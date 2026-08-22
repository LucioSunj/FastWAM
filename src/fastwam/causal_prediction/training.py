"""Dual-mode shared-LoRA action training surface for the causal work package."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from fastwam.models.wan22.adaptive_action import CachedActionCondition
from fastwam.uncond_bc import (
    FastWAMUncondBCConfig,
    compute_action_flow_matching_bc_loss,
)

from .cached_action import (
    SharedCachedActionVelocity,
    compact_current_condition,
    splice_exact_current_prefix,
)
from .contracts import CausalComputeMode
from .one_pass import C1IntervalKVFusion, C1OnePassConfig
from .shared_lora import (
    SharedActionDiTLoRAAdapter,
    SharedLoRAConfig,
    inject_shared_action_dit_lora,
)


@dataclass(frozen=True)
class CausalDualModeTrainingConfig:
    """Frozen v1 loss and validation-selection contract."""

    distillation_weight: float = 1.0
    c2_teacher_degradation_limit: float = 0.02

    def __post_init__(self) -> None:
        if self.distillation_weight not in {1.0, 4.0}:
            raise ValueError(
                "Only the preregistered distillation weights 1.0 and 4.0 are allowed."
            )
        if self.c2_teacher_degradation_limit != 0.02:
            raise ValueError("The C2 teacher degradation limit is frozen at 2%.")


def deterministic_dual_mode_sequence(
    batch_size: int,
    *,
    optimizer_step: int,
    accumulation_index: int,
) -> tuple[CausalComputeMode, ...]:
    """Assign exactly half C0 and half C2 without consuming RNG state."""

    if batch_size < 2 or batch_size % 2:
        raise ValueError("Every dual-mode microbatch must have positive even size.")
    if optimizer_step < 0 or accumulation_index < 0:
        raise ValueError("Training sequence indices must be non-negative.")
    offset = (int(optimizer_step) + int(accumulation_index)) % 2
    pair = (
        (CausalComputeMode.C0_CURRENT, CausalComputeMode.C2_FULL)
        if offset == 0
        else (CausalComputeMode.C2_FULL, CausalComputeMode.C0_CURRENT)
    )
    return tuple(pair[index % 2] for index in range(batch_size))


def deterministic_current_only_sequence(
    batch_size: int,
    *,
    optimizer_step: int,
    accumulation_index: int,
) -> tuple[CausalComputeMode, ...]:
    """Assign every diagnostic-training sample to current-only conditioning."""

    if batch_size < 1 or optimizer_step < 0 or accumulation_index < 0:
        raise ValueError("Current-only sequence inputs must be non-negative.")
    return (CausalComputeMode.C0_CURRENT,) * batch_size


class FastWAMCausalDualModePolicy(nn.Module):
    """Frozen FastWAM parent plus one shared C0/C2 ActionDiT LoRA."""

    def __init__(
        self,
        *,
        actor: nn.Module,
        lora_config: SharedLoRAConfig | None = None,
        bc_config: FastWAMUncondBCConfig | None = None,
        training_config: CausalDualModeTrainingConfig | None = None,
        lora_adapter: SharedActionDiTLoRAAdapter | None = None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.bc_config = bc_config or FastWAMUncondBCConfig()
        self.training_config = training_config or CausalDualModeTrainingConfig()
        resolved_lora = lora_config or SharedLoRAConfig()
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
        self.lora_adapter = lora_adapter or inject_shared_action_dit_lora(
            self.actor.action_expert,
            resolved_lora,
        )
        if lora_adapter is not None:
            self.lora_adapter.freeze_base()
        if self.lora_adapter.config != resolved_lora:
            raise ValueError("Provided shared LoRA does not match its configuration.")
        self.assert_only_shared_lora_trainable()

    @property
    def device(self) -> torch.device:
        return next(self.actor.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.actor.parameters()).dtype

    def assert_only_shared_lora_trainable(self) -> None:
        """Require the complete policy trainable set to equal the shared LoRA."""

        self.lora_adapter.audit_freeze().assert_valid()
        allowed = {id(item) for item in self.lora_adapter.lora_parameters()}
        unexpected = [
            name
            for name, parameter in self.actor.named_parameters()
            if parameter.requires_grad and id(parameter) not in allowed
        ]
        actual = {
            id(parameter)
            for parameter in self.actor.parameters()
            if parameter.requires_grad
        }
        if unexpected or actual != allowed:
            raise RuntimeError(
                "Causal dual-mode freeze audit failed: "
                f"unexpected={unexpected}, expected={len(allowed)}, actual={len(actual)}."
            )

    def trainable_parameter_names(self) -> tuple[str, ...]:
        """Return the exact manifest trainable set."""

        allowed = {id(item) for item in self.lora_adapter.lora_parameters()}
        return tuple(
            name
            for name, parameter in self.actor.named_parameters()
            if parameter.requires_grad and id(parameter) in allowed
        )

    def _validate_batch(self, batch: Mapping[str, Any]) -> None:
        required = {"video", "context", "context_mask", "proprio", "action"}
        missing = sorted(required - set(batch))
        if missing:
            raise KeyError(f"Causal dual-mode batch is missing: {missing}.")
        video = batch["video"]
        action = batch["action"]
        if not isinstance(video, torch.Tensor) or video.ndim != 5:
            raise ValueError("Causal training video must be [B,3,T,H,W].")
        expected_video = (
            3,
            self.bc_config.expected_video_frames,
            self.bc_config.expected_video_height,
            self.bc_config.expected_video_width,
        )
        if tuple(video.shape[1:]) != expected_video:
            raise ValueError(
                f"Causal training video expected {expected_video}, got {video.shape[1:]}."
            )
        if not isinstance(action, torch.Tensor) or tuple(action.shape[1:]) != (
            self.bc_config.action_horizon,
            self.bc_config.action_dim,
        ):
            raise ValueError("Causal training action has the wrong [B,H,D] shape.")
        if action.shape[0] != video.shape[0]:
            raise ValueError("Causal video and action batch sizes differ.")

    @staticmethod
    def _subset(batch: Mapping[str, Any], indices: torch.Tensor) -> dict[str, Any]:
        result = {}
        batch_size = int(batch["video"].shape[0])
        for name, value in batch.items():
            if isinstance(value, torch.Tensor) and value.shape[:1] == (batch_size,):
                result[name] = value.index_select(0, indices.to(value.device))
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                if len(value) == batch_size:
                    result[name] = [value[int(index)] for index in indices.tolist()]
                else:
                    result[name] = value
            else:
                result[name] = value
        return result

    @torch.no_grad()
    def _condition_from_video_latents(
        self,
        *,
        video_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> CachedActionCondition:
        """Prefill one full-shape video condition."""

        fuse_flag = bool(
            getattr(self.actor.video_expert, "fuse_vae_embedding_in_latents", False)
        )
        video_pre = self.actor.video_expert.pre_dit(
            x=video_latents,
            timestep=torch.zeros(
                video_latents.shape[0],
                device=self.device,
                dtype=self.dtype,
            ),
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        mask = self.actor._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=self.bc_config.action_horizon,
            video_tokens_per_frame=tokens_per_frame,
            device=self.device,
        )
        cache = self.actor.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=mask[:video_seq_len, :video_seq_len],
            gate_current_frame_video_tokens=tokens_per_frame,
        )
        return CachedActionCondition(
            context=context,
            context_mask=context_mask,
            video_kv_cache=cache,
            attention_mask=mask,
            video_seq_len=video_seq_len,
            current_frame_video_tokens=tokens_per_frame,
        )

    @torch.no_grad()
    def prepare_action_condition(
        self,
        batch: Mapping[str, Any],
        *,
        mode: CausalComputeMode,
    ) -> CachedActionCondition:
        """Build current-only C0 or full expert-trajectory C2 K/V."""

        mode = CausalComputeMode.parse(mode)
        if mode not in {CausalComputeMode.C0_CURRENT, CausalComputeMode.C2_FULL}:
            raise ValueError("Dual-mode training accepts only C0_CURRENT and C2_FULL.")
        video = batch["video"].to(
            device=self.device,
            dtype=self.dtype,
            non_blocking=True,
        )
        context = batch["context"].to(
            device=self.device,
            dtype=self.dtype,
            non_blocking=True,
        )
        context_mask = batch["context_mask"].to(
            device=self.device,
            dtype=torch.bool,
            non_blocking=True,
        )
        proprio = batch["proprio"][:, 0].to(
            device=self.device,
            dtype=self.dtype,
            non_blocking=True,
        )
        context, context_mask = self.actor._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=proprio,
        )
        current_latents = self.actor._encode_video_latents(
            video[:, :, :1],
            tiled=self.bc_config.tiled_vae,
        )
        if mode is CausalComputeMode.C0_CURRENT:
            temporal_factor = int(self.actor.vae.temporal_downsample_factor)
            latent_t = (self.bc_config.expected_video_frames - 1) // temporal_factor + 1
            null_latents = current_latents.new_zeros(
                (
                    current_latents.shape[0],
                    current_latents.shape[1],
                    latent_t,
                    current_latents.shape[3],
                    current_latents.shape[4],
                )
            )
            null_latents[:, :, :1] = current_latents
            null_full_condition = self._condition_from_video_latents(
                video_latents=null_latents,
                context=context,
                context_mask=context_mask,
            )
            return compact_current_condition(null_full_condition)

        full_latents = self.actor._encode_video_latents(
            video,
            tiled=self.bc_config.tiled_vae,
        )
        if full_latents.shape[:2] != current_latents.shape[:2] or (
            full_latents.shape[3:] != current_latents.shape[3:]
        ):
            raise RuntimeError("C0/C2 encoded video latent shapes are incompatible.")
        full_latents = full_latents.clone()
        full_latents[:, :, :1] = current_latents
        full_condition = self._condition_from_video_latents(
            video_latents=full_latents,
            context=context,
            context_mask=context_mask,
        )
        return full_condition

    def _mode_prediction(
        self,
        batch: Mapping[str, Any],
        *,
        mode: CausalComputeMode,
        timestep: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, CachedActionCondition]:
        condition = self.prepare_action_condition(batch, mode=mode)
        action = batch["action"].to(device=self.device, dtype=self.dtype)
        noisy = self.actor.train_action_scheduler.add_noise(action, noise, timestep)
        target = self.actor.train_action_scheduler.training_target(
            action,
            noise,
            timestep,
        )
        prediction = SharedCachedActionVelocity(
            action_expert=self.actor.action_expert,
            mot=self.actor.mot,
            condition=condition,
        )(noisy, timestep).velocity
        return prediction, target, condition

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        modes: Sequence[CausalComputeMode | str],
        timestep: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the exact 0.5 C0 + 0.5 C2 + distillation objective."""

        self._validate_batch(batch)
        batch_size = int(batch["action"].shape[0])
        parsed = tuple(CausalComputeMode.parse(mode) for mode in modes)
        if len(parsed) != batch_size:
            raise ValueError("Mode assignment must align with the training batch.")
        counts = {mode: parsed.count(mode) for mode in set(parsed)}
        expected = {
            CausalComputeMode.C0_CURRENT: batch_size // 2,
            CausalComputeMode.C2_FULL: batch_size // 2,
        }
        if batch_size % 2 or counts != expected:
            raise ValueError(
                f"Every optimizer microbatch must be exactly half C0/C2: {counts}."
            )
        action = batch["action"].to(device=self.device, dtype=self.dtype)
        if timestep is None:
            timestep = self.actor.train_action_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            timestep = timestep.to(device=self.device, dtype=self.dtype)
        if noise is None:
            noise = torch.randn_like(action)
        else:
            noise = noise.to(device=self.device, dtype=self.dtype)
        if timestep.shape != (batch_size,) or noise.shape != action.shape:
            raise ValueError(
                "Training timestep/noise shapes do not align with actions."
            )

        results: dict[CausalComputeMode, Any] = {}
        for mode in expected:
            indices = torch.tensor(
                [index for index, value in enumerate(parsed) if value is mode],
                device=self.device,
                dtype=torch.long,
            )
            subset = self._subset(batch, indices.cpu())
            prediction, target, condition = self._mode_prediction(
                subset,
                mode=mode,
                timestep=timestep.index_select(0, indices),
                noise=noise.index_select(0, indices),
            )
            action_loss = compute_action_flow_matching_bc_loss(
                prediction=prediction,
                target=target,
                timestep=timestep.index_select(0, indices),
                action_is_pad=subset.get("action_is_pad"),
                scheduler=self.actor.train_action_scheduler,
                gripper_dimension=self.bc_config.gripper_dimension,
                timestep_bins=self.bc_config.timestep_bins,
            ).loss_action_bc
            results[mode] = (
                indices,
                subset,
                prediction,
                target,
                condition,
                action_loss,
            )

        c2_indices, c2_batch, c2_prediction, c2_target, c2_condition, c2_loss = results[
            CausalComputeMode.C2_FULL
        ]
        c2_action = c2_batch["action"].to(device=self.device, dtype=self.dtype)
        c2_noise = noise.index_select(0, c2_indices)
        c2_timestep = timestep.index_select(0, c2_indices)
        c2_noisy = self.actor.train_action_scheduler.add_noise(
            c2_action,
            c2_noise,
            c2_timestep,
        )
        with torch.no_grad(), self.lora_adapter.base_only():
            teacher_prediction = SharedCachedActionVelocity(
                action_expert=self.actor.action_expert,
                mot=self.actor.mot,
                condition=c2_condition,
            )(c2_noisy, c2_timestep).velocity
        valid = c2_batch.get("action_is_pad")
        if valid is None:
            valid_float = torch.ones_like(c2_prediction[..., :1], dtype=torch.float32)
        else:
            valid_float = (~valid.to(device=self.device, dtype=torch.bool)).float()[
                ..., None
            ]
        distill = (
            (c2_prediction.float() - teacher_prediction.float()).square() * valid_float
        ).sum() / (valid_float.sum() * c2_prediction.shape[-1]).clamp_min(1.0)
        teacher_loss = compute_action_flow_matching_bc_loss(
            prediction=teacher_prediction,
            target=c2_target,
            timestep=c2_timestep,
            action_is_pad=c2_batch.get("action_is_pad"),
            scheduler=self.actor.train_action_scheduler,
            gripper_dimension=self.bc_config.gripper_dimension,
            timestep_bins=self.bc_config.timestep_bins,
        ).loss_action_bc
        c0_loss = results[CausalComputeMode.C0_CURRENT][-1]
        total = (
            0.5 * c0_loss
            + 0.5 * c2_loss
            + self.training_config.distillation_weight * distill
        )
        return {
            "loss": total,
            "loss_c0_action": c0_loss,
            "loss_c2_action": c2_loss,
            "loss_c2_distillation": distill,
            "loss_c2_teacher": teacher_loss,
        }

    @torch.no_grad()
    def evaluate_both_modes(
        self,
        batch: Mapping[str, Any],
        *,
        timestep: torch.Tensor,
        noise: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Evaluate every validation sample under C0, C2, and base C2."""

        self._validate_batch(batch)
        batch_size = int(batch["action"].shape[0])
        timestep = timestep.to(device=self.device, dtype=self.dtype)
        noise = noise.to(device=self.device, dtype=self.dtype)
        if timestep.shape != (batch_size,) or noise.shape != batch["action"].shape:
            raise ValueError("Validation timestep/noise shapes do not align.")
        outputs = {}
        c2_condition = None
        c2_target = None
        c2_prediction = None
        for mode in (CausalComputeMode.C0_CURRENT, CausalComputeMode.C2_FULL):
            prediction, target, condition = self._mode_prediction(
                batch,
                mode=mode,
                timestep=timestep,
                noise=noise,
            )
            loss = compute_action_flow_matching_bc_loss(
                prediction=prediction,
                target=target,
                timestep=timestep,
                action_is_pad=batch.get("action_is_pad"),
                scheduler=self.actor.train_action_scheduler,
                gripper_dimension=self.bc_config.gripper_dimension,
                timestep_bins=self.bc_config.timestep_bins,
            ).loss_action_bc
            outputs[
                "loss_c0_action"
                if mode is CausalComputeMode.C0_CURRENT
                else "loss_c2_action"
            ] = loss
            if mode is CausalComputeMode.C2_FULL:
                c2_condition = condition
                c2_target = target
                c2_prediction = prediction
        if c2_condition is None or c2_target is None or c2_prediction is None:
            raise AssertionError("C2 validation condition was not constructed.")
        action = batch["action"].to(device=self.device, dtype=self.dtype)
        noisy = self.actor.train_action_scheduler.add_noise(action, noise, timestep)
        with self.lora_adapter.base_only():
            teacher = SharedCachedActionVelocity(
                action_expert=self.actor.action_expert,
                mot=self.actor.mot,
                condition=c2_condition,
            )(noisy, timestep).velocity
        outputs["loss_c2_teacher"] = compute_action_flow_matching_bc_loss(
            prediction=teacher,
            target=c2_target,
            timestep=timestep,
            action_is_pad=batch.get("action_is_pad"),
            scheduler=self.actor.train_action_scheduler,
            gripper_dimension=self.bc_config.gripper_dimension,
            timestep_bins=self.bc_config.timestep_bins,
        ).loss_action_bc
        outputs["loss_equal_mode"] = 0.5 * (
            outputs["loss_c0_action"] + outputs["loss_c2_action"]
        )
        return outputs


class FastWAMCausalTriModePolicy(FastWAMCausalDualModePolicy):
    """Frozen FastWAM parent plus shared LoRA and conditional C1 fusion."""

    def __init__(
        self,
        *,
        actor: nn.Module,
        lora_config: SharedLoRAConfig | None = None,
        bc_config: FastWAMUncondBCConfig | None = None,
        training_config: CausalDualModeTrainingConfig | None = None,
        lora_adapter: SharedActionDiTLoRAAdapter | None = None,
        c1_config: C1OnePassConfig | None = None,
    ) -> None:
        super().__init__(
            actor=actor,
            lora_config=lora_config,
            bc_config=bc_config,
            training_config=training_config,
            lora_adapter=lora_adapter,
        )
        self.c1_fusion = C1IntervalKVFusion(c1_config)
        self.actor.causal_c1_interval_fusion = self.c1_fusion
        self.assert_only_causal_trainables()

    def assert_only_causal_trainables(self) -> None:
        """Require exactly shared LoRA plus interval-fusion parameters."""

        self.lora_adapter.audit_freeze().assert_valid()
        allowed = {
            *(id(item) for item in self.lora_adapter.lora_parameters()),
            *(id(item) for item in self.c1_fusion.parameters()),
        }
        actual = {
            id(parameter)
            for parameter in self.actor.parameters()
            if parameter.requires_grad
        }
        if actual != allowed:
            unexpected = [
                name
                for name, parameter in self.actor.named_parameters()
                if parameter.requires_grad and id(parameter) not in allowed
            ]
            raise RuntimeError(
                "Causal tri-mode freeze audit failed: "
                f"unexpected={unexpected}, expected={len(allowed)}, actual={len(actual)}."
            )

    def trainable_parameter_names(self) -> tuple[str, ...]:
        """Return LoRA and C1 fusion names for the run manifest."""

        allowed = {
            *(id(item) for item in self.lora_adapter.lora_parameters()),
            *(id(item) for item in self.c1_fusion.parameters()),
        }
        return tuple(
            name
            for name, parameter in self.actor.named_parameters()
            if parameter.requires_grad and id(parameter) in allowed
        )

    def _prepare_c1_action_condition(
        self,
        batch: Mapping[str, Any],
    ) -> CachedActionCondition:
        """Build Gaussian future slots and run one frozen video-expert pass."""

        video = batch["video"].to(
            device=self.device,
            dtype=self.dtype,
            non_blocking=True,
        )
        with torch.no_grad():
            first = self.actor._encode_video_latents(
                video[:, :, :1],
                tiled=self.bc_config.tiled_vae,
            )
            temporal_factor = int(self.actor.vae.temporal_downsample_factor)
            latent_t = (self.bc_config.expected_video_frames - 1) // temporal_factor + 1
            expected_shape = (
                first.shape[0],
                first.shape[1],
                latent_t,
                first.shape[3],
                first.shape[4],
            )
            supplied_noise = batch.get("c1_video_noise")
            if supplied_noise is None:
                slots = torch.randn(
                    expected_shape, device=self.device, dtype=self.dtype
                )
            else:
                if (
                    not isinstance(supplied_noise, torch.Tensor)
                    or tuple(supplied_noise.shape) != expected_shape
                ):
                    raise ValueError("C1 video noise does not match the future slots.")
                slots = supplied_noise.to(device=self.device, dtype=self.dtype)
            slots[:, :, :1] = first
            context = batch["context"].to(
                device=self.device,
                dtype=self.dtype,
                non_blocking=True,
            )
            context_mask = batch["context_mask"].to(
                device=self.device,
                dtype=torch.bool,
                non_blocking=True,
            )
            proprio = batch["proprio"][:, 0].to(
                device=self.device,
                dtype=self.dtype,
                non_blocking=True,
            )
            context, context_mask = self.actor._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
            fuse_flag = bool(
                getattr(
                    self.actor.video_expert,
                    "fuse_vae_embedding_in_latents",
                    False,
                )
            )
            video_pre = self.actor.video_expert.pre_dit(
                x=slots,
                timestep=torch.ones(
                    slots.shape[0], device=self.device, dtype=self.dtype
                ),
                context=context,
                context_mask=context_mask,
                action=None,
                fuse_vae_embedding_in_latents=fuse_flag,
            )
            video_seq_len = int(video_pre["tokens"].shape[1])
            tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
            mask = self.actor._build_mot_attention_mask(
                video_seq_len=video_seq_len,
                action_seq_len=self.bc_config.action_horizon,
                video_tokens_per_frame=tokens_per_frame,
                device=self.device,
            )
            one_pass_cache = self.actor.mot.prefill_video_cache(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                video_attention_mask=mask[:video_seq_len, :video_seq_len],
                gate_current_frame_video_tokens=tokens_per_frame,
            )
        cache = self.c1_fusion(
            one_pass_cache,
            current_token_count=tokens_per_frame,
        )
        full_condition = CachedActionCondition(
            context=context,
            context_mask=context_mask,
            video_kv_cache=cache,
            attention_mask=mask,
            video_seq_len=video_seq_len,
            current_frame_video_tokens=tokens_per_frame,
        )
        current_condition = self._condition_from_video_latents(
            video_latents=first,
            context=context,
            context_mask=context_mask,
        )
        return splice_exact_current_prefix(current_condition, full_condition)

    def prepare_action_condition(
        self,
        batch: Mapping[str, Any],
        *,
        mode: CausalComputeMode,
    ) -> CachedActionCondition:
        """Build a C0, one-pass C1, or demonstration-future C2 condition."""

        parsed = CausalComputeMode.parse(mode)
        if parsed is CausalComputeMode.C1_ONE_PASS:
            return self._prepare_c1_action_condition(batch)
        return super().prepare_action_condition(batch, mode=parsed)

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        modes: Sequence[CausalComputeMode | str],
        timestep: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute sample-balanced C0/C1/C2 loss plus C2 distillation."""

        self._validate_batch(batch)
        batch_size = int(batch["action"].shape[0])
        parsed = tuple(CausalComputeMode.parse(mode) for mode in modes)
        allowed = {
            CausalComputeMode.C0_CURRENT,
            CausalComputeMode.C1_ONE_PASS,
            CausalComputeMode.C2_FULL,
        }
        if len(parsed) != batch_size or not set(parsed) <= allowed:
            raise ValueError("Tri-mode assignments must align with C0/C1/C2 samples.")
        if set(parsed) != allowed:
            raise ValueError("Every tri-mode microbatch must contain C0, C1, and C2.")
        action = batch["action"].to(device=self.device, dtype=self.dtype)
        timestep = (
            self.actor.train_action_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=self.dtype,
            )
            if timestep is None
            else timestep.to(device=self.device, dtype=self.dtype)
        )
        noise = (
            torch.randn_like(action)
            if noise is None
            else noise.to(device=self.device, dtype=self.dtype)
        )
        if timestep.shape != (batch_size,) or noise.shape != action.shape:
            raise ValueError("Tri-mode flow inputs do not align with the action batch.")

        results: dict[CausalComputeMode, tuple[Any, ...]] = {}
        total_action = action.new_zeros(())
        for mode in (
            CausalComputeMode.C0_CURRENT,
            CausalComputeMode.C1_ONE_PASS,
            CausalComputeMode.C2_FULL,
        ):
            indices = torch.tensor(
                [index for index, value in enumerate(parsed) if value is mode],
                device=self.device,
                dtype=torch.long,
            )
            subset = self._subset(batch, indices.cpu())
            prediction, target, condition = self._mode_prediction(
                subset,
                mode=mode,
                timestep=timestep.index_select(0, indices),
                noise=noise.index_select(0, indices),
            )
            action_loss = compute_action_flow_matching_bc_loss(
                prediction=prediction,
                target=target,
                timestep=timestep.index_select(0, indices),
                action_is_pad=subset.get("action_is_pad"),
                scheduler=self.actor.train_action_scheduler,
                gripper_dimension=self.bc_config.gripper_dimension,
                timestep_bins=self.bc_config.timestep_bins,
            ).loss_action_bc
            total_action = total_action + (indices.numel() / batch_size) * action_loss
            results[mode] = (
                indices,
                subset,
                prediction,
                target,
                condition,
                action_loss,
            )

        c2_indices, c2_batch, c2_prediction, c2_target, c2_condition, c2_loss = results[
            CausalComputeMode.C2_FULL
        ]
        c2_timestep = timestep.index_select(0, c2_indices)
        c2_noise = noise.index_select(0, c2_indices)
        c2_action = c2_batch["action"].to(device=self.device, dtype=self.dtype)
        c2_noisy = self.actor.train_action_scheduler.add_noise(
            c2_action,
            c2_noise,
            c2_timestep,
        )
        with torch.no_grad(), self.lora_adapter.base_only():
            teacher_prediction = SharedCachedActionVelocity(
                action_expert=self.actor.action_expert,
                mot=self.actor.mot,
                condition=c2_condition,
            )(c2_noisy, c2_timestep).velocity
        valid = c2_batch.get("action_is_pad")
        valid_float = (
            torch.ones_like(c2_prediction[..., :1], dtype=torch.float32)
            if valid is None
            else (~valid.to(device=self.device, dtype=torch.bool)).float()[..., None]
        )
        distill = (
            (c2_prediction.float() - teacher_prediction.float()).square() * valid_float
        ).sum() / (valid_float.sum() * c2_prediction.shape[-1]).clamp_min(1.0)
        teacher_loss = compute_action_flow_matching_bc_loss(
            prediction=teacher_prediction,
            target=c2_target,
            timestep=c2_timestep,
            action_is_pad=c2_batch.get("action_is_pad"),
            scheduler=self.actor.train_action_scheduler,
            gripper_dimension=self.bc_config.gripper_dimension,
            timestep_bins=self.bc_config.timestep_bins,
        ).loss_action_bc
        total = (
            total_action
            + (c2_indices.numel() / batch_size)
            * self.training_config.distillation_weight
            * distill
        )
        return {
            "loss": total,
            "loss_c0_action": results[CausalComputeMode.C0_CURRENT][-1],
            "loss_c1_action": results[CausalComputeMode.C1_ONE_PASS][-1],
            "loss_c2_action": c2_loss,
            "loss_c2_distillation": distill,
            "loss_c2_teacher": teacher_loss,
        }

    @torch.no_grad()
    def evaluate_all_modes(
        self,
        batch: Mapping[str, Any],
        *,
        timestep: torch.Tensor,
        noise: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Evaluate every validation sample under all formal tri-mode experts."""

        self._validate_batch(batch)
        outputs: dict[str, torch.Tensor] = {}
        targets: dict[CausalComputeMode, tuple[torch.Tensor, ...]] = {}
        for mode in (
            CausalComputeMode.C0_CURRENT,
            CausalComputeMode.C1_ONE_PASS,
            CausalComputeMode.C2_FULL,
        ):
            prediction, target, condition = self._mode_prediction(
                batch,
                mode=mode,
                timestep=timestep.to(device=self.device, dtype=self.dtype),
                noise=noise.to(device=self.device, dtype=self.dtype),
            )
            loss = compute_action_flow_matching_bc_loss(
                prediction=prediction,
                target=target,
                timestep=timestep.to(device=self.device, dtype=self.dtype),
                action_is_pad=batch.get("action_is_pad"),
                scheduler=self.actor.train_action_scheduler,
                gripper_dimension=self.bc_config.gripper_dimension,
                timestep_bins=self.bc_config.timestep_bins,
            ).loss_action_bc
            outputs[f"loss_{mode.value.split('_')[0]}_action"] = loss
            targets[mode] = (prediction, target, condition)
        _c2_prediction, c2_target, c2_condition = targets[CausalComputeMode.C2_FULL]
        action = batch["action"].to(device=self.device, dtype=self.dtype)
        noisy = self.actor.train_action_scheduler.add_noise(
            action,
            noise.to(device=self.device, dtype=self.dtype),
            timestep.to(device=self.device, dtype=self.dtype),
        )
        with self.lora_adapter.base_only():
            teacher = SharedCachedActionVelocity(
                action_expert=self.actor.action_expert,
                mot=self.actor.mot,
                condition=c2_condition,
            )(noisy, timestep.to(device=self.device, dtype=self.dtype)).velocity
        outputs["loss_c2_teacher"] = compute_action_flow_matching_bc_loss(
            prediction=teacher,
            target=c2_target,
            timestep=timestep.to(device=self.device, dtype=self.dtype),
            action_is_pad=batch.get("action_is_pad"),
            scheduler=self.actor.train_action_scheduler,
            gripper_dimension=self.bc_config.gripper_dimension,
            timestep_bins=self.bc_config.timestep_bins,
        ).loss_action_bc
        return outputs


class FastWAMCausalCurrentOnlyPolicy(FastWAMCausalDualModePolicy):
    """Matched-budget LoRA diagnostic exposed only to C0 action conditions."""

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        modes: Sequence[CausalComputeMode | str],
        timestep: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute current-only action flow loss with the shared adapter shape."""

        self._validate_batch(batch)
        batch_size = int(batch["action"].shape[0])
        parsed = tuple(CausalComputeMode.parse(mode) for mode in modes)
        if parsed != (CausalComputeMode.C0_CURRENT,) * batch_size:
            raise ValueError("Current-only exposure accepts only C0 mode assignments.")
        action = batch["action"].to(device=self.device, dtype=self.dtype)
        if timestep is None:
            timestep = self.actor.train_action_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=self.dtype,
            )
        else:
            timestep = timestep.to(device=self.device, dtype=self.dtype)
        noise = (
            torch.randn_like(action)
            if noise is None
            else noise.to(device=self.device, dtype=self.dtype)
        )
        if timestep.shape != (batch_size,) or noise.shape != action.shape:
            raise ValueError(
                "Training timestep/noise shapes do not align with actions."
            )
        prediction, target, _ = self._mode_prediction(
            batch,
            mode=CausalComputeMode.C0_CURRENT,
            timestep=timestep,
            noise=noise,
        )
        loss = compute_action_flow_matching_bc_loss(
            prediction=prediction,
            target=target,
            timestep=timestep,
            action_is_pad=batch.get("action_is_pad"),
            scheduler=self.actor.train_action_scheduler,
            gripper_dimension=self.bc_config.gripper_dimension,
            timestep_bins=self.bc_config.timestep_bins,
        ).loss_action_bc
        return {"loss": loss, "loss_c0_action": loss}


def checkpoint_passes_dual_mode_selection(
    metrics: Mapping[str, float],
    *,
    degradation_limit: float = 0.02,
) -> bool:
    """Apply equal-mode selection plus the frozen C2 teacher guard."""

    required = {"loss_c0_action", "loss_c2_action", "loss_c2_teacher"}
    missing = sorted(required - set(metrics))
    if missing:
        raise KeyError(f"Dual-mode validation metrics are missing: {missing}.")
    if degradation_limit != 0.02:
        raise ValueError("The checkpoint degradation guard is frozen at 2%.")
    values = {name: float(metrics[name]) for name in required}
    if any(
        not torch.isfinite(torch.tensor(value)) or value < 0
        for value in values.values()
    ):
        raise ValueError(f"Dual-mode validation losses must be finite: {values}.")
    teacher = values["loss_c2_teacher"]
    if teacher == 0:
        return values["loss_c2_action"] == 0
    return values["loss_c2_action"] <= teacher * (1.0 + degradation_limit)
