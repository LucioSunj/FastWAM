"""Action-only behavior cloning for the UNCOND FastWAM ActionDiT LoRA.

This module is deliberately separate from :mod:`fastwam.trainer` and from the
joint FastWAM training losses.  It encodes only the current observation frame,
prefills the frozen video K/V cache, and evaluates the same cached UNCOND
ActionDiT velocity callable used by the adaptive RL policy.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from fastwam.adapters import (
    ActionDiTLoRAAdapter,
    PolicyRegime,
    RegimeLoRAConfig,
    inject_action_dit_lora,
)
from fastwam.models.wan22.adaptive_action import (
    CachedActionCondition,
    CachedActionVelocity,
)


@dataclass(frozen=True)
class FastWAMUncondBCConfig:
    """Shape and reporting contract for action-only UNCOND behavior cloning."""

    action_horizon: int = 32
    action_dim: int = 7
    proprio_dim: int = 8
    expected_video_frames: int = 9
    expected_video_height: int = 224
    expected_video_width: int = 448
    gripper_dimension: int = 6
    timestep_bins: int = 10
    tiled_vae: bool = False

    def __post_init__(self) -> None:
        positive_fields = {
            "action_horizon": self.action_horizon,
            "action_dim": self.action_dim,
            "proprio_dim": self.proprio_dim,
            "expected_video_frames": self.expected_video_frames,
            "expected_video_height": self.expected_video_height,
            "expected_video_width": self.expected_video_width,
            "timestep_bins": self.timestep_bins,
        }
        invalid = {name: value for name, value in positive_fields.items() if value <= 0}
        if invalid:
            raise ValueError(f"UNCOND BC dimensions must be positive: {invalid}")
        if not 0 <= self.gripper_dimension < self.action_dim:
            raise ValueError(
                "`gripper_dimension` must identify one action dimension, got "
                f"{self.gripper_dimension} for action_dim={self.action_dim}."
            )


@dataclass(frozen=True)
class ActionBCLoss:
    """Differentiable loss and compact action-only diagnostics."""

    loss_action_bc: torch.Tensor
    mse_per_dimension: torch.Tensor
    mse_pose: torch.Tensor
    mse_gripper: torch.Tensor
    mse_by_timestep_bin: torch.Tensor
    timestep_bin_count: torch.Tensor
    valid_action_count: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        """Return the public action-only metric mapping.

        No video or prediction loss is represented by this type.
        """

        return {
            "loss_action_bc": self.loss_action_bc,
            "mse_per_dimension": self.mse_per_dimension,
            "mse_pose": self.mse_pose,
            "mse_gripper": self.mse_gripper,
            "mse_by_timestep_bin": self.mse_by_timestep_bin,
            "timestep_bin_count": self.timestep_bin_count,
            "valid_action_count": self.valid_action_count,
        }


def compute_action_flow_matching_bc_loss(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    timestep: torch.Tensor,
    action_is_pad: torch.Tensor | None,
    scheduler: Any,
    gripper_dimension: int = 6,
    timestep_bins: int = 10,
) -> ActionBCLoss:
    """Compute the scheduler-weighted, padding-aware action BC objective.

    The reduction order is part of the contract: average the action dimensions,
    aggregate valid action steps per sample, apply the scheduler weight, and
    finally average samples.
    """

    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError(
            "`prediction` and `target` must have identical [B,T,D] shapes, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}."
        )
    batch_size, action_horizon, action_dim = prediction.shape
    if timestep.shape != (batch_size,):
        raise ValueError(
            f"`timestep` must have shape ({batch_size},), got {tuple(timestep.shape)}."
        )
    if not 0 <= int(gripper_dimension) < action_dim:
        raise ValueError(
            f"Invalid gripper dimension {gripper_dimension} for action_dim={action_dim}."
        )
    if timestep_bins <= 0:
        raise ValueError("`timestep_bins` must be positive.")
    if action_is_pad is None:
        valid = torch.ones(
            (batch_size, action_horizon),
            dtype=torch.bool,
            device=prediction.device,
        )
    else:
        if action_is_pad.shape != (batch_size, action_horizon):
            raise ValueError(
                "`action_is_pad` must match [B,T], got "
                f"{tuple(action_is_pad.shape)} instead of "
                f"({batch_size}, {action_horizon})."
            )
        valid = ~action_is_pad.to(device=prediction.device, dtype=torch.bool)

    squared_error = (prediction.float() - target.float()).square()
    token_mse = squared_error.mean(dim=2)
    valid_float = valid.to(dtype=token_mse.dtype)
    valid_per_sample = valid_float.sum(dim=1).clamp(min=1.0)
    sample_mse = (token_mse * valid_float).sum(dim=1) / valid_per_sample
    sample_weight = scheduler.training_weight(timestep).to(
        device=sample_mse.device,
        dtype=sample_mse.dtype,
    )
    loss = (sample_mse * sample_weight).mean()

    valid_action_count = valid.sum()
    stats_denominator = valid_action_count.to(dtype=squared_error.dtype).clamp(min=1)
    dimension_mse = (squared_error * valid_float.unsqueeze(2)).sum(
        dim=(0, 1)
    ) / stats_denominator
    pose_indices = [index for index in range(action_dim) if index != gripper_dimension]
    pose_mse = dimension_mse[pose_indices].mean()
    gripper_mse = dimension_mse[gripper_dimension]

    num_train_timesteps = int(scheduler.num_train_timesteps)
    normalized_timestep = timestep.float() / float(num_train_timesteps)
    bin_index = torch.floor(normalized_timestep * timestep_bins).to(torch.long)
    bin_index = bin_index.clamp(min=0, max=timestep_bins - 1)
    bin_sums = torch.zeros(
        timestep_bins,
        device=sample_mse.device,
        dtype=sample_mse.dtype,
    )
    bin_counts = torch.zeros(
        timestep_bins,
        device=sample_mse.device,
        dtype=torch.long,
    )
    bin_sums.scatter_add_(0, bin_index, sample_mse)
    bin_counts.scatter_add_(0, bin_index, torch.ones_like(bin_index))
    bin_mse = bin_sums / bin_counts.to(dtype=bin_sums.dtype).clamp(min=1)

    return ActionBCLoss(
        loss_action_bc=loss,
        mse_per_dimension=dimension_mse,
        mse_pose=pose_mse,
        mse_gripper=gripper_mse,
        mse_by_timestep_bin=bin_mse,
        timestep_bin_count=bin_counts,
        valid_action_count=valid_action_count,
    )


def _stateless_sample_seed(identity: str, *, seed: int) -> int:
    payload = f"fastwam-uncond-bc-v1\0{int(seed)}\0{identity}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def stateless_validation_flow_inputs(
    *,
    sample_identities: Sequence[str | int],
    action_shape: tuple[int, int, int],
    scheduler: Any,
    seed: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive validation timesteps/noise independently for every sample.

    Generation intentionally happens on CPU so results do not depend on CUDA
    rank, validation microbatch size, or traversal order.
    """

    batch_size, action_horizon, action_dim = action_shape
    if len(sample_identities) != batch_size:
        raise ValueError(
            "Sample identity count must match the validation batch: "
            f"{len(sample_identities)} != {batch_size}."
        )
    timesteps: list[torch.Tensor] = []
    noises: list[torch.Tensor] = []
    for identity in sample_identities:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stateless_sample_seed(str(identity), seed=int(seed)))
        uniform = torch.rand((), generator=generator, dtype=torch.float32)
        sigma = scheduler._phi(uniform, scheduler.shift)
        timesteps.append(sigma * float(scheduler.num_train_timesteps))
        noises.append(
            torch.randn(
                (action_horizon, action_dim),
                generator=generator,
                dtype=torch.float32,
            )
        )
    return (
        torch.stack(timesteps).to(device=device, dtype=dtype),
        torch.stack(noises).to(device=device, dtype=dtype),
    )


class SampleIdentityDataset(torch.utils.data.Dataset):
    """Attach a stable split-local identity to an existing map-style dataset."""

    def __init__(self, dataset: torch.utils.data.Dataset, *, namespace: str) -> None:
        if not namespace.strip():
            raise ValueError("Sample-identity namespace must be non-empty.")
        self.dataset = dataset
        self.namespace = namespace

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        if not isinstance(sample, Mapping):
            raise TypeError(f"BC dataset samples must be mappings, got {type(sample)}.")
        if "sample_identity" in sample:
            raise ValueError("Wrapped dataset already defines `sample_identity`.")
        result = dict(sample)
        result["sample_identity"] = f"{self.namespace}:{int(index)}"
        return result


class FastWAMUncondBCPolicy(nn.Module):
    """Frozen FastWAM-IDM plus an action-only UNCOND LoRA training surface."""

    def __init__(
        self,
        *,
        actor: nn.Module,
        lora_config: RegimeLoRAConfig,
        config: FastWAMUncondBCConfig | None = None,
        lora_adapter: ActionDiTLoRAAdapter | None = None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.config = config or FastWAMUncondBCConfig()
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
        self.lora_adapter = lora_adapter or inject_action_dit_lora(
            self.actor.action_expert,
            lora_config,
        )
        if lora_adapter is not None:
            self.lora_adapter.freeze_base()
        if self.lora_adapter.config != lora_config:
            raise ValueError("Provided LoRA adapter does not match `lora_config`.")
        self._assert_only_lora_trainable()

    @property
    def device(self) -> torch.device:
        return next(self.actor.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.actor.parameters()).dtype

    def _assert_only_lora_trainable(self) -> None:
        self.lora_adapter.audit_freeze().assert_valid()
        lora_ids = {id(parameter) for parameter in self.lora_adapter.lora_parameters()}
        unexpected = [
            name
            for name, parameter in self.actor.named_parameters()
            if parameter.requires_grad and id(parameter) not in lora_ids
        ]
        missing = [
            name
            for name, parameter in self.lora_adapter.named_lora_parameters()
            if not parameter.requires_grad
        ]
        if unexpected or missing:
            raise RuntimeError(
                "UNCOND BC freeze audit failed: "
                f"unexpected_trainables={unexpected}, frozen_lora={missing}."
            )

    def trainable_parameter_names(self) -> tuple[str, ...]:
        """Return the complete trainable set for manifests and assertions."""

        lora_ids = {id(parameter) for parameter in self.lora_adapter.lora_parameters()}
        return tuple(
            name
            for name, parameter in self.actor.named_parameters()
            if id(parameter) in lora_ids and parameter.requires_grad
        )

    def _validate_batch(self, batch: Mapping[str, Any]) -> None:
        required = {"video", "context", "context_mask", "proprio", "action"}
        missing = sorted(required - set(batch))
        if missing:
            raise KeyError(f"UNCOND BC batch is missing fields: {missing}.")
        video = batch["video"]
        action = batch["action"]
        proprio = batch["proprio"]
        if not isinstance(video, torch.Tensor) or video.ndim != 5:
            raise ValueError("`video` must be a tensor [B,3,T,H,W].")
        expected_video = (
            3,
            self.config.expected_video_frames,
            self.config.expected_video_height,
            self.config.expected_video_width,
        )
        if tuple(video.shape[1:]) != expected_video:
            raise ValueError(
                "UNCOND BC video shape mismatch: expected [B,"
                f"{','.join(str(value) for value in expected_video)}], got "
                f"{tuple(video.shape)}."
            )
        if not torch.is_floating_point(video):
            raise TypeError("UNCOND BC video tensor must be floating point.")
        if not isinstance(action, torch.Tensor) or tuple(action.shape[1:]) != (
            self.config.action_horizon,
            self.config.action_dim,
        ):
            raise ValueError(
                "UNCOND BC action must be [B,"
                f"{self.config.action_horizon},{self.config.action_dim}], got "
                f"{getattr(action, 'shape', None)}."
            )
        if action.shape[0] != video.shape[0] or not torch.is_floating_point(action):
            raise ValueError(
                "UNCOND BC action batch/dtype must match a floating video batch."
            )
        if not isinstance(proprio, torch.Tensor) or proprio.ndim != 3:
            raise ValueError("`proprio` must be a tensor [B,T,D].")
        if (
            proprio.shape[0] != video.shape[0]
            or proprio.shape[1] < 1
            or proprio.shape[2] != self.config.proprio_dim
            or not torch.is_floating_point(proprio)
        ):
            raise ValueError(
                "UNCOND BC proprio shape mismatch: "
                f"expected [B,T,{self.config.proprio_dim}], got {tuple(proprio.shape)}."
            )
        action_is_pad = batch.get("action_is_pad")
        if action_is_pad is not None and (
            not isinstance(action_is_pad, torch.Tensor)
            or tuple(action_is_pad.shape) != tuple(action.shape[:2])
        ):
            raise ValueError("UNCOND BC action_is_pad must match action [B,T].")

    @torch.no_grad()
    def prepare_action_condition(
        self,
        batch: Mapping[str, Any],
    ) -> CachedActionCondition:
        """Encode exactly the current frame and prefill frozen video K/V."""

        self._validate_batch(batch)
        video = batch["video"]
        context = batch["context"]
        context_mask = batch["context_mask"]
        proprio = batch["proprio"]
        if not isinstance(context, torch.Tensor) or context.ndim != 3:
            raise ValueError("`context` must be a tensor [B,L,D].")
        if not isinstance(context_mask, torch.Tensor) or context_mask.ndim != 2:
            raise ValueError("`context_mask` must be a tensor [B,L].")
        if (
            context.shape[:2] != context_mask.shape
            or context.shape[0] != video.shape[0]
        ):
            raise ValueError(
                "UNCOND BC context/mask batch and sequence shapes must agree."
            )
        if not torch.is_floating_point(context):
            raise TypeError("UNCOND BC cached text context must be floating point.")

        # This slice is the critical isolation boundary.  No future frame is
        # passed to the VAE, video expert, cache, or action loss.
        current_frame = video[:, :, :1].to(
            device=self.device,
            dtype=self.dtype,
            non_blocking=True,
        )
        current_latents = self.actor._encode_video_latents(
            current_frame,
            tiled=self.config.tiled_vae,
        )
        context = context.to(
            device=self.device,
            dtype=self.dtype,
            non_blocking=True,
        )
        context_mask = context_mask.to(
            device=self.device,
            dtype=torch.bool,
            non_blocking=True,
        )
        context, context_mask = self.actor._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=proprio[:, 0].to(
                device=self.device,
                dtype=self.dtype,
                non_blocking=True,
            ),
        )
        fuse_flag = bool(
            getattr(self.actor.video_expert, "fuse_vae_embedding_in_latents", False)
        )
        video_pre = self.actor.video_expert.pre_dit(
            x=current_latents,
            timestep=torch.zeros(
                current_latents.shape[0],
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
        if video_seq_len != tokens_per_frame:
            raise RuntimeError(
                "Current-frame BC condition unexpectedly contains more than one "
                f"video frame: seq_len={video_seq_len}, tokens_per_frame={tokens_per_frame}."
            )
        attention_mask = self.actor._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=self.config.action_horizon,
            video_tokens_per_frame=tokens_per_frame,
            device=video_pre["tokens"].device,
        )
        video_cache = self.actor.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
            gate_current_frame_video_tokens=tokens_per_frame,
        )
        return CachedActionCondition(
            context=context,
            context_mask=context_mask,
            video_kv_cache=video_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            current_frame_video_tokens=tokens_per_frame,
        )

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        timestep: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return only action-BC loss/statistics for one normalized batch."""

        condition = self.prepare_action_condition(batch)
        action = batch["action"].to(
            device=self.device,
            dtype=self.dtype,
            non_blocking=True,
        )
        batch_size = action.shape[0]
        if timestep is None:
            timestep = self.actor.train_action_scheduler.sample_training_t(
                batch_size=batch_size,
                device=self.device,
                dtype=action.dtype,
            )
        else:
            timestep = timestep.to(device=self.device, dtype=action.dtype)
        if timestep.shape != (batch_size,):
            raise ValueError(
                f"`timestep` must have shape ({batch_size},), got {tuple(timestep.shape)}."
            )
        if noise is None:
            noise = torch.randn_like(action)
        else:
            if noise.shape != action.shape:
                raise ValueError(
                    f"`noise` must match action shape {tuple(action.shape)}, got "
                    f"{tuple(noise.shape)}."
                )
            noise = noise.to(device=self.device, dtype=action.dtype)
        noisy_action = self.actor.train_action_scheduler.add_noise(
            action,
            noise,
            timestep,
        )
        target = self.actor.train_action_scheduler.training_target(
            action,
            noise,
            timestep,
        )
        velocity = CachedActionVelocity(
            action_expert=self.actor.action_expert,
            mot=self.actor.mot,
            condition=condition,
            regime=PolicyRegime.UNCOND,
            regime_context=self.lora_adapter.regime_context,
            capture_gate_kv=False,
        )
        prediction = velocity(noisy_action, timestep).velocity
        result = compute_action_flow_matching_bc_loss(
            prediction=prediction,
            target=target,
            timestep=timestep,
            action_is_pad=batch.get("action_is_pad"),
            scheduler=self.actor.train_action_scheduler,
            gripper_dimension=self.config.gripper_dimension,
            timestep_bins=self.config.timestep_bins,
        )
        return result.as_dict()


def lora_gradient_norm(adapter: ActionDiTLoRAAdapter) -> torch.Tensor:
    """Return the global FP32 norm of currently populated LoRA gradients."""

    squared = torch.zeros((), device=next(adapter.lora_parameters()).device)
    for parameter in adapter.lora_parameters():
        if parameter.grad is not None:
            squared = squared + parameter.grad.detach().float().square().sum()
    return squared.sqrt()


def lora_update_norm(
    adapter: ActionDiTLoRAAdapter,
    before: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Return the exact global norm of one optimizer update."""

    current = dict(adapter.named_lora_parameters())
    if set(current) != set(before):
        raise ValueError("LoRA update snapshot keys do not match the live adapter.")
    device = next(iter(current.values())).device
    squared = torch.zeros((), device=device)
    for name, parameter in current.items():
        reference = before[name].to(device=parameter.device, dtype=parameter.dtype)
        squared = squared + (parameter.detach() - reference).float().square().sum()
    return squared.sqrt()


def cosine_warmup_multiplier(
    step: int,
    *,
    total_steps: int,
    warmup_fraction: float = 0.05,
    minimum_ratio: float = 0.01,
) -> float:
    """Linear warmup followed by cosine decay to ``minimum_ratio``."""

    if total_steps <= 0:
        raise ValueError("`total_steps` must be positive.")
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("`warmup_fraction` must be in [0, 1).")
    if not 0.0 < minimum_ratio <= 1.0:
        raise ValueError("`minimum_ratio` must be in (0, 1].")
    warmup_steps = max(1, math.ceil(total_steps * warmup_fraction))
    current = max(0, min(int(step), total_steps))
    if current < warmup_steps:
        return float(current + 1) / float(warmup_steps)
    decay_steps = max(1, total_steps - warmup_steps)
    progress = float(current - warmup_steps) / float(decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum_ratio + (1.0 - minimum_ratio) * cosine
