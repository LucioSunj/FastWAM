"""BC feasibility surface for P1 DINOv3 native semantic memory.

This module deliberately reuses the isolated UNCOND action-only BC objective
without changing its historical checkpoint schema.  It adds only the frozen
DINO memory and the independently owned ActionDiT reader required by the P1
T1--T3 feasibility work package.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import nn

from fastwam.adapters import ActionDiTLoRAAdapter, PolicyRegime, RegimeLoRAConfig
from fastwam.models.wan22.adaptive_action import (
    CachedActionCondition,
    CachedActionVelocity,
    ModalityKeepMask,
    VisualReadCondition,
)
from fastwam.models.wan22.dinov3_memory import FrozenDinoV3Encoder
from fastwam.models.wan22.visual_backbone import FrozenVisualPatchEncoder
from fastwam.models.wan22.visual_contracts import (
    ActionVisualReader,
    NativePatchMemory,
    PreparedCameraBatch,
    PreparedVisualCameraBatch,
    SpatialPatchMemory,
    native_patch_layout_contract,
    validate_sha256,
)
from fastwam.uncond_bc import (
    FastWAMUncondBCConfig,
    FastWAMUncondBCPolicy,
    compute_action_flow_matching_bc_loss,
)

P1_CAMERA_PIXELS_KEY = "p1_camera_pixels"
P1_CAMERA_VALID_MASK_KEY = "p1_camera_valid_mask"
P1_CAMERA_IDS_KEY = "p1_camera_ids"
VISUAL_CAMERA_PIXELS_KEY = "visual_camera_pixels"
VISUAL_CAMERA_VALID_MASK_KEY = "visual_camera_valid_mask"
VISUAL_CAMERA_SOURCE_RESOLUTION_KEY = "visual_camera_source_resolution"
VISUAL_CAMERA_IDS_KEY = "visual_camera_ids"
P1_READER_PARAMETER_FAMILY = "reader"
P1_LORA_PARAMETER_FAMILY = "lora"
P1_MEMORY_DEPENDENCY_NEGATIVE_MODES = frozenset({"shuffled", "task_paired", "off"})
P1_MEMORY_INTERVENTION_MODES = frozenset(
    {
        "correct",
        "off",
        "sidecar_off",
        "zero",
        "shuffled",
        "cross_sample_shuffled",
        "task_paired",
        "drop_main",
        "drop_wrist",
    }
)


@dataclass(frozen=True)
class FastWAMP1DinoBCConfig:
    """Resolved P1-specific additions to the existing action BC contract."""

    action: FastWAMUncondBCConfig
    camera_ids: tuple[str, ...]
    camera_input_contract_sha256: str
    position_mode: str = "native_contextual_only"

    def __post_init__(self) -> None:
        if not isinstance(self.action, FastWAMUncondBCConfig):
            raise TypeError("`action` must be a FastWAMUncondBCConfig instance.")
        camera_ids = tuple(str(value) for value in self.camera_ids)
        if not camera_ids or any(not value for value in camera_ids):
            raise ValueError("P1 requires a non-empty fixed camera order.")
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError("P1 camera IDs must be unique and ordered.")
        if self.position_mode != "native_contextual_only":
            raise ValueError(
                "The first P1 feasibility run forbids a coordinate-score branch."
            )
        object.__setattr__(self, "camera_ids", camera_ids)
        object.__setattr__(
            self,
            "camera_input_contract_sha256",
            validate_sha256(
                self.camera_input_contract_sha256,
                label="P1 camera input contract SHA256",
            ),
        )

    @property
    def layout_contract(self) -> dict[str, Any]:
        """Return the exact non-trainable patch layout for artifacts."""

        return native_patch_layout_contract(self.camera_ids)


class FastWAMP1DinoBCPolicy(FastWAMUncondBCPolicy):
    """Frozen FastWAM/DINO plus trainable UNCOND LoRA and P1 reader."""

    def __init__(
        self,
        *,
        actor: nn.Module,
        lora_config: RegimeLoRAConfig,
        visual_encoder: FrozenDinoV3Encoder | FrozenVisualPatchEncoder,
        visual_reader: ActionVisualReader,
        config: FastWAMP1DinoBCConfig,
        lora_adapter: ActionDiTLoRAAdapter | None = None,
    ) -> None:
        super().__init__(
            actor=actor,
            lora_config=lora_config,
            config=config.action,
            lora_adapter=lora_adapter,
        )
        if not isinstance(
            visual_encoder,
            (FrozenDinoV3Encoder, FrozenVisualPatchEncoder),
        ):
            raise TypeError("P1 visual encoder must implement a registered encoder.")
        if not isinstance(visual_reader, ActionVisualReader):
            raise TypeError("P1 visual reader must implement ActionVisualReader.")
        self.p1_config = config
        self.visual_encoder = visual_encoder
        self.visual_reader = visual_reader
        self.visual_encoder.requires_grad_(False)
        self.visual_encoder.eval()
        self.visual_reader.to(device=self.device, dtype=self.dtype)
        if self.visual_encoder.device != self.device:
            raise ValueError("P1 DINO and FastWAM must use the same device.")
        if visual_reader.memory_contract_sha256 != self.expected_memory_contract:
            raise ValueError("P1 reader and encoder memory contracts differ.")
        self.audit_parameter_ownership()

    @property
    def expected_memory_contract(self) -> str:
        """Return the exact native-memory hash expected from the encoder."""

        if isinstance(self.visual_encoder, FrozenVisualPatchEncoder):
            return self.visual_encoder.memory_contract_sha256(
                camera_ids=self.p1_config.camera_ids,
                input_contract_sha256=self.p1_config.camera_input_contract_sha256,
            )
        from fastwam.models.wan22.dinov3_memory import native_memory_contract_sha256

        return native_memory_contract_sha256(
            self.visual_encoder.asset,
            camera_ids=self.p1_config.camera_ids,
            input_contract_sha256=self.p1_config.camera_input_contract_sha256,
        )

    def train(self, mode: bool = True) -> FastWAMP1DinoBCPolicy:
        """Train only LoRA/reader while keeping both frozen parents in eval."""

        super().train(mode)
        self.actor.eval()
        self.visual_encoder.eval()
        self.visual_reader.train(mode)
        return self

    def _camera_batch(
        self,
        batch: Mapping[str, Any],
    ) -> PreparedCameraBatch | PreparedVisualCameraBatch:
        if isinstance(self.visual_encoder, FrozenVisualPatchEncoder):
            missing = [
                key
                for key in (
                    VISUAL_CAMERA_PIXELS_KEY,
                    VISUAL_CAMERA_VALID_MASK_KEY,
                    VISUAL_CAMERA_SOURCE_RESOLUTION_KEY,
                )
                if key not in batch
            ]
            if missing:
                raise KeyError(f"V2 visual BC batch is missing fields: {missing}.")
            pixels = batch[VISUAL_CAMERA_PIXELS_KEY]
            valid = batch[VISUAL_CAMERA_VALID_MASK_KEY]
            source_resolution = batch[VISUAL_CAMERA_SOURCE_RESOLUTION_KEY]
            if not all(
                isinstance(value, torch.Tensor)
                for value in (pixels, valid, source_resolution)
            ):
                raise TypeError("V2 visual camera fields must be tensors.")
            return PreparedVisualCameraBatch(
                pixels=pixels,
                camera_ids=self.p1_config.camera_ids,
                camera_valid_mask=valid,
                input_size=self.visual_encoder.asset.input_size,
                input_contract_sha256=self.p1_config.camera_input_contract_sha256,
                source_resolution=source_resolution,
            )
        missing = [
            key
            for key in (P1_CAMERA_PIXELS_KEY, P1_CAMERA_VALID_MASK_KEY)
            if key not in batch
        ]
        if missing:
            raise KeyError(f"P1 BC batch is missing camera fields: {missing}.")
        pixels = batch[P1_CAMERA_PIXELS_KEY]
        valid = batch[P1_CAMERA_VALID_MASK_KEY]
        if not isinstance(pixels, torch.Tensor) or not isinstance(valid, torch.Tensor):
            raise TypeError("P1 camera pixels and validity mask must be tensors.")
        return PreparedCameraBatch(
            pixels=pixels,
            camera_ids=self.p1_config.camera_ids,
            camera_valid_mask=valid,
            input_contract_sha256=self.p1_config.camera_input_contract_sha256,
        )

    @torch.no_grad()
    def prepare_action_condition(
        self,
        batch: Mapping[str, Any],
        *,
        include_visual: bool = True,
        modality_keep_mask: ModalityKeepMask | None = None,
    ) -> CachedActionCondition:
        """Build the frozen current-frame cache and at most one DINO memory."""

        condition = super().prepare_action_condition(batch)
        if modality_keep_mask is not None:
            if modality_keep_mask.batch_size != condition.context.shape[0]:
                raise ValueError("Modality keep-mask batch size changed in BC prefill.")
            condition = replace(
                condition,
                modality_keep_mask=modality_keep_mask,
            )
        if not include_visual:
            return condition
        memory = self.visual_encoder.prepare_memory(
            PolicyRegime.UNCOND,
            self._camera_batch(batch),
        )
        if memory is None:
            raise RuntimeError("UNCOND P1 memory construction returned no memory.")
        if memory.memory_contract_sha256 != self.expected_memory_contract:
            raise ValueError("Runtime P1 memory contract changed unexpectedly.")
        proprio = batch["proprio"][:, 0].to(
            device=memory.tokens.device,
            dtype=self.dtype,
            non_blocking=True,
        )
        return replace(
            condition,
            visual=VisualReadCondition(
                memory=memory,
                proprio=proprio,
                video_layout_metadata={
                    "p1_native_patch_layout": memory.layout_contract,
                },
            ),
        )

    def predict_velocity(
        self,
        batch: Mapping[str, Any],
        *,
        noisy_action: torch.Tensor,
        timestep: torch.Tensor,
        regime: PolicyRegime | str = PolicyRegime.UNCOND,
        include_visual: bool = True,
    ) -> torch.Tensor:
        """Evaluate one route with an explicit P1 sidecar on/off boundary."""

        selected = PolicyRegime.parse(regime)
        if selected is PolicyRegime.IDM and include_visual:
            raise ValueError("IDM must not call or receive the P1 visual sidecar.")
        condition = self.prepare_action_condition(
            batch,
            include_visual=include_visual,
        )
        velocity = CachedActionVelocity(
            action_expert=self.actor.action_expert,
            mot=self.actor.mot,
            condition=condition,
            regime=selected,
            regime_context=self.lora_adapter.regime_context,
            capture_gate_kv=False,
            visual_reader=self.visual_reader if include_visual else None,
        )
        return velocity(noisy_action, timestep).velocity

    def forward(
        self,
        batch: Mapping[str, Any],
        *,
        timestep: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        memory_mode: str = "correct",
        memory_dependency_weight: float = 0.0,
        memory_dependency_relative_margin: float = 0.0,
        memory_dependency_negative_mode: str = "shuffled",
        memory_dependency_permutation: torch.Tensor | None = None,
        gripper_loss_multiplier: float = 1.0,
        modality_keep_mask: ModalityKeepMask | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return action BC plus an optional counterfactual memory-margin loss."""

        dependency_weight = float(memory_dependency_weight)
        dependency_margin = float(memory_dependency_relative_margin)
        negative_mode = str(memory_dependency_negative_mode).strip().lower()
        if not math.isfinite(dependency_weight) or dependency_weight < 0:
            raise ValueError("P1 memory-dependency weight must be finite and >= 0.")
        if not math.isfinite(dependency_margin) or dependency_margin < 0:
            raise ValueError(
                "P1 memory-dependency relative margin must be finite and >= 0."
            )
        if dependency_weight > 0:
            if str(memory_mode).strip().lower() != "correct":
                raise ValueError(
                    "The memory-dependency objective requires correct primary memory."
                )
            if negative_mode not in P1_MEMORY_DEPENDENCY_NEGATIVE_MODES:
                raise ValueError(
                    "P1 dependency negative memory must be shuffled, "
                    "task-paired, or off."
                )
            if negative_mode == "task_paired" and memory_dependency_permutation is None:
                raise ValueError(
                    "Task-paired dependency training requires a memory permutation."
                )
            if dependency_margin <= 0:
                raise ValueError(
                    "An enabled P1 memory-dependency objective needs a positive margin."
                )
            if timestep is None or noise is None:
                raise ValueError(
                    "P1 memory-dependency training requires explicit shared timestep "
                    "and noise tensors."
                )

        condition = self.prepare_action_condition(
            batch,
            include_visual=True,
            modality_keep_mask=modality_keep_mask,
        )
        result = self.loss_from_prepared_condition(
            batch,
            condition=condition,
            timestep=timestep,
            noise=noise,
            memory_mode=memory_mode,
            gripper_loss_multiplier=gripper_loss_multiplier,
        )
        correct_loss = result["loss_action_bc"]
        zero = correct_loss.detach().new_zeros(())
        result.update(
            {
                "loss_total": correct_loss,
                "loss_memory_dependency": zero,
                "loss_action_bc_negative_memory": zero,
                "memory_dependency_gap": zero,
                "memory_dependency_relative_gap": zero,
                "memory_dependency_required_gap": zero,
            }
        )
        if dependency_weight == 0:
            return result

        negative = self.loss_from_prepared_condition(
            batch,
            condition=condition,
            timestep=timestep,
            noise=noise,
            memory_mode=negative_mode,
            memory_permutation=memory_dependency_permutation,
            gripper_loss_multiplier=gripper_loss_multiplier,
        )
        negative_loss = negative["loss_action_bc"]
        required_gap = dependency_margin * correct_loss.detach()
        observed_gap = negative_loss - correct_loss
        dependency_loss = torch.relu(required_gap - observed_gap)
        denominator = (
            correct_loss.detach().float().clamp_min(torch.finfo(torch.float32).eps)
        )
        result.update(
            {
                "loss_total": correct_loss + dependency_weight * dependency_loss,
                "loss_memory_dependency": dependency_loss,
                "loss_action_bc_negative_memory": negative_loss,
                "memory_dependency_gap": observed_gap.detach(),
                "memory_dependency_relative_gap": (
                    observed_gap.detach().float() / denominator
                ),
                "memory_dependency_required_gap": required_gap,
            }
        )
        return result

    def loss_from_prepared_condition(
        self,
        batch: Mapping[str, Any],
        *,
        condition: CachedActionCondition,
        timestep: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        memory_mode: str = "correct",
        memory_override: NativePatchMemory | SpatialPatchMemory | None = None,
        memory_permutation: torch.Tensor | None = None,
        gripper_loss_multiplier: float = 1.0,
        return_prediction: bool = False,
        modality_keep_mask: ModalityKeepMask | None = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate BC loss while reusing one detached fixed-window cache."""

        self._validate_batch(batch)
        if not isinstance(condition, CachedActionCondition):
            raise TypeError("P1 prepared condition has an unexpected type.")
        if condition.context.shape[0] != batch["action"].shape[0]:
            raise ValueError("P1 prepared condition batch size changed.")
        if modality_keep_mask is not None:
            if condition.modality_keep_mask is not None:
                raise ValueError("Prepared condition already owns a modality keep mask.")
            condition = replace(
                condition,
                modality_keep_mask=modality_keep_mask,
            )
        normalized_mode = str(memory_mode).strip().lower()
        if normalized_mode not in P1_MEMORY_INTERVENTION_MODES:
            raise ValueError(
                "Unsupported P1 memory intervention "
                f"{normalized_mode!r}; expected "
                f"{sorted(P1_MEMORY_INTERVENTION_MODES)}."
            )
        sidecar_off = normalized_mode in {"off", "sidecar_off"}
        if condition.visual is None and not sidecar_off:
            raise ValueError("P1 prepared condition must contain native memory.")
        if memory_override is not None and (
            memory_override.memory_contract_sha256 != self.expected_memory_contract
            or memory_override.tokens.shape[0] != batch["action"].shape[0]
        ):
            raise ValueError("P1 memory override contract or batch size changed.")
        if memory_permutation is not None and memory_override is not None:
            raise ValueError("Memory override and permutation are mutually exclusive.")
        if memory_permutation is not None and normalized_mode not in {
            "shuffled",
            "cross_sample_shuffled",
            "task_paired",
        }:
            raise ValueError(
                "Memory permutations are valid only for shuffled/task-paired modes."
            )
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
            raise ValueError("P1 timestep must have shape [B].")
        if noise is None:
            noise = torch.randn_like(action)
        else:
            if noise.shape != action.shape:
                raise ValueError("P1 noise must match the normalized action shape.")
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
        visual_reader: ActionVisualReader | None = self.visual_reader
        if normalized_mode == "zero" or sidecar_off:
            if memory_override is not None:
                raise ValueError("Sidecar-off evaluation cannot accept an override.")
            condition = replace(condition, visual=None)
            visual_reader = None
        else:
            memory = condition.visual.memory
            if memory_override is not None:
                if normalized_mode != "correct":
                    raise ValueError(
                        "Explicit memory overrides require memory_mode='correct'."
                    )
                memory = memory_override
            condition = replace(
                condition,
                visual=replace(
                    condition.visual,
                    memory=_intervene_memory(
                        memory,
                        normalized_mode,
                        permutation=memory_permutation,
                    ),
                ),
            )
        prediction = CachedActionVelocity(
            action_expert=self.actor.action_expert,
            mot=self.actor.mot,
            condition=condition,
            regime=PolicyRegime.UNCOND,
            regime_context=self.lora_adapter.regime_context,
            capture_gate_kv=False,
            visual_reader=visual_reader,
        )(noisy_action, timestep).velocity
        result = compute_action_flow_matching_bc_loss(
            prediction=prediction,
            target=target,
            timestep=timestep,
            action_is_pad=batch.get("action_is_pad"),
            scheduler=self.actor.train_action_scheduler,
            gripper_dimension=self.config.gripper_dimension,
            gripper_loss_multiplier=gripper_loss_multiplier,
            timestep_bins=self.config.timestep_bins,
        )
        output = result.as_dict()
        if return_prediction:
            valid = ~batch.get(
                "action_is_pad",
                torch.zeros(
                    prediction.shape[:2],
                    dtype=torch.bool,
                    device=prediction.device,
                ),
            ).to(device=prediction.device, dtype=torch.bool)
            squared_error = (prediction.float() - target.float()).square()
            valid_float = valid.float()
            valid_per_sample = valid_float.sum(dim=1).clamp(min=1.0)
            dimension_mse_per_sample = (squared_error * valid_float.unsqueeze(2)).sum(
                dim=1
            ) / valid_per_sample.unsqueeze(1)
            if float(gripper_loss_multiplier) == 1.0:
                token_mse = squared_error.mean(dim=2)
            else:
                dimension_weights = torch.ones(
                    prediction.shape[2],
                    device=prediction.device,
                    dtype=squared_error.dtype,
                )
                dimension_weights[self.config.gripper_dimension] = float(
                    gripper_loss_multiplier
                )
                token_mse = (squared_error * dimension_weights).sum(dim=2) / (
                    dimension_weights.sum()
                )
            sample_mse = (token_mse * valid_float).sum(dim=1) / valid_per_sample
            sample_weight = self.actor.train_action_scheduler.training_weight(
                timestep
            ).to(device=sample_mse.device, dtype=sample_mse.dtype)
            output.update(
                prediction=prediction,
                target=target,
                noisy_action=noisy_action,
                timestep=timestep,
                loss_action_bc_per_sample=sample_mse * sample_weight,
                mse_per_dimension_per_sample=dimension_mse_per_sample,
            )
        return output

    def parameter_families(self) -> dict[str, tuple[nn.Parameter, ...]]:
        """Return disjoint explicit LoRA and reader optimizer ownership."""

        families = {
            P1_LORA_PARAMETER_FAMILY: tuple(self.lora_adapter.lora_parameters()),
            P1_READER_PARAMETER_FAMILY: tuple(
                parameter
                for parameter in self.visual_reader.parameters()
                if parameter.requires_grad
            ),
        }
        if not all(families.values()):
            raise ValueError("P1 LoRA and reader parameter families must be non-empty.")
        ids = [{id(parameter) for parameter in values} for values in families.values()]
        if ids[0] & ids[1]:
            raise ValueError("P1 LoRA and reader optimizer families overlap.")
        return families

    def audit_parameter_ownership(self) -> None:
        """Fail if DINO/base trainability or optimizer ownership drifts."""

        self._assert_only_lora_trainable()
        if any(
            parameter.requires_grad for parameter in self.visual_encoder.parameters()
        ):
            raise RuntimeError("Frozen P1 DINO unexpectedly owns trainable parameters.")
        families = self.parameter_families()
        allowed = {
            id(parameter) for values in families.values() for parameter in values
        }
        unexpected = [
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad and id(parameter) not in allowed
        ]
        if unexpected:
            raise RuntimeError(f"Unexpected P1 trainable parameters: {unexpected}.")


def _validated_memory_permutation(
    permutation: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    require_explicit: bool,
) -> torch.Tensor:
    """Return one strict batch permutation for a memory intervention."""

    if permutation is None:
        if require_explicit:
            raise ValueError("Task-paired memory requires an explicit permutation.")
        permutation = torch.arange(batch_size - 1, -1, -1, device=device)
    else:
        if permutation.ndim != 1 or permutation.shape[0] != batch_size:
            raise ValueError("Memory permutation must have shape [batch].")
        if permutation.dtype not in {torch.int32, torch.int64}:
            raise TypeError("Memory permutation must use an integer dtype.")
        permutation = permutation.to(device=device, dtype=torch.int64)
    if not torch.equal(
        permutation.sort().values,
        torch.arange(batch_size, device=device),
    ):
        raise ValueError("Memory permutation must be a bijection over the batch.")
    if bool((permutation == torch.arange(batch_size, device=device)).any().item()):
        raise ValueError("Memory permutation cannot contain fixed points.")
    return permutation


def _drop_memory_camera(
    memory: NativePatchMemory | SpatialPatchMemory,
    camera_id: str,
) -> NativePatchMemory | SpatialPatchMemory:
    """Return memory with one named camera invalidated and zeroed."""

    if camera_id not in memory.camera_ids:
        raise ValueError(f"Memory has no camera {camera_id!r}.")
    index = memory.camera_ids.index(camera_id)
    camera_valid = memory.camera_valid_mask.clone()
    camera_valid[:, index] = False
    if bool((~camera_valid.any(dim=1)).any().item()):
        raise ValueError("A camera-drop intervention removed every valid view.")
    patch_valid = memory.patch_valid_mask.clone()
    patch_valid[:, index] = False
    tokens = memory.tokens.clone()
    tokens[:, index] = 0
    return replace(
        memory,
        tokens=tokens,
        patch_valid_mask=patch_valid,
        camera_valid_mask=camera_valid,
    )


def _intervene_memory(
    memory: NativePatchMemory | SpatialPatchMemory,
    mode: str,
    *,
    permutation: torch.Tensor | None = None,
) -> NativePatchMemory | SpatialPatchMemory:
    """Return a detached correct/shuffled/task-paired/view-drop memory."""

    normalized = str(mode).strip().lower()
    if normalized == "correct":
        if permutation is not None:
            raise ValueError("Correct memory cannot accept a permutation.")
        return memory
    if normalized in {"shuffled", "cross_sample_shuffled"}:
        if memory.tokens.shape[0] < 2:
            raise ValueError("Cross-sample shuffled memory requires batch size >= 2.")
        resolved = _validated_memory_permutation(
            permutation,
            batch_size=memory.tokens.shape[0],
            device=memory.tokens.device,
            require_explicit=False,
        )
        return replace(
            memory,
            tokens=memory.tokens[resolved],
            patch_valid_mask=memory.patch_valid_mask[resolved],
            camera_valid_mask=memory.camera_valid_mask[resolved],
        )
    if normalized == "task_paired":
        resolved = _validated_memory_permutation(
            permutation,
            batch_size=memory.tokens.shape[0],
            device=memory.tokens.device,
            require_explicit=True,
        )
        return replace(
            memory,
            tokens=memory.tokens[resolved],
            patch_valid_mask=memory.patch_valid_mask[resolved],
            camera_valid_mask=memory.camera_valid_mask[resolved],
        )
    if normalized == "drop_main":
        return _drop_memory_camera(memory, "main")
    if normalized == "drop_wrist":
        return _drop_memory_camera(memory, "wrist")
    raise ValueError(f"Memory mode {normalized!r} is not a memory-tensor intervention.")


FastWAMP1VisualBCConfig = FastWAMP1DinoBCConfig


class FastWAMP1VisualBCPolicy(FastWAMP1DinoBCPolicy):
    """V2 public policy name for registered DINOv3 and LingBot backbones."""


def build_p1_optimizer(
    policy: FastWAMP1DinoBCPolicy,
    *,
    lora_learning_rate: float,
    reader_learning_rate: float,
    train_lora: bool,
    train_reader: bool = True,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1.0e-8,
    weight_decay: float = 0.01,
) -> torch.optim.AdamW:
    """Build explicit, disjoint P1 optimizer groups."""

    rates = (float(lora_learning_rate), float(reader_learning_rate))
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("P1 optimizer learning rates must be finite and positive.")
    families = policy.parameter_families()
    for parameter in families[P1_LORA_PARAMETER_FAMILY]:
        parameter.requires_grad_(bool(train_lora))
    for parameter in families[P1_READER_PARAMETER_FAMILY]:
        parameter.requires_grad_(bool(train_reader))
    groups = []
    if train_lora:
        groups.append(
            {
                "params": families[P1_LORA_PARAMETER_FAMILY],
                "lr": rates[0],
                "name": P1_LORA_PARAMETER_FAMILY,
            }
        )
    if train_reader:
        groups.append(
            {
                "params": families[P1_READER_PARAMETER_FAMILY],
                "lr": rates[1],
                "name": P1_READER_PARAMETER_FAMILY,
            }
        )
    if not groups:
        raise ValueError("P1 optimizer must own at least one parameter family.")
    return torch.optim.AdamW(
        groups,
        betas=betas,
        eps=float(eps),
        weight_decay=float(weight_decay),
    )
