"""Typed contracts for read-only visual memory and ActionDiT readers."""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

DINO_V3_INPUT_SIZE = 224
DINO_V3_PATCH_GRID = (14, 14)
DINO_V3_PATCH_COUNT = 196
DINO_V3_NATIVE_DIM = 384
VISUAL_READER_STATE_SCHEMA = "fastwam-action-visual-reader-v1"
VISUAL_READER_PARAMETER_FAMILY = "uncond_visual_reader"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def validate_sha256(value: str, *, label: str) -> str:
    """Return a normalized SHA256 digest or fail closed."""

    normalized = str(value).strip().lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError(f"{label} must be a 64-character lowercase SHA256 digest.")
    return normalized


def contract_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible contract with deterministic encoding."""

    encoded = json.dumps(
        dict(payload),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PreparedCameraBatch:
    """Per-view RGB pixels after the caller's pinned geometry transform."""

    pixels: torch.Tensor
    camera_ids: tuple[str, ...]
    camera_valid_mask: torch.Tensor
    input_contract_sha256: str

    def __post_init__(self) -> None:
        camera_ids = tuple(str(camera_id) for camera_id in self.camera_ids)
        if self.pixels.ndim != 5:
            raise ValueError("`pixels` must have shape [B,V,3,224,224].")
        if self.pixels.dtype != torch.uint8:
            raise TypeError("Prepared camera pixels must use uint8 dtype.")
        batch, views, channels, height, width = self.pixels.shape
        if channels != 3 or (height, width) != (
            DINO_V3_INPUT_SIZE,
            DINO_V3_INPUT_SIZE,
        ):
            raise ValueError("Prepared camera pixels must be RGB 224x224 tensors.")
        if batch < 1 or views < 1:
            raise ValueError("Prepared camera batches and view sets must be non-empty.")
        if len(camera_ids) != views:
            raise ValueError("`camera_ids` must contain one entry per camera view.")
        if any(not camera_id for camera_id in camera_ids):
            raise ValueError("Camera identifiers cannot be empty.")
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError("Camera identifiers must be unique and ordered.")
        if self.camera_valid_mask.shape != (batch, views):
            raise ValueError("`camera_valid_mask` must have shape [B,V].")
        if self.camera_valid_mask.dtype != torch.bool:
            raise TypeError("`camera_valid_mask` must use bool dtype.")
        if bool((~self.camera_valid_mask.any(dim=1)).any().item()):
            raise ValueError(
                "Every sample must contain at least one valid camera view."
            )
        object.__setattr__(self, "camera_ids", camera_ids)
        object.__setattr__(
            self,
            "input_contract_sha256",
            validate_sha256(
                self.input_contract_sha256,
                label="Camera input contract SHA256",
            ),
        )


@dataclass(frozen=True)
class NativePatchMemory:
    """Frozen native DINOv3 spatial patches in fixed camera slots."""

    tokens: torch.Tensor
    patch_valid_mask: torch.Tensor
    camera_valid_mask: torch.Tensor
    camera_ids: tuple[str, ...]
    grid: tuple[int, int]
    source_revision: str
    weights_sha256: str
    input_contract_sha256: str
    preprocess_sha256: str
    output_contract_sha256: str
    memory_contract_sha256: str

    def __post_init__(self) -> None:
        if self.tokens.ndim != 4:
            raise ValueError("Native patch tokens must have shape [B,V,N,D].")
        if not self.tokens.is_floating_point():
            raise TypeError("Native patch tokens must use a floating dtype.")
        batch, views, patches, dimension = self.tokens.shape
        if patches != DINO_V3_PATCH_COUNT or dimension != DINO_V3_NATIVE_DIM:
            raise ValueError(
                "DINOv3 ViT-S/16 memory must contain 196 patches of width 384."
            )
        camera_ids = tuple(str(camera_id) for camera_id in self.camera_ids)
        if len(camera_ids) != views or len(set(camera_ids)) != views:
            raise ValueError("Native memory camera IDs must be unique and match V.")
        if tuple(self.grid) != DINO_V3_PATCH_GRID:
            raise ValueError("Native DINOv3 memory must use the 14x14 patch grid.")
        if self.patch_valid_mask.shape != (batch, views, patches):
            raise ValueError("`patch_valid_mask` must have shape [B,V,196].")
        if self.camera_valid_mask.shape != (batch, views):
            raise ValueError("`camera_valid_mask` must have shape [B,V].")
        if self.patch_valid_mask.dtype != torch.bool:
            raise TypeError("`patch_valid_mask` must use bool dtype.")
        if self.camera_valid_mask.dtype != torch.bool:
            raise TypeError("`camera_valid_mask` must use bool dtype.")
        if self.patch_valid_mask.device != self.tokens.device:
            raise ValueError("Patch masks and native tokens must share a device.")
        if self.camera_valid_mask.device != self.tokens.device:
            raise ValueError("Camera masks and native tokens must share a device.")
        if not torch.equal(self.patch_valid_mask.any(dim=-1), self.camera_valid_mask):
            raise ValueError("Patch validity must agree exactly with camera validity.")
        if bool((~self.camera_valid_mask.any(dim=1)).any().item()):
            raise ValueError("Every native-memory sample needs one valid camera.")
        if self.tokens.requires_grad:
            raise ValueError("Native DINOv3 memory must be detached from autograd.")
        valid_tokens = self.tokens[self.patch_valid_mask]
        if not bool(torch.isfinite(valid_tokens).all().item()):
            raise ValueError("Native DINOv3 memory contains non-finite values.")
        if bool(
            (torch.linalg.vector_norm(valid_tokens.float(), dim=-1) == 0).any().item()
        ):
            raise ValueError("Native DINOv3 patches must have non-zero row norms.")
        invalid_tokens = self.tokens.masked_select(
            (~self.patch_valid_mask).unsqueeze(-1)
        )
        if invalid_tokens.numel() and bool((invalid_tokens != 0).any().item()):
            raise ValueError("Invalid native-memory slots must contain exact zeros.")
        object.__setattr__(self, "camera_ids", camera_ids)
        object.__setattr__(self, "grid", DINO_V3_PATCH_GRID)
        for field_name, label in (
            ("weights_sha256", "DINOv3 weights SHA256"),
            ("input_contract_sha256", "Camera input contract SHA256"),
            ("preprocess_sha256", "DINOv3 preprocess SHA256"),
            ("output_contract_sha256", "DINOv3 output contract SHA256"),
            ("memory_contract_sha256", "Native memory contract SHA256"),
        ):
            object.__setattr__(
                self,
                field_name,
                validate_sha256(getattr(self, field_name), label=label),
            )


@dataclass(frozen=True)
class RoutingWeights:
    """Ephemeral per-view DINO routing weights for one ActionDiT layer."""

    weights: torch.Tensor
    patch_valid_mask: torch.Tensor
    camera_valid_mask: torch.Tensor
    camera_ids: tuple[str, ...]
    router_contract_sha256: str

    def __post_init__(self) -> None:
        if self.weights.ndim != 4:
            raise ValueError("Routing weights must have shape [B,V,N_action,N_patch].")
        batch, views, _actions, patches = self.weights.shape
        if patches != DINO_V3_PATCH_COUNT:
            raise ValueError("Routing weights must target all 196 native patches.")
        if self.patch_valid_mask.shape != (batch, views, patches):
            raise ValueError("Routing patch mask shape does not match weights.")
        if self.camera_valid_mask.shape != (batch, views):
            raise ValueError("Routing camera mask shape does not match weights.")
        if self.patch_valid_mask.dtype != torch.bool:
            raise TypeError("Routing patch masks must use bool dtype.")
        if self.camera_valid_mask.dtype != torch.bool:
            raise TypeError("Routing camera masks must use bool dtype.")
        if len(self.camera_ids) != views:
            raise ValueError("Routing camera IDs must match the view dimension.")
        if not bool(torch.isfinite(self.weights).all().item()):
            raise ValueError("Routing weights contain non-finite values.")
        object.__setattr__(
            self,
            "router_contract_sha256",
            validate_sha256(
                self.router_contract_sha256,
                label="Router contract SHA256",
            ),
        )


@dataclass(frozen=True)
class LayerVideoKVView:
    """Read-only view of the current MoT layer's cached video K/V."""

    key: torch.Tensor
    value: torch.Tensor
    current_frame_tokens: int
    layout_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.key.ndim != 3 or self.value.ndim != 3:
            raise ValueError("Layer video K/V must have shape [B,S,D].")
        if self.key.shape != self.value.shape:
            raise ValueError("Layer video key and value shapes must match.")
        current = int(self.current_frame_tokens)
        if not 1 <= current <= self.key.shape[1]:
            raise ValueError("`current_frame_tokens` must lie inside video K/V.")
        object.__setattr__(self, "current_frame_tokens", current)


@dataclass(frozen=True)
class ActionLayerReadContext:
    """Stable pre-query/post-residual hook context for visual readers."""

    layer_index: int
    pre_block_hidden: torch.Tensor
    modulated_attn_input: torch.Tensor
    post_block_hidden: torch.Tensor
    base_gate_msa: torch.Tensor
    timestep_embedding: torch.Tensor
    proprio: torch.Tensor
    video: LayerVideoKVView

    def __post_init__(self) -> None:
        layer_index = int(self.layer_index)
        if layer_index < 0:
            raise ValueError("`layer_index` must be non-negative.")
        if self.pre_block_hidden.ndim != 3:
            raise ValueError("Action hidden tensors must have shape [B,N,D].")
        expected = self.pre_block_hidden.shape
        if self.modulated_attn_input.shape != expected:
            raise ValueError("Modulated attention input must match pre-block hidden.")
        if self.post_block_hidden.shape != expected:
            raise ValueError("Post-block hidden must match pre-block hidden.")
        batch = expected[0]
        if self.base_gate_msa.shape[0] != batch:
            raise ValueError("Base gate batch size does not match action hidden.")
        if (
            self.timestep_embedding.ndim != 2
            or self.timestep_embedding.shape[0] != batch
        ):
            raise ValueError("Timestep embedding must have shape [B,D_t].")
        if self.proprio.ndim != 2 or self.proprio.shape[0] != batch:
            raise ValueError("Proprioception must have shape [B,D_p].")
        if self.video.key.shape[0] != batch:
            raise ValueError("Video K/V batch size does not match action hidden.")
        object.__setattr__(self, "layer_index", layer_index)


@dataclass(frozen=True)
class VisualResidual:
    """One or more visual branches' additive post-block residual."""

    tensor: torch.Tensor
    layer_index: int
    branch_kinds: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.tensor.ndim != 3:
            raise ValueError("Visual residuals must have shape [B,N,D].")
        if int(self.layer_index) < 0:
            raise ValueError("Visual residual layer indices must be non-negative.")
        if not self.branch_kinds or any(not kind for kind in self.branch_kinds):
            raise ValueError(
                "Visual residuals must identify every contributing branch."
            )


class ActionVisualReader(nn.Module, ABC):
    """Branch-agnostic ActionDiT visual reader interface."""

    reader_kind: str
    reader_contract_sha256: str

    @property
    @abstractmethod
    def injection_layer_indices(self) -> tuple[int, ...]:
        """Return the exact ActionDiT layers that consume visual memory."""

    def should_inject(self, layer_index: int) -> bool:
        """Return whether this reader contributes at ``layer_index``."""

        return int(layer_index) in self.injection_layer_indices

    @abstractmethod
    def forward_layer(
        self,
        context: ActionLayerReadContext,
        memory: NativePatchMemory,
    ) -> VisualResidual:
        """Return the additive residual for one selected ActionDiT layer."""

    def trainable_parameter_manifest(self) -> dict[str, tuple[str, ...]]:
        """Expose explicit parameter ownership without name-substring discovery."""

        names = tuple(
            name
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        )
        if not names:
            raise ValueError("An action visual reader must own trainable parameters.")
        return {VISUAL_READER_PARAMETER_FAMILY: names}

    def export_trainable_state(self) -> dict[str, Any]:
        """Export strict reader-only state without frozen visual weights."""

        parameters = dict(self.named_parameters())
        manifest = self.trainable_parameter_manifest()
        names = manifest[VISUAL_READER_PARAMETER_FAMILY]
        return {
            "schema": VISUAL_READER_STATE_SCHEMA,
            "reader_kind": self.reader_kind,
            "reader_contract_sha256": self.reader_contract_sha256,
            "parameter_names": names,
            "state": {name: parameters[name].detach().cpu().clone() for name in names},
        }

    def load_trainable_state(self, payload: Mapping[str, Any]) -> None:
        """Strictly restore a state exported by :meth:`export_trainable_state`."""

        expected_keys = {
            "schema",
            "reader_kind",
            "reader_contract_sha256",
            "parameter_names",
            "state",
        }
        if set(payload) != expected_keys:
            raise ValueError("Visual reader state has an unexpected key set.")
        if payload["schema"] != VISUAL_READER_STATE_SCHEMA:
            raise ValueError("Unsupported visual reader state schema.")
        if payload["reader_kind"] != self.reader_kind:
            raise ValueError("Visual reader kind does not match this module.")
        if payload["reader_contract_sha256"] != self.reader_contract_sha256:
            raise ValueError("Visual reader contract hash mismatch.")
        manifest = self.trainable_parameter_manifest()[VISUAL_READER_PARAMETER_FAMILY]
        if tuple(payload["parameter_names"]) != manifest:
            raise ValueError("Visual reader parameter manifest mismatch.")
        state = payload["state"]
        if not isinstance(state, Mapping) or tuple(state) != manifest:
            raise ValueError("Visual reader state tensors do not match the manifest.")
        parameters = dict(self.named_parameters())
        with torch.no_grad():
            for name in manifest:
                tensor = state[name]
                if not isinstance(tensor, torch.Tensor):
                    raise TypeError(f"Visual reader state {name!r} is not a tensor.")
                parameter = parameters[name]
                if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype:
                    raise ValueError(
                        f"Visual reader state tensor mismatch for {name!r}."
                    )
                parameter.copy_(tensor.to(device=parameter.device))
