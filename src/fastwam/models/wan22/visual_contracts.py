"""Typed contracts for read-only visual memory and ActionDiT readers."""

from __future__ import annotations

import hashlib
import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
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
WAN_FLATTEN_ORDER = "t_h_w_row_major"
WAN_VIDEO_VALUE_LAYOUT = "batch_sequence_flat_heads"
P6_QUERY_SOURCE = "base_modulated_self_attn_input"
P6_QUERY_TIMING = "pre_block"
P6_RESIDUAL_TIMING = "post_block"

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
        if self.tokens.is_inference():
            raise ValueError(
                "Native DINOv3 memory must be materialized as an ordinary tensor."
            )
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
class WanValueSpatialMetadata:
    """Immutable provenance for the current-frame Wan value grid.

    Pixel boxes use half-open ``(top, left, bottom, right)`` coordinates in the
    combined RGB canvas. Wan supports use half-open ``(row0, row1, col0, col1)``
    coordinates in the current frame's spatial grid.
    """

    wan_grid_f: int
    wan_grid_h: int
    wan_grid_w: int
    current_frame_video_tokens: int
    wan_flatten_order: str
    vae_model_type: str
    vae_weights_sha256: str
    vae_spatial_downsample_factor: int
    video_dit_weights_sha256: str
    video_dit_patch_size: tuple[int, int, int]
    video_attention_num_heads: int
    video_attention_head_dim: int
    video_value_layout: str
    video_value_rope_applied: bool
    camera_concat_mode: str
    camera_order: tuple[str, ...]
    per_camera_post_crop_hw: tuple[tuple[int, int], ...]
    per_camera_combined_rgb_box: tuple[tuple[int, int, int, int], ...]
    per_camera_wan_grid_support: tuple[tuple[int, int, int, int], ...]
    dino_patch_grid: tuple[int, int]
    dino_preprocess_sha256: str
    invalid_mask_policy: str
    query_source: str = P6_QUERY_SOURCE
    query_timing: str = P6_QUERY_TIMING
    residual_timing: str = P6_RESIDUAL_TIMING
    spatial_transport_contract_sha256: str | None = None

    def __post_init__(self) -> None:
        integer_fields = (
            "wan_grid_f",
            "wan_grid_h",
            "wan_grid_w",
            "current_frame_video_tokens",
            "vae_spatial_downsample_factor",
            "video_attention_num_heads",
            "video_attention_head_dim",
        )
        for field_name in integer_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or int(value) < 1:
                raise ValueError(f"`{field_name}` must be a positive integer.")
            object.__setattr__(self, field_name, int(value))
        if self.wan_grid_f != 1:
            raise ValueError("P6 MVP requires exactly one current Wan frame grid.")
        if (
            self.wan_grid_f * self.wan_grid_h * self.wan_grid_w
            != self.current_frame_video_tokens
        ):
            raise ValueError("Current-frame token count does not match the Wan grid.")
        if self.wan_flatten_order != WAN_FLATTEN_ORDER:
            raise ValueError("Unsupported Wan value flatten order.")
        if self.video_value_layout != WAN_VIDEO_VALUE_LAYOUT:
            raise ValueError("Unsupported Wan video-value layout.")
        if self.video_value_rope_applied is not False:
            raise ValueError("P6 requires pre-RoPE Wan video values.")
        if self.camera_concat_mode not in {"horizontal", "vertical", "main_only"}:
            raise ValueError("Unsupported camera concatenation mode.")
        camera_order = tuple(str(item) for item in self.camera_order)
        if not camera_order or any(not item for item in camera_order):
            raise ValueError("Wan metadata needs non-empty camera identifiers.")
        if len(set(camera_order)) != len(camera_order):
            raise ValueError("Wan metadata camera identifiers must be unique.")
        if self.camera_concat_mode == "main_only" and len(camera_order) != 1:
            raise ValueError("main_only Wan metadata must contain one camera.")
        object.__setattr__(self, "camera_order", camera_order)
        patch_size = tuple(int(value) for value in self.video_dit_patch_size)
        if len(patch_size) != 3 or any(value < 1 for value in patch_size):
            raise ValueError("VideoDiT patch size must contain three positive values.")
        object.__setattr__(self, "video_dit_patch_size", patch_size)
        dino_grid = tuple(int(value) for value in self.dino_patch_grid)
        if dino_grid != DINO_V3_PATCH_GRID:
            raise ValueError("P6 MVP requires the pinned 14x14 native DINO grid.")
        object.__setattr__(self, "dino_patch_grid", dino_grid)

        sequence_fields: tuple[tuple[str, int], ...] = (
            ("per_camera_post_crop_hw", 2),
            ("per_camera_combined_rgb_box", 4),
            ("per_camera_wan_grid_support", 4),
        )
        for field_name, width in sequence_fields:
            values = tuple(
                tuple(int(item) for item in row) for row in getattr(self, field_name)
            )
            if len(values) != len(camera_order) or any(
                len(row) != width for row in values
            ):
                raise ValueError(
                    f"`{field_name}` must align exactly with camera order."
                )
            object.__setattr__(self, field_name, values)
        for height, width in self.per_camera_post_crop_hw:
            if height < 1 or width < 1:
                raise ValueError("Per-camera post-crop shapes must be positive.")
        for camera_index, (top, left, bottom, right) in enumerate(
            self.per_camera_combined_rgb_box
        ):
            crop_h, crop_w = self.per_camera_post_crop_hw[camera_index]
            if min(top, left) < 0 or bottom - top != crop_h or right - left != crop_w:
                raise ValueError(
                    "Combined RGB boxes must match post-crop camera shapes."
                )
        for row0, row1, col0, col1 in self.per_camera_wan_grid_support:
            if not (
                0 <= row0 < row1 <= self.wan_grid_h
                and 0 <= col0 < col1 <= self.wan_grid_w
            ):
                raise ValueError(
                    "Per-camera Wan support lies outside the current grid."
                )
        if not self.vae_model_type or not self.invalid_mask_policy:
            raise ValueError("Wan spatial metadata strings cannot be empty.")
        if self.query_source != P6_QUERY_SOURCE:
            raise ValueError(
                "P6 query source must be the base modulated attention input."
            )
        if self.query_timing != P6_QUERY_TIMING:
            raise ValueError("P6 query timing must be pre-block.")
        if self.residual_timing != P6_RESIDUAL_TIMING:
            raise ValueError("P6 residual timing must be post-block.")
        for field_name, label in (
            ("vae_weights_sha256", "VAE weights SHA256"),
            ("video_dit_weights_sha256", "VideoDiT weights SHA256"),
            ("dino_preprocess_sha256", "DINO preprocess SHA256"),
        ):
            object.__setattr__(
                self,
                field_name,
                validate_sha256(getattr(self, field_name), label=label),
            )
        expected_hash = contract_sha256(self.contract_payload())
        supplied_hash = self.spatial_transport_contract_sha256
        if (
            supplied_hash is not None
            and validate_sha256(
                supplied_hash,
                label="Spatial transport contract SHA256",
            )
            != expected_hash
        ):
            raise ValueError("Spatial transport contract SHA256 mismatch.")
        object.__setattr__(self, "spatial_transport_contract_sha256", expected_hash)

    @property
    def video_value_width(self) -> int:
        """Return the flattened Wan attention-head width."""

        return self.video_attention_num_heads * self.video_attention_head_dim

    def contract_payload(self) -> dict[str, Any]:
        """Return the canonical geometry and asset contract before its hash."""

        return {
            "schema": "fastwam-p6-wan-value-spatial-contract-v1",
            "wan_grid": [self.wan_grid_f, self.wan_grid_h, self.wan_grid_w],
            "current_frame_video_tokens": self.current_frame_video_tokens,
            "wan_flatten_order": self.wan_flatten_order,
            "vae_model_type": self.vae_model_type,
            "vae_weights_sha256": self.vae_weights_sha256,
            "vae_spatial_downsample_factor": self.vae_spatial_downsample_factor,
            "video_dit_weights_sha256": self.video_dit_weights_sha256,
            "video_dit_patch_size": list(self.video_dit_patch_size),
            "video_attention_num_heads": self.video_attention_num_heads,
            "video_attention_head_dim": self.video_attention_head_dim,
            "video_value_layout": self.video_value_layout,
            "video_value_rope_applied": self.video_value_rope_applied,
            "camera_concat_mode": self.camera_concat_mode,
            "camera_order": list(self.camera_order),
            "per_camera_post_crop_hw": [
                list(item) for item in self.per_camera_post_crop_hw
            ],
            "per_camera_combined_rgb_box": [
                list(item) for item in self.per_camera_combined_rgb_box
            ],
            "per_camera_wan_grid_support": [
                list(item) for item in self.per_camera_wan_grid_support
            ],
            "dino_patch_grid": list(self.dino_patch_grid),
            "dino_preprocess_sha256": self.dino_preprocess_sha256,
            "invalid_mask_policy": self.invalid_mask_policy,
            "query_source": self.query_source,
            "query_timing": self.query_timing,
            "residual_timing": self.residual_timing,
        }


@dataclass(frozen=True)
class DinoWanSpatialTransport:
    """Fixed row-stochastic transport from per-view DINO cells to Wan cells."""

    matrix: torch.Tensor
    target_valid_mask: torch.Tensor
    camera_ids: tuple[str, ...]
    spatial_contract_sha256: str
    transport_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.matrix.ndim != 3:
            raise ValueError("DINO-to-Wan transport must have shape [V,N_D,N_W].")
        views, patches, targets = self.matrix.shape
        if patches != DINO_V3_PATCH_COUNT or targets < 1:
            raise ValueError("DINO-to-Wan transport has an invalid source/target size.")
        camera_ids = tuple(str(item) for item in self.camera_ids)
        if len(camera_ids) != views or len(set(camera_ids)) != views:
            raise ValueError("Transport camera order must be unique and match V.")
        if self.target_valid_mask.shape != (views, targets):
            raise ValueError("Transport target mask must have shape [V,N_W].")
        if self.target_valid_mask.dtype is not torch.bool:
            raise TypeError("Transport target mask must use bool dtype.")
        if self.matrix.requires_grad:
            raise ValueError("Spatial transport must be permanently non-trainable.")
        matrix = self.matrix.detach().float()
        if not bool(torch.isfinite(matrix).all().item()) or bool(
            (matrix < 0).any().item()
        ):
            raise ValueError("Spatial transport must be finite and non-negative.")
        if bool((matrix[..., ~self.target_valid_mask.any(dim=0)] != 0).any().item()):
            raise ValueError("Transport writes to a target invalid for every camera.")
        for camera_index in range(views):
            invalid_targets = ~self.target_valid_mask[camera_index]
            if bool((matrix[camera_index, :, invalid_targets] != 0).any().item()):
                raise ValueError("Transport writes outside its camera Wan support.")
        row_mass = matrix.sum(dim=-1)
        if not torch.allclose(row_mass, torch.ones_like(row_mass), atol=1e-6, rtol=0):
            raise ValueError("Every DINO transport row must preserve unit mass.")
        spatial_hash = validate_sha256(
            self.spatial_contract_sha256,
            label="Spatial transport contract SHA256",
        )
        canonical = {
            "schema": "fastwam-p6-dino-wan-transport-v1",
            "spatial_contract_sha256": spatial_hash,
            "camera_ids": list(camera_ids),
            "shape": list(matrix.shape),
            "matrix": matrix.cpu().tolist(),
            "target_valid_mask": self.target_valid_mask.cpu().tolist(),
        }
        expected_hash = contract_sha256(canonical)
        supplied_hash = self.transport_sha256
        if (
            supplied_hash is not None
            and validate_sha256(
                supplied_hash,
                label="DINO-to-Wan transport SHA256",
            )
            != expected_hash
        ):
            raise ValueError("DINO-to-Wan transport tensor/hash mismatch.")
        object.__setattr__(self, "camera_ids", camera_ids)
        object.__setattr__(self, "spatial_contract_sha256", spatial_hash)
        object.__setattr__(self, "transport_sha256", expected_hash)

    @property
    def target_count(self) -> int:
        """Return the combined current-frame Wan token count."""

        return int(self.matrix.shape[-1])

    def effective_sha256(
        self,
        *,
        camera_valid_mask: torch.Tensor,
        patch_valid_mask: torch.Tensor,
    ) -> tuple[str, ...]:
        """Hash the sample-specific active-camera and source-mask behavior."""

        batch, views = camera_valid_mask.shape
        if views != len(self.camera_ids) or patch_valid_mask.shape != (
            batch,
            views,
            DINO_V3_PATCH_COUNT,
        ):
            raise ValueError("Effective transport masks have incompatible shapes.")
        if (
            camera_valid_mask.dtype is not torch.bool
            or patch_valid_mask.dtype is not torch.bool
        ):
            raise TypeError("Effective transport masks must use bool dtype.")
        if not torch.equal(patch_valid_mask.any(dim=-1), camera_valid_mask):
            raise ValueError("Effective patch and camera validity disagree.")
        if bool((~camera_valid_mask.any(dim=1)).any().item()):
            raise ValueError("Effective transport requires at least one active camera.")
        return tuple(
            contract_sha256(
                {
                    "schema": "fastwam-p6-effective-transport-v1",
                    "transport_sha256": self.transport_sha256,
                    "camera_valid_mask": camera_valid_mask[index].cpu().tolist(),
                    "patch_valid_mask": patch_valid_mask[index].cpu().tolist(),
                }
            )
            for index in range(batch)
        )


def build_area_overlap_dino_wan_transport(
    metadata: WanValueSpatialMetadata,
) -> DinoWanSpatialTransport:
    """Build the deterministic nominal-cell area-overlap transport."""

    if not isinstance(metadata, WanValueSpatialMetadata):
        raise TypeError("Area-overlap transport requires WanValueSpatialMetadata.")
    canvas_h = max(box[2] for box in metadata.per_camera_combined_rgb_box)
    canvas_w = max(box[3] for box in metadata.per_camera_combined_rgb_box)
    matrix = torch.zeros(
        len(metadata.camera_order),
        DINO_V3_PATCH_COUNT,
        metadata.current_frame_video_tokens,
        dtype=torch.float64,
    )
    target_valid = torch.zeros(
        len(metadata.camera_order),
        metadata.current_frame_video_tokens,
        dtype=torch.bool,
    )
    dino_h, dino_w = metadata.dino_patch_grid
    for camera_index, ((top, left, bottom, right), support) in enumerate(
        zip(
            metadata.per_camera_combined_rgb_box,
            metadata.per_camera_wan_grid_support,
            strict=True,
        )
    ):
        row0, row1, col0, col1 = support
        for wan_row in range(row0, row1):
            for wan_col in range(col0, col1):
                target_valid[
                    camera_index,
                    wan_row * metadata.wan_grid_w + wan_col,
                ] = True
        for dino_row in range(dino_h):
            source_top = top + (bottom - top) * dino_row / dino_h
            source_bottom = top + (bottom - top) * (dino_row + 1) / dino_h
            for dino_col in range(dino_w):
                source_left = left + (right - left) * dino_col / dino_w
                source_right = left + (right - left) * (dino_col + 1) / dino_w
                source_index = dino_row * dino_w + dino_col
                for wan_row in range(row0, row1):
                    target_top = canvas_h * wan_row / metadata.wan_grid_h
                    target_bottom = canvas_h * (wan_row + 1) / metadata.wan_grid_h
                    overlap_h = max(
                        0.0,
                        min(source_bottom, target_bottom) - max(source_top, target_top),
                    )
                    if overlap_h == 0:
                        continue
                    for wan_col in range(col0, col1):
                        target_left = canvas_w * wan_col / metadata.wan_grid_w
                        target_right = canvas_w * (wan_col + 1) / metadata.wan_grid_w
                        overlap_w = max(
                            0.0,
                            min(source_right, target_right)
                            - max(source_left, target_left),
                        )
                        if overlap_w == 0:
                            continue
                        target_index = wan_row * metadata.wan_grid_w + wan_col
                        matrix[camera_index, source_index, target_index] = (
                            overlap_h * overlap_w
                        )
                row_mass = matrix[camera_index, source_index].sum()
                if row_mass <= 0:
                    raise ValueError(
                        "A valid DINO patch has no positive Wan target support."
                    )
                matrix[camera_index, source_index] /= row_mass
    return DinoWanSpatialTransport(
        matrix=matrix.float(),
        target_valid_mask=target_valid,
        camera_ids=metadata.camera_order,
        spatial_contract_sha256=metadata.spatial_transport_contract_sha256,
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
    layout_metadata: WanValueSpatialMetadata | Mapping[str, Any] | None = None

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
    base_action_output_weight: torch.Tensor | None = None

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
        if self.base_action_output_weight is not None:
            weight = self.base_action_output_weight
            if weight.ndim != 2 or weight.shape[0] != expected[-1]:
                raise ValueError(
                    "Base action output weight must have shape [D_action,D_value]."
                )
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
    parameter_family: str = VISUAL_READER_PARAMETER_FAMILY

    def __init__(self) -> None:
        super().__init__()
        self._replay_reference: dict[str, torch.Tensor] | None = None
        self._replay_reference_actor_version: int | None = None

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
        family = str(self.parameter_family).strip()
        if not family:
            raise ValueError("Visual-reader parameter family cannot be empty.")
        return {family: names}

    def export_trainable_state(self) -> dict[str, Any]:
        """Export strict reader-only state without frozen visual weights."""

        parameters = dict(self.named_parameters())
        manifest = self.trainable_parameter_manifest()
        names = manifest[self.parameter_family]
        return {
            "schema": VISUAL_READER_STATE_SCHEMA,
            "reader_kind": self.reader_kind,
            "reader_contract_sha256": self.reader_contract_sha256,
            "parameter_names": names,
            "state": {name: parameters[name].detach().cpu().clone() for name in names},
        }

    def capture_replay_reference(self, *, actor_version: int) -> None:
        """Snapshot behavior-reader tensors for Gate-observation recomputation."""

        if isinstance(actor_version, bool) or int(actor_version) < 0:
            raise ValueError("Visual replay actor version must be non-negative.")
        parameters = dict(self.named_parameters())
        names = self.trainable_parameter_manifest()[self.parameter_family]
        self._replay_reference = {
            name: parameters[name].detach().clone() for name in names
        }
        self._replay_reference_actor_version = int(actor_version)

    @contextmanager
    def use_replay_reference(self, *, actor_version: int) -> Iterator[None]:
        """Temporarily restore behavior-reader tensors without saving parents."""

        if self._replay_reference is None:
            raise RuntimeError("No visual-reader replay reference was captured.")
        if int(actor_version) != self._replay_reference_actor_version:
            raise ValueError(
                "Visual-reader replay actor version mismatch: "
                f"expected {self._replay_reference_actor_version}, got {actor_version}."
            )
        current = self.export_trainable_state()
        reference = {
            **current,
            "state": self._replay_reference,
        }
        try:
            self.load_trainable_state(reference)
            yield
        finally:
            self.load_trainable_state(current)

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
        manifest = self.trainable_parameter_manifest()[self.parameter_family]
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
