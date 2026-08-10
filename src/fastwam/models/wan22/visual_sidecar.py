"""Reusable native DINO router and pluggable ActionDiT value branches."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .visual_contracts import (
    DINO_V3_NATIVE_DIM,
    ActionLayerReadContext,
    ActionVisualReader,
    DinoWanSpatialTransport,
    NativePatchMemory,
    RoutingWeights,
    VISUAL_READER_PARAMETER_FAMILY,
    VisualResidual,
    WanValueSpatialMetadata,
    contract_sha256,
    validate_sha256,
)

NATIVE_DINO_ROUTER_KIND = "native-dino-cosine-router-v1"
DINO_SEMANTIC_BRANCH_KIND = "dino-native-semantic-value-v1"
DINO_SEMANTIC_READER_KIND = "dinov3-native-semantic-reader-v1"
QUERY_SOURCE_CONTRACT = "base-modulated-self-attention-input-v1"
WAN_ROUTED_VALUE_BRANCH_KIND = "dino-routed-frozen-wan-value-v1"
DINOV3_ROUTER_WAN_VALUE_READER_KIND = "dinov3-router-wan-value-reader-v1"
P6_VISUAL_ROUTER_PARAMETER_FAMILY = "visual_router"


class ProjectionKind(str, Enum):
    """Supported action-side projection parameterizations."""

    FULL_LINEAR = "full_linear"
    LOW_RANK = "low_rank"

    @classmethod
    def parse(cls, value: ProjectionKind | str) -> ProjectionKind:
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as error:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown visual projection {value!r}; expected {allowed}."
            ) from error


@dataclass(frozen=True)
class ProjectionSpec:
    """Strict full-linear or low-rank projection contract."""

    kind: ProjectionKind | str
    rank: int | None

    def __post_init__(self) -> None:
        kind = ProjectionKind.parse(self.kind)
        rank = self.rank
        if kind is ProjectionKind.FULL_LINEAR:
            if rank is not None:
                raise ValueError("Full-linear projections must set `rank` to null.")
        else:
            if isinstance(rank, bool) or rank is None or int(rank) < 1:
                raise ValueError(
                    "Low-rank projections require a positive integer rank."
                )
            rank = int(rank)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "rank", rank)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ProjectionSpec:
        if set(payload) != {"kind", "rank"}:
            raise ValueError("Projection config requires exactly `kind` and `rank`.")
        return cls(kind=payload["kind"], rank=payload["rank"])

    def contract_payload(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "rank": self.rank}


@dataclass(frozen=True)
class DinoSemanticReaderConfig:
    """Fully resolved P1 action-reader configuration with no science defaults."""

    action_hidden_dim: int
    timestep_dim: int
    proprio_dim: int
    camera_ids: tuple[str, ...]
    layer_indices: tuple[int, ...]
    temperature: float
    residual_scale: float
    query_projection: ProjectionSpec
    output_projection: ProjectionSpec
    memory_contract_sha256: str

    def __post_init__(self) -> None:
        dimensions = {
            "action_hidden_dim": self.action_hidden_dim,
            "timestep_dim": self.timestep_dim,
            "proprio_dim": self.proprio_dim,
        }
        for name, value in dimensions.items():
            if isinstance(value, bool) or int(value) < 1:
                raise ValueError(f"`{name}` must be a positive integer.")
            object.__setattr__(self, name, int(value))
        camera_ids = tuple(str(camera_id) for camera_id in self.camera_ids)
        if not camera_ids or any(not camera_id for camera_id in camera_ids):
            raise ValueError("At least one non-empty camera ID is required.")
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError("Reader camera IDs must be unique and ordered.")
        layer_indices = tuple(int(index) for index in self.layer_indices)
        if not layer_indices or any(index < 0 for index in layer_indices):
            raise ValueError("At least one non-negative injection layer is required.")
        if tuple(sorted(set(layer_indices))) != layer_indices:
            raise ValueError("Injection layers must be unique and sorted.")
        temperature = float(self.temperature)
        residual_scale = float(self.residual_scale)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Cosine temperature must be finite and positive.")
        if not math.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("Residual scale must be finite and positive.")
        if not isinstance(self.query_projection, ProjectionSpec):
            raise TypeError("`query_projection` must be a ProjectionSpec.")
        if not isinstance(self.output_projection, ProjectionSpec):
            raise TypeError("`output_projection` must be a ProjectionSpec.")
        object.__setattr__(self, "camera_ids", camera_ids)
        object.__setattr__(self, "layer_indices", layer_indices)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "residual_scale", residual_scale)
        object.__setattr__(
            self,
            "memory_contract_sha256",
            validate_sha256(
                self.memory_contract_sha256,
                label="Native memory contract SHA256",
            ),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> DinoSemanticReaderConfig:
        required = {
            "action_hidden_dim",
            "timestep_dim",
            "proprio_dim",
            "camera_ids",
            "layer_indices",
            "temperature",
            "residual_scale",
            "query_projection",
            "output_projection",
            "memory_contract_sha256",
        }
        if set(payload) != required:
            missing = sorted(required - set(payload))
            unknown = sorted(set(payload) - required)
            raise ValueError(
                f"Invalid semantic reader fields; missing={missing}, unknown={unknown}."
            )
        query_payload = payload["query_projection"]
        output_payload = payload["output_projection"]
        if not isinstance(query_payload, Mapping) or not isinstance(
            output_payload, Mapping
        ):
            raise TypeError("Projection configurations must be mappings.")
        return cls(
            action_hidden_dim=payload["action_hidden_dim"],
            timestep_dim=payload["timestep_dim"],
            proprio_dim=payload["proprio_dim"],
            camera_ids=tuple(payload["camera_ids"]),
            layer_indices=tuple(payload["layer_indices"]),
            temperature=payload["temperature"],
            residual_scale=payload["residual_scale"],
            query_projection=ProjectionSpec.from_mapping(query_payload),
            output_projection=ProjectionSpec.from_mapping(output_payload),
            memory_contract_sha256=payload["memory_contract_sha256"],
        )

    def contract_payload(self) -> dict[str, Any]:
        return {
            "action_hidden_dim": self.action_hidden_dim,
            "timestep_dim": self.timestep_dim,
            "proprio_dim": self.proprio_dim,
            "camera_ids": list(self.camera_ids),
            "layer_indices": list(self.layer_indices),
            "temperature": self.temperature,
            "residual_scale": self.residual_scale,
            "query_projection": self.query_projection.contract_payload(),
            "output_projection": self.output_projection.contract_payload(),
            "memory_contract_sha256": self.memory_contract_sha256,
        }


@dataclass(frozen=True)
class DinoWanValueReaderConfig:
    """Resolved P6 router-to-frozen-Wan-value reader configuration."""

    action_hidden_dim: int
    camera_ids: tuple[str, ...]
    layer_indices: tuple[int, ...]
    temperature: float
    beta_max: float
    query_projection: ProjectionSpec
    memory_contract_sha256: str
    spatial_metadata: WanValueSpatialMetadata
    transport: DinoWanSpatialTransport
    camera_mass: tuple[float, ...]

    def __post_init__(self) -> None:
        if isinstance(self.action_hidden_dim, bool) or int(self.action_hidden_dim) < 1:
            raise ValueError("`action_hidden_dim` must be a positive integer.")
        object.__setattr__(self, "action_hidden_dim", int(self.action_hidden_dim))
        camera_ids = tuple(str(item) for item in self.camera_ids)
        if not camera_ids or len(set(camera_ids)) != len(camera_ids):
            raise ValueError("P6 camera IDs must be non-empty, unique, and ordered.")
        if camera_ids != self.spatial_metadata.camera_order:
            raise ValueError("P6 camera order differs from the Wan spatial contract.")
        if camera_ids != self.transport.camera_ids:
            raise ValueError("P6 camera order differs from the transport matrix.")
        if (
            self.transport.spatial_contract_sha256
            != self.spatial_metadata.spatial_transport_contract_sha256
        ):
            raise ValueError("P6 transport and Wan spatial contract hashes differ.")
        layer_indices = tuple(int(item) for item in self.layer_indices)
        if not layer_indices or tuple(sorted(set(layer_indices))) != layer_indices:
            raise ValueError(
                "P6 injection layers must be non-empty, unique, and sorted."
            )
        if any(item < 0 for item in layer_indices):
            raise ValueError("P6 injection layers cannot be negative.")
        temperature = float(self.temperature)
        beta_max = float(self.beta_max)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("P6 router temperature must be finite and positive.")
        if not math.isfinite(beta_max) or beta_max <= 0:
            raise ValueError("P6 beta bound must be finite and positive.")
        if not isinstance(self.query_projection, ProjectionSpec):
            raise TypeError("P6 query projection must be a ProjectionSpec.")
        camera_mass = tuple(float(value) for value in self.camera_mass)
        if len(camera_mass) != len(camera_ids):
            raise ValueError("P6 camera mass must contain one value per camera.")
        if any(not math.isfinite(value) or value <= 0 for value in camera_mass):
            raise ValueError("P6 base camera masses must be finite and positive.")
        total_mass = sum(camera_mass)
        camera_mass = tuple(value / total_mass for value in camera_mass)
        object.__setattr__(self, "camera_ids", camera_ids)
        object.__setattr__(self, "layer_indices", layer_indices)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "beta_max", beta_max)
        object.__setattr__(self, "camera_mass", camera_mass)
        object.__setattr__(
            self,
            "memory_contract_sha256",
            validate_sha256(
                self.memory_contract_sha256,
                label="Native memory contract SHA256",
            ),
        )

    def contract_payload(self) -> dict[str, Any]:
        """Return a checkpoint-stable, tensor-free P6 configuration."""

        return {
            "schema": "fastwam-p6-reader-config-v1",
            "action_hidden_dim": self.action_hidden_dim,
            "camera_ids": list(self.camera_ids),
            "layer_indices": list(self.layer_indices),
            "temperature": self.temperature,
            "beta_max": self.beta_max,
            "query_projection": self.query_projection.contract_payload(),
            "memory_contract_sha256": self.memory_contract_sha256,
            "spatial_transport_contract_sha256": (
                self.spatial_metadata.spatial_transport_contract_sha256
            ),
            "transport_sha256": self.transport.transport_sha256,
            "camera_mass": list(self.camera_mass),
            "wan_value_source": "video_cache_current_prefix",
            "output_projection": "frozen_action_self_attn_o_weight_only",
            "output_bias": False,
            "reuse_base_gate_msa": True,
            "beta_parameterization": "bounded_tanh_zero_init",
        }


class FactorizedLinear(nn.Module):
    """Bias-free low-rank projection with explicit initialization semantics."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        zero_output: bool,
    ) -> None:
        super().__init__()
        if rank > min(in_features, out_features):
            raise ValueError("Projection rank cannot exceed its input/output width.")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        nn.init.xavier_uniform_(self.down.weight)
        if zero_output:
            nn.init.zeros_(self.up.weight)
        else:
            nn.init.xavier_uniform_(self.up.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(inputs))


def _build_projection(
    in_features: int,
    out_features: int,
    spec: ProjectionSpec,
    *,
    zero_output: bool,
) -> nn.Module:
    if spec.kind is ProjectionKind.LOW_RANK:
        assert spec.rank is not None
        return FactorizedLinear(
            in_features,
            out_features,
            spec.rank,
            zero_output=zero_output,
        )
    projection = nn.Linear(in_features, out_features, bias=False)
    if zero_output:
        nn.init.zeros_(projection.weight)
    else:
        nn.init.xavier_uniform_(projection.weight)
    return projection


class NativeDinoRouter(nn.Module):
    """Map exact ActionDiT modulated inputs to per-view native patch weights."""

    def __init__(
        self,
        *,
        action_hidden_dim: int,
        temperature: float,
        projection: ProjectionSpec,
        camera_ids: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.action_hidden_dim = int(action_hidden_dim)
        self.temperature = float(temperature)
        self.camera_ids = tuple(camera_ids)
        if self.action_hidden_dim < 1:
            raise ValueError("Action hidden width must be positive.")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("Cosine temperature must be finite and positive.")
        if not self.camera_ids or len(set(self.camera_ids)) != len(self.camera_ids):
            raise ValueError("Router camera IDs must be non-empty and unique.")
        self.query_projection = _build_projection(
            self.action_hidden_dim,
            DINO_V3_NATIVE_DIM,
            projection,
            zero_output=False,
        )
        self.router_contract_sha256 = contract_sha256(
            {
                "kind": NATIVE_DINO_ROUTER_KIND,
                "query_source": QUERY_SOURCE_CONTRACT,
                "action_hidden_dim": self.action_hidden_dim,
                "native_dim": DINO_V3_NATIVE_DIM,
                "projection": projection.contract_payload(),
                "temperature": self.temperature,
                "score_dtype": "float32",
                "per_camera_softmax": True,
                "camera_ids": list(self.camera_ids),
            }
        )

    def forward(
        self,
        context: ActionLayerReadContext,
        memory: NativePatchMemory,
    ) -> RoutingWeights:
        if context.modulated_attn_input.shape[-1] != self.action_hidden_dim:
            raise ValueError("Router action hidden width does not match its contract.")
        if memory.camera_ids != self.camera_ids:
            raise ValueError("Router and native memory camera order differ.")
        if memory.tokens.device != context.modulated_attn_input.device:
            raise ValueError("Router queries and native memory must share a device.")
        queries = self.query_projection(context.modulated_attn_input).float()
        query_norms = torch.linalg.vector_norm(queries, dim=-1, keepdim=True)
        if not bool(torch.isfinite(queries).all().item()):
            raise ValueError("Projected DINO queries contain non-finite values.")
        if bool((query_norms == 0).any().item()):
            raise ValueError("Projected DINO queries must have non-zero row norms.")
        queries = queries / query_norms
        keys = F.normalize(memory.tokens.float(), dim=-1)
        scores = torch.einsum("bad,bvnd->bvan", queries, keys)
        scores = scores / self.temperature

        patch_mask = memory.patch_valid_mask
        first_patch = torch.zeros_like(patch_mask)
        first_patch[..., 0] = True
        safe_mask = patch_mask | (
            (~memory.camera_valid_mask).unsqueeze(-1) & first_patch
        )
        scores = scores.masked_fill(~safe_mask.unsqueeze(2), -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * memory.camera_valid_mask[:, :, None, None]
        weights = weights.to(dtype=memory.tokens.dtype)
        return RoutingWeights(
            weights=weights,
            patch_valid_mask=memory.patch_valid_mask,
            camera_valid_mask=memory.camera_valid_mask,
            camera_ids=memory.camera_ids,
            router_contract_sha256=self.router_contract_sha256,
        )


class VisualValueBranch(nn.Module, ABC):
    """A value consumer for a shared ephemeral native DINO routing map."""

    branch_kind: str
    branch_contract_sha256: str

    @abstractmethod
    def forward_branch(
        self,
        context: ActionLayerReadContext,
        memory: NativePatchMemory,
        routing: RoutingWeights,
    ) -> torch.Tensor:
        """Return one branch's [B,N_action,D] residual."""


class DinoSemanticValueBranch(VisualValueBranch):
    """Retrieve native DINO values and write them to ActionDiT hidden state."""

    branch_kind = DINO_SEMANTIC_BRANCH_KIND

    def __init__(
        self,
        *,
        action_hidden_dim: int,
        timestep_dim: int,
        proprio_dim: int,
        camera_ids: tuple[str, ...],
        output_projection: ProjectionSpec,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.action_hidden_dim = int(action_hidden_dim)
        self.timestep_dim = int(timestep_dim)
        self.proprio_dim = int(proprio_dim)
        self.camera_ids = tuple(camera_ids)
        self.residual_scale = float(residual_scale)
        self.output_projection = _build_projection(
            len(self.camera_ids) * DINO_V3_NATIVE_DIM,
            self.action_hidden_dim,
            output_projection,
            zero_output=True,
        )
        self.semantic_gate = nn.Linear(
            self.action_hidden_dim + self.proprio_dim + self.timestep_dim,
            1,
            bias=True,
        )
        nn.init.xavier_uniform_(self.semantic_gate.weight)
        nn.init.zeros_(self.semantic_gate.bias)
        self.branch_contract_sha256 = contract_sha256(
            {
                "kind": self.branch_kind,
                "action_hidden_dim": self.action_hidden_dim,
                "timestep_dim": self.timestep_dim,
                "proprio_dim": self.proprio_dim,
                "camera_ids": list(self.camera_ids),
                "value_source": "native_dino_patches",
                "camera_fusion": "post_retrieval_concat",
                "output_projection": output_projection.contract_payload(),
                "semantic_gate": "per_sample_per_layer_scalar_sigmoid",
                "residual_scale": self.residual_scale,
                "zero_init": "output_projection",
            }
        )

    def forward_branch(
        self,
        context: ActionLayerReadContext,
        memory: NativePatchMemory,
        routing: RoutingWeights,
    ) -> torch.Tensor:
        if (
            memory.camera_ids != self.camera_ids
            or routing.camera_ids != self.camera_ids
        ):
            raise ValueError("DINO semantic branch camera order mismatch.")
        if context.modulated_attn_input.shape[-1] != self.action_hidden_dim:
            raise ValueError("DINO semantic branch action width mismatch.")
        if context.timestep_embedding.shape[-1] != self.timestep_dim:
            raise ValueError("DINO semantic branch timestep width mismatch.")
        if context.proprio.shape[-1] != self.proprio_dim:
            raise ValueError("DINO semantic branch proprio width mismatch.")
        retrieved = torch.einsum(
            "bvan,bvnd->bvad",
            routing.weights,
            memory.tokens,
        )
        batch, views, actions, native_dim = retrieved.shape
        fused = retrieved.permute(0, 2, 1, 3).reshape(
            batch,
            actions,
            views * native_dim,
        )
        fused = fused.to(
            device=context.modulated_attn_input.device,
            dtype=context.modulated_attn_input.dtype,
        )
        projected = self.output_projection(fused)
        gate_input = torch.cat(
            (
                context.modulated_attn_input.mean(dim=1),
                context.proprio.to(
                    device=context.modulated_attn_input.device,
                    dtype=context.modulated_attn_input.dtype,
                ),
                context.timestep_embedding.to(
                    device=context.modulated_attn_input.device,
                    dtype=context.modulated_attn_input.dtype,
                ),
            ),
            dim=-1,
        )
        gate = torch.sigmoid(self.semantic_gate(gate_input)).unsqueeze(1)
        return self.residual_scale * gate * projected


class DinoRoutedWanValueBranch(VisualValueBranch):
    """Use DINO routing weights to retrieve frozen current-frame Wan values."""

    branch_kind = WAN_ROUTED_VALUE_BRANCH_KIND

    def __init__(
        self,
        *,
        action_hidden_dim: int,
        camera_ids: tuple[str, ...],
        transport: DinoWanSpatialTransport,
        spatial_metadata: WanValueSpatialMetadata,
        camera_mass: tuple[float, ...],
        beta_max: float,
    ) -> None:
        super().__init__()
        self.action_hidden_dim = int(action_hidden_dim)
        self.camera_ids = tuple(camera_ids)
        self.transport = transport
        self.spatial_metadata = spatial_metadata
        self.beta_max = float(beta_max)
        if self.action_hidden_dim < 1:
            raise ValueError("P6 action hidden width must be positive.")
        if self.camera_ids != transport.camera_ids:
            raise ValueError("P6 branch and transport camera order differ.")
        if self.camera_ids != spatial_metadata.camera_order:
            raise ValueError("P6 branch and Wan metadata camera order differ.")
        if transport.target_count != spatial_metadata.current_frame_video_tokens:
            raise ValueError("P6 transport target count differs from Wan metadata.")
        if spatial_metadata.video_value_width < 1:
            raise ValueError("P6 Wan value width must be positive.")
        if not math.isfinite(self.beta_max) or self.beta_max <= 0:
            raise ValueError("P6 beta bound must be finite and positive.")
        normalized_mass = torch.tensor(camera_mass, dtype=torch.float32)
        if normalized_mass.shape != (len(self.camera_ids),):
            raise ValueError("P6 camera mass shape differs from camera order.")
        if not bool(torch.isfinite(normalized_mass).all().item()) or bool(
            (normalized_mass <= 0).any().item()
        ):
            raise ValueError("P6 camera masses must be finite and positive.")
        normalized_mass = normalized_mass / normalized_mass.sum()
        self.register_buffer("transport_matrix", transport.matrix, persistent=False)
        self.register_buffer(
            "transport_target_valid_mask",
            transport.target_valid_mask,
            persistent=False,
        )
        self.register_buffer("base_camera_mass", normalized_mass, persistent=False)
        self.raw_beta = nn.Parameter(torch.zeros(()))
        self.branch_contract_sha256 = contract_sha256(
            {
                "kind": self.branch_kind,
                "action_hidden_dim": self.action_hidden_dim,
                "camera_ids": list(self.camera_ids),
                "transport_sha256": transport.transport_sha256,
                "spatial_transport_contract_sha256": (
                    spatial_metadata.spatial_transport_contract_sha256
                ),
                "camera_mass": normalized_mass.tolist(),
                "wan_value_source": "video_cache_current_prefix",
                "output_projection": "frozen_action_self_attn_o_weight_only",
                "output_bias": False,
                "reuse_base_gate_msa": True,
                "shared_spatial_weights_across_heads": True,
                "beta_max": self.beta_max,
                "beta_parameterization": "tanh",
                "zero_init": "raw_beta",
            }
        )

    @property
    def beta(self) -> torch.Tensor:
        """Return the bounded residual coefficient."""

        return self.beta_max * torch.tanh(self.raw_beta)

    def _transport_routing(
        self,
        memory: NativePatchMemory,
        routing: RoutingWeights,
    ) -> torch.Tensor:
        camera_valid = memory.camera_valid_mask
        prior = self.base_camera_mass.to(device=camera_valid.device)
        active_mass = prior.unsqueeze(0) * camera_valid.to(dtype=prior.dtype)
        denominator = active_mass.sum(dim=1, keepdim=True)
        if bool((denominator <= 0).any().item()):
            raise ValueError("P6 cannot route a sample with no active camera.")
        active_mass = active_mass / denominator
        matrix = self.transport_matrix.to(
            device=routing.weights.device,
            dtype=torch.float32,
        )
        transported = torch.einsum(
            "bvan,vnw->bvaw",
            routing.weights.float(),
            matrix,
        )
        combined = torch.einsum("bv,bvaw->baw", active_mass.float(), transported)
        if not bool(torch.isfinite(combined).all().item()) or bool(
            (combined < -1e-7).any().item()
        ):
            raise ValueError("P6 transported routing is non-finite or negative.")
        mass = combined.sum(dim=-1)
        if not torch.allclose(mass, torch.ones_like(mass), atol=1e-5, rtol=0):
            raise ValueError("P6 transported routing does not preserve unit mass.")
        return combined

    def resolved_runtime_shape_contract(self, *, layer_index: int) -> dict[str, Any]:
        """Return tensor-free resolved P6 shapes for one injection layer."""

        views = len(self.camera_ids)
        patch_count = (
            self.spatial_metadata.dino_patch_grid[0]
            * self.spatial_metadata.dino_patch_grid[1]
        )
        current_tokens = self.spatial_metadata.current_frame_video_tokens
        value_width = self.spatial_metadata.video_value_width
        return {
            "layer_index": int(layer_index),
            "dino_tokens_shape": ["batch", views, patch_count, DINO_V3_NATIVE_DIM],
            "wan_current_value_shape": ["batch", current_tokens, value_width],
            "transport_matrix_shape": [views, patch_count, current_tokens],
            "action_query_shape": ["batch", "action_tokens", self.action_hidden_dim],
            "routing_weights_shape": [
                "batch",
                views,
                "action_tokens",
                patch_count,
            ],
            "visual_residual_shape": [
                "batch",
                "action_tokens",
                self.action_hidden_dim,
            ],
        }

    def runtime_shape_audit(
        self,
        *,
        context: ActionLayerReadContext,
        memory: NativePatchMemory,
        routing: RoutingWeights,
        residual: torch.Tensor,
    ) -> dict[str, Any]:
        """Capture actual P6 forward shapes without retaining tensor objects."""

        current_values = context.video.value[:, : context.video.current_frame_tokens]
        return {
            "layer_index": int(context.layer_index),
            "dino_tokens_shape": list(memory.tokens.shape),
            "wan_current_value_shape": list(current_values.shape),
            "transport_matrix_shape": list(self.transport_matrix.shape),
            "action_query_shape": list(context.modulated_attn_input.shape),
            "routing_weights_shape": list(routing.weights.shape),
            "visual_residual_shape": list(residual.shape),
        }

    def forward_branch(
        self,
        context: ActionLayerReadContext,
        memory: NativePatchMemory,
        routing: RoutingWeights,
    ) -> torch.Tensor:
        if (
            memory.camera_ids != self.camera_ids
            or routing.camera_ids != self.camera_ids
        ):
            raise ValueError("P6 branch camera order mismatch.")
        layout = context.video.layout_metadata
        if not isinstance(layout, WanValueSpatialMetadata):
            raise TypeError("P6 requires typed WanValueSpatialMetadata at runtime.")
        if (
            layout.spatial_transport_contract_sha256
            != self.spatial_metadata.spatial_transport_contract_sha256
        ):
            raise ValueError("Runtime Wan spatial contract differs from P6 transport.")
        if context.video.current_frame_tokens != layout.current_frame_video_tokens:
            raise ValueError("Runtime current-frame prefix differs from P6 metadata.")
        if context.video.value.shape[-1] != layout.video_value_width:
            raise ValueError("Runtime Wan value width differs from P6 metadata.")
        if context.base_action_output_weight is None:
            raise ValueError("P6 requires the frozen action output weight.")
        output_weight = context.base_action_output_weight
        if output_weight.requires_grad:
            raise ValueError("P6 action output weight must be frozen.")
        expected_weight_shape = (self.action_hidden_dim, layout.video_value_width)
        if tuple(output_weight.shape) != expected_weight_shape:
            raise ValueError(
                "Frozen action output weight shape mismatch: "
                f"expected {expected_weight_shape}, got {tuple(output_weight.shape)}."
            )
        transported = self._transport_routing(memory, routing)
        current_values = context.video.value[
            :, : layout.current_frame_video_tokens
        ].detach()
        retrieved = torch.einsum(
            "baw,bwd->bad",
            transported.to(dtype=current_values.dtype),
            current_values,
        )
        projected = F.linear(retrieved, output_weight.detach(), bias=None)
        base_gate = context.base_gate_msa.detach().to(
            device=projected.device,
            dtype=projected.dtype,
        )
        if base_gate.ndim == 2:
            base_gate = base_gate.unsqueeze(1)
        if base_gate.shape != (projected.shape[0], 1, projected.shape[-1]):
            raise ValueError("P6 base gate_msa shape does not match action residual.")
        return self.beta.to(dtype=projected.dtype) * base_gate * projected


class RoutedVisualReader(ActionVisualReader):
    """Run one router per layer and share its weights across value branches."""

    def __init__(
        self,
        *,
        routers: Mapping[int, NativeDinoRouter],
        branches: Mapping[int, Sequence[VisualValueBranch]],
        memory_contract_sha256: str,
        reader_kind: str,
        parameter_family: str = VISUAL_READER_PARAMETER_FAMILY,
    ) -> None:
        super().__init__()
        layer_indices = tuple(sorted(int(index) for index in routers))
        if (
            not layer_indices
            or tuple(sorted(int(index) for index in branches)) != layer_indices
        ):
            raise ValueError(
                "Every visual-reader layer needs one router and branch set."
            )
        if any(not branches[index] for index in layer_indices):
            raise ValueError(
                "Every visual-reader layer needs at least one value branch."
            )
        self.routers = nn.ModuleDict(
            {str(index): routers[index] for index in layer_indices}
        )
        self.branches = nn.ModuleDict(
            {
                str(index): nn.ModuleList(list(branches[index]))
                for index in layer_indices
            }
        )
        self._injection_layer_indices = layer_indices
        self.memory_contract_sha256 = validate_sha256(
            memory_contract_sha256,
            label="Native memory contract SHA256",
        )
        self.reader_kind = str(reader_kind)
        if not self.reader_kind:
            raise ValueError("Visual reader kind cannot be empty.")
        self.parameter_family = str(parameter_family).strip()
        if not self.parameter_family:
            raise ValueError("Visual reader parameter family cannot be empty.")
        self.reader_contract_sha256 = contract_sha256(
            {
                "schema": "fastwam-routed-visual-reader-v1",
                "reader_kind": self.reader_kind,
                "memory_contract_sha256": self.memory_contract_sha256,
                "layers": [
                    {
                        "index": index,
                        "router_contract_sha256": routers[index].router_contract_sha256,
                        "branch_contract_sha256": [
                            branch.branch_contract_sha256 for branch in branches[index]
                        ],
                    }
                    for index in layer_indices
                ],
            }
        )
        self._runtime_shape_audits: dict[int, dict[str, Any]] = {}

    @property
    def injection_layer_indices(self) -> tuple[int, ...]:
        return self._injection_layer_indices

    def forward_layer(
        self,
        context: ActionLayerReadContext,
        memory: NativePatchMemory,
    ) -> VisualResidual:
        key = str(context.layer_index)
        if key not in self.routers:
            raise ValueError(
                f"Layer {context.layer_index} is not configured for visual injection."
            )
        if memory.memory_contract_sha256 != self.memory_contract_sha256:
            raise ValueError("Native memory contract does not match the visual reader.")
        routing = self.routers[key](context, memory)
        branch_modules = self.branches[key]
        residuals = [
            branch.forward_branch(context, memory, routing) for branch in branch_modules
        ]
        shape_audits = [
            branch.runtime_shape_audit(
                context=context,
                memory=memory,
                routing=routing,
                residual=residual,
            )
            for branch, residual in zip(branch_modules, residuals, strict=True)
            if isinstance(branch, DinoRoutedWanValueBranch)
        ]
        if shape_audits:
            if len(shape_audits) != 1:
                raise ValueError("P6 requires exactly one Wan value branch per layer.")
            self._runtime_shape_audits[context.layer_index] = shape_audits[0]
        total = residuals[0]
        for residual in residuals[1:]:
            total = total + residual
        if total.shape != context.post_block_hidden.shape:
            raise ValueError(
                "Visual branch residual shape does not match ActionDiT hidden."
            )
        return VisualResidual(
            tensor=total,
            layer_index=context.layer_index,
            branch_kinds=tuple(branch.branch_kind for branch in branch_modules),
        )

    def resolved_runtime_shape_contract(self) -> dict[str, Any] | None:
        """Return resolved P6 shapes without materializing runtime tensors."""

        layers: list[dict[str, Any]] = []
        for layer_index in self.injection_layer_indices:
            p6_branches = [
                branch
                for branch in self.branches[str(layer_index)]
                if isinstance(branch, DinoRoutedWanValueBranch)
            ]
            if not p6_branches:
                continue
            if len(p6_branches) != 1:
                raise ValueError("P6 requires exactly one Wan value branch per layer.")
            layers.append(
                p6_branches[0].resolved_runtime_shape_contract(layer_index=layer_index)
            )
        if not layers:
            return None
        return {
            "schema": "fastwam-p6-resolved-runtime-shapes-v1",
            "tensor_free": True,
            "layers": layers,
        }

    def runtime_shape_audit(self) -> dict[str, Any] | None:
        """Return actual shapes observed by P6 forwards, without any tensors."""

        if not self._runtime_shape_audits:
            return None
        return {
            "schema": "fastwam-p6-observed-runtime-shapes-v1",
            "tensor_free": True,
            "layers": [
                dict(self._runtime_shape_audits[index])
                for index in sorted(self._runtime_shape_audits)
            ],
        }


def build_dino_semantic_reader(
    config: DinoSemanticReaderConfig,
) -> RoutedVisualReader:
    """Build independent per-layer routers and P1 semantic value branches."""

    routers: dict[int, NativeDinoRouter] = {}
    branches: dict[int, tuple[VisualValueBranch, ...]] = {}
    for layer_index in config.layer_indices:
        routers[layer_index] = NativeDinoRouter(
            action_hidden_dim=config.action_hidden_dim,
            temperature=config.temperature,
            projection=config.query_projection,
            camera_ids=config.camera_ids,
        )
        branches[layer_index] = (
            DinoSemanticValueBranch(
                action_hidden_dim=config.action_hidden_dim,
                timestep_dim=config.timestep_dim,
                proprio_dim=config.proprio_dim,
                camera_ids=config.camera_ids,
                output_projection=config.output_projection,
                residual_scale=config.residual_scale,
            ),
        )
    return RoutedVisualReader(
        routers=routers,
        branches=branches,
        memory_contract_sha256=config.memory_contract_sha256,
        reader_kind=DINO_SEMANTIC_READER_KIND,
    )


def build_dino_wan_value_reader(
    config: DinoWanValueReaderConfig,
) -> RoutedVisualReader:
    """Build independent P6 routers with frozen-Wan value branches."""

    routers: dict[int, NativeDinoRouter] = {}
    branches: dict[int, tuple[VisualValueBranch, ...]] = {}
    for layer_index in config.layer_indices:
        routers[layer_index] = NativeDinoRouter(
            action_hidden_dim=config.action_hidden_dim,
            temperature=config.temperature,
            projection=config.query_projection,
            camera_ids=config.camera_ids,
        )
        branches[layer_index] = (
            DinoRoutedWanValueBranch(
                action_hidden_dim=config.action_hidden_dim,
                camera_ids=config.camera_ids,
                transport=config.transport,
                spatial_metadata=config.spatial_metadata,
                camera_mass=config.camera_mass,
                beta_max=config.beta_max,
            ),
        )
    reader = RoutedVisualReader(
        routers=routers,
        branches=branches,
        memory_contract_sha256=config.memory_contract_sha256,
        reader_kind=DINOV3_ROUTER_WAN_VALUE_READER_KIND,
        parameter_family=P6_VISUAL_ROUTER_PARAMETER_FAMILY,
    )
    expected_hash = contract_sha256(
        {
            "reader": reader.reader_contract_sha256,
            "config": config.contract_payload(),
        }
    )
    # Bind the transport/camera-mass contract at the top-level reader too.
    reader.reader_contract_sha256 = expected_hash
    return reader
