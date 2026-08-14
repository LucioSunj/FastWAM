"""Reusable native DINO router and pluggable ActionDiT value branches."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
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
    NativePatchMemory,
    PatchRoutingWeights,
    RoutingWeights,
    SpatialPatchMemory,
    VISUAL_READER_STATE_SCHEMA_V2,
    VisualResidual,
    contract_sha256,
    validate_sha256,
)

NATIVE_DINO_ROUTER_KIND = "native-dino-cosine-router-v1"
DINO_SEMANTIC_BRANCH_KIND = "dino-native-semantic-value-v1"
DINO_SEMANTIC_READER_KIND = "dinov3-native-semantic-reader-v1"
QUERY_SOURCE_CONTRACT = "base-modulated-self-attention-input-v1"
NATIVE_PATCH_ROUTER_KIND = "native-patch-cosine-router-v2"
SPATIAL_SEMANTIC_BRANCH_KIND = "native-spatial-semantic-value-v2"
SPATIAL_SEMANTIC_READER_KIND = "native-spatial-semantic-reader-v2"


class DinoContributionDiagnosticsCollector:
    """Collect detached per-layer DINO readout diagnostics for offline audits.

    The collector is deliberately a plain Python object rather than an
    ``nn.Module`` so enabling diagnostics cannot add parameters, buffers, or
    checkpoint state. Records retain bounded per-batch CPU tensors; callers
    own aggregation and artifact serialization.
    """

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def clear(self) -> None:
        """Discard all previously captured batches."""

        self.records.clear()

    @torch.no_grad()
    def record(
        self,
        *,
        context: ActionLayerReadContext,
        routing: RoutingWeights | PatchRoutingWeights,
        gate_logits: torch.Tensor,
        effective_gate: torch.Tensor,
        projected: torch.Tensor,
        effective_residual: torch.Tensor,
    ) -> None:
        """Append one detached diagnostic record for a reader layer."""

        weights = routing.weights.detach().float()
        valid = routing.camera_valid_mask.detach().cpu()
        safe_weights = weights.clamp_min(torch.finfo(torch.float32).tiny)
        entropy = -(weights * safe_weights.log()).sum(dim=-1)
        top1 = weights.amax(dim=-1)
        top5 = weights.topk(k=min(5, weights.shape[-1]), dim=-1).values.sum(dim=-1)
        effective_patches = entropy.exp()
        hidden_norm = torch.linalg.vector_norm(
            context.post_block_hidden.detach().float(), dim=-1
        ).clamp_min(torch.finfo(torch.float32).eps)
        projected_norm = torch.linalg.vector_norm(projected.detach().float(), dim=-1)
        residual_norm = torch.linalg.vector_norm(
            effective_residual.detach().float(), dim=-1
        )
        values = (
            entropy,
            top1,
            top5,
            effective_patches,
            hidden_norm,
            projected_norm,
            residual_norm,
            gate_logits,
            effective_gate,
        )
        if not all(bool(torch.isfinite(value).all().item()) for value in values):
            raise FloatingPointError("Non-finite DINO contribution diagnostic.")
        self.records.append(
            {
                "layer_index": int(context.layer_index),
                "camera_ids": tuple(routing.camera_ids),
                "camera_valid_mask": valid.clone(),
                "gate_logits": gate_logits.detach().float().cpu().clone(),
                "effective_gate": effective_gate.detach().float().cpu().clone(),
                "projected_norm": projected_norm.cpu(),
                "effective_residual_norm": residual_norm.cpu(),
                "projected_residual_over_hidden": (projected_norm / hidden_norm).cpu(),
                "effective_residual_over_hidden": (residual_norm / hidden_norm).cpu(),
                "effective_residual_sum": (
                    effective_residual.detach().float().sum(dim=0).cpu()
                ),
                "effective_residual_square_sum": (
                    effective_residual.detach().float().square().sum(dim=0).cpu()
                ),
                "sample_count": int(effective_residual.shape[0]),
                "attention_entropy": entropy.cpu(),
                "attention_top1": top1.cpu(),
                "attention_top5": top5.cpu(),
                "effective_patch_count": effective_patches.cpu(),
            }
        )


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
    semantic_gate_floor: float = 0.0
    semantic_gate_temperature: float = 1.0

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
        semantic_gate_floor = float(self.semantic_gate_floor)
        semantic_gate_temperature = float(self.semantic_gate_temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Cosine temperature must be finite and positive.")
        if not math.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("Residual scale must be finite and positive.")
        if (
            not math.isfinite(semantic_gate_floor)
            or semantic_gate_floor < 0
            or semantic_gate_floor >= 1
        ):
            raise ValueError("Semantic gate floor must be finite and lie in [0,1).")
        if (
            not math.isfinite(semantic_gate_temperature)
            or semantic_gate_temperature < 1
        ):
            raise ValueError("Semantic gate temperature must be finite and >= 1.")
        if not isinstance(self.query_projection, ProjectionSpec):
            raise TypeError("`query_projection` must be a ProjectionSpec.")
        if not isinstance(self.output_projection, ProjectionSpec):
            raise TypeError("`output_projection` must be a ProjectionSpec.")
        object.__setattr__(self, "camera_ids", camera_ids)
        object.__setattr__(self, "layer_indices", layer_indices)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "residual_scale", residual_scale)
        object.__setattr__(self, "semantic_gate_floor", semantic_gate_floor)
        object.__setattr__(
            self,
            "semantic_gate_temperature",
            semantic_gate_temperature,
        )
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
        optional = {"semantic_gate_floor", "semantic_gate_temperature"}
        if not required.issubset(payload) or not set(payload).issubset(
            required | optional
        ):
            missing = sorted(required - set(payload))
            unknown = sorted(set(payload) - required - optional)
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
            semantic_gate_floor=payload.get("semantic_gate_floor", 0.0),
            semantic_gate_temperature=payload.get(
                "semantic_gate_temperature",
                1.0,
            ),
        )

    def contract_payload(self) -> dict[str, Any]:
        payload = {
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
        # Keep the original reader/config contract byte-for-byte stable when the
        # contribution controls are neutral so historical checkpoints remain
        # loadable under their original config.
        if self.semantic_gate_floor != 0.0:
            payload["semantic_gate_floor"] = self.semantic_gate_floor
        if self.semantic_gate_temperature != 1.0:
            payload["semantic_gate_temperature"] = self.semantic_gate_temperature
        return payload


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
        *,
        dino_keep_mask: torch.Tensor | None = None,
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
        tokens = memory.tokens
        patch_mask = memory.patch_valid_mask
        camera_mask = memory.camera_valid_mask
        if dino_keep_mask is not None:
            if (
                dino_keep_mask.shape != (memory.tokens.shape[0],)
                or dino_keep_mask.dtype != torch.bool
                or dino_keep_mask.device != memory.tokens.device
            ):
                raise ValueError("DINO keep mask must be bool [B] on the memory device.")
            tokens = tokens * dino_keep_mask[:, None, None, None]
            patch_mask = patch_mask & dino_keep_mask[:, None, None]
            camera_mask = camera_mask & dino_keep_mask[:, None]
        keys = F.normalize(tokens.float(), dim=-1)
        scores = torch.einsum("bad,bvnd->bvan", queries, keys)
        scores = scores / self.temperature

        first_patch = F.one_hot(
            memory.patch_valid_mask.to(torch.int64).argmax(dim=-1),
            num_classes=memory.patch_valid_mask.shape[-1],
        ).to(torch.bool)
        safe_mask = patch_mask | (
            (~camera_mask).unsqueeze(-1) & first_patch
        )
        scores = scores.masked_fill(~safe_mask.unsqueeze(2), -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * camera_mask[:, :, None, None]
        weights = weights.to(dtype=memory.tokens.dtype)
        return RoutingWeights(
            weights=weights,
            patch_valid_mask=patch_mask,
            camera_valid_mask=camera_mask,
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
        *,
        dino_keep_mask: torch.Tensor | None = None,
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
        semantic_gate_floor: float = 0.0,
        semantic_gate_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.action_hidden_dim = int(action_hidden_dim)
        self.timestep_dim = int(timestep_dim)
        self.proprio_dim = int(proprio_dim)
        self.camera_ids = tuple(camera_ids)
        self.residual_scale = float(residual_scale)
        self.semantic_gate_floor = float(semantic_gate_floor)
        self.semantic_gate_temperature = float(semantic_gate_temperature)
        self._diagnostics_collector: DinoContributionDiagnosticsCollector | None = None
        if (
            not math.isfinite(self.semantic_gate_floor)
            or self.semantic_gate_floor < 0
            or self.semantic_gate_floor >= 1
        ):
            raise ValueError("Semantic gate floor must be finite and lie in [0,1).")
        if (
            not math.isfinite(self.semantic_gate_temperature)
            or self.semantic_gate_temperature < 1
        ):
            raise ValueError("Semantic gate temperature must be finite and >= 1.")
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
        branch_contract = {
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
        # Neutral controls deliberately retain the historical branch hash.
        if self.semantic_gate_floor != 0.0:
            branch_contract["semantic_gate_floor"] = self.semantic_gate_floor
        if self.semantic_gate_temperature != 1.0:
            branch_contract["semantic_gate_temperature"] = (
                self.semantic_gate_temperature
            )
            branch_contract["semantic_gate_temperature_mode"] = "negative_logits_only"
        self.branch_contract_sha256 = contract_sha256(branch_contract)

    def effective_gate(self, logits: torch.Tensor) -> torch.Tensor:
        """Map semantic-gate logits to the configured effective gate value."""

        if self.semantic_gate_temperature == 1.0:
            gate = torch.sigmoid(logits)
        else:
            tempered_logits = torch.where(
                logits < 0,
                logits / self.semantic_gate_temperature,
                logits,
            )
            gate = torch.sigmoid(tempered_logits)
        if self.semantic_gate_floor != 0.0:
            gate = gate * (1.0 - self.semantic_gate_floor)
            gate = gate + self.semantic_gate_floor
        return gate

    def forward_branch(
        self,
        context: ActionLayerReadContext,
        memory: NativePatchMemory,
        routing: RoutingWeights,
        *,
        dino_keep_mask: torch.Tensor | None = None,
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
        gate_logits = self.semantic_gate(gate_input)
        effective_gate = self.effective_gate(gate_logits)
        residual = self.residual_scale * effective_gate.unsqueeze(1) * projected
        if dino_keep_mask is not None:
            residual = residual * dino_keep_mask[:, None, None]
        collector = self._diagnostics_collector
        if collector is not None:
            collector.record(
                context=context,
                routing=routing,
                gate_logits=gate_logits,
                effective_gate=effective_gate,
                projected=projected,
                effective_residual=residual,
            )
        return residual


class RoutedVisualReader(ActionVisualReader):
    """Run one router per layer and share its weights across value branches."""

    def __init__(
        self,
        *,
        routers: Mapping[int, NativeDinoRouter],
        branches: Mapping[int, Sequence[VisualValueBranch]],
        memory_contract_sha256: str,
        reader_kind: str,
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

    @property
    def injection_layer_indices(self) -> tuple[int, ...]:
        return self._injection_layer_indices

    def forward_layer(
        self,
        context: ActionLayerReadContext,
        memory: NativePatchMemory,
        *,
        dino_keep_mask: torch.Tensor | None = None,
    ) -> VisualResidual:
        key = str(context.layer_index)
        if key not in self.routers:
            raise ValueError(
                f"Layer {context.layer_index} is not configured for visual injection."
            )
        if memory.memory_contract_sha256 != self.memory_contract_sha256:
            raise ValueError("Native memory contract does not match the visual reader.")
        routing = self.routers[key](
            context,
            memory,
            dino_keep_mask=dino_keep_mask,
        )
        branch_modules = self.branches[key]
        if dino_keep_mask is None:
            residuals = [
                branch.forward_branch(context, memory, routing)
                for branch in branch_modules
            ]
        else:
            residuals = [
                branch.forward_branch(
                    context,
                    memory,
                    routing,
                    dino_keep_mask=dino_keep_mask,
                )
                for branch in branch_modules
            ]
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

    @contextmanager
    def capture_diagnostics(
        self,
        collector: DinoContributionDiagnosticsCollector,
    ) -> Iterator[DinoContributionDiagnosticsCollector]:
        """Temporarily capture detached V1 semantic-branch diagnostics."""

        if not isinstance(collector, DinoContributionDiagnosticsCollector):
            raise TypeError("DINO diagnostics require the canonical collector.")
        semantic_branches = [
            branch
            for branches in self.branches.values()
            for branch in branches
            if isinstance(branch, DinoSemanticValueBranch)
        ]
        if not semantic_branches:
            raise ValueError("This reader has no DINO semantic branches to capture.")
        if any(
            branch._diagnostics_collector is not None for branch in semantic_branches
        ):
            raise RuntimeError("DINO diagnostics capture is already active.")
        for branch in semantic_branches:
            branch._diagnostics_collector = collector
        try:
            yield collector
        finally:
            for branch in semantic_branches:
                branch._diagnostics_collector = None


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
                semantic_gate_floor=config.semantic_gate_floor,
                semantic_gate_temperature=config.semantic_gate_temperature,
            ),
        )
    return RoutedVisualReader(
        routers=routers,
        branches=branches,
        memory_contract_sha256=config.memory_contract_sha256,
        reader_kind=DINO_SEMANTIC_READER_KIND,
    )


@dataclass(frozen=True)
class VisualPatchReaderConfig:
    """Resolved V2 reader configuration for any registered patch backbone."""

    action_hidden_dim: int
    timestep_dim: int
    proprio_dim: int
    memory_dim: int
    camera_ids: tuple[str, ...]
    layer_indices: tuple[int, ...]
    temperature: float
    residual_scale: float
    query_projection: ProjectionSpec
    output_projection: ProjectionSpec
    memory_contract_sha256: str
    semantic_gate_floor: float = 0.0
    semantic_gate_temperature: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "action_hidden_dim",
            "timestep_dim",
            "proprio_dim",
            "memory_dim",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) < 1:
                raise ValueError(f"`{name}` must be a positive integer.")
            object.__setattr__(self, name, int(value))
        camera_ids = tuple(str(value) for value in self.camera_ids)
        if not camera_ids or any(not value for value in camera_ids):
            raise ValueError("V2 reader requires a non-empty camera order.")
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError("V2 reader camera IDs must be unique and ordered.")
        layers = tuple(int(value) for value in self.layer_indices)
        if not layers or tuple(sorted(set(layers))) != layers or layers[0] < 0:
            raise ValueError(
                "V2 reader layers must be unique, sorted, and non-negative."
            )
        temperature = float(self.temperature)
        residual_scale = float(self.residual_scale)
        gate_floor = float(self.semantic_gate_floor)
        gate_temperature = float(self.semantic_gate_temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("V2 cosine temperature must be finite and positive.")
        if not math.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("V2 residual scale must be finite and positive.")
        if not math.isfinite(gate_floor) or not 0 <= gate_floor < 1:
            raise ValueError("V2 semantic gate floor must lie in [0,1).")
        if not math.isfinite(gate_temperature) or gate_temperature < 1:
            raise ValueError("V2 semantic gate temperature must be >= 1.")
        if not isinstance(self.query_projection, ProjectionSpec):
            raise TypeError("V2 query projection must be a ProjectionSpec.")
        if not isinstance(self.output_projection, ProjectionSpec):
            raise TypeError("V2 output projection must be a ProjectionSpec.")
        object.__setattr__(self, "camera_ids", camera_ids)
        object.__setattr__(self, "layer_indices", layers)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "residual_scale", residual_scale)
        object.__setattr__(self, "semantic_gate_floor", gate_floor)
        object.__setattr__(self, "semantic_gate_temperature", gate_temperature)
        object.__setattr__(
            self,
            "memory_contract_sha256",
            validate_sha256(
                self.memory_contract_sha256,
                label="V2 spatial memory contract SHA256",
            ),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> VisualPatchReaderConfig:
        """Parse an exact V2 reader mapping."""

        required = {
            "action_hidden_dim",
            "timestep_dim",
            "proprio_dim",
            "memory_dim",
            "camera_ids",
            "layer_indices",
            "temperature",
            "residual_scale",
            "query_projection",
            "output_projection",
            "memory_contract_sha256",
        }
        optional = {"semantic_gate_floor", "semantic_gate_temperature"}
        if not required.issubset(payload) or not set(payload).issubset(
            required | optional
        ):
            raise ValueError(
                "Invalid V2 reader fields; "
                f"missing={sorted(required - set(payload))}, "
                f"unknown={sorted(set(payload) - required - optional)}."
            )
        return cls(
            action_hidden_dim=payload["action_hidden_dim"],
            timestep_dim=payload["timestep_dim"],
            proprio_dim=payload["proprio_dim"],
            memory_dim=payload["memory_dim"],
            camera_ids=tuple(payload["camera_ids"]),
            layer_indices=tuple(payload["layer_indices"]),
            temperature=payload["temperature"],
            residual_scale=payload["residual_scale"],
            query_projection=ProjectionSpec.from_mapping(payload["query_projection"]),
            output_projection=ProjectionSpec.from_mapping(payload["output_projection"]),
            memory_contract_sha256=payload["memory_contract_sha256"],
            semantic_gate_floor=payload.get("semantic_gate_floor", 0.0),
            semantic_gate_temperature=payload.get(
                "semantic_gate_temperature",
                1.0,
            ),
        )

    def contract_payload(self) -> dict[str, Any]:
        """Return all behavior-affecting fields for checkpoint binding."""

        return {
            "action_hidden_dim": self.action_hidden_dim,
            "timestep_dim": self.timestep_dim,
            "proprio_dim": self.proprio_dim,
            "memory_dim": self.memory_dim,
            "camera_ids": list(self.camera_ids),
            "layer_indices": list(self.layer_indices),
            "temperature": self.temperature,
            "residual_scale": self.residual_scale,
            "query_projection": self.query_projection.contract_payload(),
            "output_projection": self.output_projection.contract_payload(),
            "memory_contract_sha256": self.memory_contract_sha256,
            "semantic_gate_floor": self.semantic_gate_floor,
            "semantic_gate_temperature": self.semantic_gate_temperature,
        }


class NativePatchRouter(nn.Module):
    """Project ActionDiT queries into a registered V2 native token space."""

    def __init__(
        self,
        *,
        action_hidden_dim: int,
        memory_dim: int,
        temperature: float,
        projection: ProjectionSpec,
        camera_ids: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.action_hidden_dim = int(action_hidden_dim)
        self.memory_dim = int(memory_dim)
        self.temperature = float(temperature)
        self.camera_ids = tuple(camera_ids)
        if min(self.action_hidden_dim, self.memory_dim) < 1:
            raise ValueError("V2 router dimensions must be positive.")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("V2 cosine temperature must be finite and positive.")
        if not self.camera_ids or len(set(self.camera_ids)) != len(self.camera_ids):
            raise ValueError("V2 router camera IDs must be non-empty and unique.")
        self.query_projection = _build_projection(
            self.action_hidden_dim,
            self.memory_dim,
            projection,
            zero_output=False,
        )
        self.router_contract_sha256 = contract_sha256(
            {
                "kind": NATIVE_PATCH_ROUTER_KIND,
                "query_source": QUERY_SOURCE_CONTRACT,
                "action_hidden_dim": self.action_hidden_dim,
                "memory_dim": self.memory_dim,
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
        memory: SpatialPatchMemory,
        *,
        dino_keep_mask: torch.Tensor | None = None,
    ) -> PatchRoutingWeights:
        if context.modulated_attn_input.shape[-1] != self.action_hidden_dim:
            raise ValueError("V2 router action hidden width differs from contract.")
        if memory.native_dim != self.memory_dim:
            raise ValueError("V2 router and memory native widths differ.")
        if memory.camera_ids != self.camera_ids:
            raise ValueError("V2 router and memory camera order differ.")
        if memory.tokens.device != context.modulated_attn_input.device:
            raise ValueError("V2 router queries and memory must share a device.")
        queries = self.query_projection(context.modulated_attn_input).float()
        query_norms = torch.linalg.vector_norm(queries, dim=-1, keepdim=True)
        if not bool(torch.isfinite(queries).all().item()) or bool(
            (query_norms == 0).any().item()
        ):
            raise ValueError("V2 projected queries must be finite and non-zero.")
        queries = queries / query_norms
        tokens = memory.tokens
        patch_mask = memory.patch_valid_mask
        camera_mask = memory.camera_valid_mask
        if dino_keep_mask is not None:
            if (
                dino_keep_mask.shape != (memory.tokens.shape[0],)
                or dino_keep_mask.dtype != torch.bool
                or dino_keep_mask.device != memory.tokens.device
            ):
                raise ValueError("DINO keep mask must be bool [B] on the memory device.")
            tokens = tokens * dino_keep_mask[:, None, None, None]
            patch_mask = patch_mask & dino_keep_mask[:, None, None]
            camera_mask = camera_mask & dino_keep_mask[:, None]
        keys = F.normalize(tokens.float(), dim=-1)
        scores = torch.einsum("bad,bvnd->bvan", queries, keys) / self.temperature
        first_patch = F.one_hot(
            memory.patch_valid_mask.to(torch.int64).argmax(dim=-1),
            num_classes=memory.patch_valid_mask.shape[-1],
        ).to(torch.bool)
        safe_mask = patch_mask | (
            (~camera_mask).unsqueeze(-1) & first_patch
        )
        scores = scores.masked_fill(~safe_mask.unsqueeze(2), -torch.inf)
        weights = torch.softmax(scores, dim=-1)
        weights = weights * camera_mask[:, :, None, None]
        return PatchRoutingWeights(
            weights=weights.to(dtype=memory.tokens.dtype),
            patch_valid_mask=patch_mask,
            camera_valid_mask=camera_mask,
            camera_ids=memory.camera_ids,
            grid=memory.grid,
            router_contract_sha256=self.router_contract_sha256,
        )


class SpatialSemanticValueBranch(nn.Module):
    """Retrieve V2 native patch values and add them to ActionDiT hidden state."""

    branch_kind = SPATIAL_SEMANTIC_BRANCH_KIND

    def __init__(
        self,
        *,
        action_hidden_dim: int,
        timestep_dim: int,
        proprio_dim: int,
        memory_dim: int,
        camera_ids: tuple[str, ...],
        output_projection: ProjectionSpec,
        residual_scale: float,
        semantic_gate_floor: float,
        semantic_gate_temperature: float,
    ) -> None:
        super().__init__()
        self.action_hidden_dim = int(action_hidden_dim)
        self.timestep_dim = int(timestep_dim)
        self.proprio_dim = int(proprio_dim)
        self.memory_dim = int(memory_dim)
        self.camera_ids = tuple(camera_ids)
        self.residual_scale = float(residual_scale)
        self.semantic_gate_floor = float(semantic_gate_floor)
        self.semantic_gate_temperature = float(semantic_gate_temperature)
        self._diagnostics_collector: DinoContributionDiagnosticsCollector | None = None
        self.output_projection = _build_projection(
            len(self.camera_ids) * self.memory_dim,
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
                "memory_dim": self.memory_dim,
                "camera_ids": list(self.camera_ids),
                "value_source": "registered_native_spatial_patches",
                "camera_fusion": "post_retrieval_concat",
                "output_projection": output_projection.contract_payload(),
                "semantic_gate": "per_sample_per_layer_scalar_sigmoid",
                "semantic_gate_floor": self.semantic_gate_floor,
                "semantic_gate_temperature": self.semantic_gate_temperature,
                "semantic_gate_temperature_mode": "negative_logits_only",
                "residual_scale": self.residual_scale,
                "zero_init": "output_projection",
            }
        )

    def effective_gate(self, logits: torch.Tensor) -> torch.Tensor:
        tempered = torch.where(
            logits < 0,
            logits / self.semantic_gate_temperature,
            logits,
        )
        gate = torch.sigmoid(tempered)
        return self.semantic_gate_floor + (1.0 - self.semantic_gate_floor) * gate

    def forward_branch(
        self,
        context: ActionLayerReadContext,
        memory: SpatialPatchMemory,
        routing: PatchRoutingWeights,
        *,
        dino_keep_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory.native_dim != self.memory_dim:
            raise ValueError("V2 semantic branch memory width changed.")
        if (
            memory.camera_ids != self.camera_ids
            or routing.camera_ids != self.camera_ids
        ):
            raise ValueError("V2 semantic branch camera order changed.")
        if context.modulated_attn_input.shape[-1] != self.action_hidden_dim:
            raise ValueError("V2 semantic branch action width changed.")
        if context.timestep_embedding.shape[-1] != self.timestep_dim:
            raise ValueError("V2 semantic branch timestep width changed.")
        if context.proprio.shape[-1] != self.proprio_dim:
            raise ValueError("V2 semantic branch proprio width changed.")
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
        gate_logits = self.semantic_gate(gate_input)
        effective_gate = self.effective_gate(gate_logits)
        residual = self.residual_scale * effective_gate.unsqueeze(1) * projected
        if dino_keep_mask is not None:
            residual = residual * dino_keep_mask[:, None, None]
        collector = self._diagnostics_collector
        if collector is not None:
            collector.record(
                context=context,
                routing=routing,
                gate_logits=gate_logits,
                effective_gate=effective_gate,
                projected=projected,
                effective_residual=residual,
            )
        return residual


class SpatialRoutedVisualReader(ActionVisualReader):
    """V2 per-layer routers and native spatial value branches."""

    reader_state_schema = VISUAL_READER_STATE_SCHEMA_V2

    def __init__(
        self,
        *,
        routers: Mapping[int, NativePatchRouter],
        branches: Mapping[int, Sequence[SpatialSemanticValueBranch]],
        memory_contract_sha256: str,
    ) -> None:
        super().__init__()
        layers = tuple(sorted(int(value) for value in routers))
        if not layers or tuple(sorted(int(value) for value in branches)) != layers:
            raise ValueError("Every V2 reader layer requires a router and branches.")
        if any(not branches[index] for index in layers):
            raise ValueError("Every V2 reader layer requires a non-empty branch set.")
        self.routers = nn.ModuleDict({str(index): routers[index] for index in layers})
        self.branches = nn.ModuleDict(
            {str(index): nn.ModuleList(list(branches[index])) for index in layers}
        )
        self._injection_layer_indices = layers
        self.memory_contract_sha256 = validate_sha256(
            memory_contract_sha256,
            label="V2 spatial memory contract SHA256",
        )
        self.reader_kind = SPATIAL_SEMANTIC_READER_KIND
        self.reader_contract_sha256 = contract_sha256(
            {
                "schema": "fastwam-routed-visual-reader-v2",
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
                    for index in layers
                ],
            }
        )

    @property
    def injection_layer_indices(self) -> tuple[int, ...]:
        return self._injection_layer_indices

    def forward_layer(
        self,
        context: ActionLayerReadContext,
        memory: NativePatchMemory | SpatialPatchMemory,
        *,
        dino_keep_mask: torch.Tensor | None = None,
    ) -> VisualResidual:
        if not isinstance(memory, SpatialPatchMemory):
            raise TypeError("The V2 reader requires SpatialPatchMemory.")
        key = str(context.layer_index)
        if key not in self.routers:
            raise ValueError(
                f"Layer {context.layer_index} is not a V2 injection layer."
            )
        if memory.memory_contract_sha256 != self.memory_contract_sha256:
            raise ValueError("V2 memory contract does not match the reader.")
        if dino_keep_mask is not None:
            if (
                dino_keep_mask.shape != (memory.tokens.shape[0],)
                or dino_keep_mask.dtype != torch.bool
                or dino_keep_mask.device != memory.tokens.device
            ):
                raise ValueError("DINO keep mask must be bool [B] on the memory device.")
        routing = self.routers[key](
            context,
            memory,
            dino_keep_mask=dino_keep_mask,
        )
        branch_modules = self.branches[key]
        residuals = [
            branch.forward_branch(
                context,
                memory,
                routing,
                dino_keep_mask=dino_keep_mask,
            )
            for branch in branch_modules
        ]
        total = residuals[0]
        for residual in residuals[1:]:
            total = total + residual
        if total.shape != context.post_block_hidden.shape:
            raise ValueError("V2 residual shape does not match ActionDiT hidden.")
        return VisualResidual(
            tensor=total,
            layer_index=context.layer_index,
            branch_kinds=tuple(branch.branch_kind for branch in branch_modules),
        )

    @contextmanager
    def capture_diagnostics(
        self,
        collector: DinoContributionDiagnosticsCollector,
    ) -> Iterator[DinoContributionDiagnosticsCollector]:
        """Temporarily capture detached V2 semantic-branch diagnostics."""

        if not isinstance(collector, DinoContributionDiagnosticsCollector):
            raise TypeError("DINO diagnostics require the canonical collector.")
        semantic_branches = [
            branch
            for branches in self.branches.values()
            for branch in branches
            if isinstance(branch, SpatialSemanticValueBranch)
        ]
        if any(
            branch._diagnostics_collector is not None for branch in semantic_branches
        ):
            raise RuntimeError("DINO diagnostics capture is already active.")
        for branch in semantic_branches:
            branch._diagnostics_collector = collector
        try:
            yield collector
        finally:
            for branch in semantic_branches:
                branch._diagnostics_collector = None


def build_visual_patch_reader(
    config: VisualPatchReaderConfig,
) -> SpatialRoutedVisualReader:
    """Build an independent V2 native-patch reader for every selected layer."""

    routers: dict[int, NativePatchRouter] = {}
    branches: dict[int, tuple[SpatialSemanticValueBranch, ...]] = {}
    for layer_index in config.layer_indices:
        routers[layer_index] = NativePatchRouter(
            action_hidden_dim=config.action_hidden_dim,
            memory_dim=config.memory_dim,
            temperature=config.temperature,
            projection=config.query_projection,
            camera_ids=config.camera_ids,
        )
        branches[layer_index] = (
            SpatialSemanticValueBranch(
                action_hidden_dim=config.action_hidden_dim,
                timestep_dim=config.timestep_dim,
                proprio_dim=config.proprio_dim,
                memory_dim=config.memory_dim,
                camera_ids=config.camera_ids,
                output_projection=config.output_projection,
                residual_scale=config.residual_scale,
                semantic_gate_floor=config.semantic_gate_floor,
                semantic_gate_temperature=config.semantic_gate_temperature,
            ),
        )
    return SpatialRoutedVisualReader(
        routers=routers,
        branches=branches,
        memory_contract_sha256=config.memory_contract_sha256,
    )
