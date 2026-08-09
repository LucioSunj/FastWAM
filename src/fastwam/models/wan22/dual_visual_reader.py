"""P7 shared-DINO routing with native-DINO and current-Wan value retrieval."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .visual_contracts import (
    DINO_V3_NATIVE_DIM,
    DINO_V3_PATCH_COUNT,
    DINO_V3_PATCH_GRID,
    ActionLayerReadContext,
    NativePatchMemory,
    RoutingWeights,
    contract_sha256,
    validate_sha256,
)
from .visual_sidecar import (
    NativeDinoRouter,
    ProjectionSpec,
    RoutedVisualReader,
    VisualValueBranch,
    _build_projection,
)

DUAL_VISUAL_READER_KIND = "shared-dino-routing-dual-retrieval-v1"
DUAL_VISUAL_READER_PARAMETER_FAMILY = "dual_visual_reader"
DINO_RETRIEVAL_BRANCH_KIND = "native-dino-dual-retrieval-value-v1"
WAN_RETRIEVAL_BRANCH_KIND = "current-wan-dual-retrieval-value-v1"
TRANSPORT_ASSET_SCHEMA = "fastwam-dino-wan-transport-v1"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor, *, dtype: torch.dtype) -> str:
    canonical = tensor.detach().to(device="cpu", dtype=dtype).contiguous()
    return hashlib.sha256(canonical.numpy().tobytes()).hexdigest()


def transport_tensor_sha256(
    *,
    transport: torch.Tensor,
    target_valid_mask: torch.Tensor,
    camera_prior: torch.Tensor,
) -> str:
    """Hash the three frozen transport tensors with explicit canonical dtypes."""

    digest = hashlib.sha256()
    for label, tensor, dtype in (
        ("transport", transport, torch.float32),
        ("target_valid_mask", target_valid_mask, torch.uint8),
        ("camera_prior", camera_prior, torch.float32),
    ):
        digest.update(label.encode("ascii"))
        digest.update(_tensor_sha256(tensor, dtype=dtype).encode("ascii"))
    return digest.hexdigest()


@dataclass(frozen=True)
class DinoWanTransportGeometry:
    """Hash-bound frozen DINO-patch to current-Wan-token transport geometry."""

    transport: torch.Tensor
    target_valid_mask: torch.Tensor
    camera_prior: torch.Tensor
    camera_ids: tuple[str, ...]
    dino_grid: tuple[int, int]
    wan_grid: tuple[int, int, int]
    asset_sha256: str
    transport_sha256: str
    geometry_contract_sha256: str

    def __post_init__(self) -> None:
        if self.transport.ndim != 3:
            raise ValueError("Transport must have shape [V,196,N_wan].")
        views, patches, wan_tokens = self.transport.shape
        if patches != DINO_V3_PATCH_COUNT:
            raise ValueError("Transport must originate from all 196 DINO patches.")
        if not self.transport.is_floating_point():
            raise TypeError("Transport must use a floating dtype.")
        if self.transport.requires_grad:
            raise ValueError("Transport must be detached and permanently frozen.")
        if not bool(torch.isfinite(self.transport).all().item()):
            raise ValueError("Transport contains non-finite values.")
        if bool((self.transport < 0).any().item()):
            raise ValueError("Transport must be non-negative.")
        row_sums = self.transport.float().sum(dim=-1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6):
            raise ValueError("Every DINO transport row must sum to one.")
        if self.target_valid_mask.shape != (views, wan_tokens):
            raise ValueError("Target-valid mask must have shape [V,N_wan].")
        if self.target_valid_mask.dtype != torch.bool:
            raise TypeError("Target-valid mask must use bool dtype.")
        if self.camera_prior.shape != (views,):
            raise ValueError("Camera prior must have shape [V].")
        if not self.camera_prior.is_floating_point():
            raise TypeError("Camera prior must use a floating dtype.")
        if self.camera_prior.requires_grad:
            raise ValueError("Camera prior must be detached and permanently frozen.")
        if not bool(torch.isfinite(self.camera_prior).all().item()):
            raise ValueError("Camera prior contains non-finite values.")
        if bool((self.camera_prior < 0).any().item()):
            raise ValueError("Camera prior must be non-negative.")
        if not math.isclose(float(self.camera_prior.sum().item()), 1.0, abs_tol=1e-6):
            raise ValueError("Camera prior must sum to one.")
        if bool((self.camera_prior == 0).any().item()):
            raise ValueError("Every configured camera needs positive prior mass.")
        camera_ids = tuple(str(camera_id) for camera_id in self.camera_ids)
        if len(camera_ids) != views or len(set(camera_ids)) != views:
            raise ValueError("Transport camera IDs must be unique and match V.")
        if tuple(self.dino_grid) != DINO_V3_PATCH_GRID:
            raise ValueError("Transport DINO source grid must be exactly 14x14.")
        wan_grid = tuple(int(value) for value in self.wan_grid)
        if len(wan_grid) != 3 or any(value < 1 for value in wan_grid):
            raise ValueError("Wan current grid must be a positive (T,H,W) triple.")
        if math.prod(wan_grid) != wan_tokens:
            raise ValueError("Wan grid product does not match transport target width.")
        masked = self.transport * self.target_valid_mask[:, None, :]
        if bool((masked.sum(dim=-1) <= 0).any().item()):
            raise ValueError(
                "Every DINO row must retain positive support after the target mask."
            )
        asset_sha256 = validate_sha256(
            self.asset_sha256, label="DINO/Wan transport asset SHA256"
        )
        expected_transport_sha256 = transport_tensor_sha256(
            transport=self.transport,
            target_valid_mask=self.target_valid_mask,
            camera_prior=self.camera_prior,
        )
        if self.transport_sha256 != expected_transport_sha256:
            raise ValueError("DINO/Wan transport tensor SHA256 mismatch.")
        expected_geometry_sha256 = contract_sha256(
            {
                "schema": TRANSPORT_ASSET_SCHEMA,
                "camera_ids": list(camera_ids),
                "dino_grid": list(DINO_V3_PATCH_GRID),
                "wan_grid": list(wan_grid),
                "asset_sha256": asset_sha256,
                "transport_sha256": expected_transport_sha256,
            }
        )
        if self.geometry_contract_sha256 != expected_geometry_sha256:
            raise ValueError("DINO/Wan geometry contract SHA256 mismatch.")
        object.__setattr__(self, "transport", self.transport.detach().clone())
        object.__setattr__(
            self,
            "target_valid_mask",
            self.target_valid_mask.detach().clone(),
        )
        object.__setattr__(
            self,
            "camera_prior",
            self.camera_prior.detach().clone(),
        )
        object.__setattr__(self, "camera_ids", camera_ids)
        object.__setattr__(self, "dino_grid", DINO_V3_PATCH_GRID)
        object.__setattr__(self, "wan_grid", wan_grid)
        object.__setattr__(self, "asset_sha256", asset_sha256)
        object.__setattr__(self, "transport_sha256", expected_transport_sha256)
        object.__setattr__(self, "geometry_contract_sha256", expected_geometry_sha256)

    @classmethod
    def from_tensors(
        cls,
        *,
        transport: torch.Tensor,
        target_valid_mask: torch.Tensor,
        camera_prior: torch.Tensor,
        camera_ids: tuple[str, ...],
        wan_grid: tuple[int, int, int],
        asset_sha256: str,
    ) -> DinoWanTransportGeometry:
        """Construct geometry while deriving both tensor and contract hashes."""

        transport_sha256 = transport_tensor_sha256(
            transport=transport,
            target_valid_mask=target_valid_mask,
            camera_prior=camera_prior,
        )
        normalized_asset = validate_sha256(
            asset_sha256, label="DINO/Wan transport asset SHA256"
        )
        geometry_contract_sha256 = contract_sha256(
            {
                "schema": TRANSPORT_ASSET_SCHEMA,
                "camera_ids": list(camera_ids),
                "dino_grid": list(DINO_V3_PATCH_GRID),
                "wan_grid": list(wan_grid),
                "asset_sha256": normalized_asset,
                "transport_sha256": transport_sha256,
            }
        )
        return cls(
            transport=transport,
            target_valid_mask=target_valid_mask,
            camera_prior=camera_prior,
            camera_ids=camera_ids,
            dino_grid=DINO_V3_PATCH_GRID,
            wan_grid=wan_grid,
            asset_sha256=normalized_asset,
            transport_sha256=transport_sha256,
            geometry_contract_sha256=geometry_contract_sha256,
        )

    @property
    def wan_token_count(self) -> int:
        return math.prod(self.wan_grid)

    def effective_transport(
        self,
        *,
        batch_size: int,
        camera_valid_mask: torch.Tensor,
        target_valid_mask: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply dynamic masks, fail closed on lost support, and renormalize rows."""

        views = len(self.camera_ids)
        if camera_valid_mask.shape != (batch_size, views):
            raise ValueError("Runtime camera mask does not match transport geometry.")
        if bool((~camera_valid_mask.any(dim=1)).any().item()):
            raise ValueError("Every sample needs at least one active P7 camera.")
        if target_valid_mask is None:
            target_mask = self.target_valid_mask.to(device=device)[None].expand(
                batch_size, -1, -1
            )
        else:
            target_mask = target_valid_mask.to(device=device)
            if target_mask.shape == self.target_valid_mask.shape:
                target_mask = target_mask[None].expand(batch_size, -1, -1)
            if target_mask.shape != (batch_size, views, self.wan_token_count):
                raise ValueError("Runtime target mask must have shape [B,V,N_wan].")
            if target_mask.dtype != torch.bool:
                raise TypeError("Runtime target mask must use bool dtype.")
            target_mask = target_mask & self.target_valid_mask.to(device=device)[None]
        transport = self.transport.to(device=device, dtype=torch.float32)
        effective = transport[None] * target_mask[:, :, None, :]
        support = effective.sum(dim=-1, keepdim=True)
        active_rows = camera_valid_mask.to(device=device)[:, :, None, None]
        if bool(((support <= 0) & active_rows).any().item()):
            raise ValueError(
                "A valid DINO row has no positive support after target masking."
            )
        safe_support = torch.where(active_rows, support, torch.ones_like(support))
        effective = effective / safe_support
        effective = effective * active_rows
        return effective.to(dtype=dtype), target_mask


def load_dino_wan_transport(
    path: str | Path,
    *,
    expected_asset_sha256: str,
    expected_transport_sha256: str,
) -> DinoWanTransportGeometry:
    """Load one local, hash-pinned transport artifact without code execution."""

    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"DINO/Wan transport asset not found: {artifact_path}")
    expected_asset = validate_sha256(
        expected_asset_sha256, label="Expected transport asset SHA256"
    )
    if _file_sha256(artifact_path) != expected_asset:
        raise ValueError("DINO/Wan transport asset SHA256 mismatch.")
    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    required = {
        "schema",
        "camera_ids",
        "dino_grid",
        "wan_grid",
        "transport",
        "target_valid_mask",
        "camera_prior",
        "transport_sha256",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise ValueError("DINO/Wan transport artifact has an unexpected key set.")
    if payload["schema"] != TRANSPORT_ASSET_SCHEMA:
        raise ValueError("Unsupported DINO/Wan transport asset schema.")
    if tuple(payload["dino_grid"]) != DINO_V3_PATCH_GRID:
        raise ValueError("Transport artifact does not use the native 14x14 grid.")
    expected_transport = validate_sha256(
        expected_transport_sha256, label="Expected transport tensor SHA256"
    )
    if payload["transport_sha256"] != expected_transport:
        raise ValueError("Pinned transport SHA256 differs from artifact metadata.")
    geometry = DinoWanTransportGeometry.from_tensors(
        transport=payload["transport"],
        target_valid_mask=payload["target_valid_mask"],
        camera_prior=payload["camera_prior"],
        camera_ids=tuple(payload["camera_ids"]),
        wan_grid=tuple(payload["wan_grid"]),
        asset_sha256=expected_asset,
    )
    if geometry.transport_sha256 != expected_transport:
        raise ValueError("DINO/Wan transport payload SHA256 mismatch.")
    return geometry


@dataclass(frozen=True)
class DualDinoWanReaderConfig:
    """Resolved P7 reader configuration; every scientific value is explicit."""

    action_hidden_dim: int
    camera_ids: tuple[str, ...]
    layer_indices: tuple[int, ...]
    temperature: float
    query_projection: ProjectionSpec
    dino_output_projection: ProjectionSpec
    gamma_dino_max: float
    gamma_wan_max: float
    memory_contract_sha256: str
    transport_sha256: str
    geometry_contract_sha256: str

    def __post_init__(self) -> None:
        if isinstance(self.action_hidden_dim, bool) or int(self.action_hidden_dim) < 1:
            raise ValueError("Action hidden width must be a positive integer.")
        camera_ids = tuple(str(value) for value in self.camera_ids)
        if not camera_ids or len(set(camera_ids)) != len(camera_ids):
            raise ValueError("P7 camera IDs must be non-empty, unique, and ordered.")
        layers = tuple(int(value) for value in self.layer_indices)
        if not layers or layers != tuple(sorted(set(layers))) or layers[0] < 0:
            raise ValueError("P7 layer indices must be non-empty, sorted, and unique.")
        temperature = float(self.temperature)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("P7 router temperature must be finite and positive.")
        for name in ("gamma_dino_max", "gamma_wan_max"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"`{name}` must be finite and positive.")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "action_hidden_dim", int(self.action_hidden_dim))
        object.__setattr__(self, "camera_ids", camera_ids)
        object.__setattr__(self, "layer_indices", layers)
        object.__setattr__(self, "temperature", temperature)
        for field_name, label in (
            ("memory_contract_sha256", "Native memory contract SHA256"),
            ("transport_sha256", "DINO/Wan transport SHA256"),
            ("geometry_contract_sha256", "DINO/Wan geometry contract SHA256"),
        ):
            object.__setattr__(
                self,
                field_name,
                validate_sha256(getattr(self, field_name), label=label),
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> DualDinoWanReaderConfig:
        required = {
            "action_hidden_dim",
            "camera_ids",
            "layer_indices",
            "temperature",
            "query_projection",
            "dino_output_projection",
            "gamma_dino_max",
            "gamma_wan_max",
            "memory_contract_sha256",
            "transport_sha256",
            "geometry_contract_sha256",
        }
        if set(payload) != required:
            raise ValueError(
                "Invalid P7 reader fields; "
                f"missing={sorted(required - set(payload))}, "
                f"unknown={sorted(set(payload) - required)}."
            )
        query = payload["query_projection"]
        output = payload["dino_output_projection"]
        if not isinstance(query, Mapping) or not isinstance(output, Mapping):
            raise TypeError("P7 projection configurations must be mappings.")
        return cls(
            action_hidden_dim=payload["action_hidden_dim"],
            camera_ids=tuple(payload["camera_ids"]),
            layer_indices=tuple(payload["layer_indices"]),
            temperature=payload["temperature"],
            query_projection=ProjectionSpec.from_mapping(query),
            dino_output_projection=ProjectionSpec.from_mapping(output),
            gamma_dino_max=payload["gamma_dino_max"],
            gamma_wan_max=payload["gamma_wan_max"],
            memory_contract_sha256=payload["memory_contract_sha256"],
            transport_sha256=payload["transport_sha256"],
            geometry_contract_sha256=payload["geometry_contract_sha256"],
        )

    def contract_payload(self) -> dict[str, Any]:
        return {
            "action_hidden_dim": self.action_hidden_dim,
            "camera_ids": list(self.camera_ids),
            "layer_indices": list(self.layer_indices),
            "temperature": self.temperature,
            "query_projection": self.query_projection.contract_payload(),
            "dino_output_projection": self.dino_output_projection.contract_payload(),
            "gamma_dino_max": self.gamma_dino_max,
            "gamma_wan_max": self.gamma_wan_max,
            "memory_contract_sha256": self.memory_contract_sha256,
            "transport_sha256": self.transport_sha256,
            "geometry_contract_sha256": self.geometry_contract_sha256,
        }


class DinoRetrievalValueBranch(VisualValueBranch):
    """Retrieve raw native DINO values in fixed camera slots."""

    branch_kind = DINO_RETRIEVAL_BRANCH_KIND

    def __init__(
        self,
        *,
        action_hidden_dim: int,
        camera_ids: tuple[str, ...],
        output_projection: ProjectionSpec,
        gamma_max: float,
    ) -> None:
        super().__init__()
        self.action_hidden_dim = int(action_hidden_dim)
        self.camera_ids = tuple(camera_ids)
        self.gamma_max = float(gamma_max)
        self.output_projection = _build_projection(
            len(self.camera_ids) * DINO_V3_NATIVE_DIM,
            self.action_hidden_dim,
            output_projection,
            zero_output=False,
        )
        self.gamma_raw = nn.Parameter(torch.zeros(()))
        self.branch_contract_sha256 = contract_sha256(
            {
                "kind": self.branch_kind,
                "camera_ids": list(self.camera_ids),
                "camera_fusion": "fixed_slot_post_retrieval_concat",
                "output_projection": output_projection.contract_payload(),
                "output_initialization": "nonzero_xavier",
                "gamma": "gamma_max*tanh(gamma_raw)",
                "gamma_max": self.gamma_max,
                "gamma_initialization": 0.0,
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
            raise ValueError("DINO retrieval camera order mismatch.")
        retrieved = torch.einsum("bvan,bvnd->bvad", routing.weights, memory.tokens)
        batch, views, actions, native_dim = retrieved.shape
        fused = retrieved.permute(0, 2, 1, 3).reshape(
            batch, actions, views * native_dim
        )
        projected = self.output_projection(
            fused.to(
                device=context.post_block_hidden.device,
                dtype=context.post_block_hidden.dtype,
            )
        )
        return self.gamma_max * torch.tanh(self.gamma_raw) * projected


class WanRetrievalValueBranch(VisualValueBranch):
    """Transport shared DINO routing to frozen current-frame Wan values."""

    branch_kind = WAN_RETRIEVAL_BRANCH_KIND

    def __init__(
        self,
        *,
        action_hidden_dim: int,
        geometry: DinoWanTransportGeometry,
        gamma_max: float,
    ) -> None:
        super().__init__()
        self.action_hidden_dim = int(action_hidden_dim)
        self.geometry = geometry
        self.gamma_max = float(gamma_max)
        self.gamma_raw = nn.Parameter(torch.zeros(()))
        self.register_buffer("transport", geometry.transport.detach(), persistent=False)
        self.register_buffer(
            "target_valid_mask",
            geometry.target_valid_mask.detach(),
            persistent=False,
        )
        self.register_buffer(
            "camera_prior", geometry.camera_prior.detach(), persistent=False
        )
        self.branch_contract_sha256 = contract_sha256(
            {
                "kind": self.branch_kind,
                "camera_ids": list(geometry.camera_ids),
                "geometry_contract_sha256": geometry.geometry_contract_sha256,
                "transport_sha256": geometry.transport_sha256,
                "wan_value_source": "current_frame_prefix_only",
                "wan_projection": "frozen_action_self_attention_o_weight_no_bias",
                "base_gate": "frozen_gate_msa",
                "camera_fusion": "active_prior_renormalized",
                "gamma": "gamma_max*tanh(gamma_raw)",
                "gamma_max": self.gamma_max,
                "gamma_initialization": 0.0,
            }
        )

    def _runtime_target_mask(
        self, context: ActionLayerReadContext
    ) -> torch.Tensor | None:
        metadata = context.video.layout_metadata
        if metadata is None:
            return None
        if tuple(metadata.get("wan_grid", ())) != self.geometry.wan_grid:
            raise ValueError("Runtime Wan grid differs from P7 geometry contract.")
        camera_ids = tuple(metadata.get("camera_ids", self.geometry.camera_ids))
        if camera_ids != self.geometry.camera_ids:
            raise ValueError("Runtime camera order differs from P7 geometry contract.")
        target_mask = metadata.get("target_valid_mask")
        if target_mask is not None and not isinstance(target_mask, torch.Tensor):
            raise TypeError("Runtime target-valid mask must be a tensor.")
        return target_mask

    def forward_branch(
        self,
        context: ActionLayerReadContext,
        memory: NativePatchMemory,
        routing: RoutingWeights,
    ) -> torch.Tensor:
        if memory.camera_ids != self.geometry.camera_ids:
            raise ValueError("Wan retrieval camera order mismatch.")
        if context.video.current_frame_tokens != self.geometry.wan_token_count:
            raise ValueError("Current Wan token count differs from transport geometry.")
        output_weight = context.base_action_output_weight
        if output_weight is None:
            raise ValueError("Wan retrieval requires the frozen action O weight.")
        if output_weight.requires_grad:
            raise ValueError("Wan retrieval action O weight must remain frozen.")
        batch = memory.tokens.shape[0]
        effective, _target_mask = self.geometry.effective_transport(
            batch_size=batch,
            camera_valid_mask=memory.camera_valid_mask,
            target_valid_mask=self._runtime_target_mask(context),
            device=routing.weights.device,
            dtype=routing.weights.dtype,
        )
        routed_to_wan = torch.einsum("bvan,bvnk->bvak", routing.weights, effective)
        current_values = context.video.value[
            :, : self.geometry.wan_token_count
        ].detach()
        per_view = torch.einsum("bvak,bkd->bvad", routed_to_wan, current_values)
        active_prior = self.camera_prior.to(
            device=per_view.device, dtype=per_view.dtype
        )[None] * memory.camera_valid_mask.to(dtype=per_view.dtype)
        prior_sum = active_prior.sum(dim=-1, keepdim=True)
        if bool((prior_sum <= 0).any().item()):
            raise ValueError("No positive camera prior remains after P7 masking.")
        active_prior = active_prior / prior_sum
        retrieved = torch.einsum("bv,bvad->bad", active_prior, per_view)
        projected = F.linear(
            retrieved,
            output_weight.detach().to(device=retrieved.device, dtype=retrieved.dtype),
            bias=None,
        )
        gate = context.base_gate_msa.detach().to(
            device=projected.device, dtype=projected.dtype
        )
        if gate.ndim == 2:
            gate = gate[:, None, :]
        if gate.shape != (batch, 1, self.action_hidden_dim):
            raise ValueError("Base gate_msa must have shape [B,D] or [B,1,D].")
        return self.gamma_max * torch.tanh(self.gamma_raw) * gate * projected


class DualDinoWanReader(RoutedVisualReader):
    """Routed reader with an explicit P7 checkpoint provenance contract."""

    def __init__(
        self,
        *,
        routers: Mapping[int, NativeDinoRouter],
        branches: Mapping[int, tuple[VisualValueBranch, ...]],
        config: DualDinoWanReaderConfig,
        geometry: DinoWanTransportGeometry,
    ) -> None:
        super().__init__(
            routers=routers,
            branches=branches,
            memory_contract_sha256=config.memory_contract_sha256,
            reader_kind=DUAL_VISUAL_READER_KIND,
            parameter_family=DUAL_VISUAL_READER_PARAMETER_FAMILY,
        )
        self._p7_config = config
        self._p7_geometry = geometry

    def checkpoint_contract(self) -> dict[str, Any]:
        """Return immutable P7 query/injection/geometry/gamma provenance."""

        return {
            "schema": "fastwam-p7-dual-reader-provenance-v1",
            "reader_kind": self.reader_kind,
            "reader_contract_sha256": self.reader_contract_sha256,
            "parameter_family": self.parameter_family,
            "reader_config": self._p7_config.contract_payload(),
            "transport_asset_sha256": self._p7_geometry.asset_sha256,
            "transport_sha256": self._p7_geometry.transport_sha256,
            "geometry_contract_sha256": (self._p7_geometry.geometry_contract_sha256),
            "dino_grid": self._p7_geometry.dino_grid,
            "wan_grid": self._p7_geometry.wan_grid,
            "query_source": "base-modulated-self-attention-input-v1",
            "injection_timing": "pre-block-query-post-block-additive-residual-v1",
            "wan_value_source": "current-frame-prefix-only",
            "wan_output_projection": ("frozen-action-self-attention-o-weight-no-bias"),
            "enabled_regime": "uncond-only-dispatch-before-encoder",
            "branch_mask_contract": "both-enabled-no-training-dropout-v1",
            "gate_visibility": "no-direct-visual-bank-indirect-later-action-kv-v1",
            "initialization_lineage": "common-parent-independent-zero-gamma-v1",
            "invalid_camera_policy": "active-prior-renormalize-or-fail-closed-v1",
            "zero_delta": "gamma_dino=gamma_wan=0",
        }


def build_dual_dino_wan_reader(
    config: DualDinoWanReaderConfig,
    geometry: DinoWanTransportGeometry,
) -> DualDinoWanReader:
    """Build P7 layers with one shared router feeding both retrieval branches."""

    if config.camera_ids != geometry.camera_ids:
        raise ValueError("P7 reader and transport camera order differ.")
    if config.transport_sha256 != geometry.transport_sha256:
        raise ValueError("P7 reader transport SHA256 mismatch.")
    if config.geometry_contract_sha256 != geometry.geometry_contract_sha256:
        raise ValueError("P7 reader geometry contract SHA256 mismatch.")
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
            DinoRetrievalValueBranch(
                action_hidden_dim=config.action_hidden_dim,
                camera_ids=config.camera_ids,
                output_projection=config.dino_output_projection,
                gamma_max=config.gamma_dino_max,
            ),
            WanRetrievalValueBranch(
                action_hidden_dim=config.action_hidden_dim,
                geometry=geometry,
                gamma_max=config.gamma_wan_max,
            ),
        )
    return DualDinoWanReader(
        routers=routers,
        branches=branches,
        config=config,
        geometry=geometry,
    )
