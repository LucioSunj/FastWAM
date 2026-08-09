"""DINOv3-guided, action-facing shadow K/V for frozen Wan current tokens.

This module implements the P8-A0/KV contract.  It never mutates the Wan token
trajectory or the base video cache: selected current-frame sources are detached
from the frozen video expert, and a trainable refiner creates a separate K/V
view consumed only by the UNCOND action expert.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch import nn

from .visual_contracts import (
    DINO_V3_NATIVE_DIM,
    NativePatchMemory,
    contract_sha256,
    validate_sha256,
)
from .wan_video_dit import rope_apply

WAN_CURRENT_REFINER_STATE_SCHEMA = "fastwam-wan-current-refiner-v1"
WAN_CURRENT_SOURCE_SCHEMA = "fastwam-wan-current-source-v1"
WAN_CURRENT_CACHE_VISIBILITY = {
    "base": ("wan_trajectory", "idm_action", "gate_direct_video"),
    "action": ("uncond_action_mixed_attention",),
}


@dataclass(frozen=True)
class WanCurrentLayerSource:
    """Detached frozen Wan tensors needed to rebuild one current K/V layer."""

    layer_index: int
    hidden_current: torch.Tensor
    attention_input_current: torch.Tensor
    key_pre_norm_current: torch.Tensor
    base_key_current: torch.Tensor
    base_value_current: torch.Tensor
    rope_freqs_current: torch.Tensor
    camera_index_current: torch.Tensor
    current_frame_video_tokens: int
    source_contract_sha256: str

    def __post_init__(self) -> None:
        layer_index = int(self.layer_index)
        current = int(self.current_frame_video_tokens)
        tensors = (
            self.hidden_current,
            self.attention_input_current,
            self.key_pre_norm_current,
            self.base_key_current,
            self.base_value_current,
        )
        if layer_index < 0:
            raise ValueError("Wan-current layer index must be non-negative.")
        if any(tensor.ndim != 3 for tensor in tensors):
            raise ValueError("Wan-current layer tensors must have shape [B,N,D].")
        if any(tensor.shape[:2] != tensors[0].shape[:2] for tensor in tensors):
            raise ValueError("Wan-current layer tensors must align on [B,N].")
        if current != self.hidden_current.shape[1] or current < 1:
            raise ValueError(
                "Wan-current sources must contain exactly the current prefix."
            )
        if self.hidden_current.shape != self.attention_input_current.shape:
            raise ValueError("Wan hidden and attention-input shapes must match.")
        if self.key_pre_norm_current.shape != self.base_key_current.shape:
            raise ValueError("Pre-norm and base key shapes must match.")
        if self.base_key_current.shape != self.base_value_current.shape:
            raise ValueError("Base key and value shapes must match.")
        if self.rope_freqs_current.shape[0] != current:
            raise ValueError(
                "RoPE frequencies must contain exactly the current prefix."
            )
        if self.camera_index_current.shape != self.hidden_current.shape[:2]:
            raise ValueError("Camera index map must have shape [B,N_current].")
        if self.camera_index_current.dtype != torch.long:
            raise TypeError("Camera index map must use torch.long dtype.")
        devices = {tensor.device for tensor in tensors}
        devices.add(self.rope_freqs_current.device)
        devices.add(self.camera_index_current.device)
        if len(devices) != 1:
            raise ValueError("Wan-current source tensors must share one device.")
        if any(tensor.requires_grad for tensor in tensors):
            raise ValueError("Frozen Wan-current sources must be detached.")
        if self.rope_freqs_current.requires_grad:
            raise ValueError("Frozen Wan RoPE frequencies must be detached.")
        if any(not tensor.is_floating_point() for tensor in tensors):
            raise TypeError("Wan-current source tensors must use floating dtypes.")
        object.__setattr__(self, "layer_index", layer_index)
        object.__setattr__(self, "current_frame_video_tokens", current)
        object.__setattr__(
            self,
            "source_contract_sha256",
            validate_sha256(
                self.source_contract_sha256,
                label="Wan-current source contract SHA256",
            ),
        )


class WanCurrentSourceCollector:
    """Single-use collector for selected frozen Wan layer sources."""

    def __init__(self) -> None:
        self._sources: dict[int, WanCurrentLayerSource] = {}

    def append(self, source: WanCurrentLayerSource) -> None:
        if source.layer_index in self._sources:
            raise ValueError(
                f"Wan-current layer {source.layer_index} was captured twice."
            )
        self._sources[source.layer_index] = source

    def snapshot(
        self, expected_layers: tuple[int, ...]
    ) -> tuple[WanCurrentLayerSource, ...]:
        actual = tuple(sorted(self._sources))
        if actual != expected_layers:
            raise ValueError(
                f"Wan-current source capture is incomplete: {actual} != {expected_layers}."
            )
        return tuple(self._sources[index] for index in expected_layers)


@dataclass(frozen=True)
class WanCurrentSourceCaptureRequest:
    """Opt-in request for current-only source capture during frozen prefill."""

    layer_indices: tuple[int, ...]
    current_frame_video_tokens: int
    camera_index_current: torch.Tensor
    source_contract_sha256: str
    collector: WanCurrentSourceCollector

    def __post_init__(self) -> None:
        layers = tuple(int(index) for index in self.layer_indices)
        if not layers or len(layers) > 2 or tuple(sorted(set(layers))) != layers:
            raise ValueError("P8-A0 requires one or two unique sorted layer indices.")
        current = int(self.current_frame_video_tokens)
        if current < 1:
            raise ValueError("Current-frame video token count must be positive.")
        if self.camera_index_current.ndim != 2:
            raise ValueError("Camera index map must have shape [B,N_current].")
        if self.camera_index_current.shape[1] != current:
            raise ValueError("Camera index map must cover exactly the current tokens.")
        if self.camera_index_current.dtype != torch.long:
            raise TypeError("Camera index map must use torch.long dtype.")
        if not isinstance(self.collector, WanCurrentSourceCollector):
            raise TypeError("`collector` must be a WanCurrentSourceCollector.")
        object.__setattr__(self, "layer_indices", layers)
        object.__setattr__(self, "current_frame_video_tokens", current)
        object.__setattr__(
            self,
            "source_contract_sha256",
            validate_sha256(
                self.source_contract_sha256,
                label="Wan-current source contract SHA256",
            ),
        )

    @classmethod
    def create(
        cls,
        *,
        layer_indices: tuple[int, ...],
        current_frame_video_tokens: int,
        camera_index_current: torch.Tensor,
        source_contract_sha256: str,
    ) -> WanCurrentSourceCaptureRequest:
        """Create a request with a fresh single-use collector."""

        return cls(
            layer_indices=layer_indices,
            current_frame_video_tokens=current_frame_video_tokens,
            camera_index_current=camera_index_current,
            source_contract_sha256=source_contract_sha256,
            collector=WanCurrentSourceCollector(),
        )

    def validate(self, *, num_layers: int, video_tokens: torch.Tensor) -> None:
        """Fail closed unless prefill is a current-only, batch-aligned pass."""

        if any(index >= num_layers for index in self.layer_indices):
            raise ValueError("Wan-current source layer lies outside the Wan expert.")
        if video_tokens.shape[1] != self.current_frame_video_tokens:
            raise ValueError(
                "P8-A0 source capture requires video_seq_len == "
                "current_frame_video_tokens; generated future video is forbidden."
            )
        if self.camera_index_current.shape != video_tokens.shape[:2]:
            raise ValueError("Camera index map does not match current Wan tokens.")
        if self.camera_index_current.device != video_tokens.device:
            raise ValueError("Camera index map and Wan tokens must share a device.")

    def should_capture(self, layer_index: int) -> bool:
        return layer_index in self.layer_indices

    def snapshot(self) -> tuple[WanCurrentLayerSource, ...]:
        return self.collector.snapshot(self.layer_indices)


@dataclass(frozen=True)
class ActionShadowKV:
    """Current-prefix K/V override for one action-facing video-cache layer."""

    layer_index: int
    key_current: torch.Tensor
    value_current: torch.Tensor
    current_frame_video_tokens: int
    actor_version: int
    refiner_contract_sha256: str
    contains_generated_future_video: bool = False

    def __post_init__(self) -> None:
        if (
            self.key_current.ndim != 3
            or self.key_current.shape != self.value_current.shape
        ):
            raise ValueError("Action shadow K/V must have equal [B,N,D] shapes.")
        current = int(self.current_frame_video_tokens)
        if self.key_current.shape[1] != current or current < 1:
            raise ValueError("Action shadow K/V must cover exactly the current prefix.")
        if self.actor_version < 0:
            raise ValueError("Action shadow actor version must be non-negative.")
        if self.contains_generated_future_video:
            raise ValueError(
                "Action shadow K/V may never contain generated future video."
            )
        object.__setattr__(
            self,
            "refiner_contract_sha256",
            validate_sha256(
                self.refiner_contract_sha256,
                label="Wan-current refiner contract SHA256",
            ),
        )


@dataclass(frozen=True)
class ActionVideoKVView:
    """Logical action cache view with sparse selected-layer shadow overrides."""

    base_video_kv_cache: list[dict[str, Any]]
    shadows: tuple[ActionShadowKV, ...]
    actor_version: int

    def __post_init__(self) -> None:
        if self.actor_version < 0:
            raise ValueError("Action cache-view actor version must be non-negative.")
        indices = tuple(shadow.layer_index for shadow in self.shadows)
        if tuple(sorted(set(indices))) != indices:
            raise ValueError("Action shadow layers must be unique and sorted.")
        if any(shadow.actor_version != self.actor_version for shadow in self.shadows):
            raise ValueError("Action shadow and cache-view actor versions differ.")
        for shadow in self.shadows:
            if shadow.layer_index >= len(self.base_video_kv_cache):
                raise ValueError("Action shadow layer lies outside the base cache.")
            base = self.base_video_kv_cache[shadow.layer_index]
            if "k" not in base or "v" not in base:
                raise ValueError("Base cache entries must contain `k` and `v`.")
            if base["k"].shape != base["v"].shape:
                raise ValueError("Base video K/V shapes must match.")
            if shadow.key_current.shape[0] != base["k"].shape[0]:
                raise ValueError("Action shadow and base cache batch sizes differ.")
            if shadow.key_current.shape[2] != base["k"].shape[2]:
                raise ValueError("Action shadow and base cache widths differ.")
            if shadow.current_frame_video_tokens > base["k"].shape[1]:
                raise ValueError("Action shadow extends beyond the base video cache.")

    @classmethod
    def base_alias(
        cls,
        base_video_kv_cache: list[dict[str, Any]],
        *,
        actor_version: int,
    ) -> ActionVideoKVView:
        """Create the disabled/force-base view without constructing shadows."""

        return cls(
            base_video_kv_cache=base_video_kv_cache,
            shadows=(),
            actor_version=actor_version,
        )

    def layer(self, layer_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve one layer, preserving exact base-object aliases when unselected."""

        base = self.base_video_kv_cache[layer_index]
        shadow = next(
            (item for item in self.shadows if item.layer_index == layer_index),
            None,
        )
        if shadow is None:
            return base["k"], base["v"]
        current = shadow.current_frame_video_tokens
        if current == base["k"].shape[1]:
            return shadow.key_current, shadow.value_current
        return (
            torch.cat((shadow.key_current, base["k"][:, current:]), dim=1),
            torch.cat((shadow.value_current, base["v"][:, current:]), dim=1),
        )


@dataclass(frozen=True)
class WanCurrentRefinerConfig:
    """Resolved P8-A0/KV refiner architecture and provenance contract."""

    wan_hidden_dim: int
    native_dim: int
    layer_indices: tuple[int, ...]
    query_rank: int
    output_rank: int
    temperature: float
    alpha: float
    memory_contract_sha256: str
    source_contract_sha256: str

    def __post_init__(self) -> None:
        layers = tuple(int(index) for index in self.layer_indices)
        if not layers or len(layers) > 2 or tuple(sorted(set(layers))) != layers:
            raise ValueError("P8-A0 requires one or two unique sorted layers.")
        for name in ("wan_hidden_dim", "native_dim", "query_rank", "output_rank"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"`{name}` must be positive.")
        if self.native_dim != DINO_V3_NATIVE_DIM:
            raise ValueError("P8-A0 requires native DINOv3 ViT-S/16 width 384.")
        if self.temperature <= 0 or self.alpha <= 0:
            raise ValueError("P8 temperature and alpha must be positive.")
        object.__setattr__(self, "layer_indices", layers)
        for name, label in (
            ("memory_contract_sha256", "Native memory contract SHA256"),
            ("source_contract_sha256", "Wan-current source contract SHA256"),
        ):
            object.__setattr__(
                self,
                name,
                validate_sha256(getattr(self, name), label=label),
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> WanCurrentRefinerConfig:
        """Parse a resolved config without accepting missing or unknown fields."""

        required = {
            "wan_hidden_dim",
            "native_dim",
            "layer_indices",
            "query_rank",
            "output_rank",
            "temperature",
            "alpha",
            "memory_contract_sha256",
            "source_contract_sha256",
        }
        if set(payload) != required:
            raise ValueError(
                "Invalid Wan-current refiner config fields; "
                f"missing={sorted(required - set(payload))}, "
                f"unknown={sorted(set(payload) - required)}."
            )
        values = dict(payload)
        values["layer_indices"] = tuple(values["layer_indices"])
        return cls(**values)

    def as_contract(self) -> dict[str, Any]:
        """Return the stable JSON-compatible refiner contract."""

        return {
            "schema": WAN_CURRENT_REFINER_STATE_SCHEMA,
            "mechanism": "p8-a0-kv",
            "wan_hidden_dim": self.wan_hidden_dim,
            "native_dim": self.native_dim,
            "layer_indices": list(self.layer_indices),
            "query_rank": self.query_rank,
            "output_rank": self.output_rank,
            "temperature": self.temperature,
            "alpha": self.alpha,
            "dropout": 0.0,
            "output_up_init": "zero",
            "source_scope": "current_frame_only",
            "cache_visibility": WAN_CURRENT_CACHE_VISIBILITY,
            "memory_contract_sha256": self.memory_contract_sha256,
            "source_contract_sha256": self.source_contract_sha256,
        }

    @property
    def contract_sha256(self) -> str:
        return contract_sha256(self.as_contract())


class _LayerRefiner(nn.Module):
    def __init__(self, config: WanCurrentRefinerConfig) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(config.wan_hidden_dim)
        self.query_down = nn.Linear(
            config.wan_hidden_dim, config.query_rank, bias=False
        )
        self.query_up = nn.Linear(config.query_rank, config.native_dim, bias=False)
        self.output_down = nn.Linear(config.native_dim, config.output_rank, bias=False)
        self.output_up = nn.Linear(
            config.output_rank, config.wan_hidden_dim, bias=False
        )
        nn.init.zeros_(self.output_up.weight)

    def retrieve(
        self,
        source: WanCurrentLayerSource,
        memory: NativePatchMemory,
        *,
        temperature: float,
    ) -> torch.Tensor:
        hidden = source.hidden_current.to(dtype=self.query_norm.weight.dtype)
        query = self.query_up(self.query_down(self.query_norm(hidden))).float()
        query = F.normalize(query, p=2.0, dim=-1)
        patches = memory.tokens.detach().float()
        patches_normalized = F.normalize(patches, p=2.0, dim=-1)
        batch, views, patch_count, width = patches.shape
        flat_patches = patches.reshape(batch, views * patch_count, width)
        flat_normalized = patches_normalized.reshape(batch, views * patch_count, width)
        scores = torch.matmul(query, flat_normalized.transpose(-1, -2)) / temperature
        patch_camera = torch.arange(views, device=scores.device).repeat_interleave(
            patch_count
        )
        same_camera = source.camera_index_current.unsqueeze(-1) == patch_camera
        valid = memory.patch_valid_mask.reshape(batch, views * patch_count).unsqueeze(1)
        support = same_camera & valid
        if bool((~support.any(dim=-1)).any().item()):
            raise ValueError(
                "Every Wan-current query must have valid same-camera patches."
            )
        scores = scores.masked_fill(~support, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, flat_patches)

    def correction(
        self, retrieval: torch.Tensor, *, alpha: float, dtype: torch.dtype
    ) -> torch.Tensor:
        value = retrieval.to(dtype=self.output_down.weight.dtype)
        delta = self.output_up(F.silu(self.output_down(value))) * alpha
        return delta.to(dtype=dtype)


@dataclass(frozen=True)
class RefinerBehaviorSnapshot:
    """CPU snapshot used only for behavior-policy Gate recomputation."""

    actor_version: int
    refiner_contract_sha256: str
    state: Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class WanCurrentTrainableParameter:
    """One typed, identity-preserving refiner optimizer-manifest entry."""

    name: str
    parameter: nn.Parameter

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.parameter, nn.Parameter):
            raise TypeError("Invalid Wan-current refiner parameter manifest entry.")
        if not self.parameter.requires_grad:
            raise ValueError(
                "Wan-current refiner manifest parameters must be trainable."
            )


class WanCurrentKVRefiner(nn.Module):
    """Build differentiable selected-layer action shadows from frozen sources."""

    parameter_family = "wan_current_refiner"

    def __init__(self, config: WanCurrentRefinerConfig) -> None:
        super().__init__()
        self.config = config
        self._replay_reference: RefinerBehaviorSnapshot | None = None
        self.layers = nn.ModuleDict(
            {str(index): _LayerRefiner(config) for index in config.layer_indices}
        )

    @property
    def refiner_contract_sha256(self) -> str:
        return self.config.contract_sha256

    def trainable_parameter_manifest(
        self,
    ) -> tuple[WanCurrentTrainableParameter, ...]:
        """Export every and only trainable refiner parameter with object identity."""

        entries = tuple(
            WanCurrentTrainableParameter(name=name, parameter=parameter)
            for name, parameter in self.named_parameters()
        )
        if not entries or len({id(entry.parameter) for entry in entries}) != len(
            entries
        ):
            raise RuntimeError("Wan-current refiner parameter manifest is invalid.")
        return entries

    def _validate_inputs(
        self,
        *,
        base_video_kv_cache: list[dict[str, Any]],
        sources: Sequence[WanCurrentLayerSource],
        memory: NativePatchMemory,
        video_blocks: Sequence[nn.Module],
    ) -> None:
        if tuple(source.layer_index for source in sources) != self.config.layer_indices:
            raise ValueError("Wan-current source layers do not match refiner config.")
        if len(base_video_kv_cache) != len(video_blocks):
            raise ValueError("Base cache and frozen Wan block counts differ.")
        if memory.memory_contract_sha256 != self.config.memory_contract_sha256:
            raise ValueError("Native memory contract does not match the P8 refiner.")
        if any(
            source.source_contract_sha256 != self.config.source_contract_sha256
            for source in sources
        ):
            raise ValueError("Wan-current source contract does not match the refiner.")
        if memory.tokens.shape[0] != sources[0].hidden_current.shape[0]:
            raise ValueError("Native memory and Wan-current source batches differ.")
        views = memory.tokens.shape[1]
        for source in sources:
            if bool(
                (
                    (source.camera_index_current < 0)
                    | (source.camera_index_current >= views)
                )
                .any()
                .item()
            ):
                raise ValueError(
                    "Wan-current camera indices lie outside native memory."
                )
            block = video_blocks[source.layer_index]
            frozen_modules = (
                block.self_attn.k,
                block.self_attn.v,
                block.self_attn.norm_k,
            )
            if any(
                parameter.requires_grad
                for module in frozen_modules
                for parameter in module.parameters()
            ):
                raise ValueError("Wan video K/V projection and norm must be frozen.")

    def build_action_view(
        self,
        *,
        base_video_kv_cache: list[dict[str, Any]],
        sources: Sequence[WanCurrentLayerSource],
        memory: NativePatchMemory,
        video_blocks: Sequence[nn.Module],
        actor_version: int,
        force_base_view: bool = False,
        allow_no_grad: bool = False,
    ) -> ActionVideoKVView:
        """Create a live action view; this call must remain grad-enabled in replay."""

        if force_base_view:
            return ActionVideoKVView.base_alias(
                base_video_kv_cache,
                actor_version=actor_version,
            )
        if not torch.is_grad_enabled() and not allow_no_grad:
            raise RuntimeError(
                "Training-time Wan-current shadow construction cannot run under no_grad."
            )
        self._validate_inputs(
            base_video_kv_cache=base_video_kv_cache,
            sources=sources,
            memory=memory,
            video_blocks=video_blocks,
        )
        shadows: list[ActionShadowKV] = []
        for source in sources:
            block = video_blocks[source.layer_index]
            layer = self.layers[str(source.layer_index)]
            retrieval = layer.retrieve(
                source,
                memory,
                temperature=self.config.temperature,
            )
            delta_input = layer.correction(
                retrieval,
                alpha=self.config.alpha,
                dtype=source.attention_input_current.dtype,
            )
            delta_key_pre = F.linear(delta_input, block.self_attn.k.weight, None)
            delta_value = F.linear(delta_input, block.self_attn.v.weight, None)

            # Base-plus-correction preserves bit-exact active-zero behavior while
            # retaining the derivative through normK and RoPE into delta_input.
            recomputed_base = rope_apply(
                block.self_attn.norm_k(source.key_pre_norm_current),
                source.rope_freqs_current,
                block.num_heads,
            )
            recomputed_refined = rope_apply(
                block.self_attn.norm_k(source.key_pre_norm_current + delta_key_pre),
                source.rope_freqs_current,
                block.num_heads,
            )
            key = source.base_key_current + (recomputed_refined - recomputed_base)
            value = source.base_value_current + delta_value
            shadows.append(
                ActionShadowKV(
                    layer_index=source.layer_index,
                    key_current=key,
                    value_current=value,
                    current_frame_video_tokens=source.current_frame_video_tokens,
                    actor_version=actor_version,
                    refiner_contract_sha256=self.refiner_contract_sha256,
                )
            )
        return ActionVideoKVView(
            base_video_kv_cache=base_video_kv_cache,
            shadows=tuple(shadows),
            actor_version=actor_version,
        )

    def checkpoint_state(self) -> dict[str, Any]:
        """Return trainable-only state and its strict architecture contract."""

        return {
            "schema": WAN_CURRENT_REFINER_STATE_SCHEMA,
            "contract": self.config.as_contract(),
            "contract_sha256": self.refiner_contract_sha256,
            "state": {
                name: tensor.detach().cpu().clone()
                for name, tensor in self.state_dict().items()
            },
        }

    def load_checkpoint_state(self, payload: Mapping[str, Any]) -> None:
        """Strictly restore only a matching P8 refiner checkpoint."""

        if set(payload) != {"schema", "contract", "contract_sha256", "state"}:
            raise ValueError("Invalid Wan-current refiner checkpoint fields.")
        if payload["schema"] != WAN_CURRENT_REFINER_STATE_SCHEMA:
            raise ValueError("Unsupported Wan-current refiner checkpoint schema.")
        if payload["contract"] != self.config.as_contract():
            raise ValueError("Wan-current refiner checkpoint contract mismatch.")
        if payload["contract_sha256"] != self.refiner_contract_sha256:
            raise ValueError("Wan-current refiner checkpoint hash mismatch.")
        self.load_state_dict(dict(payload["state"]), strict=True)

    def capture_behavior_snapshot(
        self, *, actor_version: int
    ) -> RefinerBehaviorSnapshot:
        """Capture refiner state for no-grad behavior Gate recomputation."""

        if actor_version < 0:
            raise ValueError("Behavior refiner actor version must be non-negative.")
        return RefinerBehaviorSnapshot(
            actor_version=actor_version,
            refiner_contract_sha256=self.refiner_contract_sha256,
            state={
                name: tensor.detach().cpu().clone()
                for name, tensor in self.state_dict().items()
            },
        )

    def capture_replay_reference(self, *, actor_version: int) -> None:
        """Store the behavior snapshot used by Gate K/V recomputation."""

        self._replay_reference = self.capture_behavior_snapshot(
            actor_version=actor_version
        )

    @contextmanager
    def use_replay_reference(self, *, actor_version: int) -> Iterator[None]:
        """Use the captured behavior refiner or fail closed when unavailable."""

        if self._replay_reference is None:
            raise RuntimeError("No behavior Wan-current refiner snapshot was captured.")
        with self.use_behavior_snapshot(
            self._replay_reference,
            actor_version=actor_version,
        ):
            yield

    @contextmanager
    def use_behavior_snapshot(
        self,
        snapshot: RefinerBehaviorSnapshot,
        *,
        actor_version: int,
    ) -> Iterator[None]:
        """Temporarily restore behavior state, then exactly restore live state."""

        if snapshot.actor_version != actor_version:
            raise ValueError("Behavior refiner actor version mismatch.")
        if snapshot.refiner_contract_sha256 != self.refiner_contract_sha256:
            raise ValueError("Behavior refiner contract hash mismatch.")
        live = {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.state_dict().items()
        }
        try:
            self.load_state_dict(dict(snapshot.state), strict=True)
            yield
        finally:
            self.load_state_dict(live, strict=True)
