from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import torch

from fastwam.adapters import PolicyRegime


def _mode_value(mode: object) -> PolicyRegime:
    """Return the canonical policy-regime enum."""

    value = mode.value if isinstance(mode, Enum) else mode
    return PolicyRegime.parse(str(value))


class KVSource(str, Enum):
    """Source identities understood by the read-only Gate."""

    CURRENT_FRAME_VIDEO = "current_frame_video"
    ACTION = "action"
    TEXT_STATE_CONTEXT = "text_state_context"
    CURRENT_VIDEO_AND_ACTION = "current_frame_video_and_action"


@dataclass(frozen=True)
class KeyValueBank:
    """A projected attention K/V bank plus a per-token validity mask."""

    source: KVSource
    key: torch.Tensor
    value: torch.Tensor
    valid_mask: torch.Tensor
    contains_generated_future_video: bool = False

    def __post_init__(self) -> None:
        if self.key.ndim != 3 or self.value.ndim != 3:
            raise ValueError(
                "`key` and `value` must be [B, S, D], got "
                f"{tuple(self.key.shape)} and {tuple(self.value.shape)}."
            )
        if self.key.shape != self.value.shape:
            raise ValueError(
                f"`key` and `value` shapes must match, got {tuple(self.key.shape)} "
                f"and {tuple(self.value.shape)}."
            )
        if self.key.dtype != self.value.dtype:
            raise TypeError(
                f"`key` and `value` dtypes must match, got {self.key.dtype} "
                f"and {self.value.dtype}."
            )
        expected_mask_shape = self.key.shape[:2]
        if self.valid_mask.shape != expected_mask_shape:
            raise ValueError(
                "`valid_mask` must be [B, S] matching K/V, got "
                f"{tuple(self.valid_mask.shape)} vs {tuple(expected_mask_shape)}."
            )
        if self.valid_mask.dtype != torch.bool:
            raise TypeError(f"`valid_mask` must be bool, got {self.valid_mask.dtype}.")
        if self.valid_mask.device != self.key.device or self.value.device != self.key.device:
            raise ValueError("K/V and `valid_mask` must be on the same device.")
        if self.source is KVSource.CURRENT_FRAME_VIDEO and self.contains_generated_future_video:
            raise ValueError("The current-frame video Gate bank cannot contain generated future video.")

    @property
    def batch_size(self) -> int:
        return int(self.key.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.key.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.key.shape[2])

    @property
    def nbytes(self) -> int:
        tensors = (self.key, self.value, self.valid_mask)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def detached(self) -> KeyValueBank:
        """Return a bank that cannot carry Gate gradients into FastWAM."""

        return KeyValueBank(
            source=self.source,
            key=self.key.detach(),
            value=self.value.detach(),
            valid_mask=self.valid_mask.detach(),
            contains_generated_future_video=self.contains_generated_future_video,
        )

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> KeyValueBank:
        """Move K/V for replay while preserving the boolean source mask."""

        return KeyValueBank(
            source=self.source,
            key=self.key.to(device=device, dtype=dtype, non_blocking=non_blocking),
            value=self.value.to(device=device, dtype=dtype, non_blocking=non_blocking),
            valid_mask=self.valid_mask.to(device=device, non_blocking=non_blocking),
            contains_generated_future_video=self.contains_generated_future_video,
        )


@dataclass(frozen=True)
class GateLayerKV:
    """All read-only Gate inputs captured at one MoT layer."""

    layer_index: int
    denoise_timestep: torch.Tensor
    current_mode: tuple[PolicyRegime, ...]
    current_frame_video: KeyValueBank
    action: KeyValueBank
    context: KeyValueBank
    actor_version: int = 0

    def __post_init__(self) -> None:
        banks = (self.current_frame_video, self.action, self.context)
        if self.layer_index < 0:
            raise ValueError(f"`layer_index` must be non-negative, got {self.layer_index}.")
        if self.current_frame_video.source is not KVSource.CURRENT_FRAME_VIDEO:
            raise ValueError("`current_frame_video` has the wrong K/V source identity.")
        if self.action.source is not KVSource.ACTION:
            raise ValueError("`action` has the wrong K/V source identity.")
        if self.context.source is not KVSource.TEXT_STATE_CONTEXT:
            raise ValueError("`context` has the wrong K/V source identity.")
        if self.current_frame_video.contains_generated_future_video:
            raise ValueError("Direct generated-future video K/V is forbidden in Gate payloads.")

        batch_size = banks[0].batch_size
        feature_dim = banks[0].feature_dim
        if any(bank.batch_size != batch_size for bank in banks):
            raise ValueError("Every Gate K/V bank must have the same batch size.")
        if any(bank.feature_dim != feature_dim for bank in banks):
            raise ValueError("Every Gate K/V bank must have the same projected feature dimension.")
        if any(bank.key.device != banks[0].key.device for bank in banks):
            raise ValueError("Every Gate K/V bank must be on the same device.")
        if len(self.current_mode) != batch_size:
            raise ValueError(
                f"`current_mode` must have one entry per batch item, got {len(self.current_mode)} "
                f"for batch size {batch_size}."
            )
        normalized_modes = tuple(_mode_value(mode) for mode in self.current_mode)
        object.__setattr__(self, "current_mode", normalized_modes)
        if self.denoise_timestep.shape != (batch_size,):
            raise ValueError(
                "`denoise_timestep` must be [B], got "
                f"{tuple(self.denoise_timestep.shape)} for batch size {batch_size}."
            )
        if self.denoise_timestep.device != banks[0].key.device:
            raise ValueError("`denoise_timestep` and K/V banks must be on the same device.")
        if self.actor_version < 0:
            raise ValueError(f"`actor_version` must be non-negative, got {self.actor_version}.")

    @property
    def batch_size(self) -> int:
        return self.action.batch_size

    @property
    def feature_dim(self) -> int:
        return self.action.feature_dim

    @property
    def nbytes(self) -> int:
        return (
            self.current_frame_video.nbytes
            + self.action.nbytes
            + self.context.nbytes
            + self.denoise_timestep.numel() * self.denoise_timestep.element_size()
        )

    def detached(self) -> GateLayerKV:
        """Detach every source tensor while preserving serialized metadata."""

        return GateLayerKV(
            layer_index=self.layer_index,
            denoise_timestep=self.denoise_timestep.detach(),
            current_mode=self.current_mode,
            current_frame_video=self.current_frame_video.detached(),
            action=self.action.detached(),
            context=self.context.detached(),
            actor_version=self.actor_version,
        )

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> GateLayerKV:
        """Move one layer payload to its replay or compute device."""

        timestep_dtype = dtype if dtype is not None else self.denoise_timestep.dtype
        return GateLayerKV(
            layer_index=self.layer_index,
            denoise_timestep=self.denoise_timestep.to(
                device=device,
                dtype=timestep_dtype,
                non_blocking=non_blocking,
            ),
            current_mode=self.current_mode,
            current_frame_video=self.current_frame_video.to(
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            action=self.action.to(
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            context=self.context.to(
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            actor_version=self.actor_version,
        )


@dataclass(frozen=True)
class GateKVSnapshot:
    """Layer-wise Gate K/V captured for one action denoising forward."""

    layers: tuple[GateLayerKV, ...]

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("A Gate K/V snapshot must contain at least one layer.")
        layer_indices = tuple(layer.layer_index for layer in self.layers)
        if len(set(layer_indices)) != len(layer_indices):
            raise ValueError(f"Gate K/V layer indices must be unique, got {layer_indices}.")
        if layer_indices != tuple(sorted(layer_indices)):
            raise ValueError(f"Gate K/V layers must be ordered, got {layer_indices}.")

        first = self.layers[0]
        for layer in self.layers[1:]:
            if layer.batch_size != first.batch_size:
                raise ValueError("Every layer in a Gate snapshot must have the same batch size.")
            if layer.feature_dim != first.feature_dim:
                raise ValueError("Every layer in a Gate snapshot must have the same feature dimension.")
            if layer.current_mode != first.current_mode:
                raise ValueError("Every layer in a Gate snapshot must have the same current mode.")
            if layer.actor_version != first.actor_version:
                raise ValueError("Every layer in a Gate snapshot must have the same actor version.")
            if not torch.equal(layer.denoise_timestep, first.denoise_timestep):
                raise ValueError("Every layer in a Gate snapshot must have the same denoising timestep.")

    @property
    def batch_size(self) -> int:
        return self.layers[0].batch_size

    @property
    def layer_indices(self) -> tuple[int, ...]:
        return tuple(layer.layer_index for layer in self.layers)

    @property
    def nbytes(self) -> int:
        return sum(layer.nbytes for layer in self.layers)

    def layer(self, layer_index: int) -> GateLayerKV:
        for payload in self.layers:
            if payload.layer_index == layer_index:
                return payload
        raise KeyError(f"Layer {layer_index} is absent from snapshot {self.layer_indices}.")

    def detached(self) -> GateKVSnapshot:
        return GateKVSnapshot(tuple(layer.detached() for layer in self.layers))

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> GateKVSnapshot:
        """Move a snapshot without changing its layer/mode metadata."""

        return GateKVSnapshot(
            tuple(
                layer.to(device=device, dtype=dtype, non_blocking=non_blocking)
                for layer in self.layers
            )
        )


@dataclass
class GateKVCollector:
    """Mutable collector populated as a MoT forward traverses its layers."""

    _layers: list[GateLayerKV] = field(default_factory=list, init=False, repr=False)

    def append(self, layer: GateLayerKV) -> None:
        if self._layers and layer.layer_index <= self._layers[-1].layer_index:
            raise ValueError("Gate K/V layers must be appended once in increasing order.")
        self._layers.append(layer.detached())

    def snapshot(self) -> GateKVSnapshot:
        return GateKVSnapshot(tuple(self._layers))

    def clear(self) -> None:
        self._layers.clear()


@dataclass
class GateKVTapRequest:
    """Explicit contract for a read-only Gate tap on one MoT forward."""

    current_mode: object | Sequence[object]
    denoise_timestep: float | torch.Tensor
    current_frame_video_tokens: int
    layer_indices: tuple[int, ...] | None = None
    actor_version: int = 0
    collector: GateKVCollector = field(default_factory=GateKVCollector)

    def __post_init__(self) -> None:
        if self.current_frame_video_tokens < 1:
            raise ValueError(
                "`current_frame_video_tokens` must be positive, got "
                f"{self.current_frame_video_tokens}."
            )
        if self.actor_version < 0:
            raise ValueError(f"`actor_version` must be non-negative, got {self.actor_version}.")
        if self.layer_indices is not None:
            if not self.layer_indices:
                raise ValueError("`layer_indices` cannot be empty.")
            if len(set(self.layer_indices)) != len(self.layer_indices):
                raise ValueError(f"`layer_indices` contains duplicates: {self.layer_indices}.")
            if tuple(sorted(self.layer_indices)) != self.layer_indices:
                raise ValueError("`layer_indices` must be strictly increasing.")

    def selected_layers(self, num_layers: int) -> tuple[int, ...]:
        indices = self.layer_indices if self.layer_indices is not None else tuple(range(num_layers))
        invalid = [index for index in indices if index < 0 or index >= num_layers]
        if invalid:
            raise ValueError(
                f"Gate tap layer indices {invalid} are outside a {num_layers}-layer MoT."
            )
        return indices

    def should_capture(self, layer_index: int, num_layers: int) -> bool:
        return layer_index in self.selected_layers(num_layers)

    def normalized_modes(self, batch_size: int) -> tuple[PolicyRegime, ...]:
        if isinstance(self.current_mode, Sequence) and not isinstance(self.current_mode, (str, bytes)):
            modes = tuple(_mode_value(mode) for mode in self.current_mode)
            if len(modes) != batch_size:
                raise ValueError(
                    f"`current_mode` has {len(modes)} entries for batch size {batch_size}."
                )
            return modes
        return (_mode_value(self.current_mode),) * batch_size

    def normalized_timestep(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        timestep = torch.as_tensor(self.denoise_timestep, device=device, dtype=dtype)
        if timestep.ndim == 0:
            return timestep.expand(batch_size)
        if timestep.shape != (batch_size,):
            raise ValueError(
                f"`denoise_timestep` must be scalar or [B], got {tuple(timestep.shape)}."
            )
        return timestep

    def validate(self, *, num_layers: int, video_seq_len: int, batch_size: int) -> None:
        self.selected_layers(num_layers)
        self.normalized_modes(batch_size)
        if self.current_frame_video_tokens > video_seq_len:
            raise ValueError(
                "`current_frame_video_tokens` cannot exceed the video K/V length: "
                f"{self.current_frame_video_tokens} > {video_seq_len}."
            )

    def snapshot(self) -> GateKVSnapshot:
        return self.collector.snapshot()


def context_token_mask(
    context_mask: torch.Tensor | None,
    *,
    context: torch.Tensor,
) -> torch.Tensor:
    """Reduce an action-query context mask to a Gate source-token mask."""

    batch_size, context_len = context.shape[:2]
    if context_mask is None:
        return torch.ones((batch_size, context_len), dtype=torch.bool, device=context.device)
    if context_mask.dtype != torch.bool:
        context_mask = context_mask.to(dtype=torch.bool)
    if context_mask.shape[0] != batch_size or context_mask.shape[-1] != context_len:
        raise ValueError(
            "Context mask must match context batch/length, got "
            f"{tuple(context_mask.shape)} for context {tuple(context.shape)}."
        )
    if context_mask.ndim == 2:
        return context_mask
    if context_mask.ndim in (3, 4):
        reduce_dims = tuple(range(1, context_mask.ndim - 1))
        return context_mask.any(dim=reduce_dims)
    raise ValueError(
        f"Context mask must be [B,L], [B,S,L], or [B,H,S,L], got {tuple(context_mask.shape)}."
    )
