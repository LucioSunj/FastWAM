"""Frozen-backbone one-pass future conditioning for conditional C1 experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class C1OnePassConfig:
    """Pre-registered one-pass and interval-fusion contract."""

    video_timestep: float = 1.0
    layer_stride: int = 4
    selected_layers: tuple[int, ...] = (0, 4, 8, 12, 16, 20, 24, 28)
    expected_layers: int = 30

    def __post_init__(self) -> None:
        if self.video_timestep != 1.0:
            raise ValueError("C1 video timestep is frozen at one.")
        if self.layer_stride != 4 or self.expected_layers != 30:
            raise ValueError("C1 is frozen to stride four over thirty layers.")
        if self.selected_layers != tuple(range(0, 30, 4)):
            raise ValueError("C1 must fuse the eight pre-registered interval layers.")


class C1IntervalKVFusion(nn.Module):
    """Fuse future K/V with the recipient current-prefix summary at eight layers.

    Only future-token rows are modified. The current prefix is returned byte-for-byte
    from the one-pass cache, which makes current-prefix information-flow audits
    independent of the learned fusion weights.
    """

    def __init__(self, config: C1OnePassConfig | None = None) -> None:
        super().__init__()
        self.config = config or C1OnePassConfig()
        self.future_mix_logits = nn.Parameter(
            torch.zeros(len(self.config.selected_layers), 2)
        )

    def metadata(self) -> dict[str, Any]:
        """Return the checkpointed fusion contract without tensor payloads."""

        return {
            "kind": "c1_interval_kv_fusion",
            **asdict(self.config),
            "trainable_names": ["future_mix_logits"],
        }

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Return exactly the independently trainable fusion tensors."""

        return {
            "future_mix_logits": self.future_mix_logits.detach().cpu().clone(),
        }

    def load_trainable_state_dict(
        self,
        state: Mapping[str, torch.Tensor],
        *,
        strict: bool = True,
    ) -> None:
        """Restore only the allowed interval-fusion tensor."""

        if strict and set(state) != {"future_mix_logits"}:
            raise ValueError("C1 fusion checkpoint tensor names changed.")
        value = state.get("future_mix_logits")
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != self.future_mix_logits.shape
        ):
            raise ValueError("C1 fusion checkpoint shape changed.")
        with torch.no_grad():
            self.future_mix_logits.copy_(
                value.to(
                    device=self.future_mix_logits.device,
                    dtype=self.future_mix_logits.dtype,
                )
            )

    def forward(
        self,
        video_kv_cache: Sequence[Mapping[str, Any]],
        *,
        current_token_count: int,
    ) -> list[dict[str, Any]]:
        """Apply interval fusion to future rows while retaining cache metadata."""

        if len(video_kv_cache) != self.config.expected_layers:
            raise ValueError(
                f"C1 expected {self.config.expected_layers} cache layers, got "
                f"{len(video_kv_cache)}."
            )
        if current_token_count < 1:
            raise ValueError("C1 requires a non-empty current-frame prefix.")
        selected_to_index = {
            layer: index for index, layer in enumerate(self.config.selected_layers)
        }
        fused: list[dict[str, Any]] = []
        for layer_index, layer_cache in enumerate(video_kv_cache):
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError("C1 cache layers must contain K and V tensors.")
            key = layer_cache["k"]
            value = layer_cache["v"]
            if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                raise TypeError("C1 cache K/V entries must be tensors.")
            if key.shape != value.shape or key.ndim < 3:
                raise ValueError("C1 cache K/V shapes must match and include sequence.")
            if key.shape[1] < current_token_count:
                raise ValueError("C1 current prefix exceeds the cached sequence.")
            updated = dict(layer_cache)
            if layer_index in selected_to_index and key.shape[1] > current_token_count:
                weights = torch.softmax(
                    self.future_mix_logits[selected_to_index[layer_index]], dim=0
                ).to(device=key.device, dtype=key.dtype)
                current_k = key[:, :current_token_count].mean(dim=1, keepdim=True)
                current_v = value[:, :current_token_count].mean(dim=1, keepdim=True)
                future_k = key[:, current_token_count:]
                future_v = value[:, current_token_count:]
                updated["k"] = torch.cat(
                    (
                        key[:, :current_token_count],
                        weights[0] * future_k + weights[1] * current_k,
                    ),
                    dim=1,
                )
                updated["v"] = torch.cat(
                    (
                        value[:, :current_token_count],
                        weights[0] * future_v + weights[1] * current_v,
                    ),
                    dim=1,
                )
            fused.append(updated)
        return fused


def deterministic_tri_mode_sequence(
    batch_size: int,
    *,
    optimizer_step: int,
    accumulation_index: int,
    rank: int,
    world_size: int,
    accumulation_steps: int = 4,
) -> tuple[str, ...]:
    """Assign the exact 43/43/42 rotating global tri-mode quotas.

    Four consecutive optimizer ranks/microbatches form the configured global batch
    of 128. Across every three optimizer steps each mode receives exactly 128
    samples, without consuming RNG state.
    """

    if min(batch_size, world_size, accumulation_steps) < 1:
        raise ValueError("Tri-mode sequence dimensions must be positive.")
    if min(optimizer_step, accumulation_index, rank) < 0:
        raise ValueError("Tri-mode sequence indices must be non-negative.")
    if rank >= world_size or accumulation_index >= accumulation_steps:
        raise ValueError("Tri-mode rank or accumulation index is out of range.")
    global_batch = batch_size * world_size * accumulation_steps
    if global_batch != 128:
        raise ValueError("The formal tri-mode global batch is frozen at 128.")
    quotas = (
        (43, 43, 42),
        (42, 43, 43),
        (43, 42, 43),
    )[optimizer_step % 3]
    modes = ("c0_current", "c1_one_pass", "c2_full")
    global_assignment = []
    remaining = list(quotas)
    while any(remaining):
        for mode_index, mode in enumerate(modes):
            if remaining[mode_index]:
                global_assignment.append(mode)
                remaining[mode_index] -= 1
    global_offset = (
        rank * accumulation_steps * batch_size + accumulation_index * batch_size
    )
    result = []
    for local_index in range(batch_size):
        position = global_offset + local_index
        result.append(global_assignment[position])
    return tuple(result)
