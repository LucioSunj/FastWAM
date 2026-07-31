"""Differentiable cached-action velocity calls for the adaptive RL policy."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import ContextManager

import torch
import torch.nn as nn

from fastwam.adapters import PolicyRegime, RegimeContext

from .adaptive_sampler import VelocityOutput
from .kv_tap import GateKVSnapshot, GateKVTapRequest


@dataclass(frozen=True)
class CachedActionCondition:
    """Conditioning state shared by every action denoising step in a chunk."""

    context: torch.Tensor
    context_mask: torch.Tensor
    video_kv_cache: list[dict[str, torch.Tensor]]
    attention_mask: torch.Tensor
    video_seq_len: int
    current_frame_video_tokens: int

    def __post_init__(self) -> None:
        if self.video_seq_len < 1:
            raise ValueError("`video_seq_len` must be positive.")
        if not 1 <= self.current_frame_video_tokens <= self.video_seq_len:
            raise ValueError(
                "`current_frame_video_tokens` must lie in "
                f"[1, {self.video_seq_len}], got {self.current_frame_video_tokens}."
            )
        if self.attention_mask.ndim != 2:
            raise ValueError("`attention_mask` must be a two-dimensional joint mask.")
        if self.context_mask.dtype != torch.bool:
            raise TypeError("`context_mask` must use bool dtype.")


class CachedActionVelocity:
    """Bind FastWAM conditioning while leaving the action state differentiable."""

    def __init__(
        self,
        *,
        action_expert: nn.Module,
        mot: nn.Module,
        condition: CachedActionCondition,
        regime: PolicyRegime | str,
        regime_context: RegimeContext | None = None,
        gate_layer_indices: tuple[int, ...] | None = None,
        capture_gate_kv: bool = False,
        actor_version: int = 0,
    ) -> None:
        self.action_expert = action_expert
        self.mot = mot
        self.condition = condition
        self.regime = PolicyRegime.parse(regime)
        self.regime_context = regime_context
        self.gate_layer_indices = gate_layer_indices
        self.capture_gate_kv = bool(capture_gate_kv)
        self.actor_version = int(actor_version)
        if self.actor_version < 0:
            raise ValueError("`actor_version` must be non-negative.")

    def _regime_scope(self) -> ContextManager[PolicyRegime | None]:
        if self.regime_context is None:
            return nullcontext()
        return self.regime_context.use(self.regime)

    def __call__(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
    ) -> VelocityOutput:
        """Predict one action velocity and optionally export detached Gate K/V."""

        if latents_action.ndim != 3:
            raise ValueError(
                "`latents_action` must be [B, horizon, action_dim], got "
                f"{tuple(latents_action.shape)}."
            )
        batch_size = latents_action.shape[0]
        if timestep_action.shape != (batch_size,):
            raise ValueError(
                f"`timestep_action` must be [{batch_size}], got "
                f"{tuple(timestep_action.shape)}."
            )

        tap_request = None
        if self.capture_gate_kv:
            tap_request = GateKVTapRequest(
                current_mode=self.regime,
                denoise_timestep=timestep_action,
                current_frame_video_tokens=self.condition.current_frame_video_tokens,
                layer_indices=self.gate_layer_indices,
                actor_version=self.actor_version,
            )

        with self._regime_scope():
            action_pre = self.action_expert.pre_dit(
                action_tokens=latents_action,
                timestep=timestep_action,
                context=self.condition.context,
                context_mask=self.condition.context_mask,
            )
            action_tokens = self.mot.forward_action_with_video_cache(
                action_tokens=action_pre["tokens"],
                action_freqs=action_pre["freqs"],
                action_t_mod=action_pre["t_mod"],
                action_context_payload={
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
                video_kv_cache=self.condition.video_kv_cache,
                attention_mask=self.condition.attention_mask,
                video_seq_len=self.condition.video_seq_len,
                kv_tap=tap_request,
            )
            velocity = self.action_expert.post_dit(action_tokens, action_pre)

        snapshot: GateKVSnapshot | None = (
            tap_request.snapshot() if tap_request is not None else None
        )
        return VelocityOutput(velocity=velocity, gate_tap=snapshot)
