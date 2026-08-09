"""Differentiable cached-action velocity calls for the adaptive RL policy."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from fastwam.adapters import PolicyRegime, RegimeContext, RegimeLoRALinear

from .adaptive_sampler import VelocityOutput
from .kv_tap import GateKVSnapshot, GateKVTapRequest
from .visual_contracts import ActionVisualReader, NativePatchMemory
from .wan_current_refiner import ActionVideoKVView


@dataclass(frozen=True)
class VisualReadCondition:
    """Frozen per-replan memory and non-visual inputs for an action reader."""

    memory: NativePatchMemory
    proprio: torch.Tensor
    video_layout_metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.memory, NativePatchMemory):
            raise TypeError("`memory` must be a NativePatchMemory instance.")
        if self.proprio.ndim != 2:
            raise ValueError("Visual-reader proprioception must have shape [B,D_p].")
        if not self.proprio.is_floating_point():
            raise TypeError("Visual-reader proprioception must use a floating dtype.")
        if self.proprio.shape[0] != self.memory.tokens.shape[0]:
            raise ValueError("Visual memory and proprioception batch sizes differ.")
        if self.proprio.device != self.memory.tokens.device:
            raise ValueError("Visual memory and proprioception must share a device.")


@dataclass(frozen=True)
class CachedActionCondition:
    """Conditioning state shared by every action denoising step in a chunk."""

    context: torch.Tensor
    context_mask: torch.Tensor
    video_kv_cache: list[dict[str, Any]]
    attention_mask: torch.Tensor
    video_seq_len: int
    current_frame_video_tokens: int
    visual: VisualReadCondition | None = None
    action_video_kv_view: ActionVideoKVView | None = None

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
        if self.visual is not None:
            if not isinstance(self.visual, VisualReadCondition):
                raise TypeError("`visual` must be a VisualReadCondition instance.")
            if self.visual.memory.tokens.shape[0] != self.context.shape[0]:
                raise ValueError("Visual and text/state context batch sizes differ.")
        if self.action_video_kv_view is not None:
            if not isinstance(self.action_video_kv_view, ActionVideoKVView):
                raise TypeError(
                    "`action_video_kv_view` must be an ActionVideoKVView instance."
                )
            if self.action_video_kv_view.base_video_kv_cache is not self.video_kv_cache:
                raise ValueError(
                    "Action video view must reference the condition's exact base cache."
                )


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
        visual_reader: ActionVisualReader | None = None,
    ) -> None:
        self.action_expert = action_expert
        self.mot = mot
        self.condition = condition
        self.regime = PolicyRegime.parse(regime)
        self.regime_context = regime_context
        self.gate_layer_indices = gate_layer_indices
        self.capture_gate_kv = bool(capture_gate_kv)
        self.actor_version = int(actor_version)
        self.visual_reader = visual_reader
        if self.actor_version < 0:
            raise ValueError("`actor_version` must be non-negative.")
        if self.regime is PolicyRegime.UNCOND and self.regime_context is None:
            raise ValueError(
                "UNCOND action velocity requires the injected LoRA `regime_context`; "
                "silently using IDM/base weights is forbidden."
            )
        if self.regime_context is not None and not isinstance(
            self.regime_context, RegimeContext
        ):
            raise TypeError("`regime_context` must be a RegimeContext instance.")
        if (self.visual_reader is None) != (self.condition.visual is None):
            raise ValueError(
                "Visual reader and visual condition must be supplied together."
            )
        if self.regime is PolicyRegime.IDM and self.visual_reader is not None:
            raise ValueError("IDM action velocity must bypass the visual sidecar.")
        if (
            self.regime is PolicyRegime.IDM
            and self.condition.action_video_kv_view is not None
            and self.condition.action_video_kv_view.shadows
        ):
            raise ValueError("IDM action velocity must reject P8 action shadows.")
        if (
            self.condition.action_video_kv_view is not None
            and self.condition.action_video_kv_view.actor_version != self.actor_version
        ):
            raise ValueError(
                "P8 action view and action velocity actor versions differ."
            )
        if self.visual_reader is not None and not isinstance(
            self.visual_reader, ActionVisualReader
        ):
            raise TypeError("`visual_reader` must implement ActionVisualReader.")
        if self.regime is PolicyRegime.UNCOND:
            adapted_layers = tuple(
                module
                for module in self.action_expert.modules()
                if isinstance(module, RegimeLoRALinear)
            )
            if not adapted_layers:
                raise ValueError(
                    "UNCOND action velocity requires at least one injected "
                    "RegimeLoRALinear."
                )
            if any(
                layer.regime_context is not self.regime_context
                for layer in adapted_layers
            ):
                raise ValueError(
                    "`regime_context` does not own every injected ActionDiT LoRA layer."
                )

    def _regime_scope(self) -> AbstractContextManager[PolicyRegime | None]:
        if self.regime_context is None:
            return nullcontext()
        return self.regime_context.use(self.regime)

    def _checkpoint_regime_contexts(
        self,
    ) -> tuple[
        AbstractContextManager[PolicyRegime | None],
        AbstractContextManager[PolicyRegime | None],
    ]:
        """Bind the same route to checkpoint forward and backward recomputation."""

        return self._regime_scope(), self._regime_scope()

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
            visual = self.condition.visual
            if visual is not None and "t" not in action_pre:
                raise ValueError(
                    "Visual readers require the frozen ActionDiT timestep embedding."
                )
            mot_kwargs: dict[str, Any] = {
                "action_tokens": action_pre["tokens"],
                "action_freqs": action_pre["freqs"],
                "action_t_mod": action_pre["t_mod"],
                "action_context_payload": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
                "video_kv_cache": self.condition.video_kv_cache,
                "attention_mask": self.condition.attention_mask,
                "video_seq_len": self.condition.video_seq_len,
                "kv_tap": tap_request,
                "checkpoint_context_fn": self._checkpoint_regime_contexts,
            }
            if self.condition.action_video_kv_view is not None:
                mot_kwargs["action_video_kv_view"] = self.condition.action_video_kv_view
            if visual is not None:
                mot_kwargs.update(
                    visual_reader=self.visual_reader,
                    visual_memory=visual.memory,
                    visual_proprio=visual.proprio,
                    action_time_embedding=action_pre["t"],
                    current_frame_video_tokens=(
                        self.condition.current_frame_video_tokens
                    ),
                    video_layout_metadata=visual.video_layout_metadata,
                )
            action_tokens = self.mot.forward_action_with_video_cache(**mot_kwargs)
            velocity = self.action_expert.post_dit(action_tokens, action_pre)

        snapshot: GateKVSnapshot | None = (
            tap_request.snapshot() if tap_request is not None else None
        )
        return VelocityOutput(velocity=velocity, gate_tap=snapshot)
