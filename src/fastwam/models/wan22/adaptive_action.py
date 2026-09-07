"""Differentiable cached-action velocity calls for the adaptive RL policy."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from fastwam.adapters import PolicyRegime, RegimeContext, RegimeLoRALinear

from .adaptive_sampler import VelocityOutput
from .batch_linear import BatchLinearContext
from .kv_tap import GateKVSnapshot, GateKVTapRequest


@dataclass(frozen=True)
class CachedActionCondition:
    """Conditioning state shared by every action denoising step in a chunk."""

    context: torch.Tensor
    context_mask: torch.Tensor
    video_kv_cache: list[dict[str, Any]]
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

    def index_select(self, batch_indices: torch.Tensor) -> CachedActionCondition:
        """Select batch rows while preserving the shared attention geometry."""

        if not isinstance(batch_indices, torch.Tensor):
            raise TypeError("`batch_indices` must be a tensor.")
        if batch_indices.ndim != 1:
            raise ValueError("`batch_indices` must be one-dimensional.")
        if batch_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("`batch_indices` must use an integer dtype.")
        if self.context.shape[0] != self.context_mask.shape[0]:
            raise ValueError("Cached context and context mask batch sizes differ.")

        def _select(value: torch.Tensor) -> torch.Tensor:
            indices = batch_indices.to(device=value.device, dtype=torch.long)
            return value.index_select(0, indices)

        selected_cache: list[dict[str, Any]] = []
        for layer_index, layer in enumerate(self.video_kv_cache):
            selected_layer = dict(layer)
            for bank_name in ("k", "v"):
                bank = layer.get(bank_name)
                if not isinstance(bank, torch.Tensor) or bank.ndim < 1:
                    raise ValueError(
                        "Cached video K/V must be batched tensors at "
                        f"layer={layer_index}, bank={bank_name}."
                    )
                if bank.shape[0] != self.context.shape[0]:
                    raise ValueError(
                        "Cached video K/V batch size differs from context at "
                        f"layer={layer_index}, bank={bank_name}."
                    )
                selected_layer[bank_name] = _select(bank)
            selected_cache.append(selected_layer)

        return CachedActionCondition(
            context=_select(self.context),
            context_mask=_select(self.context_mask),
            video_kv_cache=selected_cache,
            attention_mask=self.attention_mask,
            video_seq_len=self.video_seq_len,
            current_frame_video_tokens=self.current_frame_video_tokens,
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
        batch_linear_context: BatchLinearContext | None = None,
        gate_layer_indices: tuple[int, ...] | None = None,
        capture_gate_kv: bool = False,
        actor_version: int = 0,
    ) -> None:
        self.action_expert = action_expert
        self.mot = mot
        self.condition = condition
        self.regime = PolicyRegime.parse(regime)
        self.regime_context = regime_context
        self.batch_linear_context = batch_linear_context
        self.gate_layer_indices = gate_layer_indices
        self.capture_gate_kv = bool(capture_gate_kv)
        self.actor_version = int(actor_version)
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
        if self.batch_linear_context is not None and not isinstance(
            self.batch_linear_context, BatchLinearContext
        ):
            raise TypeError(
                "`batch_linear_context` must be a BatchLinearContext instance."
            )
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

    def _batch_linear_scope(self) -> AbstractContextManager[None]:
        if self.batch_linear_context is None:
            return nullcontext()
        return self.batch_linear_context.use(self.condition.context.shape[0])

    @contextmanager
    def _execution_scope(self) -> Iterator[None]:
        with self._regime_scope(), self._batch_linear_scope():
            yield

    def _checkpoint_regime_contexts(
        self,
    ) -> tuple[
        AbstractContextManager[PolicyRegime | None],
        AbstractContextManager[PolicyRegime | None],
    ]:
        """Bind the same route to checkpoint forward and backward recomputation."""

        return self._execution_scope(), self._execution_scope()

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

        with self._execution_scope():
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
                checkpoint_context_fn=self._checkpoint_regime_contexts,
                action_expert=self.action_expert,
            )
            velocity = self.action_expert.post_dit(action_tokens, action_pre)

        snapshot: GateKVSnapshot | None = (
            tap_request.snapshot() if tap_request is not None else None
        )
        return VelocityOutput(velocity=velocity, gate_tap=snapshot)


class StaticCachedActionVelocity(CachedActionVelocity):
    """Cached velocity for one frozen plain route-specific ActionDiT."""

    def __init__(
        self,
        *,
        action_expert: nn.Module,
        mot: nn.Module,
        condition: CachedActionCondition,
        regime: PolicyRegime | str,
        gate_layer_indices: tuple[int, ...] | None = None,
        capture_gate_kv: bool = False,
        actor_version: int = 0,
    ) -> None:
        self.action_expert = action_expert
        self.mot = mot
        self.condition = condition
        self.regime = PolicyRegime.parse(regime)
        self.regime_context = None
        self.batch_linear_context = None
        self.gate_layer_indices = gate_layer_indices
        self.capture_gate_kv = bool(capture_gate_kv)
        self.actor_version = int(actor_version)
        if self.actor_version < 0:
            raise ValueError("`actor_version` must be non-negative.")
        adapted_names = tuple(
            name
            for name, module in self.action_expert.named_modules()
            if isinstance(module, RegimeLoRALinear)
        )
        if adapted_names:
            raise ValueError(
                "Static action velocity requires a plain ActionDiT without LoRA: "
                f"{list(adapted_names)}."
            )
        trainable_names = tuple(
            name
            for name, parameter in self.action_expert.named_parameters()
            if parameter.requires_grad
        )
        if trainable_names:
            raise ValueError(
                "Static action velocity requires a frozen action expert: "
                f"{list(trainable_names)}."
            )
        if self.action_expert.training:
            raise ValueError(
                "Static action velocity requires an eval-mode action expert."
            )
