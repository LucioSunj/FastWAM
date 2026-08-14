"""Forced rollout ablations for the modality-dropout BC pilot."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from fastwam.modality_dropout_bc import fixed_gaussian_patch_memory
from fastwam.p1_dino_libero_inference import (
    P1DinoCompiledLiberoPolicy,
    _P1DenoiseKernel,
)

ROLLOUT_MODALITY_CONDITIONS = frozenset(
    {"clean", "wan_drop", "dino_drop", "both_drop"}
)


class _ModalityDropoutDenoiseKernel(_P1DenoiseKernel):
    """Apply a static Wan ablation before the existing ActionDiT kernel."""

    def __init__(self, policy, *, memory_enabled: bool, wan_dropped: bool) -> None:
        super().__init__(policy, memory_enabled=memory_enabled)
        self.wan_dropped = bool(wan_dropped)

    def forward(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_cache_k: list[torch.Tensor],
        video_cache_v: list[torch.Tensor],
        action_attention_mask: torch.Tensor,
        visual_memory,
        visual_proprio: torch.Tensor | None,
        current_frame_video_tokens: int,
    ) -> torch.Tensor:
        """Zero and key-mask Wan K/V for the complete rollout action chunk."""

        if self.wan_dropped:
            batch_size = int(latents_action.shape[0])
            if any(
                int(key.shape[1]) != int(current_frame_video_tokens)
                for key in video_cache_k
            ):
                raise ValueError(
                    "Wan rollout ablation requires current-frame-only K/V."
                )
            video_cache_k = [torch.zeros_like(key) for key in video_cache_k]
            video_cache_v = [torch.zeros_like(value) for value in video_cache_v]
            action_attention_mask = action_attention_mask.unsqueeze(0).unsqueeze(1)
            action_attention_mask = action_attention_mask.expand(
                batch_size,
                1,
                action_attention_mask.shape[-2],
                action_attention_mask.shape[-1],
            ).clone()
            action_attention_mask[..., :current_frame_video_tokens] = False
        return super().forward(
            latents_action,
            timestep_action,
            context,
            context_mask,
            video_cache_k,
            video_cache_v,
            action_attention_mask,
            visual_memory,
            visual_proprio,
            current_frame_video_tokens,
        )


class ModalityDropoutCompiledLiberoPolicy(P1DinoCompiledLiberoPolicy):
    """Existing compiled rollout policy with a fixed modality ablation."""

    def __init__(
        self,
        policy,
        *,
        prompt_cache_dir: str | Path,
        condition: str,
        fixed_gaussian_dino: bool,
        compile_enabled: bool = True,
        compile_mode: str = "reduce-overhead",
        parity_dtype_atol: float = 0.02,
    ) -> None:
        normalized = str(condition).strip().lower().replace("-", "_")
        if normalized not in ROLLOUT_MODALITY_CONDITIONS:
            raise ValueError(
                f"Unknown rollout modality condition {condition!r}; expected "
                f"{sorted(ROLLOUT_MODALITY_CONDITIONS)}."
            )
        dino_dropped = normalized in {"dino_drop", "both_drop"}
        self.modality_condition = normalized
        self.fixed_gaussian_dino = bool(fixed_gaussian_dino and not dino_dropped)
        super().__init__(
            policy,
            prompt_cache_dir=prompt_cache_dir,
            compile_enabled=False,
            compile_mode=compile_mode,
            parity_dtype_atol=parity_dtype_atol,
            memory_mode="off" if dino_dropped else "correct",
        )
        self.compile_enabled = bool(compile_enabled)
        self._kernel_eager = _ModalityDropoutDenoiseKernel(
            policy,
            memory_enabled=self.memory_enabled,
            wan_dropped=normalized in {"wan_drop", "both_drop"},
        ).eval()
        self._kernel_compiled = (
            torch.compile(
                self._kernel_eager,
                mode=self.compile_mode,
                fullgraph=False,
                dynamic=False,
            )
            if self.compile_enabled
            else self._kernel_eager
        )
        self._prefill_compiled = (
            torch.compile(
                self.actor._prefill_step_compiled,
                mode="default",
                fullgraph=False,
                dynamic=False,
            )
            if self.compile_enabled
            else self.actor._prefill_step_compiled
        )
        self._random_patch_metadata = None

    def _prepare_condition(self, **kwargs):
        condition = super()._prepare_condition(**kwargs)
        if not self.fixed_gaussian_dino:
            return condition
        if condition.visual is None:
            raise RuntimeError("Fixed-Gaussian rollout lost its visual condition.")
        memory, metadata = fixed_gaussian_patch_memory(
            condition.visual.memory,
            seed=42,
        )
        if self._random_patch_metadata is None:
            self._random_patch_metadata = metadata
        elif metadata != self._random_patch_metadata:
            raise RuntimeError("Fixed-Gaussian rollout patch bank changed.")
        return replace(
            condition,
            visual=replace(condition.visual, memory=memory),
        )

    @property
    def modality_dropout_audit(self) -> dict:
        """Return the static rollout condition and random-bank identity."""

        return {
            "condition": self.modality_condition,
            "wan_dropped": self.modality_condition in {"wan_drop", "both_drop"},
            "dino_dropped": self.modality_condition in {"dino_drop", "both_drop"},
            "fixed_gaussian_dino": self.fixed_gaussian_dino,
            "random_patch_bank": self._random_patch_metadata,
        }
