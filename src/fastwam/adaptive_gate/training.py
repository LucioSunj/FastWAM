"""Lightweight dual-regime objective shared by separate and fused trainers."""
from __future__ import annotations

import math

import torch


def normalized_dual_regime_action_loss(
    main_loss: torch.Tensor,
    uncond_loss: torch.Tensor,
    uncond_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return combined loss plus each regime's normalized contribution."""
    weight = float(uncond_weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(f"action_regime_weight_uncond must be finite and >= 0, got {weight}.")
    denom = 1.0 + weight
    main_contribution = main_loss / denom
    uncond_contribution = weight * uncond_loss / denom
    return main_contribution + uncond_contribution, main_contribution, uncond_contribution
