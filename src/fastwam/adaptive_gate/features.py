"""Shared, deterministic feature pooling for gate training and rollout."""
from __future__ import annotations

import torch
import torch.nn.functional as F


DEFAULT_TEXT_FEAT_DIM = 64
TEXT_FEAT_LAYOUT = "masked_mean_adaptive_avg_pool_v1"


def pool_text_context(
    context: torch.Tensor,
    context_mask: torch.Tensor,
    *,
    output_dim: int = DEFAULT_TEXT_FEAT_DIM,
) -> torch.Tensor:
    """Mask-pool text tokens and deterministically compress the embedding axis.

    Args:
        context: ``[L, D]`` or ``[B, L, D]`` text embeddings.
        context_mask: matching ``[L]`` or ``[B, L]`` validity mask.
        output_dim: fixed output width used by both online rollout and oracle BC.

    Returns:
        Float32 ``[output_dim]`` or ``[B, output_dim]`` on the input device.
    """
    if int(output_dim) <= 0:
        raise ValueError(f"output_dim must be positive, got {output_dim}.")
    squeeze = context.ndim == 2
    if squeeze:
        context = context.unsqueeze(0)
    if context_mask.ndim == 1:
        context_mask = context_mask.unsqueeze(0)
    if context.ndim != 3 or context_mask.ndim != 2:
        raise ValueError(
            "context/context_mask must be [B,L,D]/[B,L] or [L,D]/[L], got "
            f"{tuple(context.shape)} and {tuple(context_mask.shape)}"
        )
    if context.shape[:2] != context_mask.shape:
        raise ValueError(
            f"context/mask shape mismatch: {tuple(context.shape)} vs "
            f"{tuple(context_mask.shape)}"
        )

    mask = context_mask.to(device=context.device, dtype=torch.bool)
    counts = mask.sum(dim=1, keepdim=True)
    if bool((counts == 0).any()):
        raise ValueError("context_mask must contain at least one valid token per sample.")
    pooled = (
        context.float() * mask.unsqueeze(-1).to(dtype=torch.float32)
    ).sum(dim=1) / counts.to(dtype=torch.float32)
    compressed = F.adaptive_avg_pool1d(pooled.unsqueeze(1), int(output_dim)).squeeze(1)
    return compressed[0] if squeeze else compressed
