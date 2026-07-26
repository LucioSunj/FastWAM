"""Pure flow-matching objective helpers for the stage-2 sampler interface (W8).

Two facts about the FastWAM action expert anchor everything here:

* the scheduler is continuous-time rectified flow — ``x_sigma = (1-sigma)*x0 +
  sigma*eps`` with regression target ``v = eps - x0`` and timesteps
  parameterized as ``t = sigma * num_train_timesteps``
  (``models/wan22/schedulers/scheduler_continuous.py``);
* the training action loss is the weighted, pad-masked, per-token MSE reduction
  implemented inline in ``FastWAM.training_loss``.

:func:`cfm_loss` reproduces that reduction *exactly* so downstream consumers
(FPO-style ratios need ``L_CFM_old`` recomputable from cached ``(t, eps)``) can
rely on float-level equality with the training loss rather than an
approximation. :func:`predicted_clean_action` is the closed form
``x0_hat = x_sigma - sigma*v`` that velocity-field guidance evaluates at every
solver step. Everything in this module is a pure function of its inputs: no
module state, no RNG, no device assumptions.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def sigma_from_timestep(timestep: torch.Tensor, num_train_timesteps: int) -> torch.Tensor:
    """Convert scheduler timesteps ``t = sigma * N`` back to ``sigma`` in [0, 1]."""
    steps = int(num_train_timesteps)
    if steps <= 0:
        raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}")
    if not torch.is_tensor(timestep):
        raise TypeError(f"`timestep` must be a torch.Tensor, got {type(timestep).__name__}")
    return timestep.to(dtype=torch.float32) / float(steps)


def predicted_clean_action(
    x_sigma: torch.Tensor, sigma: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Closed-form clean-action estimate ``x0_hat = x_sigma - sigma * v``.

    Derivation: ``x_sigma = (1-sigma)*x0 + sigma*eps`` and ``v = eps - x0``
    give ``x_sigma - sigma*v = x0`` exactly. ``sigma`` is the *unit-interval*
    noise level — convert scheduler timesteps with :func:`sigma_from_timestep`
    first; passing a raw timestep (e.g. 800.0) here is a unit error.

    Args:
        x_sigma: Noisy action latents ``[B, T, D]``.
        sigma: Noise level in [0, 1]; scalar, ``[1]``, or per-sample ``[B]``.
        v: Predicted velocity, same shape as ``x_sigma``.
    """
    for name, value in (("x_sigma", x_sigma), ("sigma", sigma), ("v", v)):
        if not torch.is_tensor(value):
            raise TypeError(f"`{name}` must be a torch.Tensor, got {type(value).__name__}")
    if x_sigma.shape != v.shape:
        raise ValueError(
            f"`x_sigma` and `v` shapes must match, got {tuple(x_sigma.shape)} vs {tuple(v.shape)}"
        )
    if sigma.numel() not in (1, x_sigma.shape[0]):
        raise ValueError(
            f"`sigma` must have 1 or batch({x_sigma.shape[0]}) elements, got {sigma.numel()}"
        )
    sigma = sigma.to(device=x_sigma.device, dtype=x_sigma.dtype)
    if sigma.numel() == 1:
        return x_sigma - sigma.reshape(()) * v
    return x_sigma - sigma.view(-1, *([1] * (x_sigma.ndim - 1))) * v


def cfm_loss(
    pred_v: torch.Tensor,
    x0: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor,
    *,
    scheduler,
    action_is_pad: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """The FastWAM action flow-matching loss as a pure function.

    Bit-for-bit the reduction from ``FastWAM.training_loss``: per-token MSE in
    float32 against ``scheduler.training_target(x0, noise, timestep)``, meaned
    over the action dim, pad-masked mean over time (``clamp(min=1.0)`` on the
    valid count), weighted by ``scheduler.training_weight(timestep)``, then
    meaned over the batch. Replaying a cached ``(timestep, noise)`` pair through
    this function reproduces the training-time loss value exactly, which is the
    property FPO-style ratios depend on.

    Args:
        pred_v: Predicted velocity ``[B, T, D]``.
        x0: Clean action chunk ``[B, T, D]``.
        noise: The noise draw ``eps`` used to build ``x_sigma`` — NOT ``x_sigma``
            itself.
        timestep: Scheduler timesteps ``[B]`` (``t = sigma * N``).
        scheduler: A ``WanContinuousFlowMatchScheduler`` (supplies
            ``training_target`` and the shift-dependent ``training_weight``).
        action_is_pad: Optional ``[B, T]`` bool mask, True where padded.
        reduction: ``"mean"`` for the scalar training loss, ``"none"`` for the
            weighted per-sample vector ``[B]``.

    Returns:
        Scalar (``"mean"``) or ``[B]`` (``"none"``), float32.
    """
    for name, value in (("pred_v", pred_v), ("x0", x0), ("noise", noise)):
        if not torch.is_tensor(value) or value.ndim != 3:
            raise ValueError(
                f"`{name}` must be a 3D [B, T, D] tensor, got "
                f"{type(value).__name__}{tuple(value.shape) if torch.is_tensor(value) else ''}"
            )
    if not (pred_v.shape == x0.shape == noise.shape):
        raise ValueError(
            "`pred_v`, `x0`, `noise` shapes must match, got "
            f"{tuple(pred_v.shape)}, {tuple(x0.shape)}, {tuple(noise.shape)}"
        )
    if not torch.is_tensor(timestep) or timestep.ndim != 1 or timestep.shape[0] != pred_v.shape[0]:
        raise ValueError(
            f"`timestep` must be a 1D [B={pred_v.shape[0]}] tensor, got "
            f"{tuple(timestep.shape) if torch.is_tensor(timestep) else type(timestep).__name__}"
        )
    if reduction not in ("mean", "none"):
        raise ValueError(f"`reduction` must be 'mean' or 'none', got {reduction!r}")

    target = scheduler.training_target(x0, noise, timestep)
    loss_token = F.mse_loss(pred_v.float(), target.float(), reduction="none").mean(dim=2)  # [B, T]
    if action_is_pad is not None:
        if action_is_pad.shape != loss_token.shape:
            raise ValueError(
                f"`action_is_pad` must be [B, T]={tuple(loss_token.shape)}, got "
                f"{tuple(action_is_pad.shape)}"
            )
        valid = (~action_is_pad).to(device=loss_token.device, dtype=loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        per_sample = (loss_token * valid).sum(dim=1) / valid_sum
    else:
        per_sample = loss_token.mean(dim=1)

    weight = scheduler.training_weight(timestep).to(
        device=per_sample.device, dtype=per_sample.dtype
    )
    weighted = per_sample * weight
    return weighted.mean() if reduction == "mean" else weighted
