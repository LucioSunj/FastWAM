"""Flow-SDE transition utilities for FastWAM action sampling.

The FastWAM scheduler stores timesteps on ``[0, num_train_timesteps]`` and
returns a negative ODE delta.  pi-RL's Flow-SDE equations instead use a
normalized reverse-time step ``delta = t_i - t_{i+1} > 0``.  This module keeps
that conversion explicit so callers cannot silently use the wrong sign.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class FlowSDETransition:
    """One sampled reverse-time action transition."""

    sample: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    log_prob: torch.Tensor
    time: torch.Tensor
    next_time: torch.Tensor


def normalize_flow_time(
    timestep: torch.Tensor,
    *,
    num_train_timesteps: int,
) -> torch.Tensor:
    """Convert a FastWAM scheduler timestep to normalized flow time."""

    if num_train_timesteps <= 0:
        raise ValueError(
            f"`num_train_timesteps` must be positive, got {num_train_timesteps}"
        )
    time = timestep.to(dtype=torch.float32) / float(num_train_timesteps)
    if bool(((time < 0) | (time > 1)).any()):
        raise ValueError("Normalized flow timesteps must lie in [0, 1].")
    return time


def reverse_step_size(time: torch.Tensor, next_time: torch.Tensor) -> torch.Tensor:
    """Return the positive reverse-time step ``time - next_time``."""

    time, next_time = torch.broadcast_tensors(
        time.to(dtype=torch.float32), next_time.to(dtype=torch.float32)
    )
    delta = time - next_time
    if bool((delta <= 0).any()):
        raise ValueError("Flow-SDE requires strictly decreasing timesteps.")
    return delta


def _expand_like_batch(value: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
    """Broadcast a scalar or batch vector over non-batch sample dimensions."""

    value = value.to(device=sample.device, dtype=sample.dtype)
    if value.ndim == 0:
        return value
    if value.ndim != 1:
        raise ValueError(
            f"Expected a scalar or batch vector, got shape {tuple(value.shape)}"
        )
    if value.shape[0] != sample.shape[0]:
        raise ValueError(
            f"Batch mismatch: value has {value.shape[0]}, sample has {sample.shape[0]}"
        )
    return value.view(value.shape[0], *([1] * (sample.ndim - 1)))


def flow_ode_mean(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    *,
    time: torch.Tensor,
    next_time: torch.Tensor,
) -> torch.Tensor:
    """Compute the deterministic FastWAM reverse ODE update."""

    if x_t.shape != velocity.shape:
        raise ValueError(
            f"`x_t` and `velocity` must match, got {x_t.shape} and {velocity.shape}"
        )
    delta = reverse_step_size(time, next_time)
    return x_t - _expand_like_batch(delta, x_t) * velocity


def flow_sde_mean_std(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    *,
    time: torch.Tensor,
    next_time: torch.Tensor,
    noise_level: float | torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the pi-RL Flow-SDE Gaussian transition parameters.

    The first reverse step starts at ``t=1``.  Following RLinf's pi-RL
    implementation, its diffusion denominator uses the next scheduled time to
    avoid the singular ``1 - t`` term.
    """

    if x_t.shape != velocity.shape:
        raise ValueError(
            f"`x_t` and `velocity` must match, got {x_t.shape} and {velocity.shape}"
        )
    if eps <= 0:
        raise ValueError(f"`eps` must be positive, got {eps}")

    time, next_time = torch.broadcast_tensors(
        time.to(device=x_t.device, dtype=torch.float32),
        next_time.to(device=x_t.device, dtype=torch.float32),
    )
    delta = reverse_step_size(time, next_time).to(device=x_t.device)
    if bool((time <= 0).any()):
        raise ValueError("Flow-SDE cannot be evaluated at t <= 0.")

    noise_level_tensor = torch.as_tensor(
        noise_level, device=x_t.device, dtype=torch.float32
    )
    if bool((noise_level_tensor < 0).any()):
        raise ValueError("`noise_level` must be non-negative.")

    denominator_time = torch.where(
        torch.isclose(time, torch.ones_like(time)),
        next_time,
        time,
    )
    denominator = (1.0 - denominator_time).clamp_min(eps)
    sigma = noise_level_tensor * torch.sqrt(time / denominator)

    time_x = _expand_like_batch(time, x_t)
    delta_x = _expand_like_batch(delta, x_t)
    sigma_x = _expand_like_batch(sigma, x_t)

    x0_pred = x_t - velocity * time_x
    x1_pred = x_t + velocity * (1.0 - time_x)

    x0_weight = 1.0 - (time_x - delta_x)
    x1_weight = time_x - delta_x - sigma_x.square() * delta_x / (2.0 * time_x)
    mean = x0_pred * x0_weight + x1_pred * x1_weight
    std = torch.sqrt(delta_x) * sigma_x
    return mean, std


def gaussian_log_prob(
    sample: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Return elementwise Gaussian log probability in FP32."""

    if sample.shape != mean.shape:
        raise ValueError(
            f"`sample` and `mean` must match, got {sample.shape} and {mean.shape}"
        )
    try:
        std = torch.broadcast_to(std, sample.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"`std` with shape {std.shape} is not broadcastable to {sample.shape}"
        ) from exc
    if bool((std <= 0).any()):
        raise ValueError("Gaussian standard deviation must be strictly positive.")

    sample_fp32 = sample.to(dtype=torch.float32)
    mean_fp32 = mean.to(dtype=torch.float32)
    std_fp32 = std.to(dtype=torch.float32)
    return (
        -torch.log(std_fp32)
        - 0.5 * torch.log(torch.tensor(2.0 * torch.pi, device=sample.device))
        - 0.5 * ((sample_fp32 - mean_fp32) / std_fp32).square()
    )


def sample_flow_sde_transition(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    *,
    time: torch.Tensor,
    next_time: torch.Tensor,
    noise_level: float | torch.Tensor,
    generator: Optional[torch.Generator] = None,
) -> FlowSDETransition:
    """Sample one Flow-SDE transition and retain its exact log probability."""

    mean, std = flow_sde_mean_std(
        x_t,
        velocity,
        time=time,
        next_time=next_time,
        noise_level=noise_level,
    )
    noise = torch.randn(
        x_t.shape,
        generator=generator,
        device=x_t.device,
        dtype=x_t.dtype,
    )
    sample = mean + noise * std
    return FlowSDETransition(
        sample=sample,
        mean=mean,
        std=std,
        log_prob=gaussian_log_prob(sample, mean, std),
        time=time,
        next_time=next_time,
    )
