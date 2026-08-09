"""Action Flow-SDE sampling and replay independent of FastWAM preprocessing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from .flow_sde import (
    flow_ode_mean,
    flow_sde_mean_std,
    gaussian_log_prob,
    normalize_flow_time,
)


@dataclass(frozen=True)
class VelocityOutput:
    """Velocity prediction plus an optional read-only Gate tap payload."""

    velocity: torch.Tensor
    gate_tap: Any = None


@dataclass(frozen=True)
class ActionFlowRollout:
    """Replay material produced by action Flow-SDE sampling."""

    actions: torch.Tensor
    chains: torch.Tensor
    denoise_indices: torch.Tensor
    old_log_probs: torch.Tensor
    gate_taps: tuple[Any, ...]
    timesteps: torch.Tensor


@dataclass(frozen=True)
class ActionFlowReplay:
    """Selected Flow-SDE transition reconstructed under one policy."""

    log_prob: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    velocity_output: VelocityOutput


VelocityFn = Callable[[torch.Tensor, torch.Tensor], VelocityOutput | torch.Tensor]


def _as_velocity_output(result: VelocityOutput | torch.Tensor) -> VelocityOutput:
    if isinstance(result, VelocityOutput):
        return result
    if not torch.is_tensor(result):
        raise TypeError(
            "`velocity_fn` must return a Tensor or VelocityOutput, "
            f"got {type(result)!r}"
        )
    return VelocityOutput(velocity=result)


def _validate_schedule_precision(
    timesteps: torch.Tensor,
    scheduler_deltas: torch.Tensor,
) -> None:
    supported = {torch.float32, torch.float64}
    if timesteps.dtype not in supported or scheduler_deltas.dtype not in supported:
        raise TypeError(
            "Flow-SDE timesteps and deltas must remain in FP32 or FP64; "
            "cast only the timestep passed into the action model."
        )


def sample_denoise_indices(
    batch_size: int,
    num_steps: int,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
    ignore_last: bool = False,
) -> torch.Tensor:
    """Uniformly select one stochastic denoising transition per sample."""

    if batch_size <= 0:
        raise ValueError(f"`batch_size` must be positive, got {batch_size}")
    upper = num_steps - int(ignore_last)
    if upper <= 0:
        raise ValueError(
            "`num_steps` must leave at least one selectable transition, "
            f"got num_steps={num_steps}, ignore_last={ignore_last}"
        )
    return torch.randint(
        0,
        upper,
        (batch_size,),
        device=device,
        generator=generator,
    )


def sample_action_flow_sde(
    initial_noise: torch.Tensor,
    *,
    velocity_fn: VelocityFn,
    timesteps: torch.Tensor,
    scheduler_deltas: torch.Tensor,
    num_train_timesteps: int,
    noise_level: float,
    denoise_indices: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
    gate_last_n: int = 1,
    ignore_last_transition: bool = False,
    stochastic: bool = True,
    collect_replay: bool = True,
) -> ActionFlowRollout:
    """Sample an action chain with one Flow-SDE transition per sample.

    Args:
        initial_noise: Initial action noise with shape ``[B, H, A]``.
        velocity_fn: Callable receiving the current action state and the
            scheduler timestep for each batch item.
        timesteps: FastWAM scheduler timesteps with shape ``[S]``.
        scheduler_deltas: FastWAM negative ODE deltas with shape ``[S]``.
        num_train_timesteps: Scheduler normalization constant.
        noise_level: pi-RL Flow-SDE diffusion level.
        denoise_indices: Optional selected step per batch item.
        generator: Sampling generator.
        gate_last_n: Number of final velocity-call tap payloads to retain.
        ignore_last_transition: Exclude the final denoising transition from
            stochastic-index selection. This preserves a final deterministic
            ODE step after the single Flow-SDE transition.
        stochastic: If false, run the deterministic ODE and return index ``-1``.
        collect_replay: Retain the full action chain and selected log-probability.
            Evaluation can disable this while still collecting final Gate taps.
    """

    if initial_noise.ndim < 2:
        raise ValueError(
            f"`initial_noise` must include batch and action dims, got {initial_noise.shape}"
        )
    if timesteps.ndim != 1 or scheduler_deltas.ndim != 1:
        raise ValueError("`timesteps` and `scheduler_deltas` must be one-dimensional.")
    if timesteps.shape != scheduler_deltas.shape:
        raise ValueError(
            f"Schedule shape mismatch: {timesteps.shape} vs {scheduler_deltas.shape}"
        )
    _validate_schedule_precision(timesteps, scheduler_deltas)
    if gate_last_n < 1 or gate_last_n > timesteps.numel():
        raise ValueError(
            f"`gate_last_n` must be in [1, {timesteps.numel()}], got {gate_last_n}"
        )

    batch_size = initial_noise.shape[0]
    device = initial_noise.device
    num_steps = timesteps.numel()
    normalized_times = normalize_flow_time(
        timesteps,
        num_train_timesteps=num_train_timesteps,
    ).to(device=device)
    next_times = normalized_times + scheduler_deltas.to(
        device=device, dtype=torch.float32
    )

    if stochastic:
        if denoise_indices is None:
            denoise_indices = sample_denoise_indices(
                batch_size,
                num_steps,
                device=device,
                generator=generator,
                ignore_last=ignore_last_transition,
            )
        else:
            denoise_indices = denoise_indices.to(device=device, dtype=torch.long)
        if denoise_indices.shape != (batch_size,):
            raise ValueError(
                "`denoise_indices` must have shape "
                f"({batch_size},), got {tuple(denoise_indices.shape)}"
            )
        upper = num_steps - int(ignore_last_transition)
        if bool(((denoise_indices < 0) | (denoise_indices >= upper)).any()):
            raise ValueError("`denoise_indices` contains an out-of-range step.")
    else:
        denoise_indices = torch.full(
            (batch_size,),
            -1,
            device=device,
            dtype=torch.long,
        )

    x_t = initial_noise
    chains = [x_t] if collect_replay else None
    selected_log_prob = (
        torch.zeros_like(x_t, dtype=torch.float32) if collect_replay else None
    )
    gate_taps: list[Any] = []
    gate_start = num_steps - gate_last_n

    for step_idx in range(num_steps):
        time = normalized_times[step_idx].expand(batch_size)
        next_time = next_times[step_idx].expand(batch_size)
        model_timestep = (
            timesteps[step_idx].to(device=device, dtype=x_t.dtype).expand(batch_size)
        )
        output = _as_velocity_output(velocity_fn(x_t, model_timestep))
        if output.velocity.shape != x_t.shape:
            raise ValueError(
                "`velocity_fn` returned the wrong shape: "
                f"{output.velocity.shape} vs {x_t.shape}"
            )
        if step_idx >= gate_start:
            gate_taps.append(output.gate_tap)

        ode_next = flow_ode_mean(
            x_t,
            output.velocity,
            time=time,
            next_time=next_time,
        )
        selected = denoise_indices == step_idx
        if stochastic and bool(selected.any()):
            mean, std = flow_sde_mean_std(
                x_t,
                output.velocity,
                time=time,
                next_time=next_time,
                noise_level=noise_level,
            )
            noise = torch.randn(
                x_t.shape,
                generator=generator,
                device=device,
                dtype=x_t.dtype,
            )
            sde_next = mean + noise * std
            selected_mask = selected.view(batch_size, *([1] * (x_t.ndim - 1)))
            x_next = torch.where(selected_mask, sde_next, ode_next)
            if selected_log_prob is not None:
                transition_log_prob = gaussian_log_prob(sde_next, mean, std)
                selected_log_prob = torch.where(
                    selected_mask,
                    transition_log_prob,
                    selected_log_prob,
                )
        else:
            x_next = ode_next

        x_t = x_next
        if chains is not None:
            chains.append(x_t)

    return ActionFlowRollout(
        actions=x_t,
        chains=(
            torch.stack(chains, dim=1)
            if chains is not None
            else initial_noise.new_empty((batch_size, 0, *initial_noise.shape[1:]))
        ),
        denoise_indices=denoise_indices,
        old_log_probs=(
            selected_log_prob
            if selected_log_prob is not None
            else initial_noise.new_empty((batch_size, 0), dtype=torch.float32)
        ),
        gate_taps=tuple(gate_taps),
        timesteps=timesteps,
    )


def replay_action_flow_sde_transition(
    chains: torch.Tensor,
    denoise_indices: torch.Tensor,
    *,
    velocity_fn: VelocityFn,
    timesteps: torch.Tensor,
    scheduler_deltas: torch.Tensor,
    num_train_timesteps: int,
    noise_level: float,
) -> ActionFlowReplay:
    """Reconstruct the selected Gaussian transition under current parameters."""

    if chains.ndim < 3:
        raise ValueError(f"`chains` must have shape [B, S+1, ...], got {chains.shape}")
    _validate_schedule_precision(timesteps, scheduler_deltas)
    batch_size, chain_steps = chains.shape[:2]
    if chain_steps != timesteps.numel() + 1:
        raise ValueError(
            f"Chain has {chain_steps} states for {timesteps.numel()} transitions."
        )
    denoise_indices = denoise_indices.to(device=chains.device, dtype=torch.long)
    if denoise_indices.shape != (batch_size,):
        raise ValueError(
            f"`denoise_indices` must have shape ({batch_size},), "
            f"got {tuple(denoise_indices.shape)}"
        )
    if bool(((denoise_indices < 0) | (denoise_indices >= timesteps.numel())).any()):
        raise ValueError("Replay requires a valid stochastic denoising index.")

    batch = torch.arange(batch_size, device=chains.device)
    x_t = chains[batch, denoise_indices]
    sampled_next = chains[batch, denoise_indices + 1]
    selected_schedule_timestep = timesteps.to(device=chains.device)[denoise_indices]
    selected_model_timestep = selected_schedule_timestep.to(dtype=chains.dtype)
    output = _as_velocity_output(velocity_fn(x_t, selected_model_timestep))

    normalized_times = normalize_flow_time(
        selected_schedule_timestep,
        num_train_timesteps=num_train_timesteps,
    ).to(device=chains.device)
    selected_delta = scheduler_deltas.to(device=chains.device)[denoise_indices]
    next_times = normalized_times + selected_delta.to(dtype=torch.float32)
    mean, std = flow_sde_mean_std(
        x_t,
        output.velocity,
        time=normalized_times,
        next_time=next_times,
        noise_level=noise_level,
    )
    return ActionFlowReplay(
        log_prob=gaussian_log_prob(sampled_next, mean, std),
        mean=mean,
        std=std,
        velocity_output=output,
    )


def replay_action_flow_sde_log_prob(
    chains: torch.Tensor,
    denoise_indices: torch.Tensor,
    *,
    velocity_fn: VelocityFn,
    timesteps: torch.Tensor,
    scheduler_deltas: torch.Tensor,
    num_train_timesteps: int,
    noise_level: float,
) -> tuple[torch.Tensor, VelocityOutput]:
    """Recompute selected Flow-SDE log-probabilities with current parameters."""

    replay = replay_action_flow_sde_transition(
        chains,
        denoise_indices,
        velocity_fn=velocity_fn,
        timesteps=timesteps,
        scheduler_deltas=scheduler_deltas,
        num_train_timesteps=num_train_timesteps,
        noise_level=noise_level,
    )
    return replay.log_prob, replay.velocity_output
