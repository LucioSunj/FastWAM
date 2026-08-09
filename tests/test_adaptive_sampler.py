import pytest
import torch

from fastwam.models.wan22.adaptive_sampler import (
    VelocityOutput,
    replay_action_flow_sde_log_prob,
    sample_action_flow_sde,
    sample_denoise_indices,
)
from fastwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)


def _schedule(num_steps=4):
    scheduler = WanContinuousFlowMatchScheduler(
        num_train_timesteps=1000,
        shift=5.0,
    )
    timesteps, deltas = scheduler.build_inference_schedule(
        num_inference_steps=num_steps,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    return scheduler, timesteps, deltas


def test_deterministic_sampler_matches_scheduler_steps():
    scheduler, timesteps, deltas = _schedule()
    initial = torch.randn(2, 3, 4)

    def velocity_fn(x_t, timestep):
        del timestep
        return torch.full_like(x_t, 0.25)

    expected = initial
    for delta in deltas:
        expected = scheduler.step(torch.full_like(expected, 0.25), delta, expected)

    rollout = sample_action_flow_sde(
        initial,
        velocity_fn=velocity_fn,
        timesteps=timesteps,
        scheduler_deltas=deltas,
        num_train_timesteps=1000,
        noise_level=0.5,
        stochastic=False,
    )
    torch.testing.assert_close(rollout.actions, expected)
    assert torch.equal(rollout.denoise_indices, torch.full((2,), -1))


def test_sampler_uses_one_selected_transition_per_sample_and_replays_logprob():
    _, timesteps, deltas = _schedule()
    initial = torch.randn(3, 2, 4)
    weight = torch.tensor(0.1, requires_grad=True)

    def velocity_fn(x_t, timestep):
        scale = timestep.view(-1, 1, 1) / 1000.0
        return x_t * weight + scale

    selected = torch.tensor([0, 2, 3])
    rollout = sample_action_flow_sde(
        initial,
        velocity_fn=velocity_fn,
        timesteps=timesteps,
        scheduler_deltas=deltas,
        num_train_timesteps=1000,
        noise_level=0.5,
        denoise_indices=selected,
        generator=torch.Generator().manual_seed(9),
    )
    replay_log_prob, _ = replay_action_flow_sde_log_prob(
        rollout.chains.detach(),
        rollout.denoise_indices,
        velocity_fn=velocity_fn,
        timesteps=timesteps,
        scheduler_deltas=deltas,
        num_train_timesteps=1000,
        noise_level=0.5,
    )
    torch.testing.assert_close(replay_log_prob, rollout.old_log_probs)

    replay_log_prob.mean().backward()
    assert weight.grad is not None
    assert torch.isfinite(weight.grad)


def test_gate_taps_keep_only_last_n_velocity_calls():
    _, timesteps, deltas = _schedule(num_steps=5)
    initial = torch.zeros(1, 2, 3)
    seen = []

    def velocity_fn(x_t, timestep):
        tap = int(timestep.item())
        seen.append(tap)
        return VelocityOutput(torch.zeros_like(x_t), gate_tap=tap)

    rollout = sample_action_flow_sde(
        initial,
        velocity_fn=velocity_fn,
        timesteps=timesteps,
        scheduler_deltas=deltas,
        num_train_timesteps=1000,
        noise_level=0.5,
        gate_last_n=2,
        stochastic=False,
    )
    assert rollout.gate_taps == tuple(seen[-2:])


def test_eval_sampling_can_omit_replay_chain_and_log_probs():
    initial = torch.zeros(2, 3, 4)
    timesteps = torch.tensor([1000.0, 500.0])
    deltas = torch.tensor([-0.5, -0.5])

    rollout = sample_action_flow_sde(
        initial,
        velocity_fn=lambda x_t, _t: VelocityOutput(
            torch.ones_like(x_t), gate_tap={"batch": x_t.shape[0]}
        ),
        timesteps=timesteps,
        scheduler_deltas=deltas,
        num_train_timesteps=1000,
        noise_level=0.5,
        stochastic=False,
        gate_last_n=1,
        collect_replay=False,
    )

    assert rollout.chains.shape == (2, 0, 3, 4)
    assert rollout.old_log_probs.shape == (2, 0)
    assert len(rollout.gate_taps) == 1


def test_bfloat16_actions_use_fp32_default_schedule_through_final_sde_step():
    scheduler, timesteps, deltas = _schedule(num_steps=20)
    initial_noise = torch.zeros(1, 2, 3, dtype=torch.bfloat16)

    rollout = sample_action_flow_sde(
        initial_noise,
        velocity_fn=lambda action, _timestep: torch.zeros_like(action),
        timesteps=timesteps,
        scheduler_deltas=deltas,
        num_train_timesteps=scheduler.num_train_timesteps,
        noise_level=0.5,
        denoise_indices=torch.tensor([19]),
    )

    assert torch.isfinite(rollout.actions).all()
    assert torch.isfinite(rollout.old_log_probs).all()


def test_flow_sde_rejects_a_bfloat16_schedule_before_sampling():
    scheduler = WanContinuousFlowMatchScheduler(
        num_train_timesteps=1000,
        shift=5.0,
    )
    timesteps, deltas = scheduler.build_inference_schedule(
        num_inference_steps=20,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    with pytest.raises(TypeError, match="must remain in FP32 or FP64"):
        sample_action_flow_sde(
            torch.zeros(1, 2, 3, dtype=torch.bfloat16),
            velocity_fn=lambda action, _timestep: torch.zeros_like(action),
            timesteps=timesteps,
            scheduler_deltas=deltas,
            num_train_timesteps=1000,
            noise_level=0.5,
            denoise_indices=torch.tensor([19]),
        )


def test_uniform_index_sampler_respects_ignore_last():
    indices = sample_denoise_indices(
        100,
        4,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(3),
        ignore_last=True,
    )
    assert int(indices.min()) >= 0
    assert int(indices.max()) <= 2


def test_flow_sde_sampler_excludes_final_transition_when_requested():
    _, timesteps, deltas = _schedule(num_steps=4)
    initial = torch.zeros(128, 2, 3)

    rollout = sample_action_flow_sde(
        initial,
        velocity_fn=lambda action, _timestep: torch.zeros_like(action),
        timesteps=timesteps,
        scheduler_deltas=deltas,
        num_train_timesteps=1000,
        noise_level=0.5,
        generator=torch.Generator().manual_seed(42),
        ignore_last_transition=True,
    )

    assert int(rollout.denoise_indices.min()) >= 0
    assert int(rollout.denoise_indices.max()) <= 2

    with pytest.raises(ValueError, match="out-of-range"):
        sample_action_flow_sde(
            initial[:1],
            velocity_fn=lambda action, _timestep: torch.zeros_like(action),
            timesteps=timesteps,
            scheduler_deltas=deltas,
            num_train_timesteps=1000,
            noise_level=0.5,
            denoise_indices=torch.tensor([3]),
            ignore_last_transition=True,
        )


def test_bad_chain_shape_is_rejected():
    _, timesteps, deltas = _schedule()
    chains = torch.zeros(2, 4, 3, 4)

    try:
        replay_action_flow_sde_log_prob(
            chains,
            torch.tensor([0, 1]),
            velocity_fn=lambda x, t: x,
            timesteps=timesteps,
            scheduler_deltas=deltas,
            num_train_timesteps=1000,
            noise_level=0.5,
        )
    except ValueError as exc:
        assert "states" in str(exc)
    else:
        raise AssertionError("Expected a ValueError")


def test_mixed_precision_replay_matches_rollout_timestep_dtype():
    _, timesteps, deltas = _schedule()
    initial = torch.randn(2, 2, 3, dtype=torch.bfloat16)
    seen_dtypes = []

    def velocity_fn(x_t, timestep):
        seen_dtypes.append((x_t.dtype, timestep.dtype))
        assert timestep.dtype == x_t.dtype
        scale = timestep.view(-1, 1, 1) / 1000
        return x_t * 0.125 + scale

    rollout = sample_action_flow_sde(
        initial,
        velocity_fn=velocity_fn,
        timesteps=timesteps,
        scheduler_deltas=deltas,
        num_train_timesteps=1000,
        noise_level=0.5,
        denoise_indices=torch.tensor([1, 3]),
        generator=torch.Generator().manual_seed(29),
    )
    replayed, _ = replay_action_flow_sde_log_prob(
        rollout.chains.detach(),
        rollout.denoise_indices,
        velocity_fn=velocity_fn,
        timesteps=timesteps,
        scheduler_deltas=deltas,
        num_train_timesteps=1000,
        noise_level=0.5,
    )

    torch.testing.assert_close(replayed, rollout.old_log_probs)
    assert seen_dtypes
    assert all(state_dtype == time_dtype for state_dtype, time_dtype in seen_dtypes)
