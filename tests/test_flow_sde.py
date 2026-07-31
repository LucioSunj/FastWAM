import pytest
import torch

from fastwam.models.wan22.flow_sde import (
    flow_ode_mean,
    flow_sde_mean_std,
    gaussian_log_prob,
    normalize_flow_time,
    reverse_step_size,
    sample_flow_sde_transition,
)
from fastwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)


def test_normalized_time_and_reverse_delta_follow_shifted_schedule():
    scheduler = WanContinuousFlowMatchScheduler(
        num_train_timesteps=1000,
        shift=5.0,
    )
    timesteps, scheduler_deltas = scheduler.build_inference_schedule(
        num_inference_steps=5,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    times = normalize_flow_time(timesteps, num_train_timesteps=1000)
    next_times = times + scheduler_deltas

    assert torch.all(reverse_step_size(times, next_times) > 0)
    torch.testing.assert_close(
        reverse_step_size(times, next_times),
        -scheduler_deltas,
    )


def test_flow_ode_mean_matches_fastwam_scheduler_step():
    scheduler = WanContinuousFlowMatchScheduler(
        num_train_timesteps=1000,
        shift=5.0,
    )
    timesteps, scheduler_deltas = scheduler.build_inference_schedule(
        num_inference_steps=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    x_t = torch.randn(2, 3, 4)
    velocity = torch.randn_like(x_t)
    time = normalize_flow_time(timesteps[1], num_train_timesteps=1000)
    next_time = time + scheduler_deltas[1]

    expected = scheduler.step(velocity, scheduler_deltas[1], x_t)
    actual = flow_ode_mean(
        x_t,
        velocity,
        time=time,
        next_time=next_time,
    )
    torch.testing.assert_close(actual, expected)


def test_flow_sde_parameters_match_direct_pi_rl_equations():
    x_t = torch.tensor([[[0.2, -0.4]]], dtype=torch.float32)
    velocity = torch.tensor([[[0.5, 0.25]]], dtype=torch.float32)
    time = torch.tensor([0.8])
    next_time = torch.tensor([0.6])
    noise_level = 0.5

    mean, std = flow_sde_mean_std(
        x_t,
        velocity,
        time=time,
        next_time=next_time,
        noise_level=noise_level,
    )

    delta = (time - next_time).view(1, 1, 1)
    time_x = time.view(1, 1, 1)
    sigma = (
        noise_level * torch.sqrt(time / (1.0 - time))
    ).view(1, 1, 1)
    x0_pred = x_t - velocity * time_x
    x1_pred = x_t + velocity * (1.0 - time_x)
    expected_mean = (
        x0_pred * (1.0 - (time_x - delta))
        + x1_pred
        * (time_x - delta - sigma.square() * delta / (2.0 * time_x))
    )
    expected_std = torch.sqrt(delta) * sigma

    torch.testing.assert_close(mean, expected_mean)
    torch.testing.assert_close(std, expected_std)


def test_first_flow_sde_step_uses_next_time_for_finite_sigma():
    x_t = torch.zeros(1, 2, 3)
    velocity = torch.ones_like(x_t)
    mean, std = flow_sde_mean_std(
        x_t,
        velocity,
        time=torch.tensor([1.0]),
        next_time=torch.tensor([0.75]),
        noise_level=0.5,
    )
    assert torch.isfinite(mean).all()
    assert torch.isfinite(std).all()
    assert torch.all(std > 0)


def test_tiny_schedule_roundoff_below_zero_is_clamped():
    sample = torch.zeros(1, 2, 3)
    mean, std = flow_sde_mean_std(
        sample,
        sample,
        time=torch.tensor([0.1]),
        next_time=torch.tensor([-1e-8]),
        noise_level=0.5,
    )
    assert torch.isfinite(mean).all()
    assert torch.isfinite(std).all()


def test_gaussian_log_prob_matches_torch_distribution():
    sample = torch.randn(2, 3, 4)
    mean = torch.randn_like(sample)
    std = torch.rand(2, 1, 1) + 0.1
    expected = torch.distributions.Normal(mean, std).log_prob(sample)
    actual = gaussian_log_prob(sample, mean, std)
    torch.testing.assert_close(actual, expected)


def test_sampled_transition_preserves_velocity_gradients():
    x_t = torch.randn(2, 3, 4)
    velocity = torch.randn_like(x_t, requires_grad=True)
    transition = sample_flow_sde_transition(
        x_t,
        velocity,
        time=torch.tensor([0.8, 0.7]),
        next_time=torch.tensor([0.6, 0.5]),
        noise_level=0.5,
        generator=torch.Generator().manual_seed(7),
    )
    loss = transition.mean.square().mean()
    loss.backward()
    assert velocity.grad is not None
    assert torch.isfinite(velocity.grad).all()


def test_invalid_non_decreasing_schedule_is_rejected():
    try:
        reverse_step_size(torch.tensor([0.5]), torch.tensor([0.5]))
    except ValueError as exc:
        assert "strictly decreasing" in str(exc)
    else:
        raise AssertionError("Expected a ValueError")


@pytest.mark.parametrize(
    "bad_time", [float("nan"), float("inf"), float("-inf")]
)
def test_nonfinite_flow_times_are_rejected(bad_time):
    with pytest.raises(ValueError, match="finite"):
        reverse_step_size(torch.tensor([bad_time]), torch.tensor([0.5]))


@pytest.mark.parametrize(
    "noise_level", [float("nan"), float("inf"), float("-inf")]
)
def test_nonfinite_noise_level_is_rejected(noise_level):
    sample = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="noise_level.*finite"):
        flow_sde_mean_std(
            sample,
            sample,
            time=torch.tensor([0.8]),
            next_time=torch.tensor([0.6]),
            noise_level=noise_level,
        )


@pytest.mark.parametrize("eps", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_eps_is_rejected(eps):
    sample = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="eps.*positive"):
        flow_sde_mean_std(
            sample,
            sample,
            time=torch.tensor([0.8]),
            next_time=torch.tensor([0.6]),
            noise_level=0.5,
            eps=eps,
        )
