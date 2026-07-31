from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from fastwam.adapters import (
    ActionLoRATargetGroup,
    PolicyRegime,
    RegimeContext,
    RegimeLoRAConfig,
    inject_action_dit_lora,
)
from fastwam.models.wan22.adaptive_action import (
    CachedActionCondition,
    CachedActionVelocity,
)
from fastwam.models.wan22.adaptive_sampler import (
    replay_action_flow_sde_log_prob,
    sample_action_flow_sde,
)
from fastwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)


class _ActionExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 3, bias=False)

    def pre_dit(self, *, action_tokens, timestep, context, context_mask):
        del timestep
        return {
            "tokens": action_tokens,
            "freqs": torch.empty(0),
            "t_mod": torch.empty(0),
            "context": context,
            "context_mask": context_mask[:, None, :].expand(
                -1, action_tokens.shape[1], -1
            ),
        }

    def post_dit(self, tokens, _pre):
        return self.projection(tokens)


class _MoT(nn.Module):
    def forward_action_with_video_cache(self, *, action_tokens, kv_tap, **_kwargs):
        assert kv_tap is None
        return action_tokens


class _Attention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q = nn.Linear(3, 3, bias=False)
        self.k = nn.Linear(3, 3, bias=False)
        self.v = nn.Linear(3, 3, bias=False)
        self.o = nn.Linear(3, 3, bias=False)


class _AttentionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()


class _LoRAActionExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_AttentionBlock()])

    def pre_dit(self, *, action_tokens, timestep, context, context_mask):
        del timestep
        attention = self.blocks[0].self_attn
        mixed = attention.q(action_tokens) + attention.k(action_tokens)
        mixed = mixed + attention.v(action_tokens)
        return {
            "tokens": attention.o(mixed),
            "freqs": torch.empty(0),
            "t_mod": torch.empty(0),
            "context": context,
            "context_mask": context_mask[:, None, :].expand(
                -1, action_tokens.shape[1], -1
            ),
        }

    def post_dit(self, tokens, _pre):
        return tokens


def _condition() -> CachedActionCondition:
    return CachedActionCondition(
        context=torch.randn(2, 4, 5),
        context_mask=torch.ones(2, 4, dtype=torch.bool),
        video_kv_cache=[{"k": torch.randn(2, 1, 3), "v": torch.randn(2, 1, 3)}],
        attention_mask=torch.ones(3, 3, dtype=torch.bool),
        video_seq_len=1,
        current_frame_video_tokens=1,
    )


def test_cached_action_velocity_keeps_gradient_and_restores_regime():
    action_expert = _LoRAActionExpert()
    adapter = inject_action_dit_lora(
        action_expert,
        RegimeLoRAConfig(
            rank=2,
            alpha=2.0,
            target_groups=(ActionLoRATargetGroup.SELF_ATTENTION_QKVO,),
        ),
    )
    velocity = CachedActionVelocity(
        action_expert=action_expert,
        mot=_MoT(),
        condition=_condition(),
        regime=PolicyRegime.UNCOND,
        regime_context=adapter.regime_context,
    )
    action = torch.randn(2, 2, 3, requires_grad=True)

    output = velocity(action, torch.tensor([900.0, 800.0]))
    output.velocity.sum().backward()

    assert action.grad is not None
    assert any(parameter.grad is not None for parameter in adapter.lora_parameters())
    assert adapter.regime_context.current is PolicyRegime.IDM
    assert output.gate_tap is None


def test_uncond_velocity_requires_the_adapter_regime_context():
    with pytest.raises(ValueError, match="requires the injected LoRA"):
        CachedActionVelocity(
            action_expert=_ActionExpert(),
            mot=_MoT(),
            condition=_condition(),
            regime=PolicyRegime.UNCOND,
        )


def test_uncond_velocity_rejects_an_action_expert_without_lora():
    with pytest.raises(ValueError, match="at least one injected"):
        CachedActionVelocity(
            action_expert=_ActionExpert(),
            mot=_MoT(),
            condition=_condition(),
            regime=PolicyRegime.UNCOND,
            regime_context=RegimeContext(),
        )


def test_velocity_rejects_a_different_adapter_regime_context():
    action_expert = _LoRAActionExpert()
    adapter = inject_action_dit_lora(
        action_expert,
        RegimeLoRAConfig(
            rank=2,
            alpha=2.0,
            target_groups=(ActionLoRATargetGroup.SELF_ATTENTION_QKVO,),
        ),
    )

    with pytest.raises(ValueError, match="does not own every injected"):
        CachedActionVelocity(
            action_expert=action_expert,
            mot=_MoT(),
            condition=_condition(),
            regime=PolicyRegime.UNCOND,
            regime_context=RegimeContext(),
        )

    assert adapter.regime_context.current is PolicyRegime.IDM


def test_cached_action_condition_rejects_future_only_current_frame_count():
    try:
        CachedActionCondition(
            context=torch.randn(1, 1, 2),
            context_mask=torch.ones(1, 1, dtype=torch.bool),
            video_kv_cache=[],
            attention_mask=torch.ones(2, 2, dtype=torch.bool),
            video_seq_len=1,
            current_frame_video_tokens=2,
        )
    except ValueError as exc:
        assert "current_frame_video_tokens" in str(exc)
    else:
        raise AssertionError("Expected invalid current-frame count to fail.")


def test_idm_is_exact_and_uncond_flow_replay_is_consistent_with_lora():
    torch.manual_seed(23)
    action_expert = _LoRAActionExpert()
    baseline_expert = copy.deepcopy(action_expert)
    adapter = inject_action_dit_lora(
        action_expert,
        RegimeLoRAConfig(
            rank=2,
            alpha=2.0,
            target_groups=(ActionLoRATargetGroup.SELF_ATTENTION_QKVO,),
        ),
    )
    with torch.no_grad():
        for _, layer in adapter.iter_adapted_linears():
            layer.lora_B.fill_(0.1)

    baseline_velocity = CachedActionVelocity(
        action_expert=baseline_expert,
        mot=_MoT(),
        condition=_condition(),
        regime=PolicyRegime.IDM,
    )
    idm_velocity = CachedActionVelocity(
        action_expert=action_expert,
        mot=_MoT(),
        condition=_condition(),
        regime=PolicyRegime.IDM,
        regime_context=adapter.regime_context,
    )
    uncond_velocity = CachedActionVelocity(
        action_expert=action_expert,
        mot=_MoT(),
        condition=_condition(),
        regime=PolicyRegime.UNCOND,
        regime_context=adapter.regime_context,
    )
    action = torch.randn(2, 2, 3)
    timestep = torch.tensor([900.0, 700.0])

    baseline = baseline_velocity(action, timestep).velocity
    idm_first = idm_velocity(action, timestep).velocity
    idm_second = idm_velocity(action, timestep).velocity
    uncond_first = uncond_velocity(action, timestep).velocity
    uncond_second = uncond_velocity(action, timestep).velocity

    assert torch.equal(idm_first, baseline)
    assert torch.equal(idm_second, baseline)
    assert torch.equal(uncond_first, uncond_second)
    assert not torch.equal(uncond_first, baseline)
    assert adapter.regime_context.current is PolicyRegime.IDM

    scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
    timesteps, deltas = scheduler.build_inference_schedule(
        num_inference_steps=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    rollout = sample_action_flow_sde(
        torch.randn(2, 2, 3),
        velocity_fn=uncond_velocity,
        timesteps=timesteps,
        scheduler_deltas=deltas,
        num_train_timesteps=1000,
        noise_level=0.5,
        denoise_indices=torch.tensor([0, 2]),
        generator=torch.Generator().manual_seed(17),
    )
    replayed, _ = replay_action_flow_sde_log_prob(
        rollout.chains.detach(),
        rollout.denoise_indices,
        velocity_fn=uncond_velocity,
        timesteps=timesteps,
        scheduler_deltas=deltas,
        num_train_timesteps=1000,
        noise_level=0.5,
    )
    torch.testing.assert_close(replayed, rollout.old_log_probs)
