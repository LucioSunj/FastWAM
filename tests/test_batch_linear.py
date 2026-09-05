import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from fastwam.adapters import PolicyRegime, RegimeContext, RegimeLoRALinear
from fastwam.models.wan22.adaptive_action import CachedActionVelocity
from fastwam.models.wan22.batch_linear import (
    BatchInvariantLinear,
    install_batch_invariant_linears,
)


class _Expert(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.time_embedding = nn.Sequential(nn.Linear(width, width), nn.SiLU())
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(width, width))
        self.text_embedding = nn.Sequential(nn.Linear(width, width), nn.GELU())
        self.projection = nn.Linear(width, width)


class _Actor(nn.Module):
    def __init__(self, width: int = 5) -> None:
        super().__init__()
        self.video_expert = _Expert(width)
        self.action_expert = _Expert(width)


def test_install_preserves_rng_parameters_keys_and_default_forward() -> None:
    torch.manual_seed(7)
    actor = _Actor()
    before_parameters = dict(actor.named_parameters())
    before_keys = tuple(actor.state_dict())
    inputs = torch.randn(3, 4, 5)
    expected = actor.video_expert.projection(inputs)
    before_rng = torch.random.get_rng_state().clone()

    context = install_batch_invariant_linears(actor)

    assert torch.equal(torch.random.get_rng_state(), before_rng)
    assert tuple(actor.state_dict()) == before_keys
    assert all(
        dict(actor.named_parameters())[name] is parameter
        for name, parameter in before_parameters.items()
    )
    assert isinstance(actor.video_expert.projection, BatchInvariantLinear)
    assert torch.equal(actor.video_expert.projection(inputs), expected)

    serial = torch.cat(
        [actor.video_expert.projection(inputs[index : index + 1]) for index in range(3)]
    )
    with context.use(3):
        batched = actor.video_expert.projection(inputs)
    assert torch.equal(batched, serial)


def test_flattened_video_time_restores_sample_batch() -> None:
    actor = _Actor()
    context = install_batch_invariant_linears(actor)
    flattened = torch.randn(12, 5)
    module = actor.video_expert.time_embedding[0]
    serial = torch.cat([module(row) for row in flattened.reshape(3, 4, 5)])

    with context.use(3):
        batched = module(flattened)

    assert torch.equal(batched, serial.reshape(12, 5))


def test_regime_lora_rowwise_base_and_delta_backward() -> None:
    actor = _Actor()
    regime = RegimeContext()
    adapted = RegimeLoRALinear(
        actor.action_expert.projection,
        regime_context=regime,
        rank=2,
        alpha=2,
        dropout=0.0,
    )
    actor.action_expert.projection = adapted
    serial_actor = copy.deepcopy(adapted)
    context = install_batch_invariant_linears(actor)
    inputs = torch.randn(3, 4, 5, requires_grad=True)
    serial_inputs = inputs.detach().clone().requires_grad_(True)
    with serial_actor.regime_context.use(PolicyRegime.UNCOND):
        serial = torch.cat(
            [serial_actor(serial_inputs[index : index + 1]) for index in range(3)]
        )
    with regime.use(PolicyRegime.UNCOND), context.use(3):
        batched = actor.action_expert.projection(inputs)

    assert torch.equal(batched, serial)
    batched.float().sum().backward()
    serial.float().sum().backward()
    assert torch.equal(inputs.grad, serial_inputs.grad)
    assert all(
        torch.equal(left.grad, right.grad)
        for left, right in zip(
            actor.action_expert.projection.parameters(),
            serial_actor.parameters(),
            strict=True,
        )
    )


def test_nonzero_lora_dropout_is_outside_supported_contract() -> None:
    actor = _Actor()
    actor.action_expert.projection = RegimeLoRALinear(
        actor.action_expert.projection,
        regime_context=RegimeContext(),
        rank=2,
        alpha=2,
        dropout=0.1,
    )
    with pytest.raises(ValueError, match="requires dropout=0"):
        install_batch_invariant_linears(actor)


def test_checkpoint_backward_reenters_batch_linear_context() -> None:
    actor = _Actor()
    context = install_batch_invariant_linears(actor)
    module = actor.action_expert.projection
    velocity = CachedActionVelocity.__new__(CachedActionVelocity)
    velocity.regime_context = None
    velocity.batch_linear_context = context
    velocity.condition = SimpleNamespace(context=torch.empty(3, 1, 1))
    inputs = torch.randn(3, 4, 5, requires_grad=True)
    calls = []

    def function(value: torch.Tensor) -> torch.Tensor:
        calls.append(context.batch_size)
        return torch.sigmoid(module(value)).square()

    output = checkpoint(
        function,
        inputs,
        use_reentrant=False,
        context_fn=velocity._checkpoint_regime_contexts,
    )
    output.sum().backward()

    assert calls == [3, 3]
    assert inputs.grad is not None
