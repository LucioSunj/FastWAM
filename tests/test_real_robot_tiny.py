"""The tiny fixture exercises the same LoRA/cached-velocity/BC objective."""

import torch

from fastwam.adapters import PolicyRegime
from fastwam.models.wan22.adaptive_action import CachedActionVelocity
from fastwam.real_robot_tiny import TinyFastWAM, TinyUncondBCPolicy, adapt_tiny_fixture


def test_tiny_bc_is_current_only_and_idm_stays_frozen():
    torch.manual_seed(17)
    actor = TinyFastWAM()
    adapter, report, _ = adapt_tiny_fixture(actor, steps=2)
    assert len(set(report["parent_losses"] + report["bc_losses"])) == 4
    assert adapter.config.rank == adapter.config.alpha == 16
    context, mask = actor.encode_prompt(["Move the block."])
    batch = {
        "video": torch.randn(1, 3, 9, 32, 32),
        "proprio": torch.randn(1, 32, 8),
        "action": torch.randn(1, 32, 7),
        "context": context.detach(),
        "context_mask": mask,
    }
    policy = TinyUncondBCPolicy(
        actor=actor, lora_config=adapter.config, lora_adapter=adapter
    )
    timestep, noise = torch.tensor([370.0]), torch.randn(1, 32, 7)
    first = policy(batch, timestep=timestep, noise=noise)["loss_action_bc"]
    changed = {**batch, "video": batch["video"].clone()}
    changed["video"][:, :, 1:] += 100
    second = policy(changed, timestep=timestep, noise=noise)["loss_action_bc"]
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    condition = policy.prepare_action_condition(batch)
    velocity = CachedActionVelocity(
        action_expert=actor.action_expert,
        mot=actor.mot,
        condition=condition,
        regime=PolicyRegime.IDM,
        regime_context=adapter.regime_context,
    )
    before = velocity(noise, timestep).velocity.detach().clone()
    optimizer = torch.optim.Adam(adapter.lora_parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    first.backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.lora_parameters()
    )
    assert all(
        p.grad is None
        for name, p in actor.named_parameters()
        if not name.endswith(("lora_A", "lora_B"))
    )
    optimizer.step()
    torch.testing.assert_close(
        velocity(noise, timestep).velocity, before, rtol=0, atol=0
    )
