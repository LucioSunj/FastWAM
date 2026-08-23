"""Conditional same-chunk routing RL, isolated from the formal delayed PPO path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

ROUTING_RL_CHECKPOINT_SCHEMA = "causal-routing-rl-v1"


@dataclass(frozen=True)
class RoutingRLConfigV1:
    """Frozen conditional routing-RL optimization contract."""

    policy_learning_rate: float = 3e-5
    value_learning_rate: float = 1e-4
    clip_ratio: float = 0.2
    gamma: float = 1.0
    gae_lambda: float = 0.95
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    gradient_clip: float = 1.0
    episodes_per_update: int = 64
    ppo_epochs: int = 4
    chunk_minibatch: int = 256
    maximum_updates: int = 1000
    validation_interval: int = 50
    early_stopping_evaluations: int = 4
    training_budgets_percent: tuple[int, ...] = (25, 50, 75)

    def __post_init__(self) -> None:
        if asdict(self) != asdict(RoutingRLConfigV1.__new_with_defaults()):
            raise ValueError("Conditional routing-RL hyperparameters are frozen.")

    @classmethod
    def __new_with_defaults(cls) -> RoutingRLConfigV1:
        """Construct defaults without recursively invoking validation."""

        value = object.__new__(cls)
        defaults = {
            field: definition.default
            for field, definition in cls.__dataclass_fields__.items()
        }
        for name, item in defaults.items():
            object.__setattr__(value, name, item)
        return value


class ResidualBudgetRouterV1(nn.Module):
    """Zero-initialized residual over a frozen supervised utility vector."""

    def __init__(self, *, mode_count: int, state_dim: int) -> None:
        super().__init__()
        if mode_count not in {2, 3} or state_dim < mode_count:
            raise ValueError(
                "Routing RL supports two or three experts and valid state."
            )
        self.mode_count = int(mode_count)
        self.state_dim = int(state_dim)
        self.residual = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, mode_count),
        )
        self.value_head = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self,
        state: torch.Tensor,
        *,
        supervised_utility: torch.Tensor,
        affordable_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked same-chunk logits and a small routing value estimate."""

        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError("Routing RL state shape changed.")
        expected = (state.shape[0], self.mode_count)
        if supervised_utility.shape != expected or affordable_mask.shape != expected:
            raise ValueError("Routing utility/mask shape changed.")
        if (
            affordable_mask.dtype is not torch.bool
            or not affordable_mask.any(dim=1).all()
        ):
            raise ValueError("Every routing state requires an affordable expert.")
        logits = supervised_utility.detach() + self.residual(state)
        logits = logits.masked_fill(~affordable_mask, -torch.inf)
        return logits, self.value_head(state).squeeze(-1)

    def distribution(
        self,
        state: torch.Tensor,
        *,
        supervised_utility: torch.Tensor,
        affordable_mask: torch.Tensor,
    ) -> tuple[Categorical, torch.Tensor]:
        """Build the categorical policy after hard-budget masking."""

        logits, value = self(
            state,
            supervised_utility=supervised_utility,
            affordable_mask=affordable_mask,
        )
        return Categorical(logits=logits), value


def routing_ppo_loss_v1(
    *,
    new_log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantage: torch.Tensor,
    new_value: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    config: RoutingRLConfigV1 | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the conditional router's clipped PPO and value objectives."""

    cfg = config or RoutingRLConfigV1()
    tensors = (new_log_prob, old_log_prob, advantage, new_value, returns, entropy)
    if any(value.shape != new_log_prob.shape for value in tensors[1:]):
        raise ValueError("Routing PPO tensors must share shape.")
    ratio = torch.exp(new_log_prob - old_log_prob)
    clipped = ratio.clamp(1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)
    policy_loss = -torch.minimum(ratio * advantage, clipped * advantage).mean()
    value_loss = 0.5 * (new_value - returns).square().mean()
    entropy_bonus = entropy.mean()
    total = (
        policy_loss
        + cfg.value_coefficient * value_loss
        - cfg.entropy_coefficient * entropy_bonus
    )
    return {
        "loss": total,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy_bonus,
    }


def build_routing_rl_checkpoint_v1(
    *,
    router: ResidualBudgetRouterV1,
    optimizer_state: dict[str, Any],
    scheduler_state: dict[str, Any],
    training_step: int,
    fold: int,
    config: RoutingRLConfigV1 | None = None,
) -> dict[str, Any]:
    """Save only the residual router/value trainables and trainer state."""

    if training_step < 0 or fold not in range(5):
        raise ValueError("Routing RL checkpoint counters are invalid.")
    state = {
        name: value.detach().cpu().clone()
        for name, value in router.state_dict().items()
    }
    return {
        "schema": ROUTING_RL_CHECKPOINT_SCHEMA,
        "fold": fold,
        "training_step": training_step,
        "model_state_dict": state,
        "optimizer": optimizer_state,
        "scheduler": scheduler_state,
        "config": asdict(config or RoutingRLConfigV1()),
        "experts": "FROZEN-EXTERNAL",
        "supervised_outcome_heads": "FROZEN-EXTERNAL",
    }
