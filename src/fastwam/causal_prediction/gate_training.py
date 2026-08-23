"""Cross-fitted supervised training and validation-only Gate selection."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .uplift_gate import (
    CausalUpliftGate,
    UpliftGateInputs,
    causal_uplift_gate_loss,
    select_budgeted_mode,
)

BETA_CANDIDATES = (0.0, 0.5, 1.0)


@dataclass(frozen=True)
class GateSupervisedBatch:
    """Detached pre-prediction features and paired empirical mode outcomes."""

    inputs: UpliftGateInputs
    empirical_outcomes: torch.Tensor
    normalized_cost: torch.Tensor
    inclusion_weights: torch.Tensor | None = None


def validate_fold_ownership(
    labeled_tasks: Mapping[str, Sequence[str]],
    *,
    train_tasks: Sequence[str],
    validation_tasks: Sequence[str],
    test_tasks: Sequence[str],
) -> None:
    """Reject test labels in train, validation, calibration, or selection roles."""

    train = set(train_tasks)
    validation = set(validation_tasks)
    test = set(test_tasks)
    if train & validation or train & test or validation & test:
        raise ValueError("Cross-fitting task partitions overlap.")
    allowed_roles = {
        "train": train,
        "validation": validation,
        "calibration": validation,
        "beta_selection": validation,
        "checkpoint_selection": validation,
        "test": test,
    }
    if set(labeled_tasks) != set(allowed_roles):
        raise ValueError("Cross-fitting label roles changed.")
    for role, expected in allowed_roles.items():
        actual = set(labeled_tasks[role])
        if actual != expected:
            raise ValueError(
                f"Cross-fitting label ownership changed for {role}: {actual} != {expected}."
            )


def _validation_metrics(
    gate: CausalUpliftGate,
    batch: GateSupervisedBatch,
    *,
    beta: float,
    cost_weight: float,
) -> tuple[float, float]:
    with torch.no_grad():
        output = gate(batch.inputs, normalized_cost=batch.normalized_cost)
        remaining = batch.inputs.remaining_budget[:, 0]
        selected, _ = select_budgeted_mode(
            output,
            remaining_budget=remaining,
            beta=beta,
            cost_weight=cost_weight,
        )
        outcomes = batch.empirical_outcomes.to(output.q_values.device)
        value = outcomes.gather(1, selected[:, None]).float().mean()
        brier = (output.q_values - outcomes).square().mean()
    return float(value.item()), float(brier.item())


def select_validation_beta(
    gate: CausalUpliftGate,
    batch: GateSupervisedBatch,
    *,
    cost_weight: float,
) -> tuple[float, float, float]:
    """Select beta on validation value, then Brier, then smaller beta."""

    candidates = []
    for beta in BETA_CANDIDATES:
        value, brier = _validation_metrics(
            gate,
            batch,
            beta=beta,
            cost_weight=cost_weight,
        )
        candidates.append((value, -brier, -beta, beta, brier))
    _, _, _, beta, brier = max(candidates)
    value, _ = _validation_metrics(
        gate,
        batch,
        beta=beta,
        cost_weight=cost_weight,
    )
    return beta, value, brier


def fit_temperature_per_head(
    probabilities: torch.Tensor,
    outcomes: torch.Tensor,
) -> torch.Tensor:
    """Fit one positive temperature per outcome head on validation tasks only."""

    if probabilities.shape != outcomes.shape or probabilities.ndim != 2:
        raise ValueError("Calibration probabilities/outcomes must be mode-aligned.")
    probabilities = probabilities.detach().float().clamp(1e-6, 1 - 1e-6)
    outcomes = outcomes.detach().float()
    logits = torch.logit(probabilities)
    temperatures = []
    for head in range(probabilities.shape[1]):
        log_temperature = torch.zeros((), requires_grad=True)
        optimizer = torch.optim.LBFGS(
            [log_temperature],
            lr=0.1,
            max_iter=50,
            line_search_fn="strong_wolfe",
        )

        def closure(
            current_optimizer=optimizer,
            current_log_temperature=log_temperature,
            current_head=head,
        ):
            current_optimizer.zero_grad()
            temperature = current_log_temperature.exp().clamp(0.05, 20.0)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits[:, current_head] / temperature,
                outcomes[:, current_head],
            )
            loss.backward()
            return loss

        optimizer.step(closure)
        temperatures.append(log_temperature.detach().exp().clamp(0.05, 20.0))
    return torch.stack(temperatures)


def train_uplift_gate(
    gate: CausalUpliftGate,
    *,
    train_batches: Sequence[GateSupervisedBatch],
    validation_batch: GateSupervisedBatch,
    cost_weight: float,
    seed: int = 42,
) -> dict[str, Any]:
    """Train with the frozen optimizer/loss and validation-value early stopping."""

    if seed != 42:
        raise ValueError("The v1 supervised Gate seed is frozen to 42.")
    if not train_batches:
        raise ValueError("Gate training requires non-empty train batches.")
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(
        gate.parameters(),
        lr=1e-4,
        weight_decay=0.01,
    )
    best_state = None
    best_epoch = None
    best_beta = None
    best_value = -float("inf")
    best_brier = float("inf")
    epochs_without_improvement = 0
    history = []
    for epoch in range(200):
        gate.train()
        total_loss = 0.0
        sample_count = 0
        for batch in train_batches:
            batch.inputs.validate(gate.config)
            batch_size = int(batch.empirical_outcomes.shape[0])
            if batch_size > 256:
                raise ValueError("Gate training batches may not exceed 256 states.")
            optimizer.zero_grad(set_to_none=True)
            output = gate(batch.inputs, normalized_cost=batch.normalized_cost)
            losses = causal_uplift_gate_loss(
                output,
                empirical_outcomes=batch.empirical_outcomes,
                inclusion_weights=batch.inclusion_weights,
                huber_delta=0.5,
            )
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError("Supervised Gate loss became non-finite.")
            losses["loss"].backward()
            optimizer.step()
            total_loss += float(losses["loss"].item()) * batch_size
            sample_count += batch_size
        gate.eval()
        beta, value, brier = select_validation_beta(
            gate,
            validation_batch,
            cost_weight=cost_weight,
        )
        improved = value > best_value or (value == best_value and brier < best_brier)
        if improved:
            best_state = copy.deepcopy(gate.state_dict())
            best_epoch = epoch
            best_beta = beta
            best_value = value
            best_brier = brier
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / sample_count,
                "validation_value": value,
                "validation_brier": brier,
                "beta": beta,
                "improved": improved,
            }
        )
        if epochs_without_improvement >= 15:
            break
    if best_state is None or best_epoch is None or best_beta is None:
        raise RuntimeError("Gate training produced no selectable checkpoint.")
    gate.load_state_dict(best_state, strict=True)
    gate.eval()
    with torch.no_grad():
        validation_output = gate(
            validation_batch.inputs,
            normalized_cost=validation_batch.normalized_cost,
        )
    temperatures = fit_temperature_per_head(
        validation_output.q_values.cpu(),
        validation_batch.empirical_outcomes.cpu(),
    )
    return {
        "schema": "causal-uplift-gate-training-report-v1",
        "status": "PASS",
        "best_epoch": best_epoch,
        "best_beta": best_beta,
        "best_validation_value": best_value,
        "best_validation_brier": best_brier,
        "temperatures": temperatures.tolist(),
        "epochs_run": len(history),
        "history": history,
    }


def select_proposal_variant(
    *,
    one_proposal_value: float,
    one_proposal_latency_ms: float,
    two_proposal_value: float,
    two_proposal_latency_ms: float,
    latency_tolerance_ms: float = 1e-6,
) -> str:
    """Choose two proposals only for at least one-point gain at matched latency."""

    if abs(one_proposal_latency_ms - two_proposal_latency_ms) > latency_tolerance_ms:
        raise ValueError(
            "Proposal variants must be compared at matched actual latency."
        )
    return (
        "two_proposal"
        if two_proposal_value - one_proposal_value >= 0.01
        else "one_proposal"
    )
