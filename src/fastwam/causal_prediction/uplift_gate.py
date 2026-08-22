"""Detached native-feature uplift Gate for same-chunk compute routing."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .contracts import (
    UPLIFT_GATE_CHECKPOINT_SCHEMA,
    UPLIFT_GATE_CHECKPOINT_SCHEMA_V2,
    CausalComputeMode,
    UpliftGateOutput,
)


@dataclass(frozen=True)
class UpliftGateConfig:
    """Static dimensions for the four-block causal sidecar."""

    current_kv_dim: int
    language_dim: int
    history_dim: int
    proposal_dim: int
    modes: tuple[CausalComputeMode, ...] = (
        CausalComputeMode.C0_CURRENT,
        CausalComputeMode.C2_FULL,
    )
    proprio_dim: int = 8
    hidden_size: int = 256
    query_tokens: int = 4
    blocks: int = 4
    attention_heads: int = 8
    current_kv_layers: int = 30

    def __post_init__(self) -> None:
        modes = tuple(CausalComputeMode.parse(mode) for mode in self.modes)
        if any(not mode.is_routable for mode in modes) or len(set(modes)) != len(modes):
            raise ValueError("Uplift Gate modes must be unique routable modes.")
        if modes[0] is not CausalComputeMode.C0_CURRENT:
            raise ValueError("The fastest C0 mode must be the first Gate output.")
        object.__setattr__(self, "modes", modes)
        dimensions = (
            self.current_kv_dim,
            self.language_dim,
            self.history_dim,
            self.proposal_dim,
            self.proprio_dim,
            self.hidden_size,
            self.query_tokens,
            self.blocks,
            self.attention_heads,
            self.current_kv_layers,
        )
        if any(value < 1 for value in dimensions):
            raise ValueError("All Uplift Gate dimensions must be positive.")
        if self.hidden_size % self.attention_heads:
            raise ValueError("Gate hidden size must be divisible by attention heads.")
        if self.query_tokens != 4 or self.blocks != 4 or self.hidden_size != 256:
            raise ValueError(
                "The v1 causal Gate is frozen to hidden=256, queries=4, blocks=4."
            )
        if self.current_kv_layers != 30:
            raise ValueError(
                "The v1 causal Gate reads all 30 current-frame K/V layers."
            )


@dataclass(frozen=True)
class UpliftGateInputs:
    """Only pre-prediction signals available before the current route decision."""

    current_video_kv: torch.Tensor
    current_video_mask: torch.Tensor
    language: torch.Tensor
    language_mask: torch.Tensor
    proprio: torch.Tensor
    history: torch.Tensor
    history_mask: torch.Tensor
    action_proposal: torch.Tensor
    remaining_budget: torch.Tensor
    previous_mode: torch.Tensor
    steps_to_go: torch.Tensor
    current_video_layer_count: int = 30

    def validate(self, config: UpliftGateConfig) -> None:
        """Validate shapes without accepting any post-prediction tensor bag."""

        batch = self.current_video_kv.shape[0]
        if self.current_video_layer_count != config.current_kv_layers:
            raise ValueError("Gate current-frame K/V layer coverage is incomplete.")
        expected_last = {
            "current_video_kv": config.current_kv_dim,
            "language": config.language_dim,
            "proprio": config.proprio_dim,
            "history": config.history_dim,
            "action_proposal": config.proposal_dim,
        }
        for name, width in expected_last.items():
            value = getattr(self, name)
            if value.shape[0] != batch or value.shape[-1] != width:
                raise ValueError(
                    f"Gate input {name} has shape {tuple(value.shape)}; "
                    f"expected batch {batch}, last dim {width}."
                )
        if self.current_video_kv.ndim != 3 or self.language.ndim != 3:
            raise ValueError("Current K/V and language inputs must be token sequences.")
        if self.history.ndim != 3 or self.history.shape[1] != 4:
            raise ValueError("Gate history must have exactly four chunk slots.")
        mask_contracts = (
            (self.current_video_mask, self.current_video_kv.shape[:2]),
            (self.language_mask, self.language.shape[:2]),
            (self.history_mask, self.history.shape[:2]),
        )
        for mask, shape in mask_contracts:
            if mask.dtype is not torch.bool or mask.shape != shape:
                raise ValueError(f"Gate mask must be bool with shape {tuple(shape)}.")
        mode_width = len(config.modes) + 1
        if self.previous_mode.shape != (batch, mode_width):
            raise ValueError(
                "Previous-mode input must include a no-previous-route slot."
            )
        for name in ("remaining_budget", "steps_to_go"):
            if getattr(self, name).shape != (batch, 1):
                raise ValueError(f"Gate input {name} must have shape [batch, 1].")


class CausalUpliftGate(nn.Module):
    """Four-block sidecar producing per-mode success and uncertainty."""

    def __init__(self, config: UpliftGateConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.query = nn.Parameter(torch.empty(config.query_tokens, hidden))
        nn.init.normal_(self.query, std=0.02)
        self.current_projection = nn.Linear(config.current_kv_dim, hidden)
        self.language_projection = nn.Linear(config.language_dim, hidden)
        self.history_projection = nn.Linear(config.history_dim, hidden)
        scalar_width = (
            config.proprio_dim + config.proposal_dim + len(config.modes) + 1 + 2
        )
        self.scalar_projection = nn.Linear(scalar_width, hidden)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=config.attention_heads,
            dim_feedforward=hidden * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=config.blocks)
        self.norm = nn.LayerNorm(hidden)
        self.outcome_head = nn.Linear(hidden, len(config.modes))
        self.log_variance_head = nn.Linear(hidden, len(config.modes))

    def forward(
        self,
        inputs: UpliftGateInputs,
        *,
        normalized_cost: torch.Tensor,
    ) -> UpliftGateOutput:
        """Predict outcomes while detaching every policy-main-path input."""

        inputs.validate(self.config)
        batch = inputs.current_video_kv.shape[0]
        if normalized_cost.shape not in {
            (len(self.config.modes),),
            (batch, len(self.config.modes)),
        }:
            raise ValueError("Normalized mode cost has the wrong shape.")
        query = self.query.unsqueeze(0).expand(batch, -1, -1)
        current = self.current_projection(inputs.current_video_kv.detach())
        language = self.language_projection(inputs.language.detach())
        history = self.history_projection(inputs.history.detach())
        scalar = torch.cat(
            [
                inputs.proprio.detach(),
                inputs.action_proposal.detach(),
                inputs.remaining_budget.detach(),
                inputs.previous_mode.detach(),
                inputs.steps_to_go.detach(),
            ],
            dim=-1,
        )
        scalar = self.scalar_projection(scalar).unsqueeze(1)
        tokens = torch.cat([query, current, language, history, scalar], dim=1)
        valid = torch.cat(
            [
                torch.ones(
                    batch,
                    self.config.query_tokens,
                    dtype=torch.bool,
                    device=tokens.device,
                ),
                inputs.current_video_mask.detach(),
                inputs.language_mask.detach(),
                inputs.history_mask.detach(),
                torch.ones(batch, 1, dtype=torch.bool, device=tokens.device),
            ],
            dim=1,
        )
        encoded = self.blocks(tokens, src_key_padding_mask=~valid)
        pooled = self.norm(encoded[:, : self.config.query_tokens]).mean(dim=1)
        q_values = torch.sigmoid(self.outcome_head(pooled))
        log_variance = self.log_variance_head(pooled).clamp(-10.0, 5.0)
        uncertainty = torch.exp(0.5 * log_variance)
        costs = normalized_cost.to(device=q_values.device, dtype=q_values.dtype)
        if costs.ndim == 1:
            costs = costs.unsqueeze(0).expand(batch, -1)
        uplift = q_values - q_values[:, :1]
        return UpliftGateOutput(
            modes=self.config.modes,
            q_values=q_values,
            uplift=uplift,
            uncertainty=uncertainty,
            normalized_cost=costs,
        )


def causal_uplift_gate_loss(
    output: UpliftGateOutput,
    *,
    empirical_outcomes: torch.Tensor,
    inclusion_weights: torch.Tensor | None = None,
    huber_delta: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Return the preregistered supervised uplift loss and components."""

    if empirical_outcomes.shape != output.q_values.shape:
        raise ValueError("Empirical outcomes must align with Gate mode heads.")
    if huber_delta != 0.5:
        raise ValueError("The v1 uplift Huber delta is frozen to 0.5.")
    target = empirical_outcomes.to(dtype=output.q_values.dtype)
    if not bool(((target >= 0) & (target <= 1)).all()):
        raise ValueError("Empirical outcome means must lie in [0, 1].")
    if inclusion_weights is None:
        weights = torch.ones_like(target)
    else:
        weights = inclusion_weights.to(device=target.device, dtype=target.dtype)
        if weights.shape == target.shape[:1]:
            weights = weights[:, None].expand_as(target)
    if weights.shape != target.shape or bool((weights <= 0).any()):
        raise ValueError(
            "Inclusion weights must be positive per-state or mode-aligned."
        )
    bce = (
        F.binary_cross_entropy(output.q_values, target, reduction="none") * weights
    ).sum() / weights.sum()
    target_uplift = target - target[:, :1]
    huber = F.huber_loss(
        output.uplift,
        target_uplift,
        delta=huber_delta,
        reduction="none",
    )
    uplift_huber = (huber * weights).sum() / weights.sum()

    treatment_prediction = output.uplift[:, 1:]
    treatment_target = target_uplift[:, 1:]
    pair_gap = treatment_target[:, None, :] - treatment_target[None, :, :]
    pair_mask = pair_gap.abs() >= 0.5
    if bool(pair_mask.any()):
        prediction_gap = (
            treatment_prediction[:, None, :] - treatment_prediction[None, :, :]
        )
        ranking = F.softplus(-prediction_gap * pair_gap.sign())[pair_mask].mean()
    else:
        ranking = treatment_prediction.sum() * 0.0
    variance = output.uncertainty.square().clamp_min(1e-6)
    uncertainty_nll = (
        0.5
        * ((target - output.q_values).square() / variance + variance.log())
        * weights
    ).sum() / weights.sum()
    brier = (((output.q_values - target).square()) * weights).sum() / weights.sum()
    total = (
        bce + 0.5 * uplift_huber + 0.2 * ranking + 0.1 * uncertainty_nll + 0.1 * brier
    )
    return {
        "loss": total,
        "bce": bce,
        "uplift_huber": uplift_huber,
        "ranking": ranking,
        "uncertainty_nll": uncertainty_nll,
        "brier": brier,
    }


def apply_temperature_calibration(
    output: UpliftGateOutput,
    temperatures: torch.Tensor,
) -> UpliftGateOutput:
    """Apply validation-only per-head temperatures before policy routing."""

    if temperatures.shape != (len(output.modes),):
        raise ValueError("Gate calibration must provide one temperature per mode.")
    temperature = temperatures.to(
        device=output.q_values.device,
        dtype=output.q_values.dtype,
    )
    if not torch.isfinite(temperature).all() or bool((temperature <= 0).any()):
        raise ValueError("Gate calibration temperatures must be finite and positive.")
    logits = torch.logit(output.q_values.clamp(1e-6, 1 - 1e-6))
    q_values = torch.sigmoid(logits / temperature[None, :])
    return UpliftGateOutput(
        modes=output.modes,
        q_values=q_values,
        uplift=q_values - q_values[:, :1],
        uncertainty=output.uncertainty,
        normalized_cost=output.normalized_cost,
    )


def select_budgeted_mode(
    output: UpliftGateOutput,
    *,
    remaining_budget: torch.Tensor,
    beta: float,
    cost_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose the current chunk's best affordable mode and debit its cost."""

    if remaining_budget.shape != (output.q_values.shape[0],):
        raise ValueError("Remaining budget must have shape [batch].")
    utilities = output.utilities(beta=beta, cost_weight=cost_weight)
    affordable = output.normalized_cost <= remaining_budget[:, None]
    if not bool(affordable[:, 0].all()) or not bool(
        output.normalized_cost[:, 0].eq(0).all()
    ):
        raise ValueError("The fastest C0 mode must remain affordable at zero cost.")
    utilities = utilities.masked_fill(~affordable, -math.inf)
    selected = utilities.argmax(dim=-1)
    spent = output.normalized_cost.gather(1, selected[:, None]).squeeze(1)
    return selected, remaining_budget - spent


def save_uplift_gate_checkpoint(
    path: str | os.PathLike[str],
    *,
    gate: CausalUpliftGate,
    calibration: Mapping[str, Any],
    training_state: Mapping[str, Any],
) -> None:
    """Save only the sidecar Gate, calibration, and its trainer state."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": UPLIFT_GATE_CHECKPOINT_SCHEMA,
        "metadata": {
            "config": {
                **asdict(gate.config),
                "modes": [mode.value for mode in gate.config.modes],
            },
            "calibration": dict(calibration),
        },
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in gate.state_dict().items()
        },
        "training_state": dict(training_state),
    }
    inspect_uplift_gate_checkpoint_payload(payload, gate=gate)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def inspect_uplift_gate_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    gate: CausalUpliftGate | None = None,
) -> dict[str, Any]:
    """Validate that a Gate checkpoint contains only its sidecar state."""

    if set(payload) != {"schema", "metadata", "state_dict", "training_state"}:
        raise ValueError("Uplift Gate checkpoint top-level keys changed.")
    if payload["schema"] != UPLIFT_GATE_CHECKPOINT_SCHEMA:
        raise ValueError(f"Unsupported Uplift Gate schema {payload['schema']!r}.")
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping) or set(metadata) != {"config", "calibration"}:
        raise ValueError("Uplift Gate checkpoint metadata keys changed.")
    config_payload = dict(metadata["config"])
    config_payload["modes"] = tuple(config_payload["modes"])
    config = UpliftGateConfig(**config_payload)
    state = payload["state_dict"]
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Uplift Gate checkpoint state must be non-empty.")
    if any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Uplift Gate model state contains a non-tensor value.")
    if gate is not None and (
        gate.config != config or set(state) != set(gate.state_dict())
    ):
        raise ValueError("Uplift Gate checkpoint does not match the live sidecar.")
    return {
        "schema": payload["schema"],
        "modes": [mode.value for mode in config.modes],
        "state_tensor_count": len(state),
    }


def load_uplift_gate_checkpoint(
    path: str | os.PathLike[str],
    *,
    gate: CausalUpliftGate,
) -> dict[str, Any]:
    """Load a schema-checked Gate sidecar without touching policy parameters."""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("Uplift Gate checkpoint payload must be a mapping.")
    inspect_uplift_gate_checkpoint_payload(payload, gate=gate)
    gate.load_state_dict(payload["state_dict"], strict=True)
    return dict(payload)


def build_uplift_gate_checkpoint_v2(
    *,
    gate: CausalUpliftGate,
    feature_specification: Mapping[str, Any],
    fold: int,
    proposal_variant: str,
    calibration: Mapping[str, Any],
    measured_cost_ms: Mapping[str, float],
    training_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the v2 Gate payload with fold, features, and measured costs."""

    if fold not in range(5):
        raise ValueError("Gate v2 fold must lie in [0, 4].")
    if proposal_variant not in {"one_proposal", "two_proposal"}:
        raise ValueError("Gate v2 proposal variant is invalid.")
    allowed_features = {
        "current_video_kv",
        "language",
        "proprio",
        "history",
        "action_proposal",
        "proposal_disagreement",
        "remaining_budget",
        "previous_mode",
        "steps_to_go",
    }
    if not feature_specification or not set(feature_specification) <= allowed_features:
        raise ValueError("Gate v2 feature specification changed from the whitelist.")
    expected_cost_modes = {mode.value for mode in gate.config.modes}
    if set(measured_cost_ms) != expected_cost_modes or any(
        not math.isfinite(float(value)) or float(value) < 0
        for value in measured_cost_ms.values()
    ):
        raise ValueError("Gate v2 measured costs must cover every mode.")
    payload = {
        "schema": UPLIFT_GATE_CHECKPOINT_SCHEMA_V2,
        "metadata": {
            "config": {
                **asdict(gate.config),
                "modes": [mode.value for mode in gate.config.modes],
            },
            "feature_specification": dict(feature_specification),
            "fold": int(fold),
            "proposal_variant": proposal_variant,
            "calibration": dict(calibration),
            "measured_cost_ms": {
                str(key): float(value) for key, value in measured_cost_ms.items()
            },
        },
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in gate.state_dict().items()
        },
        "training_state": dict(training_state),
    }
    inspect_uplift_gate_checkpoint_payload_v2(payload, gate=gate)
    return payload


def inspect_uplift_gate_checkpoint_payload_v2(
    payload: Mapping[str, Any],
    *,
    gate: CausalUpliftGate | None = None,
) -> dict[str, Any]:
    """Validate the exact v2 Gate-only checkpoint surface."""

    if set(payload) != {"schema", "metadata", "state_dict", "training_state"}:
        raise ValueError("Uplift Gate v2 checkpoint top-level keys changed.")
    if payload["schema"] != UPLIFT_GATE_CHECKPOINT_SCHEMA_V2:
        raise ValueError(f"Unsupported Uplift Gate v2 schema {payload['schema']!r}.")
    metadata = payload["metadata"]
    expected_metadata = {
        "config",
        "feature_specification",
        "fold",
        "proposal_variant",
        "calibration",
        "measured_cost_ms",
    }
    if not isinstance(metadata, Mapping) or set(metadata) != expected_metadata:
        raise ValueError("Uplift Gate v2 metadata keys changed.")
    config_payload = dict(metadata["config"])
    config_payload["modes"] = tuple(config_payload["modes"])
    config = UpliftGateConfig(**config_payload)
    state = payload["state_dict"]
    if (
        not isinstance(state, Mapping)
        or not state
        or any(not isinstance(value, torch.Tensor) for value in state.values())
    ):
        raise ValueError("Uplift Gate v2 state must contain only tensors.")
    if gate is not None and (
        gate.config != config or set(state) != set(gate.state_dict())
    ):
        raise ValueError("Uplift Gate v2 checkpoint does not match the live sidecar.")
    # Reuse construction validation for feature/cost/fold semantics.
    if int(metadata["fold"]) not in range(5):
        raise ValueError("Uplift Gate v2 fold is invalid.")
    if metadata["proposal_variant"] not in {"one_proposal", "two_proposal"}:
        raise ValueError("Uplift Gate v2 proposal variant is invalid.")
    expected_cost_modes = {mode.value for mode in config.modes}
    costs = metadata["measured_cost_ms"]
    if set(costs) != expected_cost_modes or any(
        not math.isfinite(float(value)) or float(value) < 0 for value in costs.values()
    ):
        raise ValueError("Uplift Gate v2 measured costs are invalid.")
    return {
        "schema": payload["schema"],
        "fold": int(metadata["fold"]),
        "proposal_variant": str(metadata["proposal_variant"]),
        "modes": [mode.value for mode in config.modes],
        "state_tensor_count": len(state),
    }


def save_uplift_gate_checkpoint_v2(
    path: str | os.PathLike[str],
    **kwargs: Any,
) -> dict[str, Any]:
    """Atomically save the v2 Gate-only checkpoint."""

    payload = build_uplift_gate_checkpoint_v2(**kwargs)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return inspect_uplift_gate_checkpoint_payload_v2(
        payload,
        gate=kwargs.get("gate"),
    )


def load_uplift_gate_checkpoint_v2(
    path: str | os.PathLike[str],
    *,
    gate: CausalUpliftGate,
) -> dict[str, Any]:
    """Load a schema-checked v2 Gate without touching policy parameters."""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("Uplift Gate v2 checkpoint payload must be a mapping.")
    inspect_uplift_gate_checkpoint_payload_v2(payload, gate=gate)
    gate.load_state_dict(payload["state_dict"], strict=True)
    return dict(payload)
