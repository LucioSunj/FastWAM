"""Pre-treatment native-feature and paired-label construction for the v2 Gate."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .contracts import (
    CausalComputeMode,
    CausalGateFeatureRecordV1,
    GateTrainingExampleV1,
    ModeOutcomeSummaryV1,
    PairedInterventionRecordV2,
)


def pool_current_kv_layers(
    layer_caches: Sequence[Mapping[str, torch.Tensor]],
    *,
    current_token_count: int,
) -> torch.Tensor:
    """Mean-pool current-token K/V into one ordered token per Wan layer."""

    if len(layer_caches) != 30:
        raise ValueError("Gate feature extraction requires all 30 Wan layers.")
    if current_token_count < 1:
        raise ValueError("Current-token count must be positive.")
    tokens = []
    batch_size = None
    width = None
    for index, cache in enumerate(layer_caches):
        if set(cache) < {"k", "v"}:
            raise ValueError(f"Layer {index} lacks K/V tensors.")
        key, value = cache["k"], cache["v"]
        if (
            key.ndim != 3
            or value.ndim != 3
            or key.shape != value.shape
            or key.shape[1] < current_token_count
        ):
            raise ValueError(f"Layer {index} K/V shape is incompatible with pooling.")
        if batch_size is None:
            batch_size, width = int(key.shape[0]), int(key.shape[2])
        elif (int(key.shape[0]), int(key.shape[2])) != (batch_size, width):
            raise ValueError("Gate K/V layer widths or batch sizes differ.")
        pooled = torch.cat(
            (
                key[:, :current_token_count].detach().mean(dim=1),
                value[:, :current_token_count].detach().mean(dim=1),
            ),
            dim=-1,
        )
        tokens.append(pooled)
    return torch.stack(tokens, dim=1)


def validate_pre_prediction_feature_payload(payload: Mapping[str, Any]) -> None:
    """Accept only the frozen online feature surface."""

    allowed = {
        "current_video_kv",
        "current_video_mask",
        "language",
        "language_mask",
        "proprio",
        "history",
        "history_mask",
        "action_proposal",
        "proposal_disagreement",
        "remaining_budget",
        "previous_mode",
        "steps_to_go",
    }
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise ValueError(
            f"Gate feature payload contains unsupported fields: {unexpected}."
        )
    required = allowed - {"proposal_disagreement"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Gate feature payload is missing: {missing}.")
    if any(
        isinstance(value, torch.Tensor) and value.requires_grad
        for value in payload.values()
    ):
        raise ValueError("Stored Gate features must be detached.")


def build_proposal_gate_feature(
    action_proposal: torch.Tensor,
    proposal_disagreement: torch.Tensor | None,
    *,
    proposal_variant: str,
) -> torch.Tensor:
    """Append disagreement only for the separately trained two-proposal Gate."""

    if action_proposal.ndim != 2 or not torch.isfinite(action_proposal).all():
        raise ValueError("Gate action proposal must be finite with shape [B,D].")
    if proposal_variant == "one_proposal":
        return action_proposal.detach()
    if proposal_variant != "two_proposal":
        raise ValueError("Unknown Gate proposal variant.")
    if (
        proposal_disagreement is None
        or proposal_disagreement.shape != (action_proposal.shape[0], 1)
        or not torch.isfinite(proposal_disagreement).all()
    ):
        raise ValueError("Two-proposal Gate requires finite disagreement [B,1].")
    return torch.cat(
        (action_proposal.detach(), proposal_disagreement.detach()),
        dim=-1,
    )


def _sample_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    tensor = torch.tensor(values, dtype=torch.float64)
    return float(tensor.var(unbiased=True).item())


def build_gate_training_example(
    *,
    feature: CausalGateFeatureRecordV1,
    records: Sequence[PairedInterventionRecordV2],
    modes: Sequence[CausalComputeMode | str],
    fold: int,
    split: str,
    treatment_chunks: int = 1,
    continuation_mode: CausalComputeMode | str = CausalComputeMode.C2_FULL,
) -> GateTrainingExampleV1:
    """Aggregate replicates without forcing an unstable state into a hard label."""

    parsed_modes = tuple(CausalComputeMode.parse(mode) for mode in modes)
    parsed_continuation = CausalComputeMode.parse(continuation_mode)
    if not records:
        raise ValueError("Gate label construction requires paired branch records.")
    if any(record.identity != feature.state for record in records):
        raise ValueError("Gate features and outcomes refer to different states.")
    grouped: dict[CausalComputeMode, list[PairedInterventionRecordV2]] = defaultdict(
        list
    )
    for record in records:
        if (
            record.intervention.control.value == "standard"
            and record.intervention.treatment_chunks == treatment_chunks
            and record.intervention.continuation_mode is parsed_continuation
        ):
            grouped[record.intervention.mode].append(record)
    if set(grouped) != set(parsed_modes):
        raise ValueError("Gate example lacks a complete formal-mode outcome grid.")
    summaries = []
    for mode in parsed_modes:
        mode_records = grouped[mode]
        replicates = [record.intervention.replicate for record in mode_records]
        if len(replicates) != len(set(replicates)):
            raise ValueError("Gate labels repeat a mode/replicate primary outcome.")
        successes = [float(record.outcome.final_success) for record in mode_records]
        progresses = [
            float(record.outcome.progress_terminal) for record in mode_records
        ]
        summaries.append(
            ModeOutcomeSummaryV1(
                mode=mode,
                success_mean=sum(successes) / len(successes),
                success_variance=_sample_variance(successes),
                predicate_progress_mean=sum(progresses) / len(progresses),
                replicate_count=len(successes),
            )
        )
    baseline = summaries[0].success_mean
    uplift = {item.mode.value: item.success_mean - baseline for item in summaries}
    by_replicate: dict[int, dict[str, bool]] = defaultdict(dict)
    for mode in parsed_modes:
        for record in grouped[mode]:
            by_replicate[record.intervention.replicate][mode.value] = bool(
                record.outcome.final_success
            )
    expected = {mode.value for mode in parsed_modes}
    if any(set(row) != expected for row in by_replicate.values()):
        raise ValueError("Gate labels do not have common-random replicate alignment.")
    selected_records = [record for mode in parsed_modes for record in grouped[mode]]
    probability = selected_records[0].sampling.joint_inclusion_probability
    if any(
        record.sampling.joint_inclusion_probability != probability
        for record in selected_records
    ):
        raise ValueError("Gate label inclusion probabilities differ within a state.")
    return GateTrainingExampleV1(
        state=feature.state,
        feature_shard=feature.tensor_shard,
        feature_row=feature.tensor_row,
        proposal_variant=feature.proposal_variant,
        outcomes=tuple(summaries),
        empirical_uplift=uplift,
        inclusion_weight=1.0 / probability,
        fold=fold,
        split=split,
        replicate_joint_outcomes=tuple(
            by_replicate[index] for index in sorted(by_replicate)
        ),
    )
