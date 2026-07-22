#!/usr/bin/env python3
"""Apply the preregistered E1-P2 intermediate hard-stop contract."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from decide_sdr_learning_probe import _diagnostic_summary, _training_summary
from fastwam.adaptive_gate.sdr_contracts import (
    artifact_record,
    atomic_json,
    read_json,
    validate_learning_probe_contract,
)


SCHEMA = "fastwam-sdr-formal-monitor-v1"
ALLOWED_FRACTIONS = (0.05, 0.10, 0.25, 0.50, 0.75, 1.00)


def _parse_evaluation(value: str) -> tuple[float, Path]:
    fraction_text, separator, path_text = value.partition("=")
    if not separator or not path_text:
        raise argparse.ArgumentTypeError(
            "--evaluation must be FRACTION=/absolute/diagnostics_dir"
        )
    fraction = float(fraction_text)
    if fraction not in ALLOWED_FRACTIONS:
        raise argparse.ArgumentTypeError(
            f"unsupported formal evaluation fraction {fraction}"
        )
    return fraction, Path(path_text).expanduser().resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-decision", required=True)
    parser.add_argument("--baseline-diagnostics", required=True)
    parser.add_argument(
        "--evaluation",
        action="append",
        type=_parse_evaluation,
        required=True,
    )
    parser.add_argument("--training-metrics", required=True)
    parser.add_argument("--current-delta", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    preflight = read_json(args.preflight_decision)
    contract = validate_learning_probe_contract(preflight)
    evaluations = sorted(args.evaluation, key=lambda item: item[0])
    fractions = [fraction for fraction, _ in evaluations]
    if len(fractions) != len(set(fractions)):
        raise ValueError("Formal monitor fractions must be unique.")
    expected_prefix = list(ALLOWED_FRACTIONS[: len(fractions)])
    if fractions != expected_prefix:
        raise ValueError(
            "Formal monitor evaluations must be a contiguous preregistered prefix: "
            f"expected={expected_prefix}, observed={fractions}."
        )

    delta = torch.load(args.current_delta, map_location="cpu", weights_only=False)
    provenance = delta.get("fastwam_provenance")
    if not isinstance(provenance, dict):
        raise ValueError("Formal monitor delta has no FastWAM provenance.")
    training_contract = provenance.get("dual_regime_training_contract")
    if not isinstance(training_contract, dict):
        raise ValueError("Formal monitor delta has no training contract.")
    total_steps = int(training_contract["total_optimizer_steps"])
    current_step = int(delta.get("step"))
    expected_current_step = int(math.ceil(fractions[-1] * total_steps))
    if current_step != expected_current_step:
        raise ValueError(
            "Formal checkpoint is not at its preregistered successful-step "
            f"fraction: observed={current_step}, expected={expected_current_step}."
        )
    if provenance.get("parent_checkpoint_sha256") != contract[
        "artifact_bindings"
    ]["e_i_checkpoint"]:
        raise ValueError("Formal monitor delta has a different E-I parent.")

    training = _training_summary(
        args.training_metrics,
        expected_steps=current_step,
        schedule=contract["schedule"],
        schedule_total_steps=total_steps,
    )
    baseline = _diagnostic_summary(
        args.baseline_diagnostics,
        weight=contract["w_cap"],
    )
    summaries = [
        {
            "fraction": fraction,
            **_diagnostic_summary(path, weight=contract["w_cap"]),
        }
        for fraction, path in evaluations
    ]

    immediate_failures = []
    current = summaries[-1]
    if not training["all_finite"] or not current["finite"]:
        immediate_failures.append("non-finite training or diagnostic metric")
    if not current["no_read_uncond_parity"]:
        immediate_failures.append("no-read/forced-UNCOND parity failed")

    condition_names = {
        "gt_idm_error_over_10pct": lambda row: (
            row["idm_raw_loss"] > 1.10 * baseline["idm_raw_loss"]
        ),
        "generated_idm_error_over_10pct": lambda row: (
            row["generated_idm_action_l2"]
            > 1.10 * baseline["generated_idm_action_l2"]
        ),
        "sensitivity_below_half": lambda row: (
            row["sensitivity_median"] < 0.5 * baseline["sensitivity_median"]
        ),
        "idm_margin_nonpositive": lambda row: (
            row["common_action_all_idm_margin"] <= 0.0
        ),
        "final_block_conflict_fraction": lambda row: (
            row["final_blocks_negative_idm_margin_fraction"] >= 0.20
        ),
    }
    condition_history = {
        name: [bool(predicate(row)) for row in summaries]
        for name, predicate in condition_names.items()
    }
    consecutive_failures = [
        name
        for name, history in condition_history.items()
        if len(history) >= 2 and history[-2:] == [True, True]
    ]
    failures = immediate_failures + [
        f"two consecutive evaluation points: {name}"
        for name in consecutive_failures
    ]
    result = {
        "schema": SCHEMA,
        "status": "CONTINUE" if not failures else "FAIL-DIAGNOSED",
        "plus_full_used": False,
        "artifact_bindings": preflight["artifact_bindings"],
        "total_optimizer_steps": total_steps,
        "current_optimizer_step": current_step,
        "evaluated_fractions": fractions,
        "training": training,
        "baseline": baseline,
        "evaluations": summaries,
        "condition_history": condition_history,
        "immediate_failures": immediate_failures,
        "consecutive_failures": consecutive_failures,
        "failure_conditions": failures,
        "current_delta": artifact_record(args.current_delta),
        "no_plus_full_statement": (
            "No Plus-Full or task rollout outcome was loaded or used."
        ),
    }
    atomic_json(args.out, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "current_optimizer_step": current_step,
                "evaluated_fractions": fractions,
            },
            sort_keys=True,
        )
    )
    if failures:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
