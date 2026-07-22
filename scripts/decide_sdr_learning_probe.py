#!/usr/bin/env python3
"""Apply preregistered Canary and 500-step S-DR learning-probe gates."""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fastwam.adaptive_gate.sdr_contracts import (
    LEARNING_PROBE_DECISION_SCHEMA,
    artifact_record,
    atomic_json,
    read_json,
    validate_learning_probe_contract,
)
from fastwam.adaptive_gate.training import uncond_weight_at_step


CANARY_SCHEMA = "fastwam-sdr-canary-decision-v1"


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object.")
            rows.append(row)
    return rows


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence.")
    return sum(float(value) for value in values) / len(values)


def _longest_true_streak(values: Sequence[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _diagnostic_summary(run_dir: str | Path, *, weight: float) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    gradient = read_json(root / "gradient_diagnostics.json")
    generated = read_json(root / "generated_future_validation.json")
    common_losses = [
        row
        for row in gradient["loss_records"]
        if row["coupling"] == "common"
    ]
    valid_errors = [
        row["conditions"]["valid_self_generated_future"]["normalized_error"]["l2"]
        for row in generated["records"]
    ]
    gt_errors = [
        row["conditions"]["gt_teacher_forced_future"]["normalized_error"]["l2"]
        for row in generated["records"]
    ]
    margin = gradient["common"]["groups"]["action_all"]["margins_by_weight"][
        str(float(weight))
    ]
    final_summary = gradient["common"]["shard_margin_summaries"][
        "action_blocks_final"
    ][f"idm_margin@{float(weight)}"]
    return {
        "run_dir": str(root),
        "uncond_raw_loss": _mean(
            [row["raw_uncond"] for row in common_losses]
        ),
        "idm_raw_loss": _mean([row["raw_idm"] for row in common_losses]),
        "generated_idm_action_l2": _mean(valid_errors),
        "gt_idm_action_l2": _mean(gt_errors),
        "sensitivity_median": float(
            generated["sensitivity_gate"]["median"]
        ),
        "common_action_all_idm_margin": float(margin["idm_margin"]),
        "common_action_all_uncond_margin": float(margin["uncond_margin"]),
        "final_blocks_negative_idm_margin_fraction": float(
            final_summary["negative_fraction"]
        ),
        "no_read_uncond_parity": bool(
            generated["no_read_uncond_parity"]["pass"]
        ),
        "finite": _finite_tree(gradient) and _finite_tree(generated),
    }


def _finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _training_summary(
    path: str,
    *,
    expected_steps: int,
    schedule: Sequence[Sequence[float]] | None = None,
    schedule_total_steps: int | None = None,
) -> dict[str, Any]:
    rows = _read_jsonl(path)
    steps = [int(row["global_step"]) for row in rows]
    expected = list(range(1, int(expected_steps) + 1))
    if steps != expected:
        raise ValueError(
            "Training ledger does not contain exactly the successful steps "
            f"1..{expected_steps}."
        )
    clipped = [bool(row["gradient_clipped"]) for row in rows]
    schedule_steps = [int(row["dual_regime_optimizer_steps"]) for row in rows]
    if schedule_steps != expected:
        raise ValueError("Dual-regime schedule did not advance by successful updates.")
    if any(bool(row.get("optimizer_step_was_skipped")) for row in rows):
        raise ValueError("Successful-step ledger contains a skipped optimizer step.")
    schedule_verified = schedule is not None
    if schedule is not None:
        schedule_total = (
            int(schedule_total_steps)
            if schedule_total_steps is not None
            else int(expected_steps)
        )
        if schedule_total < expected_steps:
            raise ValueError(
                "Schedule total cannot be smaller than the observed ledger."
            )
        for step, row in zip(expected, rows):
            observed = float(
                row["losses"]["action_regime_weight_uncond"]
            )
            expected_weight = uncond_weight_at_step(
                schedule,
                optimizer_step=step - 1,
                total_optimizer_steps=schedule_total,
            )
            if not math.isclose(
                observed,
                expected_weight,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "Dual-regime weight schedule drift at successful step "
                    f"{step}: observed={observed}, expected={expected_weight}."
                )
    step_durations = [
        float(row.get("step_duration_seconds", 0.0)) for row in rows
    ]
    measured_throughput = [
        float(row.get("samples_per_second", 0.0)) for row in rows
    ]
    return {
        "successful_steps": len(rows),
        "clip_fraction": sum(clipped) / len(clipped),
        "longest_clipping_streak": _longest_true_streak(clipped),
        "peak_gpu_memory_bytes": max(
            int(row["peak_gpu_memory_bytes"]) for row in rows
        ),
        "all_finite": _finite_tree(rows),
        "schedule_verified": schedule_verified,
        "total_step_time_seconds": sum(step_durations),
        "mean_samples_per_second": (
            _mean(measured_throughput)
            if all(value > 0.0 for value in measured_throughput)
            else None
        ),
        "final_segment_elapsed_seconds": float(rows[-1]["elapsed_seconds"]),
    }


def _slope_upper_95(xs: Sequence[float], ys: Sequence[float]) -> tuple[float, float]:
    if len(xs) != len(ys) or len(xs) < 3:
        raise ValueError("Slope CI requires at least three paired points.")
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    sxx = sum((x - x_mean) ** 2 for x in xs)
    slope = sum(
        (x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)
    ) / sxx
    intercept = y_mean - slope * x_mean
    residual = sum(
        (y - intercept - slope * x) ** 2 for x, y in zip(xs, ys)
    )
    standard_error = math.sqrt(residual / (len(xs) - 2) / sxx)
    t_critical = 3.182 if len(xs) == 5 else 4.303
    return slope, slope + t_critical * standard_error


def decide_canary(args: argparse.Namespace) -> None:
    preflight_payload = read_json(args.preflight_decision)
    contract = validate_learning_probe_contract(preflight_payload)
    baseline = _diagnostic_summary(
        args.baseline_diagnostics,
        weight=contract["w_cap"],
    )
    final = _diagnostic_summary(
        args.canary_diagnostics,
        weight=contract["w_cap"],
    )
    training = _training_summary(
        args.training_metrics,
        expected_steps=50,
        schedule=contract["schedule"],
    )
    failures = []
    if training["clip_fraction"] >= 0.20:
        failures.append("gradient clipping fraction is not below 20%")
    if training["longest_clipping_streak"] >= 10:
        failures.append("gradient clipping streak reached 10 steps")
    if final["generated_idm_action_l2"] > 1.05 * baseline[
        "generated_idm_action_l2"
    ]:
        failures.append("generated-IDM action error worsened by more than 5%")
    if final["common_action_all_idm_margin"] <= 0.0:
        failures.append("common-noise IDM margin is not positive")
    if not training["all_finite"] or not final["finite"]:
        failures.append("non-finite metric")
    if not final["no_read_uncond_parity"]:
        failures.append("no-read/forced-UNCOND parity failed")
    result = {
        "schema": CANARY_SCHEMA,
        "status": "PASS" if not failures else "FAIL-DIAGNOSED",
        "plus_full_used": False,
        "artifact_bindings": preflight_payload["artifact_bindings"],
        "initializer_stage": "e_i_s0",
        "training": training,
        "baseline": baseline,
        "step50": final,
        "failure_conditions": failures,
        "probe_500_authorized": not failures,
        "artifacts": {
            "preflight_decision": artifact_record(args.preflight_decision),
            "training_metrics": artifact_record(args.training_metrics),
            "canary_delta": artifact_record(args.canary_delta),
        },
    }
    atomic_json(args.out, result)
    if failures:
        raise SystemExit(3)


def decide_probe(args: argparse.Namespace) -> None:
    preflight_payload = read_json(args.preflight_decision)
    contract = validate_learning_probe_contract(preflight_payload)
    canary = read_json(args.canary_decision)
    if canary.get("schema") != CANARY_SCHEMA or canary.get("status") != "PASS":
        raise ValueError("500-step probe requires a PASS Canary decision.")
    if canary.get("artifact_bindings") != preflight_payload.get(
        "artifact_bindings"
    ):
        raise ValueError("Canary and preflight artifact bindings differ.")
    steps = [0, 50, 100, 250, 500]
    directories = [
        args.step0_diagnostics,
        args.step50_diagnostics,
        args.step100_diagnostics,
        args.step250_diagnostics,
        args.step500_diagnostics,
    ]
    summaries = {
        str(step): _diagnostic_summary(
            directory,
            weight=contract["w_cap"],
        )
        for step, directory in zip(steps, directories)
    }
    training = _training_summary(
        args.training_metrics,
        expected_steps=500,
        schedule=contract["schedule"],
    )
    baseline = summaries["0"]
    final = summaries["500"]
    uncond_values = [summaries[str(step)]["uncond_raw_loss"] for step in steps]
    relative_uncond_drop = (
        baseline["uncond_raw_loss"] - final["uncond_raw_loss"]
    ) / max(baseline["uncond_raw_loss"], 1e-12)
    slope, slope_upper = _slope_upper_95(steps, uncond_values)
    consistent_trend = sum(
        right <= left
        for left, right in zip(uncond_values, uncond_values[1:])
    ) >= 3
    failures = []
    if not (
        (relative_uncond_drop >= 0.10 and consistent_trend)
        or slope_upper < 0.0
    ):
        failures.append("UNCOND held-out raw loss did not improve sufficiently")
    if final["idm_raw_loss"] > 1.05 * baseline["idm_raw_loss"]:
        failures.append("GT-IDM raw loss worsened by more than 5%")
    if final["generated_idm_action_l2"] > 1.05 * baseline[
        "generated_idm_action_l2"
    ]:
        failures.append("generated-IDM action error worsened by more than 5%")
    if final["sensitivity_median"] < 0.5 * baseline["sensitivity_median"]:
        failures.append("valid/no-read sensitivity retained less than 50%")
    if final["common_action_all_idm_margin"] <= 0.0:
        failures.append("final common-noise IDM margin is not positive")
    if final["final_blocks_negative_idm_margin_fraction"] >= 0.20:
        failures.append("final-block negative IDM margin fraction reached 20%")
    if not training["all_finite"] or not all(
        summary["finite"] for summary in summaries.values()
    ):
        failures.append("non-finite metric")
    if not all(
        summary["no_read_uncond_parity"] for summary in summaries.values()
    ):
        failures.append("no-read/forced-UNCOND parity failed")
    result = {
        "schema": LEARNING_PROBE_DECISION_SCHEMA,
        "status": "PASS" if not failures else "FAIL-DIAGNOSED",
        "plus_full_used": False,
        "artifact_bindings": preflight_payload["artifact_bindings"],
        "initializer_stage": "e_i_s0",
        "canary": "PASS",
        "probe_500_step": "PASS" if not failures else "FAIL-DIAGNOSED",
        "training": training,
        "evaluation_steps": summaries,
        "uncond_raw_loss": {
            "relative_drop": relative_uncond_drop,
            "consistent_trend": consistent_trend,
            "linear_slope": slope,
            "slope_95_upper": slope_upper,
        },
        "failure_conditions": failures,
        "formal_training_authorized": not failures,
        "final_probe_delta": artifact_record(args.step500_delta),
        "no_plus_full_statement": (
            "No Plus-Full outcome was loaded or used for this decision."
        ),
    }
    atomic_json(args.out, result)
    if failures:
        raise SystemExit(3)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    canary = sub.add_parser("canary")
    canary.add_argument("--preflight-decision", required=True)
    canary.add_argument("--baseline-diagnostics", required=True)
    canary.add_argument("--canary-diagnostics", required=True)
    canary.add_argument("--training-metrics", required=True)
    canary.add_argument("--canary-delta", required=True)
    canary.add_argument("--out", required=True)
    canary.set_defaults(func=decide_canary)
    probe = sub.add_parser("probe")
    probe.add_argument("--preflight-decision", required=True)
    probe.add_argument("--canary-decision", required=True)
    probe.add_argument("--step0-diagnostics", required=True)
    probe.add_argument("--step50-diagnostics", required=True)
    probe.add_argument("--step100-diagnostics", required=True)
    probe.add_argument("--step250-diagnostics", required=True)
    probe.add_argument("--step500-diagnostics", required=True)
    probe.add_argument("--training-metrics", required=True)
    probe.add_argument("--step500-delta", required=True)
    probe.add_argument("--out", required=True)
    probe.set_defaults(func=decide_probe)
    return ap


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
