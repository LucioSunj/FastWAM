"""Apply the preregistered E1 causal-audit decision rules."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from fastwam.uncond_bc_trainer import _atomic_json


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "PASS" or not value.get("finite", False):
        raise ValueError(f"Audit is not finite/PASS: {path}.")
    return value


def decide(best: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    if best["ledger"]["sha256"] != final["ledger"]["sha256"]:
        raise ValueError("Best/final audits use different frozen ledgers.")
    if best["draw_seeds"] != final["draw_seeds"]:
        raise ValueError("Best/final audits use different stateless draws.")
    gap = float(final["relative_loss_gaps"]["task_paired"])
    action_delta = float(final["metrics"]["task_paired"]["action_relative_delta"])
    residual_median = float(final["residual_hidden_median"])
    residual_p95 = float(final["residual_hidden_p95"])
    finite = all(
        math.isfinite(value)
        for value in (gap, action_delta, residual_median, residual_p95)
    )
    if not finite or residual_p95 > 0.20:
        decision = "UNSAFE_SCALE"
        allow_pilots = False
    elif action_delta >= 0.02 and gap <= 0.0:
        decision = "REPRESENTATION_MISALIGNED"
        allow_pilots = False
    elif residual_median >= 0.05 and gap < 0.03 and action_delta < 0.01:
        decision = "ARCHITECTURE_BYPASS"
        allow_pilots = False
    elif gap >= 0.05 or action_delta >= 0.02:
        decision = "PASS_TO_V2_PILOTS"
        allow_pilots = True
    else:
        decision = "NO_GO_AUDIT"
        allow_pilots = False

    def records(payload: dict[str, Any]) -> dict[tuple[str, int, str], dict[str, Any]]:
        return {
            (value["anchor_id"], int(value["draw_seed"]), value["mode"]): value
            for value in payload["paired_records"]
        }

    best_records = records(best)
    final_records = records(final)
    common = sorted(set(best_records) & set(final_records))
    if len(common) != len(best_records) or len(common) != len(final_records):
        raise ValueError("Best/final audit paired-record identities differ.")
    paired_differences = {}
    for mode in final["modes"]:
        keys = [key for key in common if key[2] == mode]
        paired_differences[mode] = {
            metric: sum(
                float(final_records[key][metric]) - float(best_records[key][metric])
                for key in keys
            )
            / len(keys)
            for metric in (
                "loss_action_bc",
                "mse_pose",
                "mse_gripper",
                "velocity_relative_delta",
                "action_relative_delta",
            )
        }
        paired_differences[mode]["pair_count"] = len(keys)
    return {
        "schema": "fastwam-p1-dino-contribution-v2-e1-decision",
        "status": "PASS" if allow_pilots else "STOP",
        "decision": decision,
        "allow_pilots": allow_pilots,
        "threshold_inputs": {
            "final_task_paired_relative_loss_gap": gap,
            "final_task_paired_action_relative_delta": action_delta,
            "final_residual_hidden_median": residual_median,
            "final_residual_hidden_p95": residual_p95,
            "finite": finite,
        },
        "paired_final_minus_best": paired_differences,
        "best_checkpoint": best["checkpoint"],
        "final_checkpoint": final["checkpoint"],
        "ledger": final["ledger"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-audit", required=True, type=Path)
    parser.add_argument("--final-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = decide(_load(args.best_audit), _load(args.final_audit))
    _atomic_json(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
