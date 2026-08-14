"""Select P1/P2 or request the sole preregistered P3 pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fastwam.uncond_bc_trainer import _atomic_json


def _epoch_result(root: Path) -> dict[str, Any]:
    records = [
        json.loads(line)
        for line in (root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    epochs = [value for value in records if "causal_assessment" in value]
    if len(epochs) != 1:
        raise ValueError(f"Pilot {root} must contain exactly one epoch assessment.")
    result = epochs[0]
    result["root"] = str(root)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    result["dependency_weight"] = float(manifest["memory_dependency"]["weight"])
    result["gripper_loss_multiplier"] = float(
        manifest["provenance"]["resolved_config"]["training"]["gripper_loss_multiplier"]
    )
    return result


def select(pilots: list[dict[str, Any]]) -> dict[str, Any]:
    if len(pilots) not in {2, 3}:
        raise ValueError("Pilot selection expects P1/P2 and optionally P3.")
    eligible = [value for value in pilots if value["causal_assessment"]["eligible"]]
    if eligible:
        eligible.sort(key=lambda value: value["validation"]["loss_action_bc"])
        best = eligible[0]
        if len(eligible) > 1:
            first_loss = float(eligible[0]["validation"]["loss_action_bc"])
            second_loss = float(eligible[1]["validation"]["loss_action_bc"])
            if abs(second_loss - first_loss) / min(first_loss, second_loss) < 0.002:
                lower_weight = [
                    value for value in eligible if value["dependency_weight"] == 0.25
                ]
                if lower_weight:
                    best = lower_weight[0]
        return {
            "status": "SELECTED",
            "decision": "ELIGIBLE_PILOT_SELECTED",
            "selected": best,
            "pilots": pilots,
        }

    reasons = [set(value["causal_assessment"]["reasons"]) for value in pilots]
    only_gripper = all(reason == {"gripper_mse"} for reason in reasons)
    has_p3 = any(value["gripper_loss_multiplier"] == 1.5 for value in pilots)
    if len(pilots) == 2 and only_gripper:
        better = min(pilots, key=lambda value: value["validation"]["loss_action_bc"])
        return {
            "status": "P3_REQUIRED",
            "decision": "RUN_CONDITIONAL_P3",
            "p3_dependency_weight": better["dependency_weight"],
            "p3_gripper_loss_multiplier": 1.5,
            "pilots": pilots,
        }
    return {
        "status": "STOP",
        "decision": "NO_ELIGIBLE_PILOT" if has_p3 or not only_gripper else "INVALID",
        "pilots": pilots,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = select([_epoch_result(path.resolve()) for path in args.pilot_dirs])
    result["schema"] = "fastwam-p1-dino-contribution-v2-pilot-selection"
    _atomic_json(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
