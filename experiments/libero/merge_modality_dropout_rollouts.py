"""Merge per-condition 30-episode rollout artifacts for pilot decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastwam.modality_dropout_bc import PILOT_ARMS
from fastwam.modality_dropout_libero_inference import ROLLOUT_MODALITY_CONDITIONS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    arms = {}
    for value in args.inputs:
        path = Path(value).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != "fastwam-modality-dropout-rollout-condition-v1":
            raise ValueError(f"Unsupported rollout condition artifact: {path}.")
        arm = str(payload["arm"])
        condition = str(payload["condition"])
        if arm not in PILOT_ARMS or condition not in ROLLOUT_MODALITY_CONDITIONS:
            raise ValueError(f"Unexpected rollout arm/condition in {path}.")
        if condition in arms.setdefault(arm, {}):
            raise ValueError(f"Duplicate rollout condition {arm}/{condition}.")
        arms[arm][condition] = payload
    required = {"clean", "wan_drop", "dino_drop"}
    missing = {
        arm: sorted(required - set(arms.get(arm, {})))
        for arm in PILOT_ARMS
        if required - set(arms.get(arm, {}))
    }
    if missing:
        raise ValueError(f"Rollout merge is missing conditions: {missing}.")
    result = {
        "schema": "fastwam-modality-dropout-rollout-v1",
        "status": "COMPLETE",
        "arms": arms,
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "COMPLETE", "output": str(output)}))


if __name__ == "__main__":
    main()
