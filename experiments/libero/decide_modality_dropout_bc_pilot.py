"""Merge six arm artifacts into the preregistered pilot result table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastwam.modality_dropout_bc import PILOT_ARMS
from fastwam.modality_dropout_bc_decision import (
    decide_modality_dropout_pilot,
    load_arm_evidence,
)


def _optional_json(path: str | None):
    if path is None:
        return None
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    for arm in PILOT_ARMS:
        parser.add_argument(f"--{arm.lower()}", required=True)
    parser.add_argument("--rollout")
    parser.add_argument("--language-canary")
    parser.add_argument("--ood")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    evidence = {
        arm: load_arm_evidence(getattr(args, arm.lower())) for arm in PILOT_ARMS
    }
    result = decide_modality_dropout_pilot(
        evidence,
        rollout=_optional_json(args.rollout),
        language_canary=_optional_json(args.language_canary),
        ood=_optional_json(args.ood),
        bootstrap_draws=args.bootstrap_draws,
    )
    target = Path(args.output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"decision": result["decision"], "output": str(target)}))


if __name__ == "__main__":
    main()
