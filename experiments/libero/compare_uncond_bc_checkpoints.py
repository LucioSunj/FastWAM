"""Compare resumable state in two UNCOND-LoRA BC checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from fastwam.uncond_bc_checkpoint import compare_uncond_bc_checkpoints


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Comparison output already exists: {output}")
    report = compare_uncond_bc_checkpoints(args.first, args.second)
    report["command"] = list(sys.argv)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        json.dumps(
            {
                "result": report["result"],
                "exact_training_state": report["exact_training_state"],
                "mismatch_count": report["mismatch_count"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    if report["result"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
