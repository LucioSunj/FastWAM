"""Inspect a FastWAM UNCOND-LoRA BC training checkpoint without model loading."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from fastwam.uncond_bc_checkpoint import inspect_uncond_bc_checkpoint


def _atomic_json(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite inspector artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output")
    args = parser.parse_args()

    report = inspect_uncond_bc_checkpoint(args.checkpoint)
    if args.output:
        _atomic_json(Path(args.output).expanduser().resolve(), report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
