"""Evaluate one P1 checkpoint under a fixed visual-memory intervention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir

from fastwam.p1_dino_bc_eval import (
    P1_MEMORY_MODES,
    evaluate_p1_checkpoint,
    write_p1_offline_result,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--memory-mode",
        required=True,
        choices=sorted(P1_MEMORY_MODES),
    )
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    """Load the fixed P1 config and write a compact offline metric record."""

    args = _parse_args()
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    with initialize_config_dir(
        version_base="1.3",
        config_dir=str(config_dir),
    ):
        cfg = compose(config_name="p1_dino_bc")
    result = evaluate_p1_checkpoint(
        cfg,
        checkpoint=args.checkpoint,
        memory_mode=args.memory_mode,
        max_batches=args.max_batches,
    )
    output = (
        Path(args.output)
        if args.output
        else Path(args.checkpoint).resolve().parent
        / f"offline_memory_{args.memory_mode}.json"
    )
    write_p1_offline_result(output, result)
    print(json.dumps({**result, "output": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
