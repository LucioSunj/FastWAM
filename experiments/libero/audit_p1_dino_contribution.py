"""CLI for the frozen DINO-to-action paired causal audit."""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

import torch
from omegaconf import OmegaConf

from fastwam.p1_dino_contribution_audit import (
    build_causal_audit_ledger,
    run_causal_audit_shard,
)
from fastwam.uncond_bc_trainer import _atomic_json


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in _csv(value))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--checkpoint-kind",
        default="auto",
        choices=("auto", "best", "full"),
    )
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--modes",
        default="correct,off,shuffled,task_paired,drop_main,drop_wrist",
    )
    parser.add_argument("--draw-seeds", default="42,43,44")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--build-ledger-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    cfg = OmegaConf.load(Path(args.config).expanduser().resolve())
    output_dir = Path(args.output_dir).expanduser().resolve()
    try:
        if args.build_ledger_only:
            ledger = build_causal_audit_ledger(cfg, args.ledger)
            output_dir.mkdir(parents=True, exist_ok=True)
            result = {
                "status": "PASS",
                "ledger": str(Path(args.ledger).expanduser().resolve()),
                "ledger_file_sha256": ledger["ledger_file_sha256"],
                "content_sha256": ledger["content_sha256"],
                "anchor_count": len(ledger["anchors"]),
            }
            _atomic_json(output_dir / "ledger_build.json", result)
        else:
            if not args.checkpoint:
                raise ValueError("--checkpoint is required unless building the ledger.")
            visible = _csv(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
            if len(visible) != 1 or visible[0] not in {"4", "5", "6", "7"}:
                raise ValueError(
                    "Each causal-audit process must expose exactly one of GPUs 4--7."
                )
            result = run_causal_audit_shard(
                cfg,
                checkpoint=args.checkpoint,
                checkpoint_kind=args.checkpoint_kind,
                ledger_path=args.ledger,
                shard_index=args.shard_index,
                num_shards=args.num_shards,
                modes=_csv(args.modes),
                draw_seeds=_int_csv(args.draw_seeds),
                output_dir=output_dir,
                device=torch.device("cuda:0"),
            )
        print(json.dumps(result, sort_keys=True))
    except Exception as error:
        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(
            output_dir / "failure.json",
            {
                "status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "args": vars(args),
            },
        )
        raise


if __name__ == "__main__":
    main()
