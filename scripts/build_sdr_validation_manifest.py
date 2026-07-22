#!/usr/bin/env python3
"""Build or verify the fixed episode-disjoint S-DR validation manifest."""
from __future__ import annotations

import argparse
import json

from fastwam.adaptive_gate.sdr_validation import (
    build_validation_manifest,
    episodes_for_manifest_split,
    read_manifest,
    validate_validation_manifest,
    write_manifest,
)


def build(args: argparse.Namespace) -> None:
    manifest = build_validation_manifest(
        dataset_dirs=args.dataset_dir,
        dataset_stats=args.dataset_stats,
        sample_count=args.sample_count,
        seed=args.seed,
        num_frames=args.num_frames,
    )
    write_manifest(args.out, manifest)
    split = episodes_for_manifest_split(
        dataset_dirs=args.dataset_dir,
        manifest=manifest,
        split="train",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "manifest": args.out,
                "sample_count": manifest["sample_count"],
                "validation_episode_count": len(manifest["validation_episodes"]),
                "train_episode_count": sum(len(value) for value in split.values()),
                "selection_fingerprint": manifest["selection_fingerprint"],
            },
            sort_keys=True,
        )
    )


def check(args: argparse.Namespace) -> None:
    manifest = read_manifest(args.manifest)
    validate_validation_manifest(
        manifest,
        dataset_dirs=args.dataset_dir,
        dataset_stats=args.dataset_stats,
    )
    train = episodes_for_manifest_split(
        dataset_dirs=args.dataset_dir,
        manifest=manifest,
        split="train",
    )
    validation = episodes_for_manifest_split(
        dataset_dirs=args.dataset_dir,
        manifest=manifest,
        split="validation",
    )
    overlaps = {
        path: sorted(set(train[path]) & set(validation[path]))
        for path in train
        if set(train[path]) & set(validation[path])
    }
    if overlaps:
        raise ValueError(f"Train/validation episodes overlap: {overlaps}.")
    print(
        json.dumps(
            {
                "status": "PASS",
                "manifest": args.manifest,
                "sample_count": manifest["sample_count"],
                "episode_disjoint": True,
                "selection_fingerprint": manifest["selection_fingerprint"],
            },
            sort_keys=True,
        )
    )


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--dataset-dir", action="append", required=True)
    build_parser.add_argument("--dataset-stats", required=True)
    build_parser.add_argument("--sample-count", type=int, default=40)
    build_parser.add_argument("--seed", type=int, default=20260721)
    build_parser.add_argument("--num-frames", type=int, default=33)
    build_parser.add_argument("--out", required=True)
    build_parser.set_defaults(func=build)
    check_parser = sub.add_parser("check")
    check_parser.add_argument("--dataset-dir", action="append", required=True)
    check_parser.add_argument("--dataset-stats", required=True)
    check_parser.add_argument("--manifest", required=True)
    check_parser.set_defaults(func=check)
    return ap


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
