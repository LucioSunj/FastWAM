"""Merge deterministic DINO contribution audit shards."""

from __future__ import annotations

import argparse
import json

from fastwam.p1_dino_contribution_audit import merge_causal_audit_shards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dirs", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = merge_causal_audit_shards(args.shard_dirs, args.output_dir)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
