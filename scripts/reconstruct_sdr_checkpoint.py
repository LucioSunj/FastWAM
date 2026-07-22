#!/usr/bin/env python3
"""Reconstruct a full S-DR checkpoint from an E-I parent and ActionDiT delta."""
from __future__ import annotations

import argparse
import json

from fastwam.adaptive_gate.sdr_contracts import atomic_json
from fastwam.adaptive_gate.sdr_delta import reconstruct_action_dit_delta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--delta-checkpoint", required=True)
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--decision-out", required=True)
    args = parser.parse_args()
    result = reconstruct_action_dit_delta(
        parent_checkpoint=args.parent_checkpoint,
        delta_checkpoint=args.delta_checkpoint,
        output_checkpoint=args.output_checkpoint,
    )
    atomic_json(args.decision_out, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
