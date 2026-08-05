"""Compare full zero-LoRA and trained-LoRA offline BC evaluations."""

from __future__ import annotations

import argparse
import json

from fastwam.adapters import sha256_file
from fastwam.uncond_bc_offline_summary import (
    compare_uncond_bc_offline,
    write_offline_comparison,
)


def main() -> None:
    """Validate paired manifests and emit one immutable comparison."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-manifest", required=True)
    parser.add_argument("--bc-manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = compare_uncond_bc_offline(args.zero_manifest, args.bc_manifest)
    path = write_offline_comparison(args.output, payload)
    print(
        json.dumps(
            {
                "status": "PASS",
                "artifact": str(path),
                "sha256": sha256_file(path),
                "relative_loss_reduction": payload["relative_loss_reduction"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
