"""Select the preregistered UNCOND action-only BC learning rate."""

from __future__ import annotations

import argparse
import json

from fastwam.adapters import sha256_file
from fastwam.uncond_bc_selection import (
    select_uncond_bc_learning_rate,
    write_lr_selection_manifest,
)


def main() -> None:
    """Validate exactly three pilot directories and emit the selection artifact."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = select_uncond_bc_learning_rate(args.pilot)
    path = write_lr_selection_manifest(args.output, payload)
    print(
        json.dumps(
            {
                "status": "PASS",
                "selected_learning_rate": payload["selected_learning_rate"],
                "artifact": str(path),
                "sha256": sha256_file(path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
