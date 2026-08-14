"""Hydra entrypoint for P1 DINO semantic-memory BC feasibility stages."""

from __future__ import annotations

import json
import traceback

import hydra

from fastwam.p1_dino_bc_runner import (
    claim_p1_output,
    record_p1_failure,
    run_p1_dino_bc,
)


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="p1_dino_bc",
)
def main(cfg) -> None:
    """Run one fail-fast P1 stage and preserve failures as evidence."""

    try:
        claim_p1_output(cfg)
        result = run_p1_dino_bc(cfg)
        print(
            json.dumps(
                {
                    "status": result.get("status"),
                    "stage": str(cfg.runner.stage),
                    "arm": str(cfg.runner.arm),
                    "output_dir": str(cfg.runner.output_dir),
                },
                sort_keys=True,
            )
        )
    except Exception as error:
        record_p1_failure(
            cfg,
            error,
            traceback_text=traceback.format_exc(),
        )
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
