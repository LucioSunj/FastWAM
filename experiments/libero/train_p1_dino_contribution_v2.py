"""Hydra entrypoint for DINO-to-action contribution-v2 training."""

from __future__ import annotations

import json
import os
import traceback

import hydra
import torch.distributed as dist

from fastwam.p1_dino_bc_full_trainer import (
    claim_p1_full_output,
    record_p1_full_failure,
    run_p1_dino_bc_full,
)


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="p1_dino_bc_dino_contribution_v2",
)
def main(cfg) -> None:
    """Run one strict canary, pilot, or formal contribution-v2 job."""

    try:
        claim_p1_full_output(cfg)
        result = run_p1_dino_bc_full(cfg)
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                json.dumps(
                    {
                        "status": result.get("status"),
                        "completion": result.get("completion"),
                        "optimizer_steps": result.get("optimizer_steps"),
                        "best_step": result.get("best_step"),
                        "output_dir": str(cfg.runner.output_dir),
                    },
                    sort_keys=True,
                )
            )
    except Exception as error:
        record_p1_full_failure(
            cfg,
            error,
            traceback_text=traceback.format_exc(),
        )
        traceback.print_exc()
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
