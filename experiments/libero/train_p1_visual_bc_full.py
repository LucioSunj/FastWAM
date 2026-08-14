"""Hydra entrypoint for four-GPU V2 visual-backbone action BC."""

from __future__ import annotations

import json
import os
import traceback

import hydra
import torch.distributed as dist

from fastwam.p1_dino_bc_full_trainer import (
    claim_p1_full_output,
    record_p1_full_failure,
    run_p1_visual_bc_full,
)


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="p1_visual_bc_full",
)
def main(cfg) -> None:
    """Run one registered-backbone V2 canary or full training job."""

    try:
        claim_p1_full_output(cfg)
        result = run_p1_visual_bc_full(cfg)
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                json.dumps(
                    {
                        "status": result.get("status"),
                        "stage": result.get("stage"),
                        "lora_rank": result.get("lora_rank"),
                        "optimizer_steps": result.get("optimizer_steps"),
                        "best_validation_loss_action_bc": result.get(
                            "best_validation_loss_action_bc"
                        ),
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
