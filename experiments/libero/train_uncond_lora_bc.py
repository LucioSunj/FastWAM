"""Hydra entrypoint for frozen-IDM UNCOND-LoRA action-only behavior cloning."""

from __future__ import annotations

import json
import os
import traceback

import hydra
import torch.distributed as dist

from fastwam.uncond_bc_trainer import (
    claim_uncond_bc_output,
    record_uncond_bc_failure,
    run_uncond_bc,
)


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="uncond_bc",
)
def main(cfg) -> None:
    """Run the isolated BC work package without constructing any RL component."""

    try:
        claim_uncond_bc_output(cfg)
        result = run_uncond_bc(cfg)
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                json.dumps(
                    {
                        "status": result.get("status"),
                        "stage": result.get("stage"),
                        "optimizer_steps": result.get("optimizer_steps"),
                        "loss_action_bc": result.get("loss_action_bc"),
                        "best_validation_loss_action_bc": result.get(
                            "best_validation_loss_action_bc"
                        ),
                        "output_dir": str(cfg.runner.output_dir),
                    },
                    sort_keys=True,
                    default=str,
                )
            )
    except Exception as error:
        record_uncond_bc_failure(
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
