"""Hydra entrypoint for full held-out UNCOND LoRA BC evaluation."""

from __future__ import annotations

import json
import os

import hydra
import torch.distributed as dist

from fastwam.uncond_bc_offline import (
    claim_uncond_bc_offline_output,
    failure_traceback,
    record_uncond_bc_offline_failure,
    run_uncond_bc_offline,
)


@hydra.main(
    version_base="1.3", config_path="../../configs", config_name="uncond_bc_eval"
)
def main(cfg) -> None:
    """Evaluate one explicit zero- or BC-LoRA policy without RL components."""

    try:
        claim_uncond_bc_offline_output(cfg)
        result = run_uncond_bc_offline(cfg)
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "policy": result["policy"],
                        "loss_action_bc": result["validation"]["loss_action_bc"],
                        "sample_count": result["validation"]["sample_count"],
                        "output_dir": str(cfg.runner.output_dir),
                    },
                    sort_keys=True,
                )
            )
    except Exception as error:
        record_uncond_bc_offline_failure(
            cfg,
            error,
            traceback_text=failure_traceback(),
        )
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
