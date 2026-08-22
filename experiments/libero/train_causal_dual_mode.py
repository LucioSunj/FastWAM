"""Hydra entrypoint for the isolated shared C0/C2 causal policy."""

from __future__ import annotations

import json
import os

import hydra
import torch.distributed as dist

from fastwam.causal_prediction_trainer import (
    claim_causal_output,
    run_causal_dual_mode,
)


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="causal_dual_mode",
)
def main(cfg) -> None:
    """Run dual-mode training without constructing adaptive RL components."""

    try:
        if int(os.environ.get("RANK", "0")) == 0:
            claim_causal_output(cfg)
        result = run_causal_dual_mode(cfg)
        if int(os.environ.get("RANK", "0")) == 0:
            print(json.dumps(result, sort_keys=True, default=str))
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
