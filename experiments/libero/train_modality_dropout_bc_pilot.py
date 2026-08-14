"""Hydra entrypoint for one symmetric modality-dropout BC pilot arm."""

from __future__ import annotations

import json

import hydra

from fastwam.modality_dropout_bc_runner import run_modality_dropout_bc_pilot


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="modality_dropout_bc_pilot",
)
def main(cfg) -> None:
    """Run A calibration or one arm against A's frozen endpoint K."""

    result = run_modality_dropout_bc_pilot(cfg)
    print(
        json.dumps(
            {
                "status": result["status"],
                "arm": result["arm"]["name"],
                "global_step": result["global_step"],
                "decision": result["decision"],
                "output_dir": str(cfg.runner.output_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
