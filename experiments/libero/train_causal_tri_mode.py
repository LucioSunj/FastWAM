"""Hydra entrypoint for the conditionally authorized tri-mode checkpoint."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from fastwam.causal_prediction_trainer import run_causal_tri_mode


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="causal_tri_mode",
)
def main(cfg: DictConfig) -> None:
    """Launch formal C0/C1/C2 training after its external phase gate."""

    run_causal_tri_mode(cfg)


if __name__ == "__main__":
    main()
