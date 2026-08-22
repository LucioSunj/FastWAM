"""Launch the matched-budget current-only causal adapter diagnostic."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from fastwam.causal_prediction_trainer import run_causal_current_only


@hydra.main(
    config_path="../../configs",
    config_name="causal_current_only",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Run current-only adapter exposure training."""

    run_causal_current_only(cfg)


if __name__ == "__main__":
    main()
