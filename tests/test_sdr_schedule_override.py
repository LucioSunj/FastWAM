import json
import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from fastwam.adaptive_gate.sdr_contracts import PREFLIGHT_DECISION_SCHEMA


def test_print_schedule_emits_hydra_compatible_sequence(tmp_path):
    schedule = [
        [0.0, 0.05],
        [0.1, 0.05],
        [0.3, 0.2],
        [0.6, 0.5],
        [1.0, 0.5],
    ]
    bindings = {
        name: {"path": f"/immutable/{name}", "sha256": char * 64}
        for name, char in (
            ("base_model_manifest", "a"),
            ("e_i_checkpoint", "b"),
            ("e_i_config", "c"),
            ("dataset_stats", "d"),
            ("validation_manifest", "e"),
            ("solver_contract", "1"),
            ("code_commit", "2"),
        )
    }
    decision = {
        "schema": PREFLIGHT_DECISION_SCHEMA,
        "status": "PASS",
        "plus_full_used": False,
        "artifact_bindings": bindings,
        "w0": 0.05,
        "w_cap": 0.5,
        "allowed_schedule": schedule,
    }
    decision_path = tmp_path / "preflight_decision.json"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "sdr_stage_contract.py"),
            "print-schedule",
            "--preflight-decision",
            str(decision_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    schedule_override = result.stdout.strip()
    assert json.loads(schedule_override) == schedule

    with initialize_config_dir(
        version_base="1.3",
        config_dir=str((project_root / "configs").resolve()),
    ):
        cfg = compose(
            config_name="train",
            overrides=[
                "task=libero_dual_regime_fused_2cam224_1e-4",
                "dual_regime_training.uncond_weight_schedule="
                + schedule_override,
            ],
        )
    resolved_schedule = OmegaConf.to_container(
        cfg.dual_regime_training.uncond_weight_schedule,
        resolve=True,
    )
    assert resolved_schedule == schedule
