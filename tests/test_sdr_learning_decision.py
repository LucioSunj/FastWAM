import importlib.util
import json
import struct
from pathlib import Path


def _decision_module():
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts/decide_sdr_learning_probe.py"
    spec = importlib.util.spec_from_file_location("sdr_learning_decision", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_training_summary_compares_logged_float32_schedule_weight(tmp_path):
    module = _decision_module()
    weight = 0.009558040980013546
    logged_weight = struct.unpack("!f", struct.pack("!f", weight))[0]
    ledger = tmp_path / "training_metrics.jsonl"
    row = {
        "global_step": 1,
        "dual_regime_optimizer_steps": 1,
        "gradient_clipped": False,
        "optimizer_step_was_skipped": False,
        "peak_gpu_memory_bytes": 1,
        "elapsed_seconds": 1.0,
        "step_duration_seconds": 1.0,
        "samples_per_second": 1.0,
        "losses": {"action_regime_weight_uncond": logged_weight},
    }
    ledger.write_text(json.dumps(row) + "\n", encoding="utf-8")

    summary = module._training_summary(
        str(ledger),
        expected_steps=1,
        schedule=[[0.0, weight], [1.0, weight]],
    )

    assert summary["schedule_verified"] is True
    assert summary["successful_steps"] == 1
