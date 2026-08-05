import json
from pathlib import Path

import pytest

from fastwam.uncond_bc_offline import EXPECTED_VALIDATION_WINDOWS
from fastwam.uncond_bc_offline_summary import (
    OFFLINE_COMPARISON_SCHEMA,
    compare_uncond_bc_offline,
    write_offline_comparison,
)


def _manifest(path: Path, *, policy: str, loss: float) -> Path:
    contract = {
        "parent_checkpoint_sha256": "a" * 64,
        "statistics_sha256": "b" * 64,
        "dataset_sha256": {"suite": "c" * 64},
        "text_cache_sha256": "d" * 64,
        "world_size": 4,
        "lora": {"rank": 16},
        "bc_policy": {"action_dim": 7},
    }
    payload = {
        "schema": "fastwam-uncond-bc-offline-eval-v1",
        "status": "PASS",
        "policy": policy,
        "future_prediction_calls": 0,
        "lora_gradients_absent": True,
        "frozen_parameter_versions_unchanged": True,
        "contains_gate": False,
        "contains_critic": False,
        "contains_value_head": False,
        "contract": contract,
        "validation": {
            "sample_count": EXPECTED_VALIDATION_WINDOWS,
            "loss_action_bc": loss,
            "mse_per_dimension": [loss + index for index in range(7)],
        },
        "sidecar": None if policy == "zero_lora" else {"sha256": "e" * 64},
    }
    path.write_text(json.dumps(payload) + "\n")
    return path


def test_offline_comparison_reports_signed_and_relative_changes(tmp_path) -> None:
    zero = _manifest(tmp_path / "zero.json", policy="zero_lora", loss=2.0)
    trained = _manifest(tmp_path / "bc.json", policy="bc_lora", loss=1.5)

    result = compare_uncond_bc_offline(zero, trained)

    assert result["schema"] == OFFLINE_COMPARISON_SCHEMA
    assert result["bc_minus_zero_loss_action_bc"] == -0.5
    assert result["relative_loss_reduction"] == 0.25
    assert result["bc_improves_offline_loss"] is True
    assert result["bc_minus_zero_mse_per_dimension"] == [-0.5] * 7


def test_offline_comparison_rejects_contract_or_sample_drift(tmp_path) -> None:
    zero = _manifest(tmp_path / "zero.json", policy="zero_lora", loss=2.0)
    trained = _manifest(tmp_path / "bc.json", policy="bc_lora", loss=1.5)
    payload = json.loads(trained.read_text())
    payload["contract"]["statistics_sha256"] = "f" * 64
    trained.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="paired contract mismatch"):
        compare_uncond_bc_offline(zero, trained)

    trained = _manifest(tmp_path / "bc.json", policy="bc_lora", loss=1.5)
    payload = json.loads(trained.read_text())
    payload["validation"]["sample_count"] -= 1
    trained.write_text(json.dumps(payload) + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        compare_uncond_bc_offline(zero, trained)


def test_offline_comparison_writer_refuses_overwrite(tmp_path) -> None:
    zero = _manifest(tmp_path / "zero.json", policy="zero_lora", loss=2.0)
    trained = _manifest(tmp_path / "bc.json", policy="bc_lora", loss=1.5)
    payload = compare_uncond_bc_offline(zero, trained)
    output = tmp_path / "comparison" / "result.json"

    write_offline_comparison(output, payload)

    assert json.loads(output.read_text()) == payload
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_offline_comparison(output, payload)
