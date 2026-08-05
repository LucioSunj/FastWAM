import json
from pathlib import Path

import pytest
import torch

from fastwam.adapters import sha256_file
from fastwam.uncond_bc_selection import (
    EXPECTED_VALIDATION_SAMPLES,
    LR_SELECTION_SCHEMA,
    select_uncond_bc_learning_rate,
    write_lr_selection_manifest,
)


def _pilot(tmp_path: Path, *, learning_rate: float, loss: float) -> Path:
    root = tmp_path / f"lr_{learning_rate}"
    root.mkdir()
    sidecar = root / "best_uncond_lora.pt"
    torch.save({"marker": torch.tensor([learning_rate])}, sidecar)
    metrics = root / "metrics.jsonl"
    metrics.write_text(
        json.dumps(
            {
                "epoch": 0,
                "global_step": 1000,
                "validation": {
                    "loss_action_bc": loss,
                    "sample_count": EXPECTED_VALIDATION_SAMPLES,
                    "valid_action_count": 900000,
                },
            }
        )
        + "\n"
    )
    contract = {
        "parent_checkpoint_sha256": "a" * 64,
        "statistics_sha256": "b" * 64,
        "dataset_sha256": {"suite": "c" * 64},
        "text_cache_sha256": "d" * 64,
    }
    config = {
        "runner": {"stage": "pilot", "output_dir": str(root)},
        "training": {"max_steps": 1000},
        "optimizer": {"learning_rate": learning_rate},
    }
    manifest = {
        "schema": "fastwam-uncond-bc-run-manifest-v1",
        "status": "PASS",
        "stage": "pilot",
        "optimizer_steps": 1000,
        "future_prediction_calls": 0,
        "frozen_parameter_versions_unchanged": True,
        "zero_lora_at_start": True,
        "nonzero_update_count": 1000,
        "best_validation_loss_action_bc": loss,
        "contract": contract,
        "provenance": {
            "resolved_config": config,
            "training_contract_sha256": f"contract-{learning_rate}",
        },
        "best_sidecar": {
            "path": str(sidecar),
            "sha256": sha256_file(sidecar),
            "strict_reload": True,
            "tensor_exact": True,
            "current_state_restored": True,
            "bc_step": 1000,
        },
    }
    (root / "run_manifest.json").write_text(json.dumps(manifest) + "\n")
    return root


def test_lr_selection_applies_strict_less_than_one_percent_tie_rule(tmp_path) -> None:
    low = _pilot(tmp_path, learning_rate=3e-5, loss=1.009)
    middle = _pilot(tmp_path, learning_rate=1e-4, loss=1.0)
    high = _pilot(tmp_path, learning_rate=3e-4, loss=1.2)

    result = select_uncond_bc_learning_rate([high, low, middle])

    assert result["schema"] == LR_SELECTION_SCHEMA
    assert result["status"] == "PASS"
    assert result["selected_learning_rate"] == 3e-5
    assert [item["learning_rate"] for item in result["candidates"]] == [
        3e-5,
        1e-4,
        3e-4,
    ]


def test_lr_selection_does_not_tie_at_exactly_one_percent(tmp_path) -> None:
    low = _pilot(tmp_path, learning_rate=3e-5, loss=1.01)
    middle = _pilot(tmp_path, learning_rate=1e-4, loss=1.0)
    high = _pilot(tmp_path, learning_rate=3e-4, loss=1.2)

    result = select_uncond_bc_learning_rate([low, middle, high])

    assert result["selected_learning_rate"] == 1e-4


def test_lr_selection_fails_closed_on_incomplete_validation(tmp_path) -> None:
    roots = [
        _pilot(tmp_path, learning_rate=learning_rate, loss=loss)
        for learning_rate, loss in ((3e-5, 1.0), (1e-4, 0.9), (3e-4, 0.8))
    ]
    metrics = roots[0] / "metrics.jsonl"
    record = json.loads(metrics.read_text())
    record["validation"]["sample_count"] -= 1
    metrics.write_text(json.dumps(record) + "\n")

    with pytest.raises(ValueError, match="all held-out windows"):
        select_uncond_bc_learning_rate(roots)


def test_lr_selection_rejects_non_lr_config_drift(tmp_path) -> None:
    roots = [
        _pilot(tmp_path, learning_rate=learning_rate, loss=loss)
        for learning_rate, loss in ((3e-5, 1.0), (1e-4, 0.9), (3e-4, 0.8))
    ]
    path = roots[-1] / "run_manifest.json"
    payload = json.loads(path.read_text())
    payload["provenance"]["resolved_config"]["training"]["max_steps"] = 999
    path.write_text(json.dumps(payload) + "\n")

    with pytest.raises(ValueError, match="differ outside"):
        select_uncond_bc_learning_rate(roots)


def test_lr_selection_manifest_is_atomic_and_non_overwriting(tmp_path) -> None:
    roots = [
        _pilot(tmp_path, learning_rate=learning_rate, loss=loss)
        for learning_rate, loss in ((3e-5, 1.0), (1e-4, 0.9), (3e-4, 0.8))
    ]
    payload = select_uncond_bc_learning_rate(roots)
    destination = tmp_path / "selection" / "manifest.json"

    written = write_lr_selection_manifest(destination, payload)

    assert json.loads(written.read_text()) == payload
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_lr_selection_manifest(destination, payload)
