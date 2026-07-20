"""Fail-closed standalone-IDM -> S0 parity evidence tests."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "verify_dual_regime_warm_start.py"
    )
    spec = importlib.util.spec_from_file_location(
        "verify_dual_regime_warm_start", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verifier = _module()


def _metadata() -> dict:
    return {
        "source_task": "libero_idm_2cam224_1e-4",
        "target_task": "libero_dual_regime_fused_2cam224_1e-4",
        "artifacts": {
            "e_i_checkpoint": {"path": "/inputs/e_i.pt", "sha256": "a" * 64},
            "e_i_config": {"path": "/inputs/config.yaml", "sha256": "b" * 64},
            "e_i_stats": {"path": "/inputs/stats.json", "sha256": "c" * 64},
        },
        "sample_index": 0,
        "seed": 0,
        "model_dtype": "bfloat16",
        "inference_steps": 20,
        "sigma_shift": 5.0,
        "device": {"requested": "cuda", "name": "NVIDIA A800-SXM4-80GB"},
        "solver_contract": {
            "schema": "fastwam-inference-solver-v1",
            "video": {"inference_steps": 20},
            "action": {"inference_steps": 20},
        },
        "solver_contract_sha256": "d" * 64,
    }


def test_cli_requires_output_path():
    required = [
        "--source-config",
        "config.yaml",
        "--source-task",
        "source",
        "--target-task",
        "target",
        "--ckpt",
        "checkpoint.pt",
        "--checkpoint-sha256",
        "a" * 64,
        "--dataset-stats",
        "stats.json",
    ]
    with pytest.raises(SystemExit):
        verifier.build_parser().parse_args(required)


def test_success_atomically_writes_complete_parity_schema(tmp_path):
    output = tmp_path / "parity_result.json"
    source = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
    target = torch.tensor([[1.0001, -2.0001]], dtype=torch.float32)

    result = verifier.assert_parity_and_write_result(
        source_action=source,
        target_action=target,
        output_path=output,
        metadata=_metadata(),
        atol=5e-4,
        rtol=5e-3,
    )

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted == result
    assert persisted["schema"] == "fastwam-warmstart-parity-v1"
    assert persisted["kind"] == "standalone_idm_to_s0_fixed_seed_parity"
    assert persisted["status"] == "PASS"
    assert persisted["source_task"] == "libero_idm_2cam224_1e-4"
    assert persisted["target_task"] == "libero_dual_regime_fused_2cam224_1e-4"
    assert persisted["comparison"]["worst_normalized_error"] <= 1.0
    assert persisted["comparison"]["allclose_margin"] >= 0.0
    assert len(persisted["actions"]["source_sha256"]) == 64
    assert len(persisted["actions"]["target_sha256"]) == 64
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_parity_does_not_create_result(tmp_path):
    output = tmp_path / "parity_result.json"
    source = torch.zeros(2, dtype=torch.float32)
    target = torch.ones(2, dtype=torch.float32)

    with pytest.raises(AssertionError):
        verifier.assert_parity_and_write_result(
            source_action=source,
            target_action=target,
            output_path=output,
            metadata=_metadata(),
            atol=5e-4,
            rtol=5e-3,
        )

    assert not output.exists()
    assert not list(tmp_path.glob("*.tmp"))
