import copy
import hashlib
import json

import pytest

from fastwam.adaptive_gate.sdr_contracts import (
    E_I_LINEAGE_SCHEMA,
    LEARNING_PROBE_DECISION_SCHEMA,
    PREFLIGHT_DECISION_SCHEMA,
    validate_e_i_lineage_manifest,
    validate_formal_training_contract,
    validate_warmstart_parity_evidence,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _lineage(tmp_path):
    artifacts = {}
    for name in (
        "wan_robot_base_checkpoint",
        "wan_robot_base_config",
        "e_i_checkpoint",
        "e_i_config",
        "dataset_stats",
    ):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        artifacts[name] = {
            "path": str(path),
            "sha256": _sha(path),
            "size_bytes": path.stat().st_size,
        }
    lineage = {
        "schema": E_I_LINEAGE_SCHEMA,
        "status": "PASS",
        "config_origin": "training_emitted",
        "plus_full_used": False,
        "artifacts": artifacts,
        "training": {
            "initializer_kind": "wan_robot_base",
            "parent_checkpoint_sha256": artifacts[
                "wan_robot_base_checkpoint"
            ]["sha256"],
            "parent_config_sha256": artifacts["wan_robot_base_config"]["sha256"],
            "output_checkpoint_sha256": artifacts["e_i_checkpoint"]["sha256"],
            "output_config_sha256": artifacts["e_i_config"]["sha256"],
            "dataset_stats_sha256": artifacts["dataset_stats"]["sha256"],
            "completed_step": 43400,
            "task": "libero_fastwam_idm",
        },
    }
    return lineage, artifacts


def _decisions(artifacts):
    bindings = {
        name: {"path": value["path"], "sha256": value["sha256"]}
        for name, value in artifacts.items()
    }
    bindings.update(
        {
            "validation_manifest": {
                "path": "/fixed/validation.json",
                "sha256": "f" * 64,
            },
            "solver_contract": {
                "path": "/fixed/solver.json",
                "sha256": "1" * 64,
            },
            "code_commit": {"path": "git", "sha256": "2" * 64},
        }
    )
    preflight = {
        "schema": PREFLIGHT_DECISION_SCHEMA,
        "status": "PASS",
        "plus_full_used": False,
        "artifact_bindings": bindings,
        "w0": 0.03,
        "w_cap": 0.25,
        "allowed_schedule": [
            [0.0, 0.03],
            [0.1, 0.03],
            [0.3, 0.2],
            [0.6, 0.25],
            [1.0, 0.25],
        ],
    }
    probe = {
        "schema": LEARNING_PROBE_DECISION_SCHEMA,
        "status": "PASS",
        "plus_full_used": False,
        "artifact_bindings": copy.deepcopy(bindings),
        "initializer_stage": "e_i_s0",
    }
    return preflight, probe


def test_formal_training_requires_same_verified_e_i_and_restarts_from_s0(tmp_path):
    lineage, artifacts = _lineage(tmp_path)
    preflight, probe = _decisions(artifacts)

    result = validate_formal_training_contract(
        preflight_decision=preflight,
        learning_probe_decision=probe,
        lineage_manifest=lineage,
    )

    assert result["base_learning_rate"] == 1e-5
    assert result["num_epochs"] == 10
    assert result["initializer_stage"] == "e_i_s0"


def test_formal_training_rejects_probe_checkpoint_continuation(tmp_path):
    lineage, artifacts = _lineage(tmp_path)
    preflight, probe = _decisions(artifacts)
    probe["initializer_stage"] = "probe_step500"

    with pytest.raises(ValueError, match="Canary/probe"):
        validate_formal_training_contract(
            preflight_decision=preflight,
            learning_probe_decision=probe,
            lineage_manifest=lineage,
        )


def test_formal_training_rejects_cross_checkpoint_decision(tmp_path):
    lineage, artifacts = _lineage(tmp_path)
    preflight, probe = _decisions(artifacts)
    probe["artifact_bindings"]["e_i_checkpoint"]["sha256"] = "9" * 64

    with pytest.raises(ValueError, match="bindings changed"):
        validate_formal_training_contract(
            preflight_decision=preflight,
            learning_probe_decision=probe,
            lineage_manifest=lineage,
        )


def test_lineage_rejects_reconstructed_config(tmp_path):
    lineage, _ = _lineage(tmp_path)
    lineage["config_origin"] = "reconstructed"

    with pytest.raises(ValueError, match="training-emitted"):
        validate_e_i_lineage_manifest(lineage)


def test_warmstart_parity_is_bound_to_exact_e_i_artifacts(tmp_path):
    _, artifacts = _lineage(tmp_path)
    parity_path = tmp_path / "parity.json"
    parity = {
        "schema": "fastwam-warmstart-parity-v1",
        "status": "PASS",
        "kind": "standalone_idm_to_s0_fixed_seed_parity",
        "inference_steps": 20,
        "solver_contract_sha256": "7" * 64,
        "actions": {"shape": [32, 7]},
        "comparison": {"worst_normalized_error": 0.0},
        "artifacts": {
            "e_i_checkpoint": artifacts["e_i_checkpoint"],
            "e_i_config": artifacts["e_i_config"],
            "e_i_stats": artifacts["dataset_stats"],
        },
    }
    parity_path.write_text(json.dumps(parity), encoding="utf-8")
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "metrics": {
                    "evidence": [
                        {
                            "path": str(parity_path),
                            "sha256": _sha(parity_path),
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    result = validate_warmstart_parity_evidence(
        decision_path,
        expected_artifacts=artifacts,
        expected_solver_fingerprint="7" * 64,
    )
    assert result["parity_result"]["sha256"] == _sha(parity_path)

    changed = copy.deepcopy(artifacts)
    changed["e_i_config"] = {
        **changed["e_i_config"],
        "sha256": "9" * 64,
    }
    with pytest.raises(ValueError, match="different E-I artifacts"):
        validate_warmstart_parity_evidence(
            decision_path,
            expected_artifacts=changed,
            expected_solver_fingerprint="7" * 64,
        )
