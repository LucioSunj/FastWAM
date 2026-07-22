import copy
import hashlib

import pytest
import torch

from fastwam.adaptive_gate.sdr_contracts import (
    PREFLIGHT_DECISION_SCHEMA,
    audit_e_i_lineage_inputs,
    validate_learning_probe_contract,
)
from fastwam.adaptive_gate.sdr_delta import reconstruct_action_dit_delta


def _bindings():
    return {
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


def _preflight():
    return {
        "schema": PREFLIGHT_DECISION_SCHEMA,
        "status": "PASS",
        "plus_full_used": False,
        "artifact_bindings": _bindings(),
        "w0": 0.05,
        "w_cap": 0.5,
        "allowed_schedule": [
            [0.0, 0.05],
            [0.1, 0.05],
            [0.3, 0.2],
            [0.6, 0.5],
            [1.0, 0.5],
        ],
    }


def test_learning_probe_contract_accepts_locked_schedule_and_bindings():
    result = validate_learning_probe_contract(_preflight())

    assert result["w0"] == 0.05
    assert result["w_cap"] == 0.5
    assert result["artifact_bindings"]["e_i_checkpoint"] == "b" * 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "NOT-RUN"),
        ("plus_full_used", True),
        ("w0", 0.06),
        ("w_cap", 1.0),
        ("allowed_schedule", [[0.0, 0.05], [1.0, 0.5]]),
    ],
)
def test_learning_probe_contract_fails_closed(field, value):
    decision = copy.deepcopy(_preflight())
    decision[field] = value

    with pytest.raises(ValueError):
        validate_learning_probe_contract(decision)


def test_lineage_audit_is_not_run_when_training_artifacts_are_missing(tmp_path):
    checkpoint = tmp_path / "e_i.pt"
    checkpoint.write_bytes(b"checkpoint")

    result = audit_e_i_lineage_inputs(
        base_model_manifest=None,
        e_i_checkpoint=checkpoint,
        e_i_config=None,
        dataset_stats=None,
        lineage_manifest=None,
    )

    assert result["status"] == "NOT-RUN"
    assert result["blockers"]
    assert result["claim"] == "No training update is authorized by this audit."


def test_action_dit_delta_reconstructs_only_parent_action_state(tmp_path):
    parent_path = tmp_path / "parent.pt"
    parent = {
        "mot": {
            "mixtures.video.weight": torch.tensor([1.0]),
            "mixtures.action.weight": torch.tensor([2.0]),
            "mixtures.action.bias": torch.tensor([3.0]),
        },
        "proprio_encoder": {"weight": torch.tensor([4.0])},
        "step": 10,
    }
    torch.save(parent, parent_path)
    parent_sha = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    delta_path = tmp_path / "delta.pt"
    provenance = {"parent_checkpoint_sha256": parent_sha}
    torch.save(
        {
            "schema": "fastwam-action-dit-delta-v1",
            "step": 50,
            "torch_dtype": "torch.bfloat16",
            "parent_checkpoint_sha256": parent_sha,
            "action_expert": {
                "weight": torch.tensor([20.0]),
                "bias": torch.tensor([30.0]),
            },
            "fastwam_provenance": provenance,
        },
        delta_path,
    )
    output = tmp_path / "rebuilt.pt"

    result = reconstruct_action_dit_delta(
        parent_checkpoint=parent_path,
        delta_checkpoint=delta_path,
        output_checkpoint=output,
    )
    rebuilt = torch.load(output, weights_only=False)

    assert result["status"] == "PASS"
    torch.testing.assert_close(
        rebuilt["mot"]["mixtures.video.weight"],
        parent["mot"]["mixtures.video.weight"],
    )
    torch.testing.assert_close(
        rebuilt["mot"]["mixtures.action.weight"],
        torch.tensor([20.0]),
    )
    torch.testing.assert_close(
        rebuilt["proprio_encoder"]["weight"],
        parent["proprio_encoder"]["weight"],
    )


def test_action_dit_delta_rejects_parent_drift(tmp_path):
    parent_path = tmp_path / "parent.pt"
    torch.save(
        {"mot": {"mixtures.action.weight": torch.tensor([1.0])}},
        parent_path,
    )
    delta_path = tmp_path / "delta.pt"
    torch.save(
        {
            "schema": "fastwam-action-dit-delta-v1",
            "parent_checkpoint_sha256": "0" * 64,
            "action_expert": {"weight": torch.tensor([2.0])},
            "fastwam_provenance": {
                "parent_checkpoint_sha256": "0" * 64,
            },
        },
        delta_path,
    )

    with pytest.raises(ValueError, match="parent checkpoint"):
        reconstruct_action_dit_delta(
            parent_checkpoint=parent_path,
            delta_checkpoint=delta_path,
            output_checkpoint=tmp_path / "rebuilt.pt",
        )
