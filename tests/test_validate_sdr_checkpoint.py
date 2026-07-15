"""Completed S-DR and two-LR selection contracts."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_sdr_checkpoint.py"
    spec = importlib.util.spec_from_file_location("validate_sdr_checkpoint", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


validator = _module()


def _candidate(tmp_path: Path, *, lr: float, steps: int = 10, name: str = "a") -> Path:
    stats = tmp_path / f"stats_{name}.json"
    stats.write_text("{}\n", encoding="utf-8")
    stats_sha = validator.sha256_file(stats)
    config = tmp_path / f"config_{name}.yaml"
    config.write_text(yaml.safe_dump({"learning_rate": lr}), encoding="utf-8")
    checkpoint = tmp_path / f"checkpoint_{name}.pt"
    contract = {
        "base_learning_rate": lr,
        "total_optimizer_steps": steps,
        "uncond_weight_schedule": [
            [0.0, 0.05],
            [0.1, 0.05],
            [0.4, 0.5],
            [1.0, 1.0],
        ],
    }
    torch.save(
        {
            "mot": {"weight": torch.ones(2)},
            "step": steps,
            "fastwam_provenance": {
                "schema_version": 2,
                "checkpoint_id": f"checkpoint-{name}",
                "adaptive_regimes": ["uncond", "idm"],
                "adaptive_backbone_kind": "idm",
                "initialization_type": "standalone_idm",
                "dual_regime_optimizer_steps": steps,
                "action_regime_weight_uncond": 1.0,
                "dataset_stats_fingerprint": stats_sha,
                "parent_checkpoint_sha256": "a" * 64,
                "parent_config_sha256": "b" * 64,
                "parent_dataset_stats_sha256": stats_sha,
                "schedule_fingerprint": validator.dual_regime_schedule_fingerprint(
                    contract
                ),
                "dual_regime_training_contract": contract,
            },
        },
        checkpoint,
    )
    out = tmp_path / f"completion_{name}.json"
    validator.validate(
        SimpleNamespace(
            checkpoint=str(checkpoint),
            resolved_config=str(config),
            dataset_stats=str(stats),
            expected_base_lr=lr,
            out=str(out),
        )
    )
    return out


def test_completed_checkpoint_requires_full_schedule(tmp_path):
    completion = _candidate(tmp_path, lr=1e-5)
    assert json.loads(completion.read_text())["status"] == "PASS"

    payload = torch.load(tmp_path / "checkpoint_a.pt", weights_only=False)
    payload["fastwam_provenance"]["dual_regime_optimizer_steps"] = 9
    torch.save(payload, tmp_path / "checkpoint_a.pt")
    with pytest.raises(ValueError, match="did not complete"):
        validator.validate(
            SimpleNamespace(
                checkpoint=str(tmp_path / "checkpoint_a.pt"),
                resolved_config=str(tmp_path / "config_a.yaml"),
                dataset_stats=str(tmp_path / "stats_a.json"),
                expected_base_lr=1e-5,
                out=str(tmp_path / "bad.json"),
            )
        )


@pytest.mark.parametrize("mutation", ["anchors", "fingerprint"])
def test_completed_checkpoint_rejects_noncanonical_schedule_provenance(
    tmp_path, mutation
):
    _candidate(tmp_path, lr=1e-5)
    checkpoint = tmp_path / "checkpoint_a.pt"
    payload = torch.load(checkpoint, weights_only=False)
    provenance = payload["fastwam_provenance"]
    if mutation == "anchors":
        provenance["dual_regime_training_contract"]["uncond_weight_schedule"][1] = [
            0.2,
            0.05,
        ]
        provenance["schedule_fingerprint"] = validator.dual_regime_schedule_fingerprint(
            provenance["dual_regime_training_contract"]
        )
    else:
        provenance["schedule_fingerprint"] = "c" * 64
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="schedule|fingerprint"):
        validator.validate(
            SimpleNamespace(
                checkpoint=str(checkpoint),
                resolved_config=str(tmp_path / "config_a.yaml"),
                dataset_stats=str(tmp_path / "stats_a.json"),
                expected_base_lr=1e-5,
                out=str(tmp_path / "bad_schedule.json"),
            )
        )


def test_two_lr_selection_binds_exact_checkpoint(tmp_path):
    first = _candidate(tmp_path, lr=1e-5, name="one")
    second = _candidate(tmp_path, lr=3e-5, name="three")
    selection = tmp_path / "selection.json"
    validator.select(
        SimpleNamespace(
            candidate_validation=[str(first), str(second)],
            selected_lr=1e-5,
            selection_basis="lower validation loss and stable gradients",
            out=str(selection),
        )
    )
    chosen = json.loads(first.read_text())["checkpoint"]
    validator.check(SimpleNamespace(selection=str(selection), checkpoint=chosen))
    with pytest.raises(ValueError, match="not the selected"):
        validator.check(
            SimpleNamespace(
                selection=str(selection),
                checkpoint=json.loads(second.read_text())["checkpoint"],
            )
        )


def test_selection_rechecks_selected_checkpoint_schedule(tmp_path):
    first = _candidate(tmp_path, lr=1e-5, name="one")
    second = _candidate(tmp_path, lr=3e-5, name="three")
    selection = tmp_path / "selection.json"
    validator.select(
        SimpleNamespace(
            candidate_validation=[str(first), str(second)],
            selected_lr=1e-5,
            selection_basis="stable validation",
            out=str(selection),
        )
    )
    checkpoint = tmp_path / "checkpoint_one.pt"
    payload = torch.load(checkpoint, weights_only=False)
    payload["fastwam_provenance"]["schedule_fingerprint"] = "d" * 64
    torch.save(payload, checkpoint)
    selection_payload = json.loads(selection.read_text())
    replacement_sha = validator.sha256_file(checkpoint)
    selection_payload["selected_checkpoint_sha256"] = replacement_sha
    for candidate in selection_payload["candidates"]:
        if candidate["base_learning_rate"] == 1e-5:
            candidate["checkpoint_sha256"] = replacement_sha
    selection.write_text(json.dumps(selection_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schedule_fingerprint"):
        validator.check(
            SimpleNamespace(selection=str(selection), checkpoint=str(checkpoint))
        )
