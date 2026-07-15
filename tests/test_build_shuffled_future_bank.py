"""Pure-CPU provenance tests for the shuffled-future bank builder."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
yaml = pytest.importorskip("yaml")


def _load_builder():
    source = Path(__file__).resolve().parents[1] / "scripts" / "build_shuffled_future_bank.py"
    spec = importlib.util.spec_from_file_location("build_shuffled_future_bank", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


def _artifacts(tmp_path: Path, *, compute_matched: bool = True):
    stats = tmp_path / "dataset_stats.json"
    stats.write_text('{"mean": 0}\n', encoding="utf-8")
    stats_sha = hashlib.sha256(stats.read_bytes()).hexdigest()
    checkpoint_id = "shared-checkpoint-id"
    solver_fingerprint = "d" * 64
    checkpoint = tmp_path / "shared.pt"
    torch.save(
        {
            "fastwam_provenance": {
                "schema_version": 2,
                "checkpoint_id": checkpoint_id,
                "adaptive_regimes": ["uncond", "idm"],
                "adaptive_backbone_kind": "idm",
                "dual_regime_optimizer_steps": 10,
                "dataset_stats_fingerprint": stats_sha,
            }
        },
        checkpoint,
    )
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "kind": "fastwam_control_profile",
                "controls": {
                    "valid_idm": {"latency_ms": 2.0, "action_steps": 20},
                    "no_read": {
                        "latency_ms": 2.0,
                        "action_steps": 20,
                        "compute_matched": compute_matched,
                    },
                    "extra_compute": {
                        "latency_ms": 2.0,
                        "action_steps": 40,
                        "compute_matched": compute_matched,
                    },
                },
                "meta": {
                    "task": "task-config",
                    "ckpt_fingerprint": checkpoint_id,
                    "dataset_stats_fingerprint": stats_sha,
                    "inference_steps": 20,
                    "solver_fingerprint": solver_fingerprint,
                    "num_video_frames": 9,
                    "action_horizon": 32,
                },
            }
        ),
        encoding="utf-8",
    )
    return stats, checkpoint, profile, stats_sha, checkpoint_id, solver_fingerprint


def test_profile_requires_compute_matched_no_read_and_extra_compute(tmp_path):
    _stats, _checkpoint, profile, _stats_sha, _checkpoint_id, _solver = _artifacts(
        tmp_path, compute_matched=False
    )
    with pytest.raises(ValueError, match="unmatched"):
        builder._load_profile(profile)


def test_lineage_binds_capture_to_profile_checkpoint_and_stats(tmp_path):
    stats, checkpoint, profile, stats_sha, checkpoint_id, solver_fingerprint = _artifacts(tmp_path)
    loaded_profile = builder._load_profile(profile)
    metadata = {
        "ckpt_fingerprint": checkpoint_id,
        "dataset_stats_fingerprint": stats_sha,
        "solver_steps": 20,
        "solver_fingerprint": solver_fingerprint,
        "num_video_frames": 9,
        "action_horizon": 32,
        "wam_task": "task-config",
        "control_profile_sha256": builder._sha256_file(profile),
    }
    builder._validate_lineage(
        torch=torch,
        global_meta=metadata,
        profile=loaded_profile,
        profile_path=str(profile),
        shared_ckpt=str(checkpoint),
        dataset_stats=str(stats),
    )
    with pytest.raises(ValueError, match="lineage"):
        builder._validate_lineage(
            torch=torch,
            global_meta={**metadata, "solver_steps": 19},
            profile=loaded_profile,
            profile_path=str(profile),
            shared_ckpt=str(checkpoint),
            dataset_stats=str(stats),
        )
