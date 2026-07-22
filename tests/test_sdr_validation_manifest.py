"""Fixed S-DR validation manifests must be reproducible and episode-disjoint."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastwam.adaptive_gate.sdr_validation import (
    STANDARD_LIBERO_SUITES,
    build_validation_manifest,
    episodes_for_manifest_split,
    validate_validation_manifest,
)


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _dataset_tree(tmp_path: Path) -> tuple[list[Path], Path]:
    dataset_dirs = []
    for suite_index, suite in enumerate(STANDARD_LIBERO_SUITES):
        root = tmp_path / f"{suite}_no_noops_lerobot"
        meta = root / "meta"
        meta.mkdir(parents=True)
        tasks = [
            {"task_index": task_index, "task": f"{suite} task {task_index}"}
            for task_index in range(2)
        ]
        episodes = []
        for episode_index in range(6):
            task_index = episode_index % 2
            episodes.append(
                {
                    "episode_index": episode_index,
                    "tasks": [tasks[task_index]["task"]],
                    "length": 40 + suite_index + episode_index,
                }
            )
        total_frames = sum(row["length"] for row in episodes)
        (meta / "info.json").write_text(
            json.dumps(
                {
                    "total_episodes": len(episodes),
                    "total_frames": total_frames,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        _write_jsonl(meta / "tasks.jsonl", tasks)
        _write_jsonl(meta / "episodes.jsonl", episodes)
        dataset_dirs.append(root)
    stats = tmp_path / "dataset_stats.json"
    stats.write_text('{"bound": "E-I"}\n', encoding="utf-8")
    return dataset_dirs, stats


def test_manifest_is_reproducible_balanced_and_episode_disjoint(tmp_path):
    dataset_dirs, stats = _dataset_tree(tmp_path)
    kwargs = {
        "dataset_dirs": dataset_dirs,
        "dataset_stats": stats,
        "sample_count": 8,
        "seed": 20260721,
        "num_frames": 33,
        "require_standard_size": False,
    }
    first = build_validation_manifest(
        **kwargs, generated_at_utc="2026-07-21T00:00:00+00:00"
    )
    second = build_validation_manifest(
        **kwargs, generated_at_utc="2026-07-22T00:00:00+00:00"
    )

    assert first["selection_fingerprint"] == second["selection_fingerprint"]
    assert first["samples"] == second["samples"]
    assert first["shuffle_donors"] == second["shuffle_donors"]
    assert {row["suite"] for row in first["samples"]} == set(
        STANDARD_LIBERO_SUITES
    )
    donor_by_id = {
        row["donor_id"]: row for row in first["shuffle_donors"]
    }
    for sample in first["samples"]:
        donor = donor_by_id[sample["shuffle_donor_id"]]
        assert donor["suite"] == sample["suite"]
        assert donor["task_index"] == sample["task_index"]
        assert donor["episode_index"] != sample["episode_index"]
    validate_validation_manifest(
        first, dataset_dirs=dataset_dirs, dataset_stats=stats
    )

    train = episodes_for_manifest_split(
        dataset_dirs=dataset_dirs, manifest=first, split="train"
    )
    validation = episodes_for_manifest_split(
        dataset_dirs=dataset_dirs, manifest=first, split="validation"
    )
    for dataset_dir in dataset_dirs:
        key = str(dataset_dir)
        assert set(train[key]).isdisjoint(validation[key])
        assert set(train[key]) | set(validation[key]) == set(range(6))


def test_manifest_rejects_dataset_or_stats_drift(tmp_path):
    dataset_dirs, stats = _dataset_tree(tmp_path)
    manifest = build_validation_manifest(
        dataset_dirs=dataset_dirs,
        dataset_stats=stats,
        sample_count=8,
        seed=20260721,
        require_standard_size=False,
    )

    stats.write_text('{"bound": "different"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="stats changed"):
        validate_validation_manifest(
            manifest, dataset_dirs=dataset_dirs, dataset_stats=stats
        )

    stats.write_text('{"bound": "E-I"}\n', encoding="utf-8")
    info = dataset_dirs[0] / "meta/info.json"
    payload = json.loads(info.read_text(encoding="utf-8"))
    payload["metadata_revision"] = 2
    info.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint changed"):
        validate_validation_manifest(
            manifest, dataset_dirs=dataset_dirs, dataset_stats=stats
        )


def test_manifest_requires_explicit_split_pairing(tmp_path):
    dataset_dirs, stats = _dataset_tree(tmp_path)
    manifest = build_validation_manifest(
        dataset_dirs=dataset_dirs,
        dataset_stats=stats,
        sample_count=8,
        seed=20260721,
        require_standard_size=False,
    )
    with pytest.raises(ValueError, match="manifest split"):
        episodes_for_manifest_split(
            dataset_dirs=dataset_dirs, manifest=manifest, split="test"
        )
