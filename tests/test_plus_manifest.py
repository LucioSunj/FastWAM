from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from fastwam.adaptive_gate.plus_manifest import load_plus_manifest


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_plus_manifest_resolves_and_hashes_bddl(tmp_path):
    bddl = tmp_path / "task.bddl"
    bddl.write_text("problem", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "libero-plus-episode-manifest-v1",
                "split": "test",
                "libero_plus_commit": "abc",
                "episodes": [
                    {
                        "episode_id": "episode-0",
                        "task_suite_name": "libero_10",
                        "base_task": "put cup on plate",
                        "task_id": 0,
                        "factor": "layout",
                        "level": "L3",
                        "bddl_path": "task.bddl",
                        "bddl_sha256": _sha(bddl),
                        "reset_state_id": 2,
                        "trial_id": 4,
                        "env_seed": 7,
                        "perturbation_id": "layout-3",
                        "asset_ids": ["cup", "plate"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    loaded = load_plus_manifest(
        manifest,
        libero_plus_root=tmp_path,
        libero_plus_commit="abc",
        verify_git=False,
    )
    assert loaded.episodes[0].bddl_path == str(bddl.resolve())
    assert loaded.episodes[0].asset_ids == ("cup", "plate")


def test_load_plus_manifest_fails_closed_on_split_or_hash(tmp_path):
    bddl = tmp_path / "task.bddl"
    bddl.write_text("problem", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "libero-plus-episode-manifest-v1",
                "split": "train",
                "libero_plus_commit": "abc",
                "episodes": [
                    {
                        "episode_id": "episode-0",
                        "task_suite_name": "libero_10",
                        "base_task": "task",
                        "task_id": 0,
                        "factor": "layout",
                        "level": "L1",
                        "bddl_path": "task.bddl",
                        "bddl_sha256": "0" * 64,
                        "reset_state_id": 0,
                        "trial_id": 0,
                        "env_seed": 0,
                        "perturbation_id": "p0",
                        "asset_ids": ["cup"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected split"):
        load_plus_manifest(
            manifest,
            libero_plus_root=tmp_path,
            libero_plus_commit="abc",
            verify_git=False,
        )


def test_endpoint_router_uses_official_libero_namespace(tmp_path):
    package = tmp_path / "libero"
    core = package / "libero"
    core.mkdir(parents=True)
    for path in (
        package / "__init__.py",
        core / "__init__.py",
        core / "benchmark.py",
        core / "envs.py",
    ):
        path.write_text("", encoding="utf-8")
    script = Path(__file__).parents[1] / "scripts/evaluate_libero_plus_manifest.py"
    spec = importlib.util.spec_from_file_location("plus_endpoint_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    saved_modules = {
        name: value for name, value in sys.modules.items() if name == "libero" or name.startswith("libero.")
    }
    for name in list(saved_modules):
        sys.modules.pop(name, None)
    saved_path = list(sys.path)
    try:
        benchmark, envs = module._route_libero_plus(tmp_path)
        assert Path(benchmark.__file__).resolve().is_relative_to(tmp_path)
        assert Path(envs.__file__).resolve().is_relative_to(tmp_path)
    finally:
        sys.path[:] = saved_path
        for name in list(sys.modules):
            if name == "libero" or name.startswith("libero."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
