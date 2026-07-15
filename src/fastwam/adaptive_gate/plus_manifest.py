"""Frozen LIBERO-Plus episode contract used by endpoint evaluation."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PLUS_MANIFEST_SCHEMA = "libero-plus-episode-manifest-v1"


def sha256_path(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PlusEpisode:
    episode_id: str
    task_suite_name: str
    base_task: str
    task_id: int
    factor: str
    level: str
    bddl_path: str
    bddl_sha256: str
    reset_state_id: int
    trial_id: int
    env_seed: int
    perturbation_id: str
    asset_ids: tuple[str, ...]


@dataclass(frozen=True)
class PlusManifest:
    path: str
    sha256: str
    split: str
    libero_plus_root: str
    libero_plus_commit: str
    episodes: tuple[PlusEpisode, ...]


def _git_head(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"LIBERO_PLUS_ROOT is not a readable git checkout: {root}") from exc
    return result.stdout.strip()


def _text(record: dict[str, Any], key: str, source: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: {key} must be a non-empty string")
    return value.strip()


def _integer(record: dict[str, Any], key: str, source: str) -> int:
    value = record.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{source}: {key} must be a non-negative integer")
    try:
        value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: {key} must be a non-negative integer") from exc
    if value < 0:
        raise ValueError(f"{source}: {key} must be a non-negative integer")
    return value


def load_plus_manifest(
    path: str | os.PathLike[str],
    *,
    libero_plus_root: str | os.PathLike[str] | None = None,
    libero_plus_commit: str | None = None,
    expected_split: str = "test",
    verify_git: bool = True,
) -> PlusManifest:
    manifest_path = Path(path).expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != PLUS_MANIFEST_SCHEMA:
        raise ValueError(
            f"{manifest_path}: expected schema={PLUS_MANIFEST_SCHEMA!r}"
        )
    split = _text(payload, "split", str(manifest_path))
    if split != expected_split:
        raise ValueError(
            f"{manifest_path}: expected split={expected_split!r}, got {split!r}"
        )
    root_value = libero_plus_root or os.environ.get("LIBERO_PLUS_ROOT")
    commit_value = libero_plus_commit or os.environ.get("LIBERO_PLUS_COMMIT")
    if not root_value or not commit_value:
        raise ValueError("LIBERO_PLUS_ROOT and LIBERO_PLUS_COMMIT are required")
    root = Path(root_value).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LIBERO_PLUS_ROOT is not a directory: {root}")
    manifest_commit = _text(payload, "libero_plus_commit", str(manifest_path))
    if str(commit_value) != manifest_commit:
        raise ValueError(
            f"LIBERO_PLUS_COMMIT={commit_value!r} does not match {manifest_commit!r}"
        )
    if verify_git and _git_head(root) != manifest_commit:
        raise ValueError("LIBERO-Plus checkout HEAD does not match the frozen manifest")

    raw_episodes = payload.get("episodes")
    if not isinstance(raw_episodes, list) or not raw_episodes:
        raise ValueError(f"{manifest_path}: episodes must be a non-empty list")
    episodes = []
    seen = set()
    for index, raw in enumerate(raw_episodes):
        source = f"{manifest_path}:episodes[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(f"{source} must be an object")
        episode_id = _text(raw, "episode_id", source)
        if episode_id in seen:
            raise ValueError(f"{source}: duplicate episode_id {episode_id!r}")
        seen.add(episode_id)
        relative = Path(_text(raw, "bddl_path", source))
        bddl = relative if relative.is_absolute() else root / relative
        bddl = bddl.resolve()
        try:
            bddl.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{source}: bddl_path escapes LIBERO_PLUS_ROOT") from exc
        expected_sha = _text(raw, "bddl_sha256", source).lower()
        if not bddl.is_file() or sha256_path(bddl) != expected_sha:
            raise ValueError(f"{source}: BDDL file/hash mismatch: {bddl}")
        assets = raw.get("asset_ids")
        if not isinstance(assets, list) or not assets or not all(
            isinstance(value, str) and value for value in assets
        ):
            raise ValueError(f"{source}: asset_ids must be a non-empty string list")
        episodes.append(
            PlusEpisode(
                episode_id=episode_id,
                task_suite_name=_text(raw, "task_suite_name", source),
                base_task=_text(raw, "base_task", source),
                task_id=_integer(raw, "task_id", source),
                factor=_text(raw, "factor", source),
                level=_text(raw, "level", source),
                bddl_path=str(bddl),
                bddl_sha256=expected_sha,
                reset_state_id=_integer(raw, "reset_state_id", source),
                trial_id=_integer(raw, "trial_id", source),
                env_seed=_integer(raw, "env_seed", source),
                perturbation_id=_text(raw, "perturbation_id", source),
                asset_ids=tuple(map(str, assets)),
            )
        )
    return PlusManifest(
        path=str(manifest_path),
        sha256=sha256_path(manifest_path),
        split=split,
        libero_plus_root=str(root),
        libero_plus_commit=manifest_commit,
        episodes=tuple(episodes),
    )
