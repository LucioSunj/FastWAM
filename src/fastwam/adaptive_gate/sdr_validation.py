"""Episode-disjoint fixed validation manifests for S-DR experiments."""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "fastwam-sdr-validation-manifest-v1"
FINGERPRINT_SCOPE = "lerobot-info-episodes-tasks-v1"
STANDARD_LIBERO_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object.")
            rows.append(value)
    return rows


def _suite_name(dataset_dir: Path) -> str:
    name = dataset_dir.name.lower()
    for suite in STANDARD_LIBERO_SUITES:
        if name.startswith(suite):
            return suite
    raise ValueError(
        f"Cannot infer a standard LIBERO suite from dataset directory {dataset_dir}."
    )


def dataset_metadata_fingerprint(
    dataset_dirs: Sequence[str | Path],
) -> dict[str, Any]:
    entries = []
    for dataset_index, raw_dir in enumerate(dataset_dirs):
        dataset_dir = Path(raw_dir).expanduser().resolve()
        metadata = {}
        for relative in ("meta/info.json", "meta/episodes.jsonl", "meta/tasks.jsonl"):
            path = dataset_dir / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            metadata[relative] = {
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        entries.append(
            {
                "dataset_index": dataset_index,
                "dataset_path": str(dataset_dir),
                "suite": _suite_name(dataset_dir),
                "metadata": metadata,
            }
        )
    payload = {
        "scope": FINGERPRINT_SCOPE,
        "datasets": entries,
    }
    payload["sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def collect_episode_records(
    dataset_dirs: Sequence[str | Path],
) -> list[dict[str, Any]]:
    records = []
    global_dataset_offset = 0
    for dataset_index, raw_dir in enumerate(dataset_dirs):
        dataset_dir = Path(raw_dir).expanduser().resolve()
        suite = _suite_name(dataset_dir)
        info = json.loads((dataset_dir / "meta/info.json").read_text(encoding="utf-8"))
        episodes = _read_jsonl(dataset_dir / "meta/episodes.jsonl")
        tasks = _read_jsonl(dataset_dir / "meta/tasks.jsonl")
        task_by_text = {str(row["task"]): int(row["task_index"]) for row in tasks}
        episode_offset = 0
        for expected_index, episode in enumerate(episodes):
            episode_index = int(episode["episode_index"])
            if episode_index != expected_index:
                raise ValueError(
                    f"{dataset_dir} episode indices must be contiguous from zero; "
                    f"expected {expected_index}, got {episode_index}."
                )
            task_names = episode.get("tasks")
            if not isinstance(task_names, list) or len(task_names) != 1:
                raise ValueError(
                    f"{dataset_dir} episode {episode_index} must have exactly one task."
                )
            task = str(task_names[0])
            if task not in task_by_text:
                raise ValueError(
                    f"{dataset_dir} episode {episode_index} references unknown task {task!r}."
                )
            length = int(episode["length"])
            if length <= 0:
                raise ValueError(
                    f"{dataset_dir} episode {episode_index} has invalid length {length}."
                )
            records.append(
                {
                    "dataset_index": dataset_index,
                    "dataset_path": str(dataset_dir),
                    "suite": suite,
                    "task_index": task_by_text[task],
                    "task": task,
                    "episode_index": episode_index,
                    "episode_length": length,
                    "episode_start_index": global_dataset_offset + episode_offset,
                }
            )
            episode_offset += length
        expected_frames = int(info["total_frames"])
        if episode_offset != expected_frames:
            raise ValueError(
                f"{dataset_dir} episode lengths sum to {episode_offset}, "
                f"but info.json declares {expected_frames} frames."
            )
        global_dataset_offset += expected_frames
    return records


def _selection_payload(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": manifest.get("schema"),
        "seed": manifest.get("seed"),
        "num_frames": manifest.get("num_frames"),
        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
        "dataset_stats_sha256": manifest.get("dataset_stats", {}).get("sha256"),
        "samples": manifest.get("samples"),
        "shuffle_donors": manifest.get("shuffle_donors"),
        "validation_episodes": manifest.get("validation_episodes"),
    }


def build_validation_manifest(
    *,
    dataset_dirs: Sequence[str | Path],
    dataset_stats: str | Path,
    sample_count: int = 40,
    seed: int = 20260721,
    num_frames: int = 33,
    generated_at_utc: str | None = None,
    require_standard_size: bool = True,
) -> dict[str, Any]:
    sample_count = int(sample_count)
    num_frames = int(num_frames)
    if require_standard_size and not 32 <= sample_count <= 64:
        raise ValueError("S-DR validation sample_count must be in [32, 64].")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    if num_frames <= 1:
        raise ValueError("num_frames must be greater than one.")

    stats_path = Path(dataset_stats).expanduser().resolve()
    if not stats_path.is_file():
        raise FileNotFoundError(stats_path)
    fingerprint = dataset_metadata_fingerprint(dataset_dirs)
    suites = tuple(entry["suite"] for entry in fingerprint["datasets"])
    if set(suites) != set(STANDARD_LIBERO_SUITES) or len(suites) != 4:
        raise ValueError(
            "S-DR validation requires exactly the four standard LIBERO suites; "
            f"got {suites}."
        )

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in collect_episode_records(dataset_dirs):
        grouped[(record["suite"], record["task_index"])].append(record)
    if not grouped:
        raise ValueError("No episode metadata was found.")

    rng = random.Random(int(seed))
    candidates = {}
    for key in sorted(grouped):
        values = list(grouped[key])
        rng.shuffle(values)
        candidates[key] = values

    selected = []
    round_index = 0
    keys = sorted(candidates)
    while len(selected) < sample_count:
        made_progress = False
        for key in keys:
            if len(selected) >= sample_count:
                break
            if round_index >= len(candidates[key]):
                continue
            episode = candidates[key][round_index]
            max_start = max(int(episode["episode_length"]) - num_frames, 0)
            frame_offset = rng.randint(0, max_start) if max_start else 0
            sample_index = int(episode["episode_start_index"]) + frame_offset
            prompt = (
                "A video recorded from a robot's point of view executing the "
                f"following instruction: {episode['task']}"
            )
            prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            sample_id_payload = {
                "dataset_index": episode["dataset_index"],
                "episode_index": episode["episode_index"],
                "frame_offset": frame_offset,
                "sample_index": sample_index,
            }
            selected.append(
                {
                    **sample_id_payload,
                    "sample_id": hashlib.sha256(
                        _canonical_bytes(sample_id_payload)
                    ).hexdigest(),
                    "dataset_path": episode["dataset_path"],
                    "suite": episode["suite"],
                    "task_index": episode["task_index"],
                    "task": episode["task"],
                    "episode_length": episode["episode_length"],
                    "prompt_sha256": prompt_sha256,
                    "context_identity": {
                        "kind": "canonical_prompt_sha256",
                        "sha256": prompt_sha256,
                    },
                }
            )
            made_progress = True
        if not made_progress:
            raise ValueError(
                f"Only {len(selected)} episode-disjoint samples are available; "
                f"cannot select {sample_count}."
            )
        round_index += 1

    selected_episode_keys = {
        (int(sample["dataset_index"]), int(sample["episode_index"]))
        for sample in selected
    }
    shuffle_donors = []
    for sample in selected:
        key = (str(sample["suite"]), int(sample["task_index"]))
        donor_candidates = [
            record
            for record in candidates[key]
            if (
                int(record["dataset_index"]),
                int(record["episode_index"]),
            )
            not in selected_episode_keys
        ]
        if not donor_candidates:
            raise ValueError(
                "Matched shuffled-future validation requires a different "
                f"held-out episode for suite/task={key}."
            )
        donor = donor_candidates[0]
        max_start = max(int(donor["episode_length"]) - num_frames, 0)
        frame_offset = rng.randint(0, max_start) if max_start else 0
        sample_index = int(donor["episode_start_index"]) + frame_offset
        donor_id_payload = {
            "kind": "matched_shuffle_donor",
            "recipient_sample_id": sample["sample_id"],
            "dataset_index": donor["dataset_index"],
            "episode_index": donor["episode_index"],
            "frame_offset": frame_offset,
            "sample_index": sample_index,
        }
        donor_id = hashlib.sha256(_canonical_bytes(donor_id_payload)).hexdigest()
        sample["shuffle_donor_id"] = donor_id
        shuffle_donors.append(
            {
                **donor_id_payload,
                "donor_id": donor_id,
                "dataset_path": donor["dataset_path"],
                "suite": donor["suite"],
                "task_index": donor["task_index"],
                "task": donor["task"],
                "episode_length": donor["episode_length"],
            }
        )

    validation_episodes = sorted(
        {
            (
                int(record["dataset_index"]),
                str(record["dataset_path"]),
                int(record["episode_index"]),
            )
            for record in (*selected, *shuffle_donors)
        }
    )
    manifest = {
        "schema": SCHEMA,
        "generated_at_utc": generated_at_utc
        or datetime.now(timezone.utc).isoformat(),
        "seed": int(seed),
        "num_frames": num_frames,
        "sample_count": len(selected),
        "fingerprint_scope": FINGERPRINT_SCOPE,
        "dataset_fingerprint": fingerprint,
        "dataset_stats": {
            "path": str(stats_path),
            "sha256": sha256_file(stats_path),
        },
        "samples": selected,
        "shuffle_donors": shuffle_donors,
        "validation_episodes": [
            {
                "dataset_index": dataset_index,
                "dataset_path": dataset_path,
                "episode_index": episode_index,
            }
            for dataset_index, dataset_path, episode_index in validation_episodes
        ],
        "train_exclusion": "complete_sample_and_shuffle_donor_episodes",
    }
    manifest["selection_fingerprint"] = hashlib.sha256(
        _canonical_bytes(_selection_payload(manifest))
    ).hexdigest()
    return manifest


def validate_validation_manifest(
    manifest: Mapping[str, Any],
    *,
    dataset_dirs: Sequence[str | Path],
    dataset_stats: str | Path | None = None,
) -> None:
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"Unsupported S-DR validation manifest schema: {manifest.get('schema')!r}.")
    current_fingerprint = dataset_metadata_fingerprint(dataset_dirs)
    if manifest.get("dataset_fingerprint") != current_fingerprint:
        raise ValueError("S-DR validation manifest dataset fingerprint changed.")
    if dataset_stats is not None:
        stats_path = Path(dataset_stats).expanduser().resolve()
        expected = manifest.get("dataset_stats", {})
        if str(stats_path) != expected.get("path") or sha256_file(stats_path) != expected.get(
            "sha256"
        ):
            raise ValueError("S-DR validation manifest dataset stats changed.")

    samples = manifest.get("samples")
    donors = manifest.get("shuffle_donors")
    episodes = manifest.get("validation_episodes")
    if not isinstance(samples, list) or not samples:
        raise ValueError("S-DR validation manifest has no samples.")
    if int(manifest.get("sample_count", -1)) != len(samples):
        raise ValueError("S-DR validation manifest sample_count is inconsistent.")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("S-DR validation manifest has no validation episodes.")
    if not isinstance(donors, list) or len(donors) != len(samples):
        raise ValueError(
            "S-DR validation manifest requires one matched shuffle donor per sample."
        )
    episode_keys = {
        (int(row["dataset_index"]), int(row["episode_index"])) for row in episodes
    }
    sample_episode_keys = {
        (int(row["dataset_index"]), int(row["episode_index"])) for row in samples
    }
    donor_episode_keys = {
        (int(row["dataset_index"]), int(row["episode_index"])) for row in donors
    }
    if not (sample_episode_keys | donor_episode_keys).issubset(episode_keys):
        raise ValueError(
            "A validation sample/donor belongs to an unlisted validation episode."
        )
    sample_ids = [str(row["sample_id"]) for row in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("S-DR validation sample IDs must be unique.")
    donor_by_id = {str(row.get("donor_id")): row for row in donors}
    if len(donor_by_id) != len(donors):
        raise ValueError("S-DR validation shuffle donor IDs must be unique.")
    for sample in samples:
        context_identity = sample.get("context_identity")
        if (
            not isinstance(context_identity, Mapping)
            or context_identity.get("kind") != "canonical_prompt_sha256"
            or context_identity.get("sha256") != sample.get("prompt_sha256")
        ):
            raise ValueError(
                "A validation sample has no valid prompt/context identity."
            )
        donor = donor_by_id.get(str(sample.get("shuffle_donor_id")))
        if donor is None:
            raise ValueError("A validation sample has no registered shuffle donor.")
        if (
            str(sample.get("suite")),
            int(sample.get("task_index")),
        ) != (
            str(donor.get("suite")),
            int(donor.get("task_index")),
        ):
            raise ValueError("A shuffle donor is not matched to the recipient task.")
        if (
            int(sample.get("dataset_index")),
            int(sample.get("episode_index")),
        ) == (
            int(donor.get("dataset_index")),
            int(donor.get("episode_index")),
        ):
            raise ValueError("A shuffle donor must come from a different episode.")

    expected_selection = hashlib.sha256(
        _canonical_bytes(_selection_payload(manifest))
    ).hexdigest()
    if manifest.get("selection_fingerprint") != expected_selection:
        raise ValueError("S-DR validation selection_fingerprint is invalid.")


def episodes_for_manifest_split(
    *,
    dataset_dirs: Sequence[str | Path],
    manifest: Mapping[str, Any],
    split: str,
) -> dict[str, list[int]]:
    validate_validation_manifest(manifest, dataset_dirs=dataset_dirs)
    if split not in {"train", "validation"}:
        raise ValueError("manifest split must be 'train' or 'validation'.")

    selected_by_path: dict[str, set[int]] = defaultdict(set)
    for row in manifest["validation_episodes"]:
        selected_by_path[str(Path(row["dataset_path"]).resolve())].add(
            int(row["episode_index"])
        )

    output = {}
    for raw_dir in dataset_dirs:
        resolved = str(Path(raw_dir).expanduser().resolve())
        info = json.loads(
            (Path(resolved) / "meta/info.json").read_text(encoding="utf-8")
        )
        all_episodes = set(range(int(info["total_episodes"])))
        validation = selected_by_path.get(resolved, set())
        if not validation or not validation.issubset(all_episodes):
            raise ValueError(
                f"Manifest validation episodes are missing or invalid for {resolved}."
            )
        chosen = all_episodes - validation if split == "train" else validation
        if not chosen:
            raise ValueError(f"Manifest split {split!r} is empty for {resolved}.")
        output[str(raw_dir)] = sorted(chosen)
    return output


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def read_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"S-DR validation manifest must be an object: {path}.")
    return payload
