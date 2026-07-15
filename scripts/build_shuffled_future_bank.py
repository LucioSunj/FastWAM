"""Build a provenance-bound matched donor bank from pre-generated future latents.

Each input must be a schema-v1 ``valid_idm`` donor capture containing
``state_id``, ``video_latents`` and ``metadata``. Metadata must include
task/factor/level/phase plus checkpoint, dataset-stats, control-profile and
solver/seed identities. At least two distinct states are required per matched
cell. The shared checkpoint, dataset stats and compute profile are verified
before the bank is written.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
from pathlib import Path
from typing import Any, Mapping


GLOBAL_FIELDS = (
    "ckpt_fingerprint",
    "dataset_stats_fingerprint",
    "solver_steps",
    "solver_fingerprint",
    "num_video_frames",
    "action_horizon",
    "wam_seed",
    "wam_task",
    "control_profile_sha256",
)


def _resolve_inputs(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(os.path.expanduser(pattern)))
        paths.extend(matches or ([pattern] if os.path.isfile(pattern) else []))
    unique = sorted({os.path.abspath(path) for path in paths})
    if not unique:
        raise FileNotFoundError(f"No future-latent artifacts match {patterns}")
    return unique


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_profile(path: str | os.PathLike[str]) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or payload.get("kind") != "fastwam_control_profile"
    ):
        raise ValueError(f"unsupported FastWAM control profile: {path}")
    controls = payload.get("controls")
    metadata = payload.get("meta")
    if not isinstance(controls, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError(f"control profile lacks controls/meta mappings: {path}")
    for name in ("valid_idm", "no_read", "extra_compute"):
        if not isinstance(controls.get(name), Mapping):
            raise ValueError(f"control profile is missing controls.{name}")
    unmatched = [
        name
        for name in ("no_read", "extra_compute")
        if not bool(controls[name].get("compute_matched", False))
    ]
    if unmatched:
        raise ValueError(
            f"control profile marks required compute controls unmatched: {unmatched}"
        )
    return dict(payload)


def _validate_lineage(
    *,
    torch,
    global_meta: Mapping[str, Any],
    profile: Mapping[str, Any],
    profile_path: str,
    shared_ckpt: str,
    dataset_stats: str,
) -> None:
    profile_meta = profile["meta"]
    stats_sha = _sha256_file(dataset_stats)
    checkpoint = torch.load(shared_ckpt, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("shared FastWAM checkpoint must contain a mapping")
    provenance = checkpoint.get("fastwam_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("shared FastWAM checkpoint lacks strict provenance")
    if provenance.get("schema_version") != 2:
        raise ValueError("shared FastWAM checkpoint provenance must be schema_version=2")
    if list(provenance.get("adaptive_regimes", ())) != ["uncond", "idm"]:
        raise ValueError("shuffled donors require a shared UNCOND/IDM checkpoint")
    if provenance.get("adaptive_backbone_kind") != "idm":
        raise ValueError("shuffled donors require adaptive_backbone_kind='idm'")
    dual_steps = provenance.get("dual_regime_optimizer_steps")
    if isinstance(dual_steps, bool) or not isinstance(dual_steps, int) or dual_steps <= 0:
        raise ValueError("S0/untrained shared checkpoints cannot build an E2 donor bank")

    expected = {
        "ckpt_fingerprint": provenance.get("checkpoint_id"),
        "dataset_stats_fingerprint": stats_sha,
        "solver_steps": profile_meta.get("inference_steps"),
        "solver_fingerprint": profile_meta.get("solver_fingerprint"),
        "num_video_frames": profile_meta.get("num_video_frames"),
        "action_horizon": profile_meta.get("action_horizon"),
        "wam_task": profile_meta.get("task"),
        "control_profile_sha256": _sha256_file(profile_path),
    }
    mismatches = {
        key: (global_meta.get(key), value)
        for key, value in expected.items()
        if global_meta.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "valid_idm donor capture does not match the shared checkpoint/profile "
            f"lineage (actual, expected): {mismatches}"
        )
    if provenance.get("dataset_stats_fingerprint") != stats_sha:
        raise ValueError("dataset_stats.json does not match the shared checkpoint")
    profile_expected = {
        "ckpt_fingerprint": provenance.get("checkpoint_id"),
        "dataset_stats_fingerprint": stats_sha,
        "solver_fingerprint": global_meta.get("solver_fingerprint"),
    }
    profile_mismatches = {
        key: (profile_meta.get(key), value)
        for key, value in profile_expected.items()
        if profile_meta.get(key) != value
    }
    if profile_mismatches:
        raise ValueError(
            "control profile does not match shared artifacts (actual, expected): "
            f"{profile_mismatches}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True, help="paths/globs to latent .pt files")
    ap.add_argument("--profile", required=True, help="compute-matched WAM control profile")
    ap.add_argument("--shared-ckpt", required=True, help="final trained shared checkpoint")
    ap.add_argument("--dataset-stats", required=True, help="exact shared dataset_stats.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch

    from fastwam.adaptive_gate import ShuffledFutureBank, donor_cell

    profile = _load_profile(args.profile)

    records = []
    global_meta = None
    seen_state_ids: set[str] = set()
    for path in _resolve_inputs(args.inputs):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"future artifact must be a dict: {path}")
        if (
            payload.get("schema_version") != 1
            or payload.get("kind") != "fastwam_shuffled_future_donor"
        ):
            raise ValueError(
                f"future artifact is not a valid_idm donor capture: {path}"
            )
        state_id = payload.get("state_id")
        latents = payload.get("video_latents")
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"future artifact metadata must be a dict: {path}")
        if not isinstance(state_id, str) or not state_id:
            raise ValueError(f"future artifact needs a non-empty state_id: {path}")
        if state_id in seen_state_ids:
            raise ValueError(f"future artifacts duplicate state_id {state_id!r}")
        seen_state_ids.add(state_id)
        donor_cell(metadata)
        current_global = {field: metadata.get(field) for field in GLOBAL_FIELDS}
        missing = [field for field, value in current_global.items() if value in (None, "")]
        if missing:
            raise ValueError(f"future artifact {path} is missing provenance fields {missing}")
        if global_meta is None:
            global_meta = current_global
        elif current_global != global_meta:
            raise ValueError(
                f"future artifact provenance differs from the bank contract: {path}"
            )
        records.append({"state_id": state_id, "latents": latents, "metadata": metadata})

    _validate_lineage(
        torch=torch,
        global_meta=global_meta or {},
        profile=profile,
        profile_path=args.profile,
        shared_ckpt=args.shared_ckpt,
        dataset_stats=args.dataset_stats,
    )

    bank = ShuffledFutureBank(records, metadata=global_meta or {})
    cells: dict[tuple[str, str, str, str], int] = {}
    for record in records:
        cell = donor_cell(record["metadata"])
        cells[cell] = cells.get(cell, 0) + 1
    undersized = {cell: count for cell, count in cells.items() if count < 2}
    if undersized:
        raise ValueError(f"matched donor cells need at least two states: {undersized}")

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank.to_payload(), out)
    print(f"wrote {out}: {len(records)} states across {len(cells)} cells")


if __name__ == "__main__":
    main()
