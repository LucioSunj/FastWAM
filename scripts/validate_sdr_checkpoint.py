#!/usr/bin/env python3
"""Validate completed S-DR checkpoints and register the two-LR pilot choice."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

from fastwam.adaptive_gate.provenance import dual_regime_schedule_fingerprint


SCHEMA = "fastwam-sdr-completion-v1"
SELECTION_SCHEMA = "fastwam-sdr-pilot-selection-v1"
CANONICAL_UNCOND_WEIGHT_SCHEDULE = (
    (0.0, 0.05),
    (0.1, 0.05),
    (0.4, 0.5),
    (1.0, 1.0),
)


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def read_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _validate_schedule_provenance(provenance: Mapping[str, Any]) -> Mapping[str, Any]:
    contract = provenance.get("dual_regime_training_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("S-DR checkpoint lacks its training contract")
    raw_schedule = contract.get("uncond_weight_schedule")
    try:
        schedule = tuple(tuple(float(value) for value in point) for point in raw_schedule)
    except (TypeError, ValueError) as exc:
        raise ValueError("S-DR checkpoint has an invalid UNCOND weight schedule") from exc
    if schedule != CANONICAL_UNCOND_WEIGHT_SCHEDULE:
        raise ValueError(
            "S-DR checkpoint does not use the preregistered UNCOND schedule: "
            f"expected={CANONICAL_UNCOND_WEIGHT_SCHEDULE}, observed={schedule}"
        )
    expected_fingerprint = dual_regime_schedule_fingerprint(contract)
    if provenance.get("schedule_fingerprint") != expected_fingerprint:
        raise ValueError(
            "S-DR schedule_fingerprint does not match its training contract"
        )
    return contract


def validate(args: argparse.Namespace) -> None:
    import torch
    import yaml

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    config_path = Path(args.resolved_config).expanduser().resolve()
    stats_path = Path(args.dataset_stats).expanduser().resolve()
    for path in (checkpoint_path, config_path, stats_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("S-DR checkpoint must be a mapping")
    provenance = payload.get("fastwam_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("S-DR checkpoint lacks fastwam_provenance")
    if provenance.get("schema_version") != 2:
        raise ValueError("S-DR checkpoint provenance must be schema_version=2")
    if not isinstance(provenance.get("checkpoint_id"), str) or not provenance.get(
        "checkpoint_id"
    ):
        raise ValueError("S-DR checkpoint provenance has no checkpoint_id")
    if list(provenance.get("adaptive_regimes", ())) != ["uncond", "idm"]:
        raise ValueError("S-DR checkpoint must contain exactly UNCOND and IDM regimes")
    if provenance.get("adaptive_backbone_kind") != "idm":
        raise ValueError("S-DR checkpoint must use the IDM backbone")
    if provenance.get("initialization_type") != "standalone_idm":
        raise ValueError("S-DR checkpoint was not initialized from standalone E-I")
    contract = _validate_schedule_provenance(provenance)
    dual_steps = provenance.get("dual_regime_optimizer_steps")
    total_steps = contract.get("total_optimizer_steps")
    if (
        isinstance(dual_steps, bool)
        or not isinstance(dual_steps, int)
        or isinstance(total_steps, bool)
        or not isinstance(total_steps, int)
        or dual_steps <= 0
        or dual_steps != total_steps
    ):
        raise ValueError(
            "S-DR did not complete its optimizer-step schedule: "
            f"dual={dual_steps!r}, contracted={total_steps!r}"
        )
    final_weight = float(provenance.get("action_regime_weight_uncond", float("nan")))
    if not math.isfinite(final_weight) or abs(final_weight - 1.0) > 1e-9:
        raise ValueError(f"final S-DR UNCOND loss weight is not 1.0: {final_weight}")
    stats_sha = sha256_file(stats_path)
    if provenance.get("dataset_stats_fingerprint") != stats_sha:
        raise ValueError("S-DR checkpoint and dataset_stats.json do not match")
    for key in (
        "parent_checkpoint_sha256",
        "parent_config_sha256",
        "parent_dataset_stats_sha256",
    ):
        if not _is_sha256(provenance.get(key)):
            raise ValueError(f"S-DR provenance has invalid {key}")
    if provenance["parent_dataset_stats_sha256"] != stats_sha:
        raise ValueError("S-DR parent and shared stats lineage differ")
    for state_key in ("mot", "proprio_encoder"):
        state = payload.get(state_key)
        if state_key == "proprio_encoder" and state is None:
            continue
        if not isinstance(state, Mapping) or not state:
            raise ValueError(f"checkpoint has no non-empty {state_key} state")
        nonfinite = [
            name
            for name, tensor in state.items()
            if not torch.is_tensor(tensor) or not bool(torch.isfinite(tensor).all())
        ]
        if nonfinite:
            raise ValueError(f"checkpoint {state_key} contains invalid/non-finite tensors: {nonfinite[:5]}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("resolved S-DR config must be a mapping")
    config_lr = float(config.get("learning_rate"))
    contract_lr = float(contract.get("base_learning_rate"))
    expected_lr = float(args.expected_base_lr)
    if config_lr != expected_lr or contract_lr != expected_lr:
        raise ValueError(
            "S-DR learning-rate lineage mismatch: "
            f"config={config_lr}, contract={contract_lr}, expected={expected_lr}"
        )
    result = {
        "schema": SCHEMA,
        "status": "PASS",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_id": str(provenance.get("checkpoint_id")),
        "checkpoint_step": int(payload.get("step")),
        "dual_regime_optimizer_steps": dual_steps,
        "total_optimizer_steps": total_steps,
        "base_learning_rate": expected_lr,
        "resolved_config": str(config_path),
        "resolved_config_sha256": sha256_file(config_path),
        "dataset_stats": str(stats_path),
        "dataset_stats_sha256": stats_sha,
    }
    atomic_json(args.out, result)


def select(args: argparse.Namespace) -> None:
    candidates = [read_json(path) for path in args.candidate_validation]
    if len(candidates) != 2:
        raise ValueError("E1 LR comparison requires exactly two candidate validations")
    expected_lrs = {1e-5, 3e-5}
    by_lr: dict[float, dict[str, Any]] = {}
    for candidate in candidates:
        if candidate.get("schema") != SCHEMA or candidate.get("status") != "PASS":
            raise ValueError("every S-DR pilot candidate must have PASS completion evidence")
        lr = float(candidate["base_learning_rate"])
        if lr in by_lr:
            raise ValueError(f"duplicate S-DR pilot learning rate {lr}")
        checkpoint = Path(candidate["checkpoint"]).expanduser().resolve()
        if not checkpoint.is_file() or sha256_file(checkpoint) != candidate.get("checkpoint_sha256"):
            raise ValueError(f"S-DR pilot checkpoint changed after validation: {checkpoint}")
        by_lr[lr] = candidate
    if set(by_lr) != expected_lrs:
        raise ValueError(f"E1 pilots must be exactly {sorted(expected_lrs)}, got {sorted(by_lr)}")
    selected_lr = float(args.selected_lr)
    if selected_lr not in by_lr:
        raise ValueError(f"selected LR {selected_lr} is not a validated candidate")
    selected = by_lr[selected_lr]
    result = {
        "schema": SELECTION_SCHEMA,
        "status": "PASS",
        "selection_basis": str(args.selection_basis),
        "selected_base_learning_rate": selected_lr,
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_checkpoint_step": selected["checkpoint_step"],
        "candidates": [by_lr[lr] for lr in sorted(by_lr)],
    }
    atomic_json(args.out, result)


def check(args: argparse.Namespace) -> None:
    selection = read_json(args.selection)
    if selection.get("schema") != SELECTION_SCHEMA or selection.get("status") != "PASS":
        raise ValueError("shared checkpoint requires a PASS two-LR pilot selection")
    if not str(selection.get("selection_basis", "")).strip():
        raise ValueError("shared checkpoint selection has no auditable selection basis")
    candidates = selection.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError("shared checkpoint selection must contain both pilot candidates")
    selected_lr = float(selection.get("selected_base_learning_rate"))
    selected_candidates = [
        candidate
        for candidate in candidates
        if isinstance(candidate, Mapping)
        and float(candidate.get("base_learning_rate", float("nan"))) == selected_lr
    ]
    if len(selected_candidates) != 1:
        raise ValueError("shared checkpoint selection is internally inconsistent")
    selected_candidate = selected_candidates[0]
    if (
        selected_candidate.get("checkpoint") != selection.get("selected_checkpoint")
        or selected_candidate.get("checkpoint_sha256")
        != selection.get("selected_checkpoint_sha256")
        or selected_candidate.get("checkpoint_step")
        != selection.get("selected_checkpoint_step")
    ):
        raise ValueError("shared checkpoint selection disagrees with its pilot evidence")
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual = sha256_file(checkpoint)
    expected = selection.get("selected_checkpoint_sha256")
    if actual != expected or checkpoint != Path(selection["selected_checkpoint"]).resolve():
        raise ValueError(
            "configured SHARED_CKPT is not the selected E1 pilot: "
            f"path={checkpoint}, sha={actual}, expected={selection.get('selected_checkpoint')}, {expected}"
        )
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("selected S-DR checkpoint must be a mapping")
    provenance = payload.get("fastwam_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("selected S-DR checkpoint lacks fastwam_provenance")
    _validate_schedule_provenance(provenance)
    print(json.dumps({"status": "PASS", "checkpoint_sha256": actual}, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    validate_parser = sub.add_parser("validate")
    validate_parser.add_argument("--checkpoint", required=True)
    validate_parser.add_argument("--resolved-config", required=True)
    validate_parser.add_argument("--dataset-stats", required=True)
    validate_parser.add_argument("--expected-base-lr", type=float, required=True)
    validate_parser.add_argument("--out", required=True)
    validate_parser.set_defaults(func=validate)
    select_parser = sub.add_parser("select")
    select_parser.add_argument("--candidate-validation", action="append", required=True)
    select_parser.add_argument("--selected-lr", type=float, required=True)
    select_parser.add_argument("--selection-basis", required=True)
    select_parser.add_argument("--out", required=True)
    select_parser.set_defaults(func=select)
    check_parser = sub.add_parser("check-selection")
    check_parser.add_argument("--selection", required=True)
    check_parser.add_argument("--checkpoint", required=True)
    check_parser.set_defaults(func=check)
    return ap


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
