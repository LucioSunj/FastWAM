"""Fail-closed learning-rate selection for UNCOND action-only BC pilots."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fastwam.adapters import sha256_file

LR_SELECTION_SCHEMA = "fastwam-uncond-bc-lr-selection-v1"
PILOT_LRS = (3e-5, 1e-4, 3e-4)
RELATIVE_TIE_TOLERANCE = 0.01
EXPECTED_VALIDATION_SAMPLES = 29052


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Required BC pilot artifact is missing: {path}"
        ) from None
    if not isinstance(payload, dict):
        raise TypeError(f"BC pilot artifact must be a JSON object: {path}")
    return payload


def _read_validation_record(path: Path) -> dict[str, Any]:
    matches = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"BC pilot metrics are missing: {path}") from None
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Malformed BC pilot metrics JSON at {path}:{line_number}."
            ) from error
        if isinstance(record, dict) and "validation" in record:
            matches.append(record)
    if len(matches) != 1:
        raise ValueError(
            f"Each 1000-step pilot must contain exactly one full validation record; "
            f"observed {len(matches)} in {path}."
        )
    return matches[0]


def _canonical_pilot_config(config: Mapping[str, Any]) -> dict[str, Any]:
    canonical = json.loads(json.dumps(config, sort_keys=True))
    runner = canonical.get("runner", {})
    if isinstance(runner, dict):
        runner.pop("output_dir", None)
    optimizer = canonical.get("optimizer", {})
    if isinstance(optimizer, dict):
        optimizer.pop("learning_rate", None)
    return canonical


def _candidate(
    directory: str | os.PathLike[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(directory).expanduser().resolve()
    manifest_path = root / "run_manifest.json"
    metrics_path = root / "metrics.jsonl"
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != "fastwam-uncond-bc-run-manifest-v1":
        raise ValueError(f"Unsupported BC pilot manifest schema: {manifest_path}")
    required = {
        "status": "PASS",
        "stage": "pilot",
        "optimizer_steps": 1000,
        "future_prediction_calls": 0,
        "frozen_parameter_versions_unchanged": True,
        "zero_lora_at_start": True,
    }
    mismatches = {
        key: {"expected": expected, "observed": manifest.get(key)}
        for key, expected in required.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            f"BC pilot acceptance mismatch in {manifest_path}: {mismatches}"
        )
    if int(manifest.get("nonzero_update_count", 0)) <= 0:
        raise ValueError(f"BC pilot has no nonzero LoRA update: {manifest_path}")

    provenance = manifest.get("provenance")
    contract = manifest.get("contract")
    if not isinstance(provenance, Mapping) or not isinstance(contract, Mapping):
        raise TypeError(f"BC pilot manifest lacks provenance/contract: {manifest_path}")
    config = provenance.get("resolved_config")
    if not isinstance(config, Mapping):
        raise TypeError(f"BC pilot manifest lacks resolved config: {manifest_path}")
    learning_rate = float(config.get("optimizer", {}).get("learning_rate", math.nan))
    if not math.isfinite(learning_rate) or learning_rate not in PILOT_LRS:
        raise ValueError(
            f"Unexpected BC pilot learning rate {learning_rate!r}: {manifest_path}"
        )

    validation_record = _read_validation_record(metrics_path)
    if validation_record.get("global_step") != 1000:
        raise ValueError(
            f"BC pilot validation did not occur at step 1000: {metrics_path}"
        )
    validation = validation_record.get("validation")
    if not isinstance(validation, Mapping):
        raise TypeError(f"BC pilot validation payload is malformed: {metrics_path}")
    sample_count = validation.get("sample_count")
    if sample_count != EXPECTED_VALIDATION_SAMPLES:
        raise ValueError(
            "BC pilot validation must cover all held-out windows exactly once: "
            f"expected {EXPECTED_VALIDATION_SAMPLES}, got {sample_count}."
        )
    validation_loss = float(validation.get("loss_action_bc", math.nan))
    best_loss = float(manifest.get("best_validation_loss_action_bc", math.nan))
    if not math.isfinite(validation_loss) or validation_loss < 0:
        raise ValueError(f"BC pilot validation loss is invalid: {validation_loss!r}")
    if validation_loss != best_loss:
        raise ValueError(
            f"BC pilot manifest/metrics validation loss mismatch: {best_loss} != "
            f"{validation_loss}."
        )

    sidecar = manifest.get("best_sidecar")
    if not isinstance(sidecar, Mapping):
        raise TypeError(f"BC pilot lacks best-sidecar report: {manifest_path}")
    sidecar_path = Path(str(sidecar.get("path", ""))).expanduser().resolve()
    sidecar_sha256 = sha256_file(sidecar_path)
    sidecar_requirements = {
        "sha256": sidecar_sha256,
        "strict_reload": True,
        "tensor_exact": True,
        "current_state_restored": True,
        "bc_step": 1000,
    }
    sidecar_mismatches = {
        key: {"expected": expected, "observed": sidecar.get(key)}
        for key, expected in sidecar_requirements.items()
        if sidecar.get(key) != expected
    }
    if sidecar_mismatches:
        raise ValueError(
            f"BC pilot sidecar acceptance mismatch in {manifest_path}: "
            f"{sidecar_mismatches}"
        )

    result = {
        "output_dir": str(root),
        "learning_rate": learning_rate,
        "validation_loss_action_bc": validation_loss,
        "validation_sample_count": sample_count,
        "validation_valid_action_count": validation.get("valid_action_count"),
        "optimizer_steps": 1000,
        "nonzero_update_count": manifest["nonzero_update_count"],
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "metrics": {"path": str(metrics_path), "sha256": sha256_file(metrics_path)},
        "sidecar": {"path": str(sidecar_path), "sha256": sidecar_sha256},
        "training_contract_sha256": provenance.get("training_contract_sha256"),
        "parent_checkpoint_sha256": contract.get("parent_checkpoint_sha256"),
        "statistics_sha256": contract.get("statistics_sha256"),
        "dataset_sha256": contract.get("dataset_sha256"),
        "text_cache_sha256": contract.get("text_cache_sha256"),
    }
    return result, _canonical_pilot_config(config)


def select_uncond_bc_learning_rate(
    pilot_directories: Sequence[str | os.PathLike[str]],
) -> dict[str, Any]:
    """Validate the three preregistered pilots and deterministically select LR."""

    if len(pilot_directories) != len(PILOT_LRS):
        raise ValueError(
            "LR selection requires exactly the three preregistered pilots."
        )
    parsed = [_candidate(directory) for directory in pilot_directories]
    candidates = [item[0] for item in parsed]
    configs = [item[1] for item in parsed]
    observed_lrs = sorted(candidate["learning_rate"] for candidate in candidates)
    if observed_lrs != sorted(PILOT_LRS):
        raise ValueError(
            f"LR pilot set mismatch: expected {sorted(PILOT_LRS)}, got {observed_lrs}."
        )
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("LR pilot configs differ outside learning_rate/output_dir.")
    shared_keys = (
        "parent_checkpoint_sha256",
        "statistics_sha256",
        "dataset_sha256",
        "text_cache_sha256",
    )
    for key in shared_keys:
        values = [candidate[key] for candidate in candidates]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"LR pilot provenance mismatch for {key}: {values}")

    candidates.sort(key=lambda candidate: candidate["learning_rate"])
    best_loss = min(candidate["validation_loss_action_bc"] for candidate in candidates)
    tied = [
        candidate
        for candidate in candidates
        if (candidate["validation_loss_action_bc"] - best_loss)
        / max(best_loss, float.fromhex("0x1.0p-1022"))
        < RELATIVE_TIE_TOLERANCE
    ]
    selected = min(tied, key=lambda candidate: candidate["learning_rate"])
    return {
        "schema": LR_SELECTION_SCHEMA,
        "status": "PASS",
        "selection_rule": (
            "minimum full-validation action-BC loss; choose smaller learning rate "
            "among candidates less than 1% above the minimum"
        ),
        "relative_tie_tolerance": RELATIVE_TIE_TOLERANCE,
        "expected_learning_rates": list(PILOT_LRS),
        "best_observed_validation_loss_action_bc": best_loss,
        "selected_learning_rate": selected["learning_rate"],
        "selected_pilot_output_dir": selected["output_dir"],
        "selected_sidecar": selected["sidecar"],
        "candidates": candidates,
    }


def write_lr_selection_manifest(
    output_path: str | os.PathLike[str],
    payload: Mapping[str, Any],
) -> Path:
    """Atomically write a new non-overwriting LR-selection manifest."""

    if payload.get("schema") != LR_SELECTION_SCHEMA or payload.get("status") != "PASS":
        raise ValueError("Only a validated PASS LR-selection payload may be written.")
    target = Path(output_path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite LR-selection artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
