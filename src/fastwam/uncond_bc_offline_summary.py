"""Strict comparison of zero- and trained-LoRA BC offline evaluations."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from fastwam.adapters import sha256_file
from fastwam.uncond_bc_offline import EXPECTED_VALIDATION_WINDOWS, OFFLINE_EVAL_SCHEMA

OFFLINE_COMPARISON_SCHEMA = "fastwam-uncond-bc-offline-comparison-v1"


def _load_accepted(path: str | os.PathLike[str], *, policy: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"BC offline manifest is missing: {source}") from None
    if not isinstance(payload, dict):
        raise TypeError(f"BC offline manifest must be an object: {source}")
    expected = {
        "schema": OFFLINE_EVAL_SCHEMA,
        "status": "PASS",
        "policy": policy,
        "future_prediction_calls": 0,
        "lora_gradients_absent": True,
        "frozen_parameter_versions_unchanged": True,
        "contains_gate": False,
        "contains_critic": False,
        "contains_value_head": False,
    }
    mismatches = {
        key: {"expected": value, "observed": payload.get(key)}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise ValueError(f"BC offline acceptance mismatch in {source}: {mismatches}")
    validation = payload.get("validation")
    if not isinstance(validation, Mapping):
        raise TypeError(f"BC offline validation payload is malformed: {source}")
    if validation.get("sample_count") != EXPECTED_VALIDATION_WINDOWS:
        raise ValueError(f"BC offline validation is incomplete: {source}")
    loss = validation.get("loss_action_bc")
    if (
        isinstance(loss, bool)
        or not isinstance(loss, (int, float))
        or not math.isfinite(float(loss))
        or float(loss) < 0
    ):
        raise ValueError(f"BC offline loss is invalid: {source}")
    dimensions = validation.get("mse_per_dimension")
    if (
        not isinstance(dimensions, list)
        or len(dimensions) != 7
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in dimensions
        )
    ):
        raise ValueError(f"BC offline per-dimension MSE is invalid: {source}")
    if policy == "zero_lora" and payload.get("sidecar") is not None:
        raise ValueError("zero_lora offline manifest unexpectedly records a sidecar.")
    if policy == "bc_lora" and not isinstance(payload.get("sidecar"), Mapping):
        raise ValueError("bc_lora offline manifest is missing its sidecar.")
    payload["_artifact"] = {"path": str(source), "sha256": sha256_file(source)}
    return payload


def compare_uncond_bc_offline(
    zero_manifest: str | os.PathLike[str],
    bc_manifest: str | os.PathLike[str],
) -> dict[str, Any]:
    """Validate the paired manifests and compute signed BC-minus-zero changes."""

    zero = _load_accepted(zero_manifest, policy="zero_lora")
    trained = _load_accepted(bc_manifest, policy="bc_lora")
    shared_contract_keys = (
        "parent_checkpoint_sha256",
        "statistics_sha256",
        "dataset_sha256",
        "text_cache_sha256",
        "world_size",
        "lora",
        "bc_policy",
    )
    mismatches = {
        key: {
            "zero_lora": zero["contract"].get(key),
            "bc_lora": trained["contract"].get(key),
        }
        for key in shared_contract_keys
        if zero["contract"].get(key) != trained["contract"].get(key)
    }
    if mismatches:
        raise ValueError(f"BC offline paired contract mismatch: {mismatches}")

    zero_validation = zero["validation"]
    trained_validation = trained["validation"]
    zero_loss = float(zero_validation["loss_action_bc"])
    trained_loss = float(trained_validation["loss_action_bc"])
    denominator = max(zero_loss, float.fromhex("0x1.0p-1022"))
    per_dimension_change = [
        float(trained_value) - float(zero_value)
        for zero_value, trained_value in zip(
            zero_validation["mse_per_dimension"],
            trained_validation["mse_per_dimension"],
            strict=True,
        )
    ]
    return {
        "schema": OFFLINE_COMPARISON_SCHEMA,
        "status": "PASS",
        "sample_count": EXPECTED_VALIDATION_WINDOWS,
        "zero_lora": {
            "artifact": zero["_artifact"],
            "loss_action_bc": zero_loss,
            "mse_per_dimension": zero_validation["mse_per_dimension"],
        },
        "bc_lora": {
            "artifact": trained["_artifact"],
            "sidecar": dict(trained["sidecar"]),
            "loss_action_bc": trained_loss,
            "mse_per_dimension": trained_validation["mse_per_dimension"],
        },
        "bc_minus_zero_loss_action_bc": trained_loss - zero_loss,
        "relative_loss_change": (trained_loss - zero_loss) / denominator,
        "relative_loss_reduction": (zero_loss - trained_loss) / denominator,
        "bc_minus_zero_mse_per_dimension": per_dimension_change,
        "bc_improves_offline_loss": trained_loss < zero_loss,
        "shared_contract": {key: zero["contract"][key] for key in shared_contract_keys},
    }


def write_offline_comparison(
    output: str | os.PathLike[str],
    payload: Mapping[str, Any],
) -> Path:
    """Atomically write one non-overwriting accepted comparison artifact."""

    if (
        payload.get("schema") != OFFLINE_COMPARISON_SCHEMA
        or payload.get("status") != "PASS"
    ):
        raise ValueError("Only an accepted BC offline comparison may be written.")
    target = Path(output).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite BC offline comparison: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
