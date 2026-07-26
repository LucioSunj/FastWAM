#!/usr/bin/env python3
"""Apply the preregistered E1-P1D-LC low-cap (``w_cap=0.09``) Canary gates.

This is the post-Canary single-variable diagnostic defined by
``docs/sft_method_discussion/post_canary_low_cap_execution_plan.md``. It is a
separate stage with its own schema. It never re-judges the archived
``FAIL-DIAGNOSED`` E1-P1 Canary and never authorizes the 500-step probe or the
10-epoch formal training by itself.

The persisted preflight/diagnostic artifacts only carry a fixed candidate weight
grid that does not contain ``0.09``. Every margin used here is therefore
recomputed at ``w=0.09`` from the persisted raw ``idm_sq``/``uncond_sq``/``dot``
statistics, and cross-checked against a grid weight that the producer did store.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastwam.adaptive_gate.sdr_contracts import (
    ARCHIVED_FAILED_CANARY_DECISION_SHA256,
    ARCHIVED_P0_5_W0,
    LOW_CAP_CANARY_DECISION_SCHEMA,
    LOW_CAP_DELTA_STEPS,
    LOW_CAP_DIAGNOSTIC_SEED,
    LOW_CAP_FINAL_STEP,
    LOW_CAP_LEARNING_RATE,
    LOW_CAP_MAX_STEPS,
    LOW_CAP_NEW_W_CAP,
    LOW_CAP_OLD_W_CAP,
    LOW_CAP_PREREGISTRATION_SCHEMA,
    LOW_CAP_STAGE,
    LOW_CAP_TRAINING_SEED,
    LOW_CAP_W0_RELATIVE_TOLERANCE,
    artifact_record,
    atomic_json,
    float32,
    read_json,
    validate_low_cap_canary_contract,
)
from fastwam.adaptive_gate.sdr_preflight import (
    negative_margin_fraction,
    weighted_descent_margins,
)
from fastwam.adaptive_gate.training import uncond_weight_at_step


NO_PLUS_FULL_STATEMENT = (
    "No Plus-Full outcome, episode manifest, or task rollout was loaded or used "
    "for this decision."
)
POST_CANARY_STATEMENT = (
    "E1-P1D-LC is a post-Canary single-variable diagnostic. It does not re-judge "
    "the archived FAIL-DIAGNOSED E1-P1 Canary, and a PASS authorizes only a "
    "separately preregistered 500-step low-cap probe."
)
EVALUATED_STEPS = (0, *LOW_CAP_DELTA_STEPS)
# A grid weight the diagnostics producer actually stored, used to prove that the
# local recomputation reproduces the persisted margins bit-for-bit.
CROSS_CHECK_WEIGHT = 0.1
MARGIN_GROUPS = ("action_all", "action_blocks_final")
GENERATED_CONDITIONS = (
    "gt_teacher_forced_future",
    "valid_self_generated_future",
    "no_read",
    "repeat_current",
    "matched_shuffled_future",
    "forced_uncond",
    "standalone_e_i_reference",
)


class LowCapContractError(RuntimeError):
    """A provenance or contract failure that forces a ``NOT-RUN`` decision."""


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise LowCapContractError(f"{path}:{line_number} must be a JSON object.")
            rows.append(row)
    if not rows:
        raise LowCapContractError(f"Training ledger is empty: {path}.")
    return rows


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise LowCapContractError("Cannot average an empty sequence.")
    return sum(float(value) for value in values) / len(values)


def _finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _longest_true_streak(values: Sequence[bool]) -> int:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _linear_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 3:
        raise LowCapContractError("A linear slope needs at least three paired points.")
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    sxx = sum((x - x_mean) ** 2 for x in xs)
    if sxx <= 0.0:
        raise LowCapContractError("Slope requires distinct step positions.")
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / sxx


def _cross_check_recomputation(side: Mapping[str, Any], *, coupling: str) -> dict[str, Any]:
    """Prove the local margin recomputation matches the persisted grid."""
    key = str(float(CROSS_CHECK_WEIGHT))
    worst = 0.0
    checked = 0
    for group_name, values in side.get("groups", {}).items():
        stored = (values.get("margins_by_weight") or {}).get(key)
        if not isinstance(stored, Mapping):
            continue
        recomputed = weighted_descent_margins(values, CROSS_CHECK_WEIGHT)
        for field in ("idm_margin", "uncond_margin", "weighted_gradient_norm_ratio"):
            worst = max(worst, abs(float(stored[field]) - float(recomputed[field])))
        checked += 1
    if checked == 0:
        raise LowCapContractError(
            f"{coupling} diagnostics store no margins_by_weight[{key}] to cross-check."
        )
    if worst > 0.0:
        raise LowCapContractError(
            f"{coupling} margin recomputation disagrees with the persisted grid "
            f"at w={CROSS_CHECK_WEIGHT}: worst_abs={worst}."
        )
    return {
        "cross_check_weight": CROSS_CHECK_WEIGHT,
        "groups_checked": checked,
        "worst_absolute_difference": worst,
    }


def _side_margins(
    side: Mapping[str, Any], *, weight: float, coupling: str
) -> dict[str, Any]:
    groups = side.get("groups")
    if not isinstance(groups, Mapping):
        raise LowCapContractError(f"{coupling} diagnostics have no groups mapping.")
    shards = side.get("shards") or ()
    if not shards:
        raise LowCapContractError(f"{coupling} diagnostics have no gradient shards.")
    result: dict[str, Any] = {
        "coupling": coupling,
        "weight": float(weight),
        "margin_source": (
            "recomputed from persisted idm_sq/uncond_sq/dot; the stored "
            "margins_by_weight grid does not contain this weight"
        ),
        "shard_count": len(shards),
        "recomputation_cross_check": _cross_check_recomputation(
            side, coupling=coupling
        ),
    }
    for group_name in MARGIN_GROUPS:
        if group_name not in groups:
            raise LowCapContractError(
                f"{coupling} diagnostics are missing group {group_name!r}."
            )
        result[group_name] = {
            **weighted_descent_margins(groups[group_name], weight),
            "negative_idm_shard_fraction": negative_margin_fraction(
                shards, group_name=group_name, weight=weight, objective="idm"
            ),
            "negative_uncond_shard_fraction": negative_margin_fraction(
                shards, group_name=group_name, weight=weight, objective="uncond"
            ),
        }
    return result


def _diagnostic_summary(run_dir: str | Path, *, weight: float) -> dict[str, Any]:
    root = Path(run_dir).expanduser().resolve()
    gradient_path = root / "gradient_diagnostics.json"
    generated_path = root / "generated_future_validation.json"
    for path in (gradient_path, generated_path):
        if not path.is_file():
            raise LowCapContractError(f"Missing diagnostic artifact: {path}.")
    gradient = read_json(gradient_path)
    generated = read_json(generated_path)

    common_losses = [
        row for row in gradient.get("loss_records", ()) if row.get("coupling") == "common"
    ]
    if not common_losses:
        raise LowCapContractError(f"No common-noise loss records in {gradient_path}.")
    records = generated.get("records")
    if not isinstance(records, list) or not records:
        raise LowCapContractError(f"No generated-future records in {generated_path}.")

    action_validation = {}
    for condition in GENERATED_CONDITIONS:
        errors = []
        for row in records:
            payload = (row.get("conditions") or {}).get(condition)
            if not isinstance(payload, Mapping):
                raise LowCapContractError(
                    f"Generated-future record is missing condition {condition!r}."
                )
            errors.append(float(payload["normalized_error"]["l2"]))
        action_validation[condition] = {
            "normalized_action_l2_mean": _mean(errors),
            "sample_count": len(errors),
        }

    parity = generated.get("no_read_uncond_parity")
    if not isinstance(parity, Mapping):
        raise LowCapContractError(f"No no_read_uncond_parity block in {generated_path}.")
    sensitivity = generated.get("sensitivity_gate")
    if not isinstance(sensitivity, Mapping):
        raise LowCapContractError(f"No sensitivity_gate block in {generated_path}.")

    return {
        "run_dir": str(root),
        "gradient_diagnostics": artifact_record(gradient_path),
        "generated_future_validation": artifact_record(generated_path),
        "uncond_raw_loss": _mean([row["raw_uncond"] for row in common_losses]),
        "idm_raw_loss": _mean([row["raw_idm"] for row in common_losses]),
        "common_loss_record_count": len(common_losses),
        "gt_idm_action_l2": action_validation["gt_teacher_forced_future"][
            "normalized_action_l2_mean"
        ],
        "generated_idm_action_l2": action_validation["valid_self_generated_future"][
            "normalized_action_l2_mean"
        ],
        "action_validation": action_validation,
        "sensitivity_median": float(sensitivity["median"]),
        "sensitivity_gate_pass": bool(sensitivity.get("pass")),
        "no_read_uncond_parity": {
            "max_abs": float(parity["max_abs"]),
            "tolerance": float(parity.get("tolerance", 1e-4)),
            "pass": bool(parity.get("pass")),
        },
        "common": _side_margins(gradient["common"], weight=weight, coupling="common"),
        "independent": _side_margins(
            gradient["independent"], weight=weight, coupling="independent"
        ),
        "finite": _finite_tree(gradient) and _finite_tree(generated),
    }


def _training_summary(
    path: str | Path,
    *,
    schedule: Sequence[Sequence[float]],
    cap_float32: float,
    expected_steps: int = LOW_CAP_MAX_STEPS,
    reference_learning_rates: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Summarize the ledger without raising on acceptance-condition failures."""
    rows = _read_jsonl(path)
    expected = list(range(1, int(expected_steps) + 1))
    steps = [int(row["global_step"]) for row in rows]
    schedule_steps = [int(row["dual_regime_optimizer_steps"]) for row in rows]
    skipped = [bool(row.get("optimizer_step_was_skipped")) for row in rows]
    clipped = [bool(row["gradient_clipped"]) for row in rows]

    weight_errors: list[float] = []
    weight_exceedances: list[dict[str, Any]] = []
    observed_weights: list[float] = []
    for index, row in enumerate(rows):
        observed = float(row["losses"]["action_regime_weight_uncond"])
        observed_weights.append(observed)
        expected_weight = float32(
            uncond_weight_at_step(
                schedule,
                optimizer_step=index,
                total_optimizer_steps=int(expected_steps),
            )
        )
        weight_errors.append(abs(observed - expected_weight))
        if float32(observed) > cap_float32:
            weight_exceedances.append(
                {"global_step": int(row["global_step"]), "observed": observed}
            )

    learning_rates = [float(row["learning_rate"]) for row in rows]
    learning_rate_check: dict[str, Any] = {
        "compared": False,
        "worst_absolute_difference": None,
        "matches_reference": None,
        "reference_step_count": None,
    }
    if reference_learning_rates is not None:
        reference = [float(value) for value in reference_learning_rates]
        learning_rate_check["compared"] = True
        learning_rate_check["reference_step_count"] = len(reference)
        if len(reference) != len(learning_rates):
            learning_rate_check["matches_reference"] = False
            learning_rate_check["worst_absolute_difference"] = None
        else:
            worst = max(
                (abs(a - b) for a, b in zip(learning_rates, reference)), default=0.0
            )
            learning_rate_check["worst_absolute_difference"] = worst
            learning_rate_check["matches_reference"] = worst == 0.0

    durations = [float(row.get("step_duration_seconds", 0.0)) for row in rows]
    throughput = [float(row.get("samples_per_second", 0.0)) for row in rows]
    return {
        "ledger": artifact_record(path),
        "applied_optimizer_steps": len(rows),
        "expected_optimizer_steps": int(expected_steps),
        "steps_are_exact_successful_sequence": steps == expected,
        "schedule_counter_tracks_successful_steps": schedule_steps == expected,
        "skipped_optimizer_steps": sum(skipped),
        "clip_fraction": sum(clipped) / len(clipped),
        "longest_clipping_streak": _longest_true_streak(clipped),
        "all_finite": _finite_tree(rows),
        "observed_uncond_weights": observed_weights,
        "worst_schedule_weight_error": max(weight_errors, default=0.0),
        "schedule_matches_locked_weights": max(weight_errors, default=1.0) == 0.0,
        "weight_cap_exceedances": weight_exceedances,
        "weight_cap_float32": cap_float32,
        "learning_rates": learning_rates,
        "learning_rate_check": learning_rate_check,
        "peak_gpu_memory_bytes": max(
            (int(row["peak_gpu_memory_bytes"]) for row in rows), default=0
        ),
        "total_step_time_seconds": sum(durations),
        "wall_time_seconds": float(rows[-1]["elapsed_seconds"]),
        "mean_samples_per_second": (
            _mean(throughput) if all(value > 0.0 for value in throughput) else None
        ),
    }


def _git_value(cwd: Path, *command: str) -> str | None:
    result = subprocess.run(
        ["git", *command],
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _reverify_bindings(bindings: Mapping[str, Any], *, repo: str | None) -> dict[str, Any]:
    """Recompute every bound artifact SHA; ``code_commit`` is verified via git."""
    verified: dict[str, Any] = {}
    failures: list[str] = []
    for name, record in bindings.items():
        if not isinstance(record, Mapping):
            failures.append(f"{name}: binding is not an object")
            continue
        raw_path = str(record.get("path", ""))
        if name == "code_commit" or raw_path.startswith("git:"):
            observed = None
            dirty = None
            if repo is not None:
                repo_path = Path(repo).expanduser().resolve()
                commit = _git_value(repo_path, "rev-parse", "HEAD")
                dirty = bool(_git_value(repo_path, "status", "--porcelain"))
                if commit is not None:
                    import hashlib

                    observed = hashlib.sha256(commit.encode("ascii")).hexdigest()
            matches = observed == record.get("sha256") and dirty is False
            verified[name] = {
                "path": raw_path,
                "recorded_sha256": record.get("sha256"),
                "observed_sha256": observed,
                "worktree_dirty": dirty,
                "verified": bool(matches),
            }
            if not matches:
                failures.append(f"{name}: code commit does not match the preflight")
            continue
        path = Path(raw_path).expanduser()
        if not path.is_file():
            verified[name] = {
                "path": raw_path,
                "recorded_sha256": record.get("sha256"),
                "observed_sha256": None,
                "verified": False,
            }
            failures.append(f"{name}: bound artifact is missing ({raw_path})")
            continue
        actual = artifact_record(path)
        matches = actual["sha256"] == record.get("sha256")
        verified[name] = {
            "path": actual["path"],
            "recorded_sha256": record.get("sha256"),
            "observed_sha256": actual["sha256"],
            "size_bytes": actual["size_bytes"],
            "verified": bool(matches),
        }
        if not matches:
            failures.append(f"{name}: bound artifact SHA256 changed")
    return {"artifacts": verified, "failures": failures, "pass": not failures}


def _validation_exclusion(
    resolved_config: str | Path, *, validation_manifest_sha256: str | None
) -> dict[str, Any]:
    """Prove the fixed validation episodes stay out of the training sampler."""
    import yaml

    path = Path(resolved_config).expanduser().resolve()
    if not path.is_file():
        raise LowCapContractError(f"Missing low-cap resolved config: {path}.")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    train = ((config or {}).get("data") or {}).get("train")
    if not isinstance(train, Mapping):
        raise LowCapContractError("Low-cap resolved config has no data.train mapping.")
    manifest_path = train.get("episode_split_manifest")
    split = train.get("manifest_split")
    reasons: list[str] = []
    manifest_sha = None
    if not manifest_path:
        reasons.append("data.train.episode_split_manifest is not set")
    else:
        candidate = Path(str(manifest_path)).expanduser()
        if not candidate.is_file():
            reasons.append(f"episode split manifest is missing: {manifest_path}")
        else:
            manifest_sha = artifact_record(candidate)["sha256"]
            if (
                validation_manifest_sha256 is not None
                and manifest_sha != validation_manifest_sha256
            ):
                reasons.append(
                    "episode split manifest differs from the bound validation manifest"
                )
    if split != "train":
        reasons.append(f"data.train.manifest_split must be 'train', got {split!r}")
    return {
        "resolved_config": artifact_record(path),
        "episode_split_manifest": (
            str(manifest_path) if manifest_path else None
        ),
        "episode_split_manifest_sha256": manifest_sha,
        "manifest_split": split,
        "reasons": reasons,
        "pass": not reasons,
    }


def _single_variable_audit(resolved_config: Mapping[str, Any]) -> dict[str, Any]:
    """Confirm that only ``w_cap`` moved relative to the failed Canary."""
    optimizer = (
        ((resolved_config.get("dual_regime_training") or {}).get("optimizer")) or {}
    )
    observed = {
        "learning_rate": resolved_config.get("learning_rate"),
        "max_steps": resolved_config.get("max_steps"),
        "seed": resolved_config.get("seed"),
        "batch_size": resolved_config.get("batch_size"),
        "gradient_accumulation_steps": resolved_config.get(
            "gradient_accumulation_steps"
        ),
        "mixed_precision": resolved_config.get("mixed_precision"),
        "max_grad_norm": resolved_config.get("max_grad_norm"),
        "weight_decay": resolved_config.get("weight_decay"),
        "lr_scheduler_type": resolved_config.get("lr_scheduler_type"),
        "weights_checkpoint_kind": resolved_config.get("weights_checkpoint_kind"),
        "action_lr_scale": optimizer.get("action_lr_scale"),
        "proprio_lr_scale": optimizer.get("proprio_lr_scale"),
        "video_lr_scale": optimizer.get("video_lr_scale"),
    }
    expected = {
        "learning_rate": LOW_CAP_LEARNING_RATE,
        "max_steps": LOW_CAP_MAX_STEPS,
        "seed": LOW_CAP_TRAINING_SEED,
        "batch_size": 1,
        "gradient_accumulation_steps": 64,
        "mixed_precision": "bf16",
        "max_grad_norm": 1.0,
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "weights_checkpoint_kind": "action_dit_delta",
        "action_lr_scale": 1.0,
        "proprio_lr_scale": 0.0,
        "video_lr_scale": 0.0,
    }
    mismatches = {
        name: {"observed": observed[name], "expected": value}
        for name, value in expected.items()
        if observed[name] != value
    }
    return {
        "observed": observed,
        "expected": expected,
        "mismatches": mismatches,
        "pass": not mismatches,
    }


def _resolve_reference_metrics(
    decision_path: Path, payload: Mapping[str, Any], override: str | None
) -> dict[str, Any]:
    recorded = (payload.get("artifacts") or {}).get("training_metrics")
    if not isinstance(recorded, Mapping):
        raise LowCapContractError(
            "Failed Canary decision does not bind its training_metrics ledger."
        )
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    else:
        candidates.append(decision_path.parent / "canary_training_metrics.jsonl")
        candidates.append(decision_path.parent / "training_metrics.jsonl")
        candidates.append(Path(str(recorded.get("path", ""))).expanduser())
    for candidate in candidates:
        if candidate.is_file():
            actual = artifact_record(candidate)
            if actual["sha256"] != recorded.get("sha256"):
                raise LowCapContractError(
                    "Reference Canary training ledger changed: "
                    f"expected={recorded.get('sha256')}, actual={actual['sha256']}."
                )
            rows = _read_jsonl(candidate)
            return {
                "artifact": actual,
                "learning_rates": [float(row["learning_rate"]) for row in rows],
            }
    raise LowCapContractError(
        "Cannot resolve the failed Canary training ledger for LR comparison; "
        f"searched {[str(item) for item in candidates]}."
    )


def _environment(output_dir: Path) -> dict[str, Any]:
    try:
        import torch

        torch_environment: dict[str, Any] = {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "gpu_total_memory_bytes": (
                torch.cuda.get_device_properties(0).total_memory
                if torch.cuda.is_available()
                else None
            ),
        }
    except Exception as exc:  # pragma: no cover - environment probing only
        torch_environment = {"inspection_error": str(exc)}
    disk = shutil.disk_usage(output_dir)
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        **torch_environment,
        "disk": {
            "path": str(output_dir),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
    }


def _acceptance_conditions(
    *,
    training: Mapping[str, Any],
    steps: Mapping[str, Mapping[str, Any]],
    binding_verification: Mapping[str, Any],
    validation_exclusion: Mapping[str, Any],
    plus_full_clean: bool,
) -> dict[str, bool]:
    baseline = steps["0"]
    final = steps[str(LOW_CAP_FINAL_STEP)]
    common = final["common"]
    independent = final["independent"]
    uncond_series = [steps[str(step)]["uncond_raw_loss"] for step in EVALUATED_STEPS]
    base_uncond = baseline["uncond_raw_loss"]
    relative_drop = (
        (base_uncond - final["uncond_raw_loss"]) / base_uncond
        if base_uncond > 0.0
        else float("-inf")
    )
    return {
        "condition_01_fifty_applied_steps_no_skip_no_nonfinite": bool(
            training["applied_optimizer_steps"] == LOW_CAP_MAX_STEPS
            and training["steps_are_exact_successful_sequence"]
            and training["schedule_counter_tracks_successful_steps"]
            and training["skipped_optimizer_steps"] == 0
            and training["all_finite"]
        ),
        "condition_02_weight_matches_locked_schedule_and_never_exceeds_cap": bool(
            training["schedule_matches_locked_weights"]
            and not training["weight_cap_exceedances"]
        ),
        "condition_03_clip_fraction_below_20_percent_no_streak_of_ten": bool(
            training["clip_fraction"] < 0.20
            and training["longest_clipping_streak"] < 10
        ),
        "condition_04_uncond_raw_loss_dropped_at_least_10_percent": bool(
            relative_drop >= 0.10
        ),
        "condition_05_uncond_raw_loss_linear_slope_is_negative": bool(
            _linear_slope(list(EVALUATED_STEPS), uncond_series) < 0.0
        ),
        "condition_06_common_idm_raw_loss_within_5_percent": bool(
            final["idm_raw_loss"] <= 1.05 * baseline["idm_raw_loss"]
        ),
        "condition_07_gt_idm_action_l2_within_5_percent": bool(
            final["gt_idm_action_l2"] <= 1.05 * baseline["gt_idm_action_l2"]
        ),
        "condition_08_generated_idm_action_l2_within_5_percent": bool(
            final["generated_idm_action_l2"]
            <= 1.05 * baseline["generated_idm_action_l2"]
        ),
        "condition_09_sensitivity_median_retains_half": bool(
            final["sensitivity_median"] >= 0.5 * baseline["sensitivity_median"]
        ),
        "condition_10_common_action_all_margins_positive": bool(
            common["action_all"]["idm_margin"] > 0.0
            and common["action_all"]["uncond_margin"] > 0.0
        ),
        "condition_11_independent_action_all_margins_positive": bool(
            independent["action_all"]["idm_margin"] > 0.0
            and independent["action_all"]["uncond_margin"] > 0.0
        ),
        "condition_12_common_action_all_negative_shard_fractions_at_most_20_percent": (
            bool(
                common["action_all"]["negative_idm_shard_fraction"] <= 0.20
                and common["action_all"]["negative_uncond_shard_fraction"] <= 0.20
            )
        ),
        "condition_13_common_final_block_negative_idm_fraction_at_most_20_percent": (
            bool(common["action_blocks_final"]["negative_idm_shard_fraction"] <= 0.20)
        ),
        "condition_14_common_action_all_weighted_norm_ratio_at_most_1_25": bool(
            common["action_all"]["weighted_gradient_norm_ratio"] <= 1.25
        ),
        "condition_15_no_read_forced_uncond_parity_within_1e_4": bool(
            all(
                steps[str(step)]["no_read_uncond_parity"]["max_abs"] <= 1e-4
                for step in EVALUATED_STEPS
            )
        ),
        "condition_16_all_bound_artifact_sha256_reverified": bool(
            binding_verification["pass"]
        ),
        "condition_17_validation_episodes_excluded_from_training_sampler": bool(
            validation_exclusion["pass"]
        ),
        "condition_18_plus_full_outcome_not_loaded_or_used": bool(plus_full_clean),
    }


PLUS_FULL_PATH_TOKENS = ("libero_plus", "libero-plus", "plus_full", "plus-full")


def _plus_full_clean(
    payloads: Sequence[Mapping[str, Any]], bindings: Mapping[str, Any]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for index, payload in enumerate(payloads):
        if payload.get("plus_full_used") is not False:
            reasons.append(f"input payload {index} does not state plus_full_used=false")
    for name, record in bindings.items():
        path = str(record.get("path", "")) if isinstance(record, Mapping) else ""
        if "plus" in str(name).lower() or any(
            token in path.lower() for token in PLUS_FULL_PATH_TOKENS
        ):
            reasons.append(f"artifact binding {name!r} references a Plus artifact")
    return (not reasons), reasons


def build_low_cap_decision(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.selected_step) != LOW_CAP_FINAL_STEP:
        raise LowCapContractError(
            "The low-cap decision is defined on the step-50 checkpoint only; "
            f"re-selecting step {args.selected_step} is prohibited."
        )
    preflight_path = Path(args.preflight_decision).expanduser().resolve()
    failed_path = Path(args.failed_canary_decision).expanduser().resolve()
    preflight_record = artifact_record(preflight_path)
    failed_record = artifact_record(failed_path)
    preflight = read_json(preflight_path)
    failed = read_json(failed_path)

    contract = validate_low_cap_canary_contract(
        preflight,
        failed_canary_decision=failed,
        failed_canary_sha256=failed_record["sha256"],
        expected_failed_canary_sha256=args.expected_failed_canary_sha256,
        archived_w0=float(args.archived_w0),
        w0_relative_tolerance=float(args.w0_relative_tolerance),
    )

    preregistration_path = Path(args.preregistration).expanduser().resolve()
    preregistration = read_json(preregistration_path)
    if preregistration.get("schema") != LOW_CAP_PREREGISTRATION_SCHEMA:
        raise LowCapContractError(
            "Preregistration schema must be "
            f"{LOW_CAP_PREREGISTRATION_SCHEMA!r}, got {preregistration.get('schema')!r}."
        )
    for field, expected in (
        ("stage", LOW_CAP_STAGE),
        ("old_w_cap", LOW_CAP_OLD_W_CAP),
        ("new_w_cap", LOW_CAP_NEW_W_CAP),
        ("learning_rate", LOW_CAP_LEARNING_RATE),
        ("max_steps", LOW_CAP_MAX_STEPS),
        ("training_seed", LOW_CAP_TRAINING_SEED),
        ("diagnostic_seed", LOW_CAP_DIAGNOSTIC_SEED),
        ("plus_full_used", False),
    ):
        if preregistration.get(field) != expected:
            raise LowCapContractError(
                f"Preregistration {field!r} must be {expected!r}, got "
                f"{preregistration.get(field)!r}."
            )

    reference = _resolve_reference_metrics(failed_path, failed, args.reference_training_metrics)
    diagnostics_by_step = {
        "0": args.baseline_diagnostics,
        "10": args.step10_diagnostics,
        "25": args.step25_diagnostics,
        "50": args.step50_diagnostics,
    }
    steps = {
        key: _diagnostic_summary(value, weight=LOW_CAP_NEW_W_CAP)
        for key, value in diagnostics_by_step.items()
    }
    training = _training_summary(
        args.training_metrics,
        schedule=contract["schedule"],
        cap_float32=contract["w_cap_float32"],
        reference_learning_rates=reference["learning_rates"],
    )
    deltas = {
        f"step{step}": artifact_record(path)
        for step, path in (
            (10, args.step10_delta),
            (25, args.step25_delta),
            (50, args.step50_delta),
        )
    }

    import yaml

    resolved_config_path = Path(args.resolved_config).expanduser().resolve()
    if not resolved_config_path.is_file():
        raise LowCapContractError(f"Missing low-cap resolved config: {resolved_config_path}.")
    resolved_config = yaml.safe_load(resolved_config_path.read_text(encoding="utf-8")) or {}
    single_variable = _single_variable_audit(resolved_config)
    configured_schedule = (resolved_config.get("dual_regime_training") or {}).get(
        "uncond_weight_schedule"
    )
    schedule_serialized = [
        [float(point[0]), float(point[1])] for point in (configured_schedule or ())
    ]
    if schedule_serialized != contract["schedule"]:
        raise LowCapContractError(
            "Low-cap resolved config does not serialize the locked schedule: "
            f"{schedule_serialized} != {contract['schedule']}."
        )

    binding_verification = _reverify_bindings(
        preflight.get("artifact_bindings") or {}, repo=args.repo
    )
    validation_exclusion = _validation_exclusion(
        resolved_config_path,
        validation_manifest_sha256=contract["artifact_bindings"].get(
            "validation_manifest"
        ),
    )
    plus_full_clean, plus_full_reasons = _plus_full_clean(
        [preflight, failed, preregistration],
        preflight.get("artifact_bindings") or {},
    )

    conditions = _acceptance_conditions(
        training=training,
        steps=steps,
        binding_verification=binding_verification,
        validation_exclusion=validation_exclusion,
        plus_full_clean=plus_full_clean,
    )
    failures = [name for name, value in conditions.items() if not value]

    contract_violations: list[str] = []
    if not single_variable["pass"]:
        contract_violations.append(
            f"non-w_cap training settings changed: {sorted(single_variable['mismatches'])}"
        )
    if training["learning_rate_check"]["matches_reference"] is not True:
        contract_violations.append(
            "per-step learning rate differs from the failed Canary reference"
        )
    if not all(summary["finite"] for summary in steps.values()):
        contract_violations.append("non-finite diagnostic metric")
    failures.extend(contract_violations)
    if plus_full_reasons:
        failures.extend(plus_full_reasons)

    status = "PASS" if not failures else "FAIL-DIAGNOSED"
    output_dir = Path(args.out).expanduser().resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "schema": LOW_CAP_CANARY_DECISION_SCHEMA,
        "stage": LOW_CAP_STAGE,
        "status": status,
        "plus_full_used": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "post_canary_statement": POST_CANARY_STATEMENT,
        "no_plus_full_statement": NO_PLUS_FULL_STATEMENT,
        "single_variable": {
            "name": "w_cap",
            "old_w_cap": LOW_CAP_OLD_W_CAP,
            "new_w_cap": LOW_CAP_NEW_W_CAP,
            "audit": single_variable,
        },
        "trigger": {
            "failed_canary_decision": failed_record,
            "status": contract["trigger"]["status"],
            "failure_conditions": contract["trigger"]["failure_conditions"],
            "required_condition": contract["trigger"]["required_condition"],
            "reinterpretation": (
                "The archived E1-P1 Canary remains FAIL-DIAGNOSED and is not "
                "re-judged by this decision."
            ),
        },
        "fresh_preflight": {
            "decision": preflight_record,
            "w0": contract["w0"],
            "archived_w0": contract["archived_w0"],
            "w0_relative_drift": contract["w0_relative_drift"],
            "w0_relative_tolerance": float(args.w0_relative_tolerance),
        },
        "preregistration": artifact_record(preregistration_path),
        "low_cap_plan": artifact_record(args.low_cap_plan),
        "artifact_bindings": preflight["artifact_bindings"],
        "invariant_artifact_bindings": contract["invariant_artifact_bindings"],
        "artifact_reverification": binding_verification,
        "initializer_stage": "e_i_s0",
        "schedule": {
            "locked": contract["schedule"],
            "w_cap": contract["w_cap"],
            "w_cap_float32": contract["w_cap_float32"],
            "resolved_config_schedule": schedule_serialized,
            "per_step_verification": {
                "worst_absolute_error": training["worst_schedule_weight_error"],
                "matches_locked_weights": training["schedule_matches_locked_weights"],
                "cap_exceedances": training["weight_cap_exceedances"],
                "observed_weights": training["observed_uncond_weights"],
            },
        },
        "learning_rate_reference": {
            "artifact": reference["artifact"],
            **training["learning_rate_check"],
        },
        "training": training,
        "evaluation_steps": steps,
        "checkpoint_deltas": deltas,
        "selected_checkpoint_step": LOW_CAP_FINAL_STEP,
        "intermediate_step_reselection": "prohibited",
        "validation_exclusion": validation_exclusion,
        "runtime": {
            "wall_time_seconds": training["wall_time_seconds"],
            "mean_samples_per_second": training["mean_samples_per_second"],
            "peak_gpu_memory_bytes": training["peak_gpu_memory_bytes"],
            "environment": _environment(output_dir),
        },
        "acceptance_conditions": conditions,
        "contract_violations": contract_violations,
        "failure_conditions": failures,
        "probe_500_authorized": status == "PASS",
        "formal_training_authorized": False,
        "next_stage": (
            "Separately preregister and implement a 500-step low-cap probe from "
            "E-I/S0."
            if status == "PASS"
            else "Follow the single diagnostic branch in section 9 of the low-cap plan."
        ),
    }


def _not_run(args: argparse.Namespace, blocker: str) -> dict[str, Any]:
    return {
        "schema": LOW_CAP_CANARY_DECISION_SCHEMA,
        "stage": LOW_CAP_STAGE,
        "status": "NOT-RUN",
        "plus_full_used": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "post_canary_statement": POST_CANARY_STATEMENT,
        "no_plus_full_statement": NO_PLUS_FULL_STATEMENT,
        "single_variable": {
            "name": "w_cap",
            "old_w_cap": LOW_CAP_OLD_W_CAP,
            "new_w_cap": LOW_CAP_NEW_W_CAP,
        },
        "selected_checkpoint_step": LOW_CAP_FINAL_STEP,
        "intermediate_step_reselection": "prohibited",
        "acceptance_conditions": {},
        "blockers": [blocker],
        "failure_conditions": [blocker],
        "probe_500_authorized": False,
        "formal_training_authorized": False,
    }


def decide(args: argparse.Namespace) -> None:
    try:
        result = build_low_cap_decision(args)
    except Exception as exc:  # provenance/contract failure -> NOT-RUN, never PASS
        result = _not_run(args, f"{type(exc).__name__}: {exc}")
    atomic_json(args.out, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "failure_conditions": result.get("failure_conditions", []),
            },
            sort_keys=True,
        )
    )
    if result["status"] != "PASS":
        raise SystemExit(3)


def preregister(args: argparse.Namespace) -> None:
    preflight = read_json(args.preflight_decision)
    failed_record = artifact_record(args.failed_canary_decision)
    failed = read_json(args.failed_canary_decision)
    contract = validate_low_cap_canary_contract(
        preflight,
        failed_canary_decision=failed,
        failed_canary_sha256=failed_record["sha256"],
        expected_failed_canary_sha256=args.expected_failed_canary_sha256,
        archived_w0=float(args.archived_w0),
        w0_relative_tolerance=float(args.w0_relative_tolerance),
    )
    result = {
        "schema": LOW_CAP_PREREGISTRATION_SCHEMA,
        "stage": LOW_CAP_STAGE,
        "old_w_cap": LOW_CAP_OLD_W_CAP,
        "new_w_cap": LOW_CAP_NEW_W_CAP,
        "learning_rate": LOW_CAP_LEARNING_RATE,
        "max_steps": LOW_CAP_MAX_STEPS,
        "training_seed": LOW_CAP_TRAINING_SEED,
        "diagnostic_seed": LOW_CAP_DIAGNOSTIC_SEED,
        "plus_full_used": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "w0": contract["w0"],
        "archived_w0": contract["archived_w0"],
        "w0_relative_drift": contract["w0_relative_drift"],
        "locked_schedule": contract["schedule"],
        "w_cap_float32": contract["w_cap_float32"],
        "delta_steps": list(LOW_CAP_DELTA_STEPS),
        "selected_checkpoint_step": LOW_CAP_FINAL_STEP,
        "intermediate_step_reselection": "prohibited",
        "artifact_bindings": preflight["artifact_bindings"],
        "invariant_artifact_bindings": contract["invariant_artifact_bindings"],
        "trigger": {
            "failed_canary_decision": failed_record,
            "status": contract["trigger"]["status"],
            "failure_conditions": contract["trigger"]["failure_conditions"],
        },
        "low_cap_plan": artifact_record(args.low_cap_plan),
        "post_canary_statement": POST_CANARY_STATEMENT,
        "no_plus_full_statement": NO_PLUS_FULL_STATEMENT,
        "authorizes": (
            "Exactly one 50-step low-cap Canary. It does not authorize the "
            "500-step probe or formal training."
        ),
    }
    atomic_json(args.out, result)
    print(json.dumps({"status": "PREREGISTERED", "w0": contract["w0"]}, sort_keys=True))


def print_schedule(args: argparse.Namespace) -> None:
    """Emit the locked low-cap schedule without writing any artifact."""
    failed_record = artifact_record(args.failed_canary_decision)
    contract = validate_low_cap_canary_contract(
        read_json(args.preflight_decision),
        failed_canary_decision=read_json(args.failed_canary_decision),
        failed_canary_sha256=failed_record["sha256"],
        expected_failed_canary_sha256=args.expected_failed_canary_sha256,
        archived_w0=float(args.archived_w0),
        w0_relative_tolerance=float(args.w0_relative_tolerance),
    )
    print(json.dumps(contract["schedule"], separators=(",", ":"), allow_nan=False))


def write_evidence_index(args: argparse.Namespace) -> None:
    """Write a mechanical SHA256SUMS.txt and README.md index for the run."""
    run_dir = Path(args.run_dir).expanduser().resolve()
    decision_path = Path(args.decision).expanduser().resolve()
    decision = read_json(decision_path)
    suffixes = {".json", ".jsonl", ".yaml", ".yml", ".txt", ".md"}
    sums_path = run_dir / "SHA256SUMS.txt"
    readme_path = run_dir / "README.md"
    listed = sorted(
        path
        for path in run_dir.rglob("*")
        if path.is_file()
        and path.suffix in suffixes
        and path not in {sums_path, readme_path}
    )
    lines = [
        f"{artifact_record(path)['sha256']}  {path.relative_to(run_dir)}"
        for path in listed
    ]
    sums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    deltas = decision.get("checkpoint_deltas") or {}
    delta_lines = [
        f"- `{name}`: `{record.get('sha256')}` ({record.get('size_bytes')} bytes)"
        for name, record in sorted(deltas.items())
    ]
    readme_path.write_text(
        "\n".join(
            [
                f"# {LOW_CAP_STAGE} low-cap Canary evidence",
                "",
                "Machine-generated index. No interpretation is asserted here; read",
                "`low_cap_canary_decision.json` for the recorded outcome.",
                "",
                f"- stage: `{LOW_CAP_STAGE}`",
                f"- decision status: `{decision.get('status')}`",
                f"- single variable: `w_cap` "
                f"`{LOW_CAP_OLD_W_CAP}` -> `{LOW_CAP_NEW_W_CAP}`",
                f"- selected checkpoint step: `{decision.get('selected_checkpoint_step')}`",
                f"- 500-step probe authorized: `{decision.get('probe_500_authorized')}`",
                f"- formal training authorized: "
                f"`{decision.get('formal_training_authorized')}`",
                f"- plus_full_used: `{decision.get('plus_full_used')}`",
                "",
                "## ActionDiT delta checkpoints",
                "",
                *(delta_lines or ["- none recorded"]),
                "",
                "## Hashed provenance files",
                "",
                "`SHA256SUMS.txt` lists every JSON/JSONL/YAML/text artifact under this",
                "run directory. Large checkpoint deltas are hashed in the decision",
                "instead of being re-read here.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "WRITTEN",
                "sha256sums": str(sums_path),
                "readme": str(readme_path),
                "file_count": len(listed),
            },
            sort_keys=True,
        )
    )


def _add_trigger_arguments(sub: argparse.ArgumentParser, *, plan: bool = True) -> None:
    sub.add_argument("--preflight-decision", required=True)
    sub.add_argument("--failed-canary-decision", required=True)
    if plan:
        sub.add_argument("--low-cap-plan", required=True)
    sub.add_argument(
        "--expected-failed-canary-sha256",
        default=ARCHIVED_FAILED_CANARY_DECISION_SHA256,
    )
    sub.add_argument("--archived-w0", type=float, default=ARCHIVED_P0_5_W0)
    sub.add_argument(
        "--w0-relative-tolerance",
        type=float,
        default=LOW_CAP_W0_RELATIVE_TOLERANCE,
    )


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    show = sub.add_parser("print-schedule")
    _add_trigger_arguments(show, plan=False)
    show.set_defaults(func=print_schedule)

    index = sub.add_parser("write-evidence-index")
    index.add_argument("--decision", required=True)
    index.add_argument("--run-dir", required=True)
    index.set_defaults(func=write_evidence_index)

    pre = sub.add_parser("preregister")
    _add_trigger_arguments(pre)
    pre.add_argument("--out", required=True)
    pre.set_defaults(func=preregister)

    dec = sub.add_parser("decide")
    _add_trigger_arguments(dec)
    dec.add_argument("--out", required=True)
    dec.add_argument("--preregistration", required=True)
    dec.add_argument("--resolved-config", required=True)
    dec.add_argument("--baseline-diagnostics", required=True)
    dec.add_argument("--step10-diagnostics", required=True)
    dec.add_argument("--step25-diagnostics", required=True)
    dec.add_argument("--step50-diagnostics", required=True)
    dec.add_argument("--training-metrics", required=True)
    dec.add_argument("--step10-delta", required=True)
    dec.add_argument("--step25-delta", required=True)
    dec.add_argument("--step50-delta", required=True)
    dec.add_argument("--reference-training-metrics")
    dec.add_argument("--repo")
    dec.add_argument("--selected-step", type=int, default=LOW_CAP_FINAL_STEP)
    dec.set_defaults(func=decide)
    return ap


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
