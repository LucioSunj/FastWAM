"""Checkpoint-bound normalization provenance helpers."""
from __future__ import annotations

import hashlib
import json
import os
import warnings
from collections.abc import Mapping

import torch


def sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dual_regime_schedule_fingerprint(contract: Mapping) -> str:
    """Hash the deterministic optimizer-step schedule contract."""
    if not isinstance(contract, Mapping):
        raise TypeError("dual-regime training contract must be a mapping")
    schedule = contract.get("uncond_weight_schedule")
    total_steps = contract.get("total_optimizer_steps")
    if not isinstance(schedule, (list, tuple)) or len(schedule) < 2:
        raise ValueError("dual-regime contract requires an UNCOND weight schedule")
    if isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0:
        raise ValueError("dual-regime contract requires positive total_optimizer_steps")
    encoded = json.dumps(
        {
            "uncond_weight_schedule": schedule,
            "total_optimizer_steps": total_steps,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def inference_solver_contract(
    model,
    *,
    video_inference_steps: int,
    action_inference_steps: int,
    sigma_shift: float | None = None,
) -> dict:
    """Describe the exact video/action inference schedules used by a Gate run.

    A step count alone is not a solver identity: changing the scheduler class,
    its training grid, or its effective shift changes every denoising state.  The
    returned mapping is deliberately JSON-only so it can be copied unchanged
    into profiles, paired-v1 metadata, donor banks and Gate sidecars.
    """

    def positive_steps(value, label: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{label} must be a positive integer")
        value = int(value)
        if value <= 0:
            raise ValueError(f"{label} must be a positive integer")
        return value

    def scheduler_contract(attribute: str, steps: int) -> dict:
        scheduler = getattr(model, attribute, None)
        if scheduler is None:
            raise ValueError(
                f"model is missing {attribute}; inference solver provenance "
                "cannot be established"
            )
        configured_shift = getattr(scheduler, "shift", None)
        train_steps = getattr(scheduler, "num_train_timesteps", None)
        if isinstance(train_steps, bool) or not isinstance(train_steps, int) or train_steps <= 0:
            raise ValueError(f"{attribute}.num_train_timesteps must be positive")
        try:
            configured_shift = float(configured_shift)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{attribute}.shift must be numeric") from exc
        if configured_shift <= 0:
            raise ValueError(f"{attribute}.shift must be positive")
        effective_shift = configured_shift if sigma_shift is None else float(sigma_shift)
        if effective_shift <= 0:
            raise ValueError("sigma_shift must be positive when provided")
        scheduler_type = type(scheduler)
        return {
            "scheduler_class": f"{scheduler_type.__module__}.{scheduler_type.__qualname__}",
            "num_train_timesteps": train_steps,
            "configured_shift": configured_shift,
            "effective_shift": effective_shift,
            "inference_steps": steps,
        }

    video_steps = positive_steps(video_inference_steps, "video_inference_steps")
    action_steps = positive_steps(action_inference_steps, "action_inference_steps")
    if sigma_shift is not None:
        sigma_shift = float(sigma_shift)
        if sigma_shift <= 0:
            raise ValueError("sigma_shift must be positive when provided")
    return {
        "schema": "fastwam-inference-solver-v1",
        "sigma_shift_override": sigma_shift,
        "video": scheduler_contract("infer_video_scheduler", video_steps),
        "action": scheduler_contract("infer_action_scheduler", action_steps),
        "branch_semantics": {
            "uncond": "action_only",
            "idm": "video_then_future_conditioned_action",
        },
    }


def inference_solver_fingerprint(contract: Mapping) -> str:
    """Return a canonical SHA256 for :func:`inference_solver_contract`."""
    if not isinstance(contract, Mapping) or contract.get("schema") != "fastwam-inference-solver-v1":
        raise ValueError("unsupported or missing FastWAM inference solver contract")
    encoded = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def module_state_schema_sha256(module: torch.nn.Module) -> str:
    """Fingerprint parameter/buffer names, shapes and dtypes, not their values."""
    schema = [
        (name, list(tensor.shape), str(tensor.dtype))
        for name, tensor in module.state_dict().items()
    ]
    encoded = json.dumps(schema, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_model_contract(model) -> dict:
    """Return architecture facts needed for strict checkpoint lineage checks."""
    video = getattr(model, "video_expert", None)
    action = model.action_expert
    return {
        "mot_state_schema_sha256": module_state_schema_sha256(model.mot),
        "video_expert_class": type(video).__name__,
        "action_expert_class": type(action).__name__,
        "video_attention_mask_mode": getattr(video, "video_attention_mask_mode", None),
        "video_action_conditioned": bool(getattr(video, "action_conditioned", False)),
        "video_patch_size": list(getattr(video, "patch_size", ())),
        "video_num_layers": len(getattr(video, "blocks", ())),
        "action_num_layers": len(getattr(action, "blocks", ())),
        "action_hidden_dim": getattr(action, "hidden_dim", None),
        "action_num_heads": getattr(action, "num_heads", None),
        "action_attn_head_dim": getattr(action, "attn_head_dim", None),
    }


def validate_dataset_stats_fingerprint(model, stats_path: str | os.PathLike) -> str:
    """Verify stats for modern adaptive checkpoints; leave vanilla models alone."""
    actual = sha256_file(stats_path)
    provenance = getattr(model, "_loaded_checkpoint_provenance", None)
    live_regimes = tuple(getattr(model, "adaptive_regimes", ()))
    if provenance is None:
        if live_regimes:
            warnings.warn(
                "Adaptive legacy checkpoint has no dataset-stats provenance; "
                "normalization compatibility cannot be verified.",
                RuntimeWarning,
                stacklevel=2,
            )
        return actual
    regimes = tuple(provenance.get("adaptive_regimes", ()))
    if not regimes:
        return actual
    expected = provenance.get("dataset_stats_fingerprint")
    if not isinstance(expected, str) or not expected:
        raise ValueError(
            "Adaptive checkpoint is missing dataset_stats_fingerprint and cannot "
            "be evaluated safely."
        )
    if actual != expected:
        raise ValueError(
            "Dataset stats do not match the adaptive checkpoint: "
            f"checkpoint={expected}, file={actual}, path={os.fspath(stats_path)!r}."
        )
    return actual
