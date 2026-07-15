"""Strict standalone-IDM to shared dual-regime warm-start contract."""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from .provenance import checkpoint_model_contract, sha256_file


_KIND = "standalone_idm"
_REQUIRED_KEYS = {
    "kind",
    "checkpoint",
    "expected_checkpoint_sha256",
    "source_task",
    "source_config",
    "source_dataset_stats",
}
_ARCHITECTURE_KEYS = (
    "model_id",
    "tokenizer_model_id",
    "tokenizer_max_len",
    "proprio_dim",
    "video_dit_config",
    "action_dit_config",
    "video_scheduler",
    "action_scheduler",
)


def _plain_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value)}.")
    return dict(value)


def warm_start_is_enabled(config: Any) -> bool:
    if config is None:
        return False
    plain = _plain_mapping(config, name="warm_start")
    kind = plain.get("kind")
    populated = {key for key, value in plain.items() if value not in (None, "")}
    if kind in (None, ""):
        if populated:
            raise ValueError(
                "warm_start.kind is null but other warm-start fields are populated: "
                f"{sorted(populated)}."
            )
        return False
    return True


def _load_model_config(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Source config must contain a mapping: {path}.")
    model_cfg = payload.get("model", payload)
    if not isinstance(model_cfg, Mapping):
        raise ValueError(f"Source config has no mapping-valued `model` block: {path}.")
    return dict(model_cfg)


def _state_schema(state: Mapping[str, torch.Tensor]) -> dict[str, tuple[tuple[int, ...], str]]:
    return {
        str(key): (tuple(value.shape), str(value.dtype))
        for key, value in state.items()
    }


def _assert_state_compatible(
    *, source: Mapping[str, torch.Tensor], target: Mapping[str, torch.Tensor], name: str
) -> None:
    source_schema = _state_schema(source)
    target_schema = _state_schema(target)
    if source_schema != target_schema:
        missing = sorted(set(target_schema) - set(source_schema))
        unexpected = sorted(set(source_schema) - set(target_schema))
        mismatched = {
            key: (source_schema[key], target_schema[key])
            for key in sorted(set(source_schema) & set(target_schema))
            if source_schema[key] != target_schema[key]
        }
        raise ValueError(
            f"Warm-start {name} state is incompatible: missing={missing}, "
            f"unexpected={unexpected}, mismatched={mismatched}."
        )


def _task_family(task: str) -> str:
    family = task.split("_", 1)[0].strip().lower()
    if family not in {"libero", "robotwin"}:
        raise ValueError(
            f"Cannot infer benchmark family from task {task!r}; expected libero_* or robotwin_*."
        )
    return family


def strict_standalone_idm_warm_start(
    model,
    config: Any,
    *,
    target_model_config: Any,
    target_dataset_stats: str | os.PathLike[str],
) -> dict[str, Any]:
    """Import only model tensors from a fully verified standalone IDM checkpoint.

    Optimizer, scheduler, source step and loaded-checkpoint identity are never
    restored. The returned record is embedded into later adaptive checkpoints.
    """
    plain = _plain_mapping(config, name="warm_start")
    if set(plain) != _REQUIRED_KEYS:
        raise ValueError(
            "warm_start keys must be exactly "
            f"{sorted(_REQUIRED_KEYS)}, got {sorted(plain)}."
        )
    if plain.get("kind") != _KIND:
        raise ValueError(
            f"warm_start.kind must be {_KIND!r}, got {plain.get('kind')!r}."
        )
    missing_values = sorted(key for key in _REQUIRED_KEYS if plain.get(key) in (None, ""))
    if missing_values:
        raise ValueError(f"Warm-start fields cannot be empty: {missing_values}.")

    if tuple(getattr(model, "adaptive_regimes", ())) != ("uncond", "idm") or (
        getattr(model, "adaptive_backbone_kind", None) != "idm"
    ):
        raise ValueError("Warm-start target must be the shared UNCOND+IDM model.")
    if type(model).__name__ != "FusedDualRegimeFastWAM":
        raise ValueError(
            "Production warm-start target must be FusedDualRegimeFastWAM, got "
            f"{type(model).__name__}."
        )

    checkpoint_path = Path(os.path.expanduser(str(plain["checkpoint"]))).resolve()
    source_config_path = Path(os.path.expanduser(str(plain["source_config"]))).resolve()
    source_stats_path = Path(os.path.expanduser(str(plain["source_dataset_stats"]))).resolve()
    target_stats_path = Path(os.path.expanduser(os.fspath(target_dataset_stats))).resolve()
    for label, path in (
        ("checkpoint", checkpoint_path),
        ("source_config", source_config_path),
        ("source_dataset_stats", source_stats_path),
        ("target_dataset_stats", target_stats_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Warm-start {label} does not exist: {path}.")

    expected_sha = str(plain["expected_checkpoint_sha256"]).strip().lower()
    if len(expected_sha) != 64 or any(char not in "0123456789abcdef" for char in expected_sha):
        raise ValueError("expected_checkpoint_sha256 must be a 64-character lowercase SHA256.")
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != expected_sha:
        raise ValueError(
            "Standalone IDM checkpoint SHA256 mismatch: "
            f"expected={expected_sha}, actual={checkpoint_sha}."
        )

    source_task = str(plain["source_task"])
    target_task = getattr(model, "checkpoint_task", None)
    if not isinstance(target_task, str) or not target_task:
        raise ValueError("Warm-start target must declare a non-empty checkpoint_task.")
    if _task_family(source_task) != _task_family(target_task):
        raise ValueError(
            f"Warm-start task family mismatch: source={source_task!r}, target={target_task!r}."
        )

    source_model_cfg = _load_model_config(source_config_path)
    target_model_cfg = _plain_mapping(target_model_config, name="target model config")
    if source_model_cfg.get("_target_") != "fastwam.runtime.create_fastwam_idm":
        raise ValueError(
            "Warm-start source config must instantiate fastwam.runtime.create_fastwam_idm."
        )
    configured_source_task = source_model_cfg.get("checkpoint_task")
    if configured_source_task not in (None, source_task):
        raise ValueError(
            "Source config checkpoint_task does not match warm_start.source_task: "
            f"{configured_source_task!r} != {source_task!r}."
        )
    architecture_mismatches = {
        key: (source_model_cfg.get(key), target_model_cfg.get(key))
        for key in _ARCHITECTURE_KEYS
        if source_model_cfg.get(key) != target_model_cfg.get(key)
    }
    if architecture_mismatches:
        raise ValueError(
            "Standalone IDM and dual-regime architecture configs differ "
            f"(source, target): {architecture_mismatches}."
        )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "mot" not in payload:
        raise ValueError("Standalone IDM warm-start checkpoint must contain a `mot` state dict.")
    raw_provenance = payload.get("fastwam_provenance")
    has_checkpoint_provenance = raw_provenance is not None
    if has_checkpoint_provenance and not isinstance(raw_provenance, Mapping):
        raise ValueError("Malformed standalone IDM FastWAM provenance metadata.")
    provenance: Mapping[str, Any] = raw_provenance or {}
    if has_checkpoint_provenance:
        if provenance.get("schema_version") != 2 or not provenance.get("checkpoint_id"):
            raise ValueError(
                "Standalone IDM checkpoint provenance must use schema_version=2 "
                "and contain checkpoint_id."
            )
        if provenance.get("model_class") != "FastWAMIDM":
            raise ValueError(
                "Warm-start source must be a standalone FastWAMIDM checkpoint, got "
                f"{provenance.get('model_class')!r}."
            )
        if tuple(provenance.get("adaptive_regimes", ())) or provenance.get(
            "adaptive_backbone_kind"
        ) is not None:
            raise ValueError("Warm-start source must not already be an adaptive checkpoint.")
    provenance_task = provenance.get("task")
    if provenance_task not in (None, source_task):
        raise ValueError(
            f"Checkpoint task {provenance_task!r} != source task {source_task!r}."
        )
    expected_dims = {
        "action_dim": int(model.action_expert.action_dim),
        "proprio_dim": getattr(model, "proprio_dim", None),
        "video_latent_dim": int(model.vae.model.z_dim),
    }
    if has_checkpoint_provenance:
        dimension_mismatches = {
            key: (provenance.get(key), expected)
            for key, expected in expected_dims.items()
            if provenance.get(key) != expected
        }
        if dimension_mismatches:
            raise ValueError(
                "Standalone IDM checkpoint dimensions do not match the target "
                f"(source, target): {dimension_mismatches}."
            )

    source_stats_sha = sha256_file(source_stats_path)
    target_stats_sha = sha256_file(target_stats_path)
    if has_checkpoint_provenance and provenance.get("dataset_stats_fingerprint") != source_stats_sha:
        raise ValueError("Source dataset-stats SHA does not match checkpoint provenance.")
    if target_stats_sha != source_stats_sha:
        raise ValueError(
            "Target training dataset stats differ from the standalone IDM stats: "
            f"source={source_stats_sha}, target={target_stats_sha}."
        )

    source_contract = provenance.get("model_contract")
    target_contract = checkpoint_model_contract(model)
    if source_contract is not None and source_contract != target_contract:
        raise ValueError(
            "Standalone IDM checkpoint model contract does not match the target dual-regime model."
        )

    _assert_state_compatible(source=payload["mot"], target=model.mot.state_dict(), name="mot")
    source_has_proprio = "proprio_encoder" in payload
    target_proprio = getattr(model, "proprio_encoder", None)
    if source_has_proprio != (target_proprio is not None):
        raise ValueError(
            "Warm-start proprio presence mismatch: "
            f"source={source_has_proprio}, target={target_proprio is not None}."
        )
    if target_proprio is not None:
        _assert_state_compatible(
            source=payload["proprio_encoder"],
            target=target_proprio.state_dict(),
            name="proprio_encoder",
        )

    model.mot.load_state_dict(payload["mot"], strict=True)
    if target_proprio is not None:
        target_proprio.load_state_dict(payload["proprio_encoder"], strict=True)

    record = {
        "initialization_type": _KIND,
        "parent_checkpoint_id": str(
            provenance.get("checkpoint_id", f"standalone-idm-{checkpoint_sha[:16]}")
        ),
        "parent_checkpoint_sha256": checkpoint_sha,
        "parent_checkpoint_path": str(checkpoint_path),
        "parent_task": source_task,
        "task_binding": (
            "checkpoint_provenance"
            if has_checkpoint_provenance and provenance_task == source_task
            else "explicit_source_task_and_hashed_config"
        ),
        "model_contract_binding": (
            "checkpoint_provenance"
            if source_contract is not None
            else "hashed_config_and_strict_state_schema"
        ),
        "parent_step": payload.get("step"),
        "source_config_sha256": sha256_file(source_config_path),
        "parent_config_sha256": sha256_file(source_config_path),
        "parent_dataset_stats_sha256": source_stats_sha,
        "dataset_stats_fingerprint": source_stats_sha,
        "source_checkpoint_provenance_present": has_checkpoint_provenance,
        "dataset_stats_binding": (
            "checkpoint_provenance"
            if has_checkpoint_provenance
            else "explicit_hashed_source_and_target_artifacts"
        ),
    }
    model.warm_start_provenance = record
    model.dual_regime_optimizer_steps = 0
    return record
