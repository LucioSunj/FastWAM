"""Atomic LoRA-only checkpoints for FastWAM UNCOND behavior cloning."""

from __future__ import annotations

import math
import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fastwam.adapters import (
    REGIME_LORA_SIDECAR_SCHEMA,
    ActionDiTLoRAAdapter,
    sha256_file,
)

UNCOND_BC_TRAINING_SCHEMA = "fastwam-uncond-bc-training-v1"
_TRAINING_CHECKPOINT_KEYS = {
    "schema",
    "global_step",
    "epoch",
    "sampler_offset",
    "parent_checkpoint_sha256",
    "adapter",
    "optimizer",
    "lr_scheduler",
    "grad_scaler",
    "rng_by_rank",
    "contract",
    "provenance",
    "trainer_state",
}
_ADAPTER_KEYS = {"metadata", "state_dict"}
_TRAINER_STATE_KEYS = {
    "best_validation_loss_action_bc",
    "best_step",
    "epochs_without_improvement",
    "nonzero_update_count",
}
_STRUCTURAL_ADAPTER_METADATA_KEYS = {
    "schema",
    "parent_checkpoint_sha256",
    "active_regime",
    "rank",
    "alpha",
    "dropout",
    "target_groups",
    "target_names",
}


def _cpu_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return value


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, CPU Torch, and all visible CUDA RNG streams."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
    }


def restore_rng_state(payload: Mapping[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state` exactly."""

    expected = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(payload) != expected:
        raise ValueError(f"BC RNG state keys changed: {sorted(payload)}.")
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch_cpu"])
    cuda_states = payload["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint has CUDA RNG state but CUDA is unavailable.")
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "CUDA RNG device count mismatch: "
                f"checkpoint={len(cuda_states)}, runtime={torch.cuda.device_count()}."
            )
        torch.cuda.set_rng_state_all(cuda_states)


def build_lora_sidecar_payload(
    adapter: ActionDiTLoRAAdapter,
    *,
    parent_checkpoint_sha256: str,
    extra_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the existing v1 sidecar payload without writing a second file."""

    return {
        "metadata": adapter.sidecar_metadata(
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            extra_metadata=extra_metadata,
        ),
        "state_dict": adapter.lora_state_dict(),
    }


def _validate_adapter_payload(
    adapter: ActionDiTLoRAAdapter,
    payload: Mapping[str, Any],
    *,
    expected_parent_checkpoint_sha256: str,
) -> dict[str, Any]:
    if set(payload) != _ADAPTER_KEYS:
        raise ValueError(f"BC adapter payload keys changed: {sorted(payload)}.")
    metadata = payload["metadata"]
    state_dict = payload["state_dict"]
    if not isinstance(metadata, Mapping) or not isinstance(state_dict, Mapping):
        raise TypeError("BC adapter metadata and state_dict must be mappings.")
    non_fp32 = sorted(
        name
        for name, value in state_dict.items()
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32
    )
    if non_fp32:
        raise ValueError(
            f"BC adapter must contain FP32 LoRA master tensors: {non_fp32[:16]}."
        )
    expected_metadata = adapter.sidecar_metadata(
        parent_checkpoint_sha256=expected_parent_checkpoint_sha256,
        extra_metadata=metadata.get("extra", {}),
    )
    mismatches = {
        key: (expected_metadata.get(key), metadata.get(key))
        for key in sorted(_STRUCTURAL_ADAPTER_METADATA_KEYS)
        if metadata.get(key) != expected_metadata.get(key)
    }
    if mismatches:
        raise ValueError(f"BC LoRA sidecar contract mismatch: {mismatches}.")
    adapter.load_lora_state_dict(state_dict, strict=True)
    return dict(metadata)


def _validate_trainer_state(
    payload: Mapping[str, Any] | None,
    *,
    global_step: int,
) -> dict[str, Any]:
    if payload is None:
        payload = {
            "best_validation_loss_action_bc": None,
            "best_step": None,
            "epochs_without_improvement": 0,
            "nonzero_update_count": 0,
        }
    if not isinstance(payload, Mapping) or set(payload) != _TRAINER_STATE_KEYS:
        keys = sorted(payload) if isinstance(payload, Mapping) else type(payload)
        raise ValueError(f"BC trainer-state keys changed: {keys}.")
    best_loss = payload["best_validation_loss_action_bc"]
    if best_loss is not None and (
        isinstance(best_loss, bool)
        or not isinstance(best_loss, (int, float))
        or not math.isfinite(float(best_loss))
        or float(best_loss) < 0
    ):
        raise ValueError(
            "BC best validation loss must be finite, non-negative, or null."
        )
    best_step = payload["best_step"]
    if best_step is not None and (
        isinstance(best_step, bool)
        or not isinstance(best_step, int)
        or not 0 <= best_step <= int(global_step)
    ):
        raise ValueError("BC best_step must be null or lie within [0, global_step].")
    counters = {
        name: payload[name]
        for name in ("epochs_without_improvement", "nonzero_update_count")
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counters.values()
    ):
        raise ValueError(f"BC trainer-state counters are invalid: {counters}.")
    if counters["nonzero_update_count"] > int(global_step):
        raise ValueError("BC nonzero_update_count cannot exceed global_step.")
    return {
        "best_validation_loss_action_bc": (
            None if best_loss is None else float(best_loss)
        ),
        "best_step": best_step,
        **counters,
    }


def save_uncond_bc_checkpoint(
    path: str | os.PathLike[str],
    *,
    adapter: ActionDiTLoRAAdapter,
    parent_checkpoint_sha256: str,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    grad_scaler: Any,
    global_step: int,
    epoch: int,
    sampler_offset: int,
    rng_by_rank: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
    provenance: Mapping[str, Any],
    trainer_state: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save LoRA plus resumable trainer state, never frozen weights."""

    counters = {
        "global_step": global_step,
        "epoch": epoch,
        "sampler_offset": sampler_offset,
    }
    invalid = {name: value for name, value in counters.items() if int(value) < 0}
    if invalid:
        raise ValueError(f"BC checkpoint counters must be non-negative: {invalid}.")
    if not rng_by_rank:
        raise ValueError("BC checkpoint requires RNG state for every rank.")
    normalized_trainer_state = _validate_trainer_state(
        trainer_state,
        global_step=int(global_step),
    )
    sidecar_extra = {
        "bc_step": int(global_step),
        "bc_config_sha256": str(contract.get("resolved_config_sha256", "")),
    }
    payload = {
        "schema": UNCOND_BC_TRAINING_SCHEMA,
        "global_step": int(global_step),
        "epoch": int(epoch),
        "sampler_offset": int(sampler_offset),
        "parent_checkpoint_sha256": str(parent_checkpoint_sha256).lower(),
        "adapter": build_lora_sidecar_payload(
            adapter,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            extra_metadata=sidecar_extra,
        ),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "grad_scaler": grad_scaler.state_dict(),
        "rng_by_rank": list(rng_by_rank),
        "contract": dict(contract),
        "provenance": dict(provenance),
        "trainer_state": normalized_trainer_state,
    }
    payload = _cpu_clone(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def load_uncond_bc_checkpoint(
    path: str | os.PathLike[str],
    *,
    adapter: ActionDiTLoRAAdapter,
    expected_parent_checkpoint_sha256: str,
    expected_contract: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    grad_scaler: Any,
) -> dict[str, Any]:
    """Strictly restore the complete BC trainer state except rank-local RNG."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or set(payload) != _TRAINING_CHECKPOINT_KEYS:
        keys = sorted(payload) if isinstance(payload, dict) else type(payload)
        raise ValueError(f"BC training checkpoint keys changed: {keys}.")
    if payload.get("schema") != UNCOND_BC_TRAINING_SCHEMA:
        raise ValueError(f"Unsupported BC checkpoint schema {payload.get('schema')!r}.")
    expected_parent = str(expected_parent_checkpoint_sha256).lower()
    if payload.get("parent_checkpoint_sha256") != expected_parent:
        raise ValueError(
            "BC checkpoint parent hash mismatch: "
            f"expected {expected_parent}, got {payload.get('parent_checkpoint_sha256')}."
        )
    if payload.get("contract") != dict(expected_contract):
        raise ValueError("BC checkpoint resolved config/data contract mismatch.")
    metadata = _validate_adapter_payload(
        adapter,
        payload["adapter"],
        expected_parent_checkpoint_sha256=expected_parent,
    )
    if metadata.get("extra", {}).get("bc_step") != int(payload["global_step"]):
        raise ValueError("BC sidecar step does not match its training checkpoint.")
    payload["trainer_state"] = _validate_trainer_state(
        payload["trainer_state"],
        global_step=int(payload["global_step"]),
    )
    optimizer.load_state_dict(payload["optimizer"])
    lr_scheduler.load_state_dict(payload["lr_scheduler"])
    grad_scaler.load_state_dict(payload["grad_scaler"])
    return payload


def load_uncond_bc_adapter_checkpoint(
    path: str | os.PathLike[str],
    *,
    adapter: ActionDiTLoRAAdapter,
    expected_parent_checkpoint_sha256: str,
) -> dict[str, Any]:
    """Strictly load only the LoRA state from a complete trainer checkpoint."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or set(payload) != _TRAINING_CHECKPOINT_KEYS:
        keys = sorted(payload) if isinstance(payload, dict) else type(payload)
        raise ValueError(f"BC training checkpoint keys changed: {keys}.")
    if payload.get("schema") != UNCOND_BC_TRAINING_SCHEMA:
        raise ValueError(f"Unsupported BC checkpoint schema {payload.get('schema')!r}.")
    expected_parent = str(expected_parent_checkpoint_sha256).lower()
    if payload.get("parent_checkpoint_sha256") != expected_parent:
        raise ValueError(
            "BC checkpoint parent hash mismatch: "
            f"expected {expected_parent}, got {payload.get('parent_checkpoint_sha256')}."
        )
    global_step = int(payload["global_step"])
    trainer_state = _validate_trainer_state(
        payload["trainer_state"],
        global_step=global_step,
    )
    metadata = _validate_adapter_payload(
        adapter,
        payload["adapter"],
        expected_parent_checkpoint_sha256=expected_parent,
    )
    if metadata.get("extra", {}).get("bc_step") != global_step:
        raise ValueError("BC sidecar step does not match its training checkpoint.")
    return {
        "schema": payload["schema"],
        "global_step": global_step,
        "epoch": int(payload["epoch"]),
        "sampler_offset": int(payload["sampler_offset"]),
        "parent_checkpoint_sha256": expected_parent,
        "adapter_metadata": metadata,
        "contract": dict(payload["contract"]),
        "provenance": dict(payload["provenance"]),
        "trainer_state": trainer_state,
    }


def _walk_tensors(value: Any, *, prefix: str = ""):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk_tensors(value[key], prefix=child)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            child = f"{prefix}.{index}" if prefix else str(index)
            yield from _walk_tensors(item, prefix=child)


def inspect_uncond_bc_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return a machine-readable payload audit and fail closed on model data."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or set(payload) != _TRAINING_CHECKPOINT_KEYS:
        raise ValueError("Not a complete FastWAM UNCOND BC training checkpoint.")
    trainer_state = _validate_trainer_state(
        payload["trainer_state"],
        global_step=int(payload["global_step"]),
    )
    adapter = payload.get("adapter")
    if not isinstance(adapter, Mapping) or set(adapter) != _ADAPTER_KEYS:
        raise ValueError("BC checkpoint is missing its v1 adapter sidecar payload.")
    metadata = adapter["metadata"]
    state_dict = adapter["state_dict"]
    if metadata.get("schema") != REGIME_LORA_SIDECAR_SCHEMA:
        raise ValueError("BC checkpoint does not contain a v1 regime-LoRA sidecar.")
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError("BC checkpoint adapter state is empty or malformed.")
    invalid_lora_names = sorted(
        name for name in state_dict if not str(name).endswith((".lora_A", ".lora_B"))
    )
    if invalid_lora_names:
        raise ValueError(f"BC sidecar contains non-LoRA tensors: {invalid_lora_names}.")

    allowed_tensor_prefixes = (
        "adapter.state_dict.",
        "optimizer.",
        "lr_scheduler.",
        "grad_scaler.",
        "rng_by_rank.",
    )
    tensor_entries = list(_walk_tensors(payload))
    forbidden_tensor_paths = sorted(
        name
        for name, _ in tensor_entries
        if not name.startswith(allowed_tensor_prefixes)
    )
    if forbidden_tensor_paths:
        raise ValueError(
            "BC checkpoint contains tensors outside LoRA/trainer state: "
            f"{forbidden_tensor_paths[:16]}."
        )
    lowered_tensor_paths = [name.lower() for name, _ in tensor_entries]
    forbidden_fragments = (
        "gate",
        "value_head",
        "video_expert",
        "proprio_encoder",
        "image",
        "observation",
        "action_sample",
        "kv_cache",
    )
    semantic_forbidden = sorted(
        name
        for name in lowered_tensor_paths
        if any(fragment in name for fragment in forbidden_fragments)
    )
    if semantic_forbidden:
        raise ValueError(
            "BC checkpoint contains forbidden model/data tensor names: "
            f"{semantic_forbidden[:16]}."
        )

    lora_tensors = [
        value for value in state_dict.values() if isinstance(value, torch.Tensor)
    ]
    non_fp32_lora = sorted(
        name
        for name, value in state_dict.items()
        if isinstance(value, torch.Tensor) and value.dtype != torch.float32
    )
    if non_fp32_lora:
        raise ValueError(
            f"BC checkpoint LoRA master tensors are not FP32: {non_fp32_lora[:16]}."
        )
    optimizer_tensors = [
        (name, value)
        for name, value in tensor_entries
        if name.startswith("optimizer.") and torch.is_floating_point(value)
    ]
    non_fp32_optimizer = sorted(
        name for name, value in optimizer_tensors if value.dtype != torch.float32
    )
    if non_fp32_optimizer:
        raise ValueError(
            "BC checkpoint optimizer floating-point state is not FP32: "
            f"{non_fp32_optimizer[:16]}."
        )
    lora_bytes = sum(value.numel() * value.element_size() for value in lora_tensors)
    return {
        "schema": payload["schema"],
        "global_step": int(payload["global_step"]),
        "epoch": int(payload["epoch"]),
        "sampler_offset": int(payload["sampler_offset"]),
        "trainer_state": trainer_state,
        "parent_checkpoint_sha256": payload["parent_checkpoint_sha256"],
        "checkpoint_sha256": sha256_file(path),
        "adapter_schema": metadata["schema"],
        "adapter_rank": int(metadata["rank"]),
        "adapter_alpha": float(metadata["alpha"]),
        "adapter_target_names": list(metadata["target_names"]),
        "lora_tensor_count": len(lora_tensors),
        "lora_bytes": int(lora_bytes),
        "lora_master_dtype": "torch.float32",
        "optimizer_tensor_count": sum(
            1 for name, _ in tensor_entries if name.startswith("optimizer.")
        ),
        "optimizer_floating_state_dtype": "torch.float32",
        "rng_rank_count": len(payload["rng_by_rank"]),
        "forbidden_tensor_paths": [],
        "contains_frozen_fastwam_tensors": False,
        "contains_gate_tensors": False,
        "contains_value_head_tensors": False,
        "contains_raw_training_samples": False,
        "result": "PASS",
    }


def _nested_mismatch_paths(first: Any, second: Any, *, path: str) -> list[str]:
    if isinstance(first, torch.Tensor):
        if isinstance(second, torch.Tensor) and torch.equal(first, second):
            return []
        return [path]
    if isinstance(first, np.ndarray):
        if isinstance(second, np.ndarray) and np.array_equal(
            first, second, equal_nan=True
        ):
            return []
        return [path]
    if isinstance(first, Mapping):
        if not isinstance(second, Mapping):
            return [f"{path}.__type__"]
        mismatches = [
            f"{path}.{key}.__missing_second__"
            for key in sorted(set(first) - set(second), key=str)
        ]
        mismatches.extend(
            f"{path}.{key}.__missing_first__"
            for key in sorted(set(second) - set(first), key=str)
        )
        for key in sorted(set(first) & set(second), key=str):
            mismatches.extend(
                _nested_mismatch_paths(
                    first[key],
                    second[key],
                    path=f"{path}.{key}",
                )
            )
        return mismatches
    if isinstance(first, (list, tuple)):
        if type(first) is not type(second) or len(first) != len(second):
            return [f"{path}.__sequence__"]
        mismatches = []
        for index, (left, right) in enumerate(zip(first, second, strict=True)):
            mismatches.extend(
                _nested_mismatch_paths(left, right, path=f"{path}.{index}")
            )
        return mismatches
    if (
        isinstance(first, float)
        and isinstance(second, float)
        and math.isnan(first)
        and math.isnan(second)
    ):
        return []
    return [] if first == second else [path]


def compare_uncond_bc_checkpoints(
    first_path: str | os.PathLike[str],
    second_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Compare all resumable training state while excluding launch provenance."""

    first_report = inspect_uncond_bc_checkpoint(first_path)
    second_report = inspect_uncond_bc_checkpoint(second_path)
    first = torch.load(first_path, map_location="cpu", weights_only=False)
    second = torch.load(second_path, map_location="cpu", weights_only=False)
    groups = {
        "adapter": (first["adapter"], second["adapter"]),
        "optimizer": (first["optimizer"], second["optimizer"]),
        "lr_scheduler": (first["lr_scheduler"], second["lr_scheduler"]),
        "grad_scaler": (first["grad_scaler"], second["grad_scaler"]),
        "rng_by_rank": (first["rng_by_rank"], second["rng_by_rank"]),
        "contract": (first["contract"], second["contract"]),
        "counters": (
            {
                key: first[key]
                for key in (
                    "schema",
                    "global_step",
                    "epoch",
                    "sampler_offset",
                    "parent_checkpoint_sha256",
                    "trainer_state",
                )
            },
            {
                key: second[key]
                for key in (
                    "schema",
                    "global_step",
                    "epoch",
                    "sampler_offset",
                    "parent_checkpoint_sha256",
                    "trainer_state",
                )
            },
        ),
    }
    group_reports = {}
    all_mismatches = []
    for name, (left, right) in groups.items():
        mismatches = _nested_mismatch_paths(left, right, path=name)
        group_reports[name] = {
            "exact": not mismatches,
            "mismatch_count": len(mismatches),
            "mismatch_paths": mismatches[:64],
        }
        all_mismatches.extend(mismatches)
    return {
        "schema": "fastwam-uncond-bc-checkpoint-comparison-v1",
        "result": "PASS" if not all_mismatches else "FAIL",
        "exact_training_state": not all_mismatches,
        "first": first_report,
        "second": second_report,
        "groups": group_reports,
        "excluded_paths": ["provenance"],
        "excluded_reason": (
            "Commands, output paths, repository dirtiness, and launch hashes are "
            "expected to differ between interrupted and uninterrupted launches."
        ),
        "mismatch_count": len(all_mismatches),
    }
