"""Strict V2 LoRA-plus-reader checkpoints for registered visual backbones."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from fastwam.adapters import (
    REGIME_LORA_SIDECAR_SCHEMA,
    ActionDiTLoRAAdapter,
    sha256_file,
)
from fastwam.models.wan22.visual_backbone import (
    VISUAL_BACKBONE_METADATA_SCHEMA,
    get_visual_backbone_preset,
)
from fastwam.models.wan22.visual_contracts import (
    ActionVisualReader,
    VISUAL_READER_STATE_SCHEMA_V2,
    validate_sha256,
)
from fastwam.uncond_bc_checkpoint import build_lora_sidecar_payload

P1_VISUAL_BC_CHECKPOINT_SCHEMA = "fastwam-p1-visual-bc-checkpoint-v2"

_CHECKPOINT_KEYS = {
    "schema",
    "global_step",
    "stage",
    "arm",
    "parent_checkpoint_sha256",
    "visual_backbone",
    "memory_contract_sha256",
    "reader_contract_sha256",
    "adapter",
    "reader",
    "contract",
    "provenance",
    "trainer_state",
}
_VISUAL_METADATA_KEYS = {
    "schema",
    "family",
    "variant",
    "model_name",
    "input_size",
    "patch_size",
    "patch_grid",
    "patch_count",
    "native_dim",
    "depth",
    "source_root",
    "source_revision",
    "weights_repo_id",
    "weights_revision",
    "weights_path",
    "weights_sha256",
    "asset_contract_sha256",
    "input_contract_sha256",
    "preprocess_sha256",
    "output_contract_sha256",
    "memory_contract_sha256",
    "compute_dtype",
    "encode_microbatch_size",
    "license_id",
}
_TRAINER_STATE_KEYS = {
    "last_loss_action_bc",
    "best_dev_loss_action_bc",
    "nonzero_update_count",
}


def _cpu_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return value


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


def validate_visual_backbone_metadata(payload: Any) -> dict[str, Any]:
    """Validate the complete non-tensor backbone identity stored in V2."""

    if not isinstance(payload, Mapping) or set(payload) != _VISUAL_METADATA_KEYS:
        keys = sorted(payload) if isinstance(payload, Mapping) else type(payload)
        raise ValueError(f"Invalid V2 visual-backbone metadata fields: {keys}.")
    result = dict(payload)
    if result["schema"] != VISUAL_BACKBONE_METADATA_SCHEMA:
        raise ValueError("V2 visual-backbone metadata schema changed.")
    preset = get_visual_backbone_preset(
        result["family"],
        result["variant"],
        input_size=int(result["input_size"]),
    )
    if (
        result["model_name"] != preset.model_name
        or int(result["native_dim"]) != preset.native_dim
        or int(result["depth"]) != preset.depth
        or int(result["patch_size"]) != 16
        or result["weights_repo_id"] != preset.weights_repo_id
        or result["weights_revision"] != preset.weights_revision
        or result["source_revision"] != preset.source_revision
        or result["license_id"] != preset.license_id
    ):
        raise ValueError("V2 visual-backbone metadata differs from the registry.")
    grid = tuple(int(value) for value in result["patch_grid"])
    expected_grid = (int(result["input_size"]) // 16,) * 2
    if grid != expected_grid or int(result["patch_count"]) != grid[0] * grid[1]:
        raise ValueError("V2 visual patch grid/count is inconsistent.")
    for key in (
        "weights_sha256",
        "asset_contract_sha256",
        "input_contract_sha256",
        "preprocess_sha256",
        "output_contract_sha256",
        "memory_contract_sha256",
    ):
        result[key] = validate_sha256(result[key], label=key)
    if result["weights_sha256"] != preset.weights_sha256:
        raise ValueError("V2 visual weights hash differs from the registry.")
    if (
        not str(result["source_root"]).strip()
        or not str(result["weights_path"]).strip()
    ):
        raise ValueError("V2 visual local asset paths cannot be empty.")
    if int(result["encode_microbatch_size"]) < 1:
        raise ValueError("V2 visual encode microbatch size must be positive.")
    return result


def _validate_trainer_state(
    payload: Any,
    *,
    global_step: int,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != _TRAINER_STATE_KEYS:
        raise ValueError("V2 compact trainer-state keys changed.")
    result: dict[str, Any] = {}
    for name in ("last_loss_action_bc", "best_dev_loss_action_bc"):
        value = payload[name]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(f"V2 trainer state {name!r} is invalid.")
        result[name] = None if value is None else float(value)
    updates = payload["nonzero_update_count"]
    if (
        isinstance(updates, bool)
        or not isinstance(updates, int)
        or not 0 <= updates <= global_step
    ):
        raise ValueError("V2 nonzero updates must lie in [0, global_step].")
    result["nonzero_update_count"] = updates
    return result


def _validate_structure(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != _CHECKPOINT_KEYS:
        keys = sorted(payload) if isinstance(payload, dict) else type(payload)
        raise ValueError(f"Not a complete V2 visual BC checkpoint: {keys}.")
    if payload["schema"] != P1_VISUAL_BC_CHECKPOINT_SCHEMA:
        raise ValueError(f"Unsupported V2 visual schema {payload['schema']!r}.")
    step = payload["global_step"]
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("V2 checkpoint global_step must be non-negative.")
    if not str(payload["stage"]).strip() or not str(payload["arm"]).strip():
        raise ValueError("V2 checkpoint stage/arm cannot be empty.")
    payload["parent_checkpoint_sha256"] = validate_sha256(
        payload["parent_checkpoint_sha256"],
        label="parent_checkpoint_sha256",
    )
    payload["memory_contract_sha256"] = validate_sha256(
        payload["memory_contract_sha256"],
        label="memory_contract_sha256",
    )
    payload["reader_contract_sha256"] = validate_sha256(
        payload["reader_contract_sha256"],
        label="reader_contract_sha256",
    )
    payload["visual_backbone"] = validate_visual_backbone_metadata(
        payload["visual_backbone"]
    )
    if (
        payload["visual_backbone"]["memory_contract_sha256"]
        != payload["memory_contract_sha256"]
    ):
        raise ValueError("V2 visual and checkpoint memory hashes disagree.")
    if not isinstance(payload["contract"], Mapping) or not isinstance(
        payload["provenance"], Mapping
    ):
        raise TypeError("V2 contract and provenance must be mappings.")
    payload["trainer_state"] = _validate_trainer_state(
        payload["trainer_state"],
        global_step=step,
    )
    adapter = payload["adapter"]
    if not isinstance(adapter, Mapping) or set(adapter) != {
        "metadata",
        "state_dict",
    }:
        raise ValueError("V2 checkpoint is missing its strict LoRA sidecar.")
    if adapter["metadata"].get("schema") != REGIME_LORA_SIDECAR_SCHEMA:
        raise ValueError("V2 LoRA sidecar schema changed.")
    state = adapter["state_dict"]
    if (
        not isinstance(state, Mapping)
        or not state
        or any(not str(name).endswith((".lora_A", ".lora_B")) for name in state)
    ):
        raise ValueError("V2 adapter state must contain LoRA tensors only.")
    reader = payload["reader"]
    expected_reader_keys = {
        "schema",
        "reader_kind",
        "reader_contract_sha256",
        "parameter_names",
        "state",
    }
    if not isinstance(reader, Mapping) or set(reader) != expected_reader_keys:
        raise ValueError("V2 checkpoint reader state is malformed.")
    if reader["schema"] != VISUAL_READER_STATE_SCHEMA_V2:
        raise ValueError("V2 checkpoint requires the V2 reader state schema.")
    if reader["reader_contract_sha256"] != payload["reader_contract_sha256"]:
        raise ValueError("V2 reader contract hashes disagree.")
    names = tuple(reader["parameter_names"])
    if not names or tuple(reader["state"]) != names:
        raise ValueError("V2 reader state manifest is empty or inconsistent.")
    tensor_entries = list(_walk_tensors(payload))
    forbidden = sorted(
        name
        for name, _ in tensor_entries
        if not name.startswith(("adapter.state_dict.", "reader.state."))
    )
    if forbidden:
        raise ValueError(
            f"V2 compact checkpoint has forbidden tensors: {forbidden[:16]}."
        )
    return payload


def save_p1_visual_bc_checkpoint(
    path: str | os.PathLike[str],
    *,
    adapter: ActionDiTLoRAAdapter,
    reader: ActionVisualReader,
    global_step: int,
    stage: str,
    arm: str,
    parent_checkpoint_sha256: str,
    visual_backbone: Mapping[str, Any],
    memory_contract_sha256: str,
    contract: Mapping[str, Any],
    provenance: Mapping[str, Any],
    trainer_state: Mapping[str, Any],
) -> None:
    """Atomically save V2 LoRA/reader state without either frozen model."""

    parent_hash = validate_sha256(
        parent_checkpoint_sha256,
        label="parent_checkpoint_sha256",
    )
    memory_hash = validate_sha256(
        memory_contract_sha256,
        label="memory_contract_sha256",
    )
    metadata = validate_visual_backbone_metadata(visual_backbone)
    if reader.memory_contract_sha256 != memory_hash:
        raise ValueError("V2 reader and memory contracts differ.")
    step = int(global_step)
    payload = {
        "schema": P1_VISUAL_BC_CHECKPOINT_SCHEMA,
        "global_step": step,
        "stage": str(stage),
        "arm": str(arm),
        "parent_checkpoint_sha256": parent_hash,
        "visual_backbone": metadata,
        "memory_contract_sha256": memory_hash,
        "reader_contract_sha256": reader.reader_contract_sha256,
        "adapter": build_lora_sidecar_payload(
            adapter,
            parent_checkpoint_sha256=parent_hash,
            extra_metadata={
                "p1_visual_step": step,
                "p1_visual_reader_contract_sha256": reader.reader_contract_sha256,
                "p1_visual_asset_contract_sha256": metadata["asset_contract_sha256"],
            },
        ),
        "reader": reader.export_trainable_state(),
        "contract": dict(contract),
        "provenance": dict(provenance),
        "trainer_state": dict(trainer_state),
    }
    payload = _validate_structure(_cpu_clone(payload))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def load_p1_visual_bc_checkpoint(
    path: str | os.PathLike[str],
    *,
    adapter: ActionDiTLoRAAdapter,
    reader: ActionVisualReader,
    expected_parent_checkpoint_sha256: str,
    expected_visual_backbone: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly restore V2 state after exact backbone/contract matching."""

    payload = _validate_structure(
        torch.load(path, map_location="cpu", weights_only=False)
    )
    expected = {
        "parent_checkpoint_sha256": validate_sha256(
            expected_parent_checkpoint_sha256,
            label="expected parent checkpoint SHA256",
        ),
        "visual_backbone": validate_visual_backbone_metadata(expected_visual_backbone),
        "memory_contract_sha256": reader.memory_contract_sha256,
        "reader_contract_sha256": reader.reader_contract_sha256,
        "contract": dict(expected_contract),
    }
    mismatches = {
        key: (value, payload[key])
        for key, value in expected.items()
        if payload[key] != value
    }
    if mismatches:
        raise ValueError(f"V2 visual checkpoint contract mismatch: {mismatches}.")
    adapter.load_lora_state_dict(payload["adapter"]["state_dict"], strict=True)
    reader.load_trainable_state(payload["reader"])
    return payload


def inspect_p1_visual_bc_checkpoint(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Audit a V2 compact artifact without allocating frozen models."""

    payload = _validate_structure(
        torch.load(path, map_location="cpu", weights_only=False)
    )
    entries = list(_walk_tensors(payload))
    return {
        "schema": payload["schema"],
        "global_step": payload["global_step"],
        "stage": payload["stage"],
        "arm": payload["arm"],
        "checkpoint_sha256": sha256_file(path),
        "parent_checkpoint_sha256": payload["parent_checkpoint_sha256"],
        "visual_backbone": payload["visual_backbone"],
        "memory_contract_sha256": payload["memory_contract_sha256"],
        "reader_contract_sha256": payload["reader_contract_sha256"],
        "lora_tensor_count": sum(
            name.startswith("adapter.state_dict.") for name, _ in entries
        ),
        "reader_tensor_count": sum(
            name.startswith("reader.state.") for name, _ in entries
        ),
        "trainable_bytes": sum(
            tensor.numel() * tensor.element_size() for _, tensor in entries
        ),
        "contains_frozen_fastwam_tensors": False,
        "contains_visual_backbone_tensors": False,
        "contains_gate_tensors": False,
        "contains_value_head_tensors": False,
        "contains_raw_training_samples": False,
        "forbidden_tensor_paths": [],
        "result": "PASS",
    }
