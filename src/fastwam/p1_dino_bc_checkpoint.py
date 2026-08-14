"""Strict LoRA-plus-reader artifacts for P1 DINO semantic-memory BC."""

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
from fastwam.models.wan22.visual_contracts import (
    ActionVisualReader,
    VISUAL_READER_STATE_SCHEMA,
    validate_sha256,
)
from fastwam.uncond_bc_checkpoint import build_lora_sidecar_payload

P1_DINO_BC_CHECKPOINT_SCHEMA = "fastwam-p1-dino-bc-checkpoint-v1"

_CHECKPOINT_KEYS = {
    "schema",
    "global_step",
    "stage",
    "arm",
    "parent_checkpoint_sha256",
    "dinov3_weights_sha256",
    "memory_contract_sha256",
    "reader_contract_sha256",
    "adapter",
    "reader",
    "contract",
    "provenance",
    "trainer_state",
}
_ADAPTER_KEYS = {"metadata", "state_dict"}
_READER_KEYS = {
    "schema",
    "reader_kind",
    "reader_contract_sha256",
    "parameter_names",
    "state",
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


def _validate_trainer_state(
    payload: Mapping[str, Any],
    *,
    global_step: int,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != _TRAINER_STATE_KEYS:
        keys = sorted(payload) if isinstance(payload, Mapping) else type(payload)
        raise ValueError(f"P1 trainer-state keys changed: {keys}.")
    result: dict[str, Any] = {}
    for name in ("last_loss_action_bc", "best_dev_loss_action_bc"):
        value = payload[name]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            raise ValueError(f"P1 trainer state {name!r} must be finite or null.")
        result[name] = None if value is None else float(value)
    updates = payload["nonzero_update_count"]
    if (
        isinstance(updates, bool)
        or not isinstance(updates, int)
        or not 0 <= updates <= global_step
    ):
        raise ValueError("P1 nonzero_update_count must lie within [0, global_step].")
    result["nonzero_update_count"] = int(updates)
    return result


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


def _validate_payload_structure(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != _CHECKPOINT_KEYS:
        keys = sorted(payload) if isinstance(payload, dict) else type(payload)
        raise ValueError(f"Not a complete P1 DINO BC checkpoint: {keys}.")
    if payload.get("schema") != P1_DINO_BC_CHECKPOINT_SCHEMA:
        raise ValueError(f"Unsupported P1 checkpoint schema {payload.get('schema')!r}.")
    global_step = payload["global_step"]
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < 0
    ):
        raise ValueError("P1 checkpoint global_step must be non-negative.")
    if not str(payload["stage"]).strip() or not str(payload["arm"]).strip():
        raise ValueError("P1 checkpoint stage and arm must be non-empty.")
    for key in (
        "parent_checkpoint_sha256",
        "dinov3_weights_sha256",
        "memory_contract_sha256",
        "reader_contract_sha256",
    ):
        payload[key] = validate_sha256(payload[key], label=key)
    if not isinstance(payload["contract"], Mapping):
        raise TypeError("P1 checkpoint contract must be a mapping.")
    if not isinstance(payload["provenance"], Mapping):
        raise TypeError("P1 checkpoint provenance must be a mapping.")
    payload["trainer_state"] = _validate_trainer_state(
        payload["trainer_state"],
        global_step=global_step,
    )
    adapter = payload["adapter"]
    if not isinstance(adapter, Mapping) or set(adapter) != _ADAPTER_KEYS:
        raise ValueError("P1 checkpoint is missing its strict LoRA sidecar.")
    metadata = adapter["metadata"]
    state = adapter["state_dict"]
    if not isinstance(metadata, Mapping) or not isinstance(state, Mapping) or not state:
        raise ValueError("P1 LoRA metadata/state must be non-empty mappings.")
    if metadata.get("schema") != REGIME_LORA_SIDECAR_SCHEMA:
        raise ValueError("P1 checkpoint LoRA sidecar schema changed.")
    invalid_lora = sorted(
        name for name in state if not str(name).endswith((".lora_A", ".lora_B"))
    )
    if invalid_lora:
        raise ValueError(
            f"P1 checkpoint contains non-LoRA adapter state: {invalid_lora}."
        )
    reader = payload["reader"]
    if not isinstance(reader, Mapping) or set(reader) != _READER_KEYS:
        raise ValueError("P1 checkpoint is missing its strict reader state.")
    if reader.get("schema") != VISUAL_READER_STATE_SCHEMA:
        raise ValueError("P1 checkpoint visual-reader state schema changed.")
    if reader.get("reader_contract_sha256") != payload["reader_contract_sha256"]:
        raise ValueError("P1 reader contract hashes disagree inside the checkpoint.")
    names = tuple(reader.get("parameter_names", ()))
    reader_state = reader.get("state")
    if (
        not names
        or not isinstance(reader_state, Mapping)
        or tuple(reader_state) != names
    ):
        raise ValueError("P1 reader manifest/state is empty or inconsistent.")

    tensors = list(_walk_tensors(payload))
    allowed_prefixes = ("adapter.state_dict.", "reader.state.")
    forbidden_paths = sorted(
        name for name, _ in tensors if not name.startswith(allowed_prefixes)
    )
    if forbidden_paths:
        raise ValueError(
            "P1 checkpoint contains tensors outside LoRA/reader state: "
            f"{forbidden_paths[:16]}."
        )
    forbidden_fragments = (
        "visual_encoder",
        "dinov3_model",
        "frozen_base",
        "actor.",
        "mot.",
        "video_expert",
        "gate_transformer",
        "adaptive_gate",
        "value_head",
        "image",
        "observation",
        "sample",
        "kv_cache",
    )
    semantic_forbidden = sorted(
        name
        for name, _ in tensors
        if any(fragment in name.lower() for fragment in forbidden_fragments)
    )
    if semantic_forbidden:
        raise ValueError(
            "P1 checkpoint contains forbidden model/data tensor names: "
            f"{semantic_forbidden[:16]}."
        )
    return payload


def save_p1_dino_bc_checkpoint(
    path: str | os.PathLike[str],
    *,
    adapter: ActionDiTLoRAAdapter,
    reader: ActionVisualReader,
    global_step: int,
    stage: str,
    arm: str,
    parent_checkpoint_sha256: str,
    dinov3_weights_sha256: str,
    memory_contract_sha256: str,
    contract: Mapping[str, Any],
    provenance: Mapping[str, Any],
    trainer_state: Mapping[str, Any],
) -> None:
    """Atomically write only trainable P1 weights plus compact metadata."""

    parent_hash = validate_sha256(
        parent_checkpoint_sha256,
        label="parent_checkpoint_sha256",
    )
    dino_hash = validate_sha256(
        dinov3_weights_sha256,
        label="dinov3_weights_sha256",
    )
    memory_hash = validate_sha256(
        memory_contract_sha256,
        label="memory_contract_sha256",
    )
    if (
        reader.reader_contract_sha256 == ""
        or reader.memory_contract_sha256 != memory_hash
    ):
        raise ValueError("P1 reader does not match the supplied memory contract.")
    step = int(global_step)
    payload = {
        "schema": P1_DINO_BC_CHECKPOINT_SCHEMA,
        "global_step": step,
        "stage": str(stage),
        "arm": str(arm),
        "parent_checkpoint_sha256": parent_hash,
        "dinov3_weights_sha256": dino_hash,
        "memory_contract_sha256": memory_hash,
        "reader_contract_sha256": reader.reader_contract_sha256,
        "adapter": build_lora_sidecar_payload(
            adapter,
            parent_checkpoint_sha256=parent_hash,
            extra_metadata={
                "p1_step": step,
                "p1_stage": str(stage),
                "p1_arm": str(arm),
                "p1_reader_contract_sha256": reader.reader_contract_sha256,
            },
        ),
        "reader": reader.export_trainable_state(),
        "contract": dict(contract),
        "provenance": dict(provenance),
        "trainer_state": dict(trainer_state),
    }
    payload = _validate_payload_structure(_cpu_clone(payload))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def load_p1_dino_bc_checkpoint(
    path: str | os.PathLike[str],
    *,
    adapter: ActionDiTLoRAAdapter,
    reader: ActionVisualReader,
    expected_parent_checkpoint_sha256: str,
    expected_dinov3_weights_sha256: str,
    expected_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly restore LoRA and reader state into contract-compatible modules."""

    payload = _validate_payload_structure(
        torch.load(path, map_location="cpu", weights_only=False)
    )
    expected = {
        "parent_checkpoint_sha256": validate_sha256(
            expected_parent_checkpoint_sha256,
            label="expected parent checkpoint SHA256",
        ),
        "dinov3_weights_sha256": validate_sha256(
            expected_dinov3_weights_sha256,
            label="expected DINOv3 weights SHA256",
        ),
        "memory_contract_sha256": reader.memory_contract_sha256,
        "reader_contract_sha256": reader.reader_contract_sha256,
    }
    mismatches = {
        key: (value, payload[key])
        for key, value in expected.items()
        if payload[key] != value
    }
    if mismatches:
        raise ValueError(f"P1 checkpoint contract mismatch: {mismatches}.")
    if payload["contract"] != dict(expected_contract):
        raise ValueError("P1 checkpoint resolved config/data contract mismatch.")
    adapter_metadata = payload["adapter"]["metadata"]
    if (
        adapter_metadata.get("parent_checkpoint_sha256")
        != expected["parent_checkpoint_sha256"]
    ):
        raise ValueError("P1 LoRA sidecar parent hash mismatch.")
    if adapter_metadata.get("extra", {}).get("p1_step") != payload["global_step"]:
        raise ValueError("P1 LoRA sidecar step does not match the checkpoint.")
    adapter.load_lora_state_dict(payload["adapter"]["state_dict"], strict=True)
    reader.load_trainable_state(payload["reader"])
    return payload


def load_p1_dino_bc_trainables(
    path: str | os.PathLike[str],
    *,
    adapter: ActionDiTLoRAAdapter,
    reader: ActionVisualReader,
    expected_parent_checkpoint_sha256: str,
    expected_dinov3_weights_sha256: str,
) -> dict[str, Any]:
    """Read LoRA/reader weights for offline audit without a config hash guess."""

    payload = _validate_payload_structure(
        torch.load(path, map_location="cpu", weights_only=False)
    )
    expected = {
        "parent_checkpoint_sha256": validate_sha256(
            expected_parent_checkpoint_sha256,
            label="expected parent checkpoint SHA256",
        ),
        "dinov3_weights_sha256": validate_sha256(
            expected_dinov3_weights_sha256,
            label="expected DINOv3 weights SHA256",
        ),
        "memory_contract_sha256": reader.memory_contract_sha256,
        "reader_contract_sha256": reader.reader_contract_sha256,
    }
    mismatches = {
        key: (value, payload[key])
        for key, value in expected.items()
        if payload[key] != value
    }
    if mismatches:
        raise ValueError(f"P1 checkpoint contract mismatch: {mismatches}.")
    metadata = payload["adapter"]["metadata"]
    if metadata.get("parent_checkpoint_sha256") != expected["parent_checkpoint_sha256"]:
        raise ValueError("P1 LoRA sidecar parent hash mismatch.")
    if adapter.config.rank != int(metadata["rank"]) or float(
        adapter.config.alpha
    ) != float(metadata["alpha"]):
        raise ValueError("P1 LoRA rank/alpha differs from the checkpoint.")
    adapter.load_lora_state_dict(payload["adapter"]["state_dict"], strict=True)
    reader.load_trainable_state(payload["reader"])
    return payload


def inspect_p1_dino_bc_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Audit a P1 artifact without allocating either frozen backbone."""

    payload = _validate_payload_structure(
        torch.load(path, map_location="cpu", weights_only=False)
    )
    tensors = list(_walk_tensors(payload))
    adapter_tensors = [
        tensor for name, tensor in tensors if name.startswith("adapter.state_dict.")
    ]
    reader_tensors = [
        tensor for name, tensor in tensors if name.startswith("reader.state.")
    ]
    return {
        "schema": payload["schema"],
        "global_step": payload["global_step"],
        "stage": payload["stage"],
        "arm": payload["arm"],
        "parent_checkpoint_sha256": payload["parent_checkpoint_sha256"],
        "dinov3_weights_sha256": payload["dinov3_weights_sha256"],
        "memory_contract_sha256": payload["memory_contract_sha256"],
        "reader_contract_sha256": payload["reader_contract_sha256"],
        "checkpoint_sha256": sha256_file(path),
        "lora_tensor_count": len(adapter_tensors),
        "reader_tensor_count": len(reader_tensors),
        "trainable_bytes": int(
            sum(tensor.numel() * tensor.element_size() for _, tensor in tensors)
        ),
        "forbidden_tensor_paths": [],
        "contains_frozen_fastwam_tensors": False,
        "contains_dinov3_tensors": False,
        "contains_gate_tensors": False,
        "contains_value_head_tensors": False,
        "contains_raw_training_samples": False,
        "contains_image_observation_or_kv": False,
        "result": "PASS",
    }
