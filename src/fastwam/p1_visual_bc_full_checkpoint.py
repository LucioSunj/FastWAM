"""Resumable V2 visual-backbone LoRA-plus-reader training checkpoints."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
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
    VISUAL_READER_STATE_SCHEMA_V2,
    validate_sha256,
)
from fastwam.p1_visual_bc_checkpoint import validate_visual_backbone_metadata
from fastwam.uncond_bc_checkpoint import build_lora_sidecar_payload

P1_VISUAL_BC_FULL_CHECKPOINT_SCHEMA = "fastwam-p1-visual-bc-full-training-v2"

_CHECKPOINT_KEYS = {
    "schema",
    "global_step",
    "epoch",
    "sampler_offset",
    "parent_checkpoint_sha256",
    "visual_backbone",
    "memory_contract_sha256",
    "reader_contract_sha256",
    "adapter",
    "reader",
    "optimizer",
    "lr_scheduler",
    "grad_scaler",
    "rng_by_rank",
    "contract",
    "provenance",
    "trainer_state",
}
_TRAINER_STATE_KEYS = {
    "best_validation_loss_action_bc",
    "best_step",
    "epochs_without_improvement",
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


def _validate_trainer_state(payload: Any, *, global_step: int) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != _TRAINER_STATE_KEYS:
        raise ValueError("V2 full trainer-state keys changed.")
    best_loss = payload["best_validation_loss_action_bc"]
    if best_loss is not None and (
        isinstance(best_loss, bool)
        or not isinstance(best_loss, (int, float))
        or not math.isfinite(float(best_loss))
        or float(best_loss) < 0
    ):
        raise ValueError("V2 full best validation loss is invalid.")
    best_step = payload["best_step"]
    if best_step is not None and (
        isinstance(best_step, bool)
        or not isinstance(best_step, int)
        or not 0 <= best_step <= global_step
    ):
        raise ValueError("V2 full best step lies outside [0, global_step].")
    counters = {
        key: payload[key]
        for key in ("epochs_without_improvement", "nonzero_update_count")
    }
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counters.values()
        )
        or counters["nonzero_update_count"] > global_step
    ):
        raise ValueError("V2 full trainer counters are invalid.")
    return {
        "best_validation_loss_action_bc": (
            None if best_loss is None else float(best_loss)
        ),
        "best_step": best_step,
        **counters,
    }


def _validate_structure(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != _CHECKPOINT_KEYS:
        keys = sorted(payload) if isinstance(payload, dict) else type(payload)
        raise ValueError(f"Not a complete V2 visual full checkpoint: {keys}.")
    if payload["schema"] != P1_VISUAL_BC_FULL_CHECKPOINT_SCHEMA:
        raise ValueError(f"Unsupported V2 full schema {payload['schema']!r}.")
    for key in ("global_step", "epoch", "sampler_offset"):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"V2 full counter {key!r} is invalid.")
    for key in (
        "parent_checkpoint_sha256",
        "memory_contract_sha256",
        "reader_contract_sha256",
    ):
        payload[key] = validate_sha256(payload[key], label=key)
    payload["visual_backbone"] = validate_visual_backbone_metadata(
        payload["visual_backbone"]
    )
    if (
        payload["visual_backbone"]["memory_contract_sha256"]
        != payload["memory_contract_sha256"]
    ):
        raise ValueError("V2 full visual and memory contracts disagree.")
    if not isinstance(payload["contract"], Mapping) or not isinstance(
        payload["provenance"], Mapping
    ):
        raise TypeError("V2 full contract/provenance must be mappings.")
    if not isinstance(payload["rng_by_rank"], list) or not payload["rng_by_rank"]:
        raise ValueError("V2 full checkpoint requires rank-local RNG states.")
    payload["trainer_state"] = _validate_trainer_state(
        payload["trainer_state"],
        global_step=payload["global_step"],
    )
    adapter = payload["adapter"]
    if not isinstance(adapter, Mapping) or set(adapter) != {
        "metadata",
        "state_dict",
    }:
        raise ValueError("V2 full checkpoint is missing its LoRA sidecar.")
    if adapter["metadata"].get("schema") != REGIME_LORA_SIDECAR_SCHEMA:
        raise ValueError("V2 full LoRA sidecar schema changed.")
    state = adapter["state_dict"]
    if (
        not isinstance(state, Mapping)
        or not state
        or any(not str(name).endswith((".lora_A", ".lora_B")) for name in state)
    ):
        raise ValueError("V2 full adapter state must contain LoRA tensors only.")
    reader = payload["reader"]
    reader_keys = {
        "schema",
        "reader_kind",
        "reader_contract_sha256",
        "parameter_names",
        "state",
    }
    if not isinstance(reader, Mapping) or set(reader) != reader_keys:
        raise ValueError("V2 full reader state is malformed.")
    if reader["schema"] != VISUAL_READER_STATE_SCHEMA_V2:
        raise ValueError("V2 full checkpoint requires the V2 reader schema.")
    if reader["reader_contract_sha256"] != payload["reader_contract_sha256"]:
        raise ValueError("V2 full reader contract hashes disagree.")
    names = tuple(reader["parameter_names"])
    if not names or tuple(reader["state"]) != names:
        raise ValueError("V2 full reader manifest/state is inconsistent.")
    allowed = (
        "adapter.state_dict.",
        "reader.state.",
        "optimizer.",
        "lr_scheduler.",
        "grad_scaler.",
        "rng_by_rank.",
    )
    forbidden = sorted(
        name for name, _ in _walk_tensors(payload) if not name.startswith(allowed)
    )
    if forbidden:
        raise ValueError(f"V2 full checkpoint has forbidden tensors: {forbidden[:16]}.")
    forbidden_fragments = (
        "visual_encoder",
        "visual_backbone.model",
        "dinov3_model",
        "lingbot",
        "frozen_base",
        "video_expert",
        "gate_transformer",
        "value_head",
        "image",
        "observation",
        "sample",
        "kv_cache",
    )
    semantic_forbidden = sorted(
        name
        for name, _ in _walk_tensors(payload)
        if any(fragment in name.lower() for fragment in forbidden_fragments)
    )
    if semantic_forbidden:
        raise ValueError(
            "V2 full checkpoint contains frozen/data tensor names: "
            f"{semantic_forbidden[:16]}."
        )
    return payload


def save_p1_visual_bc_full_checkpoint(
    path: str | os.PathLike[str],
    *,
    adapter: ActionDiTLoRAAdapter,
    reader: ActionVisualReader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    grad_scaler: Any,
    global_step: int,
    epoch: int,
    sampler_offset: int,
    rng_by_rank: Sequence[Mapping[str, Any]],
    parent_checkpoint_sha256: str,
    visual_backbone: Mapping[str, Any],
    memory_contract_sha256: str,
    contract: Mapping[str, Any],
    provenance: Mapping[str, Any],
    trainer_state: Mapping[str, Any],
) -> None:
    """Atomically save complete V2 resumable trainer state and trainables."""

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
        raise ValueError("V2 full reader and memory contracts differ.")
    step = int(global_step)
    payload = {
        "schema": P1_VISUAL_BC_FULL_CHECKPOINT_SCHEMA,
        "global_step": step,
        "epoch": int(epoch),
        "sampler_offset": int(sampler_offset),
        "parent_checkpoint_sha256": parent_hash,
        "visual_backbone": metadata,
        "memory_contract_sha256": memory_hash,
        "reader_contract_sha256": reader.reader_contract_sha256,
        "adapter": build_lora_sidecar_payload(
            adapter,
            parent_checkpoint_sha256=parent_hash,
            extra_metadata={
                "p1_visual_full_step": step,
                "p1_visual_full_config_sha256": str(
                    contract.get("resolved_config_sha256", "")
                ),
                "p1_visual_reader_contract_sha256": reader.reader_contract_sha256,
                "p1_visual_asset_contract_sha256": metadata["asset_contract_sha256"],
            },
        ),
        "reader": reader.export_trainable_state(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "grad_scaler": grad_scaler.state_dict(),
        "rng_by_rank": list(rng_by_rank),
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


def load_p1_visual_bc_full_checkpoint(
    path: str | os.PathLike[str],
    *,
    adapter: ActionDiTLoRAAdapter,
    reader: ActionVisualReader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    grad_scaler: Any,
    expected_parent_checkpoint_sha256: str,
    expected_visual_backbone: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Restore V2 training only after exact backbone/contract equality."""

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
        raise ValueError(f"V2 full checkpoint contract mismatch: {mismatches}.")
    adapter.load_lora_state_dict(payload["adapter"]["state_dict"], strict=True)
    reader.load_trainable_state(payload["reader"])
    optimizer.load_state_dict(payload["optimizer"])
    lr_scheduler.load_state_dict(payload["lr_scheduler"])
    grad_scaler.load_state_dict(payload["grad_scaler"])
    return payload


def load_p1_visual_bc_full_weights_for_evaluation(
    path: str | os.PathLike[str],
    *,
    adapter: ActionDiTLoRAAdapter,
    reader: ActionVisualReader,
    expected_parent_checkpoint_sha256: str,
    expected_visual_backbone: Mapping[str, Any],
    expected_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Load only V2 trainables from a fully validated resumable checkpoint."""

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
        raise ValueError(
            f"V2 full evaluation checkpoint contract mismatch: {mismatches}."
        )
    adapter.load_lora_state_dict(payload["adapter"]["state_dict"], strict=True)
    reader.load_trainable_state(payload["reader"])
    return payload


def inspect_p1_visual_bc_full_checkpoint(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Audit that a V2 full checkpoint excludes both frozen backbones."""

    payload = _validate_structure(
        torch.load(path, map_location="cpu", weights_only=False)
    )
    entries = list(_walk_tensors(payload))
    metadata = payload["adapter"]["metadata"]
    return {
        "schema": payload["schema"],
        "global_step": payload["global_step"],
        "epoch": payload["epoch"],
        "sampler_offset": payload["sampler_offset"],
        "checkpoint_sha256": sha256_file(path),
        "parent_checkpoint_sha256": payload["parent_checkpoint_sha256"],
        "visual_backbone": payload["visual_backbone"],
        "memory_contract_sha256": payload["memory_contract_sha256"],
        "reader_contract_sha256": payload["reader_contract_sha256"],
        "adapter_rank": int(metadata["rank"]),
        "adapter_alpha": float(metadata["alpha"]),
        "lora_tensor_count": sum(
            name.startswith("adapter.state_dict.") for name, _ in entries
        ),
        "reader_tensor_count": sum(
            name.startswith("reader.state.") for name, _ in entries
        ),
        "optimizer_tensor_count": sum(
            name.startswith("optimizer.") for name, _ in entries
        ),
        "rng_rank_count": len(payload["rng_by_rank"]),
        "contains_frozen_fastwam_tensors": False,
        "contains_visual_backbone_tensors": False,
        "contains_gate_tensors": False,
        "contains_value_head_tensors": False,
        "contains_raw_training_samples": False,
        "forbidden_tensor_paths": [],
        "result": "PASS",
    }
