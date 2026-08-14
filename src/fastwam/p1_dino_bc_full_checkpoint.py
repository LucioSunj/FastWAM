"""Resumable LoRA-plus-reader checkpoints for full P1 DINO BC training."""

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
    VISUAL_READER_STATE_SCHEMA,
    validate_sha256,
)
from fastwam.p1_dino_contribution_v2 import (
    CAUSAL_SELECTOR_SCHEMA,
    TASK_PAIRED_SAMPLER_SCHEMA,
    WARMUP_STATE_SCHEMA,
    CausalCheckpointSelector,
    DependencyWarmupController,
    NegativeModeCycle,
)
from fastwam.uncond_bc_checkpoint import build_lora_sidecar_payload

P1_DINO_BC_FULL_CHECKPOINT_SCHEMA = "fastwam-p1-dino-bc-full-training-v1"
P1_DINO_BC_FULL_CHECKPOINT_V2_SCHEMA = "fastwam-p1-dino-bc-full-training-v2"

_CHECKPOINT_KEYS = {
    "schema",
    "global_step",
    "epoch",
    "sampler_offset",
    "parent_checkpoint_sha256",
    "dinov3_weights_sha256",
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
_ADAPTER_KEYS = {"metadata", "state_dict"}
_READER_KEYS = {
    "schema",
    "reader_kind",
    "reader_contract_sha256",
    "parameter_names",
    "state",
}
_TRAINER_STATE_KEYS = {
    "best_validation_loss_action_bc",
    "best_step",
    "epochs_without_improvement",
    "nonzero_update_count",
}
_V2_STATE_KEYS = {
    "profile",
    "warmup",
    "negative_cycle",
    "task_paired_sampler_by_rank",
    "causal_selector",
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


def _validate_trainer_state(
    payload: Mapping[str, Any],
    *,
    global_step: int,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != _TRAINER_STATE_KEYS:
        keys = sorted(payload) if isinstance(payload, Mapping) else type(payload)
        raise ValueError(f"P1 full trainer-state keys changed: {keys}.")
    best_loss = payload["best_validation_loss_action_bc"]
    if best_loss is not None and (
        isinstance(best_loss, bool)
        or not isinstance(best_loss, (int, float))
        or not math.isfinite(float(best_loss))
        or float(best_loss) < 0
    ):
        raise ValueError("P1 full best validation loss must be finite or null.")
    best_step = payload["best_step"]
    if best_step is not None and (
        isinstance(best_step, bool)
        or not isinstance(best_step, int)
        or not 0 <= best_step <= global_step
    ):
        raise ValueError("P1 full best step lies outside [0, global_step].")
    counters = {
        name: payload[name]
        for name in ("epochs_without_improvement", "nonzero_update_count")
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in counters.values()
    ):
        raise ValueError(f"P1 full trainer counters are invalid: {counters}.")
    if counters["nonzero_update_count"] > global_step:
        raise ValueError("P1 full nonzero updates exceed global_step.")
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
        raise ValueError(f"Not a complete P1 full checkpoint: {keys}.")
    if payload.get("schema") != P1_DINO_BC_FULL_CHECKPOINT_SCHEMA:
        raise ValueError(f"Unsupported P1 full schema {payload.get('schema')!r}.")
    for key in ("global_step", "epoch", "sampler_offset"):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"P1 full checkpoint counter {key!r} is invalid.")
    for key in (
        "parent_checkpoint_sha256",
        "dinov3_weights_sha256",
        "memory_contract_sha256",
        "reader_contract_sha256",
    ):
        payload[key] = validate_sha256(payload[key], label=key)
    if not isinstance(payload["contract"], Mapping):
        raise TypeError("P1 full checkpoint contract must be a mapping.")
    if not isinstance(payload["provenance"], Mapping):
        raise TypeError("P1 full checkpoint provenance must be a mapping.")
    if not isinstance(payload["rng_by_rank"], list) or not payload["rng_by_rank"]:
        raise ValueError("P1 full checkpoint requires RNG state for every rank.")
    payload["trainer_state"] = _validate_trainer_state(
        payload["trainer_state"],
        global_step=payload["global_step"],
    )

    adapter = payload["adapter"]
    if not isinstance(adapter, Mapping) or set(adapter) != _ADAPTER_KEYS:
        raise ValueError("P1 full checkpoint is missing its LoRA sidecar.")
    metadata = adapter["metadata"]
    state = adapter["state_dict"]
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("schema") != REGIME_LORA_SIDECAR_SCHEMA
        or not isinstance(state, Mapping)
        or not state
    ):
        raise ValueError("P1 full LoRA metadata/state is malformed.")
    invalid_lora = sorted(
        name for name in state if not str(name).endswith((".lora_A", ".lora_B"))
    )
    if invalid_lora:
        raise ValueError(
            f"P1 full checkpoint has non-LoRA adapter state: {invalid_lora}."
        )

    reader = payload["reader"]
    if not isinstance(reader, Mapping) or set(reader) != _READER_KEYS:
        raise ValueError("P1 full checkpoint is missing reader state.")
    if reader.get("schema") != VISUAL_READER_STATE_SCHEMA:
        raise ValueError("P1 full reader state schema changed.")
    if reader.get("reader_contract_sha256") != payload["reader_contract_sha256"]:
        raise ValueError("P1 full reader contract hashes disagree.")
    names = tuple(reader.get("parameter_names", ()))
    reader_state = reader.get("state")
    if (
        not names
        or not isinstance(reader_state, Mapping)
        or tuple(reader_state) != names
    ):
        raise ValueError("P1 full reader manifest/state is empty or inconsistent.")

    allowed_prefixes = (
        "adapter.state_dict.",
        "reader.state.",
        "optimizer.",
        "lr_scheduler.",
        "grad_scaler.",
        "rng_by_rank.",
    )
    tensor_entries = list(_walk_tensors(payload))
    forbidden_paths = sorted(
        name for name, _ in tensor_entries if not name.startswith(allowed_prefixes)
    )
    if forbidden_paths:
        raise ValueError(
            "P1 full checkpoint contains tensors outside trainable/trainer state: "
            f"{forbidden_paths[:16]}."
        )
    forbidden_fragments = (
        "visual_encoder",
        "dinov3_model",
        "frozen_base",
        "video_expert",
        "proprio_encoder",
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
        for name, _ in tensor_entries
        if any(fragment in name.lower() for fragment in forbidden_fragments)
    )
    if semantic_forbidden:
        raise ValueError(
            "P1 full checkpoint contains forbidden model/data tensors: "
            f"{semantic_forbidden[:16]}."
        )
    return payload


def _validate_v2_state(
    payload: Any,
    *,
    rng_rank_count: int,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != _V2_STATE_KEYS:
        keys = sorted(payload) if isinstance(payload, Mapping) else type(payload)
        raise ValueError(f"P1 v2 state keys changed: {keys}.")
    if payload.get("profile") != "dino_contribution_v2":
        raise ValueError("P1 v2 checkpoint profile changed.")
    warmup = payload.get("warmup")
    if not isinstance(warmup, Mapping) or warmup.get("schema") != WARMUP_STATE_SCHEMA:
        raise ValueError("P1 v2 checkpoint warm-up state is malformed.")
    DependencyWarmupController.from_state_dict(warmup)
    negative_cycle = payload.get("negative_cycle")
    if not isinstance(negative_cycle, Mapping):
        raise ValueError("P1 v2 checkpoint negative cycle is malformed.")
    NegativeModeCycle.from_state_dict(negative_cycle)
    samplers = payload.get("task_paired_sampler_by_rank")
    if not isinstance(samplers, list) or len(samplers) != rng_rank_count:
        raise ValueError("P1 v2 checkpoint sampler rank count changed.")
    for rank, sampler in enumerate(samplers):
        if (
            not isinstance(sampler, Mapping)
            or sampler.get("schema") != TASK_PAIRED_SAMPLER_SCHEMA
            or int(sampler.get("rank", -1)) != rank
            or int(sampler.get("world_size", -1)) != rng_rank_count
        ):
            raise ValueError("P1 v2 rank-local sampler state is malformed.")
        order_hash = str(sampler.get("order_sha256", ""))
        if len(order_hash) != 64 or any(
            value not in "0123456789abcdef" for value in order_hash
        ):
            raise ValueError("P1 v2 sampler order SHA256 is malformed.")
    selector = payload.get("causal_selector")
    if (
        not isinstance(selector, Mapping)
        or selector.get("schema") != CAUSAL_SELECTOR_SCHEMA
    ):
        raise ValueError("P1 v2 causal-selector state is malformed.")
    CausalCheckpointSelector.from_state_dict(selector)
    tensor_paths = [name for name, _ in _walk_tensors(payload)]
    if tensor_paths:
        raise ValueError(f"P1 v2 control state contains tensors: {tensor_paths[:8]}.")
    return _cpu_clone(payload)


def _validate_v2_structure(payload: Any) -> dict[str, Any]:
    expected_keys = _CHECKPOINT_KEYS | {"v2_state"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        keys = sorted(payload) if isinstance(payload, dict) else type(payload)
        raise ValueError(f"Not a complete P1 v2 full checkpoint: {keys}.")
    if payload.get("schema") != P1_DINO_BC_FULL_CHECKPOINT_V2_SCHEMA:
        raise ValueError(f"Unsupported P1 v2 full schema {payload.get('schema')!r}.")
    legacy_view = {key: value for key, value in payload.items() if key != "v2_state"}
    legacy_view["schema"] = P1_DINO_BC_FULL_CHECKPOINT_SCHEMA
    validated = _validate_structure(legacy_view)
    validated["schema"] = P1_DINO_BC_FULL_CHECKPOINT_V2_SCHEMA
    validated["v2_state"] = _validate_v2_state(
        payload["v2_state"],
        rng_rank_count=len(validated["rng_by_rank"]),
    )
    return validated


def save_p1_dino_bc_full_checkpoint(
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
    dinov3_weights_sha256: str,
    memory_contract_sha256: str,
    contract: Mapping[str, Any],
    provenance: Mapping[str, Any],
    trainer_state: Mapping[str, Any],
) -> None:
    """Atomically save P1 trainables and complete resumable trainer state."""

    memory_hash = validate_sha256(
        memory_contract_sha256,
        label="memory_contract_sha256",
    )
    if reader.memory_contract_sha256 != memory_hash:
        raise ValueError("P1 full reader and memory contracts differ.")
    step = int(global_step)
    payload = {
        "schema": P1_DINO_BC_FULL_CHECKPOINT_SCHEMA,
        "global_step": step,
        "epoch": int(epoch),
        "sampler_offset": int(sampler_offset),
        "parent_checkpoint_sha256": str(parent_checkpoint_sha256).lower(),
        "dinov3_weights_sha256": str(dinov3_weights_sha256).lower(),
        "memory_contract_sha256": memory_hash,
        "reader_contract_sha256": reader.reader_contract_sha256,
        "adapter": build_lora_sidecar_payload(
            adapter,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            extra_metadata={
                "p1_full_step": step,
                "p1_full_config_sha256": str(
                    contract.get("resolved_config_sha256", "")
                ),
                "p1_reader_contract_sha256": reader.reader_contract_sha256,
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


def save_p1_dino_bc_full_checkpoint_v2(
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
    dinov3_weights_sha256: str,
    memory_contract_sha256: str,
    contract: Mapping[str, Any],
    provenance: Mapping[str, Any],
    trainer_state: Mapping[str, Any],
    v2_state: Mapping[str, Any],
) -> None:
    """Atomically save the independent contribution-v2 training lineage."""

    memory_hash = validate_sha256(
        memory_contract_sha256,
        label="memory_contract_sha256",
    )
    if reader.memory_contract_sha256 != memory_hash:
        raise ValueError("P1 v2 reader and memory contracts differ.")
    step = int(global_step)
    payload = {
        "schema": P1_DINO_BC_FULL_CHECKPOINT_V2_SCHEMA,
        "global_step": step,
        "epoch": int(epoch),
        "sampler_offset": int(sampler_offset),
        "parent_checkpoint_sha256": str(parent_checkpoint_sha256).lower(),
        "dinov3_weights_sha256": str(dinov3_weights_sha256).lower(),
        "memory_contract_sha256": memory_hash,
        "reader_contract_sha256": reader.reader_contract_sha256,
        "adapter": build_lora_sidecar_payload(
            adapter,
            parent_checkpoint_sha256=parent_checkpoint_sha256,
            extra_metadata={
                "p1_full_step": step,
                "p1_full_config_sha256": str(
                    contract.get("resolved_config_sha256", "")
                ),
                "p1_reader_contract_sha256": reader.reader_contract_sha256,
                "p1_dino_contribution_profile": "dino_contribution_v2",
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
        "v2_state": dict(v2_state),
    }
    payload = _validate_v2_structure(_cpu_clone(payload))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def load_p1_dino_bc_full_checkpoint(
    path: str | os.PathLike[str],
    *,
    adapter: ActionDiTLoRAAdapter,
    reader: ActionVisualReader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    grad_scaler: Any,
    expected_parent_checkpoint_sha256: str,
    expected_dinov3_weights_sha256: str,
    expected_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly restore P1 trainables and trainer state except rank-local RNG."""

    payload = _validate_structure(
        torch.load(path, map_location="cpu", weights_only=False)
    )
    expected = {
        "parent_checkpoint_sha256": validate_sha256(
            expected_parent_checkpoint_sha256,
            label="expected parent checkpoint SHA256",
        ),
        "dinov3_weights_sha256": validate_sha256(
            expected_dinov3_weights_sha256,
            label="expected DINO weights SHA256",
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
        raise ValueError(f"P1 full checkpoint contract mismatch: {mismatches}.")
    if payload["contract"] != dict(expected_contract):
        raise ValueError("P1 full resolved config/data contract mismatch.")
    adapter.load_lora_state_dict(payload["adapter"]["state_dict"], strict=True)
    reader.load_trainable_state(payload["reader"])
    optimizer.load_state_dict(payload["optimizer"])
    lr_scheduler.load_state_dict(payload["lr_scheduler"])
    grad_scaler.load_state_dict(payload["grad_scaler"])
    return payload


def _validate_expected_trainable_contract(
    payload: Mapping[str, Any],
    *,
    adapter: ActionDiTLoRAAdapter,
    reader: ActionVisualReader,
    expected_parent_checkpoint_sha256: str,
    expected_dinov3_weights_sha256: str,
) -> None:
    expected = {
        "parent_checkpoint_sha256": validate_sha256(
            expected_parent_checkpoint_sha256,
            label="expected parent checkpoint SHA256",
        ),
        "dinov3_weights_sha256": validate_sha256(
            expected_dinov3_weights_sha256,
            label="expected DINO weights SHA256",
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
        raise ValueError(f"P1 full checkpoint contract mismatch: {mismatches}.")
    metadata = payload["adapter"]["metadata"]
    if metadata.get("parent_checkpoint_sha256") != expected["parent_checkpoint_sha256"]:
        raise ValueError("P1 full LoRA sidecar parent hash mismatch.")
    if adapter.config.rank != int(metadata["rank"]) or float(
        adapter.config.alpha
    ) != float(metadata["alpha"]):
        raise ValueError("P1 full LoRA rank/alpha differs from the checkpoint.")


def load_p1_dino_bc_full_checkpoint_v2(
    path: str | os.PathLike[str],
    *,
    adapter: ActionDiTLoRAAdapter,
    reader: ActionVisualReader,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    grad_scaler: Any,
    expected_parent_checkpoint_sha256: str,
    expected_dinov3_weights_sha256: str,
    expected_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly restore a contribution-v2 full-training checkpoint."""

    payload = _validate_v2_structure(
        torch.load(path, map_location="cpu", weights_only=False)
    )
    _validate_expected_trainable_contract(
        payload,
        adapter=adapter,
        reader=reader,
        expected_parent_checkpoint_sha256=expected_parent_checkpoint_sha256,
        expected_dinov3_weights_sha256=expected_dinov3_weights_sha256,
    )
    if payload["contract"] != dict(expected_contract):
        raise ValueError("P1 v2 resolved config/data contract mismatch.")
    adapter.load_lora_state_dict(payload["adapter"]["state_dict"], strict=True)
    reader.load_trainable_state(payload["reader"])
    optimizer.load_state_dict(payload["optimizer"])
    lr_scheduler.load_state_dict(payload["lr_scheduler"])
    grad_scaler.load_state_dict(payload["grad_scaler"])
    return payload


def load_p1_dino_bc_full_trainables(
    path: str | os.PathLike[str],
    *,
    adapter: ActionDiTLoRAAdapter,
    reader: ActionVisualReader,
    expected_parent_checkpoint_sha256: str,
    expected_dinov3_weights_sha256: str,
) -> dict[str, Any]:
    """Read only LoRA/reader weights from either full-training schema."""

    raw = torch.load(path, map_location="cpu", weights_only=False)
    schema = raw.get("schema") if isinstance(raw, Mapping) else None
    if schema == P1_DINO_BC_FULL_CHECKPOINT_SCHEMA:
        payload = _validate_structure(raw)
    elif schema == P1_DINO_BC_FULL_CHECKPOINT_V2_SCHEMA:
        payload = _validate_v2_structure(raw)
    else:
        raise ValueError(f"Unsupported P1 full checkpoint schema {schema!r}.")
    _validate_expected_trainable_contract(
        payload,
        adapter=adapter,
        reader=reader,
        expected_parent_checkpoint_sha256=expected_parent_checkpoint_sha256,
        expected_dinov3_weights_sha256=expected_dinov3_weights_sha256,
    )
    adapter.load_lora_state_dict(payload["adapter"]["state_dict"], strict=True)
    reader.load_trainable_state(payload["reader"])
    return payload


def inspect_p1_dino_bc_full_checkpoint(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Audit that a full P1 checkpoint excludes all frozen models and data."""

    payload = _validate_structure(
        torch.load(path, map_location="cpu", weights_only=False)
    )
    entries = list(_walk_tensors(payload))
    lora_tensors = [
        value for name, value in entries if name.startswith("adapter.state_dict.")
    ]
    reader_tensors = [
        value for name, value in entries if name.startswith("reader.state.")
    ]
    metadata = payload["adapter"]["metadata"]
    return {
        "schema": payload["schema"],
        "global_step": payload["global_step"],
        "epoch": payload["epoch"],
        "sampler_offset": payload["sampler_offset"],
        "checkpoint_sha256": sha256_file(path),
        "parent_checkpoint_sha256": payload["parent_checkpoint_sha256"],
        "dinov3_weights_sha256": payload["dinov3_weights_sha256"],
        "memory_contract_sha256": payload["memory_contract_sha256"],
        "reader_contract_sha256": payload["reader_contract_sha256"],
        "adapter_rank": int(metadata["rank"]),
        "adapter_alpha": float(metadata["alpha"]),
        "lora_tensor_count": len(lora_tensors),
        "reader_tensor_count": len(reader_tensors),
        "optimizer_tensor_count": sum(
            1 for name, _ in entries if name.startswith("optimizer.")
        ),
        "rng_rank_count": len(payload["rng_by_rank"]),
        "contains_frozen_fastwam_tensors": False,
        "contains_dinov3_tensors": False,
        "contains_gate_tensors": False,
        "contains_value_head_tensors": False,
        "contains_raw_training_samples": False,
        "forbidden_tensor_paths": [],
        "result": "PASS",
    }


def inspect_p1_dino_bc_full_checkpoint_v2(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Audit a contribution-v2 checkpoint and its tensor-free control state."""

    payload = _validate_v2_structure(
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
        "dinov3_weights_sha256": payload["dinov3_weights_sha256"],
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
        "phase": payload["v2_state"]["warmup"]["phase"],
        "warmup_end_step": payload["v2_state"]["warmup"]["warmup_end_step"],
        "dependency_update_count": payload["v2_state"]["negative_cycle"][
            "dependency_update_count"
        ],
        "lora_ramp_progress": payload["v2_state"]["warmup"]["lora_ramp_progress"],
        "sampler_rank_count": len(payload["v2_state"]["task_paired_sampler_by_rank"]),
        "contains_frozen_fastwam_tensors": False,
        "contains_dinov3_tensors": False,
        "contains_gate_tensors": False,
        "contains_value_head_tensors": False,
        "contains_raw_training_samples": False,
        "forbidden_tensor_paths": [],
        "result": "PASS",
    }
