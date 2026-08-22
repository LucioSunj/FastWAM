"""Versioned shared-causal-policy checkpoints with an exact ownership schema."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .contracts import (
    CAUSAL_POLICY_CHECKPOINT_SCHEMA,
    CAUSAL_POLICY_CHECKPOINT_SCHEMA_V2,
    CURRENT_ONLY_CAUSAL_POLICY_CHECKPOINT_SCHEMA,
)
from .one_pass import C1IntervalKVFusion
from .shared_lora import SharedActionDiTLoRAAdapter

_TOP_LEVEL_KEYS_V1 = {
    "schema",
    "metadata",
    "adapter_state_dict",
    "optimizer",
    "lr_scheduler",
    "grad_scaler",
    "rng_by_rank",
    "trainer_state",
    "config",
}
_TOP_LEVEL_KEYS_V2 = _TOP_LEVEL_KEYS_V1 | {"fusion_state_dict"}
_METADATA_KEYS_V1 = {
    "parent_checkpoint_sha256",
    "statistics_sha256",
    "global_step",
    "epoch",
    "adapter_contract",
}
_METADATA_KEYS_V2 = _METADATA_KEYS_V1 | {"fusion_contract"}
_FORBIDDEN_KEY_PARTS = (
    "video_kv",
    "future_kv",
    "observation",
    "image",
    "frozen_parent",
    "base_weight",
)
_FORBIDDEN_KEYS = {"sample", "samples"}


def _cpu_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {str(key): _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return value


def _reject_forbidden_keys(value: Any, *, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized in _FORBIDDEN_KEYS or any(
                part in normalized for part in _FORBIDDEN_KEY_PARTS
            ):
                raise ValueError(
                    f"Causal checkpoint contains forbidden field {path}.{key}."
                )
            _reject_forbidden_keys(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_forbidden_keys(item, path=f"{path}[{index}]")


def build_causal_policy_checkpoint(
    *,
    adapter: SharedActionDiTLoRAAdapter,
    parent_checkpoint_sha256: str,
    statistics_sha256: str,
    global_step: int,
    epoch: int,
    optimizer_state: Mapping[str, Any],
    lr_scheduler_state: Mapping[str, Any],
    grad_scaler_state: Mapping[str, Any],
    rng_by_rank: Sequence[Mapping[str, Any]],
    trainer_state: Mapping[str, Any],
    config: Mapping[str, Any],
    checkpoint_schema: str = CAUSAL_POLICY_CHECKPOINT_SCHEMA,
    fusion: C1IntervalKVFusion | None = None,
) -> dict[str, Any]:
    """Build the only allowed resumable shared-policy payload."""

    if global_step < 0 or epoch < 0 or not rng_by_rank:
        raise ValueError("Checkpoint counters and rank RNG states are invalid.")
    if checkpoint_schema not in {
        CAUSAL_POLICY_CHECKPOINT_SCHEMA,
        CAUSAL_POLICY_CHECKPOINT_SCHEMA_V2,
        CURRENT_ONLY_CAUSAL_POLICY_CHECKPOINT_SCHEMA,
    }:
        raise ValueError(f"Unsupported causal policy schema {checkpoint_schema!r}.")
    if (checkpoint_schema == CAUSAL_POLICY_CHECKPOINT_SCHEMA_V2) != (
        fusion is not None
    ):
        raise ValueError("Only tri-mode v2 checkpoints may contain C1 fusion state.")
    adapter_contract = adapter.metadata(
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        statistics_sha256=statistics_sha256,
    )
    if checkpoint_schema == CAUSAL_POLICY_CHECKPOINT_SCHEMA_V2:
        adapter_contract = {
            **adapter_contract,
            "schema": CAUSAL_POLICY_CHECKPOINT_SCHEMA_V2,
            "active_modes": ["c0_current", "c1_one_pass", "c2_full"],
        }
    elif checkpoint_schema == CURRENT_ONLY_CAUSAL_POLICY_CHECKPOINT_SCHEMA:
        adapter_contract = {
            **adapter_contract,
            "schema": CURRENT_ONLY_CAUSAL_POLICY_CHECKPOINT_SCHEMA,
            "active_modes": ["c0_current"],
        }
    metadata = {
        "parent_checkpoint_sha256": str(parent_checkpoint_sha256).lower(),
        "statistics_sha256": str(statistics_sha256).lower(),
        "global_step": int(global_step),
        "epoch": int(epoch),
        "adapter_contract": adapter_contract,
    }
    if fusion is not None:
        metadata["fusion_contract"] = fusion.metadata()
    payload = {
        "schema": checkpoint_schema,
        "metadata": metadata,
        "adapter_state_dict": adapter.lora_state_dict(),
        "optimizer": _cpu_clone(optimizer_state),
        "lr_scheduler": _cpu_clone(lr_scheduler_state),
        "grad_scaler": _cpu_clone(grad_scaler_state),
        "rng_by_rank": _cpu_clone(list(rng_by_rank)),
        "trainer_state": _cpu_clone(trainer_state),
        "config": _cpu_clone(config),
    }
    if fusion is not None:
        payload["fusion_state_dict"] = fusion.trainable_state_dict()
    inspect_causal_policy_checkpoint_payload(
        payload,
        adapter=adapter,
        fusion=fusion,
    )
    return payload


def inspect_causal_policy_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    adapter: SharedActionDiTLoRAAdapter | None = None,
    fusion: C1IntervalKVFusion | None = None,
) -> dict[str, Any]:
    """Validate schema and prove that every model tensor is adapter-owned."""

    schema = payload.get("schema")
    expected_top_level = (
        _TOP_LEVEL_KEYS_V2
        if schema == CAUSAL_POLICY_CHECKPOINT_SCHEMA_V2
        else _TOP_LEVEL_KEYS_V1
    )
    if set(payload) != expected_top_level:
        raise ValueError(
            f"Causal checkpoint top-level keys changed: {sorted(payload)}."
        )
    if payload["schema"] not in {
        CAUSAL_POLICY_CHECKPOINT_SCHEMA,
        CAUSAL_POLICY_CHECKPOINT_SCHEMA_V2,
        CURRENT_ONLY_CAUSAL_POLICY_CHECKPOINT_SCHEMA,
    }:
        raise ValueError(f"Unsupported causal checkpoint schema {payload['schema']!r}.")
    metadata = payload["metadata"]
    expected_metadata = (
        _METADATA_KEYS_V2
        if payload["schema"] == CAUSAL_POLICY_CHECKPOINT_SCHEMA_V2
        else _METADATA_KEYS_V1
    )
    if not isinstance(metadata, Mapping) or set(metadata) != expected_metadata:
        raise ValueError("Causal checkpoint metadata keys changed.")
    adapter_contract = metadata["adapter_contract"]
    if not isinstance(adapter_contract, Mapping):
        raise TypeError("Causal checkpoint adapter contract must be a mapping.")
    expected_modes = {
        CAUSAL_POLICY_CHECKPOINT_SCHEMA: ["c0_current", "c2_full"],
        CAUSAL_POLICY_CHECKPOINT_SCHEMA_V2: [
            "c0_current",
            "c1_one_pass",
            "c2_full",
        ],
        CURRENT_ONLY_CAUSAL_POLICY_CHECKPOINT_SCHEMA: ["c0_current"],
    }[payload["schema"]]
    if (
        adapter_contract.get("schema") != payload["schema"]
        or adapter_contract.get("active_modes") != expected_modes
    ):
        raise ValueError("Causal checkpoint adapter mode contract changed.")
    adapter_state = payload["adapter_state_dict"]
    if not isinstance(adapter_state, Mapping) or not adapter_state:
        raise ValueError("Causal checkpoint adapter state must be non-empty.")
    if any(
        not (str(name).endswith(".lora_A") or str(name).endswith(".lora_B"))
        or not isinstance(value, torch.Tensor)
        for name, value in adapter_state.items()
    ):
        raise ValueError("Causal checkpoint model state contains a non-LoRA tensor.")
    if adapter is not None:
        expected = {name for name, _ in adapter.named_lora_parameters()}
        if set(adapter_state) != expected:
            raise ValueError(
                "Causal checkpoint does not contain exactly the live LoRA."
            )
    fusion_state = payload.get("fusion_state_dict")
    if payload["schema"] == CAUSAL_POLICY_CHECKPOINT_SCHEMA_V2:
        if not isinstance(fusion_state, Mapping) or set(fusion_state) != {
            "future_mix_logits"
        }:
            raise ValueError("Tri-mode checkpoint fusion tensor names changed.")
        if not isinstance(fusion_state["future_mix_logits"], torch.Tensor):
            raise ValueError("Tri-mode checkpoint fusion state must contain tensors.")
        if fusion is not None:
            expected = fusion.trainable_state_dict()
            if any(
                name not in expected or value.shape != expected[name].shape
                for name, value in fusion_state.items()
            ):
                raise ValueError("Tri-mode checkpoint fusion shape changed.")
    elif fusion is not None:
        raise ValueError("A C1 fusion module cannot inspect a dual-mode checkpoint.")
    # The resolved config may legitimately describe image dimensions. Runtime
    # payloads such as samples, observations, and K/V are forbidden from the
    # trainer-state channel where they could otherwise be smuggled in.
    _reject_forbidden_keys(payload["trainer_state"], path="trainer_state")
    return {
        "schema": payload["schema"],
        "global_step": int(metadata["global_step"]),
        "epoch": int(metadata["epoch"]),
        "adapter_tensor_count": len(adapter_state),
        "fusion_tensor_count": 0 if fusion_state is None else len(fusion_state),
        "parent_checkpoint_sha256": metadata["parent_checkpoint_sha256"],
        "statistics_sha256": metadata["statistics_sha256"],
    }


def save_causal_policy_checkpoint(
    path: str | os.PathLike[str],
    **kwargs: Any,
) -> dict[str, Any]:
    """Atomically save and return the inspection report."""

    adapter = kwargs.get("adapter")
    payload = build_causal_policy_checkpoint(**kwargs)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return inspect_causal_policy_checkpoint_payload(
        payload,
        adapter=adapter,
        fusion=kwargs.get("fusion"),
    )


def load_causal_policy_checkpoint(
    path: str | os.PathLike[str],
    *,
    adapter: SharedActionDiTLoRAAdapter,
    expected_parent_checkpoint_sha256: str,
    expected_statistics_sha256: str,
    expected_checkpoint_schema: str = CAUSAL_POLICY_CHECKPOINT_SCHEMA,
    fusion: C1IntervalKVFusion | None = None,
) -> dict[str, Any]:
    """Load a resumable causal checkpoint after schema and parent validation."""

    try:
        # This resumable, locally produced project checkpoint intentionally
        # contains Python/NumPy RNG state in addition to tensors.
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("Causal checkpoint payload must be a mapping.")
    inspect_causal_policy_checkpoint_payload(
        payload,
        adapter=adapter,
        fusion=fusion,
    )
    if payload["schema"] != expected_checkpoint_schema:
        raise ValueError(
            "Causal checkpoint training-exposure schema changed: "
            f"{payload['schema']!r} != {expected_checkpoint_schema!r}."
        )
    metadata = payload["metadata"]
    if (
        metadata["parent_checkpoint_sha256"]
        != str(expected_parent_checkpoint_sha256).lower()
    ):
        raise ValueError("Causal checkpoint parent identity changed.")
    if metadata["statistics_sha256"] != str(expected_statistics_sha256).lower():
        raise ValueError("Causal checkpoint statistics identity changed.")
    adapter.load_lora_state_dict(payload["adapter_state_dict"], strict=True)
    if fusion is not None:
        fusion.load_trainable_state_dict(payload["fusion_state_dict"], strict=True)
    return dict(payload)
