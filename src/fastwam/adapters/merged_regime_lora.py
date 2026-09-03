"""Offline materialization of an ActionDiT regime LoRA into plain linears."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .regime_lora import (
    REGIME_LORA_SIDECAR_SCHEMA,
    ActionDiTLoRAAdapter,
    RegimeLoRALinear,
    _replace_submodule,
    _validate_sha256,
    discover_action_dit_lora_targets,
)

FROZEN_UNCOND_ACTION_SCHEMA = "fastwam-frozen-uncond-action-v1"


@dataclass(frozen=True)
class MergedActionDiTAudit:
    """Structural and numerical summary of one offline ActionDiT merge."""

    target_names: tuple[str, ...]
    output_dtype: str
    parameter_bytes: int
    maximum_absolute_delta: float
    mean_absolute_delta: float

    @property
    def target_count(self) -> int:
        """Return the number of materialized linear projections."""

        return len(self.target_names)


def _assert_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"Cannot merge non-finite tensor {name!r}.")


def _parameter_bytes(module: nn.Module) -> int:
    return sum(
        parameter.numel() * parameter.element_size()
        for parameter in module.parameters()
    )


def _plain_action_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    state_dict = module.state_dict()
    lora_keys = tuple(
        name for name in state_dict if name.endswith((".lora_A", ".lora_B"))
    )
    if lora_keys:
        raise ValueError(
            "Frozen UNCOND action artifact cannot contain LoRA tensors: "
            f"{list(lora_keys)}."
        )
    return {name: tensor.detach().cpu().clone() for name, tensor in state_dict.items()}


def save_frozen_uncond_action_artifact(
    path: str | os.PathLike[str],
    *,
    action_dit: nn.Module,
    action_dit_config: Mapping[str, Any],
    parent_checkpoint_sha256: str,
    source_lora_sidecar_sha256: str,
    source_lora_metadata: Mapping[str, Any],
    merge_audit: MergedActionDiTAudit,
) -> None:
    """Atomically save one non-resumable plain Warm-UNCOND ActionDiT artifact."""

    target = Path(path)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite merged action artifact: {target}")
    parent_hash = _validate_sha256(parent_checkpoint_sha256)
    sidecar_hash = _validate_sha256(source_lora_sidecar_sha256)
    if source_lora_metadata.get("schema") != REGIME_LORA_SIDECAR_SCHEMA:
        raise ValueError(
            "Merged action source must use the regime-LoRA sidecar schema, got "
            f"{source_lora_metadata.get('schema')!r}."
        )
    source_parent = source_lora_metadata.get("parent_checkpoint_sha256")
    if source_parent != parent_hash:
        raise ValueError(
            "Warm UNCOND sidecar is bound to a different parent checkpoint: "
            f"expected {parent_hash}, got {source_parent}."
        )
    if tuple(source_lora_metadata.get("target_names", ())) != merge_audit.target_names:
        raise ValueError("Merge audit targets differ from the Warm UNCOND sidecar.")
    rank = int(source_lora_metadata["rank"])
    alpha = float(source_lora_metadata["alpha"])
    expected_scaling = alpha / rank
    if any(isinstance(module, RegimeLoRALinear) for module in action_dit.modules()):
        raise ValueError("Frozen UNCOND action artifact requires a plain ActionDiT.")
    if any(parameter.requires_grad for parameter in action_dit.parameters()):
        raise ValueError("Frozen UNCOND action artifact contains trainable parameters.")

    payload = {
        "schema": FROZEN_UNCOND_ACTION_SCHEMA,
        "parent_checkpoint_sha256": parent_hash,
        "source_lora_sidecar_sha256": sidecar_hash,
        "source_lora_parent_checkpoint_sha256": source_parent,
        "rank": rank,
        "alpha": alpha,
        "scaling": expected_scaling,
        "target_groups": list(source_lora_metadata["target_groups"]),
        "target_names": list(merge_audit.target_names),
        "output_dtype": merge_audit.output_dtype,
        "action_dit_config": dict(action_dit_config),
        "action_expert_state_dict": _plain_action_state_dict(action_dit),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def load_frozen_uncond_action_artifact(
    path: str | os.PathLike[str],
    *,
    action_dit: nn.Module,
    expected_action_dit_config: Mapping[str, Any],
    expected_parent_checkpoint_sha256: str,
    expected_source_lora_sidecar_sha256: str | None = None,
) -> dict[str, Any]:
    """Strictly load a plain Warm-UNCOND ActionDiT artifact into a fresh module."""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(
            "Frozen UNCOND action artifact must be a mapping, got "
            f"{type(payload).__name__}."
        )
    if payload.get("schema") != FROZEN_UNCOND_ACTION_SCHEMA:
        raise ValueError(
            f"Unsupported frozen UNCOND action schema {payload.get('schema')!r}."
        )
    expected_parent = _validate_sha256(expected_parent_checkpoint_sha256)
    if payload.get("parent_checkpoint_sha256") != expected_parent:
        raise ValueError(
            "Frozen UNCOND action parent mismatch: "
            f"expected {expected_parent}, "
            f"got {payload.get('parent_checkpoint_sha256')}."
        )
    if payload.get("source_lora_parent_checkpoint_sha256") != expected_parent:
        raise ValueError("Frozen UNCOND artifact source LoRA parent mismatch.")
    if expected_source_lora_sidecar_sha256 is not None:
        expected_sidecar = _validate_sha256(expected_source_lora_sidecar_sha256)
        if payload.get("source_lora_sidecar_sha256") != expected_sidecar:
            raise ValueError(
                "Frozen UNCOND action source sidecar mismatch: "
                f"expected {expected_sidecar}, "
                f"got {payload.get('source_lora_sidecar_sha256')}."
            )
    expected_config = dict(expected_action_dit_config)
    if payload.get("action_dit_config") != expected_config:
        raise ValueError(
            "Frozen UNCOND action config mismatch: "
            f"expected {expected_config}, got {payload.get('action_dit_config')}."
        )
    rank = int(payload.get("rank", 0))
    alpha = float(payload.get("alpha", 0.0))
    if rank <= 0 or alpha <= 0.0 or payload.get("scaling") != alpha / rank:
        raise ValueError("Frozen UNCOND action artifact has an invalid LoRA scale.")
    target_names = tuple(payload.get("target_names", ()))
    if not target_names or len(target_names) != len(set(target_names)):
        raise ValueError("Frozen UNCOND action artifact has invalid target names.")
    if any(isinstance(module, RegimeLoRALinear) for module in action_dit.modules()):
        raise ValueError("Fresh frozen UNCOND target must be a plain ActionDiT.")

    state_dict = payload.get("action_expert_state_dict")
    if not isinstance(state_dict, Mapping):
        raise TypeError("Frozen UNCOND action artifact is missing its state dict.")
    expected_state = action_dit.state_dict()
    provided_keys = set(state_dict)
    expected_keys = set(expected_state)
    if provided_keys != expected_keys:
        raise ValueError(
            "Frozen UNCOND action state key mismatch: "
            f"missing={sorted(expected_keys - provided_keys)}, "
            f"unexpected={sorted(provided_keys - expected_keys)}."
        )
    for name in sorted(expected_keys):
        value = state_dict[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Frozen UNCOND state {name!r} is not a tensor.")
        if value.shape != expected_state[name].shape:
            raise ValueError(
                f"Frozen UNCOND state shape mismatch for {name}: "
                f"expected {tuple(expected_state[name].shape)}, "
                f"got {tuple(value.shape)}."
            )
        _assert_finite(name, value)
    action_dit.load_state_dict(state_dict, strict=True)
    action_dit.requires_grad_(False)
    action_dit.eval()
    return {
        key: value
        for key, value in payload.items()
        if key != "action_expert_state_dict"
    }


def merge_action_dit_lora_(
    adapter: ActionDiTLoRAAdapter,
) -> MergedActionDiTAudit:
    """Merge one loaded ActionDiT LoRA into its CPU base, in place.

    This primitive is intentionally offline-only. The caller must work on an
    independent CPU copy of the parent ActionDiT. Every adapted projection is
    replaced by a frozen plain :class:`torch.nn.Linear` whose weight is
    ``W + (alpha / rank) * B @ A`` computed in FP32 and cast once to the base
    dtype.

    Args:
        adapter: Adapter owning the fully loaded ActionDiT copy to materialize.

    Returns:
        A summary of the materialized targets and weight deltas.

    Raises:
        ValueError: If the adapter contract, device, dropout, shapes, or tensor
            values do not permit a deterministic static merge.
        TypeError: If a declared target is not a direct ``RegimeLoRALinear``.
    """

    if not isinstance(adapter, ActionDiTLoRAAdapter):
        raise TypeError(
            f"`adapter` must be an ActionDiTLoRAAdapter, got {type(adapter).__name__}."
        )
    if adapter.config.dropout != 0.0:
        raise ValueError(
            "Static ActionDiT materialization requires LoRA dropout 0, got "
            f"{adapter.config.dropout}."
        )

    expected_targets = discover_action_dit_lora_targets(
        adapter.action_dit,
        adapter.config.target_groups,
        strict=adapter.config.strict_target_discovery,
    )
    if tuple(adapter.target_names) != expected_targets:
        raise ValueError(
            "LoRA target coverage changed before materialization: "
            f"adapter={list(adapter.target_names)}, "
            f"discovered={list(expected_targets)}."
        )
    actual_adapted = tuple(
        name
        for name, module in adapter.action_dit.named_modules()
        if isinstance(module, RegimeLoRALinear)
    )
    if len(actual_adapted) != len(expected_targets) or set(actual_adapted) != set(
        expected_targets
    ):
        raise ValueError(
            "Adapted ActionDiT modules do not exactly match the declared targets: "
            f"adapted={list(actual_adapted)}, expected={list(expected_targets)}."
        )

    deltas: list[torch.Tensor] = []
    for name, adapted in adapter.iter_adapted_linears():
        direct_module = adapter.action_dit.get_submodule(name)
        if direct_module is not adapted:
            raise TypeError(
                "Offline materialization requires direct RegimeLoRALinear targets; "
                f"{name!r} is wrapped by {type(direct_module).__name__}."
            )
        if adapted.weight.device.type != "cpu":
            raise ValueError(
                "Static ActionDiT materialization must run on a CPU model copy; "
                f"{name!r} is on {adapted.weight.device}."
            )
        if adapted.lora_A.device.type != "cpu" or adapted.lora_B.device.type != "cpu":
            raise ValueError(
                "Static ActionDiT materialization requires CPU LoRA factors; "
                f"{name!r} has A/B on {adapted.lora_A.device}/{adapted.lora_B.device}."
            )
        expected_a_shape = (adapted.rank, adapted.in_features)
        expected_b_shape = (adapted.out_features, adapted.rank)
        if tuple(adapted.lora_A.shape) != expected_a_shape:
            raise ValueError(
                f"LoRA A shape mismatch for {name}: expected {expected_a_shape}, "
                f"got {tuple(adapted.lora_A.shape)}."
            )
        if tuple(adapted.lora_B.shape) != expected_b_shape:
            raise ValueError(
                f"LoRA B shape mismatch for {name}: expected {expected_b_shape}, "
                f"got {tuple(adapted.lora_B.shape)}."
            )
        _assert_finite(f"{name}.weight", adapted.weight)
        _assert_finite(f"{name}.lora_A", adapted.lora_A)
        _assert_finite(f"{name}.lora_B", adapted.lora_B)
        if adapted.bias is not None:
            _assert_finite(f"{name}.bias", adapted.bias)

        base_fp32 = adapted.weight.detach().float()
        delta_fp32 = (
            adapted.lora_B.detach().float() @ adapted.lora_A.detach().float()
        ) * float(adapted.scaling)
        merged_fp32 = base_fp32 + delta_fp32
        _assert_finite(f"{name}.merged_weight", merged_fp32)

        replacement = nn.Linear(
            adapted.in_features,
            adapted.out_features,
            bias=adapted.bias is not None,
            device="cpu",
            dtype=adapted.weight.dtype,
        )
        with torch.no_grad():
            replacement.weight.copy_(merged_fp32.to(dtype=adapted.weight.dtype))
            if adapted.bias is not None:
                replacement.bias.copy_(adapted.bias.detach())
        replacement.train(adapted.training)
        replacement.requires_grad_(False)
        _replace_submodule(adapter.action_dit, name, replacement)
        deltas.append(delta_fp32.reshape(-1))

    remaining_lora_modules = tuple(
        name
        for name, module in adapter.action_dit.named_modules()
        if isinstance(module, RegimeLoRALinear)
    )
    if remaining_lora_modules:
        raise RuntimeError(
            "Materialized ActionDiT still contains RegimeLoRALinear modules: "
            f"{list(remaining_lora_modules)}."
        )
    remaining_lora_parameters = tuple(
        name
        for name, _ in adapter.action_dit.named_parameters()
        if name.endswith((".lora_A", ".lora_B"))
    )
    if remaining_lora_parameters:
        raise RuntimeError(
            "Materialized ActionDiT still contains LoRA parameters: "
            f"{list(remaining_lora_parameters)}."
        )
    adapter.action_dit.requires_grad_(False)

    flattened_delta = torch.cat(deltas)
    parameter_dtypes = {
        parameter.dtype for parameter in adapter.action_dit.parameters()
    }
    output_dtype = (
        str(next(iter(parameter_dtypes))) if len(parameter_dtypes) == 1 else "mixed"
    )
    return MergedActionDiTAudit(
        target_names=expected_targets,
        output_dtype=output_dtype,
        parameter_bytes=_parameter_bytes(adapter.action_dit),
        maximum_absolute_delta=float(flattened_delta.abs().max().item()),
        mean_absolute_delta=float(flattened_delta.abs().mean().item()),
    )
