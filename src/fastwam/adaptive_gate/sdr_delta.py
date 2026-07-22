"""Strict reconstruction of compact S-DR ActionDiT delta checkpoints."""
from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .provenance import sha256_file


DELTA_SCHEMA = "fastwam-action-dit-delta-v1"
ACTION_MOT_PREFIX = "mixtures.action."


def validate_action_dit_delta(
    payload: Mapping[str, Any],
    *,
    parent_checkpoint_sha256: str,
) -> Mapping[str, torch.Tensor]:
    if payload.get("schema") != DELTA_SCHEMA:
        raise ValueError(f"Unsupported ActionDiT delta schema: {payload.get('schema')!r}.")
    if payload.get("parent_checkpoint_sha256") != parent_checkpoint_sha256:
        raise ValueError("ActionDiT delta parent checkpoint SHA256 does not match.")
    action = payload.get("action_expert")
    if not isinstance(action, Mapping) or not action:
        raise ValueError("ActionDiT delta has no action_expert state.")
    invalid = [
        name
        for name, tensor in action.items()
        if not isinstance(name, str)
        or not torch.is_tensor(tensor)
        or not bool(torch.isfinite(tensor).all())
    ]
    if invalid:
        raise ValueError(f"ActionDiT delta contains invalid tensors: {invalid[:5]}.")
    provenance = payload.get("fastwam_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("ActionDiT delta has no FastWAM provenance.")
    if provenance.get("parent_checkpoint_sha256") != parent_checkpoint_sha256:
        raise ValueError("ActionDiT delta provenance has a different parent.")
    return action


def reconstruct_action_dit_delta(
    *,
    parent_checkpoint: str | os.PathLike[str],
    delta_checkpoint: str | os.PathLike[str],
    output_checkpoint: str | os.PathLike[str],
) -> dict[str, Any]:
    parent_path = Path(parent_checkpoint).expanduser().resolve()
    delta_path = Path(delta_checkpoint).expanduser().resolve()
    output_path = Path(output_checkpoint).expanduser().resolve()
    parent_sha = sha256_file(parent_path)
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    delta = torch.load(delta_path, map_location="cpu", weights_only=False)
    if not isinstance(parent, Mapping) or not isinstance(delta, Mapping):
        raise ValueError("Parent and delta checkpoints must be mappings.")
    parent_mot = parent.get("mot")
    if not isinstance(parent_mot, Mapping) or not parent_mot:
        raise ValueError("Parent E-I checkpoint has no mot state.")
    action = validate_action_dit_delta(
        delta,
        parent_checkpoint_sha256=parent_sha,
    )
    parent_action_keys = {
        name
        for name in parent_mot
        if str(name).startswith(ACTION_MOT_PREFIX)
    }
    expected_action_keys = {
        f"{ACTION_MOT_PREFIX}{name}" for name in action
    }
    if parent_action_keys != expected_action_keys:
        raise ValueError(
            "ActionDiT delta and parent action state schemas differ: "
            f"missing={sorted(parent_action_keys - expected_action_keys)[:5]}, "
            f"unexpected={sorted(expected_action_keys - parent_action_keys)[:5]}."
        )

    rebuilt_mot = dict(parent_mot)
    for name, tensor in action.items():
        target_name = f"{ACTION_MOT_PREFIX}{name}"
        target = parent_mot[target_name]
        if target.shape != tensor.shape or target.dtype != tensor.dtype:
            raise ValueError(
                f"ActionDiT tensor schema mismatch for {name}: "
                f"parent={tuple(target.shape)}/{target.dtype}, "
                f"delta={tuple(tensor.shape)}/{tensor.dtype}."
            )
        rebuilt_mot[target_name] = tensor
    rebuilt = dict(parent)
    rebuilt["mot"] = rebuilt_mot
    rebuilt["step"] = delta.get("step")
    rebuilt["torch_dtype"] = delta.get("torch_dtype")
    rebuilt["fastwam_provenance"] = delta.get("fastwam_provenance")
    rebuilt.pop("optimizer", None)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(rebuilt, temporary)
    os.replace(temporary, output_path)
    return {
        "schema": "fastwam-action-dit-reconstruction-v1",
        "status": "PASS",
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": parent_sha,
        "delta_checkpoint": str(delta_path),
        "delta_checkpoint_sha256": sha256_file(delta_path),
        "output_checkpoint": str(output_path),
        "output_checkpoint_sha256": sha256_file(output_path),
        "action_tensor_count": len(action),
    }


def load_action_dit_delta_into_model(
    model,
    *,
    delta_checkpoint: str | os.PathLike[str],
    parent_checkpoint_sha256: str,
) -> dict[str, Any]:
    delta_path = Path(delta_checkpoint).expanduser().resolve()
    delta = torch.load(delta_path, map_location="cpu", weights_only=False)
    if not isinstance(delta, Mapping):
        raise ValueError("ActionDiT delta checkpoint must be a mapping.")
    action = validate_action_dit_delta(
        delta,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
    )
    model.action_expert.load_state_dict(action, strict=True)
    provenance = delta["fastwam_provenance"]
    model.dual_regime_optimizer_steps = int(
        provenance["dual_regime_optimizer_steps"]
    )
    model.dual_regime_training_contract = dict(
        provenance["dual_regime_training_contract"]
    )
    model.action_regime_weight_uncond = float(
        provenance["action_regime_weight_uncond"]
    )
    model.warm_start_provenance = provenance.get("warm_start_provenance")
    return {
        "delta_checkpoint": str(delta_path),
        "delta_checkpoint_sha256": sha256_file(delta_path),
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "step": int(delta.get("step")),
        "action_tensor_count": len(action),
    }
