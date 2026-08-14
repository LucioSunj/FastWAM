"""Offline paired causal audit for DINO-to-action contribution."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data._utils.collate import default_collate

from fastwam.adapters import sha256_file
from fastwam.models.wan22.visual_sidecar import (
    DinoContributionDiagnosticsCollector,
)
from fastwam.p1_dino_bc_checkpoint import (
    P1_DINO_BC_CHECKPOINT_SCHEMA,
    load_p1_dino_bc_trainables,
)
from fastwam.p1_dino_bc_full_checkpoint import (
    P1_DINO_BC_FULL_CHECKPOINT_SCHEMA,
    P1_DINO_BC_FULL_CHECKPOINT_V2_SCHEMA,
    load_p1_dino_bc_full_trainables,
)
from fastwam.p1_dino_bc_runner import audit_p1_assets, build_real_p1_policy
from fastwam.p1_dino_contribution_v2 import (
    build_frozen_causal_ledger,
    load_frozen_causal_ledger,
)
from fastwam.uncond_bc import stateless_validation_flow_inputs
from fastwam.uncond_bc_trainer import (
    _atomic_json,
    _instantiate_bc_dataset,
    sha256_artifact,
)
from fastwam.utils import misc


P1_DINO_CAUSAL_AUDIT_SCHEMA = "fastwam-p1-dino-causal-audit-v2"
SUPPORTED_AUDIT_MODES = (
    "correct",
    "off",
    "shuffled",
    "task_paired",
    "drop_main",
    "drop_wrist",
)


def _load_split_dataset(cfg: DictConfig, split: str):
    from fastwam.uncond_bc import SampleIdentityDataset

    if split not in {"train", "validation"}:
        raise ValueError(f"Unsupported causal-audit dataset split {split!r}.")
    return SampleIdentityDataset(
        _instantiate_bc_dataset(
            cfg.data[split],
            expected_seed=int(cfg.seed),
        ),
        namespace=f"{split}-seed42",
    )


def _load_validation_dataset(cfg: DictConfig):
    return _load_split_dataset(cfg, "validation")


def build_causal_audit_ledger(
    cfg: DictConfig,
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Build the model-independent ledger before allocating a model."""

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(target.parent)
    dataset = _load_validation_dataset(cfg)
    training_dataset = _load_split_dataset(cfg, "train")
    return build_frozen_causal_ledger(
        dataset,
        target,
        negative_fallback_dataset=training_dataset,
    )


def _load_checkpoint_trainables(
    checkpoint: Path,
    *,
    checkpoint_kind: str,
    policy: Any,
    assets: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    schema = raw.get("schema") if isinstance(raw, Mapping) else None
    inferred = (
        "best"
        if schema == P1_DINO_BC_CHECKPOINT_SCHEMA
        else "full"
        if schema
        in {
            P1_DINO_BC_FULL_CHECKPOINT_SCHEMA,
            P1_DINO_BC_FULL_CHECKPOINT_V2_SCHEMA,
        }
        else None
    )
    requested = str(checkpoint_kind).strip().lower()
    if requested not in {"auto", "best", "full"}:
        raise ValueError("checkpoint_kind must be auto, best, or full.")
    if inferred is None or (requested != "auto" and requested != inferred):
        raise ValueError(
            f"Checkpoint kind/schema mismatch: requested={requested}, schema={schema}."
        )
    common = {
        "path": checkpoint,
        "adapter": policy.lora_adapter,
        "reader": policy.visual_reader,
        "expected_parent_checkpoint_sha256": assets["parent_checkpoint_sha256"],
        "expected_dinov3_weights_sha256": assets["dinov3_weights_sha256"],
    }
    if inferred == "best":
        payload = load_p1_dino_bc_trainables(**common)
    else:
        payload = load_p1_dino_bc_full_trainables(**common)
    return payload, inferred


def _rgb_sha256(sample: Mapping[str, Any]) -> str:
    pixels = sample["p1_camera_pixels"]
    valid = sample["p1_camera_valid_mask"]
    digest = hashlib.sha256()
    digest.update(pixels.contiguous().numpy().tobytes())
    digest.update(valid.to(torch.uint8).contiguous().numpy().tobytes())
    return digest.hexdigest()


def _quantiles(values: torch.Tensor) -> dict[str, float | int | None]:
    flattened = values.detach().float().cpu().reshape(-1)
    if flattened.numel() == 0:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p95": None}
    if not bool(torch.isfinite(flattened).all().item()):
        raise FloatingPointError("Non-finite causal-audit diagnostic.")
    return {
        "count": int(flattened.numel()),
        "mean": float(flattened.mean().item()),
        "p50": float(torch.quantile(flattened, 0.50).item()),
        "p90": float(torch.quantile(flattened, 0.90).item()),
        "p95": float(torch.quantile(flattened, 0.95).item()),
    }


def _append_diagnostics(
    destination: dict[str, list[torch.Tensor]],
    records: Sequence[Mapping[str, Any]],
) -> None:
    for record in records:
        layer = int(record["layer_index"])
        valid = record["camera_valid_mask"]
        for name in ("gate_logits", "effective_gate"):
            destination[f"layer_{layer}/{name}"].append(record[name].reshape(-1))
        for name in (
            "projected_residual_over_hidden",
            "effective_residual_over_hidden",
        ):
            value = record[name]
            destination[f"layer_{layer}/{name}"].append(value.reshape(-1))
            for token_index in range(value.shape[1]):
                destination[f"layer_{layer}/action_token_{token_index}/{name}"].append(
                    value[:, token_index].reshape(-1)
                )
        for name in (
            "attention_entropy",
            "attention_top1",
            "attention_top5",
            "effective_patch_count",
        ):
            value = record[name]
            mask = valid[:, :, None].expand_as(value)
            destination[f"layer_{layer}/{name}"].append(value[mask].reshape(-1))
            for view_index, camera_id in enumerate(record["camera_ids"]):
                view_mask = valid[:, view_index, None].expand_as(value[:, view_index])
                destination[f"layer_{layer}/view_{camera_id}/{name}"].append(
                    value[:, view_index][view_mask].reshape(-1)
                )
                for token_index in range(value.shape[2]):
                    token_valid = valid[:, view_index]
                    destination[
                        f"layer_{layer}/view_{camera_id}/"
                        f"action_token_{token_index}/{name}"
                    ].append(value[:, view_index, token_index][token_valid].reshape(-1))
            for token_index in range(value.shape[2]):
                token_values = value[:, :, token_index]
                destination[f"layer_{layer}/action_token_{token_index}/{name}"].append(
                    token_values[valid].reshape(-1)
                )


def _finalize_diagnostics(
    values: Mapping[str, Sequence[torch.Tensor]],
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    tensors = {
        key: torch.cat(tuple(parts)).float() for key, parts in values.items() if parts
    }
    return {key: _quantiles(value) for key, value in tensors.items()}, tensors


def _new_metric_accumulator() -> dict[str, Any]:
    return {
        "sample_count": 0,
        "valid_action_count": 0,
        "loss_sum": 0.0,
        "dimension_sse": None,
        "velocity_relative_delta_sum": 0.0,
        "action_relative_delta_sum": 0.0,
    }


def _accumulate_output(
    accum: dict[str, Any],
    output: Mapping[str, torch.Tensor],
    *,
    correct_prediction: torch.Tensor,
    action_is_pad: torch.Tensor,
) -> None:
    prediction = output["prediction"].detach().float()
    reference = correct_prediction.detach().float()
    batch_size = prediction.shape[0]
    valid_count = int(output["valid_action_count"].item())
    accum["sample_count"] += batch_size
    accum["valid_action_count"] += valid_count
    per_sample_loss = output.get("loss_action_bc_per_sample")
    if per_sample_loss is None or per_sample_loss.shape != (batch_size,):
        raise ValueError("Causal audit requires one action loss per anchor.")
    accum["loss_sum"] += float(per_sample_loss.detach().double().sum().item())
    dimension_sse = output["mse_per_dimension"].detach().float().cpu() * valid_count
    accum["dimension_sse"] = (
        dimension_sse
        if accum["dimension_sse"] is None
        else accum["dimension_sse"] + dimension_sse
    )
    valid = (~action_is_pad.bool()).to(prediction.device)
    valid = valid.unsqueeze(2).expand_as(prediction)
    delta = (prediction - reference).masked_fill(~valid, 0.0)
    reference_valid = reference.masked_fill(~valid, 0.0)
    velocity_relative = torch.linalg.vector_norm(
        delta.reshape(batch_size, -1), dim=1
    ) / (
        torch.linalg.vector_norm(
            reference_valid.reshape(batch_size, -1), dim=1
        ).clamp_min(torch.finfo(torch.float32).eps)
    )
    sigma = output["timestep"].detach().float() / 1000.0
    sigma = sigma[:, None, None]
    predicted_action = output["noisy_action"].detach().float() - sigma * prediction
    reference_action = output["noisy_action"].detach().float() - sigma * reference
    action_delta = (predicted_action - reference_action).masked_fill(~valid, 0.0)
    reference_action = reference_action.masked_fill(~valid, 0.0)
    action_relative = torch.linalg.vector_norm(
        action_delta.reshape(batch_size, -1), dim=1
    ) / torch.linalg.vector_norm(
        reference_action.reshape(batch_size, -1), dim=1
    ).clamp_min(torch.finfo(torch.float32).eps)
    accum["velocity_relative_delta_sum"] += float(velocity_relative.sum().item())
    accum["action_relative_delta_sum"] += float(action_relative.sum().item())


def _per_sample_output_metrics(
    output: Mapping[str, torch.Tensor],
    *,
    correct_prediction: torch.Tensor,
    action_is_pad: torch.Tensor,
    gripper_dimension: int,
) -> list[dict[str, float]]:
    prediction = output["prediction"].detach().float()
    reference = correct_prediction.detach().float()
    batch_size = prediction.shape[0]
    valid = (~action_is_pad.bool()).to(prediction.device)
    valid = valid.unsqueeze(2).expand_as(prediction)
    delta = (prediction - reference).masked_fill(~valid, 0.0)
    reference_valid = reference.masked_fill(~valid, 0.0)
    velocity_relative = torch.linalg.vector_norm(
        delta.reshape(batch_size, -1), dim=1
    ) / torch.linalg.vector_norm(
        reference_valid.reshape(batch_size, -1), dim=1
    ).clamp_min(torch.finfo(torch.float32).eps)
    sigma = output["timestep"].detach().float()[:, None, None] / 1000.0
    predicted_action = output["noisy_action"].detach().float() - sigma * prediction
    reference_action = output["noisy_action"].detach().float() - sigma * reference
    action_delta = (predicted_action - reference_action).masked_fill(~valid, 0.0)
    reference_action = reference_action.masked_fill(~valid, 0.0)
    action_relative = torch.linalg.vector_norm(
        action_delta.reshape(batch_size, -1), dim=1
    ) / torch.linalg.vector_norm(
        reference_action.reshape(batch_size, -1), dim=1
    ).clamp_min(torch.finfo(torch.float32).eps)
    dimensions = output["mse_per_dimension_per_sample"].detach().float()
    pose_indices = [
        index for index in range(dimensions.shape[1]) if index != gripper_dimension
    ]
    losses = output["loss_action_bc_per_sample"].detach().float()
    return [
        {
            "loss_action_bc": float(losses[index].item()),
            "mse_pose": float(dimensions[index, pose_indices].mean().item()),
            "mse_gripper": float(dimensions[index, gripper_dimension].item()),
            "velocity_relative_delta": float(velocity_relative[index].item()),
            "action_relative_delta": float(action_relative[index].item()),
        }
        for index in range(batch_size)
    ]


def _finalize_metric(
    accum: Mapping[str, Any], *, gripper_dimension: int
) -> dict[str, Any]:
    sample_count = int(accum["sample_count"])
    valid_count = int(accum["valid_action_count"])
    if sample_count <= 0 or valid_count <= 0 or accum["dimension_sse"] is None:
        raise ValueError("Causal audit metric accumulator is empty.")
    dimensions = accum["dimension_sse"] / valid_count
    pose = [index for index in range(len(dimensions)) if index != gripper_dimension]
    return {
        "sample_count": sample_count,
        "valid_action_count": valid_count,
        "loss_action_bc": accum["loss_sum"] / sample_count,
        "mse_per_dimension": dimensions.tolist(),
        "mse_pose": float(dimensions[pose].mean().item()),
        "mse_gripper": float(dimensions[gripper_dimension].item()),
        "velocity_relative_delta": accum["velocity_relative_delta_sum"] / sample_count,
        "action_relative_delta": accum["action_relative_delta_sum"] / sample_count,
    }


@torch.no_grad()
def run_causal_audit_shard(
    cfg: DictConfig,
    *,
    checkpoint: str | os.PathLike[str],
    checkpoint_kind: str,
    ledger_path: str | os.PathLike[str],
    shard_index: int,
    num_shards: int,
    modes: Sequence[str],
    draw_seeds: Sequence[int],
    output_dir: str | os.PathLike[str],
    device: torch.device | str,
) -> dict[str, Any]:
    """Evaluate one deterministic checkpoint/ledger shard on one GPU."""

    started = time.time()
    normalized_modes = tuple(str(mode).strip().lower() for mode in modes)
    if not normalized_modes or len(set(normalized_modes)) != len(normalized_modes):
        raise ValueError("Audit modes must be non-empty and unique.")
    invalid_modes = sorted(set(normalized_modes) - set(SUPPORTED_AUDIT_MODES))
    if invalid_modes:
        raise ValueError(f"Unsupported audit modes: {invalid_modes}.")
    if "correct" not in normalized_modes:
        raise ValueError(
            "Causal audit requires correct memory as its paired reference."
        )
    if not 0 <= int(shard_index) < int(num_shards):
        raise ValueError("Invalid audit shard index/count.")
    if not draw_seeds or any(int(seed) < 0 for seed in draw_seeds):
        raise ValueError("Audit draw seeds must be non-empty and non-negative.")

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Causal-audit output is not empty: {destination}.")
    destination.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(destination)
    ledger = load_frozen_causal_ledger(ledger_path)
    all_entries = ledger["anchors"]
    entries = [
        value
        for index, value in enumerate(all_entries)
        if index % int(num_shards) == int(shard_index)
    ]
    if not entries:
        raise ValueError("Audit shard has no ledger anchors.")
    assets = audit_p1_assets(cfg)
    resolved_device = torch.device(device)
    policy, parent_load = build_real_p1_policy(cfg, device=resolved_device)
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    checkpoint_payload, inferred_kind = _load_checkpoint_trainables(
        checkpoint_path,
        checkpoint_kind=checkpoint_kind,
        policy=policy,
        assets=assets,
    )
    policy.eval()
    dataset = _load_validation_dataset(cfg)
    training_dataset = (
        _load_split_dataset(cfg, "train")
        if any(value.get("negative_split") == "train" for value in entries)
        else None
    )

    dino_calls = {"count": 0}

    def count_dino(_module, _inputs, _outputs) -> None:
        dino_calls["count"] += 1

    hook = policy.visual_encoder.register_forward_hook(count_dino)
    metrics: dict[str, dict[str, Any]] = {
        mode: _new_metric_accumulator() for mode in normalized_modes
    }
    task_metrics: dict[str, dict[str, dict[str, Any]]] = defaultdict(
        lambda: {mode: _new_metric_accumulator() for mode in normalized_modes}
    )
    diagnostic_parts: dict[str, dict[str, list[torch.Tensor]]] = {
        mode: defaultdict(list) for mode in normalized_modes
    }
    paired_records: list[dict[str, Any]] = []
    microbatch = int(cfg.get("causal_audit", {}).get("microbatch_size", 16))
    try:
        by_task: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
        for entry in entries:
            by_task[(int(entry["suite_index"]), int(entry["task_index"]))].append(entry)
        for task_key in sorted(by_task):
            task_name = f"suite_{task_key[0]}/task_{task_key[1]}"
            task_entries = by_task[task_key]
            for offset in range(0, len(task_entries), microbatch):
                selected = task_entries[offset : offset + microbatch]
                anchor_samples = [
                    dataset[int(entry["anchor_dataset_index"])] for entry in selected
                ]
                negative_samples = [
                    (
                        dataset
                        if entry["negative_split"] == "validation"
                        else training_dataset
                    )[int(entry["negative_dataset_index"])]
                    for entry in selected
                ]
                for entry, anchor, negative in zip(
                    selected,
                    anchor_samples,
                    negative_samples,
                    strict=True,
                ):
                    if _rgb_sha256(anchor) != entry["anchor_rgb_sha256"]:
                        raise ValueError(
                            f"Anchor RGB hash drifted: {entry['anchor_id']}."
                        )
                    if _rgb_sha256(negative) != entry["negative_rgb_sha256"]:
                        raise ValueError(
                            f"Negative RGB hash drifted: {entry['negative_id']}."
                        )
                anchor_batch = default_collate(anchor_samples)
                negative_batch = default_collate(negative_samples)
                condition = policy.prepare_action_condition(
                    anchor_batch,
                    include_visual=True,
                )
                negative_memory = policy.visual_encoder.prepare_memory(
                    "uncond",
                    policy._camera_batch(negative_batch),
                )
                if negative_memory is None:
                    raise RuntimeError("Audit hard-negative DINO memory is missing.")
                identities = list(anchor_batch["sample_identity"])
                for draw_seed in draw_seeds:
                    timestep, noise = stateless_validation_flow_inputs(
                        sample_identities=identities,
                        action_shape=tuple(anchor_batch["action"].shape),
                        scheduler=policy.actor.train_action_scheduler,
                        seed=int(draw_seed),
                        device=resolved_device,
                        dtype=policy.dtype,
                    )
                    outputs: dict[str, Mapping[str, torch.Tensor]] = {}
                    for mode in normalized_modes:
                        collector = DinoContributionDiagnosticsCollector()
                        kwargs: dict[str, Any] = {"memory_mode": mode}
                        if mode == "task_paired":
                            kwargs = {
                                "memory_mode": "correct",
                                "memory_override": negative_memory,
                            }
                        elif mode == "shuffled":
                            batch_size = len(selected)
                            if batch_size < 2:
                                raise ValueError(
                                    "Shuffled audit requires batch size >= 2."
                                )
                            kwargs["memory_permutation"] = torch.roll(
                                torch.arange(batch_size),
                                shifts=1,
                            )
                        with policy.visual_reader.capture_diagnostics(collector):
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                outputs[mode] = policy.loss_from_prepared_condition(
                                    anchor_batch,
                                    condition=condition,
                                    timestep=timestep,
                                    noise=noise,
                                    return_prediction=True,
                                    **kwargs,
                                )
                        _append_diagnostics(diagnostic_parts[mode], collector.records)
                    correct_prediction = outputs["correct"]["prediction"]
                    for mode, output in outputs.items():
                        _accumulate_output(
                            metrics[mode],
                            output,
                            correct_prediction=correct_prediction,
                            action_is_pad=anchor_batch["action_is_pad"],
                        )
                        per_sample = _per_sample_output_metrics(
                            output,
                            correct_prediction=correct_prediction,
                            action_is_pad=anchor_batch["action_is_pad"],
                            gripper_dimension=policy.config.gripper_dimension,
                        )
                        for entry, sample_metric in zip(
                            selected,
                            per_sample,
                            strict=True,
                        ):
                            paired_records.append(
                                {
                                    "anchor_id": entry["anchor_id"],
                                    "suite_index": int(entry["suite_index"]),
                                    "task_index": int(entry["task_index"]),
                                    "draw_seed": int(draw_seed),
                                    "mode": mode,
                                    **sample_metric,
                                }
                            )
                        _accumulate_output(
                            task_metrics[task_name][mode],
                            output,
                            correct_prediction=correct_prediction,
                            action_is_pad=anchor_batch["action_is_pad"],
                        )
    finally:
        hook.remove()

    finalized = {
        mode: _finalize_metric(
            value,
            gripper_dimension=policy.config.gripper_dimension,
        )
        for mode, value in metrics.items()
    }
    finalized_tasks = {
        task: {
            mode: _finalize_metric(
                value,
                gripper_dimension=policy.config.gripper_dimension,
            )
            for mode, value in modes_by_task.items()
        }
        for task, modes_by_task in task_metrics.items()
    }
    diagnostic_summary: dict[str, Any] = {}
    diagnostic_tensors: dict[str, dict[str, torch.Tensor]] = {}
    for mode, parts in diagnostic_parts.items():
        summary, tensors = _finalize_diagnostics(parts)
        diagnostic_summary[mode] = summary
        diagnostic_tensors[mode] = tensors
    diagnostic_path = destination / "diagnostic_values.pt"
    temporary = diagnostic_path.with_name(f".{diagnostic_path.name}.tmp")
    try:
        torch.save(diagnostic_tensors, temporary)
        os.replace(temporary, diagnostic_path)
    finally:
        temporary.unlink(missing_ok=True)

    result = {
        "schema": P1_DINO_CAUSAL_AUDIT_SCHEMA,
        "status": "PASS",
        "checkpoint": {
            "path": str(checkpoint_path),
            "kind": inferred_kind,
            "schema": checkpoint_payload["schema"],
            "global_step": int(checkpoint_payload["global_step"]),
            "sha256": sha256_file(checkpoint_path),
        },
        "ledger": {
            "path": str(Path(ledger_path).expanduser().resolve()),
            "sha256": ledger["ledger_file_sha256"],
            "content_sha256": ledger["content_sha256"],
            "total_anchor_count": len(all_entries),
        },
        "shard_index": int(shard_index),
        "num_shards": int(num_shards),
        "anchor_count": len(entries),
        "modes": list(normalized_modes),
        "draw_seeds": [int(seed) for seed in draw_seeds],
        "metrics": finalized,
        "tasks": finalized_tasks,
        "paired_records": paired_records,
        "diagnostics": diagnostic_summary,
        "diagnostic_values": {
            "path": str(diagnostic_path),
            "sha256": sha256_artifact(diagnostic_path),
        },
        "dino_forward_calls": dino_calls["count"],
        "expected_dino_forward_calls": (
            2
            * len(draw_seeds)
            * sum(math.ceil(len(values) / microbatch) for values in by_task.values())
            / len(draw_seeds)
        ),
        "parent_load": parent_load,
        "assets": assets,
        "finite": True,
        "elapsed_seconds": time.time() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(resolved_device)),
    }
    # DINO memory is prepared once for anchors and once for negatives per data
    # microbatch, then reused across every draw and intervention.
    expected_calls = 2 * sum(
        math.ceil(len(values) / microbatch) for values in by_task.values()
    )
    result["expected_dino_forward_calls"] = expected_calls
    if dino_calls["count"] != expected_calls:
        raise RuntimeError(
            "Causal audit DINO call count changed: "
            f"{dino_calls['count']} != {expected_calls}."
        )
    _atomic_json(destination / "audit.json", result)
    return result


def merge_causal_audit_shards(
    shard_directories: Sequence[str | os.PathLike[str]],
    output_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Merge paired shard evidence and compute exact diagnostic quantiles."""

    shards = []
    diagnostic_payloads = []
    for directory in shard_directories:
        root = Path(directory).expanduser().resolve()
        shards.append(json.loads((root / "audit.json").read_text(encoding="utf-8")))
        diagnostic_payloads.append(
            torch.load(
                root / "diagnostic_values.pt", map_location="cpu", weights_only=True
            )
        )
    if not shards or any(value.get("status") != "PASS" for value in shards):
        raise ValueError("All causal-audit shards must pass before merging.")
    first = shards[0]
    identity = (
        first["checkpoint"]["sha256"],
        first["ledger"]["sha256"],
        tuple(first["modes"]),
        tuple(first["draw_seeds"]),
        first["num_shards"],
    )
    if any(
        (
            value["checkpoint"]["sha256"],
            value["ledger"]["sha256"],
            tuple(value["modes"]),
            tuple(value["draw_seeds"]),
            value["num_shards"],
        )
        != identity
        for value in shards
    ):
        raise ValueError("Causal-audit shard contracts differ.")
    if sorted(value["shard_index"] for value in shards) != list(
        range(first["num_shards"])
    ):
        raise ValueError("Causal-audit shard coverage is incomplete.")

    modes = first["modes"]
    paired_records = [
        record for shard in shards for record in shard.get("paired_records", ())
    ]
    expected_record_count = 640 * len(first["draw_seeds"]) * len(first["modes"])
    identities = {
        (record["anchor_id"], record["draw_seed"], record["mode"])
        for record in paired_records
    }
    if len(paired_records) != expected_record_count or len(identities) != len(
        paired_records
    ):
        raise ValueError("Merged audit paired-record coverage changed.")
    merged_metrics = {}
    for mode in modes:
        sample_count = sum(value["metrics"][mode]["sample_count"] for value in shards)
        valid_count = sum(
            value["metrics"][mode]["valid_action_count"] for value in shards
        )
        dimensions = (
            sum(
                torch.tensor(value["metrics"][mode]["mse_per_dimension"])
                * value["metrics"][mode]["valid_action_count"]
                for value in shards
            )
            / valid_count
        )
        merged_metrics[mode] = {
            "sample_count": sample_count,
            "valid_action_count": valid_count,
            "loss_action_bc": sum(
                value["metrics"][mode]["loss_action_bc"]
                * value["metrics"][mode]["sample_count"]
                for value in shards
            )
            / sample_count,
            "mse_per_dimension": dimensions.tolist(),
            "mse_pose": float(dimensions[:-1].mean().item()),
            "mse_gripper": float(dimensions[-1].item()),
            "velocity_relative_delta": sum(
                value["metrics"][mode]["velocity_relative_delta"]
                * value["metrics"][mode]["sample_count"]
                for value in shards
            )
            / sample_count,
            "action_relative_delta": sum(
                value["metrics"][mode]["action_relative_delta"]
                * value["metrics"][mode]["sample_count"]
                for value in shards
            )
            / sample_count,
        }
    correct_loss = merged_metrics["correct"]["loss_action_bc"]
    relative_gaps = {
        mode: (merged_metrics[mode]["loss_action_bc"] - correct_loss) / correct_loss
        for mode in modes
    }
    # Modulo sharding places every task on every shard, so recompute task
    # aggregates across shard fragments rather than accepting duplicates.
    task_values: dict[str, dict[str, Any]] = defaultdict(dict)
    task_fragments: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for shard in shards:
        for task, values in shard["tasks"].items():
            for mode, metric in values.items():
                task_fragments[task][mode].append(metric)
    for task, task_modes in task_fragments.items():
        for mode, fragments in task_modes.items():
            count = sum(int(value["sample_count"]) for value in fragments)
            valid_count = sum(int(value["valid_action_count"]) for value in fragments)
            dimensions = (
                sum(
                    torch.tensor(value["mse_per_dimension"])
                    * int(value["valid_action_count"])
                    for value in fragments
                )
                / valid_count
            )
            task_values[task][mode] = {
                "sample_count": count,
                "valid_action_count": valid_count,
                "loss_action_bc": sum(
                    value["loss_action_bc"] * value["sample_count"]
                    for value in fragments
                )
                / count,
                "mse_per_dimension": dimensions.tolist(),
                "mse_pose": float(dimensions[:-1].mean().item()),
                "mse_gripper": float(dimensions[-1].item()),
                "velocity_relative_delta": sum(
                    value["velocity_relative_delta"] * value["sample_count"]
                    for value in fragments
                )
                / count,
                "action_relative_delta": sum(
                    value["action_relative_delta"] * value["sample_count"]
                    for value in fragments
                )
                / count,
            }
    if len(task_values) != 40 or any(
        value["correct"]["sample_count"] != 16 * len(first["draw_seeds"])
        for value in task_values.values()
    ):
        raise ValueError("Merged audit task coverage changed from 40 x 16 anchors.")
    for values in task_values.values():
        correct_task_loss = values["correct"]["loss_action_bc"]
        for mode in modes:
            values[mode]["relative_loss_gap_vs_correct"] = (
                values[mode]["loss_action_bc"] - correct_task_loss
            ) / correct_task_loss
    positive_task_fraction = sum(
        task_values[task]["task_paired"]["relative_loss_gap_vs_correct"] > 0.0
        for task in task_values
    ) / len(task_values)

    merged_diagnostics: dict[str, Any] = {}
    for mode in modes:
        keys = sorted(
            set().union(
                *(set(payload.get(mode, {})) for payload in diagnostic_payloads)
            )
        )
        merged_diagnostics[mode] = {
            key: _quantiles(
                torch.cat(
                    [
                        payload[mode][key]
                        for payload in diagnostic_payloads
                        if key in payload[mode]
                    ]
                )
            )
            for key in keys
        }
    residual_keys = [
        key
        for key in merged_diagnostics["correct"]
        if key.endswith("/effective_residual_over_hidden")
    ]
    residual_values = torch.cat(
        [
            payload["correct"][key]
            for payload in diagnostic_payloads
            for key in residual_keys
            if key in payload["correct"]
        ]
    )
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    result = {
        "schema": f"{P1_DINO_CAUSAL_AUDIT_SCHEMA}-merged",
        "status": "PASS",
        "checkpoint": first["checkpoint"],
        "ledger": first["ledger"],
        "shard_count": len(shards),
        "anchor_count": sum(value["anchor_count"] for value in shards),
        "draw_seeds": first["draw_seeds"],
        "modes": modes,
        "metrics": merged_metrics,
        "relative_loss_gaps": relative_gaps,
        "positive_task_fraction": positive_task_fraction,
        "residual_hidden_p95": float(torch.quantile(residual_values.float(), 0.95)),
        "residual_hidden_median": float(torch.quantile(residual_values.float(), 0.50)),
        "tasks": dict(task_values),
        "paired_records": paired_records,
        "diagnostics": merged_diagnostics,
        "finite": True,
        "shards": [
            {
                "shard_index": value["shard_index"],
                "path": str(Path(path).expanduser().resolve()),
            }
            for value, path in zip(shards, shard_directories, strict=True)
        ],
    }
    _atomic_json(destination / "audit_merged.json", result)
    return result
