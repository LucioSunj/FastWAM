"""Four-GPU full-data trainer for P1 DINO semantic-memory action BC."""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import re
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from fastwam.models.wan22.visual_backbone import VisualBackboneAssetSpec
from fastwam.models.wan22.visual_contracts import contract_sha256
from fastwam.models.wan22.visual_sidecar import (
    DinoContributionDiagnosticsCollector,
)
from fastwam.p1_dino_bc import P1_LORA_PARAMETER_FAMILY, build_p1_optimizer
from fastwam.p1_dino_bc_checkpoint import (
    inspect_p1_dino_bc_checkpoint,
    save_p1_dino_bc_checkpoint,
)
from fastwam.p1_dino_bc_full_checkpoint import (
    inspect_p1_dino_bc_full_checkpoint,
    inspect_p1_dino_bc_full_checkpoint_v2,
    load_p1_dino_bc_full_checkpoint,
    load_p1_dino_bc_full_checkpoint_v2,
    save_p1_dino_bc_full_checkpoint,
    save_p1_dino_bc_full_checkpoint_v2,
)
from fastwam.p1_dino_bc_runner import audit_p1_assets, build_real_p1_policy
from fastwam.p1_dino_bc_runner import (
    is_visual_v2_config,
    visual_input_contract_sha256,
)
from fastwam.p1_dino_contribution_v2 import (
    DINO_CONTRIBUTION_V2_PROFILE,
    CausalCheckpointSelector,
    CausalSelectionThresholds,
    DependencyWarmupController,
    NegativeModeCycle,
    TaskPairedDistributedBatchSampler,
    extract_dataset_window_identities,
    load_frozen_causal_ledger,
)
from fastwam.p1_visual_bc_checkpoint import (
    inspect_p1_visual_bc_checkpoint,
    save_p1_visual_bc_checkpoint,
)
from fastwam.p1_visual_bc_full_checkpoint import (
    inspect_p1_visual_bc_full_checkpoint,
    load_p1_visual_bc_full_checkpoint,
    save_p1_visual_bc_full_checkpoint,
)
from fastwam.uncond_bc import cosine_warmup_multiplier, stateless_validation_flow_inputs
from fastwam.uncond_bc_checkpoint import restore_rng_state
from fastwam.uncond_bc_trainer import (
    _atomic_json,
    _atomic_text,
    _barrier,
    _broadcast_object,
    _build_datasets,
    _build_loaders,
    _canonical_config,
    _dataset_summary,
    _distributed_context,
    _gather_rng,
    _git_state,
    _set_seed,
    _trainer_state_payload,
    _validate,
    _write_resolved_config,
    _write_run_manifest,
    sha256_artifact,
)
from fastwam.utils import misc

P1_FULL_OUTPUT_MARKER = ".fastwam-p1-dino-bc-full-output-v1"
P1_DINO_V2_FULL_OUTPUT_MARKER = ".fastwam-p1-dino-contribution-v2-output-v2"
P1_VISUAL_FULL_OUTPUT_MARKER = ".fastwam-p1-visual-bc-full-output-v2"
LEGACY_JOINT_PROFILE = "legacy_joint_bc_v1"
DINO_CONTRIBUTION_PROFILE = "dino_contribution_v1"
_CHECKPOINT_NAME = re.compile(r"(?:step_[0-9]{6}|epoch_[0-9]{2}_step_[0-9]{6})[.]pt")


def _output_marker(cfg: DictConfig) -> str:
    if (
        str(cfg.training.get("dino_contribution_profile", LEGACY_JOINT_PROFILE))
        == DINO_CONTRIBUTION_V2_PROFILE
    ):
        return P1_DINO_V2_FULL_OUTPUT_MARKER
    return (
        P1_VISUAL_FULL_OUTPUT_MARKER
        if is_visual_v2_config(cfg)
        else P1_FULL_OUTPUT_MARKER
    )


def claim_p1_full_output(cfg: DictConfig) -> Path:
    """Claim a new full-training directory or validate an explicit resume."""

    output_dir = Path(str(cfg.runner.output_dir)).expanduser().resolve()
    if int(os.environ.get("RANK", "0")) != 0:
        return output_dir
    marker_name = _output_marker(cfg)
    marker = output_dir / marker_name
    if cfg.runner.get("resume"):
        if not output_dir.is_dir() or not marker.is_file():
            raise FileNotFoundError(
                "P1 full resume requires its existing owned output directory: "
                f"{output_dir}."
            )
        return output_dir
    if output_dir.exists():
        entries = {entry.name for entry in output_dir.iterdir()}
        if entries and entries != {marker_name}:
            raise FileExistsError(
                f"P1 full output directory is not empty: {output_dir}"
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    if not marker.is_file():
        _atomic_text(marker, f"{marker_name[1:]}\n")
    return output_dir


def record_p1_full_failure(
    cfg: DictConfig,
    error: BaseException,
    *,
    traceback_text: str,
) -> Path | None:
    """Record a full-training failure only in a trainer-owned directory."""

    if int(os.environ.get("RANK", "0")) != 0:
        return None
    output_dir = Path(str(cfg.runner.output_dir)).expanduser().resolve()
    if not (output_dir / _output_marker(cfg)).is_file():
        return None
    root = Path(__file__).resolve().parents[2]
    payload = {
        "schema": (
            "fastwam-p1-visual-bc-full-failure-v2"
            if is_visual_v2_config(cfg)
            else (
                "fastwam-p1-dino-contribution-full-failure-v2"
                if str(
                    cfg.training.get(
                        "dino_contribution_profile",
                        LEGACY_JOINT_PROFILE,
                    )
                )
                == DINO_CONTRIBUTION_V2_PROFILE
                else "fastwam-p1-dino-bc-full-failure-v1"
            )
        ),
        "status": "FAIL",
        "stage": str(cfg.runner.get("stage")),
        "command": list(sys.argv),
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
        "repositories": {
            "fastwam": _git_state(root),
            "outer": _git_state(root.parent),
        },
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback_text,
        "created_unix_seconds": time.time(),
    }
    target = output_dir / "failure_manifest.json"
    _atomic_json(target, payload)
    return target


def _validate_full_config(cfg: DictConfig, *, world_size: int) -> None:
    visual_v2 = is_visual_v2_config(cfg)
    stage = str(cfg.runner.stage)
    profile = str(cfg.training.get("dino_contribution_profile", LEGACY_JOINT_PROFILE))
    allowed_stages = {"canary", "benchmark", "formal"}
    if profile == DINO_CONTRIBUTION_V2_PROFILE:
        allowed_stages.add("pilot")
    if stage not in allowed_stages:
        raise ValueError(f"P1 full stage must be one of {sorted(allowed_stages)}.")
    allowed_world_sizes = (
        {1, 4} if profile == DINO_CONTRIBUTION_V2_PROFILE and stage == "canary" else {4}
    )
    if world_size not in allowed_world_sizes:
        raise ValueError(
            f"P1 full {profile} {stage} requires world size "
            f"{sorted(allowed_world_sizes)}."
        )
    if int(cfg.seed) != 42 or int(cfg.validation.seed) != 42:
        raise ValueError("P1 full train and validation seeds must remain 42.")
    if (
        int(cfg.data.train.split_seed) != 42
        or int(cfg.data.validation.split_seed) != 42
    ):
        raise ValueError("P1 full episode split seeds must remain 42.")
    if str(cfg.precision).lower() != "bf16" or bool(cfg.get("compile", False)):
        raise ValueError("P1 full training requires eager BF16 execution.")
    if not bool(cfg.training.deterministic_algorithms) or not bool(
        cfg.training.stateless_flow_inputs
    ):
        raise ValueError(
            "P1 full training requires deterministic stateless flow inputs."
        )

    rank = int(cfg.lora.rank)
    if rank not in {16, 32} or float(cfg.lora.alpha) != float(rank):
        raise ValueError("P1 full LoRA must use rank/alpha 16/16 or 32/32.")
    if float(cfg.lora.dropout) != 0.0 or list(cfg.lora.target_groups) != [
        "self_attention_qkvo",
        "cross_attention_qkvo",
        "ffn",
    ]:
        raise ValueError("P1 full LoRA targets or dropout changed.")
    if list(cfg.p1.camera_ids) != ["main", "wrist"] or list(
        cfg.p1.reader.layer_indices
    ) != [6, 12, 18, 22]:
        raise ValueError("P1 full camera or reader-layer contract changed.")
    if profile not in {
        LEGACY_JOINT_PROFILE,
        DINO_CONTRIBUTION_PROFILE,
        DINO_CONTRIBUTION_V2_PROFILE,
    }:
        raise ValueError("Unknown P1 full DINO-contribution training profile.")
    semantic_gate_floor = float(cfg.p1.reader.get("semantic_gate_floor", 0.0))
    semantic_gate_temperature = float(
        cfg.p1.reader.get("semantic_gate_temperature", 1.0)
    )
    if (
        float(cfg.p1.reader.temperature) != 0.07
        or float(cfg.p1.reader.residual_scale) != 1.0
        or str(cfg.p1.position_mode) != "native_contextual_only"
    ):
        raise ValueError("P1 full fixed reader configuration changed.")
    if profile == LEGACY_JOINT_PROFILE and (
        semantic_gate_floor != 0.0 or semantic_gate_temperature != 1.0
    ):
        raise ValueError("Legacy P1 full profile must retain its exact semantic gate.")
    if profile in {DINO_CONTRIBUTION_PROFILE, DINO_CONTRIBUTION_V2_PROFILE} and (
        semantic_gate_floor != 0.05 or semantic_gate_temperature != 1.25
    ):
        raise ValueError("DINO-contribution profile semantic-gate controls changed.")
    for name in ("query_projection", "output_projection"):
        projection = cfg.p1.reader[name]
        if str(projection.kind) != "full_linear" or projection.rank is not None:
            raise ValueError("P1 full reader projections must remain full linear.")
    camera_contract_node = (
        cfg.p1.visual_camera_input_contract
        if visual_v2
        else cfg.p1.camera_input_contract
    )
    camera_contract = OmegaConf.to_container(
        camera_contract_node,
        resolve=True,
    )
    if contract_sha256(camera_contract) != visual_input_contract_sha256(cfg):
        raise ValueError("P1 full camera input contract hash changed.")
    if visual_v2:
        asset_payload = OmegaConf.to_container(
            cfg.p1.visual_backbone,
            resolve=True,
        )
        if not isinstance(asset_payload, dict):
            raise TypeError("V2 visual backbone config must be a mapping.")
        asset = VisualBackboneAssetSpec.from_mapping(asset_payload)
        if int(camera_contract["target_input_size"]) != asset.input_size:
            raise ValueError("V2 camera and visual-backbone input sizes differ.")
        if (
            str(camera_contract["source_stage"])
            != "raw_oriented_rgb_before_fastwam_resize"
        ):
            raise ValueError("V2 visual cameras must be constructed from raw RGB.")
    else:
        camera_tolerance = float(camera_contract["source_range_tolerance"])
        if (
            float(cfg.data.train.p1_camera_range_tolerance) != camera_tolerance
            or float(cfg.data.validation.p1_camera_range_tolerance) != camera_tolerance
        ):
            raise ValueError("P1 full dataset camera-range tolerance changed.")
    if not bool(cfg.p1.fastwam_runtime.skip_download):
        raise ValueError("P1 full training must disable all model downloads.")

    if (
        int(cfg.data.expected_train_episodes) != 1539
        or int(cfg.data.expected_validation_episodes) != 173
        or int(cfg.data.expected_source_episodes) != 1712
        or int(cfg.data.expected_source_transitions) != 277713
    ):
        raise ValueError("P1 full dataset-count contract changed.")
    if visual_v2:
        for split in (cfg.data.train, cfg.data.validation):
            if not bool(split.return_visual_camera_uint8):
                raise ValueError("V2 train/dev datasets must return visual cameras.")
            if list(split.visual_camera_ids) != ["main", "wrist"]:
                raise ValueError("V2 dataset visual camera order changed.")
            if int(split.visual_camera_input_size) != asset.input_size:
                raise ValueError("V2 dataset/backbone input sizes differ.")
            if int(split.processor.visual_camera_input_size) != asset.input_size:
                raise ValueError("V2 processor/backbone input sizes differ.")
            if list(split.processor.visual_camera_ids) != ["main", "wrist"]:
                raise ValueError("V2 processor visual camera order changed.")
    else:
        if not bool(cfg.data.train.return_p1_camera_uint8) or not bool(
            cfg.data.validation.return_p1_camera_uint8
        ):
            raise ValueError(
                "P1 full train/dev datasets must return DINO camera tensors."
            )
        if list(cfg.data.train.p1_camera_ids) != ["main", "wrist"] or list(
            cfg.data.validation.p1_camera_ids
        ) != ["main", "wrist"]:
            raise ValueError("P1 full dataset camera order changed.")

    expected_accumulation = {1: 32, 2: 16, 4: 8, 8: 4, 16: 2, 32: 1}
    microbatch = int(cfg.training.microbatch_size)
    if profile == DINO_CONTRIBUTION_V2_PROFILE:
        expected_global_batch = 32 if world_size == 1 else 128
        if (
            microbatch != 32
            or int(cfg.training.gradient_accumulation_steps) != 1
            or int(cfg.training.global_batch_size) != expected_global_batch
        ):
            raise ValueError(
                "DINO contribution v2 fixes microbatch 32, accumulation 1, "
                f"and global batch {expected_global_batch}."
            )
    elif microbatch not in expected_accumulation:
        raise ValueError("P1 full microbatch must be 1, 2, 4, 8, 16, or 32 per GPU.")
    elif (
        int(cfg.training.gradient_accumulation_steps)
        != expected_accumulation[microbatch]
    ):
        raise ValueError("P1 full accumulation does not preserve global batch 128.")
    if profile != DINO_CONTRIBUTION_V2_PROFILE and (
        microbatch * world_size * int(cfg.training.gradient_accumulation_steps)
        != int(cfg.training.global_batch_size)
        or int(cfg.training.global_batch_size) != 128
    ):
        raise ValueError("P1 full effective global batch must be exactly 128.")
    expected_epoch_schedule = (
        (1, 1, 1)
        if profile == DINO_CONTRIBUTION_V2_PROFILE and stage == "pilot"
        else (10, 3, 2)
    )
    if (
        (
            int(cfg.training.max_epochs),
            int(cfg.training.minimum_epochs),
            int(cfg.training.early_stopping_patience),
        )
        != expected_epoch_schedule
        or list(cfg.training.early_checkpoint_steps) != [10]
        or int(cfg.training.save_every_steps) != 100
        or int(cfg.training.checkpoint_keep_last) != 1
    ):
        raise ValueError("P1 full epoch/checkpoint schedule changed.")
    if tuple(float(value) for value in cfg.optimizer.betas) != (0.9, 0.95):
        raise ValueError("P1 full AdamW betas changed.")
    expected_reader_lr = 1e-4 if profile == LEGACY_JOINT_PROFILE else 2e-4
    if (
        float(cfg.optimizer.lora_learning_rate) != 3e-4
        or float(cfg.optimizer.reader_learning_rate) != expected_reader_lr
        or float(cfg.optimizer.eps) != 1e-8
        or float(cfg.optimizer.weight_decay) != 0.01
        or float(cfg.optimizer.gradient_clip) != 1.0
        or float(cfg.optimizer.warmup_fraction) != 0.05
        or float(cfg.optimizer.minimum_lr_ratio) != 0.01
    ):
        raise ValueError("P1 full optimizer schedule changed.")
    reader_only_warmup_steps = int(cfg.training.get("reader_only_warmup_steps", 0))
    dependency = cfg.training.get("memory_dependency")
    dependency_enabled = bool(
        dependency is not None and dependency.get("enabled", False)
    )
    if profile == LEGACY_JOINT_PROFILE:
        if reader_only_warmup_steps != 0 or dependency_enabled:
            raise ValueError(
                "Legacy P1 full profile cannot enable contribution training."
            )
    elif profile == DINO_CONTRIBUTION_PROFILE:
        if reader_only_warmup_steps != 256 or not dependency_enabled:
            raise ValueError("DINO-contribution reader-only warmup contract changed.")
        if (
            str(dependency.negative_mode) != "shuffled"
            or float(dependency.weight) != 0.1
            or float(dependency.relative_margin) != 0.05
            or int(dependency.every_n_steps) != 4
        ):
            raise ValueError("DINO-contribution counterfactual objective changed.")
        if microbatch < 2:
            raise ValueError(
                "Shuffled-memory contribution training requires microbatch >= 2."
            )
    else:
        if reader_only_warmup_steps != 0 or not dependency_enabled:
            raise ValueError("DINO contribution v2 dynamic warm-up contract changed.")
        warmup = cfg.training.get("reader_warmup")
        if warmup is None or (
            int(warmup.min_steps) != 256
            or int(warmup.max_steps) != 1024
            or int(warmup.window_active_updates) != 128
            or float(warmup.dependency_gap_min) != 0.10
            or float(warmup.pass_rate_min) != 0.70
            or int(warmup.consecutive_pass_windows) != 2
            or int(cfg.training.lora_gradient_ramp_steps) != 512
        ):
            raise ValueError("DINO contribution v2 warm-up/ramp controls changed.")
        dependency_weight = float(dependency.weight)
        if (
            dependency_weight not in {0.25, 0.50}
            or float(dependency.relative_margin) != 0.10
            or int(dependency.warmup_every_n_steps) != 1
            or int(dependency.joint_every_n_steps) != 2
            or list(dependency.negative_cycle)
            != ["task_paired", "task_paired", "task_paired", "off"]
        ):
            raise ValueError("DINO contribution v2 dependency objective changed.")
        gripper_multiplier = float(cfg.training.gripper_loss_multiplier)
        if gripper_multiplier not in {1.0, 1.5}:
            raise ValueError(
                "DINO contribution v2 gripper multiplier must be 1 or 1.5."
            )
        selection = cfg.training.get("checkpoint_selection")
        if selection is None or (
            float(selection.reference_validation_loss) != 0.0413114168
            or float(selection.reference_pose_mse) != 0.0330594443
            or float(selection.reference_gripper_mse) != 0.0853829011
            or float(selection.validation_loss_ratio_max) != 1.005
            or float(selection.pose_mse_ratio_max) != 1.01
            or float(selection.gripper_mse_ratio_max) != 1.01
            or float(selection.hard_negative_gap_min) != 0.10
            or float(selection.off_gap_min) != 0.10
            or float(selection.positive_task_fraction_min) != 0.70
            or float(selection.residual_hidden_p95_max) != 0.20
        ):
            raise ValueError("DINO contribution v2 checkpoint selector changed.")
        audit = cfg.get("causal_audit")
        if (
            audit is None
            or [int(value) for value in audit.draw_seeds] != [42]
            or list(audit.modes) != ["correct", "off", "task_paired"]
            or int(audit.microbatch_size) != 16
        ):
            raise ValueError("DINO contribution v2 causal-audit config changed.")
        if int(cfg.data.num_workers) != 3 or int(cfg.data.prefetch_factor) != 1:
            raise ValueError("DINO contribution v2 fixes workers=3 and prefetch=1.")
        if str(cfg.training.get("gradient_sync")) != "deterministic_rank_order":
            raise ValueError(
                "DINO contribution v2 requires deterministic rank-order gradient sync."
            )
        if int(cfg.training.get("gradient_sync_chunk_mb", -1)) != 16:
            raise ValueError("DINO contribution v2 fixes 16 MiB sync chunks.")
    if not 0 <= int(cfg.data.num_workers) <= 4:
        raise ValueError("P1 full DataLoader workers must lie in [0,4] per rank.")
    if not 1 <= int(cfg.data.prefetch_factor) <= 2:
        raise ValueError("P1 full prefetch factor must lie in [1,2].")
    for split in (cfg.data.train, cfg.data.validation):
        if not bool(split.image_current_frame_only):
            raise ValueError("P1 full datasets must decode only the current frame.")
        if int(split.processor.image_num_obs_steps) != 1:
            raise ValueError("P1 full processors must receive one image frame.")
    if stage == "canary":
        if (
            int(cfg.training.max_steps or -1) != 2
            or int(cfg.runner.stop_after_steps or -1) != 2
        ):
            raise ValueError("P1 full canary must execute exactly two updates.")
    elif stage == "benchmark":
        benchmark_stop = int(cfg.runner.stop_after_steps or -1)
        valid_benchmark_stop = (
            0 < benchmark_stop <= int(cfg.training.benchmark_steps)
            if profile == DINO_CONTRIBUTION_V2_PROFILE
            else benchmark_stop == int(cfg.training.benchmark_steps)
        )
        if cfg.training.get("max_steps") is not None or not valid_benchmark_stop:
            raise ValueError(
                "P1 full benchmark must use the formal schedule and stop at "
                "or before benchmark_steps."
            )
    elif (
        cfg.training.get("max_steps") is not None
        or cfg.runner.get("stop_after_steps") is not None
    ):
        raise ValueError("P1 full formal training cannot use a step truncation.")


def _dataset_metadata_hashes(paths: list[str]) -> dict[str, str]:
    result = {}
    for value in paths:
        meta = Path(value).expanduser().resolve() / "meta"
        result[str(meta)] = sha256_artifact(meta)
    return result


def _build_contract_and_provenance(
    cfg: DictConfig,
    *,
    resolved: Mapping[str, Any],
    launch_hash: str,
    contract_hash: str,
    assets: Mapping[str, Any],
    policy,
    parent_load: Mapping[str, Any],
    rank: int,
    world_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_paths = [str(path) for path in cfg.provenance.dataset_paths]
    text_cache = str(cfg.provenance.text_cache_path)
    if rank == 0:
        dataset_hashes = _dataset_metadata_hashes(dataset_paths)
        text_cache_hash = sha256_artifact(text_cache)
    else:
        dataset_hashes = None
        text_cache_hash = None
    dataset_hashes = _broadcast_object(
        dataset_hashes,
        rank=rank,
        world_size=world_size,
    )
    text_cache_hash = _broadcast_object(
        text_cache_hash,
        rank=rank,
        world_size=world_size,
    )
    contract = {
        "resolved_config_sha256": contract_hash,
        "world_size": world_size,
        "parent_checkpoint_sha256": assets["parent_checkpoint_sha256"],
        "statistics_sha256": assets["statistics_sha256"],
        "memory_contract_sha256": policy.expected_memory_contract,
        "reader_contract_sha256": policy.visual_reader.reader_contract_sha256,
        "dataset_metadata_sha256": dataset_hashes,
        "text_cache_sha256": text_cache_hash,
        "lora": OmegaConf.to_container(cfg.lora, resolve=True),
        "reader": OmegaConf.to_container(cfg.p1.reader, resolve=True),
        "bc_policy": OmegaConf.to_container(cfg.bc_policy, resolve=True),
    }
    if is_visual_v2_config(cfg):
        contract["visual_backbone"] = dict(assets["visual_backbone"])
    else:
        contract.update(
            dinov3_weights_sha256=assets["dinov3_weights_sha256"],
            dinov3_source_revision=assets["dinov3_source_revision"],
        )
    root = Path(__file__).resolve().parents[2]
    provenance = {
        "schema": (
            "fastwam-p1-visual-bc-full-provenance-v2"
            if is_visual_v2_config(cfg)
            else "fastwam-p1-dino-bc-full-provenance-v1"
        ),
        "command": list(sys.argv),
        "resolved_config": dict(resolved),
        "launch_resolved_config_sha256": launch_hash,
        "training_contract_sha256": contract_hash,
        "repositories": {
            "fastwam": _git_state(root),
            "outer": _git_state(root.parent),
        },
        "assets": dict(assets),
        "parent_load": dict(parent_load),
        "dataset_metadata_sha256": dataset_hashes,
        "text_cache": {"path": text_cache, "sha256": text_cache_hash},
        "world_size": world_size,
        "precision": "bfloat16",
        "eager": True,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    return contract, provenance


def _trainable_snapshot(policy) -> dict[str, torch.Tensor]:
    allowed = {
        id(parameter)
        for family in policy.parameter_families().values()
        for parameter in family
    }
    return {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
        if id(parameter) in allowed
    }


def _family_norms(policy, before: Mapping[str, torch.Tensor] | None = None):
    result = {}
    named = dict(policy.named_parameters())
    for family, parameters in policy.parameter_families().items():
        ids = {id(parameter) for parameter in parameters}
        values = []
        for name, parameter in named.items():
            if id(parameter) not in ids:
                continue
            value = (
                parameter.grad if before is None else parameter.detach() - before[name]
            )
            if value is not None:
                values.append(value.detach().float().square().sum())
        result[family] = (
            torch.stack(values).sum().sqrt()
            if values
            else torch.zeros((), device=policy.device)
        )
    return result


class _DeterministicGradientSynchronizer:
    """Synchronize v2 gradients in a restart-stable rank order.

    DDP/NCCL is mathematically deterministic for a fixed reduction schedule,
    but its reducer may select a different bucket schedule for the first
    backward after a process restart. Contribution-v2 requires bitwise resume
    parity, so it gathers fixed-size chunks in rank order and performs the
    additions in that explicit order on every rank. The persistent flat buffer
    replaces, rather than supplements, DDP's gradient bucket.
    """

    def __init__(
        self,
        parameters: tuple[nn.Parameter, ...],
        *,
        world_size: int,
        chunk_bytes: int,
    ) -> None:
        if world_size < 1:
            raise ValueError("Gradient synchronizer world_size must be positive.")
        if chunk_bytes < 1:
            raise ValueError("Gradient synchronizer chunk_bytes must be positive.")
        self.parameters = parameters
        self.world_size = int(world_size)
        grouped: dict[tuple[torch.device, torch.dtype], list[nn.Parameter]] = {}
        for parameter in parameters:
            grouped.setdefault((parameter.device, parameter.dtype), []).append(
                parameter
            )
        self.groups: list[dict[str, Any]] = []
        for (device, dtype), members in grouped.items():
            element_size = torch.empty((), dtype=dtype).element_size()
            chunk_elements = max(1, int(chunk_bytes) // element_size)
            total_elements = sum(parameter.numel() for parameter in members)
            self.groups.append(
                {
                    "parameters": tuple(members),
                    "flat": torch.empty(total_elements, device=device, dtype=dtype),
                    "gathered": torch.empty(
                        self.world_size * min(chunk_elements, total_elements),
                        device=device,
                        dtype=dtype,
                    ),
                    "chunk_elements": chunk_elements,
                }
            )

    @staticmethod
    def _copy_parameters_to_flat(group: Mapping[str, Any]) -> None:
        offset = 0
        flat = group["flat"]
        for parameter in group["parameters"]:
            count = parameter.numel()
            flat[offset : offset + count].copy_(parameter.detach().reshape(-1))
            offset += count

    @staticmethod
    def _copy_flat_to_parameters(group: Mapping[str, Any]) -> None:
        offset = 0
        flat = group["flat"]
        for parameter in group["parameters"]:
            count = parameter.numel()
            parameter.data.copy_(flat[offset : offset + count].view_as(parameter))
            offset += count

    def broadcast_parameters(self) -> None:
        """Match DDP's initial rank-zero trainable-parameter broadcast."""

        if self.world_size == 1:
            return
        for group in self.groups:
            self._copy_parameters_to_flat(group)
            torch.distributed.broadcast(group["flat"], src=0)
            self._copy_flat_to_parameters(group)

    def synchronize_gradients(self) -> None:
        """Average all gradients using an explicit rank-ordered sum."""

        if self.world_size == 1:
            return
        for group in self.groups:
            flat = group["flat"]
            offset = 0
            for parameter in group["parameters"]:
                if parameter.grad is None:
                    raise RuntimeError(
                        "Contribution-v2 requires every trainable parameter to "
                        "produce a gradient before deterministic synchronization."
                    )
                count = parameter.numel()
                flat[offset : offset + count].copy_(parameter.grad.reshape(-1))
                offset += count
            chunk_elements = int(group["chunk_elements"])
            for start in range(0, flat.numel(), chunk_elements):
                local = flat[start : start + chunk_elements]
                gathered = group["gathered"][: self.world_size * local.numel()]
                torch.distributed.all_gather_into_tensor(gathered, local)
                by_rank = gathered.view(self.world_size, local.numel())
                local.copy_(by_rank[0])
                for source_rank in range(1, self.world_size):
                    local.add_(by_rank[source_rank])
                local.div_(self.world_size)
            offset = 0
            for parameter in group["parameters"]:
                count = parameter.numel()
                parameter.grad.copy_(flat[offset : offset + count].view_as(parameter))
                offset += count


def _memory_dependency_settings(cfg: DictConfig) -> dict[str, Any]:
    """Resolve the optional action-space DINO-dependency objective."""

    payload = cfg.training.get("memory_dependency")
    if payload is None or not bool(payload.get("enabled", False)):
        return {
            "enabled": False,
            "negative_mode": "shuffled",
            "weight": 0.0,
            "relative_margin": 0.0,
            "every_n_steps": 1,
        }
    if payload.get("negative_cycle") is not None:
        return {
            "enabled": True,
            "weight": float(payload.weight),
            "relative_margin": float(payload.relative_margin),
            "warmup_every_n_steps": int(payload.warmup_every_n_steps),
            "joint_every_n_steps": int(payload.joint_every_n_steps),
            "negative_cycle": [str(value) for value in payload.negative_cycle],
        }
    return {
        "enabled": True,
        "negative_mode": str(payload.negative_mode),
        "weight": float(payload.weight),
        "relative_margin": float(payload.relative_margin),
        "every_n_steps": int(payload.every_n_steps),
    }


def _memory_dependency_active(
    *,
    global_step: int,
    settings: Mapping[str, Any],
) -> bool:
    """Select deterministic optimizer updates for the extra memory forward."""

    every_n_steps = int(settings["every_n_steps"])
    if every_n_steps < 1:
        raise ValueError("Memory-dependency cadence must be a positive integer.")
    return bool(settings["enabled"]) and int(global_step) % every_n_steps == 0


def _suppress_parameter_family_gradients(policy, family: str) -> int:
    """Prevent one optimizer family from updating without changing DDP graphs."""

    families = policy.parameter_families()
    if family not in families:
        raise ValueError(f"Unknown P1 parameter family {family!r}.")
    parameters = families[family]
    for parameter in parameters:
        parameter.grad = None
    return len(parameters)


def _scale_parameter_family_gradients(
    policy,
    family: str,
    multiplier: float,
) -> int:
    """Scale one optimizer family's gradients without touching parameters."""

    value = float(multiplier)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("Gradient multiplier must be finite and lie in (0,1].")
    families = policy.parameter_families()
    if family not in families:
        raise ValueError(f"Unknown P1 parameter family {family!r}.")
    count = 0
    for parameter in families[family]:
        if parameter.grad is not None:
            parameter.grad.mul_(value)
            count += 1
    return count


def _build_v2_train_loader(
    cfg: DictConfig,
    *,
    train_dataset,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, TaskPairedDistributedBatchSampler]:
    rows = extract_dataset_window_identities(
        train_dataset,
        include_normalized_proprio=False,
    )
    sampler = TaskPairedDistributedBatchSampler(
        task_keys=[row.task_key for row in rows],
        episode_indices=[row.episode_index for row in rows],
        batch_size=int(cfg.training.microbatch_size),
        rank=rank,
        world_size=world_size,
        seed=int(cfg.seed),
    )
    loader_kwargs: dict[str, Any] = {
        "batch_sampler": sampler,
        "num_workers": int(cfg.data.num_workers),
        "pin_memory": True,
        "persistent_workers": bool(cfg.data.num_workers > 0),
        "generator": torch.Generator().manual_seed(int(cfg.seed)),
    }
    if int(cfg.data.num_workers) > 0:
        loader_kwargs["prefetch_factor"] = int(cfg.data.prefetch_factor)
    return DataLoader(train_dataset, **loader_kwargs), sampler


def _v2_warmup_controller(cfg: DictConfig) -> DependencyWarmupController:
    warmup = cfg.training.reader_warmup
    return DependencyWarmupController(
        min_steps=int(warmup.min_steps),
        max_steps=int(warmup.max_steps),
        window_active_updates=int(warmup.window_active_updates),
        dependency_gap_min=float(warmup.dependency_gap_min),
        pass_rate_min=float(warmup.pass_rate_min),
        consecutive_pass_windows_required=int(warmup.consecutive_pass_windows),
        lora_gradient_ramp_steps=int(cfg.training.lora_gradient_ramp_steps),
    )


def _v2_selector(cfg: DictConfig) -> CausalCheckpointSelector:
    selection = cfg.training.checkpoint_selection
    return CausalCheckpointSelector(
        CausalSelectionThresholds(
            validation_loss_max=float(selection.reference_validation_loss)
            * float(selection.validation_loss_ratio_max),
            pose_mse_max=float(selection.reference_pose_mse)
            * float(selection.pose_mse_ratio_max),
            gripper_mse_max=float(selection.reference_gripper_mse)
            * float(selection.gripper_mse_ratio_max),
            hard_negative_gap_min=float(selection.hard_negative_gap_min),
            off_gap_min=float(selection.off_gap_min),
            positive_task_fraction_min=float(selection.positive_task_fraction_min),
            residual_hidden_p95_max=float(selection.residual_hidden_p95_max),
        )
    )


def _frozen_versions(policy) -> dict[str, int]:
    trainable = {
        id(parameter)
        for family in policy.parameter_families().values()
        for parameter in family
    }
    return {
        name: parameter._version
        for name, parameter in policy.named_parameters()
        if id(parameter) not in trainable
    }


def _assert_frozen_versions(policy, expected: Mapping[str, int]) -> None:
    observed = _frozen_versions(policy)
    changed = sorted(
        name for name, version in expected.items() if observed.get(name) != version
    )
    if changed:
        raise RuntimeError(f"P1 full frozen parameters changed: {changed[:16]}.")


def _audit_rgb_sha256(sample: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(sample["p1_camera_pixels"].contiguous().numpy().tobytes())
    digest.update(
        sample["p1_camera_valid_mask"].to(torch.uint8).contiguous().numpy().tobytes()
    )
    return digest.hexdigest()


@torch.no_grad()
def _validate_v2_causal_ledger(
    policy,
    train_dataset,
    validation_dataset,
    ledger: Mapping[str, Any],
    *,
    cfg: DictConfig,
    rank: int,
    world_size: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run the seed-42 correct/off/task-paired eligibility audit."""

    from torch.utils.data._utils.collate import default_collate

    entries = [
        value
        for index, value in enumerate(ledger["anchors"])
        if index % world_size == rank
    ]
    by_task: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for entry in entries:
        key = (int(entry["suite_index"]), int(entry["task_index"]))
        by_task.setdefault(key, []).append(entry)
    # correct sum, off sum, negative sum, samples, then 40 x
    # (correct sum, negative sum, samples).
    accum = torch.zeros(4 + 40 * 3, dtype=torch.float64, device=device)
    residual_parts: list[torch.Tensor] = []
    microbatch = int(cfg.causal_audit.microbatch_size)
    was_training = policy.training
    policy.eval()
    for task_key in sorted(by_task):
        task_id = task_key[0] * 10 + task_key[1]
        if not 0 <= task_id < 40:
            raise ValueError(f"Causal ledger task key is invalid: {task_key}.")
        task_entries = by_task[task_key]
        for offset in range(0, len(task_entries), microbatch):
            selected = task_entries[offset : offset + microbatch]
            anchors = [
                validation_dataset[int(value["anchor_dataset_index"])]
                for value in selected
            ]
            negatives = [
                (
                    validation_dataset
                    if value["negative_split"] == "validation"
                    else train_dataset
                )[int(value["negative_dataset_index"])]
                for value in selected
            ]
            for value, anchor, negative in zip(
                selected,
                anchors,
                negatives,
                strict=True,
            ):
                if _audit_rgb_sha256(anchor) != value["anchor_rgb_sha256"]:
                    raise ValueError(
                        f"Causal anchor RGB drifted: {value['anchor_id']}."
                    )
                if _audit_rgb_sha256(negative) != value["negative_rgb_sha256"]:
                    raise ValueError(
                        f"Causal negative RGB drifted: {value['negative_id']}."
                    )
            batch = default_collate(anchors)
            negative_batch = default_collate(negatives)
            condition = policy.prepare_action_condition(batch, include_visual=True)
            negative_memory = policy.visual_encoder.prepare_memory(
                "uncond",
                policy._camera_batch(negative_batch),
            )
            if negative_memory is None:
                raise RuntimeError("Causal ledger negative memory is missing.")
            timestep, noise = stateless_validation_flow_inputs(
                sample_identities=list(batch["sample_identity"]),
                action_shape=tuple(batch["action"].shape),
                scheduler=policy.actor.train_action_scheduler,
                seed=42,
                device=device,
                dtype=policy.dtype,
            )
            collector = DinoContributionDiagnosticsCollector()
            with policy.visual_reader.capture_diagnostics(collector):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    correct = policy.loss_from_prepared_condition(
                        batch,
                        condition=condition,
                        timestep=timestep,
                        noise=noise,
                        return_prediction=True,
                    )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                off = policy.loss_from_prepared_condition(
                    batch,
                    condition=condition,
                    timestep=timestep,
                    noise=noise,
                    memory_mode="off",
                    return_prediction=True,
                )
                paired = policy.loss_from_prepared_condition(
                    batch,
                    condition=condition,
                    timestep=timestep,
                    noise=noise,
                    memory_mode="correct",
                    memory_override=negative_memory,
                    return_prediction=True,
                )
            batch_size = len(selected)
            correct_sum = correct["loss_action_bc_per_sample"].double().sum()
            off_sum = off["loss_action_bc_per_sample"].double().sum()
            paired_sum = paired["loss_action_bc_per_sample"].double().sum()
            accum[0] += correct_sum
            accum[1] += off_sum
            accum[2] += paired_sum
            accum[3] += batch_size
            task_start = 4 + task_id * 3
            accum[task_start] += correct_sum
            accum[task_start + 1] += paired_sum
            accum[task_start + 2] += batch_size
            residual_parts.extend(
                record["effective_residual_over_hidden"].reshape(-1)
                for record in collector.records
            )
    if world_size > 1:
        torch.distributed.all_reduce(accum, op=torch.distributed.ReduceOp.SUM)
    local_residual = (
        torch.cat(residual_parts).float()
        if residual_parts
        else torch.empty(0, dtype=torch.float32)
    )
    if world_size == 1:
        gathered_residual = [local_residual]
    else:
        gathered_residual: list[Any] = [None] * world_size
        torch.distributed.all_gather_object(gathered_residual, local_residual)
    residual = torch.cat(gathered_residual)
    if residual.numel() == 0 or not bool(torch.isfinite(residual).all().item()):
        raise FloatingPointError("Causal ledger residual diagnostics are non-finite.")
    count = float(accum[3].item())
    if int(count) != 640:
        raise ValueError(
            f"Causal ledger evaluated {int(count)} rather than 640 anchors."
        )
    correct_loss = float((accum[0] / count).item())
    off_loss = float((accum[1] / count).item())
    paired_loss = float((accum[2] / count).item())
    positive = 0
    task_results = {}
    for task_id in range(40):
        start = 4 + task_id * 3
        task_count = float(accum[start + 2].item())
        if int(task_count) != 16:
            raise ValueError(f"Causal task {task_id} has {int(task_count)} anchors.")
        task_correct = float((accum[start] / task_count).item())
        task_paired = float((accum[start + 1] / task_count).item())
        relative_gap = (task_paired - task_correct) / task_correct
        positive += int(relative_gap > 0.0)
        task_results[str(task_id)] = {
            "correct_loss": task_correct,
            "task_paired_loss": task_paired,
            "task_paired_relative_gap": relative_gap,
        }
    if was_training:
        policy.train()
    return {
        "correct_loss": correct_loss,
        "off_loss": off_loss,
        "task_paired_loss": paired_loss,
        "off_relative_gap": (off_loss - correct_loss) / correct_loss,
        "task_paired_relative_gap": (paired_loss - correct_loss) / correct_loss,
        "positive_task_fraction": positive / 40.0,
        "residual_hidden_p95": float(torch.quantile(residual, 0.95).item()),
        "residual_hidden_median": float(torch.quantile(residual, 0.50).item()),
        "finite": True,
        "tasks": task_results,
        "ledger_sha256": ledger["ledger_file_sha256"],
    }


def _run_acceptance_passed(
    *,
    global_step: int,
    stop_after_steps: int,
    nonzero_update_count: int,
    last_checkpoint: Mapping[str, Any] | None,
    best_step: int | None,
    controlled_stop: bool,
    early_stopped: bool,
    eligibility_required: bool = True,
) -> bool:
    """Return whether a controlled or formal full-data run completed safely."""

    reached_terminal_step = global_step == stop_after_steps
    schedule_complete = reached_terminal_step or (not controlled_stop and early_stopped)
    return (
        schedule_complete
        and nonzero_update_count > 0
        and last_checkpoint is not None
        and (not eligibility_required or best_step is not None)
    )


def _prune_checkpoints(checkpoints_dir: Path, *, keep: Path) -> list[str]:
    known = sorted(
        path
        for path in checkpoints_dir.iterdir()
        if path.is_file() and _CHECKPOINT_NAME.fullmatch(path.name)
    )
    if keep not in known:
        raise ValueError("P1 full retention keep file is not recognized.")
    pruned = []
    for path in known:
        if path != keep:
            path.unlink()
            pruned.append(path.name)
    return pruned


def _save_training_checkpoint(
    path: Path,
    *,
    policy,
    optimizer,
    scheduler,
    scaler,
    global_step: int,
    epoch: int,
    sampler_offset: int,
    contract: Mapping[str, Any],
    provenance: Mapping[str, Any],
    trainer_state: Mapping[str, Any],
    assets: Mapping[str, Any],
    rank: int,
    world_size: int,
    v2_control_state: Mapping[str, Any] | None = None,
    paired_sampler: TaskPairedDistributedBatchSampler | None = None,
) -> dict[str, Any] | None:
    rng_by_rank = _gather_rng(world_size=world_size)
    sampler_by_rank = None
    if v2_control_state is not None:
        if paired_sampler is None:
            raise ValueError("P1 v2 checkpoint requires its paired sampler.")
        local_sampler = paired_sampler.state_dict()
        if world_size == 1:
            sampler_by_rank = [local_sampler]
        else:
            sampler_by_rank = [None] * world_size
            torch.distributed.all_gather_object(sampler_by_rank, local_sampler)
    if rank == 0:
        common = {
            "path": path,
            "adapter": policy.lora_adapter,
            "reader": policy.visual_reader,
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "grad_scaler": scaler,
            "global_step": global_step,
            "epoch": epoch,
            "sampler_offset": sampler_offset,
            "rng_by_rank": rng_by_rank,
            "parent_checkpoint_sha256": assets["parent_checkpoint_sha256"],
            "memory_contract_sha256": policy.expected_memory_contract,
            "contract": contract,
            "provenance": provenance,
            "trainer_state": trainer_state,
        }
        if v2_control_state is not None:
            if "visual_backbone" in assets:
                raise ValueError(
                    "DINO contribution v2 cannot use a V2 visual backbone."
                )
            v2_state = dict(v2_control_state)
            v2_state["task_paired_sampler_by_rank"] = sampler_by_rank
            save_p1_dino_bc_full_checkpoint_v2(
                **common,
                dinov3_weights_sha256=assets["dinov3_weights_sha256"],
                v2_state=v2_state,
            )
            report = inspect_p1_dino_bc_full_checkpoint_v2(path)
        elif "visual_backbone" in assets:
            save_p1_visual_bc_full_checkpoint(
                **common,
                visual_backbone=assets["visual_backbone"],
            )
            report = inspect_p1_visual_bc_full_checkpoint(path)
        else:
            save_p1_dino_bc_full_checkpoint(
                **common,
                dinov3_weights_sha256=assets["dinov3_weights_sha256"],
            )
            report = inspect_p1_dino_bc_full_checkpoint(path)
        report["checkpoint_retention"] = {
            "keep_last": 1,
            "kept": path.name,
            "pruned": _prune_checkpoints(path.parent, keep=path),
        }
    else:
        report = None
    return _broadcast_object(report, rank=rank, world_size=world_size)


def run_p1_dino_bc_full(cfg: DictConfig) -> dict[str, Any]:
    """Run a two-update canary or complete four-GPU P1 BC training."""

    rank, world_size, local_rank, device = _distributed_context()
    _validate_full_config(cfg, world_size=world_size)
    contribution_profile = str(
        cfg.training.get("dino_contribution_profile", LEGACY_JOINT_PROFILE)
    )
    contribution_v2 = contribution_profile == DINO_CONTRIBUTION_V2_PROFILE
    _set_seed(
        int(cfg.seed),
        rank=rank,
        deterministic=bool(cfg.training.deterministic_algorithms),
    )
    output_dir = Path(str(cfg.runner.output_dir)).expanduser().resolve()
    if rank == 0:
        claim_p1_full_output(cfg)
    _barrier(world_size)
    if not (output_dir / _output_marker(cfg)).is_file():
        raise FileNotFoundError("P1 full output marker was not synchronized.")
    misc.register_work_dir(output_dir)

    resolved, launch_hash, contract_hash = _canonical_config(cfg)
    if rank == 0:
        _write_resolved_config(
            output_dir,
            launch_hash=launch_hash,
            value=OmegaConf.to_yaml(cfg, resolve=True),
        )
        assets = audit_p1_assets(cfg)
    else:
        assets = None
    assets = _broadcast_object(assets, rank=rank, world_size=world_size)

    causal_ledger = None
    if contribution_v2 and str(cfg.runner.stage) in {"pilot", "formal"}:
        causal_ledger = load_frozen_causal_ledger(str(cfg.causal_audit.ledger))

    policy, parent_load = build_real_p1_policy(cfg, device=device)
    policy.train()
    contract, provenance = _build_contract_and_provenance(
        cfg,
        resolved=resolved,
        launch_hash=launch_hash,
        contract_hash=contract_hash,
        assets=assets,
        policy=policy,
        parent_load=parent_load,
        rank=rank,
        world_size=world_size,
    )
    train_dataset, validation_dataset = _build_datasets(cfg)
    train_summary = _dataset_summary(train_dataset)
    validation_summary = _dataset_summary(validation_dataset)
    if train_summary["episodes"] != int(
        cfg.data.expected_train_episodes
    ) or validation_summary["episodes"] != int(cfg.data.expected_validation_episodes):
        raise ValueError(
            "P1 full episode split mismatch: "
            f"train={train_summary}, validation={validation_summary}."
        )
    if train_summary["episodes"] + validation_summary["episodes"] != int(
        cfg.data.expected_source_episodes
    ) or train_summary["windows"] + validation_summary["windows"] != int(
        cfg.data.expected_source_transitions
    ):
        raise ValueError("P1 full source corpus count changed.")
    train_loader, validation_loader, train_sampler = _build_loaders(
        cfg,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        rank=rank,
        world_size=world_size,
    )
    paired_sampler = None
    if contribution_v2:
        train_loader, paired_sampler = _build_v2_train_loader(
            cfg,
            train_dataset=train_dataset,
            rank=rank,
            world_size=world_size,
        )
        train_sampler = paired_sampler

    visual_encoder_calls = {"count": 0}

    def count_dino(_module, _inputs, _outputs) -> None:
        visual_encoder_calls["count"] += 1

    dino_hook = policy.visual_encoder.register_forward_hook(count_dino)
    optimizer = build_p1_optimizer(
        policy,
        lora_learning_rate=float(cfg.optimizer.lora_learning_rate),
        reader_learning_rate=float(cfg.optimizer.reader_learning_rate),
        train_lora=True,
        train_reader=True,
        betas=tuple(float(value) for value in cfg.optimizer.betas),
        eps=float(cfg.optimizer.eps),
        weight_decay=float(cfg.optimizer.weight_decay),
    )
    parameters = tuple(
        parameter for group in optimizer.param_groups for parameter in group["params"]
    )
    gradient_synchronizer = None
    if contribution_v2:
        gradient_synchronizer = _DeterministicGradientSynchronizer(
            parameters,
            world_size=world_size,
            chunk_bytes=int(cfg.training.gradient_sync_chunk_mb) * 1024 * 1024,
        )
        gradient_synchronizer.broadcast_parameters()
        model: nn.Module = policy
    else:
        model = DistributedDataParallel(
            policy,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    updates_per_epoch = len(train_loader) // int(
        cfg.training.gradient_accumulation_steps
    )
    total_steps = (
        int(cfg.training.max_steps)
        if cfg.training.get("max_steps") is not None
        else updates_per_epoch * int(cfg.training.max_epochs)
    )
    stop_after_steps = (
        int(cfg.runner.stop_after_steps)
        if cfg.runner.get("stop_after_steps") is not None
        else total_steps
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_warmup_multiplier(
            step,
            total_steps=total_steps,
            warmup_fraction=float(cfg.optimizer.warmup_fraction),
            minimum_ratio=float(cfg.optimizer.minimum_lr_ratio),
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    frozen_versions = _frozen_versions(policy)
    reader_only_warmup_steps = int(cfg.training.get("reader_only_warmup_steps", 0))
    memory_dependency = _memory_dependency_settings(cfg)
    warmup_controller = _v2_warmup_controller(cfg) if contribution_v2 else None
    negative_cycle = (
        NegativeModeCycle(tuple(memory_dependency["negative_cycle"]))
        if contribution_v2
        else None
    )
    causal_selector = _v2_selector(cfg) if contribution_v2 else None

    global_step = 0
    start_epoch = 0
    sampler_offset = 0
    best_validation = math.inf
    best_step = None
    epochs_without_improvement = 0
    nonzero_update_count = 0
    resume = cfg.runner.get("resume")
    if resume:
        common = {
            "path": str(resume),
            "adapter": policy.lora_adapter,
            "reader": policy.visual_reader,
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "grad_scaler": scaler,
            "expected_parent_checkpoint_sha256": assets["parent_checkpoint_sha256"],
            "expected_contract": contract,
        }
        if contribution_v2:
            payload = load_p1_dino_bc_full_checkpoint_v2(
                **common,
                expected_dinov3_weights_sha256=assets["dinov3_weights_sha256"],
            )
        elif is_visual_v2_config(cfg):
            payload = load_p1_visual_bc_full_checkpoint(
                **common,
                expected_visual_backbone=assets["visual_backbone"],
            )
        else:
            payload = load_p1_dino_bc_full_checkpoint(
                **common,
                expected_dinov3_weights_sha256=assets["dinov3_weights_sha256"],
            )
        if len(payload["rng_by_rank"]) != world_size:
            raise ValueError("P1 full resume RNG rank count changed.")
        restore_rng_state(payload["rng_by_rank"][rank])
        global_step = int(payload["global_step"])
        start_epoch = int(payload["epoch"])
        sampler_offset = int(payload["sampler_offset"])
        state = payload["trainer_state"]
        best = state["best_validation_loss_action_bc"]
        best_validation = math.inf if best is None else float(best)
        best_step = state["best_step"]
        epochs_without_improvement = int(state["epochs_without_improvement"])
        nonzero_update_count = int(state["nonzero_update_count"])
        if contribution_v2:
            control = payload["v2_state"]
            warmup_controller = DependencyWarmupController.from_state_dict(
                control["warmup"]
            )
            negative_cycle = NegativeModeCycle.from_state_dict(
                control["negative_cycle"]
            )
            causal_selector = CausalCheckpointSelector.from_state_dict(
                control["causal_selector"]
            )
            paired_sampler.set_epoch(start_epoch)
            paired_sampler.validate_state_dict(
                control["task_paired_sampler_by_rank"][rank]
            )

    manifest: dict[str, Any] = {
        "schema": (
            "fastwam-p1-visual-bc-full-run-v2"
            if is_visual_v2_config(cfg)
            else (
                "fastwam-p1-dino-contribution-full-run-v2"
                if contribution_v2
                else "fastwam-p1-dino-bc-full-run-v1"
            )
        ),
        "stage": str(cfg.runner.stage),
        "status": "RUNNING",
        "command": list(sys.argv),
        "contract": contract,
        "provenance": provenance,
        "parent_load": parent_load,
        "train_dataset": train_summary,
        "validation_dataset": validation_summary,
        "world_size": world_size,
        "lora_rank": int(cfg.lora.rank),
        "lora_alpha": float(cfg.lora.alpha),
        "dino_contribution_profile": contribution_profile,
        "reader_only_warmup_steps": reader_only_warmup_steps,
        "memory_dependency": memory_dependency,
        "causal_ledger": (
            None
            if causal_ledger is None
            else {
                "path": str(Path(cfg.causal_audit.ledger).expanduser().resolve()),
                "sha256": causal_ledger["ledger_file_sha256"],
                "content_sha256": causal_ledger["content_sha256"],
            }
        ),
        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
        "contains_gate": False,
        "contains_critic": False,
        "contains_value_head": False,
    }
    if rank == 0:
        _atomic_json(output_dir / "run_manifest.json", manifest)

    checkpoints_dir = output_dir / "checkpoints"
    metrics_path = output_dir / "metrics.jsonl"
    last_checkpoint = None
    stop = False
    early_stopped = False
    no_go_warmup = False
    optimizer.zero_grad(set_to_none=True)
    started_at = time.time()
    try:
        for epoch in range(start_epoch, int(cfg.training.max_epochs)):
            train_sampler.set_epoch(epoch)
            train_loader.generator.manual_seed(int(cfg.seed) + epoch)
            usable_batches = (
                len(train_loader)
                // int(cfg.training.gradient_accumulation_steps)
                * int(cfg.training.gradient_accumulation_steps)
            )
            accumulation_index = 0
            identity_digest = hashlib.sha256()
            timestep_digest = hashlib.sha256()
            noise_digest = hashlib.sha256()
            for batch_index, batch in enumerate(train_loader):
                if batch_index < sampler_offset:
                    continue
                if batch_index >= usable_batches:
                    break
                accumulation_index += 1
                sync_update = (
                    accumulation_index % int(cfg.training.gradient_accumulation_steps)
                    == 0
                )
                sync_context = (
                    contextlib.nullcontext()
                    if sync_update or contribution_v2
                    else model.no_sync()
                )
                identities = [
                    f"train-epoch-{epoch}:{identity}"
                    for identity in batch["sample_identity"]
                ]
                timestep, noise = stateless_validation_flow_inputs(
                    sample_identities=identities,
                    action_shape=tuple(batch["action"].shape),
                    scheduler=policy.actor.train_action_scheduler,
                    seed=int(cfg.seed),
                    device=device,
                    dtype=policy.dtype,
                )
                identity_digest.update(
                    json.dumps(identities, separators=(",", ":")).encode()
                )
                timestep_digest.update(
                    timestep.detach()
                    .contiguous()
                    .view(torch.uint8)
                    .cpu()
                    .numpy()
                    .tobytes()
                )
                noise_digest.update(
                    noise.detach()
                    .contiguous()
                    .view(torch.uint8)
                    .cpu()
                    .numpy()
                    .tobytes()
                )
                dependency_mode = memory_dependency.get("negative_mode", "shuffled")
                dependency_permutation = None
                lora_gradient_multiplier = 1.0
                if contribution_v2:
                    reader_only_active = warmup_controller.reader_only
                    if reader_only_active:
                        lora_gradient_multiplier = 0.0
                    cadence = (
                        int(memory_dependency["warmup_every_n_steps"])
                        if reader_only_active
                        else int(memory_dependency["joint_every_n_steps"])
                    )
                    phase_origin = (
                        0
                        if reader_only_active
                        else int(warmup_controller.warmup_end_step or 0)
                    )
                    dependency_active = (global_step - phase_origin) % cadence == 0
                    dependency_mode = (
                        negative_cycle.current if dependency_active else "off"
                    )
                    if dependency_active and dependency_mode == "task_paired":
                        dependency_permutation = (
                            paired_sampler.permutation_for_sample_identities(
                                list(batch["sample_identity"])
                            )
                        )
                    if not reader_only_active:
                        lora_gradient_multiplier = (
                            warmup_controller.next_lora_gradient_multiplier
                        )
                else:
                    dependency_active = _memory_dependency_active(
                        global_step=global_step,
                        settings=memory_dependency,
                    )
                    reader_only_active = global_step < reader_only_warmup_steps
                with sync_context, torch.autocast("cuda", dtype=torch.bfloat16):
                    output = model(
                        batch,
                        timestep=timestep,
                        noise=noise,
                        memory_dependency_weight=(
                            memory_dependency["weight"] if dependency_active else 0.0
                        ),
                        memory_dependency_relative_margin=memory_dependency[
                            "relative_margin"
                        ],
                        memory_dependency_negative_mode=dependency_mode,
                        memory_dependency_permutation=dependency_permutation,
                        gripper_loss_multiplier=float(
                            cfg.training.get("gripper_loss_multiplier", 1.0)
                        ),
                    )
                    loss = output["loss_total"] / int(
                        cfg.training.gradient_accumulation_steps
                    )
                    loss.backward()
                sampler_offset = batch_index + 1
                if not sync_update:
                    continue
                if contribution_v2:
                    gradient_synchronizer.synchronize_gradients()
                finite_loss_keys = (
                    "loss_action_bc",
                    "loss_total",
                    "loss_memory_dependency",
                    "loss_action_bc_negative_memory",
                    "memory_dependency_gap",
                    "memory_dependency_relative_gap",
                )
                if any(
                    not bool(torch.isfinite(output[key]).all().item())
                    for key in finite_loss_keys
                ):
                    raise FloatingPointError("Non-finite P1 full training objective.")
                raw_gradient_norms = _family_norms(policy)
                if not all(
                    torch.isfinite(value) for value in raw_gradient_norms.values()
                ):
                    raise FloatingPointError("Non-finite raw P1 full gradient norm.")
                if reader_only_active:
                    _suppress_parameter_family_gradients(
                        policy,
                        P1_LORA_PARAMETER_FAMILY,
                    )
                elif contribution_v2:
                    _scale_parameter_family_gradients(
                        policy,
                        P1_LORA_PARAMETER_FAMILY,
                        lora_gradient_multiplier,
                    )
                gradient_norms = _family_norms(policy)
                if not all(torch.isfinite(value) for value in gradient_norms.values()):
                    raise FloatingPointError("Non-finite P1 full gradient norm.")
                before = _trainable_snapshot(policy)
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    max_norm=float(cfg.optimizer.gradient_clip),
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_norms = _family_norms(policy, before)
                if not all(torch.isfinite(value) for value in update_norms.values()):
                    raise FloatingPointError("Non-finite P1 full update norm.")
                if any(float(value.item()) > 0 for value in update_norms.values()):
                    nonzero_update_count += 1
                global_step += 1
                completed_warmup_window = None
                global_dependency_losses = None
                if contribution_v2:
                    if dependency_active:
                        reduced_losses = torch.stack(
                            (
                                output["loss_action_bc"].detach().float(),
                                output["loss_action_bc_negative_memory"]
                                .detach()
                                .float(),
                            )
                        )
                        if world_size > 1:
                            torch.distributed.all_reduce(
                                reduced_losses,
                                op=torch.distributed.ReduceOp.SUM,
                            )
                            reduced_losses /= world_size
                        global_dependency_losses = {
                            "correct": float(reduced_losses[0].item()),
                            "negative": float(reduced_losses[1].item()),
                        }
                        negative_cycle.advance()
                        if reader_only_active:
                            completed_warmup_window = warmup_controller.observe(
                                correct_loss=global_dependency_losses["correct"],
                                negative_loss=global_dependency_losses["negative"],
                                global_step_after_update=global_step,
                            )
                    if not reader_only_active:
                        warmup_controller.record_joint_update()
                    no_go_warmup = warmup_controller.failed
                _assert_frozen_versions(policy, frozen_versions)
                record = {
                    "global_step": global_step,
                    "epoch": epoch,
                    "sampler_offset": sampler_offset,
                    "loss_action_bc": float(output["loss_action_bc"].item()),
                    "loss_total": float(output["loss_total"].item()),
                    "loss_memory_dependency": float(
                        output["loss_memory_dependency"].item()
                    ),
                    "loss_action_bc_negative_memory": float(
                        output["loss_action_bc_negative_memory"].item()
                    ),
                    "memory_dependency_gap": float(
                        output["memory_dependency_gap"].item()
                    ),
                    "memory_dependency_relative_gap": float(
                        output["memory_dependency_relative_gap"].item()
                    ),
                    "memory_dependency_active": dependency_active,
                    "reader_only_warmup_active": reader_only_active,
                    "dino_contribution_phase": (
                        warmup_controller.phase if contribution_v2 else None
                    ),
                    "dependency_negative_mode": dependency_mode,
                    "global_dependency_losses": global_dependency_losses,
                    "completed_warmup_window": completed_warmup_window,
                    "lora_gradient_multiplier": (
                        lora_gradient_multiplier if contribution_v2 else 1.0
                    ),
                    "lora_ramp_progress": (
                        warmup_controller.lora_ramp_progress
                        if contribution_v2
                        else None
                    ),
                    "negative_cycle_offset": (
                        negative_cycle.offset if contribution_v2 else None
                    ),
                    "mse_pose": float(output["mse_pose"].item()),
                    "mse_gripper": float(output["mse_gripper"].item()),
                    "raw_gradient_norms": {
                        key: float(value.item())
                        for key, value in raw_gradient_norms.items()
                    },
                    "gradient_norms": {
                        key: float(value.item())
                        for key, value in gradient_norms.items()
                    },
                    "update_norms": {
                        key: float(value.item()) for key, value in update_norms.items()
                    },
                    "learning_rates": {
                        str(group["name"]): float(group["lr"])
                        for group in optimizer.param_groups
                    },
                    "sample_identity_sha256": identity_digest.hexdigest(),
                    "timestep_sha256": timestep_digest.hexdigest(),
                    "noise_sha256": noise_digest.hexdigest(),
                }
                if rank == 0:
                    with metrics_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, sort_keys=True) + "\n")
                identity_digest = hashlib.sha256()
                timestep_digest = hashlib.sha256()
                noise_digest = hashlib.sha256()
                if (
                    global_step
                    in {int(step) for step in cfg.training.early_checkpoint_steps}
                    or global_step % int(cfg.training.save_every_steps) == 0
                    or global_step >= stop_after_steps
                    or no_go_warmup
                ):
                    last_checkpoint = _save_training_checkpoint(
                        checkpoints_dir / f"step_{global_step:06d}.pt",
                        policy=policy,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        global_step=global_step,
                        epoch=epoch,
                        sampler_offset=sampler_offset,
                        contract=contract,
                        provenance=provenance,
                        trainer_state=_trainer_state_payload(
                            best_validation=best_validation,
                            best_step=best_step,
                            epochs_without_improvement=epochs_without_improvement,
                            nonzero_update_count=nonzero_update_count,
                        ),
                        assets=assets,
                        rank=rank,
                        world_size=world_size,
                        v2_control_state=(
                            {
                                "profile": DINO_CONTRIBUTION_V2_PROFILE,
                                "warmup": warmup_controller.state_dict(),
                                "negative_cycle": negative_cycle.state_dict(),
                                "causal_selector": causal_selector.state_dict(),
                            }
                            if contribution_v2
                            else None
                        ),
                        paired_sampler=paired_sampler,
                    )
                if no_go_warmup or global_step >= stop_after_steps:
                    stop = True
                    break

            if stop and str(cfg.runner.stage) in {"canary", "benchmark"}:
                break
            if no_go_warmup:
                break
            validation = _validate(
                model,
                policy,
                validation_loader,
                cfg=cfg,
                world_size=world_size,
                device=device,
            )
            causal_validation = None
            causal_assessment = None
            if contribution_v2:
                if causal_ledger is None:
                    raise ValueError("V2 epoch validation requires its frozen ledger.")
                causal_validation = _validate_v2_causal_ledger(
                    policy,
                    train_dataset,
                    validation_dataset,
                    causal_ledger,
                    cfg=cfg,
                    rank=rank,
                    world_size=world_size,
                    device=device,
                )
                _assert_frozen_versions(policy, frozen_versions)
                causal_assessment = causal_selector.assess(
                    step=global_step,
                    validation=validation,
                    causal=causal_validation,
                    frozen_contract_unchanged=True,
                )
                improved = causal_selector.consider(causal_assessment)
            else:
                improved = validation["loss_action_bc"] < best_validation
            if improved:
                best_validation = validation["loss_action_bc"]
                best_step = global_step
                epochs_without_improvement = 0
                if rank == 0:
                    best_path = output_dir / "best_reader_lora_checkpoint.pt"
                    common = {
                        "path": best_path,
                        "adapter": policy.lora_adapter,
                        "reader": policy.visual_reader,
                        "global_step": global_step,
                        "stage": str(cfg.runner.stage),
                        "arm": (
                            str(cfg.runner.get("arm", "p1"))
                            if contribution_v2
                            else "a3_joint"
                        ),
                        "parent_checkpoint_sha256": assets["parent_checkpoint_sha256"],
                        "memory_contract_sha256": policy.expected_memory_contract,
                        "contract": contract,
                        "provenance": provenance,
                        "trainer_state": {
                            "last_loss_action_bc": validation["loss_action_bc"],
                            "best_dev_loss_action_bc": validation["loss_action_bc"],
                            "nonzero_update_count": nonzero_update_count,
                        },
                    }
                    if is_visual_v2_config(cfg):
                        save_p1_visual_bc_checkpoint(
                            **common,
                            visual_backbone=assets["visual_backbone"],
                        )
                        inspect_p1_visual_bc_checkpoint(best_path)
                    else:
                        save_p1_dino_bc_checkpoint(
                            **common,
                            dinov3_weights_sha256=assets["dinov3_weights_sha256"],
                        )
                        inspect_p1_dino_bc_checkpoint(best_path)
            else:
                epochs_without_improvement += 1
            epoch_record = {
                "epoch": epoch,
                "global_step": global_step,
                "validation": validation,
                "causal_validation": causal_validation,
                "causal_assessment": causal_assessment,
                "improved": improved,
                "best_validation_loss_action_bc": best_validation,
                "epochs_without_improvement": epochs_without_improvement,
            }
            if rank == 0:
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(epoch_record, sort_keys=True) + "\n")
            next_epoch = epoch + 1
            sampler_offset = 0
            if contribution_v2:
                paired_sampler.set_epoch(next_epoch)
            last_checkpoint = _save_training_checkpoint(
                checkpoints_dir / f"epoch_{next_epoch:02d}_step_{global_step:06d}.pt",
                policy=policy,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                global_step=global_step,
                epoch=next_epoch,
                sampler_offset=0,
                contract=contract,
                provenance=provenance,
                trainer_state=_trainer_state_payload(
                    best_validation=best_validation,
                    best_step=best_step,
                    epochs_without_improvement=epochs_without_improvement,
                    nonzero_update_count=nonzero_update_count,
                ),
                assets=assets,
                rank=rank,
                world_size=world_size,
                v2_control_state=(
                    {
                        "profile": DINO_CONTRIBUTION_V2_PROFILE,
                        "warmup": warmup_controller.state_dict(),
                        "negative_cycle": negative_cycle.state_dict(),
                        "causal_selector": causal_selector.state_dict(),
                    }
                    if contribution_v2
                    else None
                ),
                paired_sampler=paired_sampler,
            )
            if stop:
                break
            if next_epoch >= int(cfg.training.minimum_epochs) and (
                epochs_without_improvement >= int(cfg.training.early_stopping_patience)
            ):
                early_stopped = True
                break
    finally:
        dino_hook.remove()

    _assert_frozen_versions(policy, frozen_versions)
    best_value = float(best_validation) if math.isfinite(best_validation) else None
    controlled_stop = str(cfg.runner.stage) in {"canary", "benchmark"}
    pilot_run = str(cfg.runner.stage) == "pilot"
    passed = (
        _run_acceptance_passed(
            global_step=global_step,
            stop_after_steps=stop_after_steps,
            nonzero_update_count=nonzero_update_count,
            last_checkpoint=last_checkpoint,
            best_step=best_step,
            controlled_stop=controlled_stop,
            eligibility_required=not controlled_stop and not pilot_run,
            early_stopped=early_stopped,
        )
        and not no_go_warmup
    )
    if no_go_warmup:
        completion = "NO-GO_WARMUP"
    elif pilot_run and best_step is None:
        completion = "PILOT_COMPLETE_NO_ELIGIBLE_CHECKPOINT"
    elif contribution_v2 and not controlled_stop and best_step is None:
        completion = "NO_ELIGIBLE_CHECKPOINT"
    elif controlled_stop:
        completion = (
            "CONTROLLED_CANARY_STOP"
            if str(cfg.runner.stage) == "canary"
            else "CONTROLLED_BENCHMARK_STOP"
        )
    else:
        completion = "EARLY_STOPPED" if early_stopped else "TRAINING_COMPLETE"
    manifest.update(
        {
            "status": "PASS" if passed else "FAIL",
            "completion": completion,
            "early_stopped": early_stopped,
            "optimizer_steps": global_step,
            "total_planned_steps": total_steps,
            "nonzero_update_count": nonzero_update_count,
            "best_step": best_step,
            "best_validation_loss_action_bc": best_value,
            "reader_only_optimizer_updates": min(
                global_step,
                (
                    warmup_controller.active_update_count
                    if contribution_v2
                    else reader_only_warmup_steps
                ),
            ),
            "memory_dependency_optimizer_updates": (
                negative_cycle.update_count
                if contribution_v2
                else (
                    (global_step + int(memory_dependency["every_n_steps"]) - 1)
                    // int(memory_dependency["every_n_steps"])
                    if memory_dependency["enabled"]
                    else 0
                )
            ),
            "v2_warmup": (warmup_controller.state_dict() if contribution_v2 else None),
            "v2_negative_cycle": (
                negative_cycle.state_dict() if contribution_v2 else None
            ),
            "v2_causal_selector": (
                causal_selector.state_dict() if contribution_v2 else None
            ),
            (
                "visual_encoder_forward_calls_rank_local"
                if is_visual_v2_config(cfg)
                else "dino_forward_calls_rank_local"
            ): visual_encoder_calls["count"],
            "frozen_parameter_versions_unchanged": True,
            "last_checkpoint": last_checkpoint,
            "elapsed_seconds": time.time() - started_at,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
    )
    if rank == 0:
        _write_run_manifest(output_dir, manifest, launch_hash=launch_hash)
    _barrier(world_size)
    if not passed:
        raise RuntimeError("P1 full training acceptance failed; inspect manifest.")
    return manifest


def run_p1_visual_bc_full(cfg: DictConfig) -> dict[str, Any]:
    """Run the V2 registered-visual full trainer with strict lineage checks."""

    if not is_visual_v2_config(cfg):
        raise ValueError("V2 visual trainer requires `p1.lineage=visual_v2`.")
    return run_p1_dino_bc_full(cfg)
