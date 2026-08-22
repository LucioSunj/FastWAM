"""Four-GPU trainer for shared dual/tri-mode causal ActionDiT adapters."""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from fastwam.causal_prediction import (
    CAUSAL_POLICY_CHECKPOINT_SCHEMA,
    CAUSAL_POLICY_CHECKPOINT_SCHEMA_V2,
    CURRENT_ONLY_CAUSAL_POLICY_CHECKPOINT_SCHEMA,
    CausalDualModeTrainingConfig,
    FastWAMCausalCurrentOnlyPolicy,
    FastWAMCausalDualModePolicy,
    FastWAMCausalTriModePolicy,
    checkpoint_passes_dual_mode_selection,
    deterministic_current_only_sequence,
    deterministic_dual_mode_sequence,
    deterministic_tri_mode_sequence,
    load_causal_policy_checkpoint,
    save_causal_policy_checkpoint,
)
from fastwam.causal_prediction.shared_lora import SharedLoRAConfig
from fastwam.uncond_bc import (
    FastWAMUncondBCConfig,
    cosine_warmup_multiplier,
    stateless_validation_flow_inputs,
)
from fastwam.uncond_bc_checkpoint import capture_rng_state, restore_rng_state
from fastwam.uncond_bc_trainer import (
    _build_datasets,
    _build_loaders,
    _dataset_summary,
    _verify_sha256,
    load_strict_fastwam_parent,
)
from fastwam.utils import misc

CAUSAL_OUTPUT_MARKER = ".fastwam-causal-prediction-output-v1"


def _startup_event(*, rank: int, event: str) -> None:
    """Emit durable-in-launcher progress without claiming a stage result."""

    print(
        json.dumps(
            {"event": event, "rank": int(rank), "schema": "causal-startup-event-v1"},
            sort_keys=True,
        ),
        flush=True,
    )


def _distributed_context() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 4:
        raise RuntimeError(
            "Causal dual-mode formal training requires exactly four GPUs."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Causal dual-mode training requires CUDA.")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=device)
    return rank, world_size, local_rank, device


def _set_seed(seed: int, *, rank: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    effective = int(seed) + int(rank)
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    torch.cuda.manual_seed_all(effective)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def claim_causal_output(cfg: DictConfig) -> Path:
    """Claim a causal-only output or validate the explicit resume owner."""

    output = Path(str(cfg.runner.output_dir)).expanduser().resolve()
    marker = output / CAUSAL_OUTPUT_MARKER
    resume = cfg.runner.get("resume")
    if resume:
        if not marker.is_file():
            raise FileNotFoundError("Causal resume output has no ownership marker.")
        return output
    if output.exists() and any(output.iterdir()):
        entries = {item.name for item in output.iterdir()}
        if entries != {CAUSAL_OUTPUT_MARKER}:
            raise FileExistsError(f"Causal output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{CAUSAL_OUTPUT_MARKER}\n", encoding="utf-8")
    return output


def _validate_config(cfg: DictConfig, *, world_size: int) -> None:
    if str(cfg.precision).lower() != "bf16" or bool(cfg.compile):
        raise ValueError("Causal dual-mode training is frozen to eager BF16.")
    if str(cfg.data.get("multiprocessing_context", "")) != "spawn":
        raise ValueError(
            "Causal DataLoader workers must use spawn to avoid inheriting the "
            "loaded parent model."
        )
    if str(cfg.data.get("video_backend", "")) != "pyav":
        raise ValueError(
            "Causal real-video loading must select PyAV directly in the pinned "
            "environment."
        )
    canary_steps = cfg.runner.get("canary_optimizer_steps")
    if canary_steps is not None and (
        str(cfg.runner.stage) != "model_acceptance_canary" or int(canary_steps) < 1
    ):
        raise ValueError(
            "A bounded optimizer-step cap is exclusive to model acceptance canary."
        )
    canary_warmup_steps = cfg.runner.get("canary_warmup_optimizer_steps")
    canary_validation_batches = cfg.runner.get("canary_validation_batches")
    canary_timing = bool(cfg.runner.get("canary_cuda_synchronize_timing", False))
    if canary_steps is None and (
        canary_warmup_steps is not None
        or canary_validation_batches is not None
        or canary_timing
    ):
        raise ValueError("Canary timing controls require an optimizer-step cap.")
    if canary_steps is not None:
        if canary_warmup_steps is not None and not (
            0 <= int(canary_warmup_steps) < int(canary_steps)
        ):
            raise ValueError("Canary warmup steps must be below the step cap.")
        if canary_validation_batches is not None and int(canary_validation_batches) < 1:
            raise ValueError("Canary validation batch cap must be positive.")
        if canary_validation_batches is not None and not canary_timing:
            raise ValueError(
                "Bounded validation timing requires CUDA-synchronized timing."
            )
    if not bool(cfg.training.deterministic_algorithms) or not bool(
        cfg.training.stateless_flow_inputs
    ):
        raise ValueError(
            "Causal training requires deterministic stateless flow inputs."
        )
    if int(cfg.training.num_gpus) != world_size:
        raise ValueError("Configured and launched causal GPU counts differ.")
    effective = (
        world_size
        * int(cfg.training.microbatch_size)
        * int(cfg.training.gradient_accumulation_steps)
    )
    if effective != int(cfg.training.global_batch_size) or effective != 128:
        raise ValueError("Causal training global batch must be exactly 128.")
    quotas = OmegaConf.to_container(
        cfg.training.samples_per_optimizer_step,
        resolve=True,
    )
    exposure = str(cfg.training.get("exposure", "dual_mode"))
    expected_quotas = {
        "dual_mode": {"c0_current": 64, "c2_full": 64},
        "current_only": {"c0_current": 128},
        "tri_mode": {
            "step_0": [43, 43, 42],
            "step_1": [42, 43, 43],
            "step_2": [43, 42, 43],
        },
    }
    if exposure not in expected_quotas or quotas != expected_quotas[exposure]:
        raise ValueError("Causal optimizer-step exposure/quota contract changed.")
    if (
        int(cfg.training.microbatch_size) != 8
        or int(cfg.training.gradient_accumulation_steps) != 4
    ):
        raise ValueError(
            "Causal four-GPU profile is frozen to microbatch 8 / accumulation 4."
        )
    if int(cfg.training.max_epochs) != 10 or int(cfg.training.minimum_epochs) != 3:
        raise ValueError("Causal epoch bounds changed.")
    if int(cfg.training.early_stopping_patience) != 2:
        raise ValueError("Causal early-stopping patience changed.")
    optimizer_contract = {
        "learning_rate": 3e-4,
        "betas": [0.9, 0.95],
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
        "warmup_fraction": 0.05,
    }
    observed_optimizer = {
        "learning_rate": float(cfg.optimizer.learning_rate),
        "betas": [float(value) for value in cfg.optimizer.betas],
        "weight_decay": float(cfg.optimizer.weight_decay),
        "gradient_clip": float(cfg.optimizer.gradient_clip),
        "warmup_fraction": float(cfg.optimizer.warmup_fraction),
    }
    if observed_optimizer != optimizer_contract:
        raise ValueError(f"Causal optimizer contract changed: {observed_optimizer}.")
    lora_contract = OmegaConf.to_container(cfg.shared_lora, resolve=True)
    if lora_contract != {
        "rank": 16,
        "alpha": 16.0,
        "dropout": 0.0,
        "target_groups": [
            "self_attention_qkvo",
            "cross_attention_qkvo",
            "ffn",
        ],
        "freeze_base": True,
        "strict_target_discovery": True,
    }:
        raise ValueError("Causal shared LoRA contract changed.")
    causal_shape = {
        "action_horizon": 32,
        "executed_action_horizon": 10,
        "action_dim": 7,
        "proprio_dim": 8,
        "expected_video_frames": 9,
        "expected_video_height": 224,
        "expected_video_width": 448,
        "gripper_dimension": 6,
        "action_flow_steps": 10,
        "environment_max_steps": 700,
        "modes": (
            ["c0_current", "c1_one_pass", "c2_full"]
            if exposure == "tri_mode"
            else ["c0_current", "c2_full"]
        ),
    }
    observed_shape = {
        key: (
            list(cfg.causal_policy[key])
            if key == "modes"
            else int(cfg.causal_policy[key])
        )
        for key in causal_shape
    }
    if observed_shape != causal_shape:
        raise ValueError(f"Causal model/action contract changed: {observed_shape}.")
    if exposure == "tri_mode":
        c1_contract = OmegaConf.to_container(cfg.c1_one_pass, resolve=True)
        if c1_contract != {
            "video_timestep": 1.0,
            "layer_stride": 4,
            "selected_layers": [0, 4, 8, 12, 16, 20, 24, 28],
            "expected_layers": 30,
        }:
            raise ValueError("C1 one-pass/interval-fusion contract changed.")
    weight = float(cfg.selection.distillation_weight)
    attempt = str(cfg.selection.get("attempt", "primary"))
    allowed_attempts = {
        ("primary", 1.0),
        ("retry", 4.0),
        ("current_only_diagnostic", 1.0),
        ("tri_mode", 1.0),
        ("tri_mode_retry", 4.0),
    }
    if (attempt, weight) not in allowed_attempts:
        raise ValueError("Causal distillation attempt/weight is not preregistered.")


def _set_causal_video_backend(
    datasets: tuple[Any, ...],
    *,
    backend: str,
) -> int:
    """Set the selected backend before causal DataLoader workers are spawned."""

    count = 0
    for dataset in datasets:
        multi = dataset.dataset.lerobot_dataset.multi_dataset
        for subset in multi._datasets:
            subset.video_backend = backend
            count += 1
    return count


def _resume_contract_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only operational fields that cannot change training math."""

    canonical = json.loads(json.dumps(config, sort_keys=True, default=str))
    runner = canonical.get("runner", {})
    runner.pop("resume", None)
    runner.pop("output_dir", None)
    canonical.get("training", {}).pop("checkpoint_keep_last", None)
    return canonical


def _expected_global_mode_counts(
    *, exposure: str, optimizer_step: int
) -> tuple[int, int, int]:
    """Return preregistered global C0/C1/C2 samples for one optimizer step."""

    if exposure == "dual_mode":
        return (64, 0, 64)
    if exposure == "current_only":
        return (128, 0, 0)
    rotating = ((43, 43, 42), (42, 43, 43), (43, 42, 43))
    return rotating[optimizer_step % len(rotating)]


def _all_reduce(values: torch.Tensor, *, world_size: int) -> torch.Tensor:
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


@torch.no_grad()
def _validate(
    policy: FastWAMCausalDualModePolicy,
    loader,
    *,
    cfg: DictConfig,
    world_size: int,
    device: torch.device,
    max_batches_per_rank: int | None = None,
    synchronize_timing: bool = False,
) -> dict[str, Any]:
    policy.eval()
    tri_mode = isinstance(policy, FastWAMCausalTriModePolicy)
    totals = torch.zeros(5 if tri_mode else 4, device=device, dtype=torch.float64)
    if max_batches_per_rank is not None and max_batches_per_rank < 1:
        raise ValueError("Validation batch cap must be positive.")
    if synchronize_timing:
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    local_batch_count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches_per_rank is not None and batch_index >= max_batches_per_rank:
            break
        identities = list(batch["sample_identity"])
        timestep, noise = stateless_validation_flow_inputs(
            sample_identities=identities,
            action_shape=tuple(batch["action"].shape),
            scheduler=policy.actor.train_action_scheduler,
            seed=int(cfg.validation.seed),
            device=device,
            dtype=policy.dtype,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = (
                policy.evaluate_all_modes(
                    batch,
                    timestep=timestep,
                    noise=noise,
                )
                if tri_mode
                else policy.evaluate_both_modes(
                    batch,
                    timestep=timestep,
                    noise=noise,
                )
            )
        count = int(batch["action"].shape[0])
        values = [float(output["loss_c0_action"]) * count]
        if tri_mode:
            values.append(float(output["loss_c1_action"]) * count)
        values.extend(
            (
                float(output["loss_c2_action"]) * count,
                float(output["loss_c2_teacher"]) * count,
                count,
            )
        )
        totals += torch.tensor(
            values,
            device=device,
            dtype=torch.float64,
        )
        local_batch_count += 1
    if synchronize_timing:
        torch.cuda.synchronize(device)
    elapsed = torch.tensor(
        time.perf_counter() - started,
        device=device,
        dtype=torch.float64,
    )
    batch_counts = torch.tensor(
        [local_batch_count, local_batch_count],
        device=device,
        dtype=torch.int64,
    )
    if world_size > 1:
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        dist.all_reduce(batch_counts[0], op=dist.ReduceOp.MIN)
        dist.all_reduce(batch_counts[1], op=dist.ReduceOp.MAX)
    totals = _all_reduce(totals, world_size=world_size)
    count_index = 4 if tri_mode else 3
    if totals[count_index] <= 0:
        raise RuntimeError("Causal validation evaluated no samples.")
    denominator = totals[count_index]
    result = {"loss_c0_action": float((totals[0] / denominator).item())}
    if tri_mode:
        result["loss_c1_action"] = float((totals[1] / denominator).item())
        c2_index, teacher_index = 2, 3
    else:
        c2_index, teacher_index = 1, 2
    result.update(
        {
            "loss_c2_action": float((totals[c2_index] / denominator).item()),
            "loss_c2_teacher": float((totals[teacher_index] / denominator).item()),
            "sample_count": int(denominator.item()),
            "elapsed_seconds": float(elapsed.item()),
            "global_samples_per_second": float(denominator.item() / elapsed.item()),
            "batches_per_rank_min": int(batch_counts[0].item()),
            "batches_per_rank_max": int(batch_counts[1].item()),
            "batch_cap_per_rank": max_batches_per_rank,
        }
    )
    result["loss_equal_mode"] = sum(
        result[name]
        for name in (
            ("loss_c0_action", "loss_c1_action", "loss_c2_action")
            if tri_mode
            else ("loss_c0_action", "loss_c2_action")
        )
    ) / (3.0 if tri_mode else 2.0)
    result["c2_teacher_guard"] = checkpoint_passes_dual_mode_selection(result)
    return result


def _gather_rng(*, world_size: int) -> list[Mapping[str, Any]]:
    state = capture_rng_state()
    gathered: list[Mapping[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, state)
    if any(item is None for item in gathered):
        raise RuntimeError("Failed to gather causal checkpoint RNG states.")
    return [item for item in gathered if item is not None]


def _snapshot_causal_trainables(
    policy: FastWAMCausalDualModePolicy,
) -> dict[str, torch.Tensor]:
    """Clone every allowed causal trainable for update auditing."""

    allowed = {id(item) for item in policy.lora_adapter.lora_parameters()}
    if isinstance(policy, FastWAMCausalTriModePolicy):
        allowed.update(id(item) for item in policy.c1_fusion.parameters())
    return {
        name: parameter.detach().clone()
        for name, parameter in policy.actor.named_parameters()
        if id(parameter) in allowed
    }


def _causal_gradient_norm(parameters: list[nn.Parameter]) -> torch.Tensor:
    terms = [
        parameter.grad.detach().float().square().sum()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return torch.stack(terms).sum().sqrt() if terms else torch.tensor(0.0)


def _causal_update_norm(
    policy: FastWAMCausalDualModePolicy,
    before: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    current = dict(policy.actor.named_parameters())
    terms = [
        (current[name].detach().float() - value.float()).square().sum()
        for name, value in before.items()
    ]
    return torch.stack(terms).sum().sqrt() if terms else torch.tensor(0.0)


def _save(
    path: Path,
    *,
    policy: FastWAMCausalDualModePolicy,
    optimizer,
    scheduler,
    global_step: int,
    epoch: int,
    parent_hash: str,
    statistics_hash: str,
    cfg: DictConfig,
    rank: int,
    world_size: int,
    trainer_state: Mapping[str, Any],
    checkpoint_schema: str,
) -> None:
    rng_by_rank = _gather_rng(world_size=world_size)
    if rank != 0:
        return
    report = save_causal_policy_checkpoint(
        path,
        adapter=policy.lora_adapter,
        parent_checkpoint_sha256=parent_hash,
        statistics_sha256=statistics_hash,
        global_step=global_step,
        epoch=epoch,
        optimizer_state=optimizer.state_dict(),
        lr_scheduler_state=scheduler.state_dict(),
        grad_scaler_state={},
        rng_by_rank=rng_by_rank,
        trainer_state=trainer_state,
        config=OmegaConf.to_container(cfg, resolve=True),
        checkpoint_schema=checkpoint_schema,
        fusion=(
            policy.c1_fusion if isinstance(policy, FastWAMCausalTriModePolicy) else None
        ),
    )
    _atomic_json(path.with_suffix(".inspection.json"), {"status": "PASS", **report})
    recognized = sorted(path.parent.glob("checkpoint_step_*.pt"))
    keep = int(cfg.training.checkpoint_keep_last)
    for old in recognized[:-keep]:
        old.unlink()
        old.with_suffix(".inspection.json").unlink(missing_ok=True)


def _run_causal_adapter_training(cfg: DictConfig) -> dict[str, Any]:
    """Train one frozen-parent causal adapter exposure without evaluation."""

    rank, world_size, local_rank, device = _distributed_context()
    _startup_event(rank=rank, event="distributed_context_ready")
    _validate_config(cfg, world_size=world_size)
    _set_seed(int(cfg.seed), rank=rank)
    output_dir = Path(str(cfg.runner.output_dir)).expanduser().resolve()
    if rank == 0:
        claim_causal_output(cfg)
    dist.barrier()
    if not (output_dir / CAUSAL_OUTPUT_MARKER).is_file():
        raise RuntimeError("Causal output ownership marker was not synchronized.")
    misc.register_work_dir(output_dir)
    _startup_event(rank=rank, event="causal_work_dir_registered")
    identities = [
        (
            _verify_sha256(
                str(cfg.parent.checkpoint),
                str(cfg.parent.checkpoint_sha256),
                label="FastWAM parent",
            ),
            _verify_sha256(
                str(cfg.parent.statistics),
                str(cfg.parent.statistics_sha256),
                label="FastWAM statistics",
            ),
        )
        if rank == 0
        else None
    ]
    dist.broadcast_object_list(identities, src=0)
    parent_hash, statistics_hash = identities[0]
    _startup_event(rank=rank, event="asset_identity_verified")
    actor = instantiate(cfg.model, model_dtype=torch.bfloat16, device=str(device))
    _startup_event(rank=rank, event="actor_instantiated")
    parent_load = load_strict_fastwam_parent(actor, str(cfg.parent.checkpoint))
    _startup_event(rank=rank, event="parent_loaded")
    lora_config = SharedLoRAConfig(
        **OmegaConf.to_container(cfg.shared_lora, resolve=True)
    )
    bc_payload = OmegaConf.to_container(cfg.causal_policy, resolve=True)
    for key in (
        "executed_action_horizon",
        "action_flow_steps",
        "environment_max_steps",
        "modes",
    ):
        bc_payload.pop(key)
    exposure = str(cfg.training.get("exposure", "dual_mode"))
    policy_class = {
        "dual_mode": FastWAMCausalDualModePolicy,
        "current_only": FastWAMCausalCurrentOnlyPolicy,
        "tri_mode": FastWAMCausalTriModePolicy,
    }[exposure]
    checkpoint_schema = {
        "dual_mode": CAUSAL_POLICY_CHECKPOINT_SCHEMA,
        "current_only": CURRENT_ONLY_CAUSAL_POLICY_CHECKPOINT_SCHEMA,
        "tri_mode": CAUSAL_POLICY_CHECKPOINT_SCHEMA_V2,
    }[exposure]
    policy = policy_class(
        actor=actor,
        lora_config=lora_config,
        bc_config=FastWAMUncondBCConfig(**bc_payload),
        training_config=CausalDualModeTrainingConfig(
            distillation_weight=float(cfg.selection.distillation_weight),
            c2_teacher_degradation_limit=float(
                cfg.selection.c2_teacher_degradation_limit
            ),
        ),
    ).to(device)
    _startup_event(rank=rank, event="shared_policy_constructed")
    train_dataset, validation_dataset = _build_datasets(cfg)
    video_dataset_count = _set_causal_video_backend(
        (train_dataset, validation_dataset),
        backend=str(cfg.data.video_backend),
    )
    train_summary = _dataset_summary(train_dataset)
    validation_summary = _dataset_summary(validation_dataset)
    if train_summary["episodes"] != int(cfg.data.expected_train_episodes) or (
        validation_summary["episodes"] != int(cfg.data.expected_validation_episodes)
    ):
        raise ValueError("Causal train/validation episode identities changed.")
    if train_summary["windows"] + validation_summary["windows"] != int(
        cfg.data.expected_source_transitions
    ):
        raise ValueError("Causal source transition count changed.")
    _startup_event(rank=rank, event="dataset_contract_verified")
    # Do not let faster ranks fork persistent video decoders while another rank
    # is still constructing the same dataset metadata and split identities.
    dist.barrier()
    _startup_event(rank=rank, event="dataset_ranks_synchronized")
    train_loader, validation_loader, train_sampler = _build_loaders(
        cfg,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        rank=rank,
        world_size=world_size,
    )
    parameters = list(policy.lora_adapter.lora_parameters())
    if isinstance(policy, FastWAMCausalTriModePolicy):
        parameters.extend(policy.c1_fusion.parameters())
    for parameter in parameters:
        dist.broadcast(parameter.data, src=0)
    _startup_event(rank=rank, event="causal_trainables_synchronized")
    model: nn.Module = DistributedDataParallel(
        policy,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
        # Every frozen tensor was strictly restored from the same parent on every
        # rank above. Sync only the causal trainables explicitly instead of
        # broadcasting the complete 6B frozen policy during DDP construction.
        init_sync=False,
    )
    _startup_event(rank=rank, event="ddp_ready")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(cfg.optimizer.learning_rate),
        betas=tuple(float(value) for value in cfg.optimizer.betas),
        eps=float(cfg.optimizer.eps),
        weight_decay=float(cfg.optimizer.weight_decay),
    )
    updates_per_epoch = len(train_loader) // int(
        cfg.training.gradient_accumulation_steps
    )
    total_steps = updates_per_epoch * int(cfg.training.max_epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_warmup_multiplier(
            step,
            total_steps=total_steps,
            warmup_fraction=float(cfg.optimizer.warmup_fraction),
            minimum_ratio=float(cfg.optimizer.minimum_lr_ratio),
        ),
    )
    global_step = 0
    start_epoch = 0
    best_validation = math.inf
    best_step = None
    epochs_without_improvement = 0
    sampler_offset = 0
    resume = cfg.runner.get("resume")
    if resume:
        payload = load_causal_policy_checkpoint(
            str(resume),
            adapter=policy.lora_adapter,
            expected_parent_checkpoint_sha256=parent_hash,
            expected_statistics_sha256=statistics_hash,
            expected_checkpoint_schema=checkpoint_schema,
            fusion=(
                policy.c1_fusion
                if isinstance(policy, FastWAMCausalTriModePolicy)
                else None
            ),
        )
        current_config = OmegaConf.to_container(cfg, resolve=True)
        if _resume_contract_config(payload["config"]) != _resume_contract_config(
            current_config
        ):
            raise ValueError("Causal resume training-math config changed.")
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["lr_scheduler"])
        global_step = int(payload["metadata"]["global_step"])
        start_epoch = int(payload["metadata"]["epoch"])
        if len(payload["rng_by_rank"]) != world_size:
            raise ValueError("Causal checkpoint RNG rank count changed.")
        restore_rng_state(payload["rng_by_rank"][rank])
        best_validation = float(payload["trainer_state"]["best_validation"])
        best_step = payload["trainer_state"]["best_step"]
        epochs_without_improvement = int(
            payload["trainer_state"]["epochs_without_improvement"]
        )
        sampler_offset = int(payload["trainer_state"]["sampler_offset"])
    frozen_versions = {
        name: parameter._version
        for name, parameter in policy.actor.named_parameters()
        if not parameter.requires_grad
    }
    metrics_path = output_dir / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    canary_steps = cfg.runner.get("canary_optimizer_steps")
    canary_limit = None if canary_steps is None else int(canary_steps)
    canary_warmup_steps = cfg.runner.get("canary_warmup_optimizer_steps")
    canary_warmup_limit = (
        None if canary_warmup_steps is None else int(canary_warmup_steps)
    )
    canary_validation_batches = cfg.runner.get("canary_validation_batches")
    canary_validation_limit = (
        None if canary_validation_batches is None else int(canary_validation_batches)
    )
    canary_timing = bool(cfg.runner.get("canary_cuda_synchronize_timing", False))
    optimizer_step_timings: list[float] = []
    bounded_validation: dict[str, Any] | None = None
    canary_stopped = False
    last_global_mode_counts = (0, 0, 0)
    for epoch in range(start_epoch, int(cfg.training.max_epochs)):
        policy.train()
        train_sampler.set_epoch(epoch)
        train_loader.generator.manual_seed(int(cfg.seed) + epoch)
        accumulation = 0
        step_mode_counts = torch.zeros(3, device=device, dtype=torch.int64)
        usable_batches = (
            len(train_loader)
            // int(cfg.training.gradient_accumulation_steps)
            * int(cfg.training.gradient_accumulation_steps)
        )
        if canary_timing:
            torch.cuda.synchronize(device)
        optimizer_step_started = time.perf_counter()
        for batch_index, batch in enumerate(train_loader):
            if batch_index < sampler_offset:
                continue
            if batch_index >= usable_batches:
                break
            accumulation_index = accumulation % int(
                cfg.training.gradient_accumulation_steps
            )
            if exposure == "tri_mode":
                modes = deterministic_tri_mode_sequence(
                    int(batch["action"].shape[0]),
                    optimizer_step=global_step,
                    accumulation_index=accumulation_index,
                    rank=rank,
                    world_size=world_size,
                    accumulation_steps=int(cfg.training.gradient_accumulation_steps),
                )
            else:
                sequence_builder = (
                    deterministic_dual_mode_sequence
                    if exposure == "dual_mode"
                    else deterministic_current_only_sequence
                )
                modes = sequence_builder(
                    int(batch["action"].shape[0]),
                    optimizer_step=global_step,
                    accumulation_index=accumulation_index,
                )
            for mode_index, mode in enumerate(
                (
                    "c0_current",
                    "c1_one_pass",
                    "c2_full",
                )
            ):
                step_mode_counts[mode_index] += sum(
                    value.value == mode for value in modes
                )
            identities = [
                f"epoch-{epoch}:{value}" for value in batch["sample_identity"]
            ]
            timestep, noise = stateless_validation_flow_inputs(
                sample_identities=identities,
                action_shape=tuple(batch["action"].shape),
                scheduler=policy.actor.train_action_scheduler,
                seed=int(cfg.seed),
                device=device,
                dtype=policy.dtype,
            )
            sync = (
                accumulation_index == int(cfg.training.gradient_accumulation_steps) - 1
            )
            sync_context = contextlib.nullcontext() if sync else model.no_sync()
            with sync_context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    batch,
                    modes=modes,
                    timestep=timestep,
                    noise=noise,
                )
                (
                    output["loss"] / int(cfg.training.gradient_accumulation_steps)
                ).backward()
            accumulation += 1
            sampler_offset = batch_index + 1
            if not sync:
                continue
            global_mode_counts = step_mode_counts.clone()
            dist.all_reduce(global_mode_counts, op=dist.ReduceOp.SUM)
            observed_mode_counts = tuple(
                int(value) for value in global_mode_counts.tolist()
            )
            expected_mode_counts = _expected_global_mode_counts(
                exposure=exposure,
                optimizer_step=global_step,
            )
            if observed_mode_counts != expected_mode_counts:
                raise RuntimeError(
                    "Causal optimizer-step mode quota changed: "
                    f"{observed_mode_counts} != {expected_mode_counts}."
                )
            last_global_mode_counts = observed_mode_counts
            step_mode_counts.zero_()
            if not torch.isfinite(output["loss"]):
                raise FloatingPointError("Causal dual-mode loss became non-finite.")
            gradient_norm = _causal_gradient_norm(parameters)
            before = _snapshot_causal_trainables(policy)
            torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=float(cfg.optimizer.gradient_clip),
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update_norm = _causal_update_norm(policy, before)
            if not torch.isfinite(gradient_norm) or not torch.isfinite(update_norm):
                raise FloatingPointError(
                    "Causal shared LoRA update audit is non-finite."
                )
            global_step += 1
            changed = {
                name: parameter._version
                for name, parameter in policy.actor.named_parameters()
                if not parameter.requires_grad
            }
            if changed != frozen_versions:
                raise RuntimeError(
                    "A frozen parent parameter changed during causal training."
                )
            optimizer_step_elapsed = None
            if canary_timing:
                torch.cuda.synchronize(device)
                optimizer_step_elapsed = time.perf_counter() - optimizer_step_started
                optimizer_step_timings.append(optimizer_step_elapsed)
                optimizer_step_started = time.perf_counter()
            if rank == 0:
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "global_step": global_step,
                                "epoch": epoch,
                                **{
                                    name: float(value.detach().float().item())
                                    for name, value in output.items()
                                },
                                "gradient_norm": float(gradient_norm.item()),
                                "update_norm": float(update_norm.item()),
                                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                                "optimizer_step_elapsed_seconds": optimizer_step_elapsed,
                                "global_mode_sample_counts": {
                                    "c0_current": observed_mode_counts[0],
                                    "c1_one_pass": observed_mode_counts[1],
                                    "c2_full": observed_mode_counts[2],
                                },
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
            if canary_limit is not None and global_step >= canary_limit:
                _save(
                    output_dir
                    / "checkpoints"
                    / f"checkpoint_step_{global_step:06d}.pt",
                    policy=policy,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    epoch=epoch,
                    parent_hash=parent_hash,
                    statistics_hash=statistics_hash,
                    cfg=cfg,
                    rank=rank,
                    world_size=world_size,
                    trainer_state={
                        "best_validation": best_validation,
                        "best_step": best_step,
                        "epochs_without_improvement": epochs_without_improvement,
                        "sampler_offset": sampler_offset,
                        "canary_complete": True,
                    },
                    checkpoint_schema=checkpoint_schema,
                )
                canary_stopped = True
                break
            if global_step % int(cfg.training.save_every_steps) == 0:
                _save(
                    output_dir
                    / "checkpoints"
                    / f"checkpoint_step_{global_step:06d}.pt",
                    policy=policy,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    global_step=global_step,
                    epoch=epoch,
                    parent_hash=parent_hash,
                    statistics_hash=statistics_hash,
                    cfg=cfg,
                    rank=rank,
                    world_size=world_size,
                    trainer_state={
                        "best_validation": best_validation,
                        "best_step": best_step,
                        "epochs_without_improvement": epochs_without_improvement,
                        "sampler_offset": sampler_offset,
                    },
                    checkpoint_schema=checkpoint_schema,
                )
        if canary_stopped:
            break
        validation = _validate(
            policy,
            validation_loader,
            cfg=cfg,
            world_size=world_size,
            device=device,
        )
        eligible = (
            bool(validation["c2_teacher_guard"])
            if exposure in {"dual_mode", "tri_mode"}
            else True
        )
        selection_loss = (
            validation["loss_equal_mode"]
            if exposure in {"dual_mode", "tri_mode"}
            else validation["loss_c0_action"]
        )
        improved = eligible and selection_loss < best_validation
        if improved:
            best_validation = selection_loss
            best_step = global_step
            epochs_without_improvement = 0
            _save(
                output_dir / "best_causal_policy.pt",
                policy=policy,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=global_step,
                epoch=epoch + 1,
                parent_hash=parent_hash,
                statistics_hash=statistics_hash,
                cfg=cfg,
                rank=rank,
                world_size=world_size,
                trainer_state={
                    "best_validation": best_validation,
                    "best_step": best_step,
                    "epochs_without_improvement": 0,
                    "sampler_offset": 0,
                },
                checkpoint_schema=checkpoint_schema,
            )
        else:
            epochs_without_improvement += 1
        if rank == 0:
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "validation": validation,
                            "eligible": eligible,
                            "improved": improved,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        _save(
            output_dir / "checkpoints" / f"checkpoint_step_{global_step:06d}.pt",
            policy=policy,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            epoch=epoch + 1,
            parent_hash=parent_hash,
            statistics_hash=statistics_hash,
            cfg=cfg,
            rank=rank,
            world_size=world_size,
            trainer_state={
                "best_validation": best_validation,
                "best_step": best_step,
                "epochs_without_improvement": epochs_without_improvement,
                "sampler_offset": 0,
            },
            checkpoint_schema=checkpoint_schema,
        )
        sampler_offset = 0
        if epoch + 1 >= int(cfg.training.minimum_epochs) and (
            epochs_without_improvement >= int(cfg.training.early_stopping_patience)
        ):
            break
    if canary_stopped and canary_validation_limit is not None:
        bounded_validation = _validate(
            policy,
            validation_loader,
            cfg=cfg,
            world_size=world_size,
            device=device,
            max_batches_per_rank=canary_validation_limit,
            synchronize_timing=canary_timing,
        )
    if (
        not canary_stopped
        and best_step is None
        and exposure in {"dual_mode", "tri_mode"}
    ):
        raise RuntimeError(
            "No checkpoint satisfied the C2 2% teacher guard. Stop this attempt; "
            "only the preregistered independent distillation-weight-4 retry is allowed."
        )
    manifest = {
        "schema": "fastwam-causal-adapter-exposure-run-v1",
        "status": "PASS",
        "stage": str(cfg.runner.stage),
        "attempt": str(cfg.selection.get("attempt", "primary")),
        "training_exposure": exposure,
        "checkpoint_schema": checkpoint_schema,
        "parent": parent_hash,
        "statistics": statistics_hash,
        "parent_load": parent_load,
        "train_dataset": train_summary,
        "validation_dataset": validation_summary,
        "data_loader_profile": {
            "num_workers_per_loader_per_rank": int(cfg.data.num_workers),
            "prefetch_factor": int(cfg.data.prefetch_factor),
            "persistent_workers": bool(int(cfg.data.num_workers) > 0),
            "loader_count_per_rank": 2,
            "multiprocessing_context": str(cfg.data.multiprocessing_context),
            "video_backend": str(cfg.data.video_backend),
            "video_dataset_count": video_dataset_count,
        },
        "global_step": global_step,
        "best_step": best_step,
        "best_equal_mode_validation_loss": best_validation,
        "run_scope": "bounded_canary" if canary_stopped else "formal_training",
        "canary_optimizer_step_limit": canary_limit,
        "canary_warmup_optimizer_step_limit": canary_warmup_limit,
        "canary_validation_batch_limit_per_rank": canary_validation_limit,
        "canary_cuda_synchronize_timing": canary_timing,
        "optimizer_step_elapsed_seconds": optimizer_step_timings,
        "bounded_validation": bounded_validation,
        "last_global_mode_sample_counts": {
            "c0_current": last_global_mode_counts[0],
            "c1_one_pass": last_global_mode_counts[1],
            "c2_full": last_global_mode_counts[2],
        },
        "trainable_parameter_names": list(policy.trainable_parameter_names()),
        "elapsed_seconds": time.time() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
        "later_phases": "NOT-AUTHORIZED",
    }
    if rank == 0:
        _atomic_json(output_dir / "run_manifest.json", manifest)
    dist.barrier()
    return manifest


def run_causal_dual_mode(cfg: DictConfig) -> dict[str, Any]:
    """Train the preregistered shared dual-mode causal adapter."""

    if str(cfg.training.get("exposure", "dual_mode")) != "dual_mode":
        raise ValueError("Dual-mode launcher received another exposure config.")
    return _run_causal_adapter_training(cfg)


def run_causal_current_only(cfg: DictConfig) -> dict[str, Any]:
    """Train the matched-budget current-only adapter exposure diagnostic."""

    if str(cfg.training.get("exposure", "")) != "current_only":
        raise ValueError("Current-only launcher requires current_only exposure.")
    return _run_causal_adapter_training(cfg)


def run_causal_tri_mode(cfg: DictConfig) -> dict[str, Any]:
    """Train the conditional shared C0/C1/C2 causal checkpoint."""

    if str(cfg.training.get("exposure", "")) != "tri_mode":
        raise ValueError("Tri-mode launcher requires tri_mode exposure.")
    return _run_causal_adapter_training(cfg)
