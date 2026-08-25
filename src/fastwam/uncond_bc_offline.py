"""Full deterministic held-out evaluation for UNCOND action-only BC."""

from __future__ import annotations

import math
import os
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from fastwam.adapters import RegimeLoRAConfig, sha256_file
from fastwam.uncond_bc import FastWAMUncondBCConfig, FastWAMUncondBCPolicy
from fastwam.uncond_bc_checkpoint import load_uncond_bc_adapter_checkpoint
from fastwam.uncond_bc_trainer import (
    _assert_frozen_versions,
    _atomic_json,
    _barrier,
    _build_datasets,
    _build_loaders,
    _build_provenance,
    _canonical_config,
    _dataset_summary,
    _distributed_context,
    _frozen_versions,
    _set_seed,
    _validate,
    _verify_sha256,
    _zero_lora,
    load_strict_fastwam_parent,
)

OFFLINE_EVAL_SCHEMA = "fastwam-uncond-bc-offline-eval-v1"
OFFLINE_FAILURE_SCHEMA = "fastwam-uncond-bc-offline-failure-v1"
OFFLINE_OUTPUT_MARKER = ".fastwam-uncond-bc-offline-output-v1"
EXPECTED_VALIDATION_WINDOWS = 29052


def claim_uncond_bc_offline_output(cfg: DictConfig) -> Path:
    """Claim a new non-overwriting offline-validation output directory."""

    output = Path(str(cfg.runner.output_dir)).expanduser().resolve()
    if int(os.environ.get("RANK", "0")) != 0:
        return output
    marker = output / OFFLINE_OUTPUT_MARKER
    if output.exists():
        entries = {entry.name for entry in output.iterdir()}
        if entries and entries != {OFFLINE_OUTPUT_MARKER}:
            raise FileExistsError(f"BC offline output directory is not empty: {output}")
    else:
        output.mkdir(parents=True, exist_ok=False)
    if not marker.is_file():
        marker.write_text("fastwam-uncond-bc-offline-output-v1\n", encoding="utf-8")
    return output


def _validate_offline_config(cfg: DictConfig, *, world_size: int) -> None:
    if str(cfg.runner.stage) != "offline_validation":
        raise ValueError("BC offline runner.stage must be offline_validation.")
    expected_world_size = 6 if int(cfg.lora.rank) == 32 else 4
    if world_size != expected_world_size:
        raise ValueError(
            "Full BC offline validation requires exactly "
            f"{expected_world_size} GPU ranks for LoRA rank {cfg.lora.rank}."
        )
    if int(cfg.seed) != 42 or int(cfg.validation.seed) != 42:
        raise ValueError("BC offline split and validation seed must be 42.")
    if str(cfg.precision).lower() != "bf16" or bool(cfg.get("compile", False)):
        raise ValueError("BC offline validation requires eager BF16.")
    if int(cfg.distributed.collective_timeout_seconds) != 7200:
        raise ValueError("BC offline collective timeout must be 7200 seconds.")
    if int(cfg.training.microbatch_size) != 8:
        raise ValueError(
            "BC offline validation uses the selected microbatch 8 profile."
        )
    expected_parent = (
        "/XYFS02/HDD_POOL/nju_shklu/nju_shklu_1/"
        "When-will-inference-time-prediction-beneficial-/"
        "fastwam-idm-wan-robot-init/"
        "fastwam-idm-libero-wan-robot-init-step_021700.pt"
    )
    if (
        str(cfg.parent.checkpoint) != expected_parent
        or str(cfg.parent.checkpoint_sha256)
        != "e979511a2d7a1310009496c6b2f06957171bba28b96aac0d513992c6ed21ca5a"
    ):
        raise ValueError("BC offline parent path/hash differs from training.")
    expected_stats = str(Path(expected_parent).with_name("dataset_stats.json"))
    if (
        str(cfg.parent.statistics) != expected_stats
        or str(cfg.parent.statistics_sha256)
        != "30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638"
    ):
        raise ValueError("BC offline statistics path/hash differs from training.")
    expected_suites = [
        "libero_spatial_no_noops_lerobot",
        "libero_object_no_noops_lerobot",
        "libero_goal_no_noops_lerobot",
        "libero_10_no_noops_lerobot",
    ]
    if [Path(str(path)).name for path in cfg.provenance.dataset_paths] != (
        expected_suites
    ):
        raise ValueError("BC offline dataset suite/order differs from training.")
    for split in (cfg.data.train, cfg.data.validation):
        if int(split.split_seed) != 42 or str(split.pretrained_norm_stats) != (
            expected_stats
        ):
            raise ValueError("BC offline split/statistics contract changed.")
        if (
            bool(split.processor.use_stepwise_action_norm)
            or str(split.processor.norm_default_mode) != "min/max"
        ):
            raise ValueError("BC offline normalization contract changed.")
        if not bool(split.current_frame_image_only) or (
            int(split.processor.num_image_obs_steps) != 1
        ):
            raise ValueError(
                "BC offline validation must decode only the current image."
            )
    if not bool(cfg.data.train.is_training_set) or bool(
        cfg.data.validation.is_training_set
    ):
        raise ValueError("BC offline train/validation split roles changed.")
    if (
        int(cfg.model.video_dit_config.num_layers) != 30
        or int(cfg.model.action_dit_config.num_layers) != 30
    ):
        raise ValueError("BC offline requires the same 30-layer parent model.")
    expected_shape = (32, 7, 8, 9, 224, 448, 6)
    observed_shape = tuple(
        int(cfg.bc_policy[key])
        for key in (
            "action_horizon",
            "action_dim",
            "proprio_dim",
            "expected_video_frames",
            "expected_video_height",
            "expected_video_width",
            "gripper_dimension",
        )
    )
    if observed_shape != expected_shape:
        raise ValueError("BC offline action/observation shape contract changed.")
    if (
        int(cfg.lora.rank) not in {16, 32}
        or float(cfg.lora.alpha) != 16.0
        or float(cfg.lora.dropout) != 0.0
        or list(cfg.lora.target_groups)
        != ["self_attention_qkvo", "cross_attention_qkvo", "ffn"]
        or not bool(cfg.lora.freeze_base)
        or not bool(cfg.lora.strict_target_discovery)
    ):
        raise ValueError("BC offline LoRA structure differs from training.")
    policy = str(cfg.runner.policy)
    if policy not in {"zero_lora", "bc_lora", "bc_training_checkpoint"}:
        raise ValueError(
            "BC offline policy must be zero_lora, bc_lora, or bc_training_checkpoint."
        )
    sidecar = cfg.runner.get("sidecar")
    sidecar_hash = cfg.runner.get("sidecar_sha256")
    has_sidecar = sidecar is not None and bool(str(sidecar).strip())
    has_hash = sidecar_hash is not None and bool(str(sidecar_hash).strip())
    checkpoint = cfg.runner.get("training_checkpoint")
    checkpoint_hash = cfg.runner.get("training_checkpoint_sha256")
    checkpoint_step = cfg.runner.get("training_checkpoint_step")
    has_checkpoint = checkpoint is not None and bool(str(checkpoint).strip())
    has_checkpoint_hash = checkpoint_hash is not None and bool(
        str(checkpoint_hash).strip()
    )
    if policy == "zero_lora" and (
        has_sidecar or has_hash or has_checkpoint or has_checkpoint_hash
    ):
        raise ValueError("zero_lora offline validation forbids trained artifacts.")
    if policy == "bc_lora":
        if not (has_sidecar and has_hash):
            raise ValueError("bc_lora offline validation requires sidecar and SHA256.")
        if has_checkpoint or has_checkpoint_hash:
            raise ValueError("bc_lora offline validation forbids a trainer checkpoint.")
    if policy == "bc_training_checkpoint":
        if has_sidecar or has_hash:
            raise ValueError(
                "bc_training_checkpoint offline validation forbids a sidecar."
            )
        if not (has_checkpoint and has_checkpoint_hash):
            raise ValueError("bc_training_checkpoint requires a checkpoint and SHA256.")
        if (
            isinstance(checkpoint_step, bool)
            or not isinstance(checkpoint_step, int)
            or checkpoint_step <= 0
        ):
            raise ValueError(
                "bc_training_checkpoint requires a positive expected step."
            )


def _validate_sidecar_extra(
    extra: Any,
    *,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(extra, Mapping):
        raise TypeError("BC sidecar metadata.extra must be a mapping.")
    required = {
        "bc_step",
        "bc_config_sha256",
        "validation_loss_action_bc",
        "statistics_sha256",
        "dataset_sha256",
        "text_cache_sha256",
    }
    if set(extra) != required:
        raise ValueError(f"BC sidecar provenance keys changed: {sorted(extra)}.")
    step = extra["bc_step"]
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("BC sidecar bc_step must be a positive integer.")
    config_hash = str(extra["bc_config_sha256"]).lower()
    if len(config_hash) != 64 or any(
        character not in "0123456789abcdef" for character in config_hash
    ):
        raise ValueError("BC sidecar config hash is malformed.")
    loss = extra["validation_loss_action_bc"]
    if (
        isinstance(loss, bool)
        or not isinstance(loss, (int, float))
        or not math.isfinite(float(loss))
        or float(loss) < 0
    ):
        raise ValueError("BC sidecar validation loss is invalid.")
    expected = {
        "statistics_sha256": contract["statistics_sha256"],
        "dataset_sha256": contract["dataset_sha256"],
        "text_cache_sha256": contract["text_cache_sha256"],
    }
    mismatches = {
        key: {"expected": value, "observed": extra.get(key)}
        for key, value in expected.items()
        if extra.get(key) != value
    }
    if mismatches:
        raise ValueError(f"BC sidecar data provenance mismatch: {mismatches}")
    return {
        "bc_step": step,
        "bc_config_sha256": config_hash,
        "validation_loss_action_bc": float(loss),
        **expected,
    }


def record_uncond_bc_offline_failure(
    cfg: DictConfig,
    error: BaseException,
    *,
    traceback_text: str,
) -> Path | None:
    """Preserve a compact failure only in a claimed offline directory."""

    if int(os.environ.get("RANK", "0")) != 0:
        return None
    output = Path(str(cfg.runner.output_dir)).expanduser().resolve()
    if not (output / OFFLINE_OUTPUT_MARKER).is_file():
        return None
    payload = {
        "schema": OFFLINE_FAILURE_SCHEMA,
        "status": "FAIL",
        "command": list(sys.argv),
        "policy": str(cfg.runner.get("policy")),
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback_text,
        "created_unix_seconds": time.time(),
    }
    path = output / "failure_manifest.json"
    _atomic_json(path, payload)
    return path


def run_uncond_bc_offline(cfg: DictConfig) -> dict[str, Any]:
    """Evaluate zero or trained UNCOND LoRA on every held-out BC window."""

    rank, world_size, local_rank, device = _distributed_context(
        collective_timeout_seconds=int(cfg.distributed.collective_timeout_seconds)
    )
    _validate_offline_config(cfg, world_size=world_size)
    _set_seed(int(cfg.seed), rank=rank, deterministic=True)
    output = Path(str(cfg.runner.output_dir)).expanduser().resolve()
    if rank == 0 and claim_uncond_bc_offline_output(cfg) != output:
        raise RuntimeError("BC offline output ownership resolved unexpectedly.")
    _barrier(world_size)
    if not (output / OFFLINE_OUTPUT_MARKER).is_file():
        raise FileNotFoundError("BC offline output marker was not synchronized.")

    resolved, launch_hash, contract_hash = _canonical_config(cfg)
    parent_sha = (
        _verify_sha256(
            str(cfg.parent.checkpoint),
            str(cfg.parent.checkpoint_sha256),
            label="FastWAM parent",
        )
        if rank == 0
        else None
    )
    stats_sha = (
        _verify_sha256(
            str(cfg.parent.statistics),
            str(cfg.parent.statistics_sha256),
            label="FastWAM statistics",
        )
        if rank == 0
        else None
    )
    objects = [parent_sha, stats_sha]
    if world_size > 1:
        torch.distributed.broadcast_object_list(objects, src=0)
    parent_sha, stats_sha = objects
    contract, provenance = _build_provenance(
        cfg,
        resolved_config=resolved,
        launch_config_sha256=launch_hash,
        contract_config_sha256=contract_hash,
        parent_sha256=parent_sha,
        stats_sha256=stats_sha,
        rank=rank,
        world_size=world_size,
    )

    actor = instantiate(cfg.model, model_dtype=torch.bfloat16, device=str(device))
    parent_load = load_strict_fastwam_parent(actor, str(cfg.parent.checkpoint))
    policy = FastWAMUncondBCPolicy(
        actor=actor,
        lora_config=RegimeLoRAConfig(**OmegaConf.to_container(cfg.lora, resolve=True)),
        config=FastWAMUncondBCConfig(
            **OmegaConf.to_container(cfg.bc_policy, resolve=True)
        ),
    ).to(device)
    future_prediction_calls = {"count": 0}

    def forbidden_future_prediction(*args, **kwargs):
        del args, kwargs
        future_prediction_calls["count"] += 1
        raise RuntimeError("BC offline evaluation entered future-video code.")

    actor._video_denoise_step_compiled = forbidden_future_prediction
    actor.training_loss_from_inputs = forbidden_future_prediction
    actor.video_expert.post_dit = forbidden_future_prediction
    frozen_versions = _frozen_versions(policy)

    sidecar_report = None
    training_checkpoint_report = None
    if str(cfg.runner.policy) == "bc_lora":
        sidecar_path = str(cfg.runner.sidecar)
        sidecar_sha = _verify_sha256(
            sidecar_path,
            str(cfg.runner.sidecar_sha256),
            label="BC LoRA sidecar",
        )
        metadata = policy.lora_adapter.load_sidecar(
            sidecar_path,
            expected_parent_checkpoint_sha256=parent_sha,
            strict=True,
        )
        extra = _validate_sidecar_extra(metadata.get("extra"), contract=contract)
        sidecar_report = {
            "path": str(Path(sidecar_path).expanduser().resolve()),
            "sha256": sidecar_sha,
            "schema": metadata["schema"],
            "rank": int(metadata["rank"]),
            "alpha": float(metadata["alpha"]),
            "target_groups": list(metadata["target_groups"]),
            "extra": extra,
        }
        if _zero_lora(policy):
            raise ValueError("Trained BC sidecar unexpectedly leaves LoRA at zero.")
    elif str(cfg.runner.policy) == "bc_training_checkpoint":
        checkpoint_path = str(cfg.runner.training_checkpoint)
        checkpoint_sha = _verify_sha256(
            checkpoint_path,
            str(cfg.runner.training_checkpoint_sha256),
            label="BC trainer checkpoint",
        )
        loaded = load_uncond_bc_adapter_checkpoint(
            checkpoint_path,
            adapter=policy.lora_adapter,
            expected_parent_checkpoint_sha256=parent_sha,
        )
        expected_step = int(cfg.runner.training_checkpoint_step)
        if int(loaded["global_step"]) != expected_step:
            raise ValueError(
                "BC trainer checkpoint step mismatch: "
                f"expected {expected_step}, got {loaded['global_step']}."
            )
        if int(loaded["trainer_state"]["nonzero_update_count"]) != expected_step:
            raise ValueError(
                "BC trainer checkpoint does not record one nonzero LoRA update "
                "per completed optimizer step."
            )
        checkpoint_contract = loaded["contract"]
        comparable_contract_keys = {
            "world_size",
            "parent_checkpoint_sha256",
            "statistics_sha256",
            "dataset_sha256",
            "text_cache_sha256",
            "lora",
            "bc_policy",
        }
        mismatches = {
            key: {
                "expected": contract.get(key),
                "observed": checkpoint_contract.get(key),
            }
            for key in sorted(comparable_contract_keys)
            if checkpoint_contract.get(key) != contract.get(key)
        }
        if mismatches:
            raise ValueError(
                f"BC trainer checkpoint scientific contract mismatch: {mismatches}."
            )
        if _zero_lora(policy):
            raise ValueError("BC trainer checkpoint unexpectedly leaves LoRA at zero.")
        training_checkpoint_report = {
            "path": str(Path(checkpoint_path).expanduser().resolve()),
            "sha256": checkpoint_sha,
            **loaded,
        }
    elif not _zero_lora(policy):
        raise RuntimeError("zero_lora offline baseline did not start at zero.")

    train_dataset, validation_dataset = _build_datasets(cfg)
    train_summary = _dataset_summary(train_dataset)
    validation_summary = _dataset_summary(validation_dataset)
    if train_summary["episodes"] != 1539 or validation_summary["episodes"] != 173:
        raise ValueError("BC offline episode split differs from 1539/173.")
    if validation_summary["windows"] != EXPECTED_VALIDATION_WINDOWS:
        raise ValueError(
            "BC offline validation window count mismatch: "
            f"{validation_summary['windows']} != {EXPECTED_VALIDATION_WINDOWS}."
        )
    _, validation_loader, _ = _build_loaders(
        cfg,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        rank=rank,
        world_size=world_size,
    )
    model: nn.Module = DistributedDataParallel(
        policy,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    started = time.time()
    validation = _validate(
        model,
        policy,
        validation_loader,
        cfg=cfg,
        world_size=world_size,
        device=device,
        expected_sample_count=len(validation_dataset),
    )
    _assert_frozen_versions(policy, frozen_versions)
    gradients_absent = all(
        parameter.grad is None for parameter in policy.lora_adapter.lora_parameters()
    )
    passed = (
        validation["sample_count"] == EXPECTED_VALIDATION_WINDOWS
        and math.isfinite(validation["loss_action_bc"])
        and future_prediction_calls["count"] == 0
        and gradients_absent
    )
    validated_sidecar_report = None
    if passed and str(cfg.runner.policy) == "bc_training_checkpoint" and rank == 0:
        checkpoint_extra = training_checkpoint_report["adapter_metadata"].get(
            "extra", {}
        )
        training_config_sha256 = str(
            checkpoint_extra.get("bc_config_sha256", "")
        ).lower()
        if len(training_config_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in training_config_sha256
        ):
            raise ValueError("BC trainer checkpoint config hash is malformed.")
        validated_sidecar_path = output / "validated_uncond_lora.pt"
        validated_extra = {
            "bc_step": int(training_checkpoint_report["global_step"]),
            "bc_config_sha256": training_config_sha256,
            "validation_loss_action_bc": float(validation["loss_action_bc"]),
            "statistics_sha256": contract["statistics_sha256"],
            "dataset_sha256": contract["dataset_sha256"],
            "text_cache_sha256": contract["text_cache_sha256"],
        }
        before_reload = {
            name: tensor.detach().cpu().clone()
            for name, tensor in policy.lora_adapter.lora_state_dict().items()
        }
        policy.lora_adapter.save_sidecar(
            validated_sidecar_path,
            parent_checkpoint_sha256=parent_sha,
            extra_metadata=validated_extra,
        )
        reloaded_metadata = policy.lora_adapter.load_sidecar(
            validated_sidecar_path,
            expected_parent_checkpoint_sha256=parent_sha,
            strict=True,
        )
        reloaded_state = policy.lora_adapter.lora_state_dict()
        if set(reloaded_state) != set(before_reload) or any(
            not torch.equal(before_reload[name], reloaded_state[name].detach().cpu())
            for name in before_reload
        ):
            raise RuntimeError("Validated BC sidecar strict reload changed LoRA state.")
        if reloaded_metadata.get("extra") != validated_extra:
            raise RuntimeError(
                "Validated BC sidecar metadata changed on strict reload."
            )
        validated_sidecar_report = {
            "path": str(validated_sidecar_path),
            "sha256": sha256_file(validated_sidecar_path),
            "strict_reload_exact": True,
            "extra": validated_extra,
        }
    manifest = {
        "schema": OFFLINE_EVAL_SCHEMA,
        "status": "PASS" if passed else "FAIL",
        "policy": str(cfg.runner.policy),
        "command": list(sys.argv),
        "contract": contract,
        "provenance": provenance,
        "parent_load": parent_load,
        "sidecar": sidecar_report,
        "training_checkpoint": training_checkpoint_report,
        "validated_sidecar": validated_sidecar_report,
        "train_dataset": train_summary,
        "validation_dataset": validation_summary,
        "validation": validation,
        "future_prediction_calls": future_prediction_calls["count"],
        "lora_gradients_absent": gradients_absent,
        "frozen_parameter_versions_unchanged": True,
        "contains_gate": False,
        "contains_critic": False,
        "contains_value_head": False,
        "elapsed_seconds": time.time() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    if rank == 0:
        _atomic_json(output / "run_manifest.json", manifest)
    _barrier(world_size)
    if not passed:
        raise RuntimeError("Full UNCOND BC offline validation failed acceptance.")
    return manifest


def failure_traceback() -> str:
    """Expose traceback formatting to the thin Hydra entrypoint."""

    return traceback.format_exc()
