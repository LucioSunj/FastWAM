"""Single-GPU runner for the preregistered modality-dropout BC pilot."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset

from fastwam.modality_dropout_bc import (
    MODALITY_DROPOUT_PILOT_SCHEMA,
    RANDOM_PATCH_BANK_SEED,
    aggregate_dino_diagnostics,
    baseline_plateau_decision,
    fixed_gaussian_patch_memory,
    forced_modality_keep_mask,
    resolve_pilot_arm,
    sample_modality_keep_mask,
    summarize_heldout_losses,
)
from fastwam.models.wan22.visual_sidecar import DinoContributionDiagnosticsCollector
from fastwam.p1_dino_bc import build_p1_optimizer
from fastwam.p1_dino_bc_runner import audit_p1_assets, build_real_p1_policy
from fastwam.p1_visual_bc_full_checkpoint import (
    inspect_p1_visual_bc_full_checkpoint,
    save_p1_visual_bc_full_checkpoint,
)
from fastwam.uncond_bc import (
    cosine_warmup_multiplier,
    stateless_validation_flow_inputs,
)
from fastwam.uncond_bc_checkpoint import capture_rng_state
from fastwam.uncond_bc_trainer import (
    _atomic_json,
    _build_datasets,
    _build_loaders,
    _canonical_config,
    _dataset_summary,
    _git_state,
    _set_seed,
    sha256_artifact,
)

PILOT_OUTPUT_MARKER = ".fastwam-modality-dropout-bc-pilot-v1"
HELDOUT_CONDITIONS = ("clean", "wan_drop", "dino_drop", "both_drop")


def _validate_config(cfg: DictConfig) -> None:
    arm = resolve_pilot_arm(str(cfg.runner.arm))
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("The exploratory modality-dropout pilot is single-GPU only.")
    if not torch.cuda.is_available():
        raise RuntimeError("The real modality-dropout pilot requires a CUDA GPU.")
    if int(cfg.seed) != 42 or int(cfg.validation.seed) != 42:
        raise ValueError("Pilot train and held-out seeds must remain 42.")
    if (
        int(cfg.data.train.split_seed) != 42
        or int(cfg.data.validation.split_seed) != 42
    ):
        raise ValueError("Pilot episode split seeds must remain 42.")
    if list(cfg.provenance.dataset_paths) != [
        str(cfg.data.dataset_info).rsplit("/meta/", 1)[0]
    ]:
        raise ValueError("Pilot must use exactly the declared single LIBERO suite.")
    if str(cfg.precision).lower() != "bf16" or bool(cfg.get("compile", False)):
        raise ValueError("Pilot requires eager BF16 execution.")
    if int(cfg.training.microbatch_size) != 32:
        raise ValueError("Pilot microbatch size must remain 32.")
    if int(cfg.training.gradient_accumulation_steps) != 1:
        raise ValueError("Pilot does not support gradient accumulation.")
    if int(cfg.training.global_batch_size) != 32:
        raise ValueError("Pilot global batch size must remain 32.")
    if int(cfg.training.max_steps) != 3000:
        raise ValueError("Pilot maximum calibration horizon must remain 3000 updates.")
    if int(cfg.pilot.eval_interval_steps) != 500:
        raise ValueError("Pilot evaluations must remain on the 500-step grid.")
    if int(cfg.pilot.heldout_windows) != 256:
        raise ValueError("Pilot held-out set must contain exactly 256 windows.")
    if int(cfg.pilot.random_patch_seed) != RANDOM_PATCH_BANK_SEED:
        raise ValueError("Random-patch arm seed must remain 42.")
    if arm.name == "A":
        if cfg.runner.get("baseline_endpoint") is not None:
            raise ValueError(
                "Baseline calibration cannot consume an endpoint artifact."
            )
    elif cfg.runner.get("baseline_endpoint") is None:
        raise ValueError("Non-baseline arms require A's frozen endpoint artifact.")
    if str(cfg.p1.get("lineage", "")) != "visual_v2":
        raise ValueError("This pilot implements only the audited V2 spatial reader.")


def _claim_output(cfg: DictConfig) -> Path:
    output = Path(str(cfg.runner.output_dir)).expanduser().resolve()
    marker = output / PILOT_OUTPUT_MARKER
    if output.exists():
        entries = {entry.name for entry in output.iterdir()}
        if entries and entries != {PILOT_OUTPUT_MARKER}:
            raise FileExistsError(f"Pilot output directory is not empty: {output}.")
    else:
        output.mkdir(parents=True, exist_ok=False)
    marker.write_text(PILOT_OUTPUT_MARKER[1:] + "\n", encoding="utf-8")
    return output


def _tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


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
        raise RuntimeError(f"Pilot changed frozen parent/DINO weights: {changed[:16]}.")


def _frozen_parameter_sha256(policy) -> str:
    """Hash every non-trainable tensor, including FastWAM and DINO parents."""

    trainable = {
        id(parameter)
        for family in policy.parameter_families().values()
        for parameter in family
    }
    state = {
        name: parameter.detach()
        for name, parameter in policy.named_parameters()
        if id(parameter) not in trainable
    }
    return _tensor_state_sha256(state)


def _heldout_indices(dataset_length: int, *, count: int, seed: int) -> list[int]:
    if dataset_length < count:
        raise ValueError(
            f"Validation split has {dataset_length} windows; {count} are required."
        )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randperm(dataset_length, generator=generator)[:count].tolist()


def _fixed_random_override(condition, *, seed: int):
    if condition.visual is None:
        raise ValueError("Random-patch arm requires prepared visual memory.")
    return fixed_gaussian_patch_memory(condition.visual.memory, seed=seed)


@torch.no_grad()
def _evaluate_heldout(
    policy,
    loader: DataLoader,
    *,
    arm,
    step: int,
    cfg: DictConfig,
) -> tuple[dict[str, Any], dict[str, list[float]], dict[str, Any] | None]:
    policy.eval()
    losses: dict[str, list[float]] = {name: [] for name in HELDOUT_CONDITIONS}
    collector = DinoContributionDiagnosticsCollector()
    random_metadata: dict[str, Any] | None = None
    for batch in loader:
        identities = list(batch["sample_identity"])
        action_shape = tuple(batch["action"].shape)
        timestep, noise = stateless_validation_flow_inputs(
            sample_identities=identities,
            action_shape=action_shape,
            scheduler=policy.actor.train_action_scheduler,
            seed=int(cfg.validation.seed),
            device=policy.device,
            dtype=policy.dtype,
        )
        condition = policy.prepare_action_condition(batch)
        memory_override = None
        if arm.dino_input == "fixed_gaussian":
            memory_override, metadata = _fixed_random_override(
                condition,
                seed=int(cfg.pilot.random_patch_seed),
            )
            if random_metadata is None:
                random_metadata = metadata
            elif metadata != random_metadata:
                raise RuntimeError("Runtime random-patch bank metadata changed.")
        for name in HELDOUT_CONDITIONS:
            keep = forced_modality_keep_mask(
                name,
                batch_size=action_shape[0],
                device=policy.device,
            )
            context = (
                policy.visual_reader.capture_diagnostics(collector)
                if name == "clean"
                else torch.no_grad()
            )
            with context, torch.autocast("cuda", dtype=torch.bfloat16):
                output = policy.loss_from_prepared_condition(
                    batch,
                    condition=condition,
                    timestep=timestep,
                    noise=noise,
                    memory_override=memory_override,
                    return_prediction=True,
                    modality_keep_mask=keep,
                )
            losses[name].extend(
                float(value)
                for value in output["loss_action_bc_per_sample"].detach().cpu()
            )
    summary = summarize_heldout_losses(losses)
    summary.update(
        schema="fastwam-modality-dropout-heldout-v1",
        arm=arm.name,
        step=int(step),
        diagnostics=aggregate_dino_diagnostics(collector.records),
    )
    policy.train()
    return summary, losses, random_metadata


def _load_endpoint(path: str | os.PathLike[str]) -> int:
    candidate = Path(path).expanduser().resolve()
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if payload.get("schema") != "fastwam-modality-dropout-endpoint-v1":
        raise ValueError("Baseline endpoint artifact has an unsupported schema.")
    if payload.get("arm") != "A" or payload.get("status") != "PLATFORMED":
        raise ValueError("Non-baseline arm requires a platformed A endpoint.")
    endpoint = int(payload["endpoint_step"])
    if endpoint not in range(1000, 3001, 500):
        raise ValueError("Baseline endpoint lies outside the preregistered grid.")
    return endpoint


def _build_contract_and_provenance(
    cfg: DictConfig,
    *,
    resolved: Mapping[str, Any],
    launch_hash: str,
    contract_hash: str,
    assets: Mapping[str, Any],
    policy,
    parent_load: Mapping[str, Any],
    arm,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_path = str(cfg.provenance.dataset_paths[0])
    dataset_hash = sha256_artifact(Path(dataset_path) / "meta")
    text_cache = str(cfg.provenance.text_cache_path)
    text_hash = sha256_artifact(text_cache)
    contract = {
        "resolved_config_sha256": contract_hash,
        "world_size": 1,
        "parent_checkpoint_sha256": assets["parent_checkpoint_sha256"],
        "statistics_sha256": assets["statistics_sha256"],
        "memory_contract_sha256": policy.expected_memory_contract,
        "reader_contract_sha256": policy.visual_reader.reader_contract_sha256,
        "dataset_metadata_sha256": {str(Path(dataset_path) / "meta"): dataset_hash},
        "text_cache_sha256": text_hash,
        "lora": OmegaConf.to_container(cfg.lora, resolve=True),
        "reader": OmegaConf.to_container(cfg.p1.reader, resolve=True),
        "bc_policy": OmegaConf.to_container(cfg.bc_policy, resolve=True),
        "visual_backbone": dict(assets["visual_backbone"]),
        "pilot_schema": MODALITY_DROPOUT_PILOT_SCHEMA,
        "arm": arm.__dict__,
    }
    root = Path(__file__).resolve().parents[2]
    provenance = {
        "schema": "fastwam-modality-dropout-bc-provenance-v1",
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
        "dataset_metadata_sha256": contract["dataset_metadata_sha256"],
        "text_cache": {"path": text_cache, "sha256": text_hash},
        "world_size": 1,
        "precision": "bfloat16",
        "eager": True,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "pilot_arm": arm.__dict__,
    }
    return contract, provenance


def _save_checkpoint(
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
    assets: Mapping[str, Any],
    best_validation: float,
    best_step: int | None,
    nonzero_update_count: int,
) -> dict[str, Any]:
    save_p1_visual_bc_full_checkpoint(
        path,
        adapter=policy.lora_adapter,
        reader=policy.visual_reader,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        grad_scaler=scaler,
        global_step=global_step,
        epoch=epoch,
        sampler_offset=sampler_offset,
        rng_by_rank=[capture_rng_state()],
        parent_checkpoint_sha256=assets["parent_checkpoint_sha256"],
        visual_backbone=assets["visual_backbone"],
        memory_contract_sha256=policy.expected_memory_contract,
        contract=contract,
        provenance=provenance,
        trainer_state={
            "best_validation_loss_action_bc": best_validation,
            "best_step": best_step,
            "epochs_without_improvement": 0,
            "nonzero_update_count": nonzero_update_count,
        },
    )
    return inspect_p1_visual_bc_full_checkpoint(path)


def run_modality_dropout_bc_pilot(cfg: DictConfig) -> dict[str, Any]:
    """Run one arm; A calibrates K and every other arm consumes that frozen K."""

    _validate_config(cfg)
    arm = resolve_pilot_arm(str(cfg.runner.arm))
    output_dir = _claim_output(cfg)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    _set_seed(int(cfg.seed), rank=0, deterministic=True)
    assets = audit_p1_assets(cfg)
    train_dataset, validation_dataset = _build_datasets(cfg)
    train_loader, _, train_sampler = _build_loaders(
        cfg,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        rank=0,
        world_size=1,
    )
    heldout_indices = _heldout_indices(
        len(validation_dataset),
        count=int(cfg.pilot.heldout_windows),
        seed=int(cfg.validation.seed),
    )
    heldout_loader = DataLoader(
        Subset(validation_dataset, heldout_indices),
        batch_size=int(cfg.training.microbatch_size),
        shuffle=False,
        num_workers=int(cfg.data.num_workers),
        pin_memory=True,
        persistent_workers=bool(int(cfg.data.num_workers) > 0),
    )
    policy, parent_load = build_real_p1_policy(cfg, device=device)
    policy.audit_parameter_ownership()
    policy.train()
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
    total_horizon = int(cfg.training.max_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_warmup_multiplier(
            step,
            total_steps=total_horizon,
            warmup_fraction=float(cfg.optimizer.warmup_fraction),
            minimum_ratio=float(cfg.optimizer.minimum_lr_ratio),
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    frozen_versions = _frozen_versions(policy)
    resolved, launch_hash, contract_hash = _canonical_config(cfg)
    contract, provenance = _build_contract_and_provenance(
        cfg,
        resolved=resolved,
        launch_hash=launch_hash,
        contract_hash=contract_hash,
        assets=assets,
        policy=policy,
        parent_load=parent_load,
        arm=arm,
    )
    initial_trainable_hashes = {
        "lora": _tensor_state_sha256(policy.lora_adapter.lora_state_dict()),
        "reader": _tensor_state_sha256(
            policy.visual_reader.export_trainable_state()["state"]
        ),
    }
    initial_frozen_hash = _frozen_parameter_sha256(policy)
    target_step = (
        total_horizon
        if arm.name == "A"
        else _load_endpoint(str(cfg.runner.baseline_endpoint))
    )
    evaluations: list[dict[str, Any]] = []
    raw_history: list[dict[str, Any]] = []
    random_patch_metadata = None
    best_validation = math.inf
    best_step = None
    nonzero_updates = 0
    global_step = 0
    epoch = 0
    sampler_offset = 0

    def evaluate() -> dict[str, Any]:
        nonlocal random_patch_metadata, best_validation, best_step
        summary, raw_losses, metadata = _evaluate_heldout(
            policy,
            heldout_loader,
            arm=arm,
            step=global_step,
            cfg=cfg,
        )
        if metadata is not None:
            if random_patch_metadata is None:
                random_patch_metadata = metadata
            elif metadata != random_patch_metadata:
                raise RuntimeError("Random-patch bank changed between evaluations.")
        if summary["loss"]["clean"] < best_validation:
            best_validation = float(summary["loss"]["clean"])
            best_step = global_step
        payload = {**summary, "losses": raw_losses}
        evaluations.append(summary)
        raw_history.append({"step": global_step, "losses": raw_losses})
        _atomic_json(output_dir / f"heldout_step_{global_step:06d}.json", payload)
        return summary

    evaluate()
    endpoint_decision = None
    stop = False
    start_time = time.time()
    while global_step < target_step and not stop:
        train_sampler.set_epoch(epoch)
        for batch_index, batch in enumerate(train_loader):
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
            keep = sample_modality_keep_mask(
                sample_identities=identities,
                p_wan=arm.p_wan,
                p_dino=arm.p_dino,
                seed=int(cfg.seed),
                step=global_step,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                condition = policy.prepare_action_condition(
                    batch,
                    modality_keep_mask=keep,
                )
                memory_override = None
                if arm.dino_input == "fixed_gaussian":
                    memory_override, metadata = _fixed_random_override(
                        condition,
                        seed=int(cfg.pilot.random_patch_seed),
                    )
                    if random_patch_metadata is None:
                        random_patch_metadata = metadata
                    elif metadata != random_patch_metadata:
                        raise RuntimeError("Random-patch bank changed during training.")
                result = policy.loss_from_prepared_condition(
                    batch,
                    condition=condition,
                    timestep=timestep,
                    noise=noise,
                    memory_override=memory_override,
                )
                loss = result["loss_action_bc"]
            loss.backward()
            trainable = [
                parameter
                for family in policy.parameter_families().values()
                for parameter in family
            ]
            if any(parameter.grad is None for parameter in trainable):
                raise RuntimeError("A pilot optimizer parameter received no gradient.")
            if any(
                not bool(torch.isfinite(parameter.grad).all().item())
                for parameter in trainable
            ):
                raise FloatingPointError("Pilot produced a non-finite gradient.")
            before = [parameter.detach().clone() for parameter in trainable]
            torch.nn.utils.clip_grad_norm_(
                trainable,
                max_norm=float(cfg.optimizer.gradient_clip),
            )
            optimizer.step()
            scheduler.step()
            if any(
                not torch.equal(previous, parameter.detach())
                for previous, parameter in zip(before, trainable, strict=True)
            ):
                nonzero_updates += 1
            global_step += 1
            sampler_offset = batch_index + 1
            _assert_frozen_versions(policy, frozen_versions)

            if global_step % int(cfg.pilot.eval_interval_steps) == 0:
                evaluate()
                checkpoint = output_dir / "checkpoints" / f"step_{global_step:06d}.pt"
                report = _save_checkpoint(
                    checkpoint,
                    policy=policy,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    global_step=global_step,
                    epoch=epoch,
                    sampler_offset=sampler_offset,
                    contract=contract,
                    provenance=provenance,
                    assets=assets,
                    best_validation=best_validation,
                    best_step=best_step,
                    nonzero_update_count=nonzero_updates,
                )
                _atomic_json(output_dir / "checkpoint_inspection.json", report)
                for older in sorted((output_dir / "checkpoints").glob("step_*.pt")):
                    if older != checkpoint:
                        older.unlink()
                if arm.name == "A" and global_step >= 1000:
                    endpoint_decision = baseline_plateau_decision(
                        raw_history,
                        draws=int(cfg.pilot.bootstrap_draws),
                        seed=int(cfg.validation.seed),
                    )
                    if endpoint_decision["platform"]:
                        target_step = global_step
                        stop = True
                if global_step >= target_step:
                    stop = True
            if stop:
                break
        if not stop:
            epoch += 1
            sampler_offset = 0
            if epoch >= int(cfg.training.max_epochs):
                raise RuntimeError("Pilot exhausted max_epochs before target updates.")

    if arm.name == "A":
        if endpoint_decision is None:
            endpoint_decision = baseline_plateau_decision(
                raw_history,
                draws=int(cfg.pilot.bootstrap_draws),
                seed=int(cfg.validation.seed),
            )
        endpoint_payload = {
            "schema": "fastwam-modality-dropout-endpoint-v1",
            "arm": "A",
            "status": (
                "PLATFORMED"
                if endpoint_decision["platform"]
                else "EVIDENCE_INSUFFICIENT"
            ),
            "endpoint_step": endpoint_decision["endpoint"],
            "decision": endpoint_decision,
        }
        _atomic_json(output_dir / "endpoint_selection.json", endpoint_payload)
    else:
        endpoint_payload = {
            "schema": "fastwam-modality-dropout-endpoint-v1",
            "arm": arm.name,
            "status": "FROZEN_FROM_A",
            "endpoint_step": target_step,
            "baseline_endpoint": str(Path(str(cfg.runner.baseline_endpoint)).resolve()),
        }

    final_frozen_hash = _frozen_parameter_sha256(policy)
    if final_frozen_hash != initial_frozen_hash:
        raise RuntimeError("Pilot frozen parent/DINO hash changed.")

    completed = global_step == target_step and (
        arm.name != "A" or endpoint_payload["status"] == "PLATFORMED"
    )
    result = {
        "schema": "fastwam-modality-dropout-arm-result-v1",
        "status": "COMPLETE" if completed else "EVIDENCE_INSUFFICIENT",
        "arm": arm.__dict__,
        "endpoint": endpoint_payload,
        "global_step": global_step,
        "initial_trainable_sha256": initial_trainable_hashes,
        "initial_frozen_parent_and_dino_sha256": initial_frozen_hash,
        "final_frozen_parent_and_dino_sha256": final_frozen_hash,
        "random_patch_bank": random_patch_metadata,
        "heldout_indices": heldout_indices,
        "heldout_split_contract": "episode-disjoint-from-training",
        "evaluations": evaluations,
        "dataset": {
            "train": _dataset_summary(train_dataset),
            "validation": _dataset_summary(validation_dataset),
        },
        "frozen_parameter_versions_unchanged": True,
        "nonzero_update_count": nonzero_updates,
        "elapsed_seconds": time.time() - start_time,
        "rollout": "NOT-RUN",
        "ood": {
            "status": "NOT-RUN",
            "reason": "background/lighting/noise assets and hooks remain unconfirmed",
        },
        "language_canary": "NOT-RUN",
        "decision": "EVIDENCE_INSUFFICIENT",
    }
    _atomic_json(output_dir / "arm_result.json", result)
    return result
