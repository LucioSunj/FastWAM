"""Dedicated DDP trainer for frozen-IDM UNCOND-LoRA action-only BC."""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import math
import os
import random
import re
import subprocess
import sys
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
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler

from fastwam.adapters import PolicyRegime, RegimeLoRAConfig, sha256_file
from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from fastwam.models.wan22.adaptive_action import CachedActionVelocity
from fastwam.uncond_bc import (
    FastWAMUncondBCConfig,
    FastWAMUncondBCPolicy,
    SampleIdentityDataset,
    cosine_warmup_multiplier,
    lora_gradient_norm,
    lora_update_norm,
    stateless_validation_flow_inputs,
)
from fastwam.uncond_bc_checkpoint import (
    capture_rng_state,
    inspect_uncond_bc_checkpoint,
    load_uncond_bc_checkpoint,
    restore_rng_state,
    save_uncond_bc_checkpoint,
)
from fastwam.utils import misc

BC_OUTPUT_MARKER = ".fastwam-uncond-bc-output-v1"


class DistributedEvalSampler(Sampler[int]):
    """Shard validation without padding or duplicate samples."""

    def __init__(self, dataset, *, rank: int, world_size: int) -> None:
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)
        if not 0 <= self.rank < self.world_size:
            raise ValueError("Invalid validation sampler rank/world size.")

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return max(0, (remaining + self.world_size - 1) // self.world_size)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_resolved_config(
    output_dir: Path,
    *,
    launch_hash: str,
    value: str,
) -> Path:
    """Preserve every launch config while maintaining a latest convenience file."""

    versioned = output_dir / f"resolved_config_{launch_hash[:12]}.yaml"
    if versioned.exists() and versioned.read_text(encoding="utf-8") != value:
        raise FileExistsError(f"Versioned resolved config collision: {versioned}")
    if not versioned.exists():
        _atomic_text(versioned, value)
    _atomic_text(output_dir / "resolved_config.yaml", value)
    return versioned


def _write_run_manifest(
    output_dir: Path,
    manifest: Mapping[str, Any],
    *,
    launch_hash: str,
) -> Path:
    """Preserve each completed launch and update the latest manifest atomically."""

    stage = str(manifest.get("stage", "unknown"))
    step = int(manifest.get("optimizer_steps", 0))
    versioned = output_dir / (
        f"run_manifest_{stage}_step_{step:06d}_{launch_hash[:12]}.json"
    )
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if versioned.exists() and versioned.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"Versioned run manifest collision: {versioned}")
    if not versioned.exists():
        _atomic_text(versioned, encoded)
    _atomic_text(output_dir / "run_manifest.json", encoded)
    return versioned


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_artifact(path: str | os.PathLike[str]) -> str:
    """Hash a file or a deterministic recursive directory stream."""

    root = Path(path).expanduser().resolve()
    if root.is_file():
        return sha256_file(root)
    if not root.is_dir():
        raise FileNotFoundError(f"BC provenance artifact does not exist: {root}")
    files = sorted(candidate for candidate in root.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"BC provenance directory is empty: {root}")
    digest = hashlib.sha256()
    for candidate in files:
        relative = candidate.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        with candidate.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def _distributed_context() -> tuple[int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("Real UNCOND BC training requires CUDA.")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", device_id=device)
    return rank, world_size, local_rank, device


def _broadcast_object(value: Any, *, rank: int, world_size: int) -> Any:
    if world_size == 1:
        return value
    values = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def _set_seed(seed: int, *, rank: int, deterministic: bool) -> None:
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(bool(deterministic))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)
    effective = int(seed) + int(rank)
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    torch.cuda.manual_seed_all(effective)


def _git_state(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(path), *args],
            text=True,
        ).strip()

    try:
        return {
            "head": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current") or "DETACHED",
            "dirty": bool(run("status", "--short", "--untracked-files=no")),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": repr(error)}


def _canonical_config(cfg: DictConfig) -> tuple[dict[str, Any], str, str]:
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(resolved, dict):
        raise TypeError("Resolved BC Hydra config must be a mapping.")
    launch_json = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    contract_config = json.loads(launch_json)
    contract_config.get("runner", {}).pop("resume", None)
    contract_config.get("runner", {}).pop("stop_after_steps", None)
    contract_config.get("runner", {}).pop("output_dir", None)
    contract_config.get("training", {}).pop("checkpoint_keep_last", None)
    contract_json = json.dumps(
        contract_config,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        resolved,
        _sha256_bytes(launch_json.encode()),
        _sha256_bytes(contract_json.encode()),
    )


def claim_uncond_bc_output(cfg: DictConfig) -> Path:
    """Claim a new BC output safely, or validate ownership for explicit resume."""

    output_dir = Path(str(cfg.runner.output_dir)).expanduser().resolve()
    if int(os.environ.get("RANK", "0")) != 0:
        return output_dir
    marker = output_dir / BC_OUTPUT_MARKER
    resume = cfg.runner.get("resume")
    if resume:
        if not output_dir.is_dir() or not marker.is_file():
            raise FileNotFoundError(
                "BC resume requires an existing trainer-owned output directory: "
                f"{output_dir}."
            )
        return output_dir
    if output_dir.exists():
        entries = {entry.name for entry in output_dir.iterdir()}
        if entries and entries != {BC_OUTPUT_MARKER}:
            raise FileExistsError(f"BC output directory is not empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    if not marker.is_file():
        _atomic_text(marker, "fastwam-uncond-bc-output-v1\n")
    return output_dir


def record_uncond_bc_failure(
    cfg: DictConfig,
    error: BaseException,
    *,
    traceback_text: str,
) -> Path | None:
    """Write a failure artifact only inside a trainer-owned output directory."""

    if int(os.environ.get("RANK", "0")) != 0:
        return None
    output_dir = Path(str(cfg.runner.output_dir)).expanduser().resolve()
    if not (output_dir / BC_OUTPUT_MARKER).is_file():
        return None
    try:
        resolved, launch_hash, contract_hash = _canonical_config(cfg)
    except Exception:  # noqa: BLE001 - preserve the original run failure.
        resolved = {"unresolved_config": OmegaConf.to_yaml(cfg, resolve=False)}
        launch_hash = None
        contract_hash = None
    root = Path(__file__).resolve().parents[2]
    device = {
        "cuda_available": torch.cuda.is_available(),
        "visible_device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        device["names"] = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
    payload = {
        "schema": "fastwam-uncond-bc-failure-manifest-v1",
        "status": "FAIL",
        "exit_status": 1,
        "stage": str(cfg.runner.get("stage")),
        "command": list(sys.argv),
        "resolved_config": resolved,
        "launch_resolved_config_sha256": launch_hash,
        "training_contract_sha256": contract_hash,
        "repositories": {
            "fastwam": _git_state(root),
            "outer": _git_state(root.parent),
            "rlinf": _git_state(root.parent / "RLinf"),
        },
        "declared_parent": {
            "path": str(cfg.parent.checkpoint),
            "sha256": str(cfg.parent.checkpoint_sha256),
        },
        "declared_statistics": {
            "path": str(cfg.parent.statistics),
            "sha256": str(cfg.parent.statistics_sha256),
        },
        "device": device,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback_text,
        "created_unix_seconds": time.time(),
    }
    path = output_dir / "failure_manifest.json"
    _atomic_json(path, payload)
    return path


def _verify_sha256(path: str, expected: str, *, label: str) -> str:
    expected = str(expected).strip().lower()
    if len(expected) != 64 or any(
        character not in "0123456789abcdef" for character in expected
    ):
        raise ValueError(
            f"{label} expected SHA256 must be 64 lowercase hex characters."
        )
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: expected {expected}, got {actual}.")
    return actual


def load_strict_fastwam_parent(actor: nn.Module, checkpoint: str) -> dict[str, Any]:
    """Strictly restore the frozen MoT/proprio parent before LoRA injection."""

    kwargs = {"map_location": "cpu", "weights_only": False}
    try:
        payload = torch.load(checkpoint, mmap=True, **kwargs)
    except TypeError:
        payload = torch.load(checkpoint, **kwargs)
    if not isinstance(payload, Mapping):
        raise TypeError("FastWAM parent checkpoint must be a mapping.")
    mot_state = payload.get("mot")
    if not isinstance(mot_state, Mapping):
        raise TypeError("FastWAM parent checkpoint must contain `mot` weights.")
    expected_mot = set(actor.mot.state_dict())
    actual_mot = set(mot_state)
    if expected_mot != actual_mot:
        raise ValueError(
            "FastWAM parent MoT key mismatch: "
            f"missing={sorted(expected_mot - actual_mot)[:8]}, "
            f"unexpected={sorted(actual_mot - expected_mot)[:8]}."
        )
    actor.mot.load_state_dict(mot_state, strict=True)
    proprio = getattr(actor, "proprio_encoder", None)
    proprio_state = payload.get("proprio_encoder")
    if proprio is not None:
        if not isinstance(proprio_state, Mapping):
            raise ValueError("FastWAM parent is missing `proprio_encoder` weights.")
        proprio.load_state_dict(proprio_state, strict=True)
    elif proprio_state is not None:
        raise ValueError("Parent has proprio weights but configured actor does not.")
    if any("lora_" in str(name) for name in mot_state):
        raise ValueError("Frozen FastWAM parent unexpectedly contains LoRA tensors.")
    return {
        "parent_step": payload.get("step"),
        "parent_torch_dtype": payload.get("torch_dtype"),
        "mot_tensor_count": len(mot_state),
        "proprio_tensor_count": len(proprio_state or {}),
    }


def _instantiate_bc_dataset(dataset_cfg: DictConfig, *, expected_seed: int):
    payload = OmegaConf.to_container(dataset_cfg, resolve=True)
    if not isinstance(payload, dict):
        raise TypeError("Resolved BC dataset config must be a mapping.")
    split_seed = int(payload.pop("split_seed", -1))
    if split_seed != int(expected_seed):
        raise ValueError(
            f"BC episode split seed mismatch: expected {expected_seed}, got {split_seed}."
        )
    default_seed = (
        inspect.signature(BaseLerobotDataset.__init__).parameters["seed"].default
    )
    if default_seed != split_seed:
        raise RuntimeError(
            "RobotVideoDataset no longer uses the approved implicit split seed: "
            f"expected default {split_seed}, observed {default_seed}."
        )
    return instantiate(OmegaConf.create(payload))


def _build_datasets(cfg: DictConfig):
    train = SampleIdentityDataset(
        _instantiate_bc_dataset(cfg.data.train, expected_seed=int(cfg.seed)),
        namespace="train-seed42",
    )
    validation = SampleIdentityDataset(
        _instantiate_bc_dataset(cfg.data.validation, expected_seed=int(cfg.seed)),
        namespace="validation-seed42",
    )
    return train, validation


def _dataset_summary(dataset: SampleIdentityDataset) -> dict[str, Any]:
    underlying = dataset.dataset
    lerobot = getattr(underlying, "lerobot_dataset", None)
    multi = getattr(lerobot, "multi_dataset", None)
    suites = [
        {
            "dataset_dir": str(name),
            "suite": Path(str(name)).name,
            "episodes": int(subset.num_episodes),
            "windows": int(subset.num_frames),
        }
        for name, subset in zip(
            getattr(multi, "ds_names", ()),
            getattr(multi, "_datasets", ()),
            strict=True,
        )
    ]
    return {
        "episodes": int(getattr(multi, "num_episodes", -1)),
        "windows": len(dataset),
        "suites": suites,
    }


def _build_loaders(
    cfg: DictConfig,
    *,
    train_dataset,
    validation_dataset,
    rank: int,
    world_size: int,
):
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=int(cfg.seed),
        drop_last=True,
    )
    validation_sampler = DistributedEvalSampler(
        validation_dataset,
        rank=rank,
        world_size=world_size,
    )
    common = {
        "batch_size": int(cfg.training.microbatch_size),
        "num_workers": int(cfg.data.num_workers),
        "pin_memory": True,
        "persistent_workers": bool(cfg.data.num_workers > 0),
    }
    if int(cfg.data.num_workers) > 0:
        common["prefetch_factor"] = int(cfg.data.prefetch_factor)
        multiprocessing_context = cfg.data.get("multiprocessing_context")
        if multiprocessing_context is not None:
            common["multiprocessing_context"] = str(multiprocessing_context)
    train_generator = torch.Generator().manual_seed(int(cfg.seed))
    validation_generator = torch.Generator().manual_seed(int(cfg.validation.seed))
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        drop_last=True,
        generator=train_generator,
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        sampler=validation_sampler,
        drop_last=False,
        generator=validation_generator,
        **common,
    )
    return train_loader, validation_loader, train_sampler


def _all_reduce(value: torch.Tensor, *, world_size: int) -> torch.Tensor:
    if world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


@torch.no_grad()
def _validate(
    model: nn.Module,
    policy: FastWAMUncondBCPolicy,
    loader: DataLoader,
    *,
    cfg: DictConfig,
    world_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    dimension_count = policy.config.action_dim
    bin_count = policy.config.timestep_bins
    accum = torch.zeros(4 + dimension_count + 2 * bin_count, device=device)
    # Layout: loss_sum, sample_count, valid_steps, batches, dimension sums,
    # bin mse sums, bin counts.
    loader.generator.manual_seed(int(cfg.validation.seed))
    for batch in loader:
        identities = list(batch["sample_identity"])
        action = batch["action"]
        timestep, noise = stateless_validation_flow_inputs(
            sample_identities=identities,
            action_shape=tuple(action.shape),
            scheduler=policy.actor.train_action_scheduler,
            seed=int(cfg.validation.seed),
            device=device,
            dtype=policy.dtype,
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch, timestep=timestep, noise=noise)
        batch_size = action.shape[0]
        valid_count = output["valid_action_count"].float()
        accum[0] += output["loss_action_bc"].float() * batch_size
        accum[1] += batch_size
        accum[2] += valid_count
        accum[3] += 1
        accum[4 : 4 + dimension_count] += (
            output["mse_per_dimension"].float() * valid_count
        )
        bin_start = 4 + dimension_count
        bin_counts = output["timestep_bin_count"].float()
        accum[bin_start : bin_start + bin_count] += (
            output["mse_by_timestep_bin"].float() * bin_counts
        )
        accum[bin_start + bin_count :] += bin_counts
    accum = _all_reduce(accum, world_size=world_size)
    sample_count = accum[1].clamp(min=1)
    valid_count = accum[2].clamp(min=1)
    dimensions = accum[4 : 4 + dimension_count] / valid_count
    bin_start = 4 + dimension_count
    bins = accum[bin_start : bin_start + bin_count]
    bin_counts = accum[bin_start + bin_count :]
    bins = bins / bin_counts.clamp(min=1)
    pose_indices = [
        index
        for index in range(dimension_count)
        if index != policy.config.gripper_dimension
    ]
    result = {
        "loss_action_bc": float((accum[0] / sample_count).item()),
        "mse_per_dimension": dimensions.cpu().tolist(),
        "mse_pose": float(dimensions[pose_indices].mean().item()),
        "mse_gripper": float(dimensions[policy.config.gripper_dimension].item()),
        "mse_by_timestep_bin": bins.cpu().tolist(),
        "timestep_bin_count": [int(value) for value in bin_counts.cpu().tolist()],
        "valid_action_count": int(accum[2].item()),
        "sample_count": int(accum[1].item()),
        "batch_count": int(accum[3].item()),
    }
    model.train()
    return result


def _gather_rng(*, world_size: int) -> list[Mapping[str, Any]]:
    local = capture_rng_state()
    if world_size == 1:
        return [local]
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, local)
    return gathered


def _trainer_state_payload(
    *,
    best_validation: float,
    best_step: int | None,
    epochs_without_improvement: int,
    nonzero_update_count: int,
) -> dict[str, Any]:
    return {
        "best_validation_loss_action_bc": (
            float(best_validation) if math.isfinite(best_validation) else None
        ),
        "best_step": best_step,
        "epochs_without_improvement": int(epochs_without_improvement),
        "nonzero_update_count": int(nonzero_update_count),
    }


_TRAINING_CHECKPOINT_NAME = re.compile(
    r"(?:step_[0-9]{6}|epoch_[0-9]{2}_step_[0-9]{6})[.]pt"
)


def _prune_training_checkpoints(
    checkpoints_dir: Path,
    *,
    keep_path: Path,
    keep_last: int,
) -> list[str]:
    """Remove older known trainer checkpoints after a validated replacement."""

    if int(keep_last) != 1:
        raise ValueError("UNCOND BC checkpoint retention must keep exactly one file.")
    directory = checkpoints_dir.expanduser().resolve()
    keep = keep_path.expanduser().resolve()
    if keep.parent != directory or not keep.is_file() or keep.is_symlink():
        raise ValueError(
            "Retention keep_path must be a regular direct checkpoint child."
        )
    known = sorted(
        path
        for path in directory.iterdir()
        if path.is_file()
        and not path.is_symlink()
        and _TRAINING_CHECKPOINT_NAME.fullmatch(path.name)
    )
    if keep not in known:
        raise ValueError("Retention keep_path is not a recognized trainer checkpoint.")
    pruned = []
    for path in known:
        if path != keep:
            path.unlink()
            pruned.append(path.name)
    return pruned


def _save_checkpoint(
    path: Path,
    *,
    policy: FastWAMUncondBCPolicy,
    optimizer,
    scheduler,
    scaler,
    global_step: int,
    epoch: int,
    sampler_offset: int,
    contract: Mapping[str, Any],
    provenance: Mapping[str, Any],
    trainer_state: Mapping[str, Any],
    parent_sha256: str,
    rank: int,
    world_size: int,
    checkpoint_keep_last: int,
) -> dict[str, Any] | None:
    rng_by_rank = _gather_rng(world_size=world_size)
    if rank == 0:
        save_uncond_bc_checkpoint(
            path,
            adapter=policy.lora_adapter,
            parent_checkpoint_sha256=parent_sha256,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            grad_scaler=scaler,
            global_step=global_step,
            epoch=epoch,
            sampler_offset=sampler_offset,
            rng_by_rank=rng_by_rank,
            contract=contract,
            provenance=provenance,
            trainer_state=trainer_state,
        )
        report = inspect_uncond_bc_checkpoint(path)
        if report.get("result") != "PASS":
            raise RuntimeError(
                "New UNCOND BC checkpoint failed inspection; older state was retained."
            )
        report = dict(report)
        report["checkpoint_retention"] = {
            "keep_last": int(checkpoint_keep_last),
            "kept": path.name,
            "pruned": _prune_training_checkpoints(
                path.parent,
                keep_path=path,
                keep_last=checkpoint_keep_last,
            ),
        }
    else:
        report = None
    _barrier(world_size)
    return report


def _snapshot_lora(policy: FastWAMUncondBCPolicy) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in policy.lora_adapter.named_lora_parameters()
    }


def _frozen_versions(policy: FastWAMUncondBCPolicy) -> dict[str, int]:
    lora_ids = {id(parameter) for parameter in policy.lora_adapter.lora_parameters()}
    return {
        name: parameter._version
        for name, parameter in policy.actor.named_parameters()
        if id(parameter) not in lora_ids
    }


def _assert_frozen_versions(
    policy: FastWAMUncondBCPolicy,
    expected: Mapping[str, int],
) -> None:
    observed = _frozen_versions(policy)
    changed = sorted(
        name for name in expected if observed.get(name) != expected.get(name)
    )
    if changed:
        raise RuntimeError(
            f"Frozen FastWAM parameters changed in-place: {changed[:16]}."
        )


def _build_provenance(
    cfg: DictConfig,
    *,
    resolved_config: Mapping[str, Any],
    launch_config_sha256: str,
    contract_config_sha256: str,
    parent_sha256: str,
    stats_sha256: str,
    rank: int,
    world_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_paths = [str(path) for path in cfg.provenance.dataset_paths]
    text_cache = str(cfg.provenance.text_cache_path)
    if rank == 0:
        dataset_hashes = {path: sha256_artifact(path) for path in dataset_paths}
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
        "resolved_config_sha256": contract_config_sha256,
        "world_size": world_size,
        "parent_checkpoint_sha256": parent_sha256,
        "statistics_sha256": stats_sha256,
        "dataset_sha256": dataset_hashes,
        "text_cache_sha256": text_cache_hash,
        "lora": OmegaConf.to_container(cfg.lora, resolve=True),
        "bc_policy": OmegaConf.to_container(cfg.bc_policy, resolve=True),
    }
    root = Path(__file__).resolve().parents[2]
    provenance = {
        "schema": "fastwam-uncond-bc-provenance-v1",
        "command": list(sys.argv),
        "resolved_config": dict(resolved_config),
        "launch_resolved_config_sha256": launch_config_sha256,
        "training_contract_sha256": contract_config_sha256,
        "repositories": {
            "fastwam": _git_state(root),
            "outer": _git_state(root.parent),
            "rlinf": _git_state(root.parent / "RLinf"),
        },
        "parent_checkpoint": {
            "path": str(cfg.parent.checkpoint),
            "sha256": parent_sha256,
        },
        "statistics": {"path": str(cfg.parent.statistics), "sha256": stats_sha256},
        "dataset_sha256": dataset_hashes,
        "text_cache": {"path": text_cache, "sha256": text_cache_hash},
        "world_size": world_size,
        "precision": "bfloat16",
        "eager": True,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    return contract, provenance


def _validate_training_config(cfg: DictConfig, *, world_size: int) -> None:
    stage = str(cfg.runner.stage)
    allowed_stages = {"bc0", "bc1", "bc2", "pilot", "formal"}
    if stage not in allowed_stages:
        raise ValueError(f"runner.stage must be one of {sorted(allowed_stages)}.")
    if int(cfg.seed) != 42 or int(cfg.validation.seed) != 42:
        raise ValueError("The approved UNCOND BC train/split/validation seed is 42.")
    if (
        int(cfg.data.train.split_seed) != 42
        or int(cfg.data.validation.split_seed) != 42
    ):
        raise ValueError("Both BC dataset episode splits must use seed 42.")
    if str(cfg.precision).lower() != "bf16":
        raise ValueError("UNCOND BC production preset requires BF16.")
    if not bool(cfg.training.deterministic_algorithms):
        raise ValueError("UNCOND BC requires deterministic CUDA algorithms.")
    if not bool(cfg.training.stateless_flow_inputs):
        raise ValueError("UNCOND BC requires stateless per-sample flow inputs.")
    if bool(cfg.get("compile", False)):
        raise ValueError("UNCOND BC calibration/training is eager-only.")

    expected_parent = (
        "/XYFS02/HDD_POOL/nju_shklu/nju_shklu_1/"
        "When-will-inference-time-prediction-beneficial-/"
        "fastwam-idm-wan-robot-init/"
        "fastwam-idm-libero-wan-robot-init-step_021700.pt"
    )
    if str(cfg.parent.checkpoint) != expected_parent or (
        str(cfg.parent.checkpoint_sha256)
        != "e979511a2d7a1310009496c6b2f06957171bba28b96aac0d513992c6ed21ca5a"
    ):
        raise ValueError("UNCOND BC parent path/hash differs from the approved parent.")
    expected_stats = str(Path(expected_parent).with_name("dataset_stats.json"))
    if str(cfg.parent.statistics) != expected_stats or (
        str(cfg.parent.statistics_sha256)
        != "30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638"
    ):
        raise ValueError("UNCOND BC statistics path/hash differs from the parent.")
    if (
        int(cfg.data.expected_train_episodes) != 1539
        or int(cfg.data.expected_validation_episodes) != 173
    ):
        raise ValueError("UNCOND BC episode split counts changed.")
    if (
        int(cfg.data.expected_source_episodes) != 1712
        or int(cfg.data.expected_source_transitions) != 277713
    ):
        raise ValueError("UNCOND BC source-corpus counts changed.")
    dataset_suite_names = [
        Path(str(path)).name for path in cfg.provenance.dataset_paths
    ]
    if dataset_suite_names != [
        "libero_spatial_no_noops_lerobot",
        "libero_object_no_noops_lerobot",
        "libero_goal_no_noops_lerobot",
        "libero_10_no_noops_lerobot",
    ]:
        raise ValueError("UNCOND BC must use the approved four LIBERO suites in order.")
    for split in (cfg.data.train, cfg.data.validation):
        if str(split.pretrained_norm_stats) != expected_stats:
            raise ValueError("Every BC split must reuse the parent statistics.")
        if bool(split.processor.use_stepwise_action_norm) or (
            str(split.processor.norm_default_mode) != "min/max"
        ):
            raise ValueError("BC action/state normalization contract changed.")
    if not bool(cfg.data.train.is_training_set) or bool(
        cfg.data.validation.is_training_set
    ):
        raise ValueError("BC train/validation episode split roles changed.")

    expected_targets = [
        "self_attention_qkvo",
        "cross_attention_qkvo",
        "ffn",
    ]
    if (
        int(cfg.lora.rank) not in {16, 32}
        or float(cfg.lora.alpha) != 16.0
        or float(cfg.lora.dropout) != 0.0
        or list(cfg.lora.target_groups) != expected_targets
        or not bool(cfg.lora.freeze_base)
        or not bool(cfg.lora.strict_target_discovery)
    ):
        raise ValueError("UNCOND BC LoRA structure differs from the approved contract.")
    expected_bc_shape = {
        "action_horizon": 32,
        "action_dim": 7,
        "proprio_dim": 8,
        "expected_video_frames": 9,
        "expected_video_height": 224,
        "expected_video_width": 448,
        "gripper_dimension": 6,
    }
    observed_bc_shape = {key: int(cfg.bc_policy[key]) for key in expected_bc_shape}
    if observed_bc_shape != expected_bc_shape:
        raise ValueError(
            f"UNCOND BC tensor-shape contract changed: {observed_bc_shape}."
        )
    if (
        int(cfg.model.video_dit_config.num_layers) != 30
        or int(cfg.model.action_dit_config.num_layers) != 30
    ):
        raise ValueError("UNCOND BC requires the 30-layer parent architecture.")

    lora_rank = int(cfg.lora.rank)
    expected_world_size = 8 if lora_rank == 32 else 4
    expected_accumulation = {
        microbatch: 128 // (expected_world_size * microbatch)
        for microbatch in (1, 2, 4, 8)
    }
    microbatch = int(cfg.training.microbatch_size)
    if microbatch not in expected_accumulation:
        raise ValueError("BC microbatch must be one of 1, 2, 4, or 8 per GPU.")
    if stage in {"bc0", "bc1"}:
        if world_size != 1:
            raise ValueError(f"{stage} must run on exactly one GPU.")
    elif world_size != expected_world_size:
        raise ValueError(
            f"{stage} with LoRA rank {lora_rank} must run on exactly "
            f"{expected_world_size} GPUs."
        )
    if stage in {"bc2", "pilot", "formal"}:
        expected = expected_accumulation[microbatch]
        actual_accumulation = int(cfg.training.gradient_accumulation_steps)
        if actual_accumulation != expected:
            raise ValueError(
                f"{expected_world_size}-GPU microbatch {microbatch} requires "
                f"accumulation {expected}."
            )
        actual_global_batch = microbatch * world_size * actual_accumulation
        if int(cfg.training.global_batch_size) != actual_global_batch:
            raise ValueError(
                "Effective global batch mismatch: expected "
                f"{cfg.training.global_batch_size}, got {actual_global_batch}."
            )
        if int(cfg.training.global_batch_size) != 128:
            raise ValueError(
                "The approved UNCOND BC experiment requires global batch 128."
            )
    if stage == "bc1" and not bool(cfg.runner.single_gpu_diagnostic):
        raise ValueError("BC1 requires runner.single_gpu_diagnostic=true.")
    if stage != "bc1" and bool(cfg.runner.single_gpu_diagnostic):
        raise ValueError("single_gpu_diagnostic is reserved for BC1.")

    if tuple(float(value) for value in cfg.optimizer.betas) != (0.9, 0.95):
        raise ValueError("UNCOND BC AdamW betas must be (0.9, 0.95).")
    if float(cfg.optimizer.eps) != 1e-8 or float(cfg.optimizer.weight_decay) != 0.01:
        raise ValueError("UNCOND BC AdamW eps/weight_decay contract changed.")
    if float(cfg.optimizer.gradient_clip) != 1.0:
        raise ValueError("UNCOND BC gradient clip must be 1.0.")
    if (
        float(cfg.optimizer.warmup_fraction) != 0.05
        or float(cfg.optimizer.minimum_lr_ratio) != 0.01
    ):
        raise ValueError("UNCOND BC warmup/cosine floor contract changed.")
    if float(cfg.optimizer.learning_rate) not in {3e-5, 1e-4, 3e-4}:
        raise ValueError("UNCOND BC LR must be one of the three preregistered values.")
    if (
        int(cfg.training.max_epochs) != 10
        or int(cfg.training.minimum_epochs) != 3
        or int(cfg.training.early_stopping_patience) != 2
        or int(cfg.training.save_every_steps) != 500
        or int(cfg.training.checkpoint_keep_last) != 1
    ):
        raise ValueError(
            "UNCOND BC epoch/early-stop/checkpoint retention schedule changed."
        )

    max_steps = cfg.training.get("max_steps")
    stop_after = cfg.runner.get("stop_after_steps")
    if stage in {"bc1", "bc2"}:
        if int(max_steps or -1) != 2:
            raise ValueError(f"{stage} requires training.max_steps=2.")
        if int(stop_after or -1) not in {1, 2}:
            raise ValueError(f"{stage} requires stop_after_steps 1 or 2.")
    elif stage == "pilot":
        if int(max_steps or -1) != 1000 or stop_after is not None:
            raise ValueError("Each BC LR pilot must run exactly 1000 optimizer steps.")
    elif stage in {"bc0", "formal"} and (
        max_steps is not None or stop_after is not None
    ):
        raise ValueError(f"{stage} does not accept step truncation.")


def _zero_lora(policy: FastWAMUncondBCPolicy) -> bool:
    return all(
        torch.count_nonzero(parameter).item() == 0
        for name, parameter in policy.lora_adapter.named_lora_parameters()
        if name.endswith(".lora_B")
    )


def _strict_reload_best_sidecar(
    policy: FastWAMUncondBCPolicy,
    path: Path,
    *,
    parent_sha256: str,
    expected_extra: Mapping[str, Any],
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Best UNCOND BC sidecar is missing: {path}")
    before = {
        name: value.detach().cpu().clone()
        for name, value in policy.lora_adapter.lora_state_dict().items()
    }
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    expected_state = payload.get("state_dict") if isinstance(payload, Mapping) else None
    if not isinstance(expected_state, Mapping):
        raise TypeError("Best UNCOND BC sidecar has no tensor state mapping.")
    metadata = policy.lora_adapter.load_sidecar(
        path,
        expected_parent_checkpoint_sha256=parent_sha256,
        strict=True,
    )
    observed_extra = metadata.get("extra")
    if observed_extra != dict(expected_extra):
        raise ValueError(
            "Best UNCOND BC sidecar provenance mismatch: "
            f"expected={dict(expected_extra)}, observed={observed_extra}."
        )
    loaded = policy.lora_adapter.lora_state_dict()
    load_mismatches = sorted(
        name
        for name, value in loaded.items()
        if name not in expected_state
        or not torch.equal(value, expected_state[name].to(dtype=value.dtype))
    )
    policy.lora_adapter.load_lora_state_dict(before, strict=True)
    restored = policy.lora_adapter.lora_state_dict()
    restore_mismatches = sorted(
        name for name, value in before.items() if not torch.equal(value, restored[name])
    )
    if load_mismatches or restore_mismatches:
        raise RuntimeError(
            "Best-sidecar strict reload round trip failed: "
            f"load={load_mismatches[:16]}, restore={restore_mismatches[:16]}."
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "strict_reload": True,
        "tensor_exact": True,
        "current_state_restored": True,
        "schema": metadata["schema"],
        "bc_step": int(observed_extra["bc_step"]),
        "bc_config_sha256": observed_extra["bc_config_sha256"],
    }


@torch.no_grad()
def _bc0_parity_and_action_report(
    policy: FastWAMUncondBCPolicy,
    batch: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    identities = list(batch["sample_identity"])
    action = batch["action"]
    timestep, noise = stateless_validation_flow_inputs(
        sample_identities=identities,
        action_shape=tuple(action.shape),
        scheduler=policy.actor.train_action_scheduler,
        seed=int(seed),
        device=policy.device,
        dtype=policy.dtype,
    )
    condition = policy.prepare_action_condition(batch)
    normalized_action = action.to(
        device=policy.device,
        dtype=policy.dtype,
        non_blocking=True,
    )
    noisy_action = policy.actor.train_action_scheduler.add_noise(
        normalized_action,
        noise,
        timestep,
    )
    common = {
        "action_expert": policy.actor.action_expert,
        "mot": policy.actor.mot,
        "condition": condition,
        "regime_context": policy.lora_adapter.regime_context,
        "capture_gate_kv": False,
    }
    idm_velocity = CachedActionVelocity(
        **common,
        regime=PolicyRegime.IDM,
    )(noisy_action, timestep).velocity
    uncond_velocity = CachedActionVelocity(
        **common,
        regime=PolicyRegime.UNCOND,
    )(noisy_action, timestep).velocity
    delta = (idm_velocity.float() - uncond_velocity.float()).abs()
    exact = bool(torch.equal(idm_velocity, uncond_velocity))

    padding = batch.get("action_is_pad")
    valid = (
        torch.ones(action.shape[:2], dtype=torch.bool)
        if padding is None
        else ~padding.to(dtype=torch.bool, device="cpu")
    )
    action_cpu = action.detach().to(device="cpu", dtype=torch.float32)
    per_dimension = []
    for dimension in range(action.shape[2]):
        values = action_cpu[:, :, dimension][valid]
        finite = torch.isfinite(values)
        if not bool(finite.any()):
            raise ValueError(
                f"BC0 normalized action dimension {dimension} has no finite values."
            )
        finite_values = values[finite]
        per_dimension.append(
            {
                "dimension": dimension,
                "min": float(finite_values.min().item()),
                "max": float(finite_values.max().item()),
                "finite_count": int(finite.sum().item()),
                "nonfinite_count": int((~finite).sum().item()),
                "valid_count": int(values.numel()),
            }
        )
    return {
        "zero_lora_idm_uncond_exact": exact,
        "zero_lora_idm_uncond_max_abs": float(delta.max().item()),
        "velocity_shape": list(idm_velocity.shape),
        "velocity_dtype": str(idm_velocity.dtype),
        "normalized_action": {
            "shape": list(action.shape),
            "dtype": str(action.dtype),
            "gripper_dimension": policy.config.gripper_dimension,
            "per_dimension": per_dimension,
        },
        "timestep_sha256": _sha256_bytes(
            timestep.detach().float().cpu().numpy().tobytes()
        ),
        "noise_sha256": _sha256_bytes(noise.detach().float().cpu().numpy().tobytes()),
    }


def run_uncond_bc(cfg: DictConfig) -> dict[str, Any]:
    """Run BC0 or the isolated action-only DDP training loop."""

    rank, world_size, local_rank, device = _distributed_context()
    _validate_training_config(cfg, world_size=world_size)
    _set_seed(
        int(cfg.seed),
        rank=rank,
        deterministic=bool(cfg.training.deterministic_algorithms),
    )
    output_dir = Path(str(cfg.runner.output_dir)).expanduser().resolve()
    resume = cfg.runner.get("resume")
    if rank == 0:
        claimed = claim_uncond_bc_output(cfg)
        if claimed != output_dir:
            raise RuntimeError("BC output ownership resolved to an unexpected path.")
    _barrier(world_size)
    if not (output_dir / BC_OUTPUT_MARKER).is_file():
        raise FileNotFoundError("BC output ownership marker was not synchronized.")
    misc.register_work_dir(output_dir)

    resolved, launch_hash, contract_hash = _canonical_config(cfg)
    if rank == 0:
        _write_resolved_config(
            output_dir,
            launch_hash=launch_hash,
            value=OmegaConf.to_yaml(cfg, resolve=True),
        )
    parent_sha256 = (
        _verify_sha256(
            str(cfg.parent.checkpoint),
            str(cfg.parent.checkpoint_sha256),
            label="FastWAM parent",
        )
        if rank == 0
        else None
    )
    stats_sha256 = (
        _verify_sha256(
            str(cfg.parent.statistics),
            str(cfg.parent.statistics_sha256),
            label="FastWAM statistics",
        )
        if rank == 0
        else None
    )
    parent_sha256 = _broadcast_object(parent_sha256, rank=rank, world_size=world_size)
    stats_sha256 = _broadcast_object(stats_sha256, rank=rank, world_size=world_size)
    contract, provenance = _build_provenance(
        cfg,
        resolved_config=resolved,
        launch_config_sha256=launch_hash,
        contract_config_sha256=contract_hash,
        parent_sha256=parent_sha256,
        stats_sha256=stats_sha256,
        rank=rank,
        world_size=world_size,
    )

    actor = instantiate(
        cfg.model,
        model_dtype=torch.bfloat16,
        device=str(device),
    )
    parent_load = load_strict_fastwam_parent(actor, str(cfg.parent.checkpoint))
    lora_payload = OmegaConf.to_container(cfg.lora, resolve=True)
    bc_payload = OmegaConf.to_container(cfg.bc_policy, resolve=True)
    policy = FastWAMUncondBCPolicy(
        actor=actor,
        lora_config=RegimeLoRAConfig(**lora_payload),
        config=FastWAMUncondBCConfig(**bc_payload),
    ).to(device)
    future_prediction_calls = {"count": 0}

    def forbidden_future_prediction(*args, **kwargs):
        del args, kwargs
        future_prediction_calls["count"] += 1
        raise RuntimeError("Action-only BC attempted a forbidden future-video path.")

    actor._video_denoise_step_compiled = forbidden_future_prediction
    actor.training_loss_from_inputs = forbidden_future_prediction
    actor.video_expert.post_dit = forbidden_future_prediction
    train_dataset, validation_dataset = _build_datasets(cfg)
    train_summary = _dataset_summary(train_dataset)
    validation_summary = _dataset_summary(validation_dataset)
    expected_episodes = {
        "train": int(cfg.data.expected_train_episodes),
        "validation": int(cfg.data.expected_validation_episodes),
    }
    if train_summary["episodes"] != expected_episodes["train"] or (
        validation_summary["episodes"] != expected_episodes["validation"]
    ):
        raise ValueError(
            "LIBERO episode split mismatch: "
            f"train={train_summary}, validation={validation_summary}, "
            f"expected={expected_episodes}."
        )
    source_episodes = train_summary["episodes"] + validation_summary["episodes"]
    source_windows = train_summary["windows"] + validation_summary["windows"]
    if source_episodes != int(cfg.data.expected_source_episodes) or (
        source_windows != int(cfg.data.expected_source_transitions)
    ):
        raise ValueError(
            "LIBERO source corpus mismatch: "
            f"episodes={source_episodes}, transitions={source_windows}, expected="
            f"({cfg.data.expected_source_episodes}, "
            f"{cfg.data.expected_source_transitions})."
        )
    train_loader, validation_loader, train_sampler = _build_loaders(
        cfg,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        rank=rank,
        world_size=world_size,
    )

    if world_size > 1:
        model: nn.Module = DistributedDataParallel(
            policy,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    else:
        model = policy
    frozen_versions = _frozen_versions(policy)
    manifest: dict[str, Any] = {
        "schema": "fastwam-uncond-bc-run-manifest-v1",
        "stage": str(cfg.runner.stage),
        "status": "RUNNING",
        "command": list(sys.argv),
        "contract": contract,
        "provenance": provenance,
        "parent_load": parent_load,
        "train_dataset": train_summary,
        "validation_dataset": validation_summary,
        "device": {
            "type": "cuda",
            "index": local_rank,
            "name": torch.cuda.get_device_name(local_rank),
            "capability": list(torch.cuda.get_device_capability(local_rank)),
        },
        "trainable_parameter_names": list(policy.trainable_parameter_names()),
        "zero_lora_at_start": _zero_lora(policy),
        "future_prediction_calls": future_prediction_calls["count"],
        "contains_gate": False,
        "contains_critic": False,
        "contains_value_head": False,
    }

    if str(cfg.runner.stage) == "bc0":
        batch = next(iter(train_loader))
        model.eval()
        identities = list(batch["sample_identity"])
        timestep, noise = stateless_validation_flow_inputs(
            sample_identities=identities,
            action_shape=tuple(batch["action"].shape),
            scheduler=policy.actor.train_action_scheduler,
            seed=int(cfg.validation.seed),
            device=device,
            dtype=policy.dtype,
        )
        with (
            torch.no_grad(),
            torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
            ),
        ):
            parity = _bc0_parity_and_action_report(
                policy,
                batch,
                seed=int(cfg.validation.seed),
            )
            output = model(batch, timestep=timestep, noise=noise)
        finite = bool(torch.isfinite(output["loss_action_bc"]).item())
        action_finite = all(
            item["nonfinite_count"] == 0
            for item in parity["normalized_action"]["per_dimension"]
        )
        future_prediction_free = future_prediction_calls["count"] == 0
        passed = (
            finite
            and bool(manifest["zero_lora_at_start"])
            and bool(parity["zero_lora_idm_uncond_exact"])
            and action_finite
            and future_prediction_free
        )
        _assert_frozen_versions(policy, frozen_versions)
        manifest.update(
            {
                "status": "PASS" if passed else "FAIL",
                "loss_action_bc": float(output["loss_action_bc"].item()),
                "finite": finite,
                "bc0_parity": parity,
                "normalized_action_finite": action_finite,
                "frozen_parameter_versions_unchanged": True,
                "optimizer_steps": 0,
                "future_prediction_calls": future_prediction_calls["count"],
            }
        )
        if rank == 0:
            _write_run_manifest(
                output_dir,
                manifest,
                launch_hash=launch_hash,
            )
        _barrier(world_size)
        if not passed:
            raise RuntimeError("BC0 numerical, parity, or isolation acceptance failed.")
        return manifest

    parameters = list(policy.lora_adapter.lora_parameters())
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
    configured_max_steps = cfg.training.get("max_steps")
    total_steps = (
        int(configured_max_steps)
        if configured_max_steps is not None
        else updates_per_epoch * int(cfg.training.max_epochs)
    )
    requested_stop = cfg.runner.get("stop_after_steps")
    stop_after_steps = (
        int(requested_stop) if requested_stop is not None else total_steps
    )
    if not 0 < stop_after_steps <= total_steps:
        raise ValueError(
            "runner.stop_after_steps must lie in (0, total optimizer steps], got "
            f"{stop_after_steps} for total_steps={total_steps}."
        )
    diagnostic_stage = str(cfg.runner.stage) in {"bc1", "bc2"}
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
    global_step = 0
    start_epoch = 0
    sampler_offset = 0
    best_validation = math.inf
    best_step: int | None = None
    epochs_without_improvement = 0
    nonzero_update_count = 0
    if resume:
        payload = load_uncond_bc_checkpoint(
            str(resume),
            adapter=policy.lora_adapter,
            expected_parent_checkpoint_sha256=parent_sha256,
            expected_contract=contract,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            grad_scaler=scaler,
        )
        if len(payload["rng_by_rank"]) != world_size:
            raise ValueError("BC checkpoint RNG rank count differs from world size.")
        restore_rng_state(payload["rng_by_rank"][rank])
        global_step = int(payload["global_step"])
        start_epoch = int(payload["epoch"])
        sampler_offset = int(payload["sampler_offset"])
        restored_trainer_state = payload["trainer_state"]
        restored_best = restored_trainer_state["best_validation_loss_action_bc"]
        best_validation = math.inf if restored_best is None else float(restored_best)
        best_step = restored_trainer_state["best_step"]
        epochs_without_improvement = int(
            restored_trainer_state["epochs_without_improvement"]
        )
        nonzero_update_count = int(restored_trainer_state["nonzero_update_count"])
    if global_step >= stop_after_steps:
        raise ValueError(
            f"BC resume step {global_step} already reached stop {stop_after_steps}."
        )

    checkpoints_dir = output_dir / "checkpoints"
    metrics_path = output_dir / "metrics.jsonl"
    last_checkpoint_report = None
    stop = False
    optimizer.zero_grad(set_to_none=True)
    started_at = time.time()
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
                accumulation_index % int(cfg.training.gradient_accumulation_steps) == 0
            )
            sync_context = (
                contextlib.nullcontext()
                if sync_update or world_size == 1
                else model.no_sync()
            )
            training_identities = [
                f"train-epoch-{epoch}:{identity}"
                for identity in batch["sample_identity"]
            ]
            timestep, noise = stateless_validation_flow_inputs(
                sample_identities=training_identities,
                action_shape=tuple(batch["action"].shape),
                scheduler=policy.actor.train_action_scheduler,
                seed=int(cfg.seed),
                device=device,
                dtype=policy.dtype,
            )
            identity_digest.update(
                json.dumps(training_identities, separators=(",", ":")).encode()
            )
            timestep_digest.update(
                timestep.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
            )
            noise_digest.update(
                noise.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
            )
            with (
                sync_context,
                torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ),
            ):
                output = model(batch, timestep=timestep, noise=noise)
                loss = output["loss_action_bc"] / int(
                    cfg.training.gradient_accumulation_steps
                )
                loss.backward()
            sampler_offset = batch_index + 1
            if not sync_update:
                continue
            if not torch.isfinite(output["loss_action_bc"]):
                raise FloatingPointError("Non-finite UNCOND action BC loss.")
            gradient_norm = lora_gradient_norm(policy.lora_adapter)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Non-finite UNCOND LoRA gradient norm.")
            before = _snapshot_lora(policy)
            torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=float(cfg.optimizer.gradient_clip),
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update_norm = lora_update_norm(policy.lora_adapter, before)
            if not torch.isfinite(update_norm):
                raise FloatingPointError("Non-finite UNCOND LoRA update norm.")
            update_norm_value = float(update_norm.item())
            if update_norm_value > 0.0:
                nonzero_update_count += 1
            global_step += 1
            _assert_frozen_versions(policy, frozen_versions)
            record = {
                "global_step": global_step,
                "epoch": epoch,
                "sampler_offset": sampler_offset,
                "loss_action_bc": float(output["loss_action_bc"].item()),
                "mse_per_dimension": output["mse_per_dimension"]
                .detach()
                .cpu()
                .tolist(),
                "mse_pose": float(output["mse_pose"].item()),
                "mse_gripper": float(output["mse_gripper"].item()),
                "mse_by_timestep_bin": output["mse_by_timestep_bin"]
                .detach()
                .cpu()
                .tolist(),
                "valid_action_count": int(output["valid_action_count"].item()),
                "lora_gradient_norm": float(gradient_norm.item()),
                "lora_update_norm": update_norm_value,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
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
            if global_step % int(cfg.training.save_every_steps) == 0 or (
                global_step >= stop_after_steps
            ):
                last_checkpoint_report = _save_checkpoint(
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
                    parent_sha256=parent_sha256,
                    rank=rank,
                    world_size=world_size,
                    checkpoint_keep_last=int(cfg.training.checkpoint_keep_last),
                )
            if global_step >= stop_after_steps:
                stop = True
                break

        if stop and diagnostic_stage:
            break
        validation = _validate(
            model,
            policy,
            validation_loader,
            cfg=cfg,
            world_size=world_size,
            device=device,
        )
        improved = validation["loss_action_bc"] < best_validation
        if improved:
            best_validation = validation["loss_action_bc"]
            best_step = global_step
            epochs_without_improvement = 0
            if rank == 0:
                policy.lora_adapter.save_sidecar(
                    output_dir / "best_uncond_lora.pt",
                    parent_checkpoint_sha256=parent_sha256,
                    extra_metadata={
                        "bc_step": global_step,
                        "bc_config_sha256": contract_hash,
                        "validation_loss_action_bc": best_validation,
                        "statistics_sha256": stats_sha256,
                        "dataset_sha256": contract["dataset_sha256"],
                        "text_cache_sha256": contract["text_cache_sha256"],
                    },
                )
        else:
            epochs_without_improvement += 1
        epoch_record = {
            "epoch": epoch,
            "global_step": global_step,
            "validation": validation,
            "improved": improved,
            "best_validation_loss_action_bc": best_validation,
            "epochs_without_improvement": epochs_without_improvement,
        }
        if rank == 0:
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(epoch_record, sort_keys=True) + "\n")
        next_epoch = epoch + 1
        sampler_offset = 0
        last_checkpoint_report = _save_checkpoint(
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
            parent_sha256=parent_sha256,
            rank=rank,
            world_size=world_size,
            checkpoint_keep_last=int(cfg.training.checkpoint_keep_last),
        )
        if stop:
            break
        if next_epoch >= int(cfg.training.minimum_epochs) and (
            epochs_without_improvement >= int(cfg.training.early_stopping_patience)
        ):
            break

    _assert_frozen_versions(policy, frozen_versions)
    stage = str(cfg.runner.stage)
    best_validation_value = (
        float(best_validation) if math.isfinite(best_validation) else None
    )
    best_sidecar_report = None
    if stage in {"pilot", "formal"}:
        if best_step is None or best_validation_value is None:
            raise RuntimeError(
                "Validated BC training produced no finite best checkpoint."
            )
        expected_extra = {
            "bc_step": best_step,
            "bc_config_sha256": contract_hash,
            "validation_loss_action_bc": best_validation_value,
            "statistics_sha256": stats_sha256,
            "dataset_sha256": contract["dataset_sha256"],
            "text_cache_sha256": contract["text_cache_sha256"],
        }
        best_sidecar_report = _strict_reload_best_sidecar(
            policy,
            output_dir / "best_uncond_lora.pt",
            parent_sha256=parent_sha256,
            expected_extra=expected_extra,
        )

    future_prediction_free = future_prediction_calls["count"] == 0
    expected_steps_reached = (
        global_step == stop_after_steps
        if stage in {"bc1", "bc2", "pilot"}
        else global_step > 0
    )
    passed = (
        expected_steps_reached
        and nonzero_update_count > 0
        and future_prediction_free
        and (stage not in {"pilot", "formal"} or best_sidecar_report is not None)
    )
    manifest.update(
        {
            "status": "PASS" if passed else "FAIL",
            "completion": (
                "CONTROLLED_DIAGNOSTIC_STOP"
                if diagnostic_stage
                else "TRAINING_COMPLETE"
            ),
            "optimizer_steps": global_step,
            "stop_after_steps": stop_after_steps,
            "nonzero_update_count": nonzero_update_count,
            "future_prediction_calls": future_prediction_calls["count"],
            "best_step": best_step,
            "best_validation_loss_action_bc": best_validation_value,
            "trainer_state": _trainer_state_payload(
                best_validation=best_validation,
                best_step=best_step,
                epochs_without_improvement=epochs_without_improvement,
                nonzero_update_count=nonzero_update_count,
            ),
            "frozen_parameter_versions_unchanged": True,
            "last_checkpoint": last_checkpoint_report,
            "elapsed_seconds": time.time() - started_at,
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
            "best_sidecar": best_sidecar_report,
        }
    )
    if rank == 0:
        _write_run_manifest(
            output_dir,
            manifest,
            launch_hash=launch_hash,
        )
    _barrier(world_size)
    if not passed:
        raise RuntimeError(
            "UNCOND BC training acceptance failed; see run_manifest.json."
        )
    return manifest
