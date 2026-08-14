"""Offline correct/zero/shuffled-memory evaluation for P1 BC artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset

from fastwam.p1_dino_bc import FastWAMP1DinoBCPolicy
from fastwam.p1_dino_bc_checkpoint import (
    inspect_p1_dino_bc_checkpoint,
    load_p1_dino_bc_checkpoint,
)
from fastwam.p1_dino_bc_runner import (
    audit_p1_assets,
    build_real_p1_policy,
    instantiate_p1_smoke_dataset,
    validate_p1_config,
)
from fastwam.uncond_bc import stateless_validation_flow_inputs

P1_MEMORY_MODES = {"correct", "zero", "shuffled"}


@torch.no_grad()
def evaluate_p1_memory_mode(
    policy: FastWAMP1DinoBCPolicy,
    batches: Iterable[Mapping[str, Any]],
    *,
    memory_mode: str,
    seed: int,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Evaluate one fixed checkpoint without training an intervention arm."""

    mode = str(memory_mode).strip().lower()
    if mode not in P1_MEMORY_MODES:
        raise ValueError(f"P1 memory mode must be one of {sorted(P1_MEMORY_MODES)}.")
    policy.eval()
    losses = []
    samples = 0
    dino_calls = {"count": 0}

    def count_dino(_module, _inputs, _outputs) -> None:
        dino_calls["count"] += 1

    hook = policy.visual_encoder.register_forward_hook(count_dino)
    try:
        for batch_index, batch in enumerate(batches):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            action = batch["action"]
            if mode == "shuffled" and action.shape[0] < 2:
                raise ValueError("Shuffled-memory evaluation requires batch size >= 2.")
            timestep, noise = stateless_validation_flow_inputs(
                sample_identities=[
                    f"offline-batch-{batch_index}-sample-{sample_index}"
                    for sample_index in range(action.shape[0])
                ],
                action_shape=tuple(action.shape),
                scheduler=policy.actor.train_action_scheduler,
                seed=int(seed),
                device=policy.device,
                dtype=policy.dtype,
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = policy(
                    batch,
                    timestep=timestep,
                    noise=noise,
                    memory_mode=mode,
                )
            loss = output["loss_action_bc"].detach().float()
            if not torch.isfinite(loss):
                raise RuntimeError("P1 offline memory loss is non-finite.")
            losses.append(float(loss.item()) * action.shape[0])
            samples += int(action.shape[0])
    finally:
        hook.remove()
    if samples == 0:
        raise ValueError("P1 offline evaluator received no samples.")
    return {
        "schema": "fastwam-p1-dino-bc-offline-memory-v1",
        "status": "PASS",
        "memory_mode": mode,
        "sample_count": samples,
        "batch_count": len(losses),
        "mean_loss_action_bc": sum(losses) / samples,
        "dinov3_forward_calls": dino_calls["count"],
        "one_dino_call_per_batch": dino_calls["count"] == len(losses),
    }


def evaluate_p1_checkpoint(
    cfg: DictConfig,
    *,
    checkpoint: str | Path,
    memory_mode: str,
    max_batches: int,
) -> dict[str, Any]:
    """Build the verified real runtime and evaluate one memory intervention."""

    validate_p1_config(cfg)
    assets = audit_p1_assets(cfg)
    if not torch.cuda.is_available():
        raise RuntimeError("P1 real offline evaluation requires CUDA.")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    policy, _ = build_real_p1_policy(cfg, device=device)
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or not isinstance(raw.get("contract"), Mapping):
        raise ValueError("P1 checkpoint has no strict evaluation contract.")
    load_p1_dino_bc_checkpoint(
        checkpoint,
        adapter=policy.lora_adapter,
        reader=policy.visual_reader,
        expected_parent_checkpoint_sha256=assets["parent_checkpoint_sha256"],
        expected_dinov3_weights_sha256=assets["dinov3_weights_sha256"],
        expected_contract=raw["contract"],
    )
    dataset = instantiate_p1_smoke_dataset(cfg)
    indices = [int(index) for index in cfg.data.tiny_window_indices]
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    result = evaluate_p1_memory_mode(
        policy,
        loader,
        memory_mode=memory_mode,
        seed=int(cfg.seed),
        max_batches=max_batches,
    )
    result["checkpoint"] = inspect_p1_dino_bc_checkpoint(checkpoint)
    return result


def write_p1_offline_result(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Write a compact deterministic JSON result."""

    Path(path).write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
