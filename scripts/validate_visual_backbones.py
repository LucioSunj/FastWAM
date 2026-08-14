#!/usr/bin/env python3
"""Strictly load every registered local visual backbone and audit patch output."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from fastwam.adapters import sha256_file
from fastwam.models.wan22.visual_backbone import (
    VisualBackboneAssetSpec,
    build_frozen_visual_encoder,
)
from fastwam.models.wan22.visual_contracts import (
    PreparedVisualCameraBatch,
    contract_sha256,
)

PRESETS = (
    "dinov3_vits16_224",
    "dinov3_vitb16_224",
    "dinov3_vitl16_224",
    "lingbot_small_224",
    "lingbot_small_512",
    "lingbot_base_224",
    "lingbot_base_512",
    "lingbot_large_224",
    "lingbot_large_512",
)
SCHEMA = "fastwam-visual-backbone-real-asset-validation-v2"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _asset(config_dir: Path, preset_name: str) -> VisualBackboneAssetSpec:
    config_path = config_dir / f"{preset_name}.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    payload = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"Visual preset must be a mapping: {config_path}")
    return VisualBackboneAssetSpec.from_mapping(payload)


def _input(asset: VisualBackboneAssetSpec, device: torch.device):
    size = asset.input_size
    pixels = (
        torch.arange(3 * size * size, dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
        .reshape(1, 1, 3, size, size)
        .to(device)
    )
    input_hash = contract_sha256(
        {
            "schema": "fastwam-visual-real-asset-probe-input-v2",
            "input_size": size,
            "pattern": "arange_mod_251",
        }
    )
    return PreparedVisualCameraBatch(
        pixels=pixels,
        camera_ids=("main",),
        camera_valid_mask=torch.ones((1, 1), dtype=torch.bool, device=device),
        input_size=size,
        input_contract_sha256=input_hash,
        source_resolution=torch.tensor(
            [[[size, size]]],
            dtype=torch.int32,
            device=device,
        ),
    )


def _validate_one(
    config_dir: Path,
    preset_name: str,
    device: torch.device,
) -> dict[str, Any]:
    asset = _asset(config_dir, preset_name)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    encoder = build_frozen_visual_encoder(asset, device=device)
    loaded_seconds = time.perf_counter() - started
    cameras = _input(asset, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    memory = encoder(cameras)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - started
    tokens = memory.tokens.float()
    expected = (1, 1, asset.patch_count, asset.preset.native_dim)
    if tuple(tokens.shape) != expected:
        raise ValueError(f"{preset_name} output {tuple(tokens.shape)} != {expected}.")
    if not bool(torch.isfinite(tokens).all().item()):
        raise ValueError(f"{preset_name} emitted non-finite patches.")
    row_mean_max_abs = float(tokens.mean(dim=-1).abs().max().item())
    row_variance_mean = float(tokens.var(dim=-1, unbiased=False).mean().item())
    row_norm_min = float(torch.linalg.vector_norm(tokens, dim=-1).min().item())
    # ``x_norm_patchtokens`` is the official post-norm feature key. Its learned
    # affine scale and bias are checkpoint-dependent, so neither zero mean nor
    # L2 unit length is part of the public contract.
    # Reject only collapsed/exploded rows here; the strict output-key and state
    # mapping checks above establish that the registered norm was executed.
    if not 1.0e-6 <= row_variance_mean <= 100.0 or row_norm_min <= 0.0:
        raise ValueError(
            f"{preset_name} x_norm_patchtokens normalization audit failed: "
            f"variance={row_variance_mean}, min_norm={row_norm_min}."
        )
    result = {
        "preset": preset_name,
        "family": asset.family,
        "variant": asset.variant,
        "input_size": asset.input_size,
        "native_dim": asset.preset.native_dim,
        "depth": asset.preset.depth,
        "patch_grid": list(asset.grid),
        "patch_count": asset.patch_count,
        "output_shape": list(tokens.shape),
        "output_dtype": str(memory.tokens.dtype),
        "finite": True,
        "x_norm_patchtokens_row_mean_max_abs": row_mean_max_abs,
        "x_norm_patchtokens_row_variance_mean": row_variance_mean,
        "x_norm_patchtokens_row_norm_min": row_norm_min,
        "weights_path": str(asset.weights_path),
        "weights_sha256": sha256_file(asset.weights_path),
        "source_revision": asset.source_revision,
        "asset_contract_sha256": asset.asset_contract_sha256,
        "memory_contract_sha256": memory.memory_contract_sha256,
        "encoder_training": encoder.training,
        "all_parameters_frozen": all(
            not parameter.requires_grad for parameter in encoder.parameters()
        ),
        "load_seconds": loaded_seconds,
        "forward_seconds": forward_seconds,
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "peak_cuda_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda"
            else None
        ),
    }
    if result["encoder_training"] or not result["all_parameters_frozen"]:
        raise ValueError(f"{preset_name} encoder is not permanently frozen/eval.")
    del tokens, memory, cameras, encoder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=project_root / "configs/visual_backbone",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preset", action="append", choices=PRESETS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = tuple(args.preset or PRESETS)
    if len(set(selected)) != len(selected):
        raise ValueError("Validation presets must be unique.")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    records = [
        _validate_one(args.config_dir.resolve(), preset_name, device)
        for preset_name in selected
    ]
    manifest_path = (
        Path("/home/amax/data0/Checkpoints/visual-backbones")
        / "visual_backbones_manifest.json"
    )
    payload = {
        "schema": SCHEMA,
        "status": "PASS",
        "created_unix_seconds": time.time(),
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "asset_manifest": str(manifest_path),
        "asset_manifest_sha256": sha256_file(manifest_path),
        "preset_count": len(records),
        "records": records,
    }
    _atomic_json(args.output.expanduser().resolve(), payload)
    print(json.dumps({"status": "PASS", "preset_count": len(records)}))


if __name__ == "__main__":
    main()
