"""Compare continuous and interrupted/resumed contribution-v2 canaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fastwam.p1_dino_bc_full_checkpoint import (
    P1_DINO_BC_FULL_CHECKPOINT_V2_SCHEMA,
    inspect_p1_dino_bc_full_checkpoint_v2,
)
from fastwam.uncond_bc_trainer import _atomic_json


def _latest_checkpoint(root: Path) -> Path:
    values = sorted((root / "checkpoints").glob("*.pt"))
    if len(values) != 1:
        raise ValueError(f"Expected one retained checkpoint in {root}, got {values}.")
    return values[0]


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape)).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _assert_exact(first: Any, second: Any, *, path: str) -> None:
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        if not isinstance(first, np.ndarray) or not isinstance(second, np.ndarray):
            raise AssertionError(f"{path}: ndarray/type mismatch.")
        if first.dtype != second.dtype or first.shape != second.shape:
            raise AssertionError(f"{path}: ndarray metadata differs.")
        if not np.array_equal(first, second, equal_nan=True):
            raise AssertionError(f"{path}: ndarray values differ.")
        return
    if isinstance(first, torch.Tensor) or isinstance(second, torch.Tensor):
        if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
            raise AssertionError(f"{path}: tensor/type mismatch.")
        if not torch.equal(first, second):
            raise AssertionError(
                f"{path}: tensor mismatch {_tensor_sha256(first)} != "
                f"{_tensor_sha256(second)}."
            )
        return
    if isinstance(first, Mapping) or isinstance(second, Mapping):
        if not isinstance(first, Mapping) or not isinstance(second, Mapping):
            raise AssertionError(f"{path}: mapping/type mismatch.")
        if set(first) != set(second):
            raise AssertionError(f"{path}: mapping keys differ.")
        for key in sorted(first, key=str):
            _assert_exact(first[key], second[key], path=f"{path}.{key}")
        return
    if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        if not isinstance(first, (list, tuple)) or not isinstance(
            second, (list, tuple)
        ):
            raise AssertionError(f"{path}: sequence/type mismatch.")
        if len(first) != len(second):
            raise AssertionError(f"{path}: sequence lengths differ.")
        for index, (left, right) in enumerate(zip(first, second, strict=True)):
            _assert_exact(left, right, path=f"{path}.{index}")
        return
    if first != second:
        raise AssertionError(f"{path}: {first!r} != {second!r}.")


def _metric_trace(path: Path) -> list[dict[str, Any]]:
    values = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [value for value in values if "loss_action_bc" in value]


def compare_resume(
    continuous_dir: Path,
    resumed_dir: Path,
    output: Path,
) -> dict[str, Any]:
    continuous_path = _latest_checkpoint(continuous_dir)
    resumed_path = _latest_checkpoint(resumed_dir)
    continuous = torch.load(continuous_path, map_location="cpu", weights_only=False)
    resumed = torch.load(resumed_path, map_location="cpu", weights_only=False)
    if (
        continuous.get("schema") != P1_DINO_BC_FULL_CHECKPOINT_V2_SCHEMA
        or resumed.get("schema") != P1_DINO_BC_FULL_CHECKPOINT_V2_SCHEMA
    ):
        raise ValueError("Resume comparison requires two contribution-v2 checkpoints.")
    if continuous["global_step"] != 20 or resumed["global_step"] != 20:
        raise ValueError("Resume comparison requires two complete 20-update runs.")

    compared_fields: Sequence[str] = (
        "global_step",
        "epoch",
        "sampler_offset",
        "parent_checkpoint_sha256",
        "dinov3_weights_sha256",
        "memory_contract_sha256",
        "reader_contract_sha256",
        "adapter",
        "reader",
        "optimizer",
        "lr_scheduler",
        "grad_scaler",
        "rng_by_rank",
        "contract",
        "trainer_state",
        "v2_state",
    )
    for field in compared_fields:
        _assert_exact(continuous[field], resumed[field], path=field)
    continuous_trace = _metric_trace(continuous_dir / "metrics.jsonl")
    resumed_trace = _metric_trace(resumed_dir / "metrics.jsonl")
    _assert_exact(continuous_trace, resumed_trace, path="metrics_trace")
    result = {
        "schema": "fastwam-p1-dino-contribution-v2-resume-parity",
        "status": "PASS",
        "continuous": {
            "directory": str(continuous_dir),
            "checkpoint": inspect_p1_dino_bc_full_checkpoint_v2(continuous_path),
        },
        "resumed": {
            "directory": str(resumed_dir),
            "checkpoint": inspect_p1_dino_bc_full_checkpoint_v2(resumed_path),
        },
        "compared_fields": list(compared_fields),
        "metrics_update_count": len(continuous_trace),
        "tensor_and_state_exact": True,
        "metrics_trace_exact": True,
    }
    _atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous-dir", required=True, type=Path)
    parser.add_argument("--resumed-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare_resume(
        args.continuous_dir.resolve(),
        args.resumed_dir.resolve(),
        args.output.resolve(),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
