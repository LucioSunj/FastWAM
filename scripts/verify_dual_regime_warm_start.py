"""GPU acceptance check for standalone-IDM -> shared S0 numerical parity.

This is intentionally a pre-training check. It strictly imports a standalone
IDM checkpoint into FusedDualRegimeFastWAM, then runs both IDM paths on the same
real dataset state with identical solver settings and random seed.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import tempfile
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True, help="resolved config.yaml from E-I")
    parser.add_argument("--source-task", required=True)
    parser.add_argument("--target-task", required=True)
    parser.add_argument("--ckpt", required=True, help="standalone FastWAMIDM checkpoint")
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--sigma-shift", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16"
    )
    parser.add_argument("--atol", type=float, default=5e-4)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--out", required=True, help="atomic parity_result.json output")
    return parser


def _tensor_sha256(tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"shape": list(value.shape), "dtype": str(value.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _atomic_write_json(path: str | os.PathLike, payload: dict) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def assert_parity_and_write_result(
    *,
    target_action,
    source_action,
    output_path: str | os.PathLike,
    metadata: dict,
    atol: float,
    rtol: float,
) -> dict:
    import torch

    torch.testing.assert_close(
        target_action,
        source_action,
        atol=atol,
        rtol=rtol,
        msg=(
            "S0 forced-IDM output differs from its standalone IDM parent under "
            "the paired seed/solver acceptance setting."
        ),
    )
    if source_action.numel() == 0:
        raise ValueError("parity action tensors must be non-empty")
    absolute_error = (target_action - source_action).abs()
    tolerance = atol + rtol * source_action.abs()
    normalized_error = torch.where(
        tolerance > 0,
        absolute_error / tolerance,
        torch.zeros_like(absolute_error),
    )
    worst_normalized_error = float(normalized_error.max().item())
    payload = {
        **metadata,
        "schema": "fastwam-warmstart-parity-v1",
        "kind": "standalone_idm_to_s0_fixed_seed_parity",
        "status": "PASS",
        "comparison": {
            "atol": float(atol),
            "rtol": float(rtol),
            "max_abs": float(absolute_error.max().item()),
            "worst_normalized_error": worst_normalized_error,
            "allclose_margin": 1.0 - worst_normalized_error,
        },
        "actions": {
            "source_sha256": _tensor_sha256(source_action),
            "target_sha256": _tensor_sha256(target_action),
            "shape": list(source_action.shape),
            "dtype": str(source_action.dtype),
        },
    }
    _atomic_write_json(output_path, payload)
    return payload


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from fastwam.adaptive_gate.provenance import (
        inference_solver_contract,
        inference_solver_fingerprint,
        sha256_file,
    )
    from fastwam.adaptive_gate.warm_start import strict_standalone_idm_warm_start
    from fastwam.utils import misc

    if args.sample_index < 0 or args.inference_steps <= 0:
        parser.error("--sample-index must be non-negative and --inference-steps positive")
    if args.atol < 0 or args.rtol < 0:
        parser.error("--atol and --rtol must be non-negative")
    stats_path = os.path.abspath(os.path.expanduser(args.dataset_stats))
    source_config_path = os.path.abspath(os.path.expanduser(args.source_config))
    checkpoint_path = os.path.abspath(os.path.expanduser(args.ckpt))
    for path in (stats_path, source_config_path, checkpoint_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    artifact_sha256 = {
        "checkpoint": sha256_file(checkpoint_path),
        "config": sha256_file(source_config_path),
        "stats": sha256_file(stats_path),
    }
    if artifact_sha256["checkpoint"].lower() != args.checkpoint_sha256.lower():
        raise ValueError(
            "E-I checkpoint SHA256 mismatch: "
            f"expected={args.checkpoint_sha256}, actual={artifact_sha256['checkpoint']}"
        )

    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    configs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
    with initialize_config_dir(version_base="1.3", config_dir=configs_dir):
        target_cfg = compose(config_name="train", overrides=[f"task={args.target_task}"])
    source_cfg = OmegaConf.load(source_config_path)
    source_model_cfg = source_cfg.model if "model" in source_cfg else source_cfg

    with tempfile.TemporaryDirectory(prefix="fastwam-s0-parity-") as work_dir:
        misc.register_work_dir(work_dir)
        dataset = instantiate(
            target_cfg.data.train,
            pretrained_norm_stats=stats_path,
        )
        sample = dataset.get_strict(args.sample_index)

    input_image = sample["video"][:, 0].unsqueeze(0).to(args.device, dtype=dtype)
    proprio = sample.get("proprio")
    if proprio is not None:
        proprio = proprio[0].unsqueeze(0).to(args.device, dtype=dtype)
    context = sample["context"].unsqueeze(0).to(args.device, dtype=dtype)
    context_mask = sample["context_mask"].unsqueeze(0).to(args.device)
    action_horizon = int(sample["action"].shape[0])
    num_video_frames = (
        (int(dataset.num_frames) - 1) // int(dataset.action_video_freq_ratio) + 1
    )
    common = {
        "prompt": None,
        "input_image": input_image,
        "action_horizon": action_horizon,
        "num_video_frames": num_video_frames,
        "proprio": proprio,
        "context": context,
        "context_mask": context_mask,
        "num_inference_steps": args.inference_steps,
        "sigma_shift": args.sigma_shift,
        "seed": args.seed,
    }

    # Run sequentially so the acceptance check never holds two 5B models on
    # one GPU at once.
    source = instantiate(source_model_cfg, model_dtype=dtype, device=args.device)
    source.load_checkpoint(checkpoint_path)
    source.eval().requires_grad_(False)
    source_solver_contract = inference_solver_contract(
        source,
        video_inference_steps=args.inference_steps,
        action_inference_steps=args.inference_steps,
        sigma_shift=args.sigma_shift,
    )
    with torch.inference_mode():
        source_action = source.infer_action(**common)["action"].float().cpu()
    del source
    gc.collect()
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()

    target = instantiate(target_cfg.model, model_dtype=dtype, device=args.device)
    strict_standalone_idm_warm_start(
        target,
        {
            "kind": "standalone_idm",
            "checkpoint": checkpoint_path,
            "expected_checkpoint_sha256": args.checkpoint_sha256,
            "source_task": args.source_task,
            "source_config": source_config_path,
            "source_dataset_stats": stats_path,
        },
        target_model_config=target_cfg.model,
        target_dataset_stats=stats_path,
    )
    target.eval().requires_grad_(False)
    target_solver_contract = inference_solver_contract(
        target,
        video_inference_steps=args.inference_steps,
        action_inference_steps=args.inference_steps,
        sigma_shift=args.sigma_shift,
    )
    if target_solver_contract != source_solver_contract:
        raise AssertionError(
            "S0 target and standalone E-I source inference solver contracts differ"
        )
    with torch.inference_mode():
        target_action = target.infer_action(
            **common, force_branch="idm", return_routing_info=True
        )["action"].float().cpu()

    device_metadata = {"requested": str(args.device)}
    if str(args.device).startswith("cuda"):
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        device_metadata.update(
            {
                "cuda_index": index,
                "name": properties.name,
                "compute_capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": properties.total_memory,
            }
        )
    result = assert_parity_and_write_result(
        target_action=target_action,
        source_action=source_action,
        output_path=args.out,
        atol=args.atol,
        rtol=args.rtol,
        metadata={
            "source_task": args.source_task,
            "target_task": args.target_task,
            "artifacts": {
                "e_i_checkpoint": {
                    "path": checkpoint_path,
                    "sha256": artifact_sha256["checkpoint"],
                },
                "e_i_config": {
                    "path": source_config_path,
                    "sha256": artifact_sha256["config"],
                },
                "e_i_stats": {
                    "path": stats_path,
                    "sha256": artifact_sha256["stats"],
                },
            },
            "sample_index": args.sample_index,
            "seed": args.seed,
            "model_dtype": args.dtype,
            "inference_steps": args.inference_steps,
            "sigma_shift": args.sigma_shift,
            "device": device_metadata,
            "solver_contract": target_solver_contract,
            "solver_contract_sha256": inference_solver_fingerprint(
                target_solver_contract
            ),
        },
    )
    print(
        "PASS standalone->S0 forced-IDM parity: "
        f"sample={args.sample_index} seed={args.seed} "
        f"max_abs={result['comparison']['max_abs']:.6g} "
        f"atol={args.atol} rtol={args.rtol}"
    )


if __name__ == "__main__":
    main()
