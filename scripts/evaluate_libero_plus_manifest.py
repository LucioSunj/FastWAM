"""Evaluate one frozen FastWAM endpoint on an exact LIBERO-Plus episode manifest.

The timing boundary starts before image/proprio/text preprocessing and ends after
the normalized action chunk is denormalized on CPU. Simulator stepping and reset
are excluded. CUDA is synchronized at both ends of every measured decision.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ENDPOINT_RUNTIME_TRIAL_FIELDS = (
    "solver_fingerprint",
    "solver_contract",
    "dtype",
    "inference_steps",
    "action_horizon",
    "generation_horizon",
    "replan_steps",
    "execution_horizon",
    "num_video_frames",
    "wam_seed",
    "action_seed",
    "num_steps_wait",
    "binarize_gripper",
    "rand_device",
    "tiled",
    "text_cfg_scale",
    "negative_prompt",
    "use_action_ensembler",
    "input_height",
    "input_width",
    "max_episode_steps",
    "timing_boundary",
    "device",
    "hardware_fingerprint",
    "hardware_contract",
)


def _sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _within(path: str | os.PathLike[str], root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root)
    except ValueError:
        return False
    return True


def _route_libero_plus(root: Path):
    sys.path.insert(0, str(root))
    # The official LIBERO-Plus checkout keeps the upstream import namespace
    # (``libero.libero``); it does not install a ``liberoplus`` package.
    package = importlib.import_module("libero")
    package_file = getattr(package, "__file__", None)
    if package_file is None or not _within(package_file, root):
        raise RuntimeError(
            "imported libero does not come from the pinned LIBERO_PLUS_ROOT: "
            f"{package_file!r} vs {root}"
        )
    benchmark_module = importlib.import_module("libero.libero.benchmark")
    envs_module = importlib.import_module("libero.libero.envs")
    for module in (benchmark_module, envs_module):
        module_file = getattr(module, "__file__", None)
        if module_file is None or not _within(module_file, root):
            raise RuntimeError(
                "LIBERO-Plus submodule was resolved outside the pinned checkout: "
                f"{module_file!r} vs {root}"
            )
    return benchmark_module, envs_module


def _sync(device: str) -> None:
    import torch

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _hardware_contract(device: str) -> dict[str, Any]:
    """Return the canonical compute contract relevant to parity and latency."""
    import platform

    import torch

    resolved = torch.device(device)
    common = {
        "device_type": resolved.type,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": None if torch.version.cuda is None else str(torch.version.cuda),
        "cudnn_version": (
            None if torch.backends.cudnn.version() is None else int(torch.backends.cudnn.version())
        ),
    }
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA endpoint device is unavailable: {device}")
        index = torch.cuda.current_device() if resolved.index is None else int(resolved.index)
        properties = torch.cuda.get_device_properties(index)
        return {
            **common,
            "device_index": index,
            "device_name": str(properties.name),
            "compute_capability": [int(properties.major), int(properties.minor)],
            "total_memory_bytes": int(properties.total_memory),
        }
    return {
        **common,
        "device_index": None,
        "device_name": platform.processor() or platform.machine(),
        "compute_capability": None,
        "total_memory_bytes": None,
    }


def _canonical_json_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="FastWAM Hydra task config")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dataset-stats", required=True)
    parser.add_argument("--episode-manifest", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--force-branch", choices=["base", "idm"], required=True)
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--sigma-shift", type=float, default=None)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--replan-steps", type=int, default=10)
    parser.add_argument("--num-steps-wait", type=int, default=30)
    parser.add_argument("--wam-seed", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    if args.inference_steps <= 0 or args.replan_steps <= 0 or args.num_shards <= 0:
        parser.error("inference/replan steps and num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("shard-index must be in [0, num-shards)")

    plus_root_value = os.environ.get("LIBERO_PLUS_ROOT")
    plus_commit = os.environ.get("LIBERO_PLUS_COMMIT")
    if not plus_root_value or not plus_commit:
        parser.error("LIBERO_PLUS_ROOT and LIBERO_PLUS_COMMIT are required")
    plus_root = Path(plus_root_value).expanduser().resolve()
    benchmark_module, envs_module = _route_libero_plus(plus_root)

    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    from experiments.libero.eval_libero_single import (
        _get_num_video_frames,
        _load_model_checkpoint,
        _mixed_precision_to_model_dtype,
        _predict_action_chunk,
    )
    from experiments.libero.libero_utils import (
        LIBERO_ENV_RESOLUTION,
        get_libero_dummy_action,
    )
    from fastwam.adaptive_gate import (
        load_plus_manifest,
        validate_dataset_stats_fingerprint,
    )
    from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    checkpoint = Path(args.ckpt).expanduser().resolve()
    stats_path = Path(args.dataset_stats).expanduser().resolve()
    if not checkpoint.is_file() or not stats_path.is_file():
        raise FileNotFoundError("checkpoint and dataset stats must both exist")
    manifest = load_plus_manifest(
        args.episode_manifest,
        libero_plus_root=plus_root,
        libero_plus_commit=plus_commit,
        expected_split="test",
    )
    episodes = manifest.episodes[args.shard_index :: args.num_shards]
    if not episodes:
        raise ValueError("manifest shard contains no episodes")

    configs_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(configs_dir)):
        cfg = compose(config_name="sim_libero", overrides=[f"task={args.task}"])
    cfg.ckpt = str(checkpoint)
    cfg.seed = int(args.wam_seed)
    cfg.EVALUATION.dataset_stats_path = str(stats_path)
    cfg.EVALUATION.force_branch = args.force_branch
    cfg.EVALUATION.num_inference_steps = int(args.inference_steps)
    cfg.EVALUATION.sigma_shift = args.sigma_shift
    cfg.EVALUATION.replan_steps = int(args.replan_steps)
    cfg.EVALUATION.num_steps_wait = int(args.num_steps_wait)
    cfg.EVALUATION.visualize_future_video = False
    cfg.EVALUATION.use_action_ensembler = False

    dtype = _mixed_precision_to_model_dtype(
        {"float32": "no", "float16": "fp16", "bfloat16": "bf16"}[args.dtype]
    )
    model = instantiate(cfg.model, model_dtype=dtype, device=args.device)
    _load_model_checkpoint(model, str(checkpoint))
    model = model.to(args.device).eval().requires_grad_(False)
    adaptive_kind = getattr(model, "adaptive_backbone_kind", None)
    if adaptive_kind is not None:
        provenance = getattr(model, "_loaded_checkpoint_provenance", None)
        if not isinstance(provenance, dict):
            raise ValueError("adaptive endpoint checkpoint lacks strict provenance")
        if tuple(provenance.get("adaptive_regimes", ())) != ("uncond", "idm"):
            raise ValueError("adaptive endpoint is not a dual UNCOND/IDM checkpoint")
        steps = provenance.get("dual_regime_optimizer_steps")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError(
                "adaptive endpoint is an untrained S0 checkpoint: "
                f"dual_regime_optimizer_steps={steps!r}"
            )
    validate_dataset_stats_fingerprint(model, stats_path)
    stats = load_dataset_stats_from_json(str(stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(stats)
    action_horizon = (
        int(cfg.data.train.num_frames) - 1
        if args.action_horizon is None
        else int(args.action_horizon)
    )
    input_h, input_w = map(int, cfg.data.train.video_size)
    from fastwam.adaptive_gate import (
        inference_solver_contract,
        inference_solver_fingerprint,
    )

    solver_contract = inference_solver_contract(
        model,
        video_inference_steps=args.inference_steps,
        action_inference_steps=args.inference_steps,
        sigma_shift=args.sigma_shift,
    )
    solver_fingerprint = inference_solver_fingerprint(solver_contract)
    num_video_frames = _get_num_video_frames(cfg)
    generation_horizon = action_horizon
    execution_horizon = int(args.replan_steps)
    action_seed = int(args.wam_seed)
    hardware_contract = _hardware_contract(args.device)
    hardware_fingerprint = _canonical_json_sha256(hardware_contract)
    endpoint_runtime = {
        "dtype": args.dtype,
        "inference_steps": int(args.inference_steps),
        "solver_contract": solver_contract,
        "solver_fingerprint": solver_fingerprint,
        "action_horizon": action_horizon,
        "generation_horizon": generation_horizon,
        "replan_steps": int(args.replan_steps),
        "execution_horizon": execution_horizon,
        "num_video_frames": num_video_frames,
        "wam_seed": int(args.wam_seed),
        "action_seed": action_seed,
        "num_steps_wait": int(args.num_steps_wait),
        "binarize_gripper": bool(cfg.EVALUATION.get("binarize_gripper", False)),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "use_action_ensembler": bool(
            cfg.EVALUATION.get("use_action_ensembler", False)
        ),
        "input_height": input_h,
        "input_width": input_w,
        "timing_boundary": (
            "preprocess_start_to_denormalized_cpu_action_end_cuda_synchronized"
        ),
        "device": str(args.device),
        "hardware_contract": hardware_contract,
        "hardware_fingerprint": hardware_fingerprint,
    }

    benchmark_dict = benchmark_module.get_benchmark_dict()
    records: list[dict[str, Any]] = []
    all_latencies = []
    for episode in episodes:
        if episode.task_suite_name not in benchmark_dict:
            raise KeyError(f"unknown LIBERO suite {episode.task_suite_name!r}")
        suite = benchmark_dict[episode.task_suite_name]()
        task = suite.get_task(episode.task_id)
        initial_states = suite.get_task_init_states(episode.task_id)
        if episode.reset_state_id >= len(initial_states):
            raise IndexError(
                f"episode {episode.episode_id} reset_state_id={episode.reset_state_id} "
                f"outside {len(initial_states)} available states"
            )
        env = envs_module.OffScreenRenderEnv(
            bddl_file_name=episode.bddl_path,
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
        )
        env.seed(episode.env_seed)
        cfg.EVALUATION.task_suite_name = episode.task_suite_name
        obs = env.reset()
        obs = env.set_init_state(initial_states[episode.reset_state_id])
        pending_actions: list[list[float]] = []
        latencies = []
        max_steps = 700 if episode.task_suite_name in {"libero_10", "libero_90"} else 400
        success = False
        decisions = 0
        for step in range(max_steps + args.num_steps_wait):
            if step < args.num_steps_wait:
                obs, _, success, _ = env.step(get_libero_dummy_action())
                if success:
                    break
                continue
            if not pending_actions:
                _sync(args.device)
                started = time.perf_counter_ns()
                action_chunk, _, _ = _predict_action_chunk(
                    obs=obs,
                    task_description=task.language,
                    model=model,
                    processor=processor,
                    cfg=cfg,
                    action_horizon=action_horizon,
                    input_w=input_w,
                    input_h=input_h,
                    model_device=args.device,
                )
                _sync(args.device)
                latencies.append((time.perf_counter_ns() - started) / 1e6)
                decisions += 1
                pending_actions = action_chunk[: args.replan_steps].tolist()
            obs, _, success, _ = env.step(pending_actions.pop(0))
            if success:
                break
        close = getattr(env, "close", None)
        if callable(close):
            close()
        if not latencies:
            raise RuntimeError(f"episode {episode.episode_id} executed no WAM decision")
        all_latencies.extend(latencies)
        trial_runtime = {"max_episode_steps": max_steps, **endpoint_runtime}
        if set(trial_runtime) != set(ENDPOINT_RUNTIME_TRIAL_FIELDS):
            raise RuntimeError(
                "endpoint runtime emission drifted from its analyzer contract: "
                f"missing={sorted(set(ENDPOINT_RUNTIME_TRIAL_FIELDS) - set(trial_runtime))}, "
                f"extra={sorted(set(trial_runtime) - set(ENDPOINT_RUNTIME_TRIAL_FIELDS))}"
            )
        records.append(
            {
                "method": args.method,
                "episode_uid": episode.episode_id,
                "task": episode.base_task,
                "task_suite_name": episode.task_suite_name,
                "task_id": episode.task_id,
                "factor": episode.factor,
                "level": episode.level,
                "perturbation_id": episode.perturbation_id,
                "asset_ids": list(episode.asset_ids),
                "reset_state_id": episode.reset_state_id,
                "trial_id": episode.trial_id,
                "seed": episode.env_seed,
                "success": bool(success),
                "num_decisions": decisions,
                "chunk_latency_ms": latencies,
                "mean_chunk_latency_ms": statistics.fmean(latencies),
                "median_chunk_latency_ms": statistics.median(latencies),
                "manifest_sha256": manifest.sha256,
                "checkpoint_sha256": _sha256(checkpoint),
                "dataset_stats_sha256": _sha256(stats_path),
                "force_branch": args.force_branch,
                **trial_runtime,
            }
        )

    output = Path(args.out).expanduser().resolve()
    _atomic_jsonl(output / f"trials_shard_{args.shard_index}_of_{args.num_shards}.jsonl", records)
    _atomic_json(
        output / f"summary_shard_{args.shard_index}_of_{args.num_shards}.json",
        {
            "method": args.method,
            "num_episodes": len(records),
            "success_rate": sum(record["success"] for record in records) / len(records),
            "mean_chunk_latency_ms": statistics.fmean(all_latencies),
            "median_chunk_latency_ms": statistics.median(all_latencies),
            "manifest_sha256": manifest.sha256,
            "checkpoint_sha256": _sha256(checkpoint),
            "dataset_stats_sha256": _sha256(stats_path),
            **endpoint_runtime,
        },
    )


if __name__ == "__main__":
    main()
