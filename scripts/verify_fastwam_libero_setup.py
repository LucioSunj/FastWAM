#!/usr/bin/env python
"""Verify the resource-backed FastWAM LIBERO training setup."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import struct
import tarfile
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import get_method, instantiate


TASKS = (
    "libero_uncond_2cam224_1e-4",
    "libero_idm_2cam224_1e-4",
    "libero_joint_2cam224_1e-4",
    "libero_metric_adaptive_2cam224_1e-4",
    "libero_metric_adaptive_joint_2cam224_1e-4",
)
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
EXPECTED_SIZES = {
    "models/wan22_components/diffusion_pytorch_model-00001-of-00003.safetensors": 9825014472,
    "models/wan22_components/diffusion_pytorch_model-00002-of-00003.safetensors": 9995661736,
    "models/wan22_components/diffusion_pytorch_model-00003-of-00003.safetensors": 178558176,
    "models/wan22_components/diffusion_pytorch_model.safetensors.index.json": 72865,
    "models/wan22_components/Wan2.2_VAE.safetensors": 1409401152,
    "models/wan22_components/models_t5_umt5-xxl-enc-bf16.safetensors": 11361845432,
    "models/wan22_components/google/umt5-xxl/special_tokens_map.json": 6623,
    "models/wan22_components/google/umt5-xxl/spiece.model": 4548313,
    "models/wan22_components/google/umt5-xxl/tokenizer.json": 16837417,
    "models/wan22_components/google/umt5-xxl/tokenizer_config.json": 61728,
    "models/wan22_robot/checkpoint.safetensors": 9999659704,
    "checkpoints/fastwam_release/libero_uncond_2cam224.pt": 12041735140,
    "checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json": 40939,
    "checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt": 12041813092,
    "checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json": 88715,
    "data/libero_mujoco3.3.2/libero_10_no_noops_lerobot.tar.gz": 1533534552,
    "data/libero_mujoco3.3.2/libero_goal_no_noops_lerobot.tar.gz": 837172899,
    "data/libero_mujoco3.3.2/libero_object_no_noops_lerobot.tar.gz": 1352739510,
    "data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot.tar.gz": 968794594,
}


def _check_safetensors(path: Path) -> dict:
    with path.open("rb") as stream:
        raw_len = stream.read(8)
        if len(raw_len) != 8:
            raise ValueError(f"Short safetensors header: {path}")
        header_len = struct.unpack("<Q", raw_len)[0]
        if header_len <= 2 or header_len > min(path.stat().st_size - 8, 100_000_000):
            raise ValueError(f"Invalid safetensors header length {header_len}: {path}")
        header = json.loads(stream.read(header_len))
    tensor_count = sum(key != "__metadata__" for key in header)
    if tensor_count == 0:
        raise ValueError(f"No tensors in safetensors header: {path}")
    return {"size": path.stat().st_size, "tensors": tensor_count}


def verify_assets(root: Path) -> dict:
    checked = {}
    for relative, expected_size in EXPECTED_SIZES.items():
        path = root / relative
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ValueError(f"Size mismatch for {path}: {actual_size} != {expected_size}")
        checked[relative] = {"size": actual_size}

    component_dir = root / "models/wan22_components"
    index_path = component_dir / "diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    index_shards = sorted(set(index["weight_map"].values()))
    expected_shards = [
        f"diffusion_pytorch_model-{idx:05d}-of-00003.safetensors"
        for idx in range(1, 4)
    ]
    if index_shards != expected_shards:
        raise ValueError(f"Unexpected Wan shard map: {index_shards}")

    safetensors_paths = list(component_dir.glob("*.safetensors"))
    safetensors_paths.append(root / "models/wan22_robot/checkpoint.safetensors")
    for path in safetensors_paths:
        checked[str(path.relative_to(root))] = _check_safetensors(path)

    archive_counts = {}
    for path in (root / "data/libero_mujoco3.3.2").glob("*.tar.gz"):
        with tarfile.open(path, "r:gz") as archive:
            member_count = sum(1 for _ in archive)
            if member_count == 0:
                raise ValueError(f"Empty tar archive: {path}")
            archive_counts[path.name] = member_count
    checked["tar_member_counts"] = archive_counts

    for suite in SUITES:
        extracted = root / "data/libero_mujoco3.3.2" / f"{suite}_no_noops_lerobot"
        for relative in ("meta/info.json", "meta/tasks.jsonl"):
            if not (extracted / relative).is_file():
                raise FileNotFoundError(extracted / relative)

    for path in (
        root / "checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json",
        root / "checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json",
    ):
        json.loads(path.read_text())
    return checked


def _compose_tasks(repo: Path):
    configs = {}
    with initialize_config_dir(version_base="1.3", config_dir=str(repo / "configs")):
        for task_name in TASKS:
            cfg = compose(config_name="train", overrides=[f"task={task_name}"])
            target = str(cfg.model._target_)
            get_method(target)
            dirs = [str(path) for path in cfg.data.train.dataset_dirs]
            if len(dirs) != 4:
                raise ValueError(f"{task_name} has {len(dirs)} dataset dirs, expected 4")
            configs[task_name] = (cfg, target)
    return configs


def verify_configs(repo: Path) -> dict:
    return {name: target for name, (_, target) in _compose_tasks(repo).items()}


def verify_suites(run_sim: bool) -> dict:
    from libero.libero import benchmark

    benchmark_dict = benchmark.get_benchmark_dict()
    results = {}
    for suite_name in SUITES:
        suite = benchmark_dict[suite_name]()
        if suite.n_tasks != 10:
            raise ValueError(f"{suite_name} has {suite.n_tasks} tasks, expected 10")
        loaded_states = []
        for task_id in range(suite.n_tasks):
            states = suite.get_task_init_states(task_id)
            if len(states) == 0:
                raise ValueError(f"{suite_name} task {task_id} has no init states")
            loaded_states.append(len(states))
        results[suite_name] = {"tasks": suite.n_tasks, "init_states": loaded_states}

        if run_sim:
            from experiments.libero.libero_utils import (
                get_libero_dummy_action,
                get_libero_env,
                get_libero_image,
            )

            task = suite.get_task(0)
            env, _ = get_libero_env(task, resolution=256, seed=0)
            try:
                env.reset()
                obs = env.set_init_state(suite.get_task_init_states(0)[0])
                obs, _, _, _ = env.step(get_libero_dummy_action())
                images = get_libero_image(obs)
                image_stats = {}
                for key in ("image", "wrist_image"):
                    frame = np.asarray(images[key])
                    if frame.size == 0 or frame.ndim != 3 or float(frame.std()) == 0.0:
                        raise ValueError(f"{suite_name} produced invalid {key}: {frame.shape}")
                    image_stats[key] = {
                        "shape": list(frame.shape),
                        "min": int(frame.min()),
                        "max": int(frame.max()),
                    }
                results[suite_name]["egl_render"] = image_stats
            finally:
                env.close()
    return results


def verify_dataset(repo: Path, root: Path) -> dict:
    from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
    from fastwam.utils import misc

    verify_dir = root / "runs/setup_verification/dataset"
    verify_dir.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(verify_dir))
    cfg = _compose_tasks(repo)[TASKS[0]][0]
    dataset = instantiate(cfg.data.train)
    sample = dataset[0]
    required = ("video", "action", "proprio", "context", "context_mask")
    missing = [key for key in required if key not in sample]
    if missing:
        raise ValueError(f"Dataset sample missing keys: {missing}")
    expected_shapes = {
        "video": [3, 9, 224, 448],
        "action": [32, 7],
        "proprio": [32, 8],
        "context": [128, 4096],
        "context_mask": [128],
    }
    for key, expected_shape in expected_shapes.items():
        actual_shape = list(sample[key].shape)
        if actual_shape != expected_shape:
            raise ValueError(f"Unexpected {key} shape: {actual_shape} != {expected_shape}")

    prompts = set()
    for dataset_dir in cfg.data.train.dataset_dirs:
        with (repo / str(dataset_dir) / "meta/tasks.jsonl").open() as stream:
            for line in stream:
                record = json.loads(line)
                prompts.add(DEFAULT_PROMPT.format(task=record["task"]))
    if len(prompts) != 40:
        raise ValueError(f"Expected 40 unique LIBERO prompts, got {len(prompts)}")

    cache_dir = repo / str(cfg.data.train.text_embedding_cache_dir)
    cache_shapes = set()
    for prompt in prompts:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = cache_dir / f"{digest}.t5_len128.wan22ti2v5b.pt"
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        shape_pair = (tuple(payload["context"].shape), tuple(payload["mask"].shape))
        if shape_pair != ((128, 4096), (128,)):
            raise ValueError(f"Unexpected cache shapes in {cache_path}: {shape_pair}")
        cache_shapes.add(shape_pair)
    return {
        "length": len(dataset),
        "sample_shapes": expected_shapes,
        "cached_prompts": len(prompts),
        "cache_shapes": [[list(context), list(mask)] for context, mask in cache_shapes],
        "prompt": sample["prompt"],
    }


def instantiate_models(repo: Path) -> dict:
    results = {}
    for task_name, (cfg, target) in _compose_tasks(repo).items():
        model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda:0")
        results[task_name] = {"target": target, "class": type(model).__name__}
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-root", default=os.environ.get("FASTWAM_RESOURCE_ROOT", "/root/autodl-fs/fastwam"))
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--skip-sim", action="store_true")
    parser.add_argument("--skip-dataset", action="store_true")
    parser.add_argument("--instantiate-models", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.resource_root).resolve()
    repo = Path(args.repo_root).resolve()
    results = {
        "assets": verify_assets(root),
        "configs": verify_configs(repo),
        "libero": verify_suites(run_sim=not args.skip_sim),
    }
    if not args.skip_dataset:
        results["dataset"] = verify_dataset(repo, root)
    if args.instantiate_models:
        results["models"] = instantiate_models(repo)

    output = Path(args.output) if args.output else root / "manifests/verification_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"Verification results: {output}")


if __name__ == "__main__":
    main()
