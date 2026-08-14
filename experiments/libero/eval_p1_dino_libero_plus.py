"""Persistent rank-32 P1 evaluation workers for all LIBERO-Plus tasks."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero import eval_libero_single as legacy
from experiments.libero.libero_utils import LIBERO_ENV_RESOLUTION
from fastwam.adapters import sha256_file
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.models.wan22.visual_contracts import contract_sha256
from fastwam.p1_dino_bc_checkpoint import load_p1_dino_bc_checkpoint
from fastwam.p1_dino_bc_full_checkpoint import (
    P1_DINO_BC_FULL_CHECKPOINT_SCHEMA,
    P1_DINO_BC_FULL_CHECKPOINT_V2_SCHEMA,
    load_p1_dino_bc_full_trainables,
)
from fastwam.p1_dino_bc_runner import build_real_p1_policy, is_visual_v2_config
from fastwam.p1_dino_libero_inference import (
    P1DinoCompiledLiberoPolicy,
    P1VisualCompiledLiberoPolicy,
    resolve_compile_cache_seed,
)
from fastwam.p1_visual_bc_checkpoint import load_p1_visual_bc_checkpoint
from fastwam.p1_visual_bc_full_checkpoint import (
    P1_VISUAL_BC_FULL_CHECKPOINT_SCHEMA,
    load_p1_visual_bc_full_weights_for_evaluation,
)
from fastwam.utils.pytorch_utils import set_global_seed
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
EVALUATION_SETS = ("libero-plus", "libero-standard")


def _get_plus_env(task, resolution: int, seed: int):
    """Construct LIBERO-Plus with the string path required by its old API."""

    bddl_path = str(
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt-cache-dir", required=True)
    parser.add_argument("--classification", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument(
        "--evaluation-set", choices=EVALUATION_SETS, default="libero-plus"
    )
    parser.add_argument(
        "--memory-mode",
        choices=("correct", "off", "random_tensor", "random_vit"),
        default="correct",
    )
    parser.add_argument("--compile-cache-seed")
    parser.add_argument("--visual-backbone")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--worker-id", type=int)
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--task-file")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    return parser.parse_args()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_tasks(path: Path) -> list[tuple[str, int, int]]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip() and not raw.lstrip().startswith("#"):
            fields = [value.strip() for value in raw.split(",")]
            if len(fields) not in {2, 3}:
                raise ValueError(f"Malformed LIBERO task row: {raw!r}.")
            suite, task_id = fields[:2]
            reset_id = 0 if len(fields) == 2 else int(fields[2])
            rows.append((suite, int(task_id), reset_id))
    if not rows:
        raise ValueError(f"Task file is empty: {path}")
    return rows


def _load_training_config(path: str, *, visual_backbone: str | None):
    """Load a resolved job config or compose a public Hydra config file."""

    config_path = Path(path).expanduser().resolve()
    raw = OmegaConf.load(config_path)
    if "defaults" in raw:
        overrides = (
            [] if visual_backbone is None else [f"visual_backbone={visual_backbone}"]
        )
        with initialize_config_dir(
            config_dir=str(config_path.parent),
            version_base="1.3",
        ):
            return compose(config_name=config_path.stem, overrides=overrides)
    if visual_backbone is not None:
        raise ValueError(
            "`--visual-backbone` can only override an unresolved public Hydra "
            "config; a resolved training config is already identity-bound."
        )
    return raw


def _eval_cfg(training_cfg, output_dir: Path):
    cfg = OmegaConf.create(OmegaConf.to_container(training_cfg, resolve=True))
    cfg.seed = 42
    cfg.eval_num_inference_steps = 10
    cfg.EVALUATION = {
        "task_suite_name": "libero_spatial",
        "task_id": 0,
        "num_trials": 1,
        "output_dir": str(output_dir),
        "env_num": 1,
        "num_steps_wait": 30,
        "replan_steps": 10,
        "binarize_gripper": True,
        "use_action_ensembler": False,
        "visualize_future_video": False,
        "save_rollout_video": False,
        "show_progress": False,
        "action_horizon": 32,
        "num_inference_steps": 10,
        "sigma_shift": None,
        "text_cfg_scale": 1.0,
        "negative_prompt": "",
        "rand_device": "cpu",
        "tiled": False,
    }
    return cfg


def _build_model(args: argparse.Namespace, training_cfg):
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    policy, parent_load = build_real_p1_policy(training_cfg, device=device)
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or not isinstance(raw.get("contract"), dict):
        raise ValueError("P1 checkpoint is missing its strict contract.")
    common = {
        "path": args.checkpoint,
        "adapter": policy.lora_adapter,
        "reader": policy.visual_reader,
        "expected_parent_checkpoint_sha256": str(training_cfg.parent.checkpoint_sha256),
        "expected_contract": raw["contract"],
    }
    if is_visual_v2_config(training_cfg):
        expected_visual = policy.visual_encoder.asset.checkpoint_metadata(
            camera_ids=policy.p1_config.camera_ids,
            input_contract_sha256=(policy.p1_config.camera_input_contract_sha256),
        )
        if raw.get("schema") == P1_VISUAL_BC_FULL_CHECKPOINT_SCHEMA:
            loaded = load_p1_visual_bc_full_weights_for_evaluation(
                **common,
                expected_visual_backbone=expected_visual,
            )
        else:
            loaded = load_p1_visual_bc_checkpoint(
                **common,
                expected_visual_backbone=expected_visual,
            )
        wrapper = P1VisualCompiledLiberoPolicy
    else:
        dino_common = {
            "path": args.checkpoint,
            "adapter": policy.lora_adapter,
            "reader": policy.visual_reader,
            "expected_parent_checkpoint_sha256": str(
                training_cfg.parent.checkpoint_sha256
            ),
            "expected_dinov3_weights_sha256": str(
                training_cfg.p1.dinov3.weights_sha256
            ),
        }
        if raw.get("schema") in {
            P1_DINO_BC_FULL_CHECKPOINT_SCHEMA,
            P1_DINO_BC_FULL_CHECKPOINT_V2_SCHEMA,
        }:
            loaded = load_p1_dino_bc_full_trainables(**dino_common)
        else:
            loaded = load_p1_dino_bc_checkpoint(
                **dino_common,
                expected_contract=raw["contract"],
            )
        wrapper = P1DinoCompiledLiberoPolicy
    model = wrapper(
        policy,
        prompt_cache_dir=args.prompt_cache_dir,
        compile_enabled=not args.no_compile,
        memory_mode=args.memory_mode,
    )
    return model, parent_load, loaded


def _worker(args: argparse.Namespace) -> None:
    if args.worker_id is None or args.physical_gpu is None or args.task_file is None:
        raise ValueError("Worker mode requires worker/gpu/task-file arguments.")
    output_dir = Path(args.output_dir).expanduser().resolve()
    task_rows = _read_tasks(Path(args.task_file).expanduser().resolve())
    training_cfg = _load_training_config(
        args.config,
        visual_backbone=args.visual_backbone,
    )
    eval_cfg = _eval_cfg(training_cfg, output_dir)
    set_global_seed(42, get_worker_init_fn=False)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    model, parent_load, loaded = _build_model(args, training_cfg)
    checkpoint_sha256 = sha256_file(args.checkpoint)

    statistics_path = Path(str(training_cfg.parent.statistics)).resolve()
    processor: FastWAMProcessor = instantiate(training_cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(
        load_dataset_stats_from_json(str(statistics_path))
    )
    classification = json.loads(Path(args.classification).read_text(encoding="utf-8"))
    registry = benchmark.get_benchmark_dict()
    suite_cache = {}
    failures_path = output_dir / f"worker_{args.worker_id}_failures.jsonl"

    active_task: tuple[str, int] | None = None
    cached_context = None
    cached_mask = None
    for ordinal, (suite_name, task_id, reset_id) in enumerate(task_rows, start=1):
        if args.evaluation_set == "libero-plus":
            result_name = f"worker{args.worker_id}_task{task_id}_results.json"
        else:
            result_name = (
                f"worker{args.worker_id}_task{task_id}_reset{reset_id}_results.json"
            )
        result_path = output_dir / "results" / suite_name / result_name
        if args.resume and result_path.is_file():
            continue
        started = time.time()
        env = None
        try:
            if suite_name not in suite_cache:
                with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
                    suite_cache[suite_name] = registry[suite_name]()
            suite = suite_cache[suite_name]
            task = suite.get_task(task_id)
            initial_states = suite.get_task_init_states(task_id)
            if not 0 <= reset_id < len(initial_states):
                raise ValueError(
                    f"Reset {reset_id} unavailable for {suite_name}/{task_id}; "
                    f"only {len(initial_states)} initial states exist."
                )
            eval_cfg.EVALUATION.task_suite_name = suite_name
            eval_cfg.EVALUATION.task_id = task_id
            env, description = _get_plus_env(task, LIBERO_ENV_RESOLUTION, 42)
            prompt = DEFAULT_PROMPT.format(task=description)
            task_key = (suite_name, task_id)
            if active_task != task_key:
                model.clear_prompt()
                cached_context, cached_mask = model.encode_prompt(prompt)
                active_task = task_key
            if cached_context is None or cached_mask is None:
                raise RuntimeError("Encoded prompt context was not retained.")
            success, _images, _predicted, _psnr, infer_time = legacy.run_single_episode(
                env=env,
                initial_state=initial_states[reset_id],
                task_description=description,
                model=model,
                processor=processor,
                cfg=eval_cfg,
                episode_idx=reset_id,
                action_horizon=32,
                input_w=448,
                input_h=224,
                model_device="cuda:0",
                cached_context=cached_context,
                cached_context_mask=cached_mask,
            )
            category = dict(classification[suite_name][task_id])
            if int(category["id"]) != task_id + 1:
                raise ValueError("LIBERO-Plus classification index changed.")
            prompt_audit = model.prompt_audit
            dino_call_audit = model.dino_call_audit
            if (
                prompt_audit["runtime_text_encoder_calls"] != 0
                or prompt_audit["cache_load_count_for_current_task"] != 1
            ):
                raise RuntimeError("Encoded-prompt reuse contract failed.")
            if not dino_call_audit["contract_passed"]:
                raise RuntimeError("DINO memory encoder call-count contract failed.")
            parity = model.parity_report
            if not args.no_compile and (parity is None or not parity["passed"]):
                raise RuntimeError("torch.compile parity contract failed.")
            payload = {
                "schema": (
                    "fastwam-p1-visual-libero-plus-result-v2"
                    if is_visual_v2_config(training_cfg)
                    else "fastwam-p1-dino-libero-plus-result-v1"
                ),
                "status": "PASS",
                "suite": suite_name,
                "task_id": task_id,
                "reset_id": reset_id,
                "evaluation_set": args.evaluation_set,
                "memory_mode": args.memory_mode,
                "task_description": description,
                "success": bool(success),
                "total_episodes": 1,
                "worker_id": args.worker_id,
                "physical_gpu": args.physical_gpu,
                "duration_seconds": time.time() - started,
                "inference_seconds": infer_time,
                "category": category.get("category"),
                "difficulty_level": category.get("difficulty_level"),
                "encoded_prompt": prompt_audit,
                "dino_call_audit": dino_call_audit,
                "torch_compile": {
                    "enabled": not args.no_compile,
                    "mode": "reduce-overhead",
                    "prefill_mode": "default",
                    "parity": parity,
                    "visual_compile_cache_key": model.visual_compile_cache_key,
                },
                "visual_input": model.visual_input_audit,
                "checkpoint": {
                    "path": str(Path(args.checkpoint).resolve()),
                    "sha256": checkpoint_sha256,
                    "rank": int(loaded["adapter"]["metadata"]["rank"]),
                    "global_step": int(loaded["global_step"]),
                },
                "parent_load": parent_load,
                "statistics": {
                    "path": str(statistics_path),
                    "sha256": str(training_cfg.parent.statistics_sha256),
                },
            }
            _atomic_json(result_path, payload)
            print(
                f"[{ordinal}/{len(task_rows)}] {suite_name}/{task_id} "
                f"reset={reset_id} "
                f"success={bool(success)} duration={payload['duration_seconds']:.1f}s",
                flush=True,
            )
        except Exception as error:
            record = {
                "suite": suite_name,
                "task_id": task_id,
                "reset_id": reset_id,
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
            }
            failures_path.parent.mkdir(parents=True, exist_ok=True)
            with failures_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            raise
        finally:
            if env is not None and callable(getattr(env, "close", None)):
                env.close()


def _all_tasks(evaluation_set: str) -> list[tuple[str, int, int]]:
    registry = benchmark.get_benchmark_dict()
    rows = []
    for suite_name in SUITES:
        with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
            suite = registry[suite_name]()
        if evaluation_set == "libero-plus":
            rows.extend(
                (suite_name, task_id, 0) for task_id in range(int(suite.n_tasks))
            )
        else:
            if int(suite.n_tasks) < 10:
                raise RuntimeError(f"{suite_name} exposes fewer than 10 tasks.")
            rows.extend(
                (suite_name, task_id, reset_id)
                for task_id in range(10)
                for reset_id in range(5)
            )
    expected = 10030 if evaluation_set == "libero-plus" else 200
    if len(rows) != expected:
        raise RuntimeError(
            f"{evaluation_set} episode count changed: {len(rows)} != {expected}."
        )
    return rows


def _compile_identity(training_cfg, *, memory_mode: str) -> str:
    """Bind the persistent Inductor cache to the complete V2 visual contract."""

    if not is_visual_v2_config(training_cfg):
        return f"dinov3_vits16_224_{memory_mode}_v2"
    visual = OmegaConf.to_container(
        training_cfg.p1.visual_backbone,
        resolve=True,
    )
    reader = OmegaConf.to_container(training_cfg.p1.reader, resolve=True)
    identity = contract_sha256(
        {
            "schema": "fastwam-p1-visual-compile-cache-v2",
            "visual_backbone": visual,
            "camera_ids": list(training_cfg.p1.camera_ids),
            "visual_camera_input_contract_sha256": str(
                training_cfg.p1.visual_camera_input_contract_sha256
            ),
            "reader": reader,
        }
    )
    return (
        f"{visual['family']}_{visual['variant']}_{int(visual['input_size'])}_"
        f"{memory_mode}_{identity}"
    )


def _manager(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gpus = [int(value) for value in args.gpus.split(",")]
    if gpus != [4, 5, 6, 7]:
        raise ValueError("This evaluation is pinned to physical GPUs 4,5,6,7.")
    rows = _all_tasks(args.evaluation_set)
    if args.workers_per_gpu < 1:
        raise ValueError("workers-per-gpu must be positive.")
    worker_gpus = [gpu for gpu in gpus for _ in range(args.workers_per_gpu)]
    training_cfg = _load_training_config(
        args.config,
        visual_backbone=args.visual_backbone,
    )
    compile_identity = _compile_identity(training_cfg, memory_mode=args.memory_mode)
    partitions = [[] for _ in worker_gpus]
    if args.evaluation_set == "libero-standard":
        task_groups: dict[tuple[str, int], list[tuple[str, int, int]]] = {}
        for row in rows:
            task_groups.setdefault(row[:2], []).append(row)
        for index, group in enumerate(task_groups.values()):
            partitions[index % len(worker_gpus)].extend(group)
    else:
        for index, row in enumerate(rows):
            partitions[index % len(worker_gpus)].append(row)
    _atomic_json(
        output_dir / "run_manifest.json",
        {
            "schema": (
                "fastwam-p1-visual-libero-plus-run-v2"
                if is_visual_v2_config(training_cfg)
                else "fastwam-p1-dino-libero-plus-run-v1"
            ),
            "status": "RUNNING",
            "evaluation_set": args.evaluation_set,
            "memory_mode": args.memory_mode,
            "task_count": len({row[:2] for row in rows}),
            "episode_count": len(rows),
            "physical_gpus": gpus,
            "workers": len(worker_gpus),
            "workers_per_gpu": args.workers_per_gpu,
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "config": str(Path(args.config).resolve()),
            "visual_backbone_override": args.visual_backbone,
            "prompt_cache_dir": str(Path(args.prompt_cache_dir).resolve()),
            "runtime_text_encoder": False,
            "torch_compile": not args.no_compile,
            "compile_identity": compile_identity,
            "compile_cache_seed": (
                None
                if args.compile_cache_seed is None
                else str(Path(args.compile_cache_seed).expanduser().resolve())
            ),
            "started_unix_seconds": time.time(),
        },
    )
    processes = []
    logs = output_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for worker_id, (gpu, partition) in enumerate(
        zip(worker_gpus, partitions, strict=True)
    ):
        task_file = output_dir / f"worker_{worker_id}_tasks.txt"
        task_file.write_text(
            "".join(
                (
                    f"{suite},{task_id}\n"
                    if args.evaluation_set == "libero-plus"
                    else f"{suite},{task_id},{reset_id}\n"
                )
                for suite, task_id, reset_id in partition
            ),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--config",
            str(Path(args.config).resolve()),
            "--checkpoint",
            str(Path(args.checkpoint).resolve()),
            "--prompt-cache-dir",
            str(Path(args.prompt_cache_dir).resolve()),
            "--classification",
            str(Path(args.classification).resolve()),
            "--output-dir",
            str(output_dir),
            "--worker-id",
            str(worker_id),
            "--physical-gpu",
            str(gpu),
            "--task-file",
            str(task_file),
            "--evaluation-set",
            args.evaluation_set,
            "--memory-mode",
            args.memory_mode,
        ]
        if args.no_compile:
            command.append("--no-compile")
        if args.visual_backbone is not None:
            command.extend(("--visual-backbone", args.visual_backbone))
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
        inductor_cache = (
            output_dir / "torchinductor" / compile_identity / f"worker_{worker_id}"
        )
        triton_cache = output_dir / "triton" / compile_identity / f"worker_{worker_id}"
        if args.compile_cache_seed:
            seed = Path(args.compile_cache_seed).expanduser().resolve()
            seed_inductor = resolve_compile_cache_seed(
                seed,
                cache_name="torchinductor",
                compile_identity=compile_identity,
                worker_id=worker_id,
            )
            seed_triton = resolve_compile_cache_seed(
                seed,
                cache_name="triton",
                compile_identity=compile_identity,
                worker_id=worker_id,
            )
            shutil.copytree(seed_inductor, inductor_cache, dirs_exist_ok=True)
            shutil.copytree(seed_triton, triton_cache, dirs_exist_ok=True)
        environment["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_cache)
        environment["TRITON_CACHE_DIR"] = str(triton_cache)
        log_path = logs / f"worker_{worker_id}_gpu_{gpu}.log"
        handle = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((worker_id, gpu, process, handle, log_path))
        print(f"GPU {gpu}: worker {worker_id}, {len(partition)} tasks", flush=True)
    failures = []
    for worker_id, gpu, process, handle, log_path in processes:
        return_code = process.wait()
        handle.close()
        if return_code:
            failures.append(
                {
                    "worker": worker_id,
                    "gpu": gpu,
                    "rc": return_code,
                    "log": str(log_path),
                }
            )
    manifest_path = output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "FAIL" if failures else "PASS"
    manifest["failures"] = failures
    manifest["finished_unix_seconds"] = time.time()
    _atomic_json(manifest_path, manifest)
    if failures:
        raise SystemExit(f"LIBERO-Plus workers failed: {failures}")


def main() -> None:
    args = _args()
    if args.worker:
        _worker(args)
    else:
        _manager(args)


if __name__ == "__main__":
    main()
