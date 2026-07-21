import fcntl
import gc
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

import hydra
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from libero.libero import benchmark
from omegaconf import DictConfig, OmegaConf

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.eval_libero_single import (  # noqa: E402
    NumpyEncoder,
    _load_model_checkpoint,
    _mixed_precision_to_model_dtype,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
    _validate_visualize_future_video_cfg,
    run_single_task,
)
from experiments.libero.text_embedding_cache import build_text_context_cache  # noqa: E402
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor  # noqa: E402
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("max", lambda x: max(x), replace=True)
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)], replace=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _env_path(name: str, *, required: bool = True) -> Path | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        if required:
            raise ValueError(f"Environment variable {name} must be set.")
        return None
    return Path(os.path.expanduser(os.path.expandvars(value)))


def _result_exists(output_dir: Path, suite: str, task_id: int) -> bool:
    suite_dir = output_dir / suite
    return any(suite_dir.glob(f"gpu*_task{task_id}_results.json"))


def _parse_task_line(line: str) -> tuple[str, int]:
    suite, task_id = line.strip().split(",", 1)
    return suite, int(task_id)


def _claim_next_task(queue_file: Path, lock_file: Path, output_dir: Path) -> tuple[str, int] | None:
    queue_file.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        lines = queue_file.read_text(encoding="utf-8").splitlines() if queue_file.exists() else []
        selected: tuple[str, int] | None = None
        remaining: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            suite, task_id = _parse_task_line(stripped)
            if _result_exists(output_dir, suite, task_id):
                continue
            if selected is None:
                selected = (suite, task_id)
                continue
            remaining.append(f"{suite},{task_id}")

        queue_file.write_text("\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8")
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return selected


def _append_line_locked(path: Path, lock_file: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock_file.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with path.open("a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _build_runtime(cfg: DictConfig):
    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)
    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    _validate_visualize_future_video_cfg(cfg)
    text_context_cache = build_text_context_cache(cfg)

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError("Persistent LIBERO worker currently supports only env_num=1.")

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])

    output_dir = Path(os.path.expanduser(os.path.expandvars(str(cfg.EVALUATION.output_dir))))
    output_dir.mkdir(parents=True, exist_ok=True)
    if bool(cfg.EVALUATION.get("save_rollout_video", False)):
        (output_dir / "videos").mkdir(parents=True, exist_ok=True)

    return {
        "model": model,
        "processor": processor,
        "action_horizon": action_horizon,
        "input_w": input_w,
        "input_h": input_h,
        "model_device": model_device,
        "output_dir": output_dir,
        "text_context_cache": text_context_cache,
    }


def _extend_initial_states(initial_states, num_trials: int):
    while len(initial_states) < num_trials:
        initial_states.extend(initial_states[: num_trials - len(initial_states)])
    return initial_states


def _run_claimed_task(
    *,
    cfg: DictConfig,
    runtime: dict,
    suites: dict[str, object],
    suite_name: str,
    task_id: int,
) -> dict:
    cfg.EVALUATION.task_suite_name = suite_name
    cfg.EVALUATION.task_id = int(task_id)

    output_dir: Path = runtime["output_dir"]
    suite_output_dir = output_dir / suite_name
    suite_output_dir.mkdir(parents=True, exist_ok=True)

    video_dir = suite_output_dir / "videos"
    if bool(cfg.EVALUATION.get("save_rollout_video", False)):
        video_dir.mkdir(parents=True, exist_ok=True)
    predicted_video_dir = suite_output_dir / "predicted_videos"
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        predicted_video_dir.mkdir(parents=True, exist_ok=True)

    task_suite = suites[suite_name]
    task = task_suite.get_task(task_id)
    initial_states = _extend_initial_states(
        task_suite.get_task_init_states(task_id), int(cfg.EVALUATION.num_trials)
    )

    start_time = time.time()
    results = {
        "task_suite": suite_name,
        "task_id": task_id,
        "task_description": None,
        "successes": 0,
        "total_episodes": int(cfg.EVALUATION.num_trials),
        "gpu_id": int(cfg.gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
    }

    task_results = run_single_task(
        task=task,
        initial_states=initial_states,
        model=runtime["model"],
        processor=runtime["processor"],
        cfg=cfg,
        video_dir=video_dir,
        predicted_video_dir=predicted_video_dir,
        action_horizon=runtime["action_horizon"],
        input_w=runtime["input_w"],
        input_h=runtime["input_h"],
        model_device=runtime["model_device"],
        text_context_cache=runtime["text_context_cache"],
    )
    results.update(task_results)
    results["duration"] = time.time() - start_time

    output_file = suite_output_dir / f"gpu{cfg.gpu_id}_task{task_id}_results.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)
    return results


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def main(cfg: DictConfig):
    PartialState().config = cfg
    queue_file = _env_path("PERSISTENT_QUEUE_FILE")
    lock_file = _env_path("PERSISTENT_LOCK_FILE")
    status_dir = _env_path("PERSISTENT_STATUS_DIR")
    failed_file = _env_path("PERSISTENT_FAILED_TASKS_FILE")
    failed_lock = _env_path("PERSISTENT_FAILED_LOCK_FILE")

    runtime = _build_runtime(cfg)
    output_dir: Path = runtime["output_dir"]
    status_dir.mkdir(parents=True, exist_ok=True)
    status_file = status_dir / f"gpu{cfg.gpu_id}.status"

    benchmark_dict = benchmark.get_benchmark_dict()
    suite_names = list(dict.fromkeys(line.split(",", 1)[0] for line in queue_file.read_text().splitlines() if line))
    suites = {suite_name: benchmark_dict[suite_name]() for suite_name in suite_names}

    logging.info(
        "Persistent LIBERO worker started: gpu_id=%s output_dir=%s queue=%s",
        cfg.gpu_id,
        output_dir,
        queue_file,
    )

    tasks_run = 0
    while True:
        task = _claim_next_task(queue_file, lock_file, output_dir)
        if task is None:
            status_file.write_text(
                f"IDLE|gpu={cfg.gpu_id}|tasks_run={tasks_run}|ts={time.strftime('%Y-%m-%d %H:%M:%S')}\n",
                encoding="utf-8",
            )
            print(f"GPU{cfg.gpu_id}: queue is empty; worker exiting after {tasks_run} tasks.", flush=True)
            return

        suite_name, task_id = task
        status_file.write_text(
            f"RUNNING|gpu={cfg.gpu_id}|suite={suite_name}|task_id={task_id}|tasks_run={tasks_run}"
            f"|ts={time.strftime('%Y-%m-%d %H:%M:%S')}\n",
            encoding="utf-8",
        )
        print(f"GPU{cfg.gpu_id}: running {suite_name} task_id={task_id}", flush=True)
        try:
            result = _run_claimed_task(
                cfg=cfg,
                runtime=runtime,
                suites=suites,
                suite_name=suite_name,
                task_id=task_id,
            )
            tasks_run += 1
            status_file.write_text(
                f"DONE|gpu={cfg.gpu_id}|suite={suite_name}|task_id={task_id}|successes={result['successes']}"
                f"/{result['total_episodes']}|duration={result['duration']:.2f}|tasks_run={tasks_run}"
                f"|ts={time.strftime('%Y-%m-%d %H:%M:%S')}\n",
                encoding="utf-8",
            )
            print(
                f"GPU{cfg.gpu_id}: completed {suite_name} task_id={task_id} "
                f"{result['successes']}/{result['total_episodes']} in {result['duration']:.2f}s",
                flush=True,
            )
        except Exception as exc:  # Keep the worker alive for the remaining queue.
            tb = traceback.format_exc()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            failed_line = f"{timestamp},{suite_name},{task_id},gpu={cfg.gpu_id},error={type(exc).__name__}: {exc}"
            _append_line_locked(failed_file, failed_lock, failed_line)
            status_file.write_text(f"FAILED|{failed_line}\n", encoding="utf-8")
            print(tb, flush=True)
        finally:
            gc_collect_interval = int(cfg.EVALUATION.get("gc_collect_interval", 20))
            if gc_collect_interval > 0 and tasks_run > 0 and tasks_run % gc_collect_interval == 0:
                gc.collect()
            if bool(cfg.EVALUATION.get("empty_cuda_cache_each_task", False)) and torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
