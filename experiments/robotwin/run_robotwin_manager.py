import ast
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SINGLE_ENTRY = PROJECT_ROOT / "experiments" / "robotwin" / "eval_robotwin_single.py"
PRECOMPUTE_TEXT_ENTRY = PROJECT_ROOT / "experiments" / "robotwin" / "precompute_robotwin_text_embeds.py"
EVAL_STEP_LIMIT_FILE = PROJECT_ROOT / "third_party" / "RoboTwin" / "task_config" / "_eval_step_limit.yml"
TERMINATE_TIMEOUT_SEC = 10
POLL_INTERVAL_SEC = 2


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 >= len(parts):
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        task_name = parts[runs_idx + 1]
        date_dir = parts[runs_idx + 2]
        if task_name == "" or date_dir == "":
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        return f"{task_name}_{date_dir}"
    return ckpt_path.stem


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    if key in {
        "ckpt",
        "gpu_id",
        "EVALUATION.task_name",
        "EVALUATION.task_config",
        "EVALUATION.output_dir",
    }:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def _collect_worker_overrides() -> list[str]:
    return [ov for ov in HydraConfig.get().overrides.task if not _is_blocked_override(ov)]


def _load_all_tasks() -> list[str]:
    if not EVAL_STEP_LIMIT_FILE.exists():
        raise FileNotFoundError(f"Task list file not found: {EVAL_STEP_LIMIT_FILE}")
    with EVAL_STEP_LIMIT_FILE.open("r", encoding="utf-8") as f:
        task_map = yaml.safe_load(f)
    if not isinstance(task_map, dict) or len(task_map) == 0:
        raise ValueError(f"Invalid task map in: {EVAL_STEP_LIMIT_FILE}")
    tasks = list(task_map.keys())
    # Keep original order and remove duplicates.
    seen = set()
    dedup_tasks: list[str] = []
    for task in tasks:
        if task in seen:
            continue
        seen.add(task)
        dedup_tasks.append(task)
    return dedup_tasks


def _parse_success_rate(result_file: Path) -> float:
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")
    text = result_file.read_text(encoding="utf-8")
    last_value: float | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        try:
            last_value = float(stripped)
        except ValueError:
            continue
    if last_value is None:
        raise ValueError(f"Failed to parse success rate from: {result_file}")
    return last_value

def _extract_missing_text_prompt(log_file: Path) -> str | None:
    if not log_file.exists():
        return None
    for line in reversed(log_file.read_text(encoding="utf-8", errors="replace").splitlines()):
        marker = "Prompt: "
        if "Missing precomputed text embedding cache:" not in line or marker not in line:
            continue
        raw_prompt = line.split(marker, 1)[1].strip()
        try:
            prompt = ast.literal_eval(raw_prompt)
        except (SyntaxError, ValueError):
            return None
        return prompt if isinstance(prompt, str) and prompt else None


def _phase_result_filename(phase: str) -> str:
    if phase == "clean":
        return "_result_clean.txt"
    if phase == "random":
        return "_result_random.txt"
    raise ValueError(f"Unsupported phase: {phase}")


def _mean_or_none(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    if len(valid) == 0:
        return None
    return float(sum(valid) / len(valid))


def _to_jsonable(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


@dataclass
class RunningState:
    task_name: str
    gpu_id: int
    phase: str  # "clean" | "random"
    process: subprocess.Popen[str]


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if not SINGLE_ENTRY.exists():
        raise FileNotFoundError(f"Single evaluation entry not found: {SINGLE_ENTRY}")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)

    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")

    num_gpus = int(cfg.MULTIRUN.num_gpus)
    if num_gpus <= 0:
        raise ValueError("`MULTIRUN.num_gpus` must be > 0.")
    gpu_offset = int(cfg.MULTIRUN.get("gpu_offset", 0))
    if gpu_offset < 0:
        raise ValueError("`MULTIRUN.gpu_offset` must be >= 0.")
    max_tasks_per_gpu = int(cfg.MULTIRUN.max_tasks_per_gpu)
    if max_tasks_per_gpu <= 0:
        raise ValueError("`MULTIRUN.max_tasks_per_gpu` must be > 0.")
    gpu_ids = list(range(gpu_offset, gpu_offset + num_gpus))

    raw_output_dir = str(cfg.EVALUATION.output_dir)
    output_dir = _resolve_path(raw_output_dir, base=PROJECT_ROOT)
    run_ts = output_dir.name
    if run_ts == "":
        raise ValueError(f"Invalid EVALUATION.output_dir (missing run_ts): {output_dir}")
    if Path(os.path.expanduser(os.path.expandvars(raw_output_dir))).is_absolute():
        run_output_dir = output_dir
    else:
        run_output_dir = PROJECT_ROOT / "evaluate_results" / "robotwin" / ckpt_tag / run_ts
    run_output_dir.mkdir(parents=True, exist_ok=True)

    manager_log = run_output_dir / "manager.log"
    failed_tasks_file = run_output_dir / "failed_tasks.txt"
    summary_csv = run_output_dir / "summary.csv"
    summary_json = run_output_dir / "summary.json"

    task_name_cfg = cfg.EVALUATION.task_name
    if task_name_cfg is None or str(task_name_cfg).strip() == "":
        tasks = _load_all_tasks()
    else:
        tasks = [str(task_name_cfg)]

    extra_overrides = _collect_worker_overrides()

    task_rates: dict[str, dict[str, float | None]] = {
        task: {"clean": None, "random": None} for task in tasks
    }
    failed_records: list[dict[str, Any]] = []
    pending_tasks = deque(tasks)
    running_states: list[RunningState] = []

    phase_to_task_config = {
        "clean": "demo_clean",
        "random": "demo_randomized",
    }

    text_cache_value = cfg.EVALUATION.get("text_embedding_cache_dir")
    if text_cache_value is None or str(text_cache_value).strip() == "":
        raise ValueError("`EVALUATION.text_embedding_cache_dir` is required.")
    text_cache_dir = _resolve_path(str(text_cache_value), base=PROJECT_ROOT)
    context_len = int(cfg.EVALUATION.get("context_len", 128))
    recovery_dir = run_output_dir / "missing_text_cache_recovery"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    max_missing_cache_retries = int(cfg.EVALUATION.get("max_missing_cache_retries", 64))
    missing_cache_retries: dict[tuple[str, str], int] = {}

    def log(msg: str) -> None:

        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def latest_worker_log(task_name: str) -> Path | None:
        candidates = list(run_output_dir.glob(f"eval_{task_name}_*.log"))
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime_ns)

    def recover_missing_text_cache(state: RunningState) -> bool:
        log_file = latest_worker_log(state.task_name)
        if log_file is None:
            return False
        prompt = _extract_missing_text_prompt(log_file)
        if prompt is None:
            return False

        retry_key = (state.task_name, state.phase)
        retry_count = missing_cache_retries.get(retry_key, 0) + 1
        if retry_count > max_missing_cache_retries:
            log(
                f"missing text cache retry limit exceeded: task={state.task_name} "
                f"phase={state.phase} gpu={state.gpu_id} retries={retry_count - 1}"
            )
            return False
        missing_cache_retries[retry_key] = retry_count

        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_name = f"{prompt_hash}.t5_len{context_len}.wan22ti2v5b.pt"
        cache_path = text_cache_dir / cache_name
        prompt_file = recovery_dir / f"{state.task_name}_{state.phase}_{prompt_hash}.json"
        recovery_log = recovery_dir / f"{state.task_name}_{state.phase}_{prompt_hash}.log"
        prompt_file.write_text(
            json.dumps([prompt], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        cmd = [
            sys.executable,
            str(PRECOMPUTE_TEXT_ENTRY),
            "--robotwin-root",
            str(robotwin_root),
            "--cache-dir",
            str(text_cache_dir),
            "--prompts-json",
            str(prompt_file),
            "--skip-collect",
            "--context-len",
            str(context_len),
            "--device",
            "cuda:0",
            "--batch-size",
            "1",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(state.gpu_id)
        env.setdefault("FASTWAM_WAN22_COMPONENT_DIR", "/root/autodl-fs/Wan2.2-5B-Robot")
        log(
            f"recover missing text cache: task={state.task_name} phase={state.phase} "
            f"gpu={state.gpu_id} retry={retry_count} cache={cache_name}"
        )
        with recovery_log.open("w", encoding="utf-8") as recovery_f:
            recovery = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=recovery_f,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if recovery.returncode != 0 or not cache_path.is_file():
            log(
                f"missing text cache recovery failed: task={state.task_name} "
                f"phase={state.phase} gpu={state.gpu_id} return_code={recovery.returncode} "
                f"log={recovery_log}"
            )
            return False
        log(
            f"missing text cache recovered: task={state.task_name} phase={state.phase} "
            f"gpu={state.gpu_id} cache={cache_name}"
        )
        return True

    def build_cmd(*, task_name: str, gpu_id: int, phase: str) -> list[str]:
        task_config = phase_to_task_config[phase]
        cmd = [
            sys.executable,
            str(SINGLE_ENTRY),
            f"ckpt={str(ckpt_path)}",
            f"gpu_id={gpu_id}",
            f"EVALUATION.task_name={task_name}",
            f"EVALUATION.task_config={task_config}",
            f"EVALUATION.output_dir={str(output_dir)}",
        ]
        cmd.extend(extra_overrides)
        return cmd

    def launch_phase(task_name: str, gpu_id: int, phase: str) -> RunningState:
        cmd = build_cmd(task_name=task_name, gpu_id=gpu_id, phase=phase)
        log(
            f"launch task={task_name} phase={phase} gpu={gpu_id} "
            f"cmd={' '.join(cmd)}"
        )
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            text=True,
        )
        return RunningState(
            task_name=task_name,
            gpu_id=gpu_id,
            phase=phase,
            process=process,
        )

    def terminate_all_running() -> None:
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            log(f"terminating task={state.task_name} phase={state.phase} gpu={state.gpu_id}")
            state.process.terminate()
        deadline = time.time() + TERMINATE_TIMEOUT_SEC
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            remaining = max(0.0, deadline - time.time())
            try:
                state.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                log(f"killing task={state.task_name} phase={state.phase} gpu={state.gpu_id}")
                state.process.kill()
                state.process.wait()

    def gpu_running_count(gpu_id: int) -> int:
        count = 0
        for state in running_states:
            if state.gpu_id != gpu_id:
                continue
            if state.process.poll() is None:
                count += 1
        return count

    def phase_result_file(task_name: str, phase: str) -> Path:
        return run_output_dir / task_name / _phase_result_filename(phase)

    def try_record_existing_result(task_name: str, phase: str) -> bool:
        if task_rates[task_name][phase] is not None:
            return True
        result_file = phase_result_file(task_name, phase)
        if not result_file.exists():
            return False
        success_rate = _parse_success_rate(result_file)
        task_rates[task_name][phase] = success_rate
        log(
            f"skip existing task={task_name} phase={phase} "
            f"success_rate={success_rate:.4f}"
        )
        return True

    def try_launch_pending(gpu_id: int) -> None:
        while len(pending_tasks) > 0 and gpu_running_count(gpu_id) < max_tasks_per_gpu:
            task_name = pending_tasks.popleft()

            clean_done = try_record_existing_result(task_name, "clean")
            if not clean_done:
                running_states.append(launch_phase(task_name=task_name, gpu_id=gpu_id, phase="clean"))
                continue

            random_done = try_record_existing_result(task_name, "random")
            if not random_done:
                running_states.append(launch_phase(task_name=task_name, gpu_id=gpu_id, phase="random"))
                continue

    def write_outputs() -> None:
        clean_mean = _mean_or_none([task_rates[t]["clean"] for t in tasks])
        random_mean = _mean_or_none([task_rates[t]["random"] for t in tasks])

        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_name", "clean_success_rate", "random_success_rate"])
            for task in tasks:
                writer.writerow(
                    [
                        task,
                        task_rates[task]["clean"],
                        task_rates[task]["random"],
                    ]
                )
            writer.writerow(["__overall__", clean_mean, random_mean])

        payload = {
            "per_task": [
                {
                    "task_name": task,
                    "clean_success_rate": _to_jsonable(task_rates[task]["clean"]),
                    "random_success_rate": _to_jsonable(task_rates[task]["random"]),
                }
                for task in tasks
            ],
            "overall": {
                "clean_mean_success_rate": _to_jsonable(clean_mean),
                "random_mean_success_rate": _to_jsonable(random_mean),
            },
        }
        summary_json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        with failed_tasks_file.open("w", encoding="utf-8") as f:
            for rec in failed_records:
                f.write(
                    f"{rec['task_name']},{rec['phase']},gpu={rec['gpu_id']},"
                    f"return_code={rec['return_code']},reason={rec['reason']}\n"
                )

    log(
        f"manager start tasks={len(tasks)} gpu_ids={gpu_ids} "
        f"max_tasks_per_gpu={max_tasks_per_gpu} output_dir={run_output_dir}"
    )

    # Launch initial tasks for each GPU up to capacity.
    for gpu_id in gpu_ids:
        try_launch_pending(gpu_id)

    has_failure = False
    failure_message = ""

    while len(running_states) > 0:
        progressed = False
        for state in list(running_states):
            gpu_id = state.gpu_id
            return_code = state.process.poll()
            if return_code is None:
                continue
            progressed = True
            running_states.remove(state)

            if return_code != 0:
                if recover_missing_text_cache(state):
                    log(
                        f"retry phase after text cache recovery: task={state.task_name} "
                        f"phase={state.phase} gpu={gpu_id}"
                    )
                    running_states.append(
                        launch_phase(
                            task_name=state.task_name,
                            gpu_id=gpu_id,
                            phase=state.phase,
                        )
                    )
                    continue
                has_failure = True
                failure_message = (
                    f"worker failed: task={state.task_name}, phase={state.phase}, "
                    f"gpu={gpu_id}, return_code={return_code}"
                )
                failed_records.append(
                    {
                        "task_name": state.task_name,
                        "phase": state.phase,
                        "gpu_id": gpu_id,
                        "return_code": return_code,
                        "reason": "process_failed",
                    }
                )
                log(failure_message)
                terminate_all_running()
                running_states.clear()
                break

            result_file = phase_result_file(state.task_name, state.phase)
            try:
                success_rate = _parse_success_rate(result_file)
            except Exception as exc:
                has_failure = True
                failure_message = (
                    f"result parse failed: task={state.task_name}, phase={state.phase}, "
                    f"gpu={gpu_id}, error={repr(exc)}"
                )
                failed_records.append(
                    {
                        "task_name": state.task_name,
                        "phase": state.phase,
                        "gpu_id": gpu_id,
                        "return_code": return_code,
                        "reason": "result_parse_failed",
                    }
                )
                log(failure_message)
                terminate_all_running()
                running_states.clear()
                break

            task_rates[state.task_name][state.phase] = success_rate
            log(
                f"done task={state.task_name} phase={state.phase} gpu={gpu_id} "
                f"success_rate={success_rate:.4f}"
            )

            if state.phase == "clean":
                if not try_record_existing_result(state.task_name, "random"):
                    running_states.append(launch_phase(
                        task_name=state.task_name,
                        gpu_id=gpu_id,
                        phase="random",
                    ))
                else:
                    try_launch_pending(gpu_id)
                continue

            try_launch_pending(gpu_id)

        if has_failure:
            break
        if not progressed:
            time.sleep(POLL_INTERVAL_SEC)

    # Mark not started tasks when failure happened.
    if has_failure:
        for task_name in pending_tasks:
            failed_records.append(
                {
                    "task_name": task_name,
                    "phase": "not_started",
                    "gpu_id": -1,
                    "return_code": -1,
                    "reason": "aborted_not_started",
                }
            )

    write_outputs()
    log(f"summary saved: {summary_csv} and {summary_json}")

    if has_failure:
        raise RuntimeError(failure_message)

    log("manager finished successfully")


if __name__ == "__main__":
    main()
