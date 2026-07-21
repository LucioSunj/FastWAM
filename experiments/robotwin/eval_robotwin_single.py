"""
RobotWin single-task evaluation entrypoint (Hydra).

Features:
- Read `configs/sim_robotwin.yaml`.
- Check or create the symlink:
  `RoboTwin/policy/fastwam -> experiments/robotwin/fastwam`.
- Forward config overrides to the official RoboTwin entrypoint
  `script/eval_policy.py` and save logs.

Common arguments:
- `ckpt`: path to the FastWAM checkpoint (required).
- `EVALUATION.task_name`: task name to evaluate (required).
- `gpu_id`: sets `CUDA_VISIBLE_DEVICES`.

Examples:
1) Minimal run
   python experiments/robotwin/eval_robotwin_single.py \
     ckpt=/path/to/ckpt.pt \
     EVALUATION.task_name=click_alarmclock

2) Run with more evaluation overrides
   python experiments/robotwin/eval_robotwin_single.py \
     ckpt=/path/to/ckpt.pt \
     EVALUATION.task_name=click_alarmclock \
     EVALUATION.task_config=demo_randomized \
     EVALUATION.replan_steps=4 \
     EVALUATION.num_inference_steps=4 \
     gpu_id=0
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLICY_NAME = "fastwam_policy"


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_optional_path(path_value: Any, *, base: Path) -> Path | None:
    if path_value is None:
        return None
    text = str(path_value).strip()
    if text == "" or text.lower() in {"none", "null"}:
        return None
    return _resolve_path(text, base=base)


def _resolve_dataset_stats_path(cfg: DictConfig, ckpt_path: Path) -> Path:
    explicit = _resolve_optional_path(cfg.EVALUATION.dataset_stats_path, base=PROJECT_ROOT)
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)

    for parent in list(ckpt_path.parents)[:4]:
        candidates.append((parent / "dataset_stats.json").resolve())

    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    raise FileNotFoundError(
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )


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


def _ensure_policy_symlink(robotwin_root: Path, policy_source_dir: Path) -> Path:
    policy_root = robotwin_root / "policy"
    if not policy_root.is_dir():
        raise FileNotFoundError(f"RoboTwin policy directory not found: {policy_root}")

    policy_target = policy_root / POLICY_NAME
    source_resolved = policy_source_dir.resolve()

    if not policy_target.exists() and not policy_target.is_symlink():
        policy_target.symlink_to(source_resolved, target_is_directory=True)
        return policy_target

    if policy_target.is_symlink():
        target_resolved = policy_target.resolve()
        if target_resolved != source_resolved:
            raise RuntimeError(
                f"Policy symlink conflict: {policy_target} -> {target_resolved}, "
                f"expected -> {source_resolved}"
            )
        return policy_target

    raise RuntimeError(
        f"Path already exists and is not a symlink: {policy_target}. "
        "Please handle it manually to avoid overriding existing policy files."
    )


def _format_override_value(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return str(value)
    return repr(str(value))


def _append_override(overrides: list[str], key: str, value: Any, *, skip_none: bool = True) -> None:
    if skip_none and value is None:
        return
    overrides.extend([f"--{key}", _format_override_value(value)])


def _phase_result_filename(task_config: str) -> str:
    if task_config == "demo_clean":
        return "_result_clean.txt"
    if task_config == "demo_randomized":
        return "_result_random.txt"
    raise ValueError(
        f"Unsupported EVALUATION.task_config={task_config!r}; "
        "expected demo_clean or demo_randomized."
    )


def _phase_video_filename(task_config: str) -> str:
    if task_config == "demo_clean":
        return "_video_clean.mp4"
    if task_config == "demo_randomized":
        return "_video_random.mp4"
    raise ValueError(
        f"Unsupported EVALUATION.task_config={task_config!r}; "
        "expected demo_clean or demo_randomized."
    )


def _phase_video_dirname(task_config: str) -> str:
    if task_config == "demo_clean":
        return "clean"
    if task_config == "demo_randomized":
        return "random"
    raise ValueError(
        f"Unsupported EVALUATION.task_config={task_config!r}; "
        "expected demo_clean or demo_randomized."
    )


def _robotwin_result_path(*, log_file: Path, robotwin_root: Path) -> Path:
    result_path: Path | None = None
    marker = "Data has been saved to "
    for line in log_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if marker not in line:
            continue
        raw_path = line.split(marker, 1)[1].strip()
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = (robotwin_root / candidate).resolve()
        result_path = candidate

    if result_path is None:
        raise FileNotFoundError(f"Could not locate RoboTwin result path in log: {log_file}")
    if not result_path.exists():
        raise FileNotFoundError(f"RoboTwin result file not found: {result_path}")
    return result_path


def _sync_robotwin_result(
    *,
    log_file: Path,
    robotwin_root: Path,
    robotwin_eval_base: Path,
    task_config: str,
) -> Path:
    result_path = _robotwin_result_path(log_file=log_file, robotwin_root=robotwin_root)
    robotwin_eval_base.mkdir(parents=True, exist_ok=True)
    synced_path = robotwin_eval_base / _phase_result_filename(task_config)
    synced_path.write_text(result_path.read_text(encoding="utf-8"), encoding="utf-8")
    return synced_path


def _replace_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target_path.resolve())


def _read_episode_records(result_dir: Path) -> list[dict[str, Any]]:
    manifest_path = result_dir / "_episodes.json"
    if not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Episode manifest must be a list: {manifest_path}")
    return [item for item in payload if isinstance(item, dict)]


def _select_episode_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for record in records:
        outcome = "success" if bool(record.get("success", False)) else "failure"
        if outcome not in selected and record.get("real_video"):
            selected[outcome] = record
    return selected


def _sync_predicted_episode_videos(
    *,
    predicted_video_log_dir: Path,
    episode_index: int,
    target_dir: Path,
) -> list[Path]:
    synced_paths: list[Path] = []
    if not predicted_video_log_dir.exists():
        return synced_paths
    prefix = f"episode{episode_index:04d}_"
    predicted_dir = target_dir / "predicted"
    for predicted_path in sorted(predicted_video_log_dir.glob(prefix + "*_pred.mp4")):
        synced_path = predicted_dir / predicted_path.name
        _replace_symlink(synced_path, predicted_path)
        synced_paths.append(synced_path)
        metadata_path = predicted_path.with_suffix(".json")
        if metadata_path.exists():
            metadata_link = predicted_dir / metadata_path.name
            _replace_symlink(metadata_link, metadata_path)
            synced_paths.append(metadata_link)
    return synced_paths


def _cleanup_unselected_videos(
    *,
    result_dir: Path,
    predicted_video_log_dir: Path,
    selected_episode_indices: set[int],
) -> None:
    for video_path in result_dir.glob("episode*.mp4"):
        stem = video_path.stem
        if not stem.startswith("episode"):
            continue
        try:
            episode_index = int(stem[len("episode"):])
        except ValueError:
            continue
        if episode_index not in selected_episode_indices:
            video_path.unlink(missing_ok=True)

    if predicted_video_log_dir.exists():
        for predicted_path in predicted_video_log_dir.glob("episode*_pred.mp4"):
            name = predicted_path.name
            try:
                episode_index = int(name.split("_", 1)[0].replace("episode", ""))
            except ValueError:
                continue
            if episode_index not in selected_episode_indices:
                predicted_path.unlink(missing_ok=True)
                predicted_path.with_suffix(".json").unlink(missing_ok=True)


def _sync_robotwin_video(
    *,
    log_file: Path,
    robotwin_root: Path,
    robotwin_eval_base: Path,
    task_config: str,
    predicted_video_log_dir: Path,
) -> list[Path]:
    result_path = _robotwin_result_path(log_file=log_file, robotwin_root=robotwin_root)
    result_dir = result_path.parent
    records = _read_episode_records(result_dir)
    if not records:
        return []

    robotwin_eval_base.mkdir(parents=True, exist_ok=True)
    synced_paths: list[Path] = []
    phase_name = _phase_video_dirname(task_config)
    selected_records = _select_episode_records(records)
    selected_episode_indices: set[int] = set()

    for outcome, record in selected_records.items():
        episode_index = int(record["episode_index"])
        real_video_name = str(record["real_video"])
        real_video_path = result_dir / real_video_name
        if not real_video_path.exists():
            continue
        selected_episode_indices.add(episode_index)

        group_dir = robotwin_eval_base / "selected_video_groups" / phase_name / outcome
        real_link = group_dir / "real_operation.mp4"
        _replace_symlink(real_link, real_video_path)
        synced_paths.append(real_link)

        summary_path = group_dir / "episode.json"
        summary_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        synced_paths.append(summary_path)

        shortcut = robotwin_eval_base / f"_video_{phase_name}_{outcome}.mp4"
        _replace_symlink(shortcut, real_video_path)
        synced_paths.append(shortcut)

        synced_paths.extend(_sync_predicted_episode_videos(
            predicted_video_log_dir=predicted_video_log_dir,
            episode_index=episode_index,
            target_dir=group_dir,
        ))

    _cleanup_unselected_videos(
        result_dir=result_dir,
        predicted_video_log_dir=predicted_video_log_dir,
        selected_episode_indices=selected_episode_indices,
    )
    return synced_paths


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if cfg.EVALUATION.task_name is None:
        raise ValueError("`EVALUATION.task_name` must not be None.")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)

    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")

    policy_source_dir = (PROJECT_ROOT / "experiments" / "robotwin" / POLICY_NAME).resolve()
    if not policy_source_dir.is_dir():
        raise FileNotFoundError(f"Policy source directory not found: {policy_source_dir}")

    _ensure_policy_symlink(robotwin_root=robotwin_root, policy_source_dir=policy_source_dir)

    raw_output_dir = str(cfg.EVALUATION.output_dir)
    output_dir = _resolve_path(raw_output_dir, base=PROJECT_ROOT)
    run_ts = output_dir.name
    if run_ts == "":
        raise ValueError(f"Invalid EVALUATION.output_dir (missing run_ts): {output_dir}")
    if Path(os.path.expanduser(os.path.expandvars(raw_output_dir))).is_absolute():
        run_output_dir = output_dir
    else:
        run_output_dir = (
            PROJECT_ROOT
            / "evaluate_results"
            / "robotwin"
            / ckpt_tag
            / run_ts
        )
    run_output_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_output_dir / (
        f"eval_{str(cfg.EVALUATION.task_name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    robotwin_eval_base = run_output_dir / str(cfg.EVALUATION.task_name)
    phase_name = _phase_video_dirname(str(cfg.EVALUATION.task_config))
    predicted_video_log_dir = robotwin_eval_base / "predicted_videos" / phase_name

    sim_cfg_path = (PROJECT_ROOT / "configs" / "sim_robotwin.yaml").resolve()
    sim_task = HydraConfig.get().runtime.choices.get("task")

    dataset_stats_path = _resolve_dataset_stats_path(cfg, ckpt_path)

    overrides: list[str] = []
    _append_override(overrides, "task_name", cfg.EVALUATION.task_name)
    _append_override(overrides, "task_config", cfg.EVALUATION.task_config)
    _append_override(overrides, "ckpt_setting", str(ckpt_path))
    _append_override(overrides, "seed", cfg.seed)
    _append_override(overrides, "policy_name", cfg.EVALUATION.policy_name)
    _append_override(overrides, "instruction_type", cfg.EVALUATION.instruction_type)
    _append_override(overrides, "eval_num_episodes", cfg.EVALUATION.eval_num_episodes)

    _append_override(overrides, "sim_cfg_path", str(sim_cfg_path))
    _append_override(overrides, "sim_task", sim_task)
    _append_override(overrides, "eval_output_dir", str(robotwin_eval_base))
    _append_override(overrides, "mixed_precision", cfg.mixed_precision)
    _append_override(overrides, "device", cfg.EVALUATION.device)
    _append_override(overrides, "dataset_stats_path", str(dataset_stats_path))
    _append_override(overrides, "load_text_encoder", cfg.model.load_text_encoder)
    _append_override(overrides, "skip_dit_load_from_pretrain", cfg.model.skip_dit_load_from_pretrain)
    _append_override(overrides, "text_embedding_cache_dir", cfg.EVALUATION.text_embedding_cache_dir)
    _append_override(overrides, "context_len", cfg.EVALUATION.context_len)
    _append_override(overrides, "action_horizon", cfg.EVALUATION.action_horizon)
    _append_override(overrides, "replan_steps", cfg.EVALUATION.replan_steps)
    _append_override(overrides, "num_inference_steps", cfg.EVALUATION.num_inference_steps)
    _append_override(overrides, "sigma_shift", cfg.EVALUATION.sigma_shift)
    _append_override(overrides, "text_cfg_scale", cfg.EVALUATION.text_cfg_scale)
    _append_override(overrides, "negative_prompt", cfg.EVALUATION.negative_prompt)
    _append_override(overrides, "rand_device", cfg.EVALUATION.rand_device)
    _append_override(overrides, "tiled", cfg.EVALUATION.tiled)
    _append_override(overrides, "timing_enabled", cfg.EVALUATION.timing_enabled)
    _append_override(
        overrides,
        "skip_get_obs_within_replan",
        cfg.EVALUATION.skip_get_obs_within_replan,
    )
    _append_override(overrides, "eval_video_log", cfg.EVALUATION.eval_video_log)
    _append_override(overrides, "eval_video_max_episodes", cfg.EVALUATION.eval_video_max_episodes)
    _append_override(overrides, "render_every_frame", cfg.EVALUATION.render_every_frame)
    _append_override(overrides, "clear_cache_freq", cfg.EVALUATION.get("clear_cache_freq", None))
    _append_override(overrides, "visualize_future_video", cfg.EVALUATION.get("visualize_future_video", False))
    _append_override(overrides, "predicted_video_log_dir", str(predicted_video_log_dir))
    _append_override(overrides, "predicted_video_fps", cfg.EVALUATION.get("predicted_video_fps", 8))
    _append_override(
        overrides,
        "predicted_video_max_episodes",
        cfg.EVALUATION.get("predicted_video_max_episodes", None),
    )
    _append_override(
        overrides,
        "predicted_video_max_replans_per_episode",
        cfg.EVALUATION.get("predicted_video_max_replans_per_episode", None),
    )

    cmd = [
        sys.executable,
        "-u",
        "script/eval_policy.py",
        "--config",
        f"policy/{POLICY_NAME}/deploy_policy.yml",
        "--overrides",
        *overrides,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("FASTWAM_WAN22_COMPONENT_DIR", "/root/autodl-fs/Wan2.2-5B-Robot")
    env.setdefault(
        "FASTWAM_WAN_VIDEO_DIT_CHECKPOINT",
        str(Path(env["FASTWAM_WAN22_COMPONENT_DIR"]) / "checkpoint.safetensors"),
    )

    with open(log_file, "w", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            cmd,
            cwd=str(robotwin_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_f.flush()
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(f"RoboTwin evaluation failed with return code {return_code}. Log: {log_file}")

    synced_result = _sync_robotwin_result(
        log_file=log_file,
        robotwin_root=robotwin_root,
        robotwin_eval_base=robotwin_eval_base,
        task_config=str(cfg.EVALUATION.task_config),
    )
    print(f"Synced RoboTwin result to: {synced_result}")

    synced_videos = _sync_robotwin_video(
        log_file=log_file,
        robotwin_root=robotwin_root,
        robotwin_eval_base=robotwin_eval_base,
        task_config=str(cfg.EVALUATION.task_config),
        predicted_video_log_dir=predicted_video_log_dir,
    )
    if len(synced_videos) > 0:
        print(f"Synced selected video groups under: {robotwin_eval_base / 'selected_video_groups' / phase_name}")
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        print(f"Saved FastWAM predicted videos under: {predicted_video_log_dir}")

    print(f"Evaluation finished successfully. Log saved to: {log_file}")
    OmegaConf.save(
        config=cfg,
        f=str(run_output_dir / f"eval_config_{str(cfg.EVALUATION.task_name)}.yaml"),
    )


if __name__ == "__main__":
    main()
