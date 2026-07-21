import hashlib
import io
import json
import logging
import os
import sys
import time
import inspect
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.adaptive_gate import (
    explicit_eval_branch,
    validate_dataset_stats_fingerprint,
)
from fastwam.utils.video_io import save_mp4

logger = logging.getLogger(__name__)

PACKED_CACHE_BIN = "packed_cache.bin"
PACKED_CACHE_INDEX = "packed_cache.index.jsonl"


class PackedTextEmbeddingCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.bin_path = self.cache_dir / PACKED_CACHE_BIN
        self.index_path = self.cache_dir / PACKED_CACHE_INDEX
        self._index: dict[str, tuple[int, int]] = {}
        if self.bin_path.exists() and self.index_path.exists():
            self._load_index()

    @property
    def available(self) -> bool:
        return bool(self._index)

    def _load_index(self) -> None:
        with self.index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                self._index[str(record["name"])] = (int(record["offset"]), int(record["length"]))
        logger.info(
            "Loaded packed text embedding cache: entries=%d bin=%s index=%s",
            len(self._index),
            self.bin_path,
            self.index_path,
        )

    def get(self, name: str) -> Optional[dict[str, torch.Tensor]]:
        span = self._index.get(name)
        if span is None:
            return None
        offset, length = span
        with self.bin_path.open("rb") as f:
            f.seek(offset)
            payload_bytes = f.read(length)
        if len(payload_bytes) != length:
            raise IOError(
                f"Packed cache entry {name} is truncated: expected {length} bytes, got {len(payload_bytes)}"
            )
        return torch.load(io.BytesIO(payload_bytes), map_location="cpu")

    def source(self, name: str) -> str:
        span = self._index.get(name)
        if span is None:
            return str(self.bin_path)
        offset, length = span
        return f"{self.bin_path}:{offset}+{length}"


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_sim_cfg_name(sim_cfg_path: Optional[str], sim_cfg_name: Optional[str]) -> str:
    configs_root = (PROJECT_ROOT / "configs").resolve()
    if not _is_none_like(sim_cfg_path):
        cfg_path = Path(str(sim_cfg_path)).expanduser().resolve()
        try:
            relative = cfg_path.relative_to(configs_root)
        except ValueError as exc:
            raise ValueError(
                f"`sim_cfg_path` must be under {configs_root}, got: {cfg_path}"
            ) from exc
        return relative.as_posix()

    if _is_none_like(sim_cfg_name):
        return "sim_robotwin.yaml"
    return str(sim_cfg_name)


def _compose_sim_cfg(
    sim_cfg_path: Optional[str],
    sim_cfg_name: Optional[str],
    sim_task: Optional[str],
) -> DictConfig:
    config_name = _resolve_sim_cfg_name(sim_cfg_path=sim_cfg_path, sim_cfg_name=sim_cfg_name)
    configs_root = (PROJECT_ROOT / "configs").resolve()
    overrides = []
    if not _is_none_like(sim_task):
        overrides.append(f"task={str(sim_task)}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def _resolve_dataset_stats_path(dataset_stats_path: Optional[str]) -> Path:
    if _is_none_like(dataset_stats_path):
        raise FileNotFoundError(
            "`dataset_stats_path` is required. "
            "Please pass it from eval entrypoint overrides."
        )
    resolved = Path(str(dataset_stats_path)).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset stats path not found: {resolved}")
    return resolved


def _resolve_text_embedding_cache_dir(text_embedding_cache_dir: Optional[str]) -> Path:
    if _is_none_like(text_embedding_cache_dir):
        raise FileNotFoundError(
            "`text_embedding_cache_dir` is required for RoboTwin FastWAM eval. "
            "Run experiments/robotwin/precompute_robotwin_text_embeds.py first."
        )
    raw_path = Path(os.path.expanduser(os.path.expandvars(str(text_embedding_cache_dir))))
    if not raw_path.is_absolute():
        raw_path = (PROJECT_ROOT / raw_path).resolve()
    resolved = raw_path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"Text embedding cache dir not found: {resolved}. "
            "Run experiments/robotwin/precompute_robotwin_text_embeds.py first."
        )
    return resolved


def _resolve_optional_output_dir(path_value: Optional[str]) -> Optional[Path]:
    if _is_none_like(path_value):
        return None
    raw_path = Path(os.path.expanduser(os.path.expandvars(str(path_value))))
    if not raw_path.is_absolute():
        raw_path = (Path.cwd() / raw_path).resolve()
    return raw_path.resolve()


def _resize_rgb(image: np.ndarray, size_wh: tuple[int, int]) -> np.ndarray:
    pil_image = Image.fromarray(image.astype(np.uint8), mode="RGB")
    resized = pil_image.resize(size_wh, resample=Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


class WorldActionRobotWinPolicy:
    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        action_horizon: int,
        replan_steps: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        text_cfg_scale: float,
        negative_prompt: str,
        rand_device: str,
        tiled: bool,
        timing_enabled: bool,
        num_video_frames: int,
        text_embedding_cache_dir: Path,
        context_len: int,
        visualize_future_video: bool,
        predicted_video_log_dir: Optional[Path],
        predicted_video_fps: int,
        predicted_video_max_episodes: Optional[int],
        predicted_video_max_replans_per_episode: Optional[int],
        force_branch: str = "base",
    ) -> None:
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        model_cfg_copy.load_text_encoder = False

        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        self.model = self.model.to(device).eval()

        self.processor: FastWAMProcessor = instantiate(processor_cfg).eval()
        validate_dataset_stats_fingerprint(self.model, dataset_stats_path)
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.action_horizon = int(action_horizon)
        self.replan_steps = int(max(1, min(replan_steps, action_horizon)))
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.timing_enabled = bool(timing_enabled)
        self._num_video_frames = int(num_video_frames)
        self.text_embedding_cache_dir = Path(text_embedding_cache_dir)
        self._packed_text_cache = PackedTextEmbeddingCache(self.text_embedding_cache_dir)
        self.context_len = int(context_len)
        self._text_context_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.visualize_future_video = bool(visualize_future_video)
        self.predicted_video_log_dir = predicted_video_log_dir
        self.predicted_video_fps = int(max(1, predicted_video_fps))
        self.predicted_video_max_episodes = predicted_video_max_episodes
        self.predicted_video_max_replans_per_episode = predicted_video_max_replans_per_episode
        if self.visualize_future_video:
            if self.predicted_video_log_dir is None:
                self.predicted_video_log_dir = (Path.cwd() / "fastwam_predicted_videos").resolve()
            self.predicted_video_log_dir.mkdir(parents=True, exist_ok=True)

        self.force_branch = str(force_branch)
        # Validate once at construction; vanilla models return an empty mapping.
        explicit_eval_branch(
            self.model,
            "infer_joint" if self.visualize_future_video else "infer_action",
            self.force_branch,
            require_video=self.visualize_future_video,
        )

        self.pending_actions: deque[np.ndarray] = deque()
        self.episode_count = 0
        self.replan_count = 0
        self.step_count = 0
        self._timing_rollout = {"infer_s": 0.0, "sim_s": 0.0}

        logger.info(
            "Initialized WorldActionRobotWinPolicy | ckpt=%s | stats=%s | horizon=%d | replan=%d | text_cache=%s",
            checkpoint_path,
            dataset_stats_path,
            self.action_horizon,
            self.replan_steps,
            self.text_embedding_cache_dir,
        )

    def _should_save_predicted_video(self) -> bool:
        if not self.visualize_future_video or self.predicted_video_log_dir is None:
            return False
        episode_idx = max(0, self.episode_count - 1)
        if self.predicted_video_max_episodes is not None and episode_idx >= self.predicted_video_max_episodes:
            return False
        if (
            self.predicted_video_max_replans_per_episode is not None
            and self.replan_count >= self.predicted_video_max_replans_per_episode
        ):
            return False
        return True

    def _save_predicted_video_clip(self, frames: list[Image.Image], instruction: str) -> None:
        if not self._should_save_predicted_video():
            return
        if len(frames) == 0:
            logger.warning("Skip empty predicted future video clip.")
            return

        assert self.predicted_video_log_dir is not None
        episode_idx = max(0, self.episode_count - 1)
        replan_idx = self.replan_count
        step_idx = self.step_count
        filename = f"episode{episode_idx:04d}_replan{replan_idx:04d}_step{step_idx:06d}_pred.mp4"
        video_path = self.predicted_video_log_dir / filename
        save_mp4(frames, str(video_path), fps=self.predicted_video_fps)

        metadata = {
            "episode": episode_idx,
            "replan": replan_idx,
            "step": step_idx,
            "num_frames": len(frames),
            "fps": self.predicted_video_fps,
            "instruction": instruction,
        }
        video_path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        logger.info("Saved FastWAM predicted future video: %s", video_path)

    def _get_cached_text_context(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._text_context_cache.get(prompt)
        if cached is not None:
            return cached

        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_name = f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt"
        cache_path = self.text_embedding_cache_dir / cache_name
        payload = self._packed_text_cache.get(cache_name)
        cache_source = self._packed_text_cache.source(cache_name) if payload is not None else str(cache_path)
        if payload is None:
            if not cache_path.exists():
                raise FileNotFoundError(
                    f"Missing precomputed text embedding cache: {cache_path} "
                    f"or packed entry {cache_name} under {self.text_embedding_cache_dir}. "
                    f"Prompt: {prompt!r}"
                )
            payload = torch.load(str(cache_path), map_location="cpu")

        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L,D], got {tuple(context.shape)} in {cache_source}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got {tuple(context_mask.shape)} in {cache_source}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_source}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_source}"
            )

        context = context.detach().to(device="cpu", dtype=torch.bfloat16).contiguous().clone()
        context_mask = context_mask.detach().to(device="cpu", dtype=torch.bool).contiguous()
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        cached = (context, context_mask)
        self._text_context_cache[prompt] = cached
        return cached

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]

        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")

        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _build_robotwin_image_tensor(self, observation: Dict[str, Any]) -> torch.Tensor:
        obs_data = observation["observation"]
        head = _resize_rgb(obs_data["head_camera"]["rgb"], (320, 256))
        left = _resize_rgb(obs_data["left_camera"]["rgb"], (160, 128))
        right = _resize_rgb(obs_data["right_camera"]["rgb"], (160, 128))
        bottom = np.concatenate([left, right], axis=1)
        image = np.concatenate([head, bottom], axis=0)  # [384, 320, 3]

        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device,
            dtype=self.model.torch_dtype,
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        return image_tensor

    def _infer_action_chunk(self, observation: Dict[str, Any], instruction: str) -> np.ndarray:
        image_tensor = self._build_robotwin_image_tensor(observation)
        state_vector = np.asarray(observation["joint_action"]["vector"], dtype=np.float32)
        proprio = self._normalize_state(state_vector)

        prompt = DEFAULT_PROMPT.format(task=instruction)
        context, context_mask = self._get_cached_text_context(prompt)
        infer_kwargs = {
            "prompt": None,
            "context": context,
            "context_mask": context_mask,
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        infer_method_name = "infer_joint" if self.visualize_future_video else "infer_action"
        infer_method = getattr(self.model, infer_method_name)
        if "num_video_frames" in inspect.signature(infer_method).parameters:
            infer_kwargs["num_video_frames"] = int(self._num_video_frames)
        infer_kwargs.update(
            explicit_eval_branch(
                self.model,
                infer_method_name,
                self.force_branch,
                require_video=self.visualize_future_video,
            )
        )
        infer_t0 = time.perf_counter() if self.timing_enabled else 0.0
        with torch.no_grad():
            pred = infer_method(**infer_kwargs)
        if self.timing_enabled:
            self._timing_rollout["infer_s"] += time.perf_counter() - infer_t0
        if self.visualize_future_video:
            pred_video = pred.get("video")
            if pred_video is None:
                raise KeyError("`visualize_future_video=true` requires model.infer_joint() to return `video`.")
            self._save_predicted_video_clip(list(pred_video), instruction=instruction)

        action_tensor = pred["action"]  # [T, D]
        action_chunk = self._denormalize_action(action_tensor)[0]  # [T, D]
        return action_chunk

    def _fill_action_queue(self, observation: Dict[str, Any], instruction: str) -> None:
        action_chunk = self._infer_action_chunk(observation=observation, instruction=instruction)
        self.replan_count += 1
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        for i in range(n_exec):
            self.pending_actions.append(np.asarray(action_chunk[i], dtype=np.float32))

    def should_request_observation(self) -> bool:
        return not self.pending_actions

    def step(self, task_env, observation: Optional[Dict[str, Any]]) -> None:
        if not self.pending_actions:
            if observation is None:
                raise ValueError(
                    "Observation is required when action queue is empty "
                    "(replan step for fastwam)."
                )
            instruction = task_env.get_instruction()
            self._fill_action_queue(observation=observation, instruction=instruction)

        if not self.pending_actions:
            logger.warning("No action generated; skip current eval step.")
            return

        action = self.pending_actions.popleft()
        sim_t0 = time.perf_counter() if self.timing_enabled else 0.0
        task_env.take_action(action, action_type="qpos")
        if self.timing_enabled:
            self._timing_rollout["sim_s"] += time.perf_counter() - sim_t0
        self.step_count += 1

    def reset_timing_rollout(self) -> None:
        self._timing_rollout["infer_s"] = 0.0
        self._timing_rollout["sim_s"] = 0.0

    def get_timing_rollout(self) -> Dict[str, float]:
        return {
            "infer_s": float(self._timing_rollout["infer_s"]),
            "sim_s": float(self._timing_rollout["sim_s"]),
        }

    def reset(self) -> None:
        self.pending_actions.clear()
        self.episode_count += 1
        self.replan_count = 0
        self.step_count = 0
        self.reset_timing_rollout()


def encode_obs(observation: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return observation


def get_model(usr_args: Dict[str, Any]):
    sim_cfg_path = usr_args.get("sim_cfg_path")
    sim_cfg_name = usr_args.get("sim_cfg_name")
    sim_task = usr_args.get("sim_task")
    cfg = _compose_sim_cfg(
        sim_cfg_path=sim_cfg_path,
        sim_cfg_name=sim_cfg_name,
        sim_task=sim_task,
    )

    checkpoint_path = usr_args.get("ckpt_setting")
    if _is_none_like(checkpoint_path):
        raise ValueError("`ckpt_setting` is required and must be a valid checkpoint path.")

    device = str(usr_args.get("device") or cfg.EVALUATION.get("device") or "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; fallback device to cpu.")
        device = "cpu"

    mixed_precision = str(usr_args.get("mixed_precision") or cfg.get("mixed_precision", "bf16"))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
    if not _is_none_like(usr_args.get("load_text_encoder")):
        model_cfg.load_text_encoder = _parse_bool(usr_args.get("load_text_encoder"))
    if not _is_none_like(usr_args.get("skip_dit_load_from_pretrain")):
        model_cfg.skip_dit_load_from_pretrain = _parse_bool(usr_args.get("skip_dit_load_from_pretrain"))

    dataset_stats_path = _resolve_dataset_stats_path(
        dataset_stats_path=usr_args.get("dataset_stats_path"),
    )
    text_embedding_cache_dir = _resolve_text_embedding_cache_dir(
        usr_args.get("text_embedding_cache_dir", cfg.EVALUATION.get("text_embedding_cache_dir"))
    )
    context_len = _parse_optional_int(usr_args.get("context_len"))
    if context_len is None:
        context_len = int(cfg.EVALUATION.get("context_len", cfg.data.train.get("context_len", 128)))

    action_horizon = _parse_optional_int(usr_args.get("action_horizon"))
    if action_horizon is None:
        eval_horizon = _parse_optional_int(cfg.EVALUATION.get("action_horizon"))
        action_horizon = eval_horizon if eval_horizon is not None else int(cfg.data.train.num_frames) - 1
    if action_horizon <= 0:
        raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")

    replan_steps = _parse_optional_int(usr_args.get("replan_steps"))
    if replan_steps is None:
        replan_steps = int(cfg.EVALUATION.get("replan_steps", 8))

    num_inference_steps = _parse_optional_int(usr_args.get("num_inference_steps"))
    if num_inference_steps is None:
        num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.eval_num_inference_steps))

    sigma_shift = _parse_optional_float(usr_args.get("sigma_shift"))
    if sigma_shift is None:
        sigma_shift = _parse_optional_float(cfg.EVALUATION.get("sigma_shift"))

    seed = _parse_optional_int(usr_args.get("seed"))
    text_cfg_scale = float(usr_args.get("text_cfg_scale", cfg.EVALUATION.get("text_cfg_scale", 1.0)))
    negative_prompt = str(usr_args.get("negative_prompt", cfg.EVALUATION.get("negative_prompt", "")))
    rand_device = str(usr_args.get("rand_device", cfg.EVALUATION.get("rand_device", "cpu")))
    tiled = _parse_bool(usr_args.get("tiled", cfg.EVALUATION.get("tiled", False)))
    timing_enabled = _parse_bool(
        usr_args.get("timing_enabled", cfg.EVALUATION.get("timing_enabled", False))
    )
    force_branch = str(usr_args.get("force_branch", cfg.EVALUATION.get("force_branch", "base")))
    visualize_future_video = _parse_bool(
        usr_args.get("visualize_future_video", cfg.EVALUATION.get("visualize_future_video", False))
    )
    predicted_video_log_dir = _resolve_optional_output_dir(
        usr_args.get("predicted_video_log_dir", cfg.EVALUATION.get("predicted_video_log_dir", None))
    )
    predicted_video_fps = int(usr_args.get("predicted_video_fps", cfg.EVALUATION.get("predicted_video_fps", 8)))
    predicted_video_max_episodes = _parse_optional_int(
        usr_args.get(
            "predicted_video_max_episodes",
            cfg.EVALUATION.get("predicted_video_max_episodes", None),
        )
    )
    predicted_video_max_replans_per_episode = _parse_optional_int(
        usr_args.get(
            "predicted_video_max_replans_per_episode",
            cfg.EVALUATION.get("predicted_video_max_replans_per_episode", None),
        )
    )

    policy = WorldActionRobotWinPolicy(
        model_cfg=model_cfg,
        processor_cfg=cfg.data.train.processor,
        checkpoint_path=str(checkpoint_path),
        dataset_stats_path=dataset_stats_path,
        device=device,
        model_dtype=model_dtype,
        action_horizon=action_horizon,
        replan_steps=replan_steps,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        timing_enabled=timing_enabled,
        num_video_frames=(int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1,
        force_branch=force_branch,
        text_embedding_cache_dir=text_embedding_cache_dir,
        context_len=context_len,
        visualize_future_video=visualize_future_video,
        predicted_video_log_dir=predicted_video_log_dir,
        predicted_video_fps=predicted_video_fps,
        predicted_video_max_episodes=predicted_video_max_episodes,
        predicted_video_max_replans_per_episode=predicted_video_max_replans_per_episode,
    )
    return policy


def eval(TASK_ENV, model, observation: Optional[Dict[str, Any]]):
    obs = encode_obs(observation)
    model.step(TASK_ENV, obs)


def reset_model(model):
    model.reset()
