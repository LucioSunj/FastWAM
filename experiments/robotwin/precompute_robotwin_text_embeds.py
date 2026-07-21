import argparse
import hashlib
import io
import itertools
import json
import logging
import os
import random
import re
import sys
import uuid
from pathlib import Path
from contextlib import nullcontext
from typing import Any

import torch
import yaml
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.models.wan22.helpers.loader import (
    _ensure_component_path,
    _load_registered_model,
    _resolve_configs,
    _wan22_component_dir,
)
from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
CACHE_ENCODER_ID = "wan22ti2v5b"

logger = logging.getLogger("precompute_robotwin_text_embeds")


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _load_task_names(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        task_map = yaml.safe_load(f)
    if not isinstance(task_map, dict) or not task_map:
        raise ValueError(f"Invalid task list file: {path}")
    return list(task_map.keys())


def _safe_close_env(task_env: Any) -> None:
    try:
        task_env.close_env()
    except Exception:
        pass


def _prepare_robotwin_imports(robotwin_root: Path):
    old_cwd = Path.cwd()
    os.chdir(robotwin_root)
    for item in [robotwin_root, robotwin_root / "policy", robotwin_root / "description" / "utils"]:
        item_str = str(item)
        if item_str not in sys.path:
            sys.path.insert(0, item_str)

    from envs import CONFIGS_PATH
    from envs.utils.create_actor import UnStableError
    from generate_episode_instructions import filter_instructions, generate_episode_descriptions, load_task_instructions
    from script.eval_policy import class_decorator, get_embodiment_config

    return {
        "old_cwd": old_cwd,
        "CONFIGS_PATH": CONFIGS_PATH,
        "UnStableError": UnStableError,
        "filter_instructions": filter_instructions,
        "generate_episode_descriptions": generate_episode_descriptions,
        "load_task_instructions": load_task_instructions,
        "class_decorator": class_decorator,
        "get_embodiment_config": get_embodiment_config,
    }


def _build_task_args(
    *,
    robotwin_root: Path,
    task_name: str,
    task_config: str,
    eval_num_episodes: int,
    imports: dict[str, Any],
) -> dict[str, Any]:
    with (robotwin_root / "task_config" / f"{task_config}.yml").open("r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = "precompute_text_embeddings"
    args["episode_num"] = int(eval_num_episodes)
    args["eval_mode"] = True
    args["skip_render_without_obs"] = True
    args["eval_obs_save_dir"] = None
    args["eval_video_save_dir"] = None
    args["eval_video_log"] = False

    embodiment_type = args.get("embodiment")
    configs_path = imports["CONFIGS_PATH"]
    with open(os.path.join(configs_path, "_embodiment_config.yml"), "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(name: str) -> str:
        robot_file = embodiment_types[name]["file_path"]
        if robot_file is None:
            raise ValueError(f"No embodiment file for {name}")
        return robot_file

    with open(os.path.join(configs_path, "_camera_config.yml"), "r", encoding="utf-8") as f:
        camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = camera_config[head_camera_type]["h"]
    args["head_camera_w"] = camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment items should be 1 or 3")

    get_embodiment_config = imports["get_embodiment_config"]
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])
    return args


def _expand_placeholder_values(
    *,
    robotwin_root: Path,
    key: str,
    value: Any,
    instruction_type: str,
) -> list[str]:
    key = key.strip("{}")
    value = str(value)
    object_json = robotwin_root / "description" / "objects_description" / f"{value}.json"

    if object_json.exists():
        with object_json.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if instruction_type == "unseen":
            descriptions = payload.get("unseen") or payload.get("seen", [])
        else:
            descriptions = payload.get("seen", [])
        return [f"the {description}" for description in descriptions]

    if ("\\" in value or "/" in value) and not object_json.exists():
        raise FileNotFoundError(f"Placeholder {key!r} looks like an object path but is missing: {object_json}")

    if len(key) == 1 and "a" <= key <= "z":
        return [f"the {value} arm"]

    return [value]


def _expand_all_episode_prompts(
    *,
    robotwin_root: Path,
    task_name: str,
    episode_info: dict[str, Any],
    instruction_type: str,
    imports: dict[str, Any],
) -> set[str]:
    task_data = imports["load_task_instructions"](task_name)
    instructions = list(task_data.get(instruction_type, []))
    filtered_instructions = imports["filter_instructions"](instructions, episode_info)
    stripped_episode_info = {key.strip("{}"): value for key, value in episode_info.items()}

    prompts: set[str] = set()
    for instruction in filtered_instructions:
        placeholders = list(dict.fromkeys(re.findall(r"{([^}]+)}", instruction)))
        value_options = [
            _expand_placeholder_values(
                robotwin_root=robotwin_root,
                key=key,
                value=stripped_episode_info[key],
                instruction_type=instruction_type,
            )
            for key in placeholders
        ]
        for values in itertools.product(*value_options):
            expanded = instruction
            for key, value in zip(placeholders, values):
                expanded = expanded.replace("{" + key + "}", value)
            prompts.add(DEFAULT_PROMPT.format(task=str(expanded)))

    return prompts


def _collect_prompts_for_task(
    *,
    robotwin_root: Path,
    task_name: str,
    task_config: str,
    instruction_type: str,
    eval_num_episodes: int,
    seed: int,
    max_seed_attempts: int,
    expand_all_placeholder_values: bool,
    imports: dict[str, Any],
) -> set[str]:
    class_decorator = imports["class_decorator"]
    generate_episode_descriptions = imports["generate_episode_descriptions"]
    UnStableError = imports["UnStableError"]

    task_args = _build_task_args(
        robotwin_root=robotwin_root,
        task_name=task_name,
        task_config=task_config,
        eval_num_episodes=eval_num_episodes,
        imports=imports,
    )
    task_env = class_decorator(task_name)
    task_env.suc = 0
    task_env.test_num = 0

    prompts: set[str] = set()
    now_id = 0
    succ_seed = 0
    now_seed = 100000 * (1 + int(seed))
    attempts = 0
    invalid_seed_attempts = 0
    max_invalid_seed_attempts = int(task_args.get("max_invalid_seed_attempts", max(20, eval_num_episodes * 10)))
    max_total_attempts = max_seed_attempts * max(1, eval_num_episodes)

    def handle_invalid_seed(reason: str) -> None:
        nonlocal now_id, succ_seed, now_seed, invalid_seed_attempts
        invalid_seed_attempts += 1
        now_seed += 1
        if invalid_seed_attempts < max_invalid_seed_attempts:
            return
        succ_seed += 1
        now_id += 1
        logger.warning(
            "Counting failed prompt episode task=%s config=%s after %d invalid seed(s): %s",
            task_name,
            task_config,
            invalid_seed_attempts,
            reason,
        )
        invalid_seed_attempts = 0

    def add_episode_prompts(episode_info: Any, *, strict: bool) -> tuple[int, int]:
        if not isinstance(episode_info, dict) or "info" not in episode_info:
            if strict:
                raise KeyError(f"Missing episode info for task={task_name} config={task_config} seed={now_seed}")
            return 0, 0

        before = len(prompts)
        instruction_seed = int(now_seed) % (2**32 - 1)
        if expand_all_placeholder_values:
            expanded_prompts = _expand_all_episode_prompts(
                robotwin_root=robotwin_root,
                task_name=task_name,
                episode_info=episode_info["info"],
                instruction_type=instruction_type,
                imports=imports,
            )
            prompts.update(expanded_prompts)
            return len(expanded_prompts), len(prompts) - before

        random.seed(instruction_seed)
        np.random.seed(instruction_seed)
        results = generate_episode_descriptions(task_name, [episode_info["info"]], eval_num_episodes)
        candidates = results[0].get(instruction_type)
        if not candidates:
            if strict:
                raise KeyError(
                    f"No instructions for type={instruction_type!r} task={task_name} config={task_config} seed={now_seed}"
                )
            logger.warning(
                "No prompt candidates for attempted seed task=%s config=%s seed=%s type=%s",
                task_name,
                task_config,
                now_seed,
                instruction_type,
            )
            return 0, 0

        for instruction in candidates:
            prompts.add(DEFAULT_PROMPT.format(task=str(instruction)))
        return len(candidates), len(prompts) - before

    while succ_seed < eval_num_episodes:
        attempts += 1
        if attempts > max_total_attempts:
            raise RuntimeError(
                f"Exceeded max_total_attempts={max_total_attempts} for task={task_name} config={task_config}; "
                f"covered {succ_seed}/{eval_num_episodes} eval episodes."
            )

        render_freq = task_args["render_freq"]
        task_args["render_freq"] = 0
        try:
            task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **task_args)
            episode_info = task_env.play_once()
            task_env.close_env()
        except UnStableError:
            _safe_close_env(task_env)
            task_args["render_freq"] = render_freq
            handle_invalid_seed("expert_check UnStableError")
            continue
        except Exception as exc:
            _safe_close_env(task_env)
            logger.warning(
                "Skipping unstable prompt seed task=%s config=%s seed=%s error=%r",
                task_name,
                task_config,
                now_seed,
                exc,
            )
            task_args["render_freq"] = render_freq
            handle_invalid_seed(f"expert_check exception: {type(exc).__name__}")
            continue

        task_args["render_freq"] = render_freq
        candidate_count, new_prompt_count = add_episode_prompts(episode_info, strict=False)

        if task_env.plan_success and task_env.check_success():
            if candidate_count <= 0:
                add_episode_prompts(episode_info, strict=True)
            invalid_seed_attempts = 0
            succ_seed += 1
        else:
            if candidate_count > 0:
                logger.info(
                    "Cached attempted seed prompts task=%s config=%s seed=%s candidates=%d new_prompts=%d prompts_so_far=%d",
                    task_name,
                    task_config,
                    now_seed,
                    candidate_count,
                    new_prompt_count,
                    len(prompts),
                )
            handle_invalid_seed("expert_check did not reach success state")
            continue

        logger.info(
            "Collected task=%s config=%s valid_episode=%d/%d seed=%s candidates=%d new_prompts=%d prompts_so_far=%d",
            task_name,
            task_config,
            succ_seed,
            eval_num_episodes,
            now_seed,
            candidate_count,
            new_prompt_count,
            len(prompts),
        )
        now_id += 1
        now_seed += 1

    return prompts


def collect_robotwin_prompts(
    *,
    robotwin_root: Path,
    task_names: list[str],
    task_configs: list[str],
    instruction_type: str,
    eval_num_episodes: int,
    seed: int,
    max_seed_attempts: int,
    expand_all_placeholder_values: bool,
) -> list[str]:
    imports = _prepare_robotwin_imports(robotwin_root)
    all_prompts: set[str] = set()
    try:
        total_jobs = len(task_names) * len(task_configs)
        job_idx = 0
        for task_config in task_configs:
            for task_name in task_names:
                job_idx += 1
                logger.info(
                    "Collecting prompts job=%d/%d task=%s config=%s",
                    job_idx,
                    total_jobs,
                    task_name,
                    task_config,
                )
                prompts = _collect_prompts_for_task(
                    robotwin_root=robotwin_root,
                    task_name=task_name,
                    task_config=task_config,
                    instruction_type=instruction_type,
                    eval_num_episodes=eval_num_episodes,
                    seed=seed,
                    max_seed_attempts=max_seed_attempts,
                    expand_all_placeholder_values=expand_all_placeholder_values,
                    imports=imports,
                )
                all_prompts.update(prompts)
                logger.info(
                    "Finished task=%s config=%s task_prompts=%d total_unique_prompts=%d",
                    task_name,
                    task_config,
                    len(prompts),
                    len(all_prompts),
                )
    finally:
        os.chdir(imports["old_cwd"])

    return sorted(all_prompts)


def _cache_name(prompt: str, context_len: int) -> str:
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return f"{hashed}.t5_len{context_len}.{CACHE_ENCODER_ID}.pt"


def _cache_path(cache_dir: Path, prompt: str, context_len: int) -> Path:
    return cache_dir / _cache_name(prompt, context_len)


PACKED_CACHE_BIN = "packed_cache.bin"
PACKED_CACHE_INDEX = "packed_cache.index.jsonl"


class PackedTextEmbeddingCacheWriter:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.bin_path = self.cache_dir / PACKED_CACHE_BIN
        self.index_path = self.cache_dir / PACKED_CACHE_INDEX
        self.names: set[str] = set()
        self._bin = None
        self._index = None
        if self.index_path.exists():
            with self.index_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    self.names.add(str(json.loads(line)["name"]))

    def has(self, name: str) -> bool:
        return name in self.names

    def __enter__(self):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._bin = self.bin_path.open("ab")
        self._index = self.index_path.open("a", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._index is not None:
            self._index.close()
        if self._bin is not None:
            self._bin.close()

    def write(self, name: str, payload: dict[str, torch.Tensor]) -> None:
        if self._bin is None or self._index is None:
            raise RuntimeError("PackedTextEmbeddingCacheWriter must be used as a context manager.")
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        payload_bytes = buffer.getvalue()
        offset = self._bin.tell()
        self._bin.write(payload_bytes)
        self._bin.flush()
        record = {"name": name, "offset": offset, "length": len(payload_bytes)}
        self._index.write(json.dumps(record, sort_keys=True) + "\n")
        self._index.flush()
        self.names.add(name)


def _atomic_torch_save(payload: dict[str, torch.Tensor], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


def _model_id_to_enc_id(model_id: str) -> str:
    base = str(model_id).split("/")[-1]
    enc_id = re.sub(r"[^a-z0-9]+", "", base.lower())
    return enc_id or "textenc"


def encode_prompts(
    *,
    prompts: list[str],
    cache_dir: Path,
    context_len: int,
    model_id: str,
    tokenizer_model_id: str,
    device: str,
    batch_size: int,
    overwrite: bool,
    redirect_common_files: bool,
    packed_cache: bool,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if _model_id_to_enc_id(model_id) != CACHE_ENCODER_ID:
        raise ValueError(
            f"This eval path expects cache encoder id {CACHE_ENCODER_ID!r}, "
            f"but model_id={model_id!r} resolves to {_model_id_to_enc_id(model_id)!r}."
        )

    packed_writer = PackedTextEmbeddingCacheWriter(cache_dir) if packed_cache else None
    prompts_to_encode: list[str] = []
    skipped = 0
    for prompt in prompts:
        name = _cache_name(prompt, context_len)
        path = cache_dir / name
        exists = path.exists() or (packed_writer is not None and packed_writer.has(name))
        if exists and not overwrite:
            skipped += 1
        else:
            prompts_to_encode.append(prompt)

    logger.info(
        "Text cache status: total_prompts=%d skipped_existing=%d to_encode=%d cache_dir=%s packed_cache=%s",
        len(prompts),
        skipped,
        len(prompts_to_encode),
        cache_dir,
        packed_cache,
    )
    if not prompts_to_encode:
        return

    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is unavailable; falling back to CPU text encoding.")
        device = "cpu"
    torch_dtype = torch.bfloat16

    logger.info(
        "Loading text encoder model_id=%s tokenizer_model_id=%s device=%s dtype=%s context_len=%d",
        model_id,
        tokenizer_model_id,
        device,
        torch_dtype,
        context_len,
    )
    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        redirect_common_files=redirect_common_files,
    )
    component_dir = _wan22_component_dir()
    if component_dir is not None:
        logger.info("Using Wan2.2 component directory from FASTWAM_WAN22_COMPONENT_DIR=%s", component_dir)
        text_encoder_path = _ensure_component_path(
            config=text_config,
            component_dir=component_dir,
            component_name="text encoder",
        )
        tokenizer_path = _ensure_component_path(
            config=tokenizer_config,
            component_dir=component_dir,
            component_name="tokenizer",
        )
        logger.info("Using Wan text encoder from component directory: %s", text_encoder_path)
        logger.info("Using Wan tokenizer from component directory: %s", tokenizer_path)
    else:
        text_config.download_if_necessary()
        tokenizer_config.download_if_necessary()
        text_encoder_path = text_config.path
        tokenizer_path = tokenizer_config.path

    text_encoder = _load_registered_model(
        text_encoder_path,
        "wan_video_text_encoder",
        torch_dtype=torch_dtype,
        device=device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_path,
        seq_len=context_len,
        clean="whitespace",
    )

    over_length_prompts = 0
    written = 0
    overwritten = 0
    writer_context = packed_writer if packed_writer is not None else nullcontext()
    with writer_context as writer:
        with tqdm(total=len(prompts_to_encode), desc="Encoding prompts", unit="prompt", dynamic_ncols=True) as pbar:
            with torch.no_grad():
                for start in range(0, len(prompts_to_encode), batch_size):
                    batch_prompts = prompts_to_encode[start : start + batch_size]
                    ids, mask = tokenizer(batch_prompts, return_mask=True, add_special_tokens=True)
                    ids = ids.to(device)
                    mask = mask.to(device=device, dtype=torch.bool)
                    over_length_prompts += int(mask.all(dim=1).sum().item())
                    context = text_encoder(ids, mask)

                    for i, prompt in enumerate(batch_prompts):
                        name = _cache_name(prompt, context_len)
                        output_path = cache_dir / name
                        already_exists = output_path.exists() or (writer is not None and writer.has(name))
                        if already_exists:
                            overwritten += 1
                        else:
                            written += 1
                        payload = {
                            "context": context[i].detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
                            "mask": mask[i].detach().to(device="cpu", dtype=torch.bool).contiguous(),
                        }
                        if writer is not None:
                            writer.write(name, payload)
                            output_path.unlink(missing_ok=True)
                        else:
                            _atomic_torch_save(payload, output_path)
                    pbar.update(len(batch_prompts))

    logger.info(
        "Finished text embedding precompute: new=%d overwritten=%d skipped=%d over_length=%d/%d",
        written,
        overwritten,
        skipped,
        over_length_prompts,
        len(prompts_to_encode),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute RoboTwin eval text embeddings for FastWAM.")
    parser.add_argument("--robotwin-root", default="/root/RoboTwin")
    parser.add_argument(
        "--tasks-file",
        default=str(PROJECT_ROOT / "third_party" / "RoboTwin" / "task_config" / "_eval_step_limit.yml"),
    )
    parser.add_argument("--cache-dir", default=str(PROJECT_ROOT / "data" / "text_embeds_cache" / "robotwin_eval"))
    parser.add_argument("--prompts-json", default=str(PROJECT_ROOT / "data" / "text_embeds_cache" / "robotwin_eval_prompts.json"))
    parser.add_argument("--task-config", action="append", dest="task_configs", default=None)
    parser.add_argument("--task", action="append", dest="tasks", default=None)
    parser.add_argument("--instruction-type", default="unseen")
    parser.add_argument("--eval-num-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seed-attempts", type=int, default=80)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--tokenizer-model-id", default=DEFAULT_TOKENIZER_MODEL_ID)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--packed-cache", action="store_true")
    parser.add_argument("--redirect-common-files", action="store_true", default=True)
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument(
        "--expand-all-placeholder-values",
        action="store_true",
        help="For each valid eval episode, enumerate all matching templates and all seen/unseen object descriptions instead of sampling descriptions.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = parse_args()
    if args.eval_num_episodes <= 0:
        raise ValueError("--eval-num-episodes must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    robotwin_root = _resolve_path(args.robotwin_root, base=PROJECT_ROOT)
    tasks_file = _resolve_path(args.tasks_file, base=PROJECT_ROOT)
    cache_dir = _resolve_path(args.cache_dir, base=PROJECT_ROOT)
    prompts_json = _resolve_path(args.prompts_json, base=PROJECT_ROOT)

    task_names = args.tasks if args.tasks else _load_task_names(tasks_file)
    task_configs = args.task_configs if args.task_configs else ["demo_clean", "demo_randomized"]

    if args.skip_collect:
        if not prompts_json.exists():
            raise FileNotFoundError(f"--skip-collect requires an existing prompts JSON: {prompts_json}")
        prompts = json.loads(prompts_json.read_text(encoding="utf-8"))
        if not isinstance(prompts, list) or not all(isinstance(item, str) for item in prompts):
            raise ValueError(f"Invalid prompts JSON payload: {prompts_json}")
        logger.info("Loaded %d prompts from existing %s", len(prompts), prompts_json)
    else:
        logger.info(
            "Starting RoboTwin prompt collection tasks=%d task_configs=%s eval_num_episodes=%d seed=%d",
            len(task_names),
            task_configs,
            args.eval_num_episodes,
            args.seed,
        )
        prompts = collect_robotwin_prompts(
            robotwin_root=robotwin_root,
            task_names=task_names,
            task_configs=task_configs,
            instruction_type=args.instruction_type,
            eval_num_episodes=args.eval_num_episodes,
            seed=args.seed,
            max_seed_attempts=args.max_seed_attempts,
            expand_all_placeholder_values=bool(args.expand_all_placeholder_values),
        )
        prompts_json.parent.mkdir(parents=True, exist_ok=True)
        prompts_json.write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved %d unique prompts to %s", len(prompts), prompts_json)

    if args.collect_only:
        return

    encode_prompts(
        prompts=prompts,
        cache_dir=cache_dir,
        context_len=args.context_len,
        model_id=args.model_id,
        tokenizer_model_id=args.tokenizer_model_id,
        device=args.device,
        batch_size=args.batch_size,
        overwrite=bool(args.overwrite),
        redirect_common_files=bool(args.redirect_common_files),
        packed_cache=bool(args.packed_cache),
    )


if __name__ == "__main__":
    main()
