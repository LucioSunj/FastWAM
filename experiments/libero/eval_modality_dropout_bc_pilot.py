"""Run one fixed 30-episode rollout condition for a pilot arm."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero import eval_libero_single as legacy
from experiments.libero.eval_p1_dino_libero_plus import (
    _eval_cfg,
    _get_plus_env,
    _load_training_config,
)
from experiments.libero.libero_utils import LIBERO_ENV_RESOLUTION
from fastwam.adapters import sha256_file
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.modality_dropout_bc import PILOT_ARMS, resolve_pilot_arm
from fastwam.modality_dropout_libero_inference import (
    ROLLOUT_MODALITY_CONDITIONS,
    ModalityDropoutCompiledLiberoPolicy,
)
from fastwam.p1_dino_bc_runner import build_real_p1_policy, is_visual_v2_config
from fastwam.p1_visual_bc_checkpoint import load_p1_visual_bc_checkpoint
from fastwam.p1_visual_bc_full_checkpoint import (
    P1_VISUAL_BC_FULL_CHECKPOINT_SCHEMA,
    load_p1_visual_bc_full_weights_for_evaluation,
)
from fastwam.utils.pytorch_utils import set_global_seed
from libero.libero import benchmark

PILOT_RESET_IDS = (0, 8, 16)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt-cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", choices=tuple(PILOT_ARMS), required=True)
    parser.add_argument(
        "--condition",
        choices=tuple(sorted(ROLLOUT_MODALITY_CONDITIONS)),
        required=True,
    )
    parser.add_argument("--no-compile", action="store_true")
    return parser.parse_args()


def _build_model(args: argparse.Namespace, training_cfg):
    if not is_visual_v2_config(training_cfg):
        raise ValueError("The modality-dropout pilot requires a V2 visual config.")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    policy, parent_load = build_real_p1_policy(training_cfg, device=device)
    raw = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or not isinstance(raw.get("contract"), dict):
        raise ValueError("Pilot checkpoint is missing its strict contract.")
    expected_visual = policy.visual_encoder.asset.checkpoint_metadata(
        camera_ids=policy.p1_config.camera_ids,
        input_contract_sha256=policy.p1_config.camera_input_contract_sha256,
    )
    common = {
        "path": args.checkpoint,
        "adapter": policy.lora_adapter,
        "reader": policy.visual_reader,
        "expected_parent_checkpoint_sha256": str(
            training_cfg.parent.checkpoint_sha256
        ),
        "expected_contract": raw["contract"],
        "expected_visual_backbone": expected_visual,
    }
    if raw.get("schema") == P1_VISUAL_BC_FULL_CHECKPOINT_SCHEMA:
        loaded = load_p1_visual_bc_full_weights_for_evaluation(**common)
    else:
        loaded = load_p1_visual_bc_checkpoint(**common)
    model = ModalityDropoutCompiledLiberoPolicy(
        policy,
        prompt_cache_dir=args.prompt_cache_dir,
        condition=args.condition,
        fixed_gaussian_dino=args.arm == "D",
        compile_enabled=not args.no_compile,
    )
    return model, parent_load, loaded


def main() -> None:
    args = _args()
    arm = resolve_pilot_arm(args.arm)
    training_cfg = _load_training_config(args.config, visual_backbone=None)
    if str(training_cfg.runner.arm).upper() != arm.name:
        raise ValueError("Rollout config arm and requested arm differ.")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Rollout output already exists: {output}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    eval_cfg = _eval_cfg(training_cfg, output.parent)
    eval_cfg.EVALUATION.task_suite_name = "libero_spatial"
    set_global_seed(42, get_worker_init_fn=False)
    torch.set_float32_matmul_precision("high")
    model, parent_load, loaded = _build_model(args, training_cfg)
    processor: FastWAMProcessor = instantiate(training_cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(
        load_dataset_stats_from_json(str(training_cfg.parent.statistics))
    )
    registry = benchmark.get_benchmark_dict()
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        suite = registry["libero_spatial"]()
    episodes = []
    for task_id in range(10):
        task = suite.get_task(task_id)
        initial_states = suite.get_task_init_states(task_id)
        for reset_id in PILOT_RESET_IDS:
            if reset_id >= len(initial_states):
                raise ValueError(f"Task {task_id} has no reset {reset_id}.")
            env, description = _get_plus_env(task, LIBERO_ENV_RESOLUTION, 42)
            try:
                model.clear_prompt()
                context, context_mask = model.encode_prompt(
                    DEFAULT_PROMPT.format(task=description)
                )
                success, _, _, _, inference_seconds = legacy.run_single_episode(
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
                    cached_context=context,
                    cached_context_mask=context_mask,
                )
            finally:
                env.close()
            episodes.append(
                {
                    "suite": "libero_spatial",
                    "task_id": task_id,
                    "reset_id": reset_id,
                    "success": bool(success),
                    "inference_seconds": inference_seconds,
                    "task_description": description,
                }
            )
    if len(episodes) != 30:
        raise RuntimeError("Pilot rollout episode count changed from 30.")
    payload = {
        "schema": "fastwam-modality-dropout-rollout-condition-v1",
        "status": "COMPLETE",
        "arm": arm.name,
        "condition": args.condition,
        "episode_success": [row["success"] for row in episodes],
        "episodes": episodes,
        "checkpoint": {
            "path": str(Path(args.checkpoint).resolve()),
            "sha256": sha256_file(args.checkpoint),
            "global_step": int(loaded["global_step"]),
        },
        "parent_load": parent_load,
        "modality_audit": model.modality_dropout_audit,
        "dino_call_audit": model.dino_call_audit,
        "parity": model.parity_report,
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "COMPLETE", "output": str(output)}))


if __name__ == "__main__":
    main()
