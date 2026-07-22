import logging
import hashlib
import json
import inspect
import math
import os
import re
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim
from .adaptive_gate.eval_routing import explicit_eval_branch
from .adaptive_gate.training import (
    advance_successful_optimizer_steps,
    build_optimizer_parameter_groups,
    canonicalize_uncond_weight_schedule,
    classify_training_resume_source,
    raw_loss_gradient_statistics,
    uncond_weight_at_step,
    validate_dual_regime_trainer_state,
)
from .adaptive_gate.warm_start import (
    strict_standalone_idm_warm_start,
    warm_start_is_enabled,
)

logger = get_logger(__name__)


def _forward_prepared_model(model, sample):
    """Enter through the accelerator-prepared wrapper, never an unwrapped method."""
    return model(sample)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _final_video_block_parameters(
    model, count: int
) -> list[torch.nn.Parameter]:
    video = getattr(model, "video_expert", None)
    blocks = list(getattr(video, "blocks", ())) if video is not None else []
    count = int(count)
    if count <= 0 or count > len(blocks):
        raise ValueError(
            "gradient_diagnostic_video_final_blocks must select a non-empty "
            f"suffix of video_expert.blocks; requested={count}, available={len(blocks)}."
        )
    return [param for block in blocks[-count:] for param in block.parameters()]


def dual_regime_gradient_parameter_groups(
    model, *, video_final_blocks: int = 1
) -> dict[str, list[torch.nn.Parameter]]:
    """Diagnostic views over the action path; overlapping views are intentional."""
    action = getattr(model, "action_expert", None)
    if action is None:
        raise ValueError("Dual-regime gradient diagnostics require action_expert.")
    blocks = list(getattr(action, "blocks", ()))
    groups: dict[str, list[torch.nn.Parameter]] = {
        "action_all": list(action.parameters()),
    }
    action_encoder = getattr(action, "action_encoder", None)
    if action_encoder is not None:
        groups["action_embedding"] = list(action_encoder.parameters())
    attention_params = [
        param
        for block in blocks
        for param in getattr(block, "self_attn", torch.nn.Module()).parameters()
    ]
    if attention_params:
        groups["action_attention"] = attention_params
    if blocks:
        third = max(len(blocks) // 3, 1)
        groups["action_blocks_early"] = [
            param for block in blocks[:third] for param in block.parameters()
        ]
        groups["action_blocks_middle"] = [
            param for block in blocks[third : max(2 * third, third + 1)]
            for param in block.parameters()
        ]
        groups["action_blocks_final"] = [
            param for block in blocks[max(2 * third, third + 1) :] for param in block.parameters()
        ]
    proprio = getattr(model, "proprio_encoder", None)
    if proprio is not None:
        groups["proprio_all"] = list(proprio.parameters())
    video_final = _final_video_block_parameters(model, video_final_blocks)
    if any(param.requires_grad for param in video_final):
        groups["video_final"] = video_final
    return {name: params for name, params in groups.items() if params}


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        raw_run_until_step = cfg.get("run_until_step")
        self.requested_run_until_step = (
            int(raw_run_until_step)
            if raw_run_until_step is not None
            else None
        )
        raw_run_until_fraction = cfg.get("run_until_step_fraction")
        self.run_until_step_fraction = (
            float(raw_run_until_fraction)
            if raw_run_until_fraction is not None
            else None
        )
        if (
            self.requested_run_until_step is not None
            and self.run_until_step_fraction is not None
        ):
            raise ValueError(
                "run_until_step and run_until_step_fraction are mutually exclusive."
            )
        if (
            self.requested_run_until_step is not None
            and self.requested_run_until_step <= 0
        ):
            raise ValueError("run_until_step must be a positive integer.")
        if self.run_until_step_fraction is not None and not (
            0.0 < self.run_until_step_fraction <= 1.0
        ):
            raise ValueError("run_until_step_fraction must be in (0, 1].")
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        raw_save_steps = cfg.get("save_steps", ())
        self.save_steps = {
            int(step) for step in (raw_save_steps or ())
        }
        if any(step <= 0 for step in self.save_steps):
            raise ValueError("save_steps must contain only positive integers.")
        self.save_step_fractions = tuple(
            float(fraction)
            for fraction in (cfg.get("save_step_fractions", ()) or ())
        )
        if any(
            not 0.0 < fraction < 1.0
            for fraction in self.save_step_fractions
        ):
            raise ValueError(
                "save_step_fractions must be strictly between 0 and 1."
            )
        self.save_final_checkpoint = bool(cfg.get("save_final_checkpoint", True))
        self.save_optimizer_state = bool(cfg.get("save_optimizer_state", True))
        self.weights_checkpoint_kind = str(
            cfg.get("weights_checkpoint_kind", "full")
        )
        if self.weights_checkpoint_kind not in {"full", "action_dit_delta"}:
            raise ValueError(
                "weights_checkpoint_kind must be 'full' or 'action_dit_delta'."
            )
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        
        self.resume = cfg.resume
        self.warm_start_cfg = cfg.get("warm_start")
        self.warm_start_enabled = warm_start_is_enabled(self.warm_start_cfg)
        if self.resume and self.warm_start_enabled:
            raise ValueError(
                "`resume` and `warm_start` are mutually exclusive: resume restores an "
                "adaptive training lineage, while warm_start creates a new lineage."
            )
        self.is_dual_regime = (
            tuple(getattr(model, "adaptive_regimes", ())) == ("uncond", "idm")
            and getattr(model, "adaptive_backbone_kind", None) == "idm"
        )
        if (
            self.weights_checkpoint_kind == "action_dit_delta"
            and not self.is_dual_regime
        ):
            raise ValueError(
                "ActionDiT deltas are only supported for dual-regime training."
            )
        dual_cfg = cfg.get("dual_regime_training") or {}
        if isinstance(dual_cfg, DictConfig):
            dual_cfg = OmegaConf.to_container(dual_cfg, resolve=True)
        self.dual_regime_cfg = dict(dual_cfg)
        optimizer_group_cfg = dict(self.dual_regime_cfg.get("optimizer", {}))
        self.action_lr_scale = float(optimizer_group_cfg.get("action_lr_scale", 1.0))
        self.proprio_lr_scale = float(optimizer_group_cfg.get("proprio_lr_scale", 1.0))
        raw_video_lr_scale = optimizer_group_cfg.get("video_lr_scale", None)
        self.video_lr_scale = (
            0.0 if self.is_dual_regime else 1.0
        ) if raw_video_lr_scale is None else float(raw_video_lr_scale)
        raw_video_final_blocks = optimizer_group_cfg.get("video_final_blocks", None)
        self.video_final_blocks = (
            None
            if raw_video_final_blocks is None
            else int(raw_video_final_blocks)
        )
        if self.is_dual_regime and self.video_lr_scale > 0.0 and (
            self.video_final_blocks is None or self.video_final_blocks <= 0
        ):
            raise ValueError(
                "Dual-regime video updates must name a positive "
                "optimizer.video_final_blocks suffix."
            )
        self.video_train_start_fraction = float(
            optimizer_group_cfg.get("video_train_start_fraction", 0.0)
        )
        if not 0.0 <= self.video_train_start_fraction <= 1.0:
            raise ValueError("video_train_start_fraction must be in [0, 1].")
        self.gradient_diagnostics_every = int(
            self.dual_regime_cfg.get("gradient_diagnostics_every", 0)
        )
        if self.gradient_diagnostics_every < 0:
            raise ValueError("gradient_diagnostics_every must be non-negative.")
        self.gradient_diagnostic_video_final_blocks = int(
            self.dual_regime_cfg.get("gradient_diagnostic_video_final_blocks", 1)
        )
        if self.is_dual_regime and self.gradient_diagnostics_every > 0:
            # Validate the diagnostic view before any distributed workers start.
            _final_video_block_parameters(
                self.model, self.gradient_diagnostic_video_final_blocks
            )
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        if self.warm_start_enabled:
            warm_record = strict_standalone_idm_warm_start(
                self.model,
                self.warm_start_cfg,
                target_model_config=cfg.model,
                target_dataset_stats=os.path.join(self.output_dir, "dataset_stats.json"),
            )
            logger.info(
                "Strict standalone-IDM warm start accepted: parent_id=%s sha256=%s",
                warm_record["parent_checkpoint_id"],
                warm_record["parent_checkpoint_sha256"],
            )
        if self.is_dual_regime and not hasattr(self.model, "dual_regime_optimizer_steps"):
            self.model.dual_regime_optimizer_steps = 0
        self.dual_regime_optimizer_steps = int(
            getattr(self.model, "dual_regime_optimizer_steps", 0)
        )
        self.warm_start_provenance = getattr(
            self.model, "warm_start_provenance", None
        )

        # Configure disjoint expert groups before optimizer/DeepSpeed creation.
        optimizer_groups = build_optimizer_parameter_groups(
            self.model,
            base_learning_rate=self.learning_rate,
            action_lr_scale=self.action_lr_scale,
            proprio_lr_scale=self.proprio_lr_scale,
            video_lr_scale=self.video_lr_scale,
            video_final_blocks=self.video_final_blocks,
        )
        self.trainable_group_names = tuple(group["name"] for group in optimizer_groups)
        self.optimizer = torch.optim.AdamW(
            optimizer_groups,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        if self.requested_run_until_step is not None:
            self.run_until_step = self.requested_run_until_step
        elif self.run_until_step_fraction is not None:
            self.run_until_step = max(
                1,
                int(math.ceil(self.run_until_step_fraction * self.max_steps)),
            )
        else:
            self.run_until_step = self.max_steps
        if self.run_until_step > self.max_steps:
            raise ValueError(
                "run_until_step cannot exceed the contracted total optimizer "
                f"steps: {self.run_until_step} > {self.max_steps}."
            )
        self.save_steps.update(
            max(
                1,
                min(
                    self.max_steps - 1,
                    int(math.ceil(fraction * self.max_steps)),
                ),
            )
            for fraction in self.save_step_fractions
        )
        schedule_points = self.dual_regime_cfg.get("uncond_weight_schedule")
        if self.is_dual_regime:
            if schedule_points is None:
                fixed_weight = float(getattr(self.model, "action_regime_weight_uncond", 1.0))
                schedule_points = ((0.0, fixed_weight), (1.0, fixed_weight))
            self.uncond_weight_schedule = canonicalize_uncond_weight_schedule(
                schedule_points
            )
            self.dual_regime_training_contract = {
                "uncond_weight_schedule": [list(point) for point in self.uncond_weight_schedule],
                "base_learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "gradient_accumulation_steps": self.gradient_accumulation_steps,
                "mixed_precision": self.mixed_precision,
                "max_grad_norm": self.max_grad_norm,
                "action_lr_scale": self.action_lr_scale,
                "proprio_lr_scale": self.proprio_lr_scale,
                "video_lr_scale": self.video_lr_scale,
                "video_final_blocks": self.video_final_blocks,
                "video_train_start_fraction": self.video_train_start_fraction,
                "gradient_diagnostics_every": self.gradient_diagnostics_every,
                "gradient_diagnostic_video_final_blocks": (
                    self.gradient_diagnostic_video_final_blocks
                ),
                "total_optimizer_steps": self.max_steps,
            }
            self.model.dual_regime_training_contract = self.dual_regime_training_contract
        else:
            if schedule_points is not None:
                raise ValueError(
                    "uncond_weight_schedule is only valid for the dual-regime IDM model."
                )
            self.uncond_weight_schedule = None
            self.dual_regime_training_contract = None
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)
        self.training_metrics_path = os.path.join(
            self.output_dir, "training_metrics.jsonl"
        )

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()
        unwrapped = self.accelerator.unwrap_model(self.model)
        self.dual_regime_optimizer_steps = int(
            getattr(unwrapped, "dual_regime_optimizer_steps", self.dual_regime_optimizer_steps)
        )
        self.warm_start_provenance = getattr(
            unwrapped, "warm_start_provenance", self.warm_start_provenance
        )
        self._validate_loaded_training_contract(unwrapped)
        if self.global_step > self.run_until_step:
            raise ValueError(
                "Loaded training state is already beyond this staged boundary: "
                f"global_step={self.global_step}, run_until_step={self.run_until_step}."
            )

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _append_training_metrics(self, payload: dict) -> None:
        if not self.accelerator.is_main_process:
            return
        with open(self.training_metrics_path, "a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(payload, sort_keys=True, allow_nan=False) + "\n"
            )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)
        if scheduler_type not in {"cosine", "constant"}:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        remaining_steps = max(total_train_steps - warmup_steps, 1)

        def lr_multiplier(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return max((step + 1) / float(warmup_steps), 1.0 / warmup_steps)
            if scheduler_type == "constant":
                return 1.0
            progress = min(max((step - warmup_steps) / float(remaining_steps), 0.0), 1.0)
            return 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))

        # One multiplicative schedule preserves every configured group-LR ratio.
        return LambdaLR(self.optimizer, lr_lambda=lr_multiplier)
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        resume_kind = classify_training_resume_source(
            resume_path, is_dual_regime=self.is_dual_regime
        )
        if resume_kind == "full_state":
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2.")

    def _validate_loaded_training_contract(self, model) -> None:
        if not self.is_dual_regime:
            return
        loaded = getattr(model, "dual_regime_training_contract", None)
        if self.resume and loaded is not None and loaded != self.dual_regime_training_contract:
            raise ValueError(
                "Resumed dual-regime training contract differs from the current config: "
                f"checkpoint={loaded}, current={self.dual_regime_training_contract}."
            )
        model.dual_regime_training_contract = self.dual_regime_training_contract
        model.dual_regime_optimizer_steps = self.dual_regime_optimizer_steps
        model.warm_start_provenance = self.warm_start_provenance

    def _set_dit_only_train_mode(self):
        logger.info("Restoring configured train mode for groups=%s.", self.trainable_group_names)
        model = self.accelerator.unwrap_model(self.model)
        model.eval()
        model.mot.train()
        for name in self.trainable_group_names:
            module = {
                "action": getattr(model, "action_expert", None),
                "proprio": getattr(model, "proprio_encoder", None),
                "video": getattr(model, "video_expert", None),
            }[name]
            if module is not None:
                module.train()
        for name, module in (
            ("action", getattr(model, "action_expert", None)),
            ("proprio", getattr(model, "proprio_encoder", None)),
            ("video", getattr(model, "video_expert", None)),
        ):
            if module is not None and name not in self.trainable_group_names:
                module.eval()

    @staticmethod
    def _apply_dit_only_train_mode(model):
        """Legacy helper retained for callers outside the staged trainer path."""
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # eval_index = (self.global_step + self.accelerator.process_index) % len(self.val_dataset)
        rng = torch.Generator(device="cpu").manual_seed(self.global_step + self.accelerator.process_index)
        eval_index = torch.randint(0, len(self.val_dataset), (1,), generator=rng).item()
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])

        # 1. training loss
        with self.accelerator.autocast():
            val_loss, _ = model.training_loss(sample)
            val_loss = val_loss.float().item()
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        infer_kwargs.update(
            explicit_eval_branch(
                model,
                "infer",
                getattr(model, "adaptive_backbone_kind", "idm"),
                require_video=True,
            )
        )

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        mean_metrics = gathered_metrics[:, :7].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 7].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 8].mean().item() if action_l1 is not None else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(mean_metrics[0].item()),
            "psnr_rg": float(mean_metrics[1].item()),
            "ssim_rg": float(mean_metrics[2].item()),
            "psnr_rd": float(mean_metrics[3].item()),
            "ssim_rd": float(mean_metrics[4].item()),
            "psnr_dg": float(mean_metrics[5].item()),
            "ssim_dg": float(mean_metrics[6].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        if self.is_dual_regime:
            model.dual_regime_optimizer_steps = int(self.dual_regime_optimizer_steps)
            model.dual_regime_training_contract = self.dual_regime_training_contract
            model.warm_start_provenance = self.warm_start_provenance
            model.action_regime_weight_uncond = uncond_weight_at_step(
                self.uncond_weight_schedule,
                optimizer_step=self.dual_regime_optimizer_steps,
                total_optimizer_steps=self.max_steps,
            )
        stats_path = os.path.join(self.output_dir, "dataset_stats.json")
        if os.path.isfile(stats_path):
            model.dataset_stats_fingerprint = _sha256_file(stats_path)
        elif getattr(model, "adaptive_regimes", None):
            raise FileNotFoundError(
                "Adaptive checkpoint provenance requires training dataset stats at "
                f"{stats_path}."
            )
        suffix = (
            ".action_dit_delta.pt"
            if self.weights_checkpoint_kind == "action_dit_delta"
            else ".pt"
        )
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}{suffix}")
        if self.weights_checkpoint_kind == "action_dit_delta":
            model.save_action_dit_delta(ckpt_path, step=self.global_step)
        else:
            model.save_checkpoint(ckpt_path, optimizer=None, step=self.global_step)
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        stats_path = os.path.join(self.output_dir, "dataset_stats.json")
        if self.is_dual_regime and not os.path.isfile(stats_path):
            raise FileNotFoundError(
                "Adaptive trainer state requires the exact dataset_stats.json."
            )
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
            "dual_regime_optimizer_steps": int(self.dual_regime_optimizer_steps),
            "dual_regime_training_contract": self.dual_regime_training_contract,
            "warm_start_provenance": self.warm_start_provenance,
            "dataset_stats_fingerprint": (
                _sha256_file(stats_path) if os.path.isfile(stats_path) else None
            ),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = None
        if self.save_optimizer_state:
            state_path = os.path.join(self.state_dir, step_tag)
            ensure_dir(state_path)
            self.accelerator.save_state(output_dir=state_path)
            if self.accelerator.is_main_process:
                self._save_trainer_state(state_path)
            self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        state_file = Path(state_dir) / "trainer_state.json"
        payload = None
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        if self.is_dual_regime:
            stats_path = os.path.join(self.output_dir, "dataset_stats.json")
            if not os.path.isfile(stats_path):
                raise FileNotFoundError(
                    "Adaptive resume requires the current run dataset_stats.json."
                )
            current_stats = _sha256_file(stats_path)
            validate_dual_regime_trainer_state(
                payload,
                expected_contract=self.dual_regime_training_contract,
                expected_dataset_stats_fingerprint=current_stats,
            )
        self.accelerator.load_state(input_dir=state_dir)
        if payload is not None:
            self.global_step = int(payload["global_step"])
            if self.is_dual_regime:
                self.dual_regime_optimizer_steps = int(
                    payload["dual_regime_optimizer_steps"]
                )
                self.warm_start_provenance = payload.get("warm_start_provenance")
                model = self.accelerator.unwrap_model(self.model)
                model.dual_regime_optimizer_steps = self.dual_regime_optimizer_steps
                model.dual_regime_training_contract = self.dual_regime_training_contract
                model.warm_start_provenance = self.warm_start_provenance

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch(self.epoch)
                self.train_sampler.set_epoch_offset(0)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    def _set_dual_regime_weight(self) -> float | None:
        if not self.is_dual_regime:
            return None
        weight = uncond_weight_at_step(
            self.uncond_weight_schedule,
            optimizer_step=self.dual_regime_optimizer_steps,
            total_optimizer_steps=self.max_steps,
        )
        model = self.accelerator.unwrap_model(self.model)
        model.action_regime_weight_uncond = weight
        return weight

    def _capture_gradient_diagnostics_this_microbatch(self) -> bool:
        return (
            self.is_dual_regime
            and self.gradient_diagnostics_every > 0
            and self.accelerator.sync_gradients
            and (self.dual_regime_optimizer_steps + 1)
            % self.gradient_diagnostics_every
            == 0
        )

    def _consume_gradient_diagnostics(self, model) -> dict[str, float]:
        raw_losses = getattr(model, "_raw_dual_regime_losses", None)
        if raw_losses is None or "idm" not in raw_losses or "uncond" not in raw_losses:
            raise RuntimeError(
                "Fused dual-regime model did not expose raw IDM/UNCOND losses for diagnostics."
            )
        try:
            local_stats = raw_loss_gradient_statistics(
                raw_losses["idm"],
                raw_losses["uncond"],
                dual_regime_gradient_parameter_groups(
                    model,
                    video_final_blocks=self.gradient_diagnostic_video_final_blocks,
                ),
            )
        finally:
            delattr(model, "_raw_dual_regime_losses")

        metrics: dict[str, float] = {}
        for group_name, values in local_stats.items():
            gathered = self.accelerator.gather(values.reshape(1, -1))
            totals = gathered.reshape(-1, values.numel()).sum(dim=0)
            dot, idm_sq, uncond_sq, used_by_both, parameter_count = totals
            denom = torch.sqrt(idm_sq * uncond_sq)
            cosine = torch.where(
                denom > 0,
                dot / denom,
                torch.full_like(denom, float("nan")),
            )
            prefix = f"gradient_alignment/{group_name}"
            metrics[f"{prefix}/cosine"] = float(cosine.item())
            metrics[f"{prefix}/idm_norm"] = float(torch.sqrt(idm_sq).item())
            metrics[f"{prefix}/uncond_norm"] = float(torch.sqrt(uncond_sq).item())
            metrics[f"{prefix}/used_fraction"] = float(
                (used_by_both / parameter_count.clamp_min(1)).item()
            )
            metrics[f"{prefix}/unused_fraction"] = float(
                (1.0 - used_by_both / parameter_count.clamp_min(1)).item()
            )
        return metrics

    def _enable_frozen_video_diagnostic_parameters(
        self, model
    ) -> list[torch.nn.Parameter]:
        """Temporarily expose frozen video suffixes only to ``autograd.grad``.

        These leaves are switched off again before the main backward pass, so
        they never enter optimizer state or DDP gradient reduction.
        """
        toggled = []
        for param in _final_video_block_parameters(
            model, self.gradient_diagnostic_video_final_blocks
        ):
            if not param.requires_grad:
                param.requires_grad_(True)
                toggled.append(param)
        return toggled

    def _apply_staged_gradient_freezing(self) -> None:
        if "video" not in self.trainable_group_names:
            return
        fraction = self.dual_regime_optimizer_steps / float(max(self.max_steps, 1))
        if fraction >= self.video_train_start_fraction:
            return
        for group in self.optimizer.param_groups:
            if group.get("name") == "video":
                group["lr"] = 0.0
                for param in group["params"]:
                    param.grad = None

    def train(self):
        self._set_dit_only_train_mode()

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info(
            "Starting training with max_steps=%d run_until_step=%d.",
            self.max_steps,
            self.run_until_step,
        )
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()
        self.last_successful_step_time = self.run_start_time

        while self.global_step < self.run_until_step:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.set_epoch(self.epoch)
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                self._set_dual_regime_weight()
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                capture_gradient_diagnostics = (
                    self._capture_gradient_diagnostics_this_microbatch()
                )
                diagnostic_video_params = (
                    self._enable_frozen_video_diagnostic_parameters(unwrapped_model)
                    if capture_gradient_diagnostics
                    else []
                )
                if hasattr(unwrapped_model, "_raw_dual_regime_losses"):
                    delattr(unwrapped_model, "_raw_dual_regime_losses")
                unwrapped_model._capture_raw_dual_regime_losses = (
                    capture_gradient_diagnostics
                )
                try:
                    with self.accelerator.autocast():
                        # Always enter through the prepared wrapper's forward so
                        # DDP/DeepSpeed hooks and reducers observe every iteration.
                        loss, loss_dict = _forward_prepared_model(self.model, sample)
                    diagnostic_metrics = (
                        self._consume_gradient_diagnostics(unwrapped_model)
                        if capture_gradient_diagnostics
                        else {}
                    )
                finally:
                    unwrapped_model._capture_raw_dual_regime_losses = False
                    for param in diagnostic_video_params:
                        param.requires_grad_(False)
                self.accelerator.backward(loss)

                if self.accelerator.sync_gradients:
                    self._apply_staged_gradient_freezing()
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    optimizer_step_was_skipped = bool(
                        self.accelerator.optimizer_step_was_skipped
                    )
                    if not optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.global_step, self.dual_regime_optimizer_steps = (
                        advance_successful_optimizer_steps(
                            global_step=self.global_step,
                            dual_regime_optimizer_steps=self.dual_regime_optimizer_steps,
                            is_dual_regime=self.is_dual_regime,
                            optimizer_step_was_skipped=optimizer_step_was_skipped,
                        )
                    )
                    if self.is_dual_regime:
                        unwrapped_model.dual_regime_optimizer_steps = (
                            self.dual_regime_optimizer_steps
                        )
                    self.optimizer.zero_grad(set_to_none=True)
                    if optimizer_step_was_skipped:
                        logger.warning(
                            "Optimizer step was skipped; successful-step counters and "
                            "dual-regime schedules remain at global_step=%d.",
                            self.global_step,
                        )
                        continue
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    if self.is_dual_regime:
                        global_loss_metrics["dual_regime_optimizer_steps"] = float(
                            self.dual_regime_optimizer_steps
                        )
                        global_loss_metrics["dual_regime_schedule_fraction"] = float(
                            self.dual_regime_optimizer_steps / max(self.max_steps, 1)
                        )
                    global_loss_metrics.update(diagnostic_metrics)
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                    current_lr = float(self.optimizer.param_groups[0]["lr"])
                    finite_values = [
                        global_loss,
                        global_grad_norm,
                        current_lr,
                        *global_loss_metrics.values(),
                    ]
                    if not all(math.isfinite(value) for value in finite_values):
                        raise FloatingPointError(
                            "Non-finite training metric detected after optimizer step."
                        )
                    metric_time = time.perf_counter()
                    step_duration_seconds = (
                        metric_time - self.last_successful_step_time
                    )
                    self.last_successful_step_time = metric_time
                    self._append_training_metrics(
                        {
                            "schema": "fastwam-training-metric-v1",
                            "epoch": int(self.epoch),
                            "global_step": int(self.global_step),
                            "dual_regime_optimizer_steps": int(
                                self.dual_regime_optimizer_steps
                            ),
                            "loss": global_loss,
                            "losses": global_loss_metrics,
                            "grad_norm_before_clip": global_grad_norm,
                            "gradient_clipped": bool(
                                global_grad_norm > self.max_grad_norm
                            ),
                            "max_grad_norm": self.max_grad_norm,
                            "learning_rate": current_lr,
                            "elapsed_seconds": float(
                                metric_time - self.run_start_time
                            ),
                            "step_duration_seconds": float(
                                step_duration_seconds
                            ),
                            "samples_per_second": float(
                                self.batch_size
                                * self.accelerator.num_processes
                                * self.gradient_accumulation_steps
                                / max(step_duration_seconds, 1e-12)
                            ),
                            "peak_gpu_memory_bytes": (
                                int(torch.cuda.max_memory_allocated())
                                if torch.cuda.is_available()
                                else 0
                            ),
                            "optimizer_step_was_skipped": False,
                        }
                    )

                    periodic_log = self.log_every > 0 and self.global_step % self.log_every == 0
                    if (periodic_log or diagnostic_metrics) and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        for group in self.optimizer.param_groups:
                            wandb_payload[f"train/lr_{group.get('name', 'unnamed')}"] = float(
                                group["lr"]
                            )
                        self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                                metrics["psnr_rd"],
                                metrics["ssim_rd"],
                            )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                                "eval/psnr_rg": float(metrics["psnr_rg"]),
                                "eval/ssim_rg": float(metrics["ssim_rg"]),
                                "eval/psnr_rd": float(metrics["psnr_rd"]),
                                "eval/ssim_rd": float(metrics["ssim_rd"]),
                                "eval/psnr_dg": float(metrics["psnr_dg"]),
                                "eval/ssim_dg": float(metrics["ssim_dg"]),
                            }
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._wandb_log(eval_payload)

                    periodic_save = (
                        self.save_every > 0
                        and self.global_step % self.save_every == 0
                    )
                    if periodic_save or self.global_step in self.save_steps:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.run_until_step:
                        reason = (
                            "max_steps reached"
                            if self.global_step >= self.max_steps
                            else "run_until_step reached"
                        )
                        self._finish_training(reason)
                        return

        self._finish_training("training finished")

    def _finish_training(self, reason: str):
        if not self.save_final_checkpoint:
            if self.accelerator.is_main_process:
                logger.info(
                    "[done] %s step=%d final checkpoint disabled",
                    reason,
                    self.global_step,
                )
            return None
        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] %s step=%d weights=%s state=%s",
                reason,
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        return ckpt_info
