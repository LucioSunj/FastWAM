"""Rank-32 P1 DINO policy surface for compiled LIBERO inference."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from fastwam.adapters import PolicyRegime, sha256_file
from fastwam.models.wan22.adaptive_action import (
    CachedActionCondition,
    VisualReadCondition,
)
from fastwam.models.wan22.visual_backbone import FrozenVisualPatchEncoder
from fastwam.models.wan22.visual_contracts import (
    NativePatchMemory,
    PreparedCameraBatch,
    PreparedVisualCameraBatch,
    contract_sha256,
)
from fastwam.p1_dino_bc import FastWAMP1DinoBCPolicy
from fastwam.utils.text_embedding_cache import (
    load_text_embedding,
    prompt_sha256,
)


def resolve_compile_cache_seed(
    seed: Path,
    *,
    cache_name: str,
    compile_identity: str,
    worker_id: int,
) -> Path:
    """Resolve current or legacy per-worker compile-cache layouts."""

    root = seed / cache_name
    candidates = (
        root / compile_identity / f"worker_{worker_id}",
        root / f"worker_{worker_id}",
    )
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.iterdir()):
            return candidate
    if root.is_dir() and not any(
        child.is_dir() and child.name.startswith("worker_") for child in root.iterdir()
    ):
        if any(root.iterdir()):
            return root
    raise FileNotFoundError(
        f"No {cache_name} cache for worker {worker_id} and identity "
        f"{compile_identity!r} under {seed}."
    )


@dataclass(frozen=True)
class CompileParityReport:
    """Correctness gate for the compiled action path."""

    baseline_repeat_max_abs: float
    compiled_max_abs: float
    dtype_atol: float
    threshold: float
    passed: bool
    compile_warmup_seconds: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible report."""

        return {
            **self.__dict__,
            "threshold_formula": ("max(dtype_atol, 1.25 * baseline_repeat_max_abs)"),
        }


class _P1DenoiseKernel(nn.Module):
    """One ActionDiT step with a statically selected visual-memory branch."""

    def __init__(
        self,
        policy: FastWAMP1DinoBCPolicy,
        *,
        memory_enabled: bool,
    ) -> None:
        super().__init__()
        self.action_expert = policy.actor.action_expert
        self.mot = policy.actor.mot
        self.memory_enabled = bool(memory_enabled)
        self.visual_reader = policy.visual_reader if self.memory_enabled else None

    def forward(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_cache_k: list[torch.Tensor],
        video_cache_v: list[torch.Tensor],
        action_attention_mask: torch.Tensor,
        visual_memory,
        visual_proprio: torch.Tensor | None,
        current_frame_video_tokens: int,
    ) -> torch.Tensor:
        """Predict action velocity through the selected static memory branch."""

        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        if self.memory_enabled:
            if visual_memory is None or visual_proprio is None:
                raise ValueError("The correct-memory kernel requires visual inputs.")
            visual_reader = self.visual_reader
            visual_layout = {"p1_native_patch_layout": visual_memory.layout_contract}
            visual_time = action_pre["t"]
            visual_extent = current_frame_video_tokens
        else:
            visual_reader = None
            visual_memory = None
            visual_proprio = None
            visual_layout = None
            visual_time = None
            visual_extent = None
        tokens = self.mot._forward_action_with_video_cache_inner(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_cache_k=video_cache_k,
            video_cache_v=video_cache_v,
            action_attention_mask=action_attention_mask,
            visual_reader=visual_reader,
            visual_memory=visual_memory,
            visual_proprio=visual_proprio,
            action_time_embedding=visual_time,
            current_frame_video_tokens=visual_extent,
            video_layout_metadata=visual_layout,
        )
        return self.action_expert.post_dit(tokens, action_pre)


P1_POLICY_MEMORY_MODES = frozenset(
    {"correct", "off", "random_tensor", "random_vit"}
)
P1_RANDOM_PATCH_SEED = 2026081321


def _fixed_random_patch_memory(
    memory: NativePatchMemory,
    *,
    seed: int,
) -> NativePatchMemory:
    """Replace native patch tokens by one fixed, normalized random tensor."""

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    shape = (1, *memory.tokens.shape[1:])
    tokens = torch.randn(shape, generator=generator, dtype=torch.float32)
    tokens = torch.nn.functional.layer_norm(tokens, (tokens.shape[-1],))
    tokens = tokens.expand(memory.tokens.shape[0], -1, -1, -1).clone()
    tokens = tokens.to(device=memory.tokens.device, dtype=memory.tokens.dtype)
    tokens = tokens.masked_fill(~memory.patch_valid_mask.unsqueeze(-1), 0)
    return replace(memory, tokens=tokens.detach())


class P1DinoCompiledLiberoPolicy(nn.Module):
    """Adapter exposing cached-text, compiled action inference to LIBERO."""

    def __init__(
        self,
        policy: FastWAMP1DinoBCPolicy,
        *,
        prompt_cache_dir: str | Path,
        compile_enabled: bool = True,
        compile_mode: str = "reduce-overhead",
        parity_dtype_atol: float = 0.02,
        memory_mode: str = "correct",
    ) -> None:
        super().__init__()
        self.policy = policy.eval()
        self.actor = policy.actor
        self.prompt_cache_dir = Path(prompt_cache_dir).expanduser().resolve()
        self.compile_enabled = bool(compile_enabled)
        self.compile_mode = str(compile_mode)
        self.parity_dtype_atol = float(parity_dtype_atol)
        self.memory_mode = str(memory_mode).strip().lower()
        if self.memory_mode not in P1_POLICY_MEMORY_MODES:
            raise ValueError(
                "LIBERO inference memory_mode must be one of "
                f"{sorted(P1_POLICY_MEMORY_MODES)}."
            )
        self.memory_enabled = self.memory_mode != "off"
        self.random_patch_seed = P1_RANDOM_PATCH_SEED
        self._random_visual_encoder = None
        if self.memory_mode == "random_vit":
            if isinstance(policy.visual_encoder, FrozenVisualPatchEncoder):
                raise TypeError("The random-ViT P0-A control currently requires V1 DINO-S.")
            self._random_visual_encoder = copy.deepcopy(policy.visual_encoder)
            cuda_devices = (
                [self.device.index]
                if self.device.type == "cuda" and self.device.index is not None
                else []
            )
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(self.random_patch_seed)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(self.random_patch_seed)
                self._random_visual_encoder.model.init_weights()
            self._random_visual_encoder.requires_grad_(False).eval()
        if not self.prompt_cache_dir.is_dir():
            raise FileNotFoundError(
                f"Encoded prompt cache directory is missing: {self.prompt_cache_dir}"
            )
        if getattr(self.actor, "text_encoder", None) is not None:
            raise RuntimeError(
                "P1 LIBERO evaluation must not allocate a runtime text encoder."
            )
        if torch.is_tensor(self.actor.action_expert.freqs):
            self.actor.action_expert.freqs = self.actor.action_expert.freqs.to(
                self.device
            )
        if isinstance(self.actor.video_expert.freqs, tuple):
            self.actor.video_expert.freqs = tuple(
                value.to(self.device) for value in self.actor.video_expert.freqs
            )
        self._kernel_eager = _P1DenoiseKernel(
            policy,
            memory_enabled=self.memory_enabled,
        ).eval()
        self._kernel_compiled = (
            torch.compile(
                self._kernel_eager,
                mode=self.compile_mode,
                fullgraph=False,
                dynamic=False,
            )
            if self.compile_enabled
            else self._kernel_eager
        )
        self._prefill_compiled = (
            torch.compile(
                self.actor._prefill_step_compiled,
                mode="default",
                fullgraph=False,
                dynamic=False,
            )
            if self.compile_enabled
            else self.actor._prefill_step_compiled
        )
        self._prompt: str | None = None
        self._prompt_context: tuple[torch.Tensor, torch.Tensor] | None = None
        self._prompt_audit: dict[str, Any] | None = None
        self._prompt_load_count = 0
        self._runtime_encode_count = 0
        self._inference_call_count = 0
        self._visual_memory_encode_count = 0
        self._parity_report: CompileParityReport | None = None
        self._last_visual_input_audit: dict[str, Any] | None = None
        self._schedule_cache: dict[
            tuple[int, torch.dtype, float | None],
            tuple[torch.Tensor, torch.Tensor],
        ] = {}
        self._attention_mask_cache: dict[tuple[int, int, int], torch.Tensor] = {}
        if self.requires_external_visual_pixels:
            asset = self.policy.visual_encoder.asset
            self._visual_compile_cache_key = contract_sha256(
                {
                    "schema": "fastwam-p1-visual-compile-key-v2",
                    "family": asset.family,
                    "variant": asset.variant,
                    "input_size": asset.input_size,
                    "asset_contract_sha256": asset.asset_contract_sha256,
                    "memory_contract_sha256": self.policy.expected_memory_contract,
                    "reader_contract_sha256": (
                        self.policy.visual_reader.reader_contract_sha256
                    ),
                }
            )
        else:
            self._visual_compile_cache_key = None

    @property
    def device(self) -> torch.device:
        return self.policy.device

    @property
    def torch_dtype(self) -> torch.dtype:
        return self.policy.dtype

    @property
    def proprio_dim(self) -> int:
        return self.policy.p1_config.action.proprio_dim

    @property
    def requires_external_visual_pixels(self) -> bool:
        """Whether the sidecar must receive independently resized raw cameras."""

        return isinstance(self.policy.visual_encoder, FrozenVisualPatchEncoder)

    @property
    def visual_input_size(self) -> int | None:
        """Return the registered V2 sidecar size without changing FastWAM input."""

        if not self.requires_external_visual_pixels:
            return None
        return int(self.policy.visual_encoder.asset.input_size)

    @property
    def visual_camera_ids(self) -> tuple[str, ...]:
        return self.policy.p1_config.camera_ids

    @property
    def visual_compile_cache_key(self) -> str | None:
        """Return the family/variant/size-specific compile identity."""

        return self._visual_compile_cache_key

    @property
    def visual_input_audit(self) -> dict[str, Any] | None:
        """Return the last V2 raw-source and target-resolution record."""

        return (
            None
            if self._last_visual_input_audit is None
            else dict(self._last_visual_input_audit)
        )

    @property
    def parity_report(self) -> dict[str, Any] | None:
        return None if self._parity_report is None else self._parity_report.as_dict()

    @property
    def dino_call_audit(self) -> dict[str, Any]:
        """Return the causal-sidecar call-count contract for this worker."""

        parity_extra = int(self.compile_enabled and self._parity_report is not None)
        expected = (
            self._inference_call_count + parity_extra if self.memory_enabled else 0
        )
        return {
            "memory_mode": self.memory_mode,
            "random_patch_seed": (
                self.random_patch_seed
                if self.memory_mode in {"random_tensor", "random_vit"}
                else None
            ),
            "inference_calls": self._inference_call_count,
            "visual_memory_encoder_calls": self._visual_memory_encode_count,
            "compile_parity_extra_call": parity_extra,
            "expected_visual_memory_encoder_calls": expected,
            "contract_passed": self._visual_memory_encode_count == expected,
        }

    @property
    def prompt_audit(self) -> dict[str, Any]:
        if self._prompt_audit is None:
            raise RuntimeError("No encoded prompt has been loaded for this task.")
        return {
            **self._prompt_audit,
            "cache_load_count_for_current_task": self._prompt_load_count,
            "runtime_text_encoder_calls": self._runtime_encode_count,
            "context_reused_across_replans": True,
        }

    def clear_prompt(self) -> None:
        """Release the prior task context before loading the next task."""

        self._prompt = None
        self._prompt_context = None
        self._prompt_audit = None
        self._prompt_load_count = 0

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Load an offline T5 embedding exactly once for the current task."""

        if self._prompt == prompt and self._prompt_context is not None:
            return self._prompt_context
        if self._prompt is not None:
            raise RuntimeError("Prompt changed without an explicit task cache reset.")
        context, mask, path = load_text_embedding(
            self.prompt_cache_dir,
            prompt,
            device=self.device,
            dtype=self.torch_dtype,
        )
        self._prompt = prompt
        self._prompt_context = (context, mask)
        self._prompt_load_count += 1
        self._prompt_audit = {
            "mode": "offline_preencoded_t5",
            "prompt_sha256": prompt_sha256(prompt),
            "cache_path": str(path),
            "cache_file_sha256": sha256_file(path),
            "context_shape": list(context.shape),
            "context_dtype": str(context.dtype),
            "mask_shape": list(mask.shape),
            "load_text_encoder": False,
        }
        return context, mask

    def _prepare_condition(
        self,
        *,
        input_image: torch.Tensor,
        proprio: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        compiled_prefill: bool,
        visual_camera_pixels: torch.Tensor | None = None,
        visual_camera_valid_mask: torch.Tensor | None = None,
        visual_camera_source_resolution: torch.Tensor | None = None,
    ) -> CachedActionCondition:
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        current_frame = input_image.unsqueeze(2)
        current_latents = self.actor._encode_video_latents(
            current_frame,
            tiled=self.policy.p1_config.action.tiled_vae,
        )
        proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
        context = context.to(device=self.device, dtype=self.torch_dtype)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool)
        context, context_mask = self.actor._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=proprio,
        )
        fuse_flag = bool(
            getattr(self.actor.video_expert, "fuse_vae_embedding_in_latents", False)
        )
        video_pre = self.actor.video_expert.pre_dit(
            x=current_latents,
            timestep=torch.zeros(
                current_latents.shape[0],
                device=self.device,
                dtype=self.torch_dtype,
            ),
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        if video_seq_len != tokens_per_frame:
            raise RuntimeError("P1 inference prefill received more than one frame.")
        mask_key = (
            video_seq_len,
            self.policy.p1_config.action.action_horizon,
            tokens_per_frame,
        )
        if mask_key not in self._attention_mask_cache:
            self._attention_mask_cache[mask_key] = self.actor._build_mot_attention_mask(
                video_seq_len=video_seq_len,
                action_seq_len=self.policy.p1_config.action.action_horizon,
                video_tokens_per_frame=tokens_per_frame,
                device=self.device,
            )
        attention_mask = self._attention_mask_cache[mask_key]
        prefill = (
            self._prefill_compiled
            if compiled_prefill
            else self.actor._prefill_step_compiled
        )
        cache_k, cache_v = prefill(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context=video_pre["context"],
            video_context_mask=video_pre["context_mask"],
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        memory = None
        if not self.memory_enabled:
            self._last_visual_input_audit = None
        elif self.requires_external_visual_pixels:
            if visual_camera_pixels is None:
                raise ValueError(
                    "V2 visual inference requires raw-derived sidecar camera pixels."
                )
            cameras = visual_camera_pixels.to(device=self.device, dtype=torch.uint8)
            if visual_camera_valid_mask is None:
                valid_mask = torch.ones(
                    cameras.shape[:2],
                    dtype=torch.bool,
                    device=self.device,
                )
            else:
                valid_mask = visual_camera_valid_mask.to(
                    device=self.device,
                    dtype=torch.bool,
                )
            source_resolution = visual_camera_source_resolution
            if source_resolution is not None:
                source_resolution = source_resolution.to(
                    device=self.device,
                    dtype=torch.int32,
                )
            prepared = PreparedVisualCameraBatch(
                pixels=cameras,
                camera_ids=self.policy.p1_config.camera_ids,
                camera_valid_mask=valid_mask,
                input_size=int(self.visual_input_size),
                input_contract_sha256=(
                    self.policy.p1_config.camera_input_contract_sha256
                ),
                source_resolution=source_resolution,
            )
        else:
            camera_pixels = (
                ((input_image[:, :, :, :224].float() + 1.0) * 127.5)
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
            )
            wrist_pixels = (
                ((input_image[:, :, :, 224:].float() + 1.0) * 127.5)
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
            )
            cameras = torch.stack((camera_pixels, wrist_pixels), dim=1)
            prepared = PreparedCameraBatch(
                pixels=cameras,
                camera_ids=self.policy.p1_config.camera_ids,
                camera_valid_mask=torch.ones(
                    (input_image.shape[0], 2),
                    dtype=torch.bool,
                    device=self.device,
                ),
                input_contract_sha256=(
                    self.policy.p1_config.camera_input_contract_sha256
                ),
            )
        if self.memory_enabled:
            self._visual_memory_encode_count += 1
            visual_encoder = (
                self._random_visual_encoder
                if self.memory_mode == "random_vit"
                else self.policy.visual_encoder
            )
            assert visual_encoder is not None
            memory = visual_encoder.prepare_memory(PolicyRegime.UNCOND, prepared)
            if memory is None:
                raise RuntimeError("P1 visual memory was not constructed.")
            if self.memory_mode == "random_tensor":
                if not isinstance(memory, NativePatchMemory):
                    raise TypeError("The random-tensor P0-A control requires V1 memory.")
                memory = _fixed_random_patch_memory(
                    memory,
                    seed=self.random_patch_seed,
                )
            elif self.memory_mode == "random_vit":
                if not isinstance(memory, NativePatchMemory):
                    raise TypeError("The random-ViT P0-A control requires V1 memory.")
        if self.memory_enabled and self.requires_external_visual_pixels:
            assert memory is not None
            assert memory.source_resolution is not None
            source_sizes = memory.source_resolution.detach().cpu().tolist()
            target = int(self.visual_input_size)
            self._last_visual_input_audit = {
                "camera_ids": list(self.visual_camera_ids),
                "source_resolution_hw": source_sizes,
                "target_resolution_hw": [target, target],
                "source_stage": "highest_available_oriented_environment_rgb",
                "upsampled_from_lower_resolution": any(
                    valid and (int(height) < target or int(width) < target)
                    for sample_sizes, sample_valid in zip(
                        source_sizes,
                        prepared.camera_valid_mask.detach().cpu().tolist(),
                        strict=True,
                    )
                    for (height, width), valid in zip(
                        sample_sizes,
                        sample_valid,
                        strict=True,
                    )
                ),
            }
        return CachedActionCondition(
            context=context,
            context_mask=context_mask,
            video_kv_cache=[
                {"k": key, "v": value}
                for key, value in zip(cache_k, cache_v, strict=True)
            ],
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            current_frame_video_tokens=tokens_per_frame,
            visual=(
                VisualReadCondition(memory=memory, proprio=proprio)
                if memory is not None
                else None
            ),
        )

    def _solve(
        self,
        latents: torch.Tensor,
        condition: CachedActionCondition,
        *,
        num_inference_steps: int,
        sigma_shift: float | None,
        compiled: bool,
    ) -> torch.Tensor:
        kernel = self._kernel_compiled if compiled else self._kernel_eager
        video_cache_k = [item["k"] for item in condition.video_kv_cache]
        video_cache_v = [item["v"] for item in condition.video_kv_cache]
        total = condition.video_seq_len + latents.shape[1]
        action_mask = condition.attention_mask[condition.video_seq_len : total, :total]
        schedule_key = (
            int(num_inference_steps),
            latents.dtype,
            None if sigma_shift is None else float(sigma_shift),
        )
        if schedule_key not in self._schedule_cache:
            self._schedule_cache[schedule_key] = (
                self.actor.infer_action_scheduler.build_inference_schedule(
                    num_inference_steps=int(num_inference_steps),
                    device=self.device,
                    dtype=latents.dtype,
                    shift_override=sigma_shift,
                )
            )
        timesteps, deltas = self._schedule_cache[schedule_key]
        if self.memory_enabled and condition.visual is None:
            raise RuntimeError("Correct-memory inference lost its visual condition.")
        visual_memory = None if condition.visual is None else condition.visual.memory
        visual_proprio = None if condition.visual is None else condition.visual.proprio
        with self.policy.lora_adapter.regime_context.use(PolicyRegime.UNCOND):
            for step_t, delta in zip(timesteps, deltas, strict=True):
                timestep = step_t.unsqueeze(0).to(
                    device=self.device,
                    dtype=latents.dtype,
                )
                velocity = kernel(
                    latents,
                    timestep,
                    condition.context,
                    condition.context_mask,
                    video_cache_k,
                    video_cache_v,
                    action_mask,
                    visual_memory,
                    visual_proprio,
                    condition.current_frame_video_tokens,
                )
                latents = self.actor.infer_action_scheduler.step(
                    velocity,
                    delta,
                    latents,
                )
        return latents

    @torch.no_grad()
    def infer_action(
        self,
        prompt: str | None,
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        num_inference_steps: int = 10,
        sigma_shift: float | None = None,
        seed: int | None = 42,
        rand_device: str = "cpu",
        visual_camera_pixels: torch.Tensor | None = None,
        visual_camera_valid_mask: torch.Tensor | None = None,
        visual_camera_source_resolution: torch.Tensor | None = None,
        **_unused: Any,
    ) -> dict[str, torch.Tensor]:
        """Generate one action chunk through compiled P1 UNCOND inference."""

        if prompt is not None or context is None or context_mask is None:
            raise ValueError("P1 evaluation requires the reused encoded context.")
        if proprio is None or tuple(proprio.shape) != (1, self.proprio_dim):
            raise ValueError("P1 LIBERO proprio must have shape [1,8].")
        if int(action_horizon) != self.policy.p1_config.action.action_horizon:
            raise ValueError("P1 action horizon changed from the trained contract.")
        self._inference_call_count += 1
        generator = None
        if seed is not None:
            generator = torch.Generator(device=rand_device).manual_seed(int(seed))
        initial = torch.randn(
            (1, int(action_horizon), self.policy.p1_config.action.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        if self._parity_report is None and self.compile_enabled:
            eager_condition = self._prepare_condition(
                input_image=input_image,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                compiled_prefill=False,
                visual_camera_pixels=visual_camera_pixels,
                visual_camera_valid_mask=visual_camera_valid_mask,
                visual_camera_source_resolution=(visual_camera_source_resolution),
            )
            compiled_condition = self._prepare_condition(
                input_image=input_image,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                compiled_prefill=True,
                visual_camera_pixels=visual_camera_pixels,
                visual_camera_valid_mask=visual_camera_valid_mask,
                visual_camera_source_resolution=(visual_camera_source_resolution),
            )
            eager_a = self._solve(
                initial.clone(),
                eager_condition,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                compiled=False,
            )
            eager_b = self._solve(
                initial.clone(),
                eager_condition,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                compiled=False,
            )
            started = time.perf_counter()
            compiled_out = self._solve(
                initial.clone(),
                compiled_condition,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                compiled=True,
            )
            torch.cuda.synchronize(self.device)
            warmup_seconds = time.perf_counter() - started
            repeat = float((eager_a.float() - eager_b.float()).abs().max().item())
            error = float((eager_a.float() - compiled_out.float()).abs().max().item())
            threshold = max(self.parity_dtype_atol, 1.25 * repeat)
            self._parity_report = CompileParityReport(
                baseline_repeat_max_abs=repeat,
                compiled_max_abs=error,
                dtype_atol=self.parity_dtype_atol,
                threshold=threshold,
                passed=error <= threshold,
                compile_warmup_seconds=warmup_seconds,
            )
            if not self._parity_report.passed:
                raise RuntimeError(
                    "P1 torch.compile parity failed: "
                    f"max_abs={error:.6f}, threshold={threshold:.6f}."
                )
            output = compiled_out
        else:
            condition = self._prepare_condition(
                input_image=input_image,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                compiled_prefill=self.compile_enabled,
                visual_camera_pixels=visual_camera_pixels,
                visual_camera_valid_mask=visual_camera_valid_mask,
                visual_camera_source_resolution=(visual_camera_source_resolution),
            )
            output = self._solve(
                initial,
                condition,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                compiled=self.compile_enabled,
            )
        return {"action": output[0].detach().to(device="cpu", dtype=torch.float32)}


class P1VisualCompiledLiberoPolicy(P1DinoCompiledLiberoPolicy):
    """V2 public name for compiled DINOv3/LingBot LIBERO inference."""

    def __init__(self, policy: FastWAMP1DinoBCPolicy, **kwargs: Any) -> None:
        super().__init__(policy, **kwargs)
        if not self.requires_external_visual_pixels:
            raise TypeError("P1VisualCompiledLiberoPolicy requires a V2 encoder.")
