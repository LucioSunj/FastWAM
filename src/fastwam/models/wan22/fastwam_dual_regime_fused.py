"""Fused single-pass dual-regime training for the metric-adaptive FastWAM variants.

Alternative to the two-forward dual-regime training implemented inside
``MetricAdaptiveFastWAM`` / ``MetricAdaptiveFastWAMJoint``. Inference routing,
checkpoint format, and configs knobs are inherited unchanged from those classes;
ONLY ``training_loss`` is replaced.

Key idea
--------
Under the adaptive configs, three facts already hold:

1. ``video_attention_mask_mode="first_frame_causal"`` (or ``per_frame_causal``):
   the first-frame video tokens attend ONLY to themselves inside the full
   training video sequence.
2. ``fuse_vae_embedding_in_latents=true`` + ``seperated_timestep=true``: the
   first latent frame is the CLEAN input frame and receives timestep-0
   token-wise modulation (``wan_video_dit.pre_dit`` forces
   ``token_timesteps[:, 0, :] = 0``).
3. Frame-0 RoPE coordinates are identical whether the video sequence holds one
   frame or many.

Together these mean the main training forward already contains a layer-exact
replica of the base branch clean first-frame hidden state. The implementation
therefore makes one `self.mot(...)` call per step, while each MoT layer runs two
component-aligned attention groups:

1. Main: `[full main video | main action]`, with the exact reference shape.
2. Base: `[clean first frame | base action]`, also with the exact reference
   shape. The first-frame span recomputes Q/K/V and participates in SDPA but is
   read-only: its attention/post output is discarded, and the next layer reuses
   the main group updated first-frame hidden state.

Main and base action separately execute Q/K/V, output projection,
cross-attention, and FFN. The existing global boolean mask supplies each group
submatrix without changing visibility semantics. One MoT call then trains:

    video loss            <- noisy-video block (computed exactly once)
    main-regime action    <- draft 0 (joint: full video / idm: cond video)
    base-regime action    <- draft 1 (first-frame columns only)

This removes the complete base-video post/cross-attention/FFN shadow path,
yields one autograd graph, and drops the `share_inputs` copied-code or
second-VAE-encode trade-off. Base-action loss still backpropagates through the
read-only first-frame K/V into the video expert.

Randomness is isolated in ``_sample_dual_regime_draws`` so tests can replay the
same draws through both this fused forward and a reference two-forward
computation and assert numerical parity (see ``tests/test_dual_regime_fused.py``).
"""
from __future__ import annotations

import math
from typing import Any

import torch

from fastwam.utils.logging_config import get_logger
from fastwam.adaptive_gate.sdr_preflight import couple_action_noise_draws
from fastwam.adaptive_gate.training import normalized_dual_regime_action_loss

from .dual_regime_masks import build_multi_regime_attention_mask, merge_action_draft_payloads
from .mot import MoTAttentionGroup, MoTExpertSpan
from .regime import REGIME_UNCOND
from .fastwam_metric_adaptive import (
    MetricAdaptiveFastWAM,
    _maybe_instantiate,
    _mapping_value,
    _scheduler_value,
    _to_plain_dict,
)
from .fastwam_metric_adaptive_joint import MetricAdaptiveFastWAMJoint

logger = get_logger(__name__)


class _FusedDualRegimeTrainingMixin:
    """Shared fused-training machinery; variants supply the video-side layout.

    Subclasses must set ``main_regime_name`` and implement
    ``_prepare_fused_video_side``; the idm variant additionally overrides
    ``_sample_variant_draws`` for its teacher-forcing cond branch.
    """

    main_regime_name: str = "main"

    # ---- randomness (all sampling lives here, for replayable parity tests) --
    def _sample_dual_regime_draws(self, inputs: dict) -> dict[str, Any]:
        input_latents = inputs["input_latents"]
        action = inputs["action"]
        batch_size = input_latents.shape[0]
        draws: dict[str, Any] = {
            "noise_video": torch.randn_like(input_latents),
            "timestep_video": self.train_video_scheduler.sample_training_t(
                batch_size=batch_size, device=self.device, dtype=input_latents.dtype
            ),
            "action_regimes": [
                {
                    "name": name,
                    "noise": torch.randn_like(action),
                    "timestep": self.train_action_scheduler.sample_training_t(
                        batch_size=batch_size, device=self.device, dtype=action.dtype
                    ),
                }
                for name in (self.main_regime_name, "base")
            ],
        }
        self._sample_variant_draws(inputs, draws)
        coupling = getattr(self, "_diagnostic_action_noise_coupling", None)
        if coupling is None:
            return draws
        return couple_action_noise_draws(draws, mode=str(coupling))

    def _sample_variant_draws(self, inputs: dict, draws: dict) -> None:
        """Hook for variant-specific extra draws (idm cond branch)."""

    # ---- variant hook -------------------------------------------------------
    def _prepare_fused_video_side(self, inputs: dict, draws: dict) -> dict[str, Any]:
        """Build the merged video-side payload for the fused forward.

        Must return a dict with keys: ``tokens``, ``freqs``, ``t_mod``,
        ``context``, ``context_mask`` (merged video-expert payload),
        ``block_masks`` (list of square video->video masks, one per block),
        ``pred_slice`` + ``pred_pre`` (which video-token span feeds
        ``video_expert.post_dit`` for the video loss), ``target_video``,
        ``main_span`` and ``base_span`` (video-column spans, concatenated
        coordinates, that the main/base action drafts may attend).
        """
        raise NotImplementedError

    # ---- shared checks ------------------------------------------------------
    @staticmethod
    def _require_first_frame_isolated(block_mask: torch.Tensor, tokens_per_frame: int) -> None:
        """The base regime is layer-exact only if first-frame rows see nothing
        beyond first-frame columns (true for ``first_frame_causal`` and
        ``per_frame_causal``; false for ``bidirectional``)."""
        first = min(int(tokens_per_frame), int(block_mask.shape[0]))
        if bool(block_mask[:first, first:].any()):
            raise ValueError(
                "Fused dual-regime training requires the first-frame video rows to "
                "attend only first-frame columns so that they replicate base-branch "
                "inference exactly. The current `video_attention_mask_mode` violates "
                "this (use `first_frame_causal` or `per_frame_causal`)."
            )

    @staticmethod
    def _require_token_wise_t_mod(t_mod: torch.Tensor) -> None:
        if t_mod.ndim != 4:
            raise ValueError(
                "Fused dual-regime training requires token-wise video `t_mod`; "
                "ensure `seperated_timestep=true` and `fuse_vae_embedding_in_latents=true`."
            )

    @staticmethod
    def _require_temporal_patch_size(video_expert) -> None:
        patch_size = getattr(video_expert, "patch_size", None)
        if patch_size is None or len(patch_size) < 1 or int(patch_size[0]) != 1:
            raise ValueError(
                "Fused dual-regime equivalence requires temporal patch_size[0] == 1; "
                f"got {patch_size!r}."
            )

    # ---- fused forward (deterministic given `draws`) ------------------------
    def _fused_dual_regime_forward(self, inputs: dict, draws: dict) -> dict[str, Any]:
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]

        video_side = self._prepare_fused_video_side(inputs, draws)

        drafts: list[dict[str, Any]] = []
        for regime in draws["action_regimes"]:
            noisy_action = self.train_action_scheduler.add_noise(
                action, regime["noise"], regime["timestep"]
            )
            target_action = self.train_action_scheduler.training_target(
                action, regime["noise"], regime["timestep"]
            )
            action_pre = self.action_expert.pre_dit(
                action_tokens=noisy_action,
                timestep=regime["timestep"],
                context=context,
                context_mask=context_mask,
            )
            drafts.append(
                {
                    "name": regime["name"],
                    "pre": action_pre,
                    "target": target_action,
                    "timestep": regime["timestep"],
                }
            )

        merged_action = merge_action_draft_payloads([d["pre"] for d in drafts])
        span_by_name = {
            self.main_regime_name: video_side["main_span"],
            "base": video_side["base_span"],
        }
        attention_mask = build_multi_regime_attention_mask(
            video_block_masks=video_side["block_masks"],
            draft_lens=[end - start for start, end in merged_action["draft_slices"]],
            draft_video_spans=[[span_by_name[d["name"]]] for d in drafts],
            device=video_side["tokens"].device,
        )
        action_span_by_name = {
            draft["name"]: span
            for draft, span in zip(drafts, merged_action["draft_slices"])
        }
        main_action_start, main_action_end = action_span_by_name[self.main_regime_name]
        base_action_start, base_action_end = action_span_by_name["base"]
        base_video_start, base_video_end = video_side["base_span"]
        attention_groups = (
            MoTAttentionGroup(
                name=self.main_regime_name,
                spans=(
                    MoTExpertSpan("video", 0, int(video_side["tokens"].shape[1])),
                    MoTExpertSpan("action", main_action_start, main_action_end),
                ),
                # Regime labels are consumed only by regime-aware wrapped
                # submodules (stage-2 UNCOND-gated adapters); with none
                # attached they change nothing. `main_regime_name` must be a
                # registered regime — a variant with an unregistered name now
                # fails closed in `MoT._validate_attention_groups` instead of
                # silently running unlabeled.
                regime=self.main_regime_name,
            ),
            MoTAttentionGroup(
                name="base",
                spans=(
                    MoTExpertSpan(
                        "video", base_video_start, base_video_end, write=False
                    ),
                    MoTExpertSpan("action", base_action_start, base_action_end),
                ),
                regime=REGIME_UNCOND,
            ),
        )

        tokens_out = self.mot(
            embeds_all={"video": video_side["tokens"], "action": merged_action["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_side["freqs"], "action": merged_action["freqs"]},
            context_all={
                "video": {"context": video_side["context"], "mask": video_side["context_mask"]},
                "action": {"context": merged_action["context"], "mask": merged_action["context_mask"]},
            },
            t_mod_all={"video": video_side["t_mod"], "action": merged_action["t_mod"]},
            attention_groups=attention_groups,
        )

        pred_start, pred_end = video_side["pred_slice"]
        pred_video = self.video_expert.post_dit(
            tokens_out["video"][:, pred_start:pred_end], video_side["pred_pre"]
        )
        action_drafts = []
        for draft, (start, end) in zip(drafts, merged_action["draft_slices"]):
            pred_action = self.action_expert.post_dit(tokens_out["action"][:, start:end], draft["pre"])
            action_drafts.append({**draft, "pred": pred_action})

        return {
            "pred_video": pred_video,
            "target_video": video_side["target_video"],
            "timestep_video": draws["timestep_video"],
            "action_drafts": action_drafts,
        }

    # ---- training loss -------------------------------------------------------
    def training_loss(
        self,
        sample,
        tiled: bool = False,
        draws: dict | None = None,
        draws_out: dict | None = None,
    ):
        """Single fused forward training BOTH action regimes; video loss once.

        Stage-2 W8 additions, defaulting to the pre-W8 behaviour: ``draws``
        injects a complete externally supplied draw dict (the exact structure
        produced by ``_sample_dual_regime_draws``; used VERBATIM — the
        diagnostic noise-coupling hook is NOT re-applied to injected draws),
        and ``draws_out`` — a caller-supplied dict — receives references to the
        draws actually used, so ``(t, eps)`` pairs can be cached and replayed.
        """
        self._require_temporal_patch_size(self.video_expert)
        inputs = self.build_inputs(sample, tiled=tiled)
        if inputs["first_frame_latents"] is None:
            raise ValueError(
                "Fused dual-regime training requires `fuse_vae_embedding_in_latents=true` "
                "so that `first_frame_latents` is available (the adaptive configs set this)."
            )
        if draws is None:
            draws = self._sample_dual_regime_draws(inputs)
        if draws_out is not None:
            draws_out.update(draws)
        out = self._fused_dual_regime_forward(inputs, draws)

        # Video loss: identical reduction to the parents; first (clean/fused)
        # latent step excluded because `first_frame_latents` is present.
        pred_video = out["pred_video"][:, :, 1:]
        target_video = out["target_video"][:, :, 1:]
        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=inputs["image_is_pad"],
            include_initial_video_step=False,
        )
        video_weight = self.train_video_scheduler.training_weight(out["timestep_video"]).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        regime_losses: dict[str, torch.Tensor] = {}
        for draft in out["action_drafts"]:
            per_sample = self._action_loss_per_sample(
                draft["pred"], draft["target"], inputs["action_is_pad"]
            )
            action_weight = self.train_action_scheduler.training_weight(draft["timestep"]).to(
                per_sample.device, dtype=per_sample.dtype
            )
            regime_losses[draft["name"]] = (per_sample * action_weight).mean()

        w_base = float(
            getattr(self, "action_regime_weight_uncond", getattr(self, "action_regime_weight_base", 1.0))
        )
        loss_action_main = regime_losses[self.main_regime_name]
        loss_action_base = regime_losses["base"]
        if bool(getattr(self, "_capture_raw_dual_regime_losses", False)):
            # Kept on the live graph only for the sparse trainer diagnostic.
            # The trainer consumes and deletes this attribute before backward.
            self._raw_dual_regime_losses = {
                self.main_regime_name: loss_action_main,
                "uncond": loss_action_base,
            }

        combined_action, main_raw_contribution, base_raw_contribution = (
            normalized_dual_regime_action_loss(loss_action_main, loss_action_base, w_base)
        )
        loss_total = (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_action * combined_action
        )
        main_contribution = self.loss_lambda_action * main_raw_contribution
        base_contribution = self.loss_lambda_action * base_raw_contribution
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            f"loss_action_{self.main_regime_name}_raw": float(
                loss_action_main.detach().item()
            ),
            "loss_action_uncond_raw": float(loss_action_base.detach().item()),
            f"loss_action_{self.main_regime_name}": float(main_contribution.detach().item()),
            "loss_action_uncond": float(base_contribution.detach().item()),
            "loss_action_combined": self.loss_lambda_action
            * float(combined_action.detach().item()),
            "action_regime_weight_uncond": w_base,
        }
        return loss_total, loss_dict


class FusedDualRegimeFastWAMJoint(_FusedDualRegimeTrainingMixin, MetricAdaptiveFastWAMJoint):
    """base/joint adaptive model with fused single-pass dual-regime training.

    Inference routing (base <-> joint) and checkpoint format are inherited from
    ``MetricAdaptiveFastWAMJoint`` untouched. The fused sequence per step is

        [ noisy_video (S_v) | action_joint (S_a) | action_base (S_a) ]

    where ``action_joint`` attends the full video (FastWAMJoint mask) and
    ``action_base`` attends only the first-frame columns (FastWAM mask), whose
    hidden states are a layer-exact replica of base-branch inference (see the
    module docstring).
    """

    main_regime_name = "joint"

    def _prepare_fused_video_side(self, inputs: dict, draws: dict) -> dict[str, Any]:
        input_latents = inputs["input_latents"]
        latents = self.train_video_scheduler.add_noise(
            input_latents, draws["noise_video"], draws["timestep_video"]
        )
        target_video = self.train_video_scheduler.training_target(
            input_latents, draws["noise_video"], draws["timestep_video"]
        )
        latents[:, :, 0:1] = inputs["first_frame_latents"]

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=draws["timestep_video"],
            context=inputs["context"],
            context_mask=inputs["context_mask"],
            action=None,  # action_conditioned=false -> ignored
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )
        self._require_token_wise_t_mod(video_pre["t_mod"])

        video_seq_len = int(video_pre["tokens"].shape[1])
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        block_mask = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=tokens_per_frame,
            device=video_pre["tokens"].device,
        )
        self._require_first_frame_isolated(block_mask, tokens_per_frame)

        first_frame_tokens = min(tokens_per_frame, video_seq_len)
        return {
            "tokens": video_pre["tokens"],
            "freqs": video_pre["freqs"],
            "t_mod": video_pre["t_mod"],
            "context": video_pre["context"],
            "context_mask": video_pre["context_mask"],
            "block_masks": [block_mask],
            "pred_slice": (0, video_seq_len),
            "pred_pre": video_pre,
            "target_video": target_video,
            "main_span": (0, video_seq_len),  # joint regime: full video
            "base_span": (0, first_frame_tokens),  # base regime: first frame only
        }


class FusedDualRegimeFastWAM(_FusedDualRegimeTrainingMixin, MetricAdaptiveFastWAM):
    """base/idm adaptive model with fused single-pass dual-regime training.

    Inference routing (base <-> idm) is inherited from
    ``MetricAdaptiveFastWAM`` untouched. The fused sequence per step is

        [ noisy_video (S_v) | cond_video (S_v) | action_idm (S_a) | action_base (S_a) ]

    mirroring FastWAMIDM's teacher-forcing layout: ``action_idm`` attends the
    cond-video block (per-sample noised w.p. ``video_cond_noise_prob``), and
    ``action_base`` attends only the cond block's first-frame columns — the
    clean first frame at timestep 0, a layer-exact replica of base-branch
    inference.
    """

    main_regime_name = "idm"

    def _sample_variant_draws(self, inputs: dict, draws: dict) -> None:
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        # Sampled unconditionally (unlike the parent's `if any()` guard) —
        # torch.where below reduces to the clean path when the mask is all-False.
        draws["cond_noise_mask"] = (
            torch.rand((batch_size,), device=self.device) < float(self.video_cond_noise_prob)
        )
        draws["timestep_video_cond_sampled"] = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=input_latents.dtype
        )
        draws["noise_video_cond"] = torch.randn_like(input_latents)

    def _prepare_fused_video_side(self, inputs: dict, draws: dict) -> dict[str, Any]:
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        fuse_flag = inputs["fuse_vae_embedding_in_latents"]

        # Block 0: noisy video (video-denoising target), first frame kept clean.
        latents_noisy = self.train_video_scheduler.add_noise(
            input_latents, draws["noise_video"], draws["timestep_video"]
        )
        target_video = self.train_video_scheduler.training_target(
            input_latents, draws["noise_video"], draws["timestep_video"]
        )
        latents_noisy[:, :, 0:1] = inputs["first_frame_latents"]

        # Block 1: teacher-forcing cond video, per-sample noised w.p.
        # `video_cond_noise_prob` (semantics identical to FastWAMIDM.training_loss).
        cond_noise_mask = draws["cond_noise_mask"]
        timestep_video_cond = torch.where(
            cond_noise_mask,
            draws["timestep_video_cond_sampled"],
            torch.zeros_like(draws["timestep_video_cond_sampled"]),
        ).to(dtype=input_latents.dtype)
        latents_cond_noisy = self.train_video_scheduler.add_noise(
            input_latents, draws["noise_video_cond"], draws["timestep_video_cond_sampled"]
        )
        latents_cond = torch.where(
            cond_noise_mask.view(batch_size, 1, 1, 1, 1), latents_cond_noisy, input_latents
        )
        latents_cond = latents_cond.clone()
        latents_cond[:, :, 0:1] = inputs["first_frame_latents"]

        video_pre_noisy = self.video_expert.pre_dit(
            x=latents_noisy,
            timestep=draws["timestep_video"],
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_pre_cond = self.video_expert.pre_dit(
            x=latents_cond,
            timestep=timestep_video_cond,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        self._require_token_wise_t_mod(video_pre_noisy["t_mod"])
        self._require_token_wise_t_mod(video_pre_cond["t_mod"])

        noisy_seq_len = int(video_pre_noisy["tokens"].shape[1])
        cond_seq_len = int(video_pre_cond["tokens"].shape[1])
        noisy_tpf = int(video_pre_noisy["meta"]["tokens_per_frame"])
        cond_tpf = int(video_pre_cond["meta"]["tokens_per_frame"])
        if noisy_tpf != cond_tpf:
            raise ValueError(
                "Teacher-forcing requires identical `tokens_per_frame` for noisy and cond "
                f"video blocks, got {noisy_tpf} and {cond_tpf}."
            )

        mask_noisy = self.video_expert.build_video_to_video_mask(
            video_seq_len=noisy_seq_len,
            video_tokens_per_frame=noisy_tpf,
            device=video_pre_noisy["tokens"].device,
        )
        mask_cond = self.video_expert.build_video_to_video_mask(
            video_seq_len=cond_seq_len,
            video_tokens_per_frame=cond_tpf,
            device=video_pre_cond["tokens"].device,
        )
        self._require_first_frame_isolated(mask_cond, cond_tpf)

        first_frame_tokens = min(cond_tpf, cond_seq_len)
        return {
            "tokens": torch.cat([video_pre_noisy["tokens"], video_pre_cond["tokens"]], dim=1),
            "freqs": torch.cat([video_pre_noisy["freqs"], video_pre_cond["freqs"]], dim=0),
            "t_mod": torch.cat([video_pre_noisy["t_mod"], video_pre_cond["t_mod"]], dim=1),
            "context": video_pre_noisy["context"],
            "context_mask": torch.cat(
                [video_pre_noisy["context_mask"], video_pre_cond["context_mask"]], dim=1
            ),
            "block_masks": [mask_noisy, mask_cond],
            "pred_slice": (0, noisy_seq_len),  # video loss ONLY from the noisy block
            "pred_pre": video_pre_noisy,
            "target_video": target_video,
            # idm regime: the cond block (concatenated video coordinates).
            "main_span": (noisy_seq_len, noisy_seq_len + cond_seq_len),
            # base regime: the cond block's clean first frame at timestep 0.
            "base_span": (noisy_seq_len, noisy_seq_len + first_frame_tokens),
        }


# --------------------------------------------------------------------------- #
# Factories (one shared body — the two per-variant factories in the inherited
# modules are otherwise near-identical copies of each other).
# --------------------------------------------------------------------------- #
def _create_fused_dual_regime(
    model_cls,
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    adaptive=None,
    router=None,
    train=None,
    checkpoint_task: str | None = None,
):
    adaptive_cfg = _to_plain_dict(adaptive if adaptive is not None else router)
    routing_metric = _maybe_instantiate(adaptive_cfg.get("metric"))
    routing_selector = _maybe_instantiate(adaptive_cfg.get("selector"))
    annotate_outputs = bool(adaptive_cfg.get("annotate_outputs", True))
    allow_internal_routing = bool(adaptive_cfg.get("allow_internal_routing", False))

    train_cfg = _to_plain_dict(train)
    if "action_regime_weight_uncond" in train_cfg:
        action_regime_weight_base = float(train_cfg["action_regime_weight_uncond"])
    else:
        action_regime_weight_base = float(train_cfg.get("action_regime_weight_base", 1.0))
        if "action_regime_weight_base" in train_cfg:
            logger.warning(
                "`train.action_regime_weight_base` is deprecated; use "
                "`train.action_regime_weight_uncond`."
            )
    if not math.isfinite(action_regime_weight_base) or action_regime_weight_base <= 0.0:
        raise ValueError(
            "train.action_regime_weight_uncond must be finite and > 0 for a dual-regime checkpoint, got "
            f"{action_regime_weight_base}."
        )
    if "share_inputs" in train_cfg:
        logger.info(
            "`train.share_inputs` is ignored by the fused dual-regime training "
            "(inputs are always built exactly once per step)."
        )

    def _container(cfg):
        from omegaconf import DictConfig, OmegaConf

        if isinstance(cfg, DictConfig):
            return OmegaConf.to_container(cfg, resolve=True)
        return cfg

    model = model_cls.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=_container(video_dit_config),
        action_dit_config=_container(action_dit_config),
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(_scheduler_value(video_scheduler, "train_shift", 5.0)),
        video_infer_shift=float(_scheduler_value(video_scheduler, "infer_shift", 5.0)),
        video_num_train_timesteps=int(_scheduler_value(video_scheduler, "num_train_timesteps", 1000)),
        action_train_shift=float(_scheduler_value(action_scheduler, "train_shift", 5.0)),
        action_infer_shift=float(_scheduler_value(action_scheduler, "infer_shift", 5.0)),
        action_num_train_timesteps=int(_scheduler_value(action_scheduler, "num_train_timesteps", 1000)),
        loss_lambda_video=float(_mapping_value(loss, "lambda_video", 1.0)),
        loss_lambda_action=float(_mapping_value(loss, "lambda_action", 1.0)),
        routing_metric=routing_metric,
        routing_selector=routing_selector,
        annotate_outputs=annotate_outputs,
        allow_internal_routing=allow_internal_routing,
    )
    # Set training knobs AFTER construction (the FastWAM base signature has no
    # **kwargs to thread them through) — same pattern as the inherited factories.
    model.action_regime_weight_uncond = action_regime_weight_base
    model.checkpoint_task = checkpoint_task
    return model


def create_fused_dual_regime_fastwam(**kwargs):
    """Hydra factory for the base/idm fused dual-regime model."""
    return _create_fused_dual_regime(FusedDualRegimeFastWAM, **kwargs)


def create_fused_dual_regime_fastwam_joint(**kwargs):
    """Hydra factory for the base/joint fused dual-regime model."""
    return _create_fused_dual_regime(FusedDualRegimeFastWAMJoint, **kwargs)
