from __future__ import annotations

import inspect
from typing import Any, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from fastwam.routing.metrics import MetricResult, PolicyEntropyMetric, RoutingDecision, ThresholdSelector
from fastwam.utils.logging_config import get_logger

from .fastwam import FastWAM
from .fastwam_joint import FastWAMJoint

# Reuse the generic (branch-agnostic) routing + factory helpers from the
# base/idm adaptive module so this variant stays in sync with that plumbing.
from .fastwam_metric_adaptive import (
    _ROUTING_ONLY_KWARGS,
    _InheritedRoutingContext,
    _maybe_instantiate,
    _mapping_value,
    _scheduler_value,
    _to_plain_dict,
)

logger = get_logger(__name__)


class MetricAdaptiveFastWAMJoint(FastWAMJoint):
    """FastWAMJoint subclass that can fall back to base FastWAM inference.

    This is the base/joint counterpart of ``MetricAdaptiveFastWAM`` (which routes
    base/idm). At inference a metric + selector chooses between:

    - ``base``  : ``FastWAM.infer_*`` — action conditions on the clean first
      frame only (fast, no future imagination).
    - ``joint`` : ``FastWAMJoint.infer_*`` — video and action are denoised
      together and the action conditions on the full (jointly-denoised) video
      (slow, full future imagination).

    Both branches share one set of weights / one checkpoint; only the inference
    algorithm (attention conditioning) differs. Training exposes the single
    shared action expert to BOTH conditioning regimes every step, with the
    video-denoising loss computed exactly once (in the joint forward).
    """

    def __init__(
        self,
        *args,
        routing_metric=None,
        routing_selector=None,
        annotate_outputs: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.configure_routing(
            routing_metric=routing_metric,
            routing_selector=routing_selector,
            annotate_outputs=annotate_outputs,
        )

    @classmethod
    def from_wan22_pretrained(
        cls,
        *,
        routing_metric=None,
        routing_selector=None,
        annotate_outputs: bool = True,
        **kwargs,
    ):
        model = super().from_wan22_pretrained(**kwargs)
        model.configure_routing(
            routing_metric=routing_metric,
            routing_selector=routing_selector,
            annotate_outputs=annotate_outputs,
        )
        return model

    def configure_routing(
        self,
        routing_metric=None,
        routing_selector=None,
        annotate_outputs: bool = True,
    ) -> None:
        self.routing_metric = routing_metric if routing_metric is not None else PolicyEntropyMetric(probe_branch="base")
        self.routing_selector = (
            routing_selector
            if routing_selector is not None
            else ThresholdSelector(threshold=0.0, low_branch="base", high_branch="joint", mode="ge")
        )
        self.annotate_outputs = bool(annotate_outputs)
        self.last_routing_decision: dict[str, Any] | None = None

    # ---- inference routing (identical to the base/idm variant) -----------
    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int | None = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        routing_metric_value: float | None = None,
        force_branch: str | None = None,
        return_routing_info: bool = False,
    ) -> dict[str, Any]:
        kwargs = locals()
        kwargs.pop("self")
        return self._route_inherited(method_name="infer_action", kwargs=kwargs)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = False,
        routing_metric_value: float | None = None,
        force_branch: str | None = None,
        return_routing_info: bool = False,
    ) -> dict[str, Any]:
        kwargs = locals()
        kwargs.pop("self")
        return self._route_inherited(method_name="infer_joint", kwargs=kwargs)

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        routing_metric_value: float | None = None,
        force_branch: str | None = None,
        return_routing_info: bool = False,
    ) -> dict[str, Any]:
        kwargs = locals()
        kwargs.pop("self")
        return self._route_inherited(method_name="infer", kwargs=kwargs)

    def _route_inherited(self, method_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        force_branch = kwargs.get("force_branch")
        return_routing_info = bool(kwargs.get("return_routing_info", False))
        context = _InheritedRoutingContext(model=self, method_name=method_name, kwargs=kwargs)

        if force_branch is not None:
            selected_branch = str(force_branch)
            self._validate_branch_name(selected_branch)
            metric_result = MetricResult(
                name="forced",
                value=0.0,
                details={"reason": "force_branch", "selected_branch": selected_branch},
            )
            decision = RoutingDecision(
                selected_branch=selected_branch,
                metric=metric_result,
                threshold=float(getattr(self.routing_selector, "threshold", 0.0)),
                mode="forced",
                low_branch=str(getattr(self.routing_selector, "low_branch", "")),
                high_branch=str(getattr(self.routing_selector, "high_branch", "")),
            )
        else:
            metric_result = self.routing_metric(context)
            decision = self.routing_selector.select(metric_result)
            selected_branch = decision.selected_branch
            self._validate_branch_name(selected_branch)

        output = dict(context.run_branch(selected_branch))
        routing_info = decision.as_dict()
        routing_info["method_name"] = method_name
        self.last_routing_decision = routing_info
        if self.annotate_outputs or return_routing_info:
            output["_routing"] = routing_info
        return output

    def _call_inherited_branch(self, branch_name: str, method_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        self._validate_branch_name(branch_name)
        if method_name == "infer":
            method_name = "infer_joint"
            kwargs = self._infer_kwargs_to_infer_joint_kwargs(kwargs)

        if branch_name == "base":
            method = getattr(FastWAM, method_name)
        else:
            method = getattr(FastWAMJoint, method_name)

        branch_kwargs = self._filter_kwargs_for_method(method, kwargs)
        logger.debug("Adaptive FastWAMJoint call `%s` through inherited branch `%s`.", method_name, branch_name)
        return method(self, **branch_kwargs)

    @staticmethod
    def _validate_branch_name(branch_name: str) -> None:
        if branch_name not in {"base", "joint"}:
            raise ValueError(f"Unknown branch `{branch_name}`. Expected one of: ['base', 'joint'].")

    @staticmethod
    def _infer_kwargs_to_infer_joint_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        joint_kwargs = dict(kwargs)
        num_frames = joint_kwargs.pop("num_frames")
        joint_kwargs["num_video_frames"] = num_frames
        joint_kwargs.pop("action_cfg_scale", None)
        if joint_kwargs.get("action_horizon") is None:
            raise ValueError("`action_horizon` is required for adaptive `infer` routing.")
        return joint_kwargs

    @staticmethod
    def _filter_kwargs_for_method(method, kwargs: dict[str, Any]) -> dict[str, Any]:
        signature = inspect.signature(method)
        params = signature.parameters
        accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values())
        call_kwargs = {}
        for key, value in kwargs.items():
            if key in _ROUTING_ONLY_KWARGS:
                continue
            if accepts_kwargs or key in params:
                call_kwargs[key] = value
        return call_kwargs

    # ---- corrected dual-regime training ---------------------------------
    # The inference router picks base or joint per call, but both share one set
    # of weights. To keep the shared action expert in-distribution for BOTH
    # branches, every training step computes the action loss under BOTH
    # conditioning regimes while the video-denoising loss is computed ONCE.
    #
    # Knobs are set by create_metric_adaptive_fastwam_joint (read via getattr so
    # the model still runs if wiring is omitted):
    #   self.action_regime_weight_base : float = 1.0   # w_base
    #   self.train_share_inputs        : bool  = True   # single VAE encode/step

    def _action_loss_per_sample(
        self,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Per-sample action loss, identical reduction to FastWAM/FastWAMJoint
        (fastwam.py:550-556). Returns [B] (pre-weight)."""
        action_loss_token = F.mse_loss(
            pred_action.float(), target_action.float(), reduction="none"
        ).mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(
                device=action_loss_token.device, dtype=action_loss_token.dtype
            )
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            return (action_loss_token * valid).sum(dim=1) / valid_sum
        return action_loss_token.mean(dim=1)

    def _base_regime_action_loss(self, inputs: dict) -> torch.Tensor:
        """BASE-regime action loss (differentiable scalar, batch mean).

        Mirrors FastWAM.infer_action (fastwam.py:993-1012): a SINGLE CLEAN
        first-frame latent at timestep_video=0, action=None into
        video_expert.pre_dit (action_conditioned=false ignores it), and the
        FastWAM first-frame attention mask. NO video loss is computed here
        (tokens_out["video"] is discarded), so the video-denoising loss is
        never double-counted.
        """
        first_frame_latents = inputs["first_frame_latents"]
        if first_frame_latents is None:
            raise ValueError(
                "Base-regime action training requires `fuse_vae_embedding_in_latents=true` "
                "so that `first_frame_latents` is available (the adaptive config sets this)."
            )
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        fuse_flag = inputs["fuse_vae_embedding_in_latents"]

        # Independent noisy action / timestep for the base regime.
        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        # Video branch = single CLEAN first frame at timestep 0 (matches infer_action).
        timestep_video = torch.zeros(
            (batch_size,), dtype=first_frame_latents.dtype, device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,  # action_conditioned=false -> ignored; matches base inference
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        # IMPORTANT: explicitly use FastWAM's FIRST-FRAME mask, NOT
        # self._build_mot_attention_mask (which MRO-resolves to FastWAMJoint's
        # full-video mask). With a single-frame video they coincide, but the
        # explicit call is faithful to base inference and future-proof.
        attention_mask = FastWAM._build_mot_attention_mask(
            self,
            video_seq_len=int(video_pre["tokens"].shape[1]),
            action_seq_len=int(action_pre["tokens"].shape[1]),
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )

        tokens_out = self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )
        # NOTE: tokens_out["video"] is intentionally discarded -> NO video loss.
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        action_loss_per_sample = self._action_loss_per_sample(
            pred_action, target_action, action_is_pad
        )
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        return (action_loss_per_sample * action_weight).mean()

    def _joint_regime_losses(self, inputs: dict):
        """JOINT-regime video + action losses from a PRE-BUILT `inputs` dict.

        Equivalent to FastWAM.training_loss (fastwam.py:448-568) but consuming
        the caller's `inputs` (single VAE encode/step) and returning the two
        DIFFERENTIABLE loss tensors. Because `self` is a FastWAMJoint subclass,
        `self._build_mot_attention_mask` is the FULL-video (joint) mask, so the
        action attends to all video tokens — matching joint inference where the
        action conditions on the jointly-denoised video.
        Returns (loss_video, loss_action_joint) as 0-dim tensors.
        """
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        fuse_flag = inputs["fuse_vae_embedding_in_latents"]
        first_frame_latents = inputs["first_frame_latents"]

        # Noisy video (first latent frame kept clean if fused).
        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
        if first_frame_latents is not None:
            latents[:, :, 0:1] = first_frame_latents

        # Noisy action (joint-conditioned).
        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,  # action_conditioned=false -> ignored
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        # JOINT mask: action attends to the full video latent sequence.
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=int(video_pre["tokens"].shape[1]),
            action_seq_len=int(action_pre["tokens"].shape[1]),
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )

        tokens_out = self.mot(
            embeds_all={"video": video_pre["tokens"], "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": video_pre["freqs"], "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": video_pre["t_mod"], "action": action_pre["t_mod"]},
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        include_initial_video_step = first_frame_latents is None
        if first_frame_latents is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video, target_video=target_video,
            image_is_pad=image_is_pad, include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_per_sample = self._action_loss_per_sample(pred_action, target_action, action_is_pad)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action_joint = (action_loss_per_sample * action_weight).mean()
        return loss_video, loss_action_joint

    def training_loss(self, sample, tiled: bool = False):
        """Dual-regime training: action expert sees BOTH the joint and base
        conditioning regimes every step; video loss is computed exactly once."""
        w_base = float(getattr(self, "action_regime_weight_base", 1.0))
        share_inputs = bool(getattr(self, "train_share_inputs", True))

        if share_inputs:
            # Single VAE encode / context build per step (no redundant encode).
            inputs = self.build_inputs(sample, tiled=tiled)
            loss_video, loss_action_joint = self._joint_regime_losses(inputs)
            loss_action_base = self._base_regime_action_loss(inputs)
            loss_main_total = (
                self.loss_lambda_video * loss_video
                + self.loss_lambda_action * loss_action_joint
            )
        else:
            # Pure zero-drift path: delegate verbatim to the unmodified base
            # FastWAM.training_loss. Because `self` is a FastWAMJoint subclass,
            # self._build_mot_attention_mask is the joint mask, so this computes
            # the JOINT regime (video + joint-conditioned action). Then add a
            # base-regime action term from a second build_inputs.
            loss_main_total, main_dict = FastWAM.training_loss(self, sample, tiled=tiled)
            inputs = self.build_inputs(sample, tiled=tiled)
            loss_action_base = self._base_regime_action_loss(inputs)

        loss_total = loss_main_total + self.loss_lambda_action * w_base * loss_action_base

        if share_inputs:
            loss_dict = {
                "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
                "loss_action_joint": self.loss_lambda_action * float(loss_action_joint.detach().item()),
                "loss_action_base": self.loss_lambda_action * w_base * float(loss_action_base.detach().item()),
            }
        else:
            loss_dict = {
                "loss_video": float(main_dict["loss_video"]),
                "loss_action_joint": float(main_dict["loss_action"]),
                "loss_action_base": self.loss_lambda_action * w_base * float(loss_action_base.detach().item()),
            }
        return loss_total, loss_dict


def create_metric_adaptive_fastwam_joint(
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
):
    adaptive_cfg = _to_plain_dict(adaptive if adaptive is not None else router)
    routing_metric = _maybe_instantiate(adaptive_cfg.get("metric"))
    routing_selector = _maybe_instantiate(adaptive_cfg.get("selector"))
    annotate_outputs = bool(adaptive_cfg.get("annotate_outputs", True))

    # Dual-regime training knobs. NOT threaded through from_wan22_pretrained
    # (the FastWAM base signature has no **kwargs); set as attributes post-build.
    train_cfg = _to_plain_dict(train)
    action_regime_weight_base = float(train_cfg.get("action_regime_weight_base", 1.0))
    train_share_inputs = bool(train_cfg.get("share_inputs", True))

    model = MetricAdaptiveFastWAMJoint.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=(None if proprio_dim is None else int(proprio_dim)),
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=OmegaConf.to_container(video_dit_config, resolve=True)
        if isinstance(video_dit_config, DictConfig)
        else video_dit_config,
        action_dit_config=OmegaConf.to_container(action_dit_config, resolve=True)
        if isinstance(action_dit_config, DictConfig)
        else action_dit_config,
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
    )
    # Set training knobs AFTER construction (do NOT pass into from_wan22_pretrained).
    model.action_regime_weight_base = action_regime_weight_base
    model.train_share_inputs = train_share_inputs
    return model
