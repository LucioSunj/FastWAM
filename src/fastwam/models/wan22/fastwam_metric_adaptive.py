from __future__ import annotations

import inspect
import math
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fastwam.routing.metrics import MetricResult, PolicyEntropyMetric, RoutingDecision, ThresholdSelector
from fastwam.adaptive_gate.training import normalized_dual_regime_action_loss
from fastwam.utils.logging_config import get_logger

from .fastwam import FastWAM
from .fastwam_idm import FastWAMIDM

logger = get_logger(__name__)


_ROUTING_ONLY_KWARGS = {
    "force_branch",
    "return_routing_info",
    "routing_metric_value",
}


class _InheritedRoutingContext:
    def __init__(self, model: "MetricAdaptiveFastWAM", method_name: str, kwargs: dict[str, Any]):
        self.model = model
        self.method_name = method_name
        self.kwargs = kwargs
        self._outputs: dict[str, dict[str, Any]] = {}

    def run_branch(self, branch_name: str) -> dict[str, Any]:
        if branch_name not in self._outputs:
            self._outputs[branch_name] = self.model._call_inherited_branch(
                branch_name=branch_name,
                method_name=self.method_name,
                kwargs=self.kwargs,
            )
        return self._outputs[branch_name]

    def run_branch_with_overrides(self, branch_name: str, **overrides: Any) -> dict[str, Any]:
        if self._overrides_match_current_kwargs(overrides):
            return self.run_branch(branch_name)
        branch_kwargs = dict(self.kwargs)
        branch_kwargs.update(overrides)
        return self.model._call_inherited_branch(
            branch_name=branch_name,
            method_name=self.method_name,
            kwargs=branch_kwargs,
        )

    def _overrides_match_current_kwargs(self, overrides: dict[str, Any]) -> bool:
        for key, value in overrides.items():
            current_value = self.kwargs.get(key)
            if torch.is_tensor(current_value) or torch.is_tensor(value):
                return False
            if current_value != value:
                return False
        return True


class MetricAdaptiveFastWAM(FastWAMIDM):
    """FastWAMIDM subclass that can fall back to base FastWAM inference.

    This class intentionally inherits from the IDM variant instead of wrapping
    two independent model instances. The `base` branch explicitly calls
    `FastWAM` methods on this instance, while the `idm` branch calls
    `FastWAMIDM` methods. Editing this subclass changes only the adaptive
    variant and leaves the original model classes untouched.
    """

    adaptive_regimes = ("uncond", "idm")
    adaptive_backbone_kind = "idm"

    def __init__(
        self,
        *args,
        routing_metric=None,
        routing_selector=None,
        annotate_outputs: bool = True,
        allow_internal_routing: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.configure_routing(
            routing_metric=routing_metric,
            routing_selector=routing_selector,
            annotate_outputs=annotate_outputs,
            allow_internal_routing=allow_internal_routing,
        )

    @classmethod
    def from_wan22_pretrained(
        cls,
        *,
        routing_metric=None,
        routing_selector=None,
        annotate_outputs: bool = True,
        allow_internal_routing: bool = False,
        **kwargs,
    ):
        model = super().from_wan22_pretrained(**kwargs)
        model.configure_routing(
            routing_metric=routing_metric,
            routing_selector=routing_selector,
            annotate_outputs=annotate_outputs,
            allow_internal_routing=allow_internal_routing,
        )
        return model

    def configure_routing(
        self,
        routing_metric=None,
        routing_selector=None,
        annotate_outputs: bool = True,
        allow_internal_routing: bool = False,
    ) -> None:
        self.routing_metric = routing_metric if routing_metric is not None else PolicyEntropyMetric(probe_branch="base")
        self.routing_selector = (
            routing_selector
            if routing_selector is not None
            else ThresholdSelector(threshold=0.0, low_branch="base", high_branch="idm", mode="ge")
        )
        self.annotate_outputs = bool(annotate_outputs)
        self.allow_internal_routing = bool(allow_internal_routing)
        self.last_routing_decision: dict[str, Any] | None = None

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int | None = None,
        first_frame_latents: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        video_inference_steps: int | None = None,
        action_inference_steps: int | None = None,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        routing_metric_value: float | None = None,
        force_branch: str | None = None,
        return_routing_info: bool = False,
        return_video_latents: bool = False,
        idm_control: Any = "valid_idm",
        shuffled_future_donor: Any = None,
        expected_donor_metadata: Mapping[str, Any] | None = None,
        init_noise: Optional[torch.Tensor] = None,
        velocity_hook: Any = None,
        return_init_noise: bool = False,
    ) -> dict[str, Any]:
        kwargs = locals()
        kwargs.pop("self")
        return self._route_inherited(method_name="infer_action", kwargs=kwargs)

    def infer_action_with_grad(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        num_video_frames: int | None = None,
        first_frame_latents: Optional[torch.Tensor] = None,
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
        init_noise: Optional[torch.Tensor] = None,
        velocity_hook: Any = None,
        return_init_noise: bool = False,
    ) -> dict[str, Any]:
        """Gradient-carrying forced-branch action dispatch (stage-2 W17).

        Routes exactly like ``infer_action`` but through the UNDECORATED
        gradient-carrying base entry (``FastWAM.infer_action_with_grad``), so
        ``result["action"]`` keeps its autograd graph (``[T, D]`` on the model
        device/dtype; see the base method for the full contract).

        Fail-closed scope: ``force_branch="base"`` (the UNCOND regime) is
        REQUIRED. The IDM branch is refused because ``FastWAMIDM`` defines no
        grad-carrying override — plain ``getattr`` resolution would silently
        inherit the BASE method and run the wrong conditioning regime, and the
        IDM future-video rollout has no gradient path in W17 scope anyway.
        Internal routing is likewise refused: a gradient rollout is a training
        construct and must state its regime explicitly. The deliberately
        narrow signature omits every IDM-only control kwarg
        (``idm_control``/donor/video-step overrides), so passing one raises
        ``TypeError`` instead of being silently dropped.
        """
        if force_branch is None:
            raise ValueError(
                "infer_action_with_grad requires an explicit force_branch='base': "
                "the gradient-carrying rollout is a training construct and never "
                "routes internally."
            )
        if str(force_branch) != "base":
            raise ValueError(
                "infer_action_with_grad supports only force_branch='base' (the "
                f"UNCOND regime); got {force_branch!r}. The IDM branch has no "
                "gradient-carrying implementation (W17 scope: action path only)."
            )
        kwargs = locals()
        kwargs.pop("self")
        return self._route_inherited(method_name="infer_action_with_grad", kwargs=kwargs)

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
        video_inference_steps: int | None = None,
        action_inference_steps: int | None = None,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = False,
        routing_metric_value: float | None = None,
        force_branch: str | None = None,
        return_routing_info: bool = False,
        return_video_latents: bool = False,
        idm_control: Any = "valid_idm",
        shuffled_future_donor: Any = None,
        expected_donor_metadata: Mapping[str, Any] | None = None,
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
        video_inference_steps: int | None = None,
        action_inference_steps: int | None = None,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        routing_metric_value: float | None = None,
        force_branch: str | None = None,
        return_routing_info: bool = False,
        idm_control: Any = "valid_idm",
        shuffled_future_donor: Any = None,
        expected_donor_metadata: Mapping[str, Any] | None = None,
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
        elif self.allow_internal_routing:
            metric_result = self.routing_metric(context)
            decision = self.routing_selector.select(metric_result)
            selected_branch = decision.selected_branch
            self._validate_branch_name(selected_branch)
        else:
            raise ValueError(
                "Adaptive inference requires an explicit `force_branch` from the external "
                "gate. Set allow_internal_routing=True only for the legacy entropy-router "
                "ablation, which runs extra probe inference and is not compute-saving."
            )

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
            method = getattr(FastWAMIDM, method_name)

        branch_kwargs = self._filter_kwargs_for_method(method, kwargs)
        # Stage-2 W8: the sampler interface (noise injection / velocity hook)
        # is implemented on the base/UNCOND action path only. The signature
        # filter above would silently DROP these kwargs for a branch method
        # that lacks them — fail closed instead, because a caller who injected
        # a noise or a guidance hook must never get an unguided rollout back.
        # Identity checks only: a tensor value must never hit `in (None, False)`
        # membership (tensor __eq__ would raise on multi-element truthiness).
        dropped_sampler_kwargs = [
            key
            for key in ("init_noise", "velocity_hook", "return_init_noise")
            if (value := kwargs.get(key)) is not None
            and value is not False
            and key not in branch_kwargs
        ]
        if dropped_sampler_kwargs:
            raise ValueError(
                f"Branch {branch_name!r} ({method.__qualname__}) does not implement "
                f"sampler-interface kwargs {dropped_sampler_kwargs}; refusing to drop "
                "them silently. init_noise/velocity_hook/return_init_noise are "
                "base/UNCOND-only."
            )
        logger.debug("Adaptive FastWAM call `%s` through inherited branch `%s`.", method_name, branch_name)
        return method(self, **branch_kwargs)

    @staticmethod
    def _validate_branch_name(branch_name: str) -> None:
        if branch_name not in {"base", "idm"}:
            raise ValueError(f"Unknown branch `{branch_name}`. Expected one of: ['base', 'idm'].")

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
    # The inference router picks base or idm per call, but both share one set
    # of weights. To keep the shared action expert in-distribution for BOTH
    # branches, every training step computes the action loss under BOTH
    # conditioning regimes while the video-denoising loss is computed ONCE.
    #
    # Knobs are set by create_metric_adaptive_fastwam (read via getattr so the
    # model still runs if wiring is omitted):
    #   self.action_regime_weight_uncond : float = 1.0
    #   self.train_share_inputs        : bool  = True   # single VAE encode/step

    def _action_loss_per_sample(
        self,
        pred_action: torch.Tensor,
        target_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Per-sample action loss, identical reduction to FastWAM/FastWAMIDM
        (fastwam.py:550-556, fastwam_idm.py:209-215). Returns [B] (pre-weight)."""
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

    def _idm_regime_losses(self, inputs: dict):
        """IDM-regime video + action losses from a PRE-BUILT `inputs` dict.

        Line-for-line equivalent of FastWAMIDM.training_loss (fastwam_idm.py:69-220)
        EXCEPT it consumes the caller's `inputs` (single VAE encode/step) and returns
        the two DIFFERENTIABLE loss tensors. Keep in sync with the parent if it
        ever changes (or set share_inputs=false to delegate to it verbatim).
        Returns (loss_video, loss_action_idm) as 0-dim tensors.
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

        # Branch A: noisy video.
        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=input_latents.dtype,
        )
        latents_noisy = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)
        if first_frame_latents is not None:
            latents_noisy[:, :, 0:1] = first_frame_latents

        # Branch B: noisy action (idm-conditioned).
        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size, device=self.device, dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        # Branch C: teacher-forcing cond-video, per-sample noised w.p. video_cond_noise_prob.
        cond_noise_mask = torch.rand((batch_size,), device=self.device) < float(self.video_cond_noise_prob)
        timestep_video_cond = torch.zeros_like(timestep_video, dtype=input_latents.dtype, device=self.device)
        latents_cond = input_latents
        if bool(cond_noise_mask.any()):
            timestep_video_cond_sampled = self.train_video_scheduler.sample_training_t(
                batch_size=batch_size, device=self.device, dtype=input_latents.dtype,
            )
            timestep_video_cond = torch.where(cond_noise_mask, timestep_video_cond_sampled, timestep_video_cond)
            noise_video_cond = torch.randn_like(input_latents)
            latents_cond_noisy = self.train_video_scheduler.add_noise(
                input_latents, noise_video_cond, timestep_video_cond_sampled
            )
            cond_noise_selector = cond_noise_mask.view(batch_size, 1, 1, 1, 1)
            latents_cond = torch.where(cond_noise_selector, latents_cond_noisy, input_latents)
        if first_frame_latents is not None:
            latents_cond = latents_cond.clone()
            latents_cond[:, :, 0:1] = first_frame_latents

        video_pre_noisy = self.video_expert.pre_dit(
            x=latents_noisy, timestep=timestep_video, context=context,
            context_mask=context_mask, action=None, fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_pre_cond = self.video_expert.pre_dit(
            x=latents_cond, timestep=timestep_video_cond, context=context,
            context_mask=context_mask, action=None, fuse_vae_embedding_in_latents=fuse_flag,
        )
        if video_pre_noisy["t_mod"].ndim != 4 or video_pre_cond["t_mod"].ndim != 4:
            raise ValueError(
                "Teacher-forcing requires token-wise `t_mod`; "
                "ensure `seperated_timestep=true` and `fuse_vae_embedding_in_latents=true`."
            )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action, timestep=timestep_action,
            context=context, context_mask=context_mask,
        )

        noisy_video_seq_len = int(video_pre_noisy["tokens"].shape[1])
        cond_video_seq_len = int(video_pre_cond["tokens"].shape[1])
        noisy_tpf = int(video_pre_noisy["meta"]["tokens_per_frame"])
        cond_tpf = int(video_pre_cond["meta"]["tokens_per_frame"])

        merged_video_tokens = torch.cat([video_pre_noisy["tokens"], video_pre_cond["tokens"]], dim=1)
        merged_video_freqs = torch.cat([video_pre_noisy["freqs"], video_pre_cond["freqs"]], dim=0)
        merged_video_t_mod = torch.cat([video_pre_noisy["t_mod"], video_pre_cond["t_mod"]], dim=1)
        merged_video_context_mask = torch.cat(
            [video_pre_noisy["context_mask"], video_pre_cond["context_mask"]], dim=1
        )

        attention_mask = self._build_teacher_forcing_attention_mask(
            noisy_video_seq_len=noisy_video_seq_len,
            cond_video_seq_len=cond_video_seq_len,
            action_seq_len=action_pre["tokens"].shape[1],
            noisy_video_tokens_per_frame=noisy_tpf,
            cond_video_tokens_per_frame=cond_tpf,
            device=merged_video_tokens.device,
        )

        tokens_out = self.mot(
            embeds_all={"video": merged_video_tokens, "action": action_pre["tokens"]},
            attention_mask=attention_mask,
            freqs_all={"video": merged_video_freqs, "action": action_pre["freqs"]},
            context_all={
                "video": {"context": video_pre_noisy["context"], "mask": merged_video_context_mask},
                "action": {"context": action_pre["context"], "mask": action_pre["context_mask"]},
            },
            t_mod_all={"video": merged_video_t_mod, "action": action_pre["t_mod"]},
        )

        # Video loss ONLY from the noisy-video half (no double count).
        pred_video = self.video_expert.post_dit(
            tokens_out["video"][:, :noisy_video_seq_len], video_pre_noisy
        )
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
        loss_action_idm = (action_loss_per_sample * action_weight).mean()
        return loss_video, loss_action_idm

    def training_loss(self, sample, tiled: bool = False):
        """Dual-regime training: action expert sees BOTH the idm and base
        conditioning regimes every step; video loss is computed exactly once."""
        w_base = float(
            getattr(
                self,
                "action_regime_weight_uncond",
                getattr(self, "action_regime_weight_base", 1.0),
            )
        )
        share_inputs = bool(getattr(self, "train_share_inputs", True))

        if share_inputs:
            # Single VAE encode / context build per step (no redundant encode).
            inputs = self.build_inputs(sample, tiled=tiled)
            loss_video, loss_action_idm = self._idm_regime_losses(inputs)
            loss_action_base = self._base_regime_action_loss(inputs)
        else:
            idm_inputs = self.build_inputs(sample, tiled=tiled)
            loss_video, loss_action_idm = self._idm_regime_losses(idm_inputs)
            base_inputs = self.build_inputs(sample, tiled=tiled)
            loss_action_base = self._base_regime_action_loss(base_inputs)

        combined_action, idm_raw_contribution, base_raw_contribution = (
            normalized_dual_regime_action_loss(loss_action_idm, loss_action_base, w_base)
        )
        loss_total = (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_action * combined_action
        )
        idm_contribution = self.loss_lambda_action * idm_raw_contribution
        base_contribution = self.loss_lambda_action * base_raw_contribution
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action_idm": float(idm_contribution.detach().item()),
            "loss_action_uncond": float(base_contribution.detach().item()),
            "loss_action_combined": self.loss_lambda_action * float(combined_action.detach().item()),
            "action_regime_weight_uncond": w_base,
        }
        return loss_total, loss_dict


def create_metric_adaptive_fastwam(
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

    # Dual-regime training knobs. NOT threaded through from_wan22_pretrained
    # (the FastWAM base signature has no **kwargs); set as attributes post-build.
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
    train_share_inputs = bool(train_cfg.get("share_inputs", True))

    model = MetricAdaptiveFastWAM.from_wan22_pretrained(
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
        allow_internal_routing=allow_internal_routing,
    )
    # Set training knobs AFTER construction (do NOT pass into from_wan22_pretrained).
    model.action_regime_weight_uncond = action_regime_weight_base
    model.train_share_inputs = train_share_inputs
    model.checkpoint_task = checkpoint_task
    return model


def _to_plain_dict(value) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Expected mapping-like config, got {type(value)}.")


def _maybe_instantiate(value):
    if value is None:
        return None
    if isinstance(value, DictConfig):
        return instantiate(value)
    if isinstance(value, Mapping) and "_target_" in value:
        return instantiate(OmegaConf.create(value))
    return value


def _mapping_value(value, key: str, default):
    mapping = _to_plain_dict(value)
    return mapping.get(key, default)


def _scheduler_value(value, key: str, default):
    mapping = _to_plain_dict(value)
    return mapping.get(key, default)
