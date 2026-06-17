from __future__ import annotations

import inspect
from typing import Any, Mapping, Optional

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from fastwam.routing.metrics import MetricResult, PolicyEntropyMetric, RoutingDecision, ThresholdSelector
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
            else ThresholdSelector(threshold=0.0, low_branch="base", high_branch="idm", mode="ge")
        )
        self.annotate_outputs = bool(annotate_outputs)
        self.last_routing_decision: dict[str, Any] | None = None

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
            method = getattr(FastWAMIDM, method_name)

        branch_kwargs = self._filter_kwargs_for_method(method, kwargs)
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
):
    adaptive_cfg = _to_plain_dict(adaptive if adaptive is not None else router)
    routing_metric = _maybe_instantiate(adaptive_cfg.get("metric"))
    routing_selector = _maybe_instantiate(adaptive_cfg.get("selector"))
    annotate_outputs = bool(adaptive_cfg.get("annotate_outputs", True))

    return MetricAdaptiveFastWAM.from_wan22_pretrained(
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
