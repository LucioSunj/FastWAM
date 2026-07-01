"""WAMModeAdapter — mode-switched inference wrapper around a FROZEN fastwam model.

One interface over BOTH dual-regime backbones (decision #2):
- backbone_kind="joint" -> a MetricAdaptiveFastWAMJoint checkpoint (routes base/joint)
- backbone_kind="idm"   -> a MetricAdaptiveFastWAM    checkpoint (routes base/idm)

The same `act(...)` is used in training and inference. The adapter never trains
the WAM (it must arrive frozen); it only selects how much prediction compute to
spend per call via `mode`.

`act(...) -> {action_chunk, world_feat, cost, aux}`:
- action_chunk : [action_horizon, action_dim] tensor (cpu float32, as fastwam returns)
- world_feat   : the cheap current-context latent (gate input), computed in ALL
                 modes BEFORE the mode-specific compute
- cost         : relative FLOPs of the chosen mode, cost(FULL)=1
- aux          : {mode, branch, video_inference_steps, action_inference_steps, routing}

No-leakage: the future the action conditions on is the model's OWN self-generated
latent (the dual-regime model denoises from noise; only frame-0 is clamped to the
encoded current image). LATENT/FULL never decode pixels (infer_action returns the
action only). `force_branch` also bypasses the model's internal PolicyEntropy
probe so the gate is the sole decision-maker.
"""
from __future__ import annotations

import inspect
from typing import Any, Optional

import torch

from .cost import default_cost_table, load_cost_table
from .modes import coerce_mode, mode_to_branch_step_counts


class WAMModeAdapter:
    def __init__(
        self,
        model,
        *,
        backbone_kind: str,
        num_video_frames: int,
        action_horizon: int,
        # TODO(make-variable, decision #3): k_lo/k_hi/action_steps are fixed
        #   defaults for now; expose them as runtime/config variables later so the
        #   gate (or a schedule) can pick the LATENT/FULL depth dynamically.
        k_lo: int = 4,
        k_hi: int = 20,
        action_steps: Optional[int] = None,
        cost_table: Optional[dict[str, float]] = None,
        cost_table_path: Optional[str] = None,
        cost_source: Optional[str] = None,
        default_seed: Optional[int] = None,
    ):
        if not hasattr(model, "infer_action"):
            raise TypeError("`model` must expose `infer_action`.")
        sig = inspect.signature(model.infer_action)
        if "force_branch" not in sig.parameters:
            raise TypeError(
                "WAMModeAdapter requires a routing-capable model (MetricAdaptiveFastWAM or "
                "MetricAdaptiveFastWAMJoint) whose `infer_action` accepts `force_branch`. "
                "Pass a dual-regime checkpoint loaded into one of those classes."
            )
        self.model = model
        self.backbone_kind = str(backbone_kind)
        self.num_video_frames = int(num_video_frames)
        self.action_horizon = int(action_horizon)
        self.k_lo = int(k_lo)
        self.k_hi = int(k_hi)
        # SKIP drives only the ACTION denoising loop (video is not denoised); use
        # the full action schedule by default so SKIP == current fastwam behavior.
        self.action_steps = int(action_steps) if action_steps is not None else int(k_hi)
        self.default_seed = default_seed

        if cost_table is not None:
            self.cost_table = {k: float(v) for k, v in cost_table.items()}
        else:
            loaded = load_cost_table(cost_table_path, source=cost_source)
            self.cost_table = loaded if loaded is not None else default_cost_table(self.k_lo, self.k_hi)

    # ----- gate input ---------------------------------------------------- #
    @property
    def world_feat_dim(self) -> int:
        """Channel dim of the current-context latent (= gate world_feat dim)."""
        return int(self.model.vae.model.z_dim)

    @torch.no_grad()
    def encode_world_feat(self, input_image: torch.Tensor) -> torch.Tensor:
        """Cheap current-context latent, pooled to a fixed [z_dim] vector.

        This is the SINGLE encoder forward, computed before any mode-specific
        compute, and is the gate's input in every mode.
        """
        latent = self.model._encode_input_image_latents_tensor(input_image)  # [1, C, t, h, w]
        feat = latent.mean(dim=tuple(range(2, latent.ndim)))  # pool over (t,h,w) -> [1, C]
        return feat.squeeze(0).detach().float()

    # ----- mode-switched action ------------------------------------------ #
    @torch.no_grad()
    def act(
        self,
        *,
        input_image: torch.Tensor,
        mode,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        prompt: Optional[str] = None,
        action_horizon: Optional[int] = None,
        num_video_frames: Optional[int] = None,
        world_feat: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        mode = coerce_mode(mode)
        # world_feat is computed in ALL modes, BEFORE the mode-specific compute.
        if world_feat is None:
            world_feat = self.encode_world_feat(input_image)

        branch, video_steps, action_steps = mode_to_branch_step_counts(
            mode,
            backbone_kind=self.backbone_kind,
            k_lo=self.k_lo,
            k_hi=self.k_hi,
            action_steps=self.action_steps,
        )
        # `force_branch` routes to FastWAM (base) / FastWAMJoint|IDM (future) on the
        # SAME frozen weights, and bypasses the model's internal routing metric.
        # `_filter_kwargs_for_method` drops `num_video_frames` for the base branch
        # (FastWAM.infer_action has no such arg) and passes it for joint/idm.
        infer_kwargs = dict(
            prompt=prompt,
            input_image=input_image,
            action_horizon=int(action_horizon or self.action_horizon),
            num_video_frames=int(num_video_frames or self.num_video_frames),
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            num_inference_steps=int(action_steps),
            seed=seed if seed is not None else self.default_seed,
            force_branch=branch,
            return_routing_info=True,
        )
        if video_steps is not None and int(video_steps) != int(action_steps):
            infer_kwargs["video_inference_steps"] = int(video_steps)
            infer_kwargs["action_inference_steps"] = int(action_steps)

        out = self.model.infer_action(**infer_kwargs)
        return {
            "action_chunk": out["action"],
            "world_feat": world_feat,
            "cost": float(self.cost_table[mode.value]),
            "aux": {
                "mode": mode.value,
                "branch": branch,
                "video_inference_steps": None if video_steps is None else int(video_steps),
                "action_inference_steps": int(action_steps),
                "num_inference_steps": int(action_steps),
                "routing": out.get("_routing"),
            },
        }
