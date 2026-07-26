"""Binary mode adapter around one frozen dual-regime IDM FastWAM."""
from __future__ import annotations

import inspect
import math
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F

from .cost import default_cost_table, load_cost_table, validate_cost_table
from .controls import IDMControl, ShuffledFutureDonor, coerce_idm_control
from .modes import WAMMode, coerce_mode, mode_to_branch_steps
from .provenance import (
    dual_regime_schedule_fingerprint,
    inference_solver_contract,
    inference_solver_fingerprint,
)


WORLD_FEAT_LAYOUT = "spatial_2x2_plus_channel_std_v1"


@dataclass(frozen=True)
class EncodedWorldState:
    """One VAE encoding reused by both gate features and WAM inference."""

    world_feat: torch.Tensor
    first_frame_latents: torch.Tensor
    layout: str = WORLD_FEAT_LAYOUT
    image_shape: Optional[tuple[int, int]] = None


class WAMModeAdapter:
    """Choose between reactive ``UNCOND`` and complete future-conditioned ``IDM``."""

    def __init__(
        self,
        model,
        *,
        backbone_kind: str,
        task: Optional[str] = None,
        num_video_frames: int,
        generation_horizon: int,
        inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        context_len: int = 128,
        dataset_stats_fingerprint: Optional[str] = None,
        cost_table: Optional[dict[str, float]] = None,
        cost_table_path: Optional[str] = None,
        cost_source: Optional[str] = None,
        default_seed: Optional[int] = None,
        allow_legacy_checkpoint: bool = False,
        allow_unloaded_model: bool = False,
    ):
        if str(backbone_kind) != "idm":
            raise ValueError(
                "The adaptive gate supports only the dual-regime IDM backbone; "
                f"got backbone_kind={backbone_kind!r}."
            )
        if not hasattr(model, "infer_action"):
            raise TypeError("`model` must expose `infer_action`.")
        sig = inspect.signature(model.infer_action)
        required = {"force_branch", "first_frame_latents"}
        missing = required.difference(sig.parameters)
        if missing:
            raise TypeError(
                "WAMModeAdapter requires a current dual-regime IDM model whose "
                f"`infer_action` accepts {sorted(required)}; missing={sorted(missing)}."
            )

        self.model = model
        self.task = None if task is None else str(task)
        self.backbone_kind = "idm"
        self.num_video_frames = int(num_video_frames)
        self.generation_horizon = int(generation_horizon)
        self.inference_steps = int(inference_steps)
        self.sigma_shift = None if sigma_shift is None else float(sigma_shift)
        self.context_len = int(context_len)
        self.dataset_stats_fingerprint = dataset_stats_fingerprint
        self.cost_source = None if cost_source is None else str(cost_source)
        self.default_seed = default_seed
        if (
            self.num_video_frames <= 0
            or self.generation_horizon <= 0
            or self.inference_steps <= 0
            or self.context_len <= 0
            or (self.sigma_shift is not None and self.sigma_shift <= 0)
        ):
            raise ValueError(
                "num_video_frames, generation_horizon and inference_steps must be positive; "
                f"got {self.num_video_frames}, {self.generation_horizon}, {self.inference_steps}."
            )

        self.solver_contract = inference_solver_contract(
            self.model,
            video_inference_steps=self.inference_steps,
            action_inference_steps=self.inference_steps,
            sigma_shift=self.sigma_shift,
        )
        self.solver_fingerprint = inference_solver_fingerprint(self.solver_contract)

        self._validate_checkpoint_provenance(
            allow_legacy_checkpoint=allow_legacy_checkpoint,
            allow_unloaded_model=allow_unloaded_model,
        )
        if cost_table is not None and cost_table_path is not None:
            raise ValueError("Pass either cost_table or cost_table_path, not both.")
        if cost_source is not None and cost_table_path is None:
            raise ValueError("cost_source requires cost_table_path.")
        if cost_table is not None:
            self.cost_table = validate_cost_table(cost_table)
        else:
            loaded, cost_meta = load_cost_table(
                cost_table_path, source=cost_source, return_meta=True
            )
            self.cost_table = loaded if loaded is not None else default_cost_table(self.inference_steps)
            self._cost_meta = cost_meta
            if loaded is not None:
                self._validate_cost_profile_meta(cost_meta)
        if cost_table is not None:
            self._cost_meta = None

    def _validate_cost_profile_meta(self, meta: Optional[dict[str, Any]]) -> None:
        if not meta:
            raise ValueError("Profiled cost YAML must contain a non-empty `meta` block.")
        if self.task is None:
            raise ValueError("`task` is required when loading a profiled cost table.")
        expected = {
            "task": self.task,
            "backbone_kind": "idm",
            "ckpt_fingerprint": getattr(self.model, "_loaded_checkpoint_fingerprint", None),
            "inference_steps": self.inference_steps,
            "solver_fingerprint": self.solver_fingerprint,
            "num_video_frames": self.num_video_frames,
            "action_horizon": self.generation_horizon,
            "context_len": self.context_len,
            "model_dtype": str(getattr(self.model, "torch_dtype", None)),
            "proprio_dim": getattr(self.model, "proprio_dim", None),
        }
        if expected["ckpt_fingerprint"] is None:
            raise ValueError(
                "A profiled cost table requires a loaded checkpoint with a verifiable fingerprint."
            )
        mismatches = {
            key: (meta.get(key), value)
            for key, value in expected.items()
            if meta.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "Cost profile metadata does not match this adapter "
                f"(actual, expected): {mismatches}."
            )
        if self.cost_source in {"latency", "latency_ms"}:
            expected_device = self._device_name()
            if meta.get("device_name") != expected_device:
                raise ValueError(
                    "Latency profile hardware does not match this model device: "
                    f"profile={meta.get('device_name')!r}, current={expected_device!r}."
                )

    def _device_name(self) -> str:
        device = torch.device(getattr(self.model, "device", "cpu"))
        if device.type == "cuda" and torch.cuda.is_available():
            index = device.index if device.index is not None else torch.cuda.current_device()
            return torch.cuda.get_device_name(index)
        return device.type

    def _validate_cost_resolution(self, input_image: torch.Tensor) -> None:
        if self._cost_meta is None:
            return
        if input_image.ndim not in (3, 4):
            raise ValueError(f"input_image must be 3D/4D, got {tuple(input_image.shape)}")
        height, width = map(int, input_image.shape[-2:])
        expected = (self._cost_meta.get("height"), self._cost_meta.get("width"))
        if expected[0] is None or expected[1] is None:
            raise ValueError("Cost profile meta must include height and width.")
        if (height, width) != tuple(map(int, expected)):
            raise ValueError(
                f"Cost profile resolution {expected} does not match input {(height, width)}."
            )

    def _validate_checkpoint_provenance(
        self,
        *,
        allow_legacy_checkpoint: bool,
        allow_unloaded_model: bool,
    ) -> None:
        live_regimes = tuple(getattr(self.model, "adaptive_regimes", ()))
        live_kind = getattr(self.model, "adaptive_backbone_kind", None)
        if live_regimes != ("uncond", "idm") or live_kind != "idm":
            raise ValueError(
                "Live model is not the dual-regime IDM implementation: "
                f"adaptive_regimes={live_regimes!r}, adaptive_backbone_kind={live_kind!r}."
            )
        provenance = getattr(self.model, "_loaded_checkpoint_provenance", "not_loaded")
        if provenance == "not_loaded":
            if allow_unloaded_model:
                return
            raise ValueError(
                "WAMModeAdapter requires a loaded dual-regime checkpoint. Pass "
                "allow_unloaded_model=True only for tests/development."
            )
        if provenance is None:
            message = (
                "The loaded checkpoint predates FastWAM provenance metadata, so dual-regime "
                "IDM training cannot be verified. Pass allow_legacy_checkpoint=True only after "
                "manually confirming the checkpoint."
            )
            if not allow_legacy_checkpoint:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return
        schema_version = provenance.get("schema_version")
        checkpoint_id = provenance.get("checkpoint_id")
        if schema_version != 2 or not isinstance(checkpoint_id, str) or not checkpoint_id:
            message = (
                "Checkpoint provenance is incomplete or uses an unsupported schema: "
                f"schema_version={schema_version!r}, checkpoint_id={checkpoint_id!r}. "
                "Treat it as legacy and verify it manually."
            )
            if not allow_legacy_checkpoint:
                raise ValueError(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return
        regimes = tuple(provenance.get("adaptive_regimes", ()))
        if regimes != ("uncond", "idm"):
            raise ValueError(
                "Checkpoint is not a dual-regime UNCOND+IDM checkpoint: "
                f"adaptive_regimes={regimes!r}."
            )
        dual_steps = provenance.get("dual_regime_optimizer_steps")
        if (
            not isinstance(dual_steps, int)
            or isinstance(dual_steps, bool)
            or dual_steps <= 0
        ):
            raise ValueError(
                "Checkpoint is an untrained S0 artifact or does not prove any "
                "dual-regime optimizer update: "
                f"dual_regime_optimizer_steps={dual_steps!r}."
            )
        contract = provenance.get("dual_regime_training_contract")
        try:
            expected_schedule_fingerprint = dual_regime_schedule_fingerprint(contract)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Checkpoint has no valid deterministic dual-regime schedule contract."
            ) from exc
        if provenance.get("schedule_fingerprint") != expected_schedule_fingerprint:
            raise ValueError(
                "Checkpoint schedule_fingerprint does not match its training contract."
            )
        if dual_steps > int(contract["total_optimizer_steps"]):
            raise ValueError(
                "Checkpoint dual_regime_optimizer_steps exceeds its training contract."
            )
        if provenance.get("initialization_type") != "standalone_idm":
            raise ValueError(
                "Production adaptive Gate checkpoints must be initialized from the "
                "verified standalone IDM endpoint."
            )
        parent_id = provenance.get("parent_checkpoint_id")
        if not isinstance(parent_id, str) or not parent_id:
            raise ValueError("Checkpoint is missing parent_checkpoint_id lineage.")
        for key in (
            "parent_checkpoint_sha256",
            "parent_config_sha256",
            "parent_dataset_stats_sha256",
        ):
            value = provenance.get(key)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"Checkpoint has invalid {key} lineage.")
        if provenance["parent_dataset_stats_sha256"] != provenance.get(
            "dataset_stats_fingerprint"
        ):
            raise ValueError(
                "Parent and shared checkpoint dataset-stats lineage must match."
            )
        regime_weight = provenance.get("action_regime_weight_uncond")
        regime_weight_value = None if regime_weight is None else float(regime_weight)
        if (
            regime_weight_value is None
            or not math.isfinite(regime_weight_value)
            or regime_weight_value <= 0.0
        ):
            raise ValueError(
                "Checkpoint does not prove that the UNCOND regime was trained with positive "
                f"weight: action_regime_weight_uncond={regime_weight!r}."
            )
        if self.task is not None and provenance.get("task") != self.task:
            raise ValueError(
                "Checkpoint task does not match the adaptive gate task: "
                f"checkpoint={provenance.get('task')!r}, gate={self.task!r}."
            )
        checkpoint_stats = provenance.get("dataset_stats_fingerprint")
        if checkpoint_stats is None:
            if not allow_legacy_checkpoint:
                raise ValueError(
                    "Checkpoint has no dataset_stats_fingerprint; use an explicitly verified "
                    "legacy escape hatch or regenerate the checkpoint."
                )
        elif (
            self.dataset_stats_fingerprint is not None
            and checkpoint_stats != self.dataset_stats_fingerprint
        ):
            raise ValueError(
                "Dataset-stats fingerprint does not match the checkpoint: "
                f"checkpoint={checkpoint_stats}, provided={self.dataset_stats_fingerprint}."
            )

    @property
    def world_feat_dim(self) -> int:
        """Coarse 2x2 layout (4C) plus channel standard deviation (C)."""
        return 5 * int(self.model.vae.model.z_dim)

    @torch.no_grad()
    def encode_world_state(self, input_image: torch.Tensor) -> EncodedWorldState:
        """Encode the current image once and construct a fixed spatial feature."""
        dtype = getattr(self.model, "torch_dtype", input_image.dtype)
        device = getattr(self.model, "device", input_image.device)
        image = input_image.to(device=device, dtype=dtype)
        latent = self.model._encode_input_image_latents_tensor(image)
        if latent.ndim != 5 or latent.shape[0] != 1:
            raise ValueError(
                "first-frame VAE latent must be [1,C,T,H,W], got "
                f"{tuple(latent.shape)}."
            )
        coarse = F.adaptive_avg_pool3d(latent.float(), (1, 2, 2)).flatten(1)
        spread = latent.float().std(dim=(2, 3, 4), unbiased=False)
        world_feat = torch.cat([coarse, spread], dim=-1).squeeze(0).detach()
        if world_feat.numel() != self.world_feat_dim:
            raise RuntimeError(
                f"world feature dim mismatch: expected {self.world_feat_dim}, got {world_feat.numel()}."
            )
        return EncodedWorldState(
            world_feat=world_feat,
            first_frame_latents=latent.detach(),
            image_shape=tuple(map(int, input_image.shape[-2:])),
        )

    @torch.no_grad()
    def encode_world_feat(self, input_image: torch.Tensor) -> torch.Tensor:
        """Compatibility helper; prefer ``encode_world_state`` to reuse the latent."""
        return self.encode_world_state(input_image).world_feat

    def _validate_act_inputs(
        self,
        *,
        input_image: torch.Tensor,
        proprio: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        generation_horizon: Optional[int],
        num_video_frames: Optional[int],
        encoded_state: Optional[EncodedWorldState],
    ) -> tuple[int, int, EncodedWorldState]:
        """Shared ``act()``/``act_with_grad()`` input validation (stage-2 W17).

        The statements below are the verbatim pre-W17 validation prefix of
        ``act()``, moved into one helper so the gradient-carrying entry point
        cannot drift from the production checks. Pure validation plus the
        (no-grad) world-state encode — no rollout numerics live here.

        Returns ``(requested_horizon, requested_frames, encoded_state)``.
        """
        self._validate_cost_resolution(input_image)
        expected_proprio_dim = getattr(self.model, "proprio_dim", None)
        if expected_proprio_dim is not None:
            if proprio is None:
                raise ValueError(
                    f"proprio is required because model.proprio_dim={expected_proprio_dim}."
                )
            if int(proprio.shape[-1]) != int(expected_proprio_dim):
                raise ValueError(
                    f"proprio last dim must be {expected_proprio_dim}, got {proprio.shape[-1]}."
                )
        if context is not None:
            if context.ndim not in (2, 3) or int(context.shape[-2]) != self.context_len:
                raise ValueError(
                    f"context length must be {self.context_len}, got {tuple(context.shape)}."
                )
            if context_mask is None or int(context_mask.shape[-1]) != self.context_len:
                raise ValueError(
                    f"context_mask length must be {self.context_len}."
                )
        requested_horizon = (
            self.generation_horizon if generation_horizon is None else int(generation_horizon)
        )
        requested_frames = (
            self.num_video_frames if num_video_frames is None else int(num_video_frames)
        )
        if requested_horizon <= 0 or requested_frames <= 0:
            raise ValueError(
                "generation_horizon and num_video_frames must be positive, got "
                f"{requested_horizon}, {requested_frames}."
            )
        if self._cost_meta is not None:
            if requested_horizon != self.generation_horizon:
                raise ValueError(
                    "generation_horizon override is incompatible with the profiled cost: "
                    f"{requested_horizon} != {self.generation_horizon}."
                )
            if requested_frames != self.num_video_frames:
                raise ValueError(
                    "num_video_frames override is incompatible with the profiled cost: "
                    f"{requested_frames} != {self.num_video_frames}."
                )
        if encoded_state is None:
            encoded_state = self.encode_world_state(input_image)
        if encoded_state.layout != WORLD_FEAT_LAYOUT:
            raise ValueError(
                f"encoded state layout {encoded_state.layout!r} does not match {WORLD_FEAT_LAYOUT!r}."
            )
        image_shape = tuple(map(int, input_image.shape[-2:]))
        if encoded_state.image_shape is not None and encoded_state.image_shape != image_shape:
            raise ValueError(
                f"encoded state image shape {encoded_state.image_shape} does not match "
                f"input image shape {image_shape}."
            )
        if encoded_state.world_feat.numel() != self.world_feat_dim:
            raise ValueError(
                f"encoded world feature must have {self.world_feat_dim} elements, got "
                f"{encoded_state.world_feat.numel()}."
            )
        return requested_horizon, requested_frames, encoded_state

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
        generation_horizon: Optional[int] = None,
        num_video_frames: Optional[int] = None,
        encoded_state: Optional[EncodedWorldState] = None,
        seed: Optional[int] = None,
        init_noise: Optional[torch.Tensor] = None,
        velocity_hook: Optional[Any] = None,
        return_init_noise: bool = False,
    ) -> dict[str, Any]:
        selected = coerce_mode(mode)
        # Stage-2 W8 sampler interface: UNCOND-only by contract. The IDM branch
        # dispatch filters unknown kwargs, so allowing these through for IDM
        # would silently return an unguided rollout — fail closed here instead.
        if selected is not WAMMode.UNCOND and (
            init_noise is not None or velocity_hook is not None or return_init_noise
        ):
            raise ValueError(
                "init_noise/velocity_hook/return_init_noise are UNCOND-only; got "
                f"mode={selected.value!r}."
            )
        if init_noise is not None and seed is not None:
            raise ValueError("`init_noise` and `seed` are mutually exclusive.")
        requested_horizon, requested_frames, encoded_state = self._validate_act_inputs(
            input_image=input_image,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            generation_horizon=generation_horizon,
            num_video_frames=num_video_frames,
            encoded_state=encoded_state,
        )

        branch, steps = mode_to_branch_steps(selected, inference_steps=self.inference_steps)
        # An injected noise replaces the seed entirely; backfilling default_seed
        # here would trip the base model's init_noise/seed exclusivity check.
        effective_seed = (
            None
            if init_noise is not None
            else (seed if seed is not None else self.default_seed)
        )
        # Forward sampler-interface kwargs only when engaged, so the default
        # path keeps calling older model signatures unchanged.
        sampler_kwargs: dict[str, Any] = {}
        if init_noise is not None:
            sampler_kwargs["init_noise"] = init_noise
        if velocity_hook is not None:
            sampler_kwargs["velocity_hook"] = velocity_hook
        if return_init_noise:
            sampler_kwargs["return_init_noise"] = True
        out = self.model.infer_action(
            prompt=prompt,
            input_image=input_image,
            first_frame_latents=encoded_state.first_frame_latents,
            action_horizon=requested_horizon,
            num_video_frames=requested_frames,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            num_inference_steps=steps,
            sigma_shift=self.sigma_shift,
            seed=effective_seed,
            force_branch=branch,
            return_routing_info=True,
            **sampler_kwargs,
        )
        result = {
            "action_chunk": out["action"],
            "world_feat": encoded_state.world_feat,
            "cost": float(self.cost_table[selected.value]),
            "aux": {
                "mode": selected.value,
                "branch": branch,
                "video_inference_steps": steps if selected is WAMMode.IDM else None,
                "action_inference_steps": steps,
                "num_inference_steps": steps,
                "routing": out.get("_routing"),
            },
        }
        if return_init_noise:
            # Hard index: if the branch ever stops returning the key, replay
            # bookkeeping must fail loudly rather than cache a silent None.
            result["init_noise"] = out["init_noise"]
        return result

    def act_with_grad(
        self,
        *,
        input_image: torch.Tensor,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        prompt: Optional[str] = None,
        generation_horizon: Optional[int] = None,
        num_video_frames: Optional[int] = None,
        encoded_state: Optional[EncodedWorldState] = None,
        seed: Optional[int] = None,
        init_noise: Optional[torch.Tensor] = None,
        velocity_hook: Optional[Any] = None,
        return_init_noise: bool = False,
    ) -> dict[str, Any]:
        """Gradient-carrying forced-UNCOND rollout (stage-2 W17 RL entry).

        Same input contract and validation as ``act(mode=UNCOND)`` — the
        shared ``_validate_act_inputs`` helper is the single source of truth —
        but dispatches through ``infer_action_with_grad``, so
        ``result["action_chunk"]`` keeps its autograd graph: the ``[T, D]``
        action latent on the model device/dtype, NOT detached / NOT moved to
        CPU / NOT cast to float32. Back-propagating a scalar loss built from
        it populates ``.grad`` on every parameter the forward used (with the
        base frozen, only the UNCOND-gated adapter parameters accumulate).

        There is deliberately no ``mode`` parameter: the gradient path exists
        only for the base/UNCOND branch (the IDM future-video rollout has no
        gradient path in W17 scope), and the dual-regime model fails closed on
        any other ``force_branch``. ``encode_world_state`` remains no-grad —
        the first-frame latent is conditioning, not a differentiation target.
        Forward numerics are bitwise identical to ``act(mode=UNCOND)`` for the
        same inputs/seed.
        """
        if not hasattr(self.model, "infer_action_with_grad"):
            raise TypeError(
                "Loaded WAM does not expose `infer_action_with_grad`; "
                "act_with_grad requires a stage-2 W17 dual-regime model."
            )
        if init_noise is not None and seed is not None:
            raise ValueError("`init_noise` and `seed` are mutually exclusive.")
        requested_horizon, requested_frames, encoded_state = self._validate_act_inputs(
            input_image=input_image,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            generation_horizon=generation_horizon,
            num_video_frames=num_video_frames,
            encoded_state=encoded_state,
        )

        branch, steps = mode_to_branch_steps(
            WAMMode.UNCOND, inference_steps=self.inference_steps
        )
        # Mirrors act(): an injected noise replaces the seed entirely, and the
        # sampler-interface kwargs are forwarded only when engaged.
        effective_seed = (
            None
            if init_noise is not None
            else (seed if seed is not None else self.default_seed)
        )
        sampler_kwargs: dict[str, Any] = {}
        if init_noise is not None:
            sampler_kwargs["init_noise"] = init_noise
        if velocity_hook is not None:
            sampler_kwargs["velocity_hook"] = velocity_hook
        if return_init_noise:
            sampler_kwargs["return_init_noise"] = True
        out = self.model.infer_action_with_grad(
            prompt=prompt,
            input_image=input_image,
            first_frame_latents=encoded_state.first_frame_latents,
            action_horizon=requested_horizon,
            num_video_frames=requested_frames,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            num_inference_steps=steps,
            sigma_shift=self.sigma_shift,
            seed=effective_seed,
            force_branch=branch,
            return_routing_info=True,
            **sampler_kwargs,
        )
        result = {
            "action_chunk": out["action"],
            "world_feat": encoded_state.world_feat,
            "cost": float(self.cost_table[WAMMode.UNCOND.value]),
            "aux": {
                "mode": WAMMode.UNCOND.value,
                "branch": branch,
                "video_inference_steps": None,
                "action_inference_steps": steps,
                "num_inference_steps": steps,
                "grad_enabled": True,
                "routing": out.get("_routing"),
            },
        }
        if return_init_noise:
            # Hard index, mirroring act(): replay bookkeeping must fail loudly
            # rather than cache a silent None.
            result["init_noise"] = out["init_noise"]
        return result

    @torch.no_grad()
    def act_control(
        self,
        *,
        input_image: torch.Tensor,
        control: IDMControl | str,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        prompt: Optional[str] = None,
        generation_horizon: Optional[int] = None,
        num_video_frames: Optional[int] = None,
        encoded_state: Optional[EncodedWorldState] = None,
        seed: Optional[int] = None,
        shuffled_future_donor: Optional[ShuffledFutureDonor] = None,
        expected_donor_metadata: Optional[dict[str, Any]] = None,
        extra_action_steps: Optional[int] = None,
        return_video_latents: bool = False,
    ) -> dict[str, Any]:
        """Run a mechanism control without extending the production mode space."""
        selected = coerce_idm_control(control)
        if selected is IDMControl.VALID_IDM and not return_video_latents:
            if shuffled_future_donor is not None or extra_action_steps is not None:
                raise ValueError("valid_idm does not accept donor or extra-action-step overrides.")
            return self.act(
                input_image=input_image,
                mode=WAMMode.IDM,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                prompt=prompt,
                generation_horizon=generation_horizon,
                num_video_frames=num_video_frames,
                encoded_state=encoded_state,
                seed=seed,
            )

        self._validate_cost_resolution(input_image)
        requested_horizon = (
            self.generation_horizon if generation_horizon is None else int(generation_horizon)
        )
        requested_frames = (
            self.num_video_frames if num_video_frames is None else int(num_video_frames)
        )
        if requested_horizon <= 0 or requested_frames <= 0:
            raise ValueError("generation_horizon and num_video_frames must be positive.")
        if self._cost_meta is not None and (
            requested_horizon != self.generation_horizon
            or requested_frames != self.num_video_frames
        ):
            raise ValueError("control shape overrides are incompatible with the profiled WAM shape.")
        expected_proprio_dim = getattr(self.model, "proprio_dim", None)
        if expected_proprio_dim is not None:
            if proprio is None or int(proprio.shape[-1]) != int(expected_proprio_dim):
                actual = None if proprio is None else int(proprio.shape[-1])
                raise ValueError(
                    f"proprio dim must be {expected_proprio_dim} for controls, got {actual}."
                )
        if context is not None:
            if context_mask is None or int(context.shape[-2]) != self.context_len or (
                int(context_mask.shape[-1]) != self.context_len
            ):
                raise ValueError(f"control context/context_mask must use length {self.context_len}.")
        if encoded_state is None:
            encoded_state = self.encode_world_state(input_image)
        if encoded_state.layout != WORLD_FEAT_LAYOUT:
            raise ValueError(
                f"encoded state layout {encoded_state.layout!r} does not match {WORLD_FEAT_LAYOUT!r}."
            )

        branch = "base" if selected is IDMControl.EXTRA_COMPUTE else "idm"
        action_steps = self.inference_steps
        if selected is IDMControl.EXTRA_COMPUTE:
            if extra_action_steps is None or int(extra_action_steps) <= 0:
                raise ValueError("extra_compute requires positive extra_action_steps.")
            if shuffled_future_donor is not None:
                raise ValueError("extra_compute does not accept a future donor.")
            action_steps = int(extra_action_steps)
        elif extra_action_steps is not None:
            raise ValueError(f"{selected.value} does not accept extra_action_steps.")

        signature = inspect.signature(self.model.infer_action)
        if selected is not IDMControl.EXTRA_COMPUTE and "idm_control" not in signature.parameters:
            raise TypeError("Loaded WAM does not expose the experimental IDM-control API.")
        out = self.model.infer_action(
            prompt=prompt,
            input_image=input_image,
            first_frame_latents=encoded_state.first_frame_latents,
            action_horizon=requested_horizon,
            num_video_frames=requested_frames,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            # The base FastWAM branch exposes a single action-solver step
            # count.  IDM splits video/action schedules, so preserve the
            # production video count there while overriding only action steps.
            num_inference_steps=(
                action_steps
                if selected is IDMControl.EXTRA_COMPUTE
                else self.inference_steps
            ),
            action_inference_steps=action_steps,
            sigma_shift=self.sigma_shift,
            seed=seed if seed is not None else self.default_seed,
            force_branch=branch,
            return_routing_info=True,
            idm_control=selected,
            shuffled_future_donor=shuffled_future_donor,
            expected_donor_metadata=expected_donor_metadata,
            return_video_latents=return_video_latents,
        )
        reference_mode = WAMMode.UNCOND if selected is IDMControl.EXTRA_COMPUTE else WAMMode.IDM
        result = {
            "action_chunk": out["action"],
            "world_feat": encoded_state.world_feat,
            # Controls are calibrated in a separate profile. This value is only
            # the production-mode reference and must not enter Gate rewards.
            "cost": float(self.cost_table[reference_mode.value]),
            "aux": {
                "control": selected.value,
                "branch": branch,
                "action_inference_steps": action_steps,
                "video_inference_steps": (
                    None if selected is IDMControl.EXTRA_COMPUTE else self.inference_steps
                ),
                "cost_is_reference": True,
                "routing": out.get("_routing"),
            },
        }
        if "video_latents" in out:
            result["video_latents"] = out["video_latents"]
        return result
