"""Lightweight dual-regime training helpers.

The helpers in this module deliberately avoid importing Accelerate or the Wan
stack.  This keeps schedule validation and gradient-alignment diagnostics easy
to unit test on CPU while :mod:`fastwam.trainer` owns the distributed wiring.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


def classify_training_resume_source(
    resume_path: str | Path, *, is_dual_regime: bool
) -> str:
    """Classify a resume artifact and reject unsafe adaptive weight-only resumes."""
    path = Path(resume_path)
    if path.is_dir():
        return "full_state"
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    if is_dual_regime:
        raise ValueError(
            "Dual-regime training cannot resume from a weights-only file: it "
            "would restore schedule lineage without optimizer, scheduler, or "
            "global-step state. Resume from the matching Accelerate state "
            "directory, or start a new explicit warm_start lineage."
        )
    return "weights_only"


def validate_dual_regime_trainer_state(
    payload: Mapping[str, Any] | None,
    *,
    expected_contract: Mapping[str, Any],
    expected_dataset_stats_fingerprint: str,
) -> int:
    """Fail closed before restoring a dual-regime Accelerate state directory."""
    if payload is None:
        raise ValueError(
            "Adaptive training state is missing trainer_state.json; optimizer-step "
            "lineage and the dual-regime schedule cannot be verified."
        )
    if "dual_regime_optimizer_steps" not in payload:
        raise ValueError(
            "Adaptive trainer state predates dual_regime_optimizer_steps and "
            "cannot be resumed as a verified lineage."
        )
    loaded_contract = payload.get("dual_regime_training_contract")
    if loaded_contract != expected_contract:
        raise ValueError(
            "Adaptive trainer-state contract does not match current config: "
            f"state={loaded_contract}, current={dict(expected_contract)}."
        )
    loaded_steps = payload["dual_regime_optimizer_steps"]
    total_steps = loaded_contract.get("total_optimizer_steps")
    if (
        isinstance(loaded_steps, bool)
        or not isinstance(loaded_steps, int)
        or isinstance(total_steps, bool)
        or not isinstance(total_steps, int)
        or not 0 <= loaded_steps <= total_steps
    ):
        raise ValueError(
            "Adaptive trainer dual_regime_optimizer_steps is outside its "
            f"contract: {loaded_steps!r}."
        )
    global_step = payload.get("global_step")
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step != loaded_steps
    ):
        raise ValueError(
            "Adaptive trainer global_step must equal the number of successful "
            f"dual-regime optimizer updates: global={global_step!r}, dual={loaded_steps}."
        )
    stored_stats = payload.get("dataset_stats_fingerprint")
    if stored_stats != expected_dataset_stats_fingerprint:
        raise ValueError(
            "Adaptive resume dataset-stats SHA does not match trainer state: "
            f"state={stored_stats}, current={expected_dataset_stats_fingerprint}."
        )
    return loaded_steps


def advance_successful_optimizer_steps(
    *,
    global_step: int,
    dual_regime_optimizer_steps: int,
    is_dual_regime: bool,
    optimizer_step_was_skipped: bool,
) -> tuple[int, int]:
    """Advance counters only for an optimizer update that actually happened."""
    global_step = int(global_step)
    dual_steps = int(dual_regime_optimizer_steps)
    if optimizer_step_was_skipped:
        return global_step, dual_steps
    return global_step + 1, dual_steps + int(bool(is_dual_regime))


def normalized_dual_regime_action_loss(
    main_loss: torch.Tensor,
    uncond_loss: torch.Tensor,
    uncond_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return combined loss plus each regime's normalized contribution."""
    weight = float(uncond_weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError(f"action_regime_weight_uncond must be finite and >= 0, got {weight}.")
    denom = 1.0 + weight
    main_contribution = main_loss / denom
    uncond_contribution = weight * uncond_loss / denom
    return main_contribution + uncond_contribution, main_contribution, uncond_contribution


def canonicalize_uncond_weight_schedule(
    points: Sequence[Mapping[str, Any] | Sequence[float]],
) -> tuple[tuple[float, float], ...]:
    """Validate a piecewise-linear schedule expressed in optimizer fractions.

    Each point may be ``{"fraction": f, "weight": w}`` or ``(f, w)``.  The
    schedule must cover the complete run, use strictly increasing fractions,
    and keep the UNCOND objective active with a finite positive weight.
    """
    if not points:
        raise ValueError("UNCOND weight schedule must contain at least two points.")

    result: list[tuple[float, float]] = []
    for index, point in enumerate(points):
        if isinstance(point, Mapping):
            if set(point) != {"fraction", "weight"}:
                raise ValueError(
                    "UNCOND schedule mapping keys must be exactly "
                    f"['fraction', 'weight'], got {sorted(point)} at index {index}."
                )
            fraction = float(point["fraction"])
            weight = float(point["weight"])
        else:
            values = list(point)
            if len(values) != 2:
                raise ValueError(
                    f"UNCOND schedule point {index} must have two values, got {values}."
                )
            fraction, weight = map(float, values)
        if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError(
                f"UNCOND schedule fraction must be finite and in [0, 1], got {fraction}."
            )
        if not math.isfinite(weight) or weight <= 0.0:
            raise ValueError(
                f"UNCOND schedule weight must be finite and > 0, got {weight}."
            )
        if result and fraction <= result[-1][0]:
            raise ValueError("UNCOND schedule fractions must be strictly increasing.")
        result.append((fraction, weight))

    if len(result) < 2 or result[0][0] != 0.0 or result[-1][0] != 1.0:
        raise ValueError("UNCOND weight schedule must start at 0.0 and end at 1.0.")
    return tuple(result)


def uncond_weight_at_step(
    schedule: Sequence[tuple[float, float]],
    *,
    optimizer_step: int,
    total_optimizer_steps: int,
) -> float:
    """Interpolate ``schedule`` at the start of an optimizer step."""
    if int(total_optimizer_steps) <= 0:
        raise ValueError(
            f"total_optimizer_steps must be positive, got {total_optimizer_steps}."
        )
    step = min(max(int(optimizer_step), 0), int(total_optimizer_steps))
    fraction = step / float(total_optimizer_steps)
    canonical = tuple((float(f), float(w)) for f, w in schedule)
    if not canonical:
        raise ValueError("schedule cannot be empty.")
    if fraction <= canonical[0][0]:
        return canonical[0][1]
    for (left_f, left_w), (right_f, right_w) in zip(canonical, canonical[1:]):
        if fraction <= right_f:
            alpha = (fraction - left_f) / (right_f - left_f)
            return left_w + alpha * (right_w - left_w)
    return canonical[-1][1]


def raw_loss_gradient_statistics(
    main_loss: torch.Tensor,
    uncond_loss: torch.Tensor,
    parameter_groups: Mapping[str, Sequence[torch.nn.Parameter]],
) -> dict[str, torch.Tensor]:
    """Return additive gradient statistics without modifying ``param.grad``.

    Each output is ``[dot, main_sq, uncond_sq, used_by_both, parameter_count]``.
    The first three entries can be summed across ranks before computing cosine.
    Parameter groups may overlap; gradients are evaluated only once for the
    unique union and then accumulated for each requested diagnostic view.
    """
    if main_loss.ndim != 0 or uncond_loss.ndim != 0:
        raise ValueError("raw regime losses must be scalar tensors.")

    unique: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    normalized_groups: dict[str, list[torch.nn.Parameter]] = {}
    for name, params in parameter_groups.items():
        group: list[torch.nn.Parameter] = []
        for param in params:
            if not isinstance(param, torch.nn.Parameter):
                raise TypeError(f"gradient diagnostic group {name!r} contains a non-Parameter.")
            if not param.requires_grad:
                continue
            group.append(param)
            if id(param) not in seen:
                seen.add(id(param))
                unique.append(param)
        if group:
            normalized_groups[str(name)] = group
    if not unique:
        raise ValueError("gradient diagnostics require at least one trainable parameter.")

    main_grads = torch.autograd.grad(
        main_loss, unique, retain_graph=True, allow_unused=True
    )
    uncond_grads = torch.autograd.grad(
        uncond_loss, unique, retain_graph=True, allow_unused=True
    )
    by_id = {
        id(param): (main_grad, uncond_grad)
        for param, main_grad, uncond_grad in zip(unique, main_grads, uncond_grads)
    }

    output: dict[str, torch.Tensor] = {}
    device = main_loss.device
    for name, params in normalized_groups.items():
        dot = torch.zeros((), device=device, dtype=torch.float64)
        main_sq = torch.zeros_like(dot)
        uncond_sq = torch.zeros_like(dot)
        used_by_both = 0
        for param in params:
            main_grad, uncond_grad = by_id[id(param)]
            if main_grad is not None:
                main_sq = main_sq + main_grad.detach().double().pow(2).sum()
            if uncond_grad is not None:
                uncond_sq = uncond_sq + uncond_grad.detach().double().pow(2).sum()
            if main_grad is not None and uncond_grad is not None:
                dot = dot + (main_grad.detach().double() * uncond_grad.detach().double()).sum()
                used_by_both += 1
        output[name] = torch.stack(
            [
                dot,
                main_sq,
                uncond_sq,
                torch.tensor(float(used_by_both), device=device, dtype=torch.float64),
                torch.tensor(float(len(params)), device=device, dtype=torch.float64),
            ]
        )
    return output


def build_optimizer_parameter_groups(
    model,
    *,
    base_learning_rate: float,
    action_lr_scale: float = 1.0,
    proprio_lr_scale: float = 1.0,
    video_lr_scale: float = 1.0,
    video_final_blocks: int | None = None,
) -> list[dict]:
    """Freeze the model and build named, identity-deduplicated expert groups."""
    base_learning_rate = float(base_learning_rate)
    scales = {
        "action": float(action_lr_scale),
        "proprio": float(proprio_lr_scale),
        "video": float(video_lr_scale),
    }
    if not math.isfinite(base_learning_rate) or base_learning_rate <= 0.0:
        raise ValueError("base_learning_rate must be finite and positive.")
    for name, scale in scales.items():
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError(f"{name}_lr_scale must be finite and non-negative, got {scale}.")

    model.eval()
    model.requires_grad_(False)
    modules = {
        "action": getattr(model, "action_expert", None),
        "proprio": getattr(model, "proprio_encoder", None),
        "video": getattr(model, "video_expert", None),
    }
    groups: list[dict] = []
    seen: set[int] = set()
    for name in ("action", "proprio", "video"):
        module = modules[name]
        scale = scales[name]
        if module is None or scale == 0.0:
            continue
        if name == "video" and video_final_blocks is not None:
            if isinstance(video_final_blocks, bool) or int(video_final_blocks) <= 0:
                raise ValueError(
                    "video_final_blocks must be a positive integer when specified."
                )
            blocks = list(getattr(module, "blocks", ()))
            count = int(video_final_blocks)
            if not blocks or count > len(blocks):
                raise ValueError(
                    "video_final_blocks cannot be resolved against video_expert.blocks: "
                    f"requested={count}, available={len(blocks)}."
                )
            candidate_params = [
                param for block in blocks[-count:] for param in block.parameters()
            ]
        else:
            candidate_params = list(module.parameters())
        params = []
        for param in candidate_params:
            if id(param) in seen:
                continue
            seen.add(id(param))
            param.requires_grad_(True)
            params.append(param)
        if params:
            module.train()
            groups.append(
                {
                    "name": name,
                    "params": params,
                    "lr": base_learning_rate * scale,
                    "lr_scale": scale,
                }
            )
    if not groups:
        raise ValueError("Optimizer configuration freezes every parameter group.")
    return groups
