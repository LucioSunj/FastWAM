"""Fail-closed S-DR gradient diagnostics and preflight weight selection."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch


DEFAULT_CANDIDATE_WEIGHTS = (
    0.001,
    0.003,
    0.01,
    0.03,
    0.05,
    0.1,
    0.2,
    0.5,
    1.0,
)
CAP_CANDIDATE_WEIGHTS = (0.2, 0.25, 0.5)


def couple_action_noise_draws(
    draws: Mapping[str, Any], *, mode: str
) -> dict[str, Any]:
    """Return replay draws with independent or common action noise/timestep."""
    if mode not in {"independent", "common"}:
        raise ValueError("action noise coupling must be 'independent' or 'common'.")
    copied = dict(draws)
    regimes = [dict(regime) for regime in draws.get("action_regimes", ())]
    if len(regimes) != 2:
        raise ValueError("Dual-regime replay requires exactly two action regimes.")
    names = [str(regime.get("name")) for regime in regimes]
    if names.count("base") != 1:
        raise ValueError(f"Dual-regime replay requires one base draft, got {names}.")
    main_index = 1 - names.index("base")
    base_index = names.index("base")
    for index, regime in enumerate(regimes):
        for key in ("noise", "timestep"):
            if not torch.is_tensor(regime.get(key)):
                raise TypeError(f"action regime {index} has no tensor-valued {key}.")
    if mode == "common":
        regimes[base_index]["noise"] = regimes[main_index]["noise"].clone()
        regimes[base_index]["timestep"] = regimes[main_index]["timestep"].clone()
    copied["action_regimes"] = regimes
    copied["diagnostic_action_noise_coupling"] = mode
    return copied


@contextmanager
def diagnostic_action_noise_coupling(model, mode: str):
    """Temporarily select replay coupling without changing the training default."""
    if mode not in {"independent", "common"}:
        raise ValueError("action noise coupling must be 'independent' or 'common'.")
    attribute = "_diagnostic_action_noise_coupling"
    missing = object()
    previous = getattr(model, attribute, missing)
    setattr(model, attribute, mode)
    try:
        yield
    finally:
        if previous is missing:
            delattr(model, attribute)
        else:
            setattr(model, attribute, previous)


def diagnostic_parameter_groups(
    model, *, video_final_blocks: int = 1
) -> dict[str, list[torch.nn.Parameter]]:
    """Build overlapping diagnostic views required by the S-DR preflight."""
    action = getattr(model, "action_expert", None)
    if action is None:
        raise ValueError("S-DR diagnostics require action_expert.")
    blocks = list(getattr(action, "blocks", ()))
    groups: dict[str, list[torch.nn.Parameter]] = {
        "action_all": list(action.parameters()),
    }

    input_modules = [
        getattr(action, name, None)
        for name in (
            "action_encoder",
            "text_embedding",
            "time_embedding",
            "time_projection",
        )
    ]
    input_params = [
        parameter
        for module in input_modules
        if module is not None
        for parameter in module.parameters()
    ]
    if input_params:
        groups["action_input_embedding"] = input_params

    if blocks:
        first_end = max(len(blocks) // 3, 1)
        middle_end = max(2 * len(blocks) // 3, first_end + 1)
        groups["action_blocks_early"] = [
            parameter
            for block in blocks[:first_end]
            for parameter in block.parameters()
        ]
        groups["action_blocks_middle"] = [
            parameter
            for block in blocks[first_end:middle_end]
            for parameter in block.parameters()
        ]
        groups["action_blocks_final"] = [
            parameter
            for block in blocks[middle_end:]
            for parameter in block.parameters()
        ]
        groups["action_attention_projections"] = [
            parameter
            for block in blocks
            for module_name in ("self_attn", "cross_attn")
            for parameter in getattr(block, module_name).parameters()
        ]
        groups["action_mlp"] = [
            parameter
            for block in blocks
            for parameter in getattr(block, "ffn").parameters()
        ]
    head = getattr(action, "head", None)
    if head is not None:
        groups["action_output_projection"] = list(head.parameters())

    proprio = getattr(model, "proprio_encoder", None)
    if proprio is not None:
        groups["proprio_all"] = list(proprio.parameters())

    video = getattr(model, "video_expert", None)
    video_blocks = list(getattr(video, "blocks", ())) if video is not None else []
    count = int(video_final_blocks)
    if count <= 0 or count > len(video_blocks):
        raise ValueError(
            "video_final_blocks must select a non-empty suffix of "
            f"video_expert.blocks; requested={count}, available={len(video_blocks)}."
        )
    groups["video_final"] = [
        parameter
        for block in video_blocks[-count:]
        for parameter in block.parameters()
    ]
    return {name: parameters for name, parameters in groups.items() if parameters}


def _gradient_statistics(
    idm_vectors: Mapping[int, torch.Tensor],
    uncond_vectors: Mapping[int, torch.Tensor],
    *,
    count: int,
    group_indices: Mapping[str, Sequence[int]],
    used_idm: set[int] | None = None,
    used_uncond: set[int] | None = None,
) -> dict[str, dict[str, Any]]:
    if count <= 0:
        raise ValueError("Gradient statistics require at least one sample.")
    output = {}
    for name, indices in group_indices.items():
        dot = 0.0
        idm_sq = 0.0
        uncond_sq = 0.0
        for index in indices:
            idm = idm_vectors.get(index)
            uncond = uncond_vectors.get(index)
            if idm is not None:
                idm = idm / float(count)
                idm_sq += float(torch.sum(idm * idm, dtype=torch.float64).item())
            if uncond is not None:
                uncond = uncond / float(count)
                uncond_sq += float(
                    torch.sum(uncond * uncond, dtype=torch.float64).item()
                )
            if idm is not None and uncond is not None:
                dot += float(torch.sum(idm * uncond, dtype=torch.float64).item())
        denom = math.sqrt(idm_sq * uncond_sq)
        output[name] = {
            "dot": dot,
            "idm_sq": idm_sq,
            "uncond_sq": uncond_sq,
            "idm_norm": math.sqrt(idm_sq),
            "uncond_norm": math.sqrt(uncond_sq),
            "cosine": dot / denom if denom > 0.0 else None,
            "parameter_tensor_count": len(indices),
            "idm_used_fraction": (
                sum(index in used_idm for index in indices) / max(len(indices), 1)
                if used_idm is not None
                else None
            ),
            "uncond_used_fraction": (
                sum(index in used_uncond for index in indices)
                / max(len(indices), 1)
                if used_uncond is not None
                else None
            ),
        }
    return output


class ExactGradientAccumulator:
    """Accumulate gradient vectors before computing dot products.

    The global and current-shard sums live on CPU in FP32. Overlapping
    parameter groups share one stored vector, so action_all and its diagnostic
    subviews do not duplicate storage.
    """

    def __init__(
        self,
        parameter_groups: Mapping[str, Sequence[torch.nn.Parameter]],
    ) -> None:
        unique = []
        index_by_id = {}
        group_indices = {}
        for name, parameters in parameter_groups.items():
            indices = []
            for parameter in parameters:
                if not isinstance(parameter, torch.nn.Parameter):
                    raise TypeError(
                        f"Gradient group {name!r} contains a non-Parameter."
                    )
                if not parameter.requires_grad:
                    continue
                identifier = id(parameter)
                if identifier not in index_by_id:
                    index_by_id[identifier] = len(unique)
                    unique.append(parameter)
                indices.append(index_by_id[identifier])
            if indices:
                group_indices[str(name)] = tuple(dict.fromkeys(indices))
        if not unique:
            raise ValueError("Exact gradient accumulation needs trainable parameters.")
        self.parameters = tuple(unique)
        self.group_indices = group_indices
        self._global_idm: dict[int, torch.Tensor] = {}
        self._global_uncond: dict[int, torch.Tensor] = {}
        self._global_count = 0
        self._used_idm: set[int] = set()
        self._used_uncond: set[int] = set()
        self._shard_index: int | None = None
        self._shard_idm: dict[int, torch.Tensor] = {}
        self._shard_uncond: dict[int, torch.Tensor] = {}
        self._shard_count = 0
        self._shards: list[dict[str, Any]] = []

    @staticmethod
    def _add(
        bank: dict[int, torch.Tensor],
        index: int,
        gradient: torch.Tensor | None,
    ) -> None:
        if gradient is None:
            return
        value = gradient.detach().to(device="cpu", dtype=torch.float32)
        if index in bank:
            bank[index].add_(value)
        else:
            bank[index] = value.clone()

    def start_shard(self, shard_index: int) -> None:
        if self._shard_index is not None:
            raise RuntimeError("Finish the active gradient shard before starting another.")
        self._shard_index = int(shard_index)
        self._shard_idm = {}
        self._shard_uncond = {}
        self._shard_count = 0

    def accumulate(
        self, idm_loss: torch.Tensor, uncond_loss: torch.Tensor
    ) -> None:
        if self._shard_index is None:
            raise RuntimeError("start_shard() must be called before accumulate().")
        if idm_loss.ndim != 0 or uncond_loss.ndim != 0:
            raise ValueError("Raw IDM and UNCOND losses must be scalar tensors.")
        idm_gradients = torch.autograd.grad(
            idm_loss,
            self.parameters,
            retain_graph=True,
            allow_unused=True,
        )
        uncond_gradients = torch.autograd.grad(
            uncond_loss,
            self.parameters,
            retain_graph=False,
            allow_unused=True,
        )
        for index, (idm_gradient, uncond_gradient) in enumerate(
            zip(idm_gradients, uncond_gradients)
        ):
            self._add(self._global_idm, index, idm_gradient)
            self._add(self._global_uncond, index, uncond_gradient)
            self._add(self._shard_idm, index, idm_gradient)
            self._add(self._shard_uncond, index, uncond_gradient)
            if idm_gradient is not None:
                self._used_idm.add(index)
            if uncond_gradient is not None:
                self._used_uncond.add(index)
        self._global_count += 1
        self._shard_count += 1
        del idm_gradients, uncond_gradients

    def finish_shard(self) -> dict[str, Any]:
        if self._shard_index is None:
            raise RuntimeError("No active gradient shard.")
        if self._shard_count <= 0:
            raise ValueError("A gradient shard cannot be empty.")
        result = {
            "shard_index": self._shard_index,
            "sample_count": self._shard_count,
            "groups": _gradient_statistics(
                self._shard_idm,
                self._shard_uncond,
                count=self._shard_count,
                group_indices=self.group_indices,
            ),
        }
        self._shards.append(result)
        self._shard_index = None
        self._shard_idm = {}
        self._shard_uncond = {}
        self._shard_count = 0
        return result

    def finalize(self) -> dict[str, Any]:
        if self._shard_index is not None:
            raise RuntimeError("Finish the active gradient shard before finalize().")
        return {
            "sample_count": self._global_count,
            "accumulator_device": "cpu",
            "accumulator_dtype": "float32",
            "cross_microbatch_terms_included": True,
            "groups": _gradient_statistics(
                self._global_idm,
                self._global_uncond,
                count=self._global_count,
                group_indices=self.group_indices,
                used_idm=self._used_idm,
                used_uncond=self._used_uncond,
            ),
            "shards": list(self._shards),
        }

    def storage_bytes(self) -> int:
        banks = (
            self._global_idm,
            self._global_uncond,
            self._shard_idm,
            self._shard_uncond,
        )
        return sum(
            value.numel() * value.element_size()
            for bank in banks
            for value in bank.values()
        )


def weighted_descent_margins(
    statistics: Mapping[str, Any],
    weight: float,
    *,
    epsilon: float = 1e-12,
) -> dict[str, Any]:
    weight = float(weight)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("UNCOND weight must be finite and non-negative.")
    a = float(statistics["idm_sq"])
    b = float(statistics["uncond_sq"])
    c = float(statistics["dot"])
    if not all(math.isfinite(value) for value in (a, b, c)):
        raise ValueError("Gradient statistics must be finite.")
    if a < 0.0 or b < 0.0:
        raise ValueError("Squared gradient norms cannot be negative.")

    denominator = 1.0 + weight
    idm_margin = (a + weight * c) / denominator
    uncond_margin = (c + weight * b) / denominator
    if c < 0.0:
        interval = {
            "kind": "conflicting",
            "lower": (-c / b) if b > 0.0 else None,
            "upper": (a / -c) if a > 0.0 else 0.0,
        }
    else:
        interval = {
            "kind": "nonconflicting",
            "lower": 0.0,
            "upper": None,
        }
    return {
        "weight": weight,
        "idm_margin": idm_margin,
        "uncond_margin": uncond_margin,
        "normalized_idm_margin": idm_margin / (a + epsilon),
        "normalized_uncond_margin": uncond_margin / (b + epsilon),
        "weighted_gradient_norm_ratio": (
            weight * math.sqrt(b) / (math.sqrt(a) + epsilon)
        ),
        "simultaneous_descent_interval": interval,
    }


def negative_margin_fraction(
    shards: Sequence[Mapping[str, Any]],
    *,
    group_name: str,
    weight: float,
    objective: str,
) -> float:
    if objective not in {"idm", "uncond"}:
        raise ValueError("objective must be 'idm' or 'uncond'.")
    if not shards:
        raise ValueError("negative-margin fraction requires at least one shard.")
    key = f"{objective}_margin"
    negative = 0
    for shard in shards:
        groups = shard.get("groups", {})
        if group_name not in groups:
            raise ValueError(f"Gradient shard is missing group {group_name!r}.")
        if weighted_descent_margins(groups[group_name], weight)[key] <= 0.0:
            negative += 1
    return negative / len(shards)


def select_preflight_weights(
    diagnostics: Mapping[str, Any],
    *,
    max_negative_fraction: float = 0.20,
) -> dict[str, Any]:
    groups = diagnostics.get("groups", {})
    shards = diagnostics.get("shards", ())
    if "action_all" not in groups or "action_blocks_final" not in groups:
        raise ValueError(
            "Preflight weight selection requires action_all and action_blocks_final."
        )
    action = groups["action_all"]
    idm_norm = float(action["idm_norm"])
    uncond_norm = float(action["uncond_norm"])
    w0 = min(0.05, 0.25 * idm_norm / (uncond_norm + 1e-12))

    candidate_weights = sorted(
        set(DEFAULT_CANDIDATE_WEIGHTS) | set(CAP_CANDIDATE_WEIGHTS)
    )
    table = {}
    for weight in candidate_weights:
        row = weighted_descent_margins(action, weight)
        row["action_all_negative_idm_fraction"] = negative_margin_fraction(
            shards,
            group_name="action_all",
            weight=weight,
            objective="idm",
        )
        row["action_all_negative_uncond_fraction"] = negative_margin_fraction(
            shards,
            group_name="action_all",
            weight=weight,
            objective="uncond",
        )
        row["final_blocks_negative_idm_fraction"] = negative_margin_fraction(
            shards,
            group_name="action_blocks_final",
            weight=weight,
            objective="idm",
        )
        table[str(weight)] = row

    safe = []
    for weight in CAP_CANDIDATE_WEIGHTS:
        row = table[str(weight)]
        if (
            row["idm_margin"] > 0.0
            and row["uncond_margin"] > 0.0
            and row["action_all_negative_idm_fraction"]
            <= max_negative_fraction
            and row["action_all_negative_uncond_fraction"]
            <= max_negative_fraction
            and row["final_blocks_negative_idm_fraction"]
            <= max_negative_fraction
        ):
            safe.append(weight)
    w_cap = max(safe) if safe else None
    w0_margins = weighted_descent_margins(action, w0)
    schedule = (
        None
        if w_cap is None or w0 <= 0.0
        else [
            [0.0, w0],
            [0.1, w0],
            [0.3, min(0.2, w_cap)],
            [0.6, w_cap],
            [1.0, w_cap],
        ]
    )
    return {
        "w0": w0,
        "w_cap": w_cap,
        "safe_candidate_weights": safe,
        "candidate_margins": table,
        "w0_margins": w0_margins,
        "schedule": schedule,
        "go": bool(
            w_cap is not None
            and w_cap >= 0.2
            and w0 > 0.0
            and w0_margins["idm_margin"] > 0.0
        ),
    }


def check_loss_arithmetic(
    *,
    idm_raw: float,
    uncond_raw: float,
    weight: float,
    idm_contribution: float,
    uncond_contribution: float,
    combined: float,
    tolerance: float = 1e-7,
) -> dict[str, Any]:
    denominator = 1.0 + float(weight)
    expected_idm = float(idm_raw) / denominator
    expected_uncond = float(weight) * float(uncond_raw) / denominator
    expected_combined = expected_idm + expected_uncond
    errors = {
        "idm_contribution": abs(float(idm_contribution) - expected_idm),
        "uncond_contribution": abs(float(uncond_contribution) - expected_uncond),
        "combined": abs(float(combined) - expected_combined),
    }
    return {
        "pass": all(value <= tolerance for value in errors.values()),
        "absolute_errors": errors,
        "expected": {
            "idm_contribution": expected_idm,
            "uncond_contribution": expected_uncond,
            "combined": expected_combined,
        },
    }
