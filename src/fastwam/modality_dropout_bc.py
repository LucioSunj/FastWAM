"""Contracts and statistics for the symmetric modality-dropout BC pilot."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from fastwam.models.wan22.adaptive_action import ModalityKeepMask
from fastwam.models.wan22.visual_contracts import SpatialPatchMemory

MODALITY_DROPOUT_PILOT_SCHEMA = "fastwam-modality-dropout-bc-pilot-v1"
RANDOM_PATCH_BANK_SEED = 42


@dataclass(frozen=True)
class ModalityDropoutArm:
    """One preregistered pilot arm."""

    name: str
    p_wan: float
    p_dino: float
    dino_input: str

    def __post_init__(self) -> None:
        for name, value in (("p_wan", self.p_wan), ("p_dino", self.p_dino)):
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and lie in [0,1].")
        if self.dino_input not in {"real", "fixed_gaussian"}:
            raise ValueError("DINO input must be real or fixed_gaussian.")


PILOT_ARMS: dict[str, ModalityDropoutArm] = {
    "A": ModalityDropoutArm("A", 0.0, 0.0, "real"),
    "B15": ModalityDropoutArm("B15", 0.15, 0.15, "real"),
    "B30": ModalityDropoutArm("B30", 0.30, 0.30, "real"),
    "B50": ModalityDropoutArm("B50", 0.50, 0.50, "real"),
    "C": ModalityDropoutArm("C", 0.30, 0.0, "real"),
    "D": ModalityDropoutArm("D", 0.30, 0.30, "fixed_gaussian"),
}


def resolve_pilot_arm(name: str) -> ModalityDropoutArm:
    """Resolve an exact preregistered arm name."""

    normalized = str(name).strip().upper()
    if normalized not in PILOT_ARMS:
        raise ValueError(
            f"Unknown modality-dropout arm {name!r}; expected {sorted(PILOT_ARMS)}."
        )
    return PILOT_ARMS[normalized]


def _stateless_uniform(identity: str, *, seed: int, step: int, domain: str) -> float:
    payload = (
        f"{MODALITY_DROPOUT_PILOT_SCHEMA}\0{int(seed)}\0{int(step)}\0"
        f"{domain}\0{identity}"
    ).encode()
    integer = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return (integer + 0.5) / float(1 << 64)


def sample_modality_keep_mask(
    *,
    sample_identities: Sequence[str | int],
    p_wan: float,
    p_dino: float,
    seed: int,
    step: int,
    device: torch.device | str,
) -> ModalityKeepMask | None:
    """Sample independent per-sample masks once for a complete action chunk.

    Sampling is stateless and happens logically on CPU. Thus rank, batch traversal,
    and microbatch partitioning cannot change a sample's draw. Returning ``None``
    for the exact zero-probability arm preserves the pre-pilot forward path.
    """

    probabilities = (float(p_wan), float(p_dino))
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in probabilities
    ):
        raise ValueError("Modality dropout probabilities must be finite in [0,1].")
    if int(step) < 0:
        raise ValueError("Modality dropout step must be non-negative.")
    identities = tuple(str(identity) for identity in sample_identities)
    if not identities:
        raise ValueError("Modality dropout requires at least one sample identity.")
    if len(set(identities)) != len(identities):
        raise ValueError("Sample identities must be unique within a BC chunk.")
    if probabilities == (0.0, 0.0):
        return None
    wan = [
        _stateless_uniform(identity, seed=seed, step=step, domain="wan") >= p_wan
        for identity in identities
    ]
    dino = [
        _stateless_uniform(identity, seed=seed, step=step, domain="dino") >= p_dino
        for identity in identities
    ]
    return ModalityKeepMask(
        wan=torch.tensor(wan, dtype=torch.bool, device=device),
        dino=torch.tensor(dino, dtype=torch.bool, device=device),
    )


def forced_modality_keep_mask(
    condition: str,
    *,
    batch_size: int,
    device: torch.device | str,
) -> ModalityKeepMask | None:
    """Build one of the four held-out ablation conditions."""

    normalized = str(condition).strip().lower().replace("-", "_")
    if normalized == "clean":
        return None
    choices = {
        "wan_drop": (False, True),
        "dino_drop": (True, False),
        "both_drop": (False, False),
    }
    if normalized not in choices:
        raise ValueError(f"Unknown held-out modality condition {condition!r}.")
    if int(batch_size) < 1:
        raise ValueError("Forced modality masks require a positive batch size.")
    wan, dino = choices[normalized]
    return ModalityKeepMask(
        wan=torch.full((batch_size,), wan, dtype=torch.bool, device=device),
        dino=torch.full((batch_size,), dino, dtype=torch.bool, device=device),
    )


def _tensor_sha256(tensor: torch.Tensor) -> str:
    canonical = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(canonical.dtype).encode())
    digest.update(str(tuple(canonical.shape)).encode())
    digest.update(canonical.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def fixed_gaussian_patch_memory(
    memory: SpatialPatchMemory,
    *,
    seed: int = RANDOM_PATCH_BANK_SEED,
) -> tuple[SpatialPatchMemory, dict[str, Any]]:
    """Replace V2 tokens with one runtime-shaped Gaussian bank shared over B."""

    if not isinstance(memory, SpatialPatchMemory):
        raise TypeError("The random-patch pilot arm requires SpatialPatchMemory.")
    _, views, patches, dimension = memory.tokens.shape
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    bank = torch.randn(
        (views, patches, dimension),
        generator=generator,
        dtype=torch.float32,
    ).to(dtype=memory.tokens.dtype)
    metadata = {
        "schema": "fastwam-fixed-gaussian-patch-bank-v1",
        "seed": int(seed),
        "shape": [views, patches, dimension],
        "dtype": str(bank.dtype),
        "sha256": _tensor_sha256(bank),
    }
    tokens = bank.to(device=memory.tokens.device)
    tokens = tokens.unsqueeze(0).expand(memory.tokens.shape[0], -1, -1, -1)
    tokens = tokens * memory.patch_valid_mask.unsqueeze(-1)
    return replace(memory, tokens=tokens.clone()), metadata


def _as_finite_vector(
    values: Sequence[float] | torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float64).flatten().cpu()
    if tensor.numel() < 2:
        raise ValueError(f"{name} requires at least two paired samples.")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} contains non-finite values.")
    return tensor


def _ratio(clean: torch.Tensor, dropped: torch.Tensor) -> torch.Tensor:
    denominator = clean.mean()
    if float(denominator) <= 0.0:
        raise ValueError("Clean held-out loss mean must be positive.")
    return (dropped.mean() - denominator) / denominator


def _bootstrap_indices(
    sample_count: int,
    *,
    draws: int,
    seed: int,
    chunk_size: int = 512,
):
    if draws < 1:
        raise ValueError("Bootstrap draw count must be positive.")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    remaining = int(draws)
    while remaining:
        count = min(remaining, int(chunk_size))
        yield torch.randint(
            sample_count,
            (count, sample_count),
            generator=generator,
        )
        remaining -= count


def paired_bootstrap_reliance_change(
    first: Mapping[str, Sequence[float] | torch.Tensor],
    second: Mapping[str, Sequence[float] | torch.Tensor],
    *,
    dropped_key: str = "dino_drop",
    draws: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap a paired change in relative ablation reliance."""

    vectors = {
        "first_clean": _as_finite_vector(first["clean"], name="first clean"),
        "first_drop": _as_finite_vector(first[dropped_key], name="first drop"),
        "second_clean": _as_finite_vector(second["clean"], name="second clean"),
        "second_drop": _as_finite_vector(second[dropped_key], name="second drop"),
    }
    sizes = {value.numel() for value in vectors.values()}
    if len(sizes) != 1:
        raise ValueError("Paired checkpoint loss arrays must have identical lengths.")
    sample_count = sizes.pop()
    point = _ratio(vectors["second_clean"], vectors["second_drop"]) - _ratio(
        vectors["first_clean"], vectors["first_drop"]
    )
    samples = []
    for index in _bootstrap_indices(sample_count, draws=draws, seed=seed):
        first_clean = vectors["first_clean"][index].mean(dim=1)
        first_drop = vectors["first_drop"][index].mean(dim=1)
        second_clean = vectors["second_clean"][index].mean(dim=1)
        second_drop = vectors["second_drop"][index].mean(dim=1)
        first_ratio = (first_drop - first_clean) / first_clean
        second_ratio = (second_drop - second_clean) / second_clean
        samples.append(second_ratio - first_ratio)
    boot = torch.cat(samples)
    interval = torch.quantile(
        boot,
        torch.tensor([0.025, 0.975], dtype=boot.dtype),
    ).tolist()
    return {
        "point": float(point),
        "ci95": [float(interval[0]), float(interval[1])],
        "paired_samples": int(sample_count),
        "bootstrap_draws": int(draws),
        "seed": int(seed),
    }


def summarize_heldout_losses(
    losses: Mapping[str, Sequence[float] | torch.Tensor],
) -> dict[str, Any]:
    """Summarize the four fixed held-out modality conditions."""

    required = ("clean", "wan_drop", "dino_drop", "both_drop")
    missing = [key for key in required if key not in losses]
    if missing:
        raise KeyError(f"Held-out losses are missing conditions: {missing}.")
    vectors = {key: _as_finite_vector(losses[key], name=key) for key in required}
    if len({value.numel() for value in vectors.values()}) != 1:
        raise ValueError("Held-out condition arrays must be sample-aligned.")
    clean = vectors["clean"]
    return {
        "sample_count": int(clean.numel()),
        "loss": {key: float(value.mean()) for key, value in vectors.items()},
        "d_dino_loss": float(_ratio(clean, vectors["dino_drop"])),
        "d_wan_loss": float(_ratio(clean, vectors["wan_drop"])),
    }


def baseline_plateau_decision(
    history: Sequence[Mapping[str, Any]],
    *,
    draws: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Apply the preregistered baseline plateau rule without extrapolation."""

    by_step = {int(item["step"]): item for item in history}
    steps = sorted(by_step)
    if len(steps) != len(history):
        raise ValueError("Baseline history contains duplicate steps.")
    if not steps or steps[0] != 0 or any(step % 500 for step in steps):
        raise ValueError("Baseline evaluations must start at 0 on a 500-step grid.")
    current = steps[-1]
    if current < 1000:
        return {"platform": False, "endpoint": None, "reason": "need_step_1000"}
    if current > 3000:
        raise ValueError("Baseline plateau calibration cannot exceed 3000 steps.")
    if current - 500 not in by_step or 1000 not in by_step:
        raise ValueError("Baseline plateau history has a missing required checkpoint.")

    early = paired_bootstrap_reliance_change(
        by_step[0]["losses"],
        by_step[1000]["losses"],
        draws=draws,
        seed=seed,
    )
    recent = paired_bootstrap_reliance_change(
        by_step[current - 500]["losses"],
        by_step[current]["losses"],
        draws=draws,
        seed=seed + current,
    )
    recent_ci_contains_zero = recent["ci95"][0] <= 0.0 <= recent["ci95"][1]
    threshold = 0.1 * abs(float(early["point"]))
    regular = abs(float(recent["point"])) <= threshold and recent_ci_contains_zero
    early_ci_contains_zero = early["ci95"][0] <= 0.0 <= early["ci95"][1]
    all_changes = []
    all_no_change = early_ci_contains_zero
    if early_ci_contains_zero:
        for step in steps[1:]:
            change = paired_bootstrap_reliance_change(
                by_step[0]["losses"],
                by_step[step]["losses"],
                draws=draws,
                seed=seed + 10_000 + step,
            )
            all_changes.append({"step": step, **change})
            all_no_change &= change["ci95"][0] <= 0.0 <= change["ci95"][1]
    no_change_exception = all_no_change and recent_ci_contains_zero
    platform = bool(regular or no_change_exception)
    reason = (
        "regular_plateau"
        if regular
        else "all_changes_indistinguishable_from_zero"
        if no_change_exception
        else "not_platformed"
    )
    if current == 3000 and not platform:
        reason = "max_step_without_platform"
    return {
        "platform": platform,
        "endpoint": current if platform else None,
        "reason": reason,
        "early_0_to_1000": early,
        "recent_500": recent,
        "regular_absolute_threshold": threshold,
        "all_changes_from_step_0": all_changes,
    }


def aggregate_dino_diagnostics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate gate, pre-gate projection, residual, and cross-sample variance."""

    if not records:
        raise ValueError("DINO diagnostics aggregation requires records.")
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["layer_index"])].append(record)

    layers: dict[str, dict[str, float | int]] = {}
    for layer_index, layer_records in sorted(grouped.items()):
        gate_values = torch.cat(
            [
                torch.as_tensor(record["effective_gate"]).reshape(-1)
                for record in layer_records
            ]
        ).double()
        projected_values = torch.cat(
            [
                torch.as_tensor(record["projected_norm"]).reshape(-1)
                for record in layer_records
            ]
        ).double()
        residual_values = torch.cat(
            [
                torch.as_tensor(record["effective_residual_norm"]).reshape(-1)
                for record in layer_records
            ]
        ).double()
        sample_count = sum(int(record["sample_count"]) for record in layer_records)
        residual_sum = sum(
            (
                torch.as_tensor(record["effective_residual_sum"]).double()
                for record in layer_records
            ),
            start=torch.zeros_like(
                torch.as_tensor(layer_records[0]["effective_residual_sum"]).double()
            ),
        )
        residual_square_sum = sum(
            (
                torch.as_tensor(record["effective_residual_square_sum"]).double()
                for record in layer_records
            ),
            start=torch.zeros_like(
                torch.as_tensor(
                    layer_records[0]["effective_residual_square_sum"]
                ).double()
            ),
        )
        variance = residual_square_sum / sample_count - (
            residual_sum / sample_count
        ).square()
        layers[str(layer_index)] = {
            "sample_count": sample_count,
            "gate_mean": float(gate_values.mean()),
            "projected_norm": float(projected_values.mean()),
            "residual_norm": float(residual_values.mean()),
            "residual_cross_sample_variance": float(variance.clamp_min(0).mean()),
        }
    metric_names = (
        "gate_mean",
        "projected_norm",
        "residual_norm",
        "residual_cross_sample_variance",
    )
    overall = {
        name: float(sum(float(value[name]) for value in layers.values()) / len(layers))
        for name in metric_names
    }
    return {"layers": layers, "overall": overall}


def random_patch_kill_test(
    *,
    baseline: Mapping[str, Sequence[float] | torch.Tensor],
    semantic: Mapping[str, Sequence[float] | torch.Tensor],
    random_patch: Mapping[str, Sequence[float] | torch.Tensor],
    draws: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Test whether random patches reproduce at least 80% of semantic reliance."""

    vectors = {
        f"{arm}_{condition}": _as_finite_vector(
            payload[condition], name=f"{arm} {condition}"
        )
        for arm, payload in (
            ("a", baseline),
            ("b", semantic),
            ("d", random_patch),
        )
        for condition in ("clean", "dino_drop")
    }
    sizes = {value.numel() for value in vectors.values()}
    if len(sizes) != 1:
        raise ValueError("Random-patch kill-test arrays must be paired and aligned.")
    count = sizes.pop()

    def contrast(index: torch.Tensor | None = None) -> torch.Tensor:
        def arm_ratio(prefix: str) -> torch.Tensor:
            clean = vectors[f"{prefix}_clean"]
            dropped = vectors[f"{prefix}_dino_drop"]
            if index is None:
                return _ratio(clean, dropped)
            clean_mean = clean[index].mean(dim=1)
            drop_mean = dropped[index].mean(dim=1)
            return (drop_mean - clean_mean) / clean_mean

        a = arm_ratio("a")
        return (arm_ratio("d") - a) - 0.8 * (arm_ratio("b") - a)

    samples = [
        contrast(index)
        for index in _bootstrap_indices(count, draws=draws, seed=seed)
    ]
    boot = torch.cat(samples)
    interval = torch.quantile(
        boot,
        torch.tensor([0.025, 0.975], dtype=boot.dtype),
    ).tolist()
    if interval[0] >= 0.0:
        outcome = "REPRODUCED"
    elif interval[1] < 0.0:
        outcome = "NOT_REPRODUCED"
    else:
        outcome = "INCONCLUSIVE"
    return {
        "contrast": float(contrast()),
        "ci95": [float(interval[0]), float(interval[1])],
        "outcome": outcome,
        "paired_samples": int(count),
        "bootstrap_draws": int(draws),
        "seed": int(seed),
    }
