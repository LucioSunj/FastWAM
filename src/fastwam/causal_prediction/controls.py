"""Deterministic generic-compute controls for causal interventions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from .contracts import CausalControlKind

GENERIC_PROPOSAL_GRID = (2, 3, 4, 5, 6, 8)


def derive_generic_proposal_seed(base_seed: int, proposal_index: int) -> int:
    """Derive proposal seeds while preserving candidate zero as the paired seed."""

    if base_seed < 0 or proposal_index < 0:
        raise ValueError("Proposal seed inputs must be non-negative.")
    return int(base_seed) + 104_729 * int(proposal_index)


def normalized_action_medoid(proposals: torch.Tensor) -> tuple[int, torch.Tensor]:
    """Return the deterministic medoid of ``[proposal, time, action]`` chunks."""

    if proposals.ndim != 3 or proposals.shape[0] < 1:
        raise ValueError("Action proposals must have shape [M,T,D] with M >= 1.")
    if not torch.isfinite(proposals).all():
        raise ValueError("Action proposals must be finite.")
    flattened = proposals.detach().float().reshape(proposals.shape[0], -1)
    distances = torch.cdist(flattened, flattened, p=2).square()
    mean_distance = distances.mean(dim=1)
    # torch.argmin returns the first minimum, which is the frozen tie-break.
    index = int(torch.argmin(mean_distance).item())
    return index, mean_distance


def select_latency_matched_proposal_count(
    *,
    c2_latency_ms: Sequence[float],
    proposal_latency_ms: Mapping[int, Sequence[float]],
) -> dict[str, float | int]:
    """Choose the preregistered proposal count using canary latency only."""

    if set(proposal_latency_ms) != set(GENERIC_PROPOSAL_GRID):
        raise ValueError("Generic proposal latency grid changed from preregistration.")
    c2 = torch.as_tensor(c2_latency_ms, dtype=torch.float64)
    if c2.numel() < 1 or not torch.isfinite(c2).all() or bool((c2 < 0).any()):
        raise ValueError("C2 canary latency must be finite and non-negative.")
    target = float(c2.median().item())
    candidates: list[tuple[float, int, float]] = []
    for count in GENERIC_PROPOSAL_GRID:
        values = torch.as_tensor(proposal_latency_ms[count], dtype=torch.float64)
        if (
            values.numel() < 1
            or not torch.isfinite(values).all()
            or bool((values < 0).any())
        ):
            raise ValueError("Proposal canary latency must be finite and non-negative.")
        median = float(values.median().item())
        candidates.append((abs(median - target), count, median))
    gap, count, median = min(candidates)
    return {
        "proposal_count": count,
        "c2_median_latency_ms": target,
        "proposal_median_latency_ms": median,
        "absolute_latency_gap_ms": gap,
    }


def compose_controlled_video_latents(
    *,
    first_frame: torch.Tensor,
    generated_full: torch.Tensor,
    control: CausalControlKind | str,
    donor_future: torch.Tensor | None = None,
    ground_truth_future: torch.Tensor | None = None,
) -> torch.Tensor:
    """Replace future slots while preserving the recipient current prefix exactly."""

    selected = CausalControlKind.parse(control)
    if generated_full.ndim != 5 or first_frame.shape != generated_full[:, :, :1].shape:
        raise ValueError("Video latent prefix/full shapes are incompatible.")
    if not torch.equal(first_frame, generated_full[:, :, :1]):
        raise ValueError("Generated video does not preserve the recipient prefix.")
    if selected is CausalControlKind.REPEAT_CURRENT:
        result = first_frame.expand_as(generated_full).clone()
    elif selected in {
        CausalControlKind.SHUFFLED_WRONG_STATE,
        CausalControlKind.TEMPORAL_SHIFT,
    }:
        if donor_future is None or donor_future.shape != generated_full[:, :, 1:].shape:
            raise ValueError("Wrong-future control requires a shape-matched donor.")
        result = torch.cat(
            (first_frame, donor_future.to(first_frame.device, first_frame.dtype)),
            dim=2,
        )
    elif selected is CausalControlKind.GT_FUTURE_OFFLINE:
        if (
            ground_truth_future is None
            or ground_truth_future.shape != generated_full[:, :, 1:].shape
        ):
            raise ValueError(
                "GT-future control requires shape-matched privileged latents."
            )
        result = torch.cat(
            (
                first_frame,
                ground_truth_future.to(first_frame.device, first_frame.dtype),
            ),
            dim=2,
        )
    else:
        result = generated_full
    if not torch.equal(result[:, :, :1], first_frame):
        raise AssertionError("A future control changed the recipient current prefix.")
    return result
