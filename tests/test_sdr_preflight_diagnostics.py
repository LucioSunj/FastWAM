"""S-DR preflight gradients must include effective-batch cross terms."""
from __future__ import annotations

import math

import pytest
import torch

from fastwam.adaptive_gate.sdr_preflight import (
    ExactGradientAccumulator,
    check_loss_arithmetic,
    couple_action_noise_draws,
    select_preflight_weights,
    weighted_descent_margins,
)


def _stats(idm, uncond):
    idm_tensor = torch.tensor(idm, dtype=torch.float64)
    uncond_tensor = torch.tensor(uncond, dtype=torch.float64)
    return {
        "dot": float(torch.dot(idm_tensor, uncond_tensor)),
        "idm_sq": float(torch.dot(idm_tensor, idm_tensor)),
        "uncond_sq": float(torch.dot(uncond_tensor, uncond_tensor)),
        "idm_norm": float(torch.linalg.vector_norm(idm_tensor)),
        "uncond_norm": float(torch.linalg.vector_norm(uncond_tensor)),
    }


@pytest.mark.parametrize(
    ("idm", "uncond", "expected_cosine"),
    [
        ([1.0, 2.0], [1.0, 2.0], 1.0),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([1.0, 0.0], [-0.5, 1.0], -1.0 / math.sqrt(5.0)),
        ([1.0, -2.0], [-1.0, 2.0], -1.0),
    ],
)
def test_weighted_margins_cover_alignment_cases(idm, uncond, expected_cosine):
    statistics = _stats(idm, uncond)
    denom = math.sqrt(statistics["idm_sq"] * statistics["uncond_sq"])
    assert statistics["dot"] / denom == pytest.approx(expected_cosine)
    result = weighted_descent_margins(statistics, 0.2)
    assert math.isfinite(result["idm_margin"])
    assert math.isfinite(result["uncond_margin"])


def test_weighted_margins_zero_norm_and_conflict_interval():
    zero = weighted_descent_margins(_stats([0.0, 0.0], [1.0, 0.0]), 0.2)
    assert zero["idm_margin"] == 0.0
    assert math.isfinite(zero["normalized_idm_margin"])

    opposing = weighted_descent_margins(_stats([2.0], [-1.0]), 1.0)
    interval = opposing["simultaneous_descent_interval"]
    assert interval == {"kind": "conflicting", "lower": 2.0, "upper": 2.0}
    assert opposing["idm_margin"] > 0.0
    assert opposing["uncond_margin"] < 0.0


@pytest.mark.parametrize("weight", [-1.0, float("inf"), float("nan")])
def test_weighted_margins_reject_invalid_weight(weight):
    with pytest.raises(ValueError, match="weight"):
        weighted_descent_margins(_stats([1.0], [1.0]), weight)


def test_exact_accumulator_includes_cross_microbatch_terms():
    parameter = torch.nn.Parameter(torch.zeros(2))
    accumulator = ExactGradientAccumulator({"action_all": [parameter]})
    accumulator.start_shard(0)

    idm_loss = (parameter * torch.tensor([1.0, 0.0])).sum()
    uncond_loss = (parameter * torch.tensor([0.0, 1.0])).sum()
    accumulator.accumulate(idm_loss, uncond_loss)

    idm_loss = (parameter * torch.tensor([0.0, 1.0])).sum()
    uncond_loss = (parameter * torch.tensor([1.0, 0.0])).sum()
    accumulator.accumulate(idm_loss, uncond_loss)
    accumulator.finish_shard()
    result = accumulator.finalize()

    group = result["groups"]["action_all"]
    assert result["cross_microbatch_terms_included"] is True
    assert group["dot"] == pytest.approx(0.5)
    assert group["idm_sq"] == pytest.approx(0.5)
    assert group["uncond_sq"] == pytest.approx(0.5)
    assert group["dot"] != 0.0


def test_common_noise_replay_clones_main_draw_without_mutating_source():
    original = {
        "action_regimes": [
            {
                "name": "idm",
                "noise": torch.tensor([1.0]),
                "timestep": torch.tensor([2.0]),
            },
            {
                "name": "base",
                "noise": torch.tensor([3.0]),
                "timestep": torch.tensor([4.0]),
            },
        ]
    }
    common = couple_action_noise_draws(original, mode="common")
    assert torch.equal(
        common["action_regimes"][0]["noise"],
        common["action_regimes"][1]["noise"],
    )
    assert torch.equal(
        common["action_regimes"][0]["timestep"],
        common["action_regimes"][1]["timestep"],
    )
    assert original["action_regimes"][1]["noise"].item() == 3.0

    independent = couple_action_noise_draws(original, mode="independent")
    assert independent["action_regimes"][0]["noise"].item() == 1.0
    assert independent["action_regimes"][1]["noise"].item() == 3.0


def test_preflight_selection_uses_weighted_shard_margins():
    groups = {
        "action_all": _stats([2.0, 0.0], [1.0, 0.0]),
        "action_blocks_final": _stats([2.0, 0.0], [1.0, 0.0]),
    }
    diagnostics = {
        "groups": groups,
        "shards": [
            {"groups": groups, "shard_index": index, "sample_count": 1}
            for index in range(8)
        ],
    }
    decision = select_preflight_weights(diagnostics)
    assert decision["go"] is True
    assert decision["w0"] == pytest.approx(0.05)
    assert decision["w_cap"] == pytest.approx(0.5)
    assert decision["schedule"][-1] == [1.0, 0.5]
    assert "1.0" in decision["candidate_margins"]


def test_loss_arithmetic_reports_weighted_contributions():
    result = check_loss_arithmetic(
        idm_raw=2.0,
        uncond_raw=4.0,
        weight=0.5,
        idm_contribution=2.0 / 1.5,
        uncond_contribution=2.0 / 1.5,
        combined=4.0 / 1.5,
    )
    assert result["pass"] is True
    failed = check_loss_arithmetic(
        idm_raw=2.0,
        uncond_raw=4.0,
        weight=0.5,
        idm_contribution=0.0,
        uncond_contribution=0.0,
        combined=0.0,
    )
    assert failed["pass"] is False
