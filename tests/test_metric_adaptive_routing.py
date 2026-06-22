"""Tier-1 pure-logic tests for metric-adaptive routing.

These need only `torch` (no Wan weights, no GPU, no datasets) and exercise the
routing metric / selector layer used by `MetricAdaptiveFastWAM` at inference.

Run:
    cd FastWAM
    pytest tests/test_metric_adaptive_routing.py -v
"""
from __future__ import annotations

import math

import pytest
import torch

from fastwam.routing.metrics import (
    ActionOutputMetric,
    ExternalValueMetric,
    InputImageStdMetric,
    MetricResult,
    PolicyEntropyMetric,
    ThresholdSelector,
)


class FakeContext:
    """Minimal RoutingContext stand-in (matches the Protocol in metrics.py)."""

    def __init__(self, kwargs=None, branch_outputs=None):
        self.method_name = "infer_action"
        self.kwargs = dict(kwargs or {})
        self._branch_outputs = dict(branch_outputs or {})
        self.branch_calls = []
        self.override_calls = []

    def run_branch(self, name):
        self.branch_calls.append(name)
        return self._branch_outputs[name]

    def run_branch_with_overrides(self, name, **overrides):
        self.override_calls.append((name, overrides))
        return self._branch_outputs[name]


# --------------------------------------------------------------------------- #
# ThresholdSelector
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "mode,value,threshold,expected",
    [
        ("ge", 0.5, 0.0, "idm"),
        ("ge", 0.0, 0.0, "idm"),   # >= boundary -> high branch
        ("ge", -0.1, 0.0, "base"),
        ("gt", 0.0, 0.0, "base"),  # not strictly greater -> low branch
        ("le", 0.0, 0.0, "idm"),
        ("lt", 0.0, 0.0, "base"),
    ],
)
def test_threshold_selector_modes(mode, value, threshold, expected):
    sel = ThresholdSelector(threshold=threshold, mode=mode)
    decision = sel.select(MetricResult(name="m", value=value))
    assert decision.selected_branch == expected
    assert decision.as_dict()["selected_branch"] == expected


def test_threshold_selector_custom_branches():
    sel = ThresholdSelector(threshold=1.0, low_branch="A", high_branch="B", mode="ge")
    assert sel.select(MetricResult("m", 2.0)).selected_branch == "B"
    assert sel.select(MetricResult("m", 0.0)).selected_branch == "A"


def test_threshold_selector_rejects_unknown_mode():
    with pytest.raises(ValueError):
        ThresholdSelector(threshold=0.0, mode="nope")


# --------------------------------------------------------------------------- #
# ExternalValueMetric
# --------------------------------------------------------------------------- #
def test_external_value_from_kwargs():
    metric = ExternalValueMetric()
    result = metric(FakeContext(kwargs={"routing_metric_value": 0.8}))
    assert result.value == pytest.approx(0.8)


def test_external_value_default_when_missing():
    metric = ExternalValueMetric(default_value=0.3)
    assert metric(FakeContext()).value == pytest.approx(0.3)


def test_external_value_required_raises():
    metric = ExternalValueMetric(required=True)
    with pytest.raises(ValueError):
        metric(FakeContext())


# --------------------------------------------------------------------------- #
# InputImageStdMetric
# --------------------------------------------------------------------------- #
def test_input_image_std_matches_tensor_std():
    image = torch.randn(1, 3, 16, 16)
    result = InputImageStdMetric()(FakeContext(kwargs={"input_image": image}))
    assert result.value == pytest.approx(float(image.float().std().item()), rel=1e-4)


def test_input_image_std_missing_raises():
    with pytest.raises(ValueError):
        InputImageStdMetric()(FakeContext())


# --------------------------------------------------------------------------- #
# ActionOutputMetric
# --------------------------------------------------------------------------- #
def test_action_output_mean_abs():
    action = torch.tensor([[[-2.0], [2.0]]])  # [B=1, T=2, a=1] -> mean abs = 2.0
    metric = ActionOutputMetric(probe_branch="base", statistic="mean_abs")
    result = metric(FakeContext(branch_outputs={"base": {"action": action}}))
    assert result.value == pytest.approx(2.0)
    # The probe branch must have been executed exactly once.
    # (ActionOutputMetric uses run_branch, not the override variant.)


def test_action_output_temporal_delta_mean_abs():
    action = torch.tensor([[[0.0], [3.0], [3.0]]])  # diffs 3,0 -> mean abs 1.5
    metric = ActionOutputMetric(statistic="temporal_delta_mean_abs")
    result = metric(FakeContext(branch_outputs={"base": {"action": action}}))
    assert result.value == pytest.approx(1.5)


def test_action_output_rejects_unknown_statistic():
    with pytest.raises(ValueError):
        ActionOutputMetric(statistic="nope")


def test_action_output_missing_action_key_raises():
    metric = ActionOutputMetric(probe_branch="base", statistic="mean_abs")
    with pytest.raises(KeyError):
        metric(FakeContext(branch_outputs={"base": {"not_action": torch.zeros(1)}}))


# --------------------------------------------------------------------------- #
# PolicyEntropyMetric (plumbing; KDE value is not asserted exactly)
# --------------------------------------------------------------------------- #
def test_policy_entropy_runs_and_probes_num_samples_times():
    action = torch.randn(1, 4, 3)  # [B, horizon, a_dim]
    ctx = FakeContext(kwargs={"seed": 0}, branch_outputs={"base": {"action": action}})
    metric = PolicyEntropyMetric(probe_branch="base", num_samples=4)
    result = metric(ctx)
    assert math.isfinite(result.value)
    assert len(ctx.override_calls) == 4          # one probe sample per draw
    assert result.name == "base_policy_entropy"
    assert result.details["num_samples"] == 4


def test_policy_entropy_requires_two_samples():
    with pytest.raises(ValueError):
        PolicyEntropyMetric(num_samples=1)


def test_policy_entropy_action_dims_subset():
    action = torch.randn(1, 4, 7)
    ctx = FakeContext(kwargs={"seed": 1}, branch_outputs={"base": {"action": action}})
    metric = PolicyEntropyMetric(num_samples=3, action_dims=[0, 1, 2])
    result = metric(ctx)
    assert math.isfinite(result.value)
    # action_shape recorded in details reflects the 3 selected dims.
    assert result.details["action_shape"][-1] == 3
