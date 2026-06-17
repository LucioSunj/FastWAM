from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Mapping, Protocol, Sequence

import torch


@dataclass(frozen=True)
class MetricResult:
    """Scalar routing metric plus optional debug metadata."""

    name: str
    value: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutingDecision:
    selected_branch: str
    metric: MetricResult
    threshold: float
    mode: str
    low_branch: str
    high_branch: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_branch": self.selected_branch,
            "metric_name": self.metric.name,
            "metric_value": self.metric.value,
            "metric_details": self.metric.details,
            "threshold": self.threshold,
            "mode": self.mode,
            "low_branch": self.low_branch,
            "high_branch": self.high_branch,
        }


class RoutingContext(Protocol):
    method_name: str
    kwargs: dict[str, Any]

    def run_branch(self, branch_name: str) -> dict[str, Any]:
        ...

    def run_branch_with_overrides(self, branch_name: str, **overrides: Any) -> dict[str, Any]:
        ...


class ThresholdSelector:
    """Select one of two branches by comparing a metric with a threshold."""

    _MODES: dict[str, Callable[[float, float], bool]] = {
        "ge": lambda value, threshold: value >= threshold,
        "gt": lambda value, threshold: value > threshold,
        "le": lambda value, threshold: value <= threshold,
        "lt": lambda value, threshold: value < threshold,
    }

    def __init__(
        self,
        threshold: float,
        low_branch: str = "base",
        high_branch: str = "idm",
        mode: str = "ge",
    ):
        if mode not in self._MODES:
            raise ValueError(f"Unsupported selector mode: {mode}. Expected one of {sorted(self._MODES)}.")
        self.threshold = float(threshold)
        self.low_branch = str(low_branch)
        self.high_branch = str(high_branch)
        self.mode = str(mode)

    def select(self, metric: MetricResult) -> RoutingDecision:
        use_high = self._MODES[self.mode](float(metric.value), self.threshold)
        selected = self.high_branch if use_high else self.low_branch
        return RoutingDecision(
            selected_branch=selected,
            metric=metric,
            threshold=self.threshold,
            mode=self.mode,
            low_branch=self.low_branch,
            high_branch=self.high_branch,
        )


class ExternalValueMetric:
    """Use a caller-provided scalar as the routing score.

    Pass the score through `routing_metric_value=...` when calling `infer_action`,
    `infer_joint`, or `infer`. This supports threshold-only routing without
    changing existing model or evaluation code.
    """

    def __init__(
        self,
        metric_key: str = "routing_metric_value",
        default_value: float | None = None,
        required: bool = False,
        name: str = "external_value",
        **_ignored,
    ):
        self.metric_key = str(metric_key)
        self.default_value = default_value
        self.required = bool(required)
        self.name = str(name)

    def __call__(self, context: RoutingContext) -> MetricResult:
        raw_value = context.kwargs.get(self.metric_key, self.default_value)
        if raw_value is None and self.required:
            raise ValueError(f"Missing required routing metric value: {self.metric_key}")
        value = 0.0 if raw_value is None else float(raw_value)
        return MetricResult(
            name=self.name,
            value=value,
            details={"metric_key": self.metric_key, "source": "caller" if raw_value is not None else "default"},
        )


class InputImageStdMetric:
    """Route by the standard deviation of the normalized input image tensor."""

    def __init__(self, name: str = "input_image_std", **_ignored):
        self.name = str(name)

    def __call__(self, context: RoutingContext) -> MetricResult:
        image = context.kwargs.get("input_image")
        if image is None:
            raise ValueError("InputImageStdMetric requires `input_image` in inference kwargs.")
        if not torch.is_tensor(image):
            raise TypeError(f"`input_image` must be a torch.Tensor, got {type(image)}.")
        value = float(image.detach().float().std().item())
        return MetricResult(name=self.name, value=value, details={"source": "input_image"})


class ActionOutputMetric:
    """Run a probe branch and compute a scalar metric from its predicted action.

    This supports routing from model output without modifying the existing
    FastWAM or FastWAMIDM implementations. If the selector chooses the same
    branch used for probing, the routed model reuses that cached output.
    """

    _STATISTICS = {
        "mean_abs",
        "max_abs",
        "std",
        "l2",
        "temporal_delta_mean_abs",
        "temporal_delta_l2",
    }

    def __init__(
        self,
        probe_branch: str = "base",
        statistic: str = "temporal_delta_mean_abs",
        name: str | None = None,
        **_ignored,
    ):
        if statistic not in self._STATISTICS:
            raise ValueError(f"Unsupported action statistic: {statistic}. Expected one of {sorted(self._STATISTICS)}.")
        self.probe_branch = str(probe_branch)
        self.statistic = str(statistic)
        self.name = name or f"{self.probe_branch}_action_{self.statistic}"

    def __call__(self, context: RoutingContext) -> MetricResult:
        output = context.run_branch(self.probe_branch)
        if "action" not in output:
            raise KeyError(
                f"Probe branch `{self.probe_branch}` output has no `action` key. "
                f"Available keys: {sorted(output.keys())}"
            )
        action = output["action"]
        if not torch.is_tensor(action):
            action = torch.as_tensor(action)
        action = action.detach().float()
        value = self._compute(action)
        return MetricResult(
            name=self.name,
            value=value,
            details={
                "probe_branch": self.probe_branch,
                "statistic": self.statistic,
                "action_shape": list(action.shape),
            },
        )

    def _compute(self, action: torch.Tensor) -> float:
        if self.statistic == "mean_abs":
            return float(action.abs().mean().item())
        if self.statistic == "max_abs":
            return float(action.abs().amax().item())
        if self.statistic == "std":
            return float(action.std().item())
        if self.statistic == "l2":
            return float(action.pow(2).mean().sqrt().item())

        if action.ndim == 3:
            time_dim = 1
        else:
            time_dim = 0
        if action.shape[time_dim] < 2:
            return 0.0
        delta = torch.diff(action, dim=time_dim)
        if self.statistic == "temporal_delta_mean_abs":
            return float(delta.abs().mean().item())
        if self.statistic == "temporal_delta_l2":
            return float(delta.pow(2).mean().sqrt().item())
        raise AssertionError(f"Unhandled statistic: {self.statistic}")


class PolicyEntropyMetric:
    """Estimate policy entropy by repeatedly sampling a probe branch.

    This follows the online part of DemoSpeedup's entropy signal: for one
    observation, sample several action chunks from a policy, estimate the local
    action density with a Gaussian kernel, and use the resulting conditional
    entropy as the routing score.
    """

    _REDUCTIONS = {"mean", "max", "min", "first", "last"}

    def __init__(
        self,
        probe_branch: str = "base",
        num_samples: int = 4,
        bandwidth: float | str = "silverman",
        min_bandwidth: float = 1.0e-3,
        leave_one_out: bool = True,
        reduction: str = "mean",
        normalize_by_dim: bool = True,
        action_dims: Sequence[int] | None = None,
        probe_overrides: Mapping[str, Any] | None = None,
        seed_base: int | None = None,
        seed_stride: int = 1009,
        fallback_seed: int = 0,
        output_key: str = "action",
        name: str | None = None,
        **_ignored,
    ):
        if num_samples < 2:
            raise ValueError("PolicyEntropyMetric requires `num_samples >= 2`.")
        if isinstance(bandwidth, str) and bandwidth != "silverman":
            raise ValueError("`bandwidth` must be a positive float or the string 'silverman'.")
        if not isinstance(bandwidth, str) and float(bandwidth) <= 0:
            raise ValueError("`bandwidth` must be positive.")
        if float(min_bandwidth) <= 0:
            raise ValueError("`min_bandwidth` must be positive.")
        if reduction not in self._REDUCTIONS:
            raise ValueError(f"Unsupported entropy reduction: {reduction}. Expected one of {sorted(self._REDUCTIONS)}.")
        if int(seed_stride) == 0:
            raise ValueError("`seed_stride` must be non-zero.")
        if action_dims is not None and len(action_dims) == 0:
            raise ValueError("`action_dims` must be null or contain at least one dimension index.")

        self.probe_branch = str(probe_branch)
        self.num_samples = int(num_samples)
        self.bandwidth = bandwidth
        self.min_bandwidth = float(min_bandwidth)
        self.leave_one_out = bool(leave_one_out)
        self.reduction = str(reduction)
        self.normalize_by_dim = bool(normalize_by_dim)
        self.action_dims = None if action_dims is None else tuple(int(index) for index in action_dims)
        self.probe_overrides = {} if probe_overrides is None else dict(probe_overrides)
        self.seed_base = None if seed_base is None else int(seed_base)
        self.seed_stride = int(seed_stride)
        self.fallback_seed = int(fallback_seed)
        self.output_key = str(output_key)
        self.name = name or f"{self.probe_branch}_policy_entropy"

    def __call__(self, context: RoutingContext) -> MetricResult:
        actions = []
        seeds = []
        for sample_index in range(self.num_samples):
            seed = self._sample_seed(context, sample_index)
            probe_kwargs = dict(self.probe_overrides)
            probe_kwargs["seed"] = seed
            output = context.run_branch_with_overrides(self.probe_branch, **probe_kwargs)
            if self.output_key not in output:
                raise KeyError(
                    f"Probe branch `{self.probe_branch}` output has no `{self.output_key}` key. "
                    f"Available keys: {sorted(output.keys())}"
                )
            actions.append(self._as_batched_action(output[self.output_key]))
            seeds.append(seed)

        samples = self._stack_action_samples(actions)
        entropy = self._entropy_per_observation_step(samples)
        value = self._reduce(entropy)
        bandwidth_summary = self._bandwidth_summary(samples)
        return MetricResult(
            name=self.name,
            value=value,
            details={
                "probe_branch": self.probe_branch,
                "num_samples": self.num_samples,
                "seeds": seeds,
                "probe_overrides": self.probe_overrides,
                "output_key": self.output_key,
                "action_shape": list(samples.shape[1:]),
                "bandwidth": bandwidth_summary,
                "leave_one_out": self.leave_one_out,
                "reduction": self.reduction,
                "normalize_by_dim": self.normalize_by_dim,
                "entropy_mean": float(entropy.mean().item()),
                "entropy_min": float(entropy.amin().item()),
                "entropy_max": float(entropy.amax().item()),
            },
        )

    def _sample_seed(self, context: RoutingContext, sample_index: int) -> int:
        if self.seed_base is not None:
            base_seed = self.seed_base
        else:
            context_seed = context.kwargs.get("seed")
            base_seed = self.fallback_seed if context_seed is None else int(context_seed)
        return int(base_seed + sample_index * self.seed_stride)

    def _as_batched_action(self, action: Any) -> torch.Tensor:
        if not torch.is_tensor(action):
            action = torch.as_tensor(action)
        action = action.detach().float()
        if action.ndim == 1:
            action = action.view(1, 1, -1)
        elif action.ndim == 2:
            action = action.unsqueeze(0)
        elif action.ndim == 3:
            pass
        else:
            action = action.reshape(action.shape[0], action.shape[1], -1)
        if self.action_dims is not None:
            action = action.index_select(-1, torch.tensor(self.action_dims, device=action.device))
        return action

    @staticmethod
    def _stack_action_samples(actions: list[torch.Tensor]) -> torch.Tensor:
        first_shape = actions[0].shape
        for index, action in enumerate(actions[1:], start=1):
            if action.shape != first_shape:
                raise ValueError(
                    "PolicyEntropyMetric requires all sampled action chunks to have the same shape. "
                    f"Sample 0 has {list(first_shape)}, sample {index} has {list(action.shape)}."
                )
        return torch.stack(actions, dim=0)

    def _entropy_per_observation_step(self, samples: torch.Tensor) -> torch.Tensor:
        # samples: [num_samples, batch, horizon, action_dim]
        num_samples, batch_size, horizon, action_dim = samples.shape
        groups = samples.permute(1, 2, 0, 3).reshape(batch_size * horizon, num_samples, action_dim)
        bandwidth = self._resolve_bandwidth(groups)

        diff = groups.unsqueeze(2) - groups.unsqueeze(1)
        dist2 = diff.pow(2).sum(dim=-1)
        log_kernel = -dist2 / (2.0 * bandwidth.square().view(-1, 1, 1))
        denom_count = num_samples
        if self.leave_one_out and num_samples > 1:
            diagonal = torch.eye(num_samples, dtype=torch.bool, device=samples.device).unsqueeze(0)
            log_kernel = log_kernel.masked_fill(diagonal, float("-inf"))
            denom_count = num_samples - 1

        log_norm = action_dim * torch.log(math.sqrt(2.0 * math.pi) * bandwidth)
        log_density = torch.logsumexp(log_kernel, dim=-1) - math.log(denom_count) - log_norm.view(-1, 1)
        entropy = -log_density.mean(dim=-1)
        if self.normalize_by_dim:
            entropy = entropy / float(action_dim)
        return entropy

    def _resolve_bandwidth(self, groups: torch.Tensor) -> torch.Tensor:
        if isinstance(self.bandwidth, str):
            num_samples = groups.shape[1]
            action_dim = groups.shape[2]
            scale = groups.std(dim=1, unbiased=False).mean(dim=-1)
            factor = (num_samples * (action_dim + 2.0) / 4.0) ** (-1.0 / (action_dim + 4.0))
            bandwidth = scale * factor
        else:
            bandwidth = torch.full(
                (groups.shape[0],),
                float(self.bandwidth),
                dtype=groups.dtype,
                device=groups.device,
            )
        return bandwidth.clamp_min(self.min_bandwidth)

    def _bandwidth_summary(self, samples: torch.Tensor) -> dict[str, float | str]:
        groups = samples.permute(1, 2, 0, 3).reshape(-1, samples.shape[0], samples.shape[-1])
        bandwidth = self._resolve_bandwidth(groups)
        kind = self.bandwidth if isinstance(self.bandwidth, str) else "fixed"
        return {
            "kind": kind,
            "mean": float(bandwidth.mean().item()),
            "min": float(bandwidth.amin().item()),
            "max": float(bandwidth.amax().item()),
        }

    def _reduce(self, entropy: torch.Tensor) -> float:
        if self.reduction == "mean":
            return float(entropy.mean().item())
        if self.reduction == "max":
            return float(entropy.amax().item())
        if self.reduction == "min":
            return float(entropy.amin().item())
        if self.reduction == "first":
            return float(entropy[0].item())
        if self.reduction == "last":
            return float(entropy[-1].item())
        raise AssertionError(f"Unhandled reduction: {self.reduction}")
