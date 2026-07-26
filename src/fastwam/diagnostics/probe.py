"""Linear read-out probe for the E7 three-point diagnostic.

Fits the *same* probe at several activation loci and reports where AUC collapses
under texture shift. See `fastwam.diagnostics.taps` for the extraction side and
for what each locus implies about the remedy.

Three properties are load-bearing and are all covered by tests:

**Grouped cross-validation is mandatory, not optional.** Adjacent frames of one
trajectory are near-duplicates. Splitting folds by sample lets a frame's own
neighbours sit in the training set, which inflates AUC toward 1.0 regardless of
whether the information is really there — exactly the false positive this
diagnostic exists to avoid. Every fold here is disjoint *by group*.

**Standardisation is fitted on the training fold only.** Computing feature means
over the full dataset leaks test-fold statistics into training and biases AUC
upward.

**Everything is deterministic.** Same inputs and seed give bitwise-identical
pooled features and identical folds, so a probe result can be re-derived rather
than trusted.

No third-party dependency beyond `torch` and `numpy`: the logistic regression is
penalised IRLS in float64, which for this problem size is both faster and more
reproducible than an iterative optimiser.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "DEFAULT_POOL_DIM",
    "POOL_LAYOUT",
    "ProbeResult",
    "cross_fitted_probe",
    "fit_logistic",
    "grouped_kfold",
    "pool_activation",
    "roc_auc",
]

DEFAULT_POOL_DIM = 64
POOL_LAYOUT = "mean_over_non_feature_axes_then_adaptive_avg_pool_v1"


# --------------------------------------------------------------------------- #
# Pooling
# --------------------------------------------------------------------------- #

def pool_activation(
    activation: torch.Tensor,
    *,
    feature_dim: int = -1,
    output_dim: int = DEFAULT_POOL_DIM,
) -> torch.Tensor:
    """Reduce ``[N, ...]`` activations to a deterministic ``[N, output_dim]``.

    Averages over every axis except the batch axis and `feature_dim`, then
    compresses the feature axis with `adaptive_avg_pool1d`. This mirrors the
    pooling already used for gate features
    (`fastwam/adaptive_gate/features.py:12`) so the two feature layouts stay
    comparable, and it is exactly reproducible.

    `feature_dim` indexes the full tensor including the batch axis, and must not
    be the batch axis itself; for ``[B, T, D]`` token streams pass ``-1``, for
    ``[B, C, T, H, W]`` video latents pass ``1``.
    """
    if not isinstance(activation, torch.Tensor):
        raise TypeError(
            f"activation must be a Tensor, got {type(activation).__name__}."
        )
    if activation.ndim < 2:
        raise ValueError(
            f"activation must be at least 2-D [N, ...], got shape "
            f"{tuple(activation.shape)}."
        )
    if int(output_dim) <= 0:
        raise ValueError(f"output_dim must be positive, got {output_dim}.")

    ndim = activation.ndim
    axis = feature_dim + ndim if feature_dim < 0 else feature_dim
    if not 0 <= axis < ndim:
        raise ValueError(
            f"feature_dim={feature_dim} is out of range for shape "
            f"{tuple(activation.shape)}."
        )
    if axis == 0:
        raise ValueError(
            "feature_dim must not be the batch axis (0); the batch axis is never "
            "pooled."
        )

    x = activation.float()
    x = x.movedim(axis, -1)                       # [N, ..., F]
    if x.ndim > 2:
        x = x.mean(dim=tuple(range(1, x.ndim - 1)))  # [N, F]
    if not torch.isfinite(x).all():
        raise ValueError(
            "activation contains non-finite values after reduction; refusing to "
            "pool. Check the tap and the forward pass that produced it."
        )
    pooled = F.adaptive_avg_pool1d(x.unsqueeze(1), int(output_dim)).squeeze(1)
    return pooled.contiguous()


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def roc_auc(y_true: Sequence[int] | np.ndarray, scores: Sequence[float] | np.ndarray) -> float:
    """Tie-aware ROC AUC via the Mann-Whitney statistic.

    Returns NaN when either class is absent, which is a real outcome for a small
    held-out fold and must not be silently reported as 0.5.
    """
    y = np.asarray(y_true).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    if y.shape != s.shape:
        raise ValueError(f"shape mismatch: y={y.shape} scores={s.shape}")
    if y.size == 0:
        return float("nan")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("roc_auc expects binary labels in {0, 1}.")

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(s.size, dtype=np.float64)
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0  # 1-based average rank
        i = j + 1

    rank_sum_pos = float(ranks[y == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# --------------------------------------------------------------------------- #
# Grouped cross-validation
# --------------------------------------------------------------------------- #

def _group_labels(g: np.ndarray, y: np.ndarray, unique: np.ndarray) -> dict:
    """Map each group to its single label, refusing to guess if it is not single."""
    mapping: dict = {}
    for value in unique:
        present = np.unique(y[g == value])
        if present.size != 1:
            raise ValueError(
                f"group {value!r} carries labels {present.tolist()}. Stratified "
                "grouped folds need one label per group -- in E7 the label is "
                "episode success, which is constant along a trajectory, so a "
                "mixed group means the labels and groups arrays are misaligned."
            )
        mapping[value.item()] = int(present[0])
    return mapping


def grouped_kfold(
    groups: Sequence | np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 0,
    labels: Sequence[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Deterministic per-sample fold assignment, disjoint by group.

    Every sample sharing a group lands in the same fold, so no trajectory is ever
    split across the train/test boundary.

    When `labels` is given the folds are additionally **stratified** by the
    group's label, dealing each class's groups round-robin so every fold gets a
    near-identical class ratio. That is not a refinement -- it is required for the
    result to mean anything:

    * An unstratified fold is routinely class-imbalanced, its training folds
      carry the opposite imbalance, the fitted intercept shifts, and every score
      in that fold moves together. Per-fold AUC is immune (a constant shift does
      not reorder) but the pooled out-of-fold AUC is not. Measured on pure noise
      with 40 groups, unstratified pooling settled at a *stable* ~0.34 across
      20/40/60 groups -- reproducible enough to be mistaken for signal.
    * Unstratified folds also go entirely single-class often enough to make
      per-fold AUC undefined, which throws away folds.

    Correcting the scores after the fact does not work: standardising within fold
    removes the offset, but when a fold really is class-pure that offset *is* the
    signal, and the measured AUC on genuinely separable data drops from 1.00 to
    0.89. Balancing the folds fixes the cause instead.
    """
    g = np.asarray(groups).reshape(-1)
    if g.size == 0:
        raise ValueError("groups is empty.")
    n_splits = int(n_splits)
    if n_splits < 2:
        raise ValueError(f"n_splits must be >= 2, got {n_splits}.")

    unique = np.unique(g)  # np.unique sorts -> assignment is seed-only, not input-order
    if unique.size < n_splits:
        raise ValueError(
            f"cannot build {n_splits} group-disjoint folds from {unique.size} "
            "distinct group(s). Collect more trajectories or reduce n_splits."
        )

    rng = np.random.default_rng(seed)
    fold_of_group: dict = {}

    if labels is None:
        permuted = rng.permutation(unique.size)
        bounds = np.linspace(0, unique.size, n_splits + 1).astype(int)
        for fold in range(n_splits):
            for pos in permuted[bounds[fold] : bounds[fold + 1]]:
                fold_of_group[unique[pos].item()] = fold
    else:
        y = np.asarray(labels).reshape(-1)
        if y.size != g.size:
            raise ValueError(
                f"labels has {y.size} entries but groups has {g.size}."
            )
        per_group = _group_labels(g, y, unique)
        for cls in sorted(set(per_group.values())):
            members = np.array(
                [v for v in unique if per_group[v.item()] == cls], dtype=unique.dtype
            )
            if members.size < n_splits:
                raise ValueError(
                    f"class {cls} has only {members.size} group(s); {n_splits} "
                    "stratified folds would leave folds without it. Collect more "
                    "trajectories of that class or reduce n_splits."
                )
            members = members[rng.permutation(members.size)]
            for offset, value in enumerate(members):
                fold_of_group[value.item()] = offset % n_splits

    return np.array([fold_of_group[v.item()] for v in g], dtype=np.int64)


# --------------------------------------------------------------------------- #
# Penalised logistic regression (IRLS, float64)
# --------------------------------------------------------------------------- #

def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Branch on sign so neither exp overflows; plain 1/(1+exp(-z)) overflows for
    # z <~ -710 and separable data reaches that quickly.
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1.0,
    max_iter: int = 100,
    tol: float = 1e-9,
) -> np.ndarray:
    """Ridge-penalised logistic regression by IRLS. Returns weights of size d+1.

    A bias column is appended internally and is **not** penalised. With ``l2 > 0``
    the objective is strictly convex, so perfectly separable data converges to a
    finite optimum instead of diverging.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D [n, d], got shape {X.shape}.")
    if X.shape[0] != y.size:
        raise ValueError(f"X has {X.shape[0]} rows but y has {y.size} entries.")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("fit_logistic expects binary labels in {0, 1}.")
    if len(np.unique(y)) < 2:
        raise ValueError(
            "fit_logistic needs both classes present in the training split; got "
            f"only label {np.unique(y).tolist()}. With grouped folds this usually "
            "means one group holds every positive - collect more trajectories."
        )
    if float(l2) <= 0.0:
        raise ValueError(
            f"l2 must be > 0 for a strictly convex objective, got {l2}."
        )

    n, d = X.shape
    design = np.hstack([X, np.ones((n, 1), dtype=np.float64)])
    penalty = np.full(d + 1, float(l2), dtype=np.float64)
    penalty[-1] = 0.0  # never shrink the intercept
    w = np.zeros(d + 1, dtype=np.float64)

    for _ in range(int(max_iter)):
        eta = design @ w
        p = _sigmoid(eta)
        # Floor the IRLS weights: p(1-p) underflows to 0 once the model is
        # confident, which would make the normal equations singular.
        s = np.clip(p * (1.0 - p), 1e-10, None)
        z = eta + (y - p) / s
        lhs = design.T @ (design * s[:, None]) + np.diag(penalty)
        rhs = design.T @ (s * z)
        try:
            w_new = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:  # pragma: no cover - defensive
            w_new = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        if not np.all(np.isfinite(w_new)):  # pragma: no cover - defensive
            raise FloatingPointError("IRLS produced non-finite weights.")
        delta = np.max(np.abs(w_new - w))
        w = w_new
        if delta < tol:
            break
    return w


def _standardise_params(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)  # constant features -> pass through as 0
    return mean, std


# --------------------------------------------------------------------------- #
# Cross-fitted probe
# --------------------------------------------------------------------------- #

@dataclass
class ProbeResult:
    """`auc` is the pooled out-of-fold AUC over stratified group-disjoint folds.

    `mean_fold_auc` averages the per-fold AUCs and is reported as an independent
    cross-check. Stratification keeps every fold class-balanced, so the two
    should agree closely; a large gap means the folds are not balanced and the
    pooled number should not be trusted.
    """

    auc: float
    ci_low: float
    ci_high: float
    mean_fold_auc: float
    n_samples: int
    n_positive: int
    n_groups: int
    n_splits: int
    l2: float
    seed: int
    per_fold_auc: list = field(default_factory=list)
    oof_scores: list = field(default_factory=list)

    def to_dict(self, *, include_scores: bool = False) -> dict:
        payload = asdict(self)
        if not include_scores:
            payload.pop("oof_scores")
        return payload


def _bootstrap_group_ci(
    y: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    *,
    seed: int,
    n_bootstrap: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile CI, resampling **groups** rather than samples.

    Resampling individual frames would treat correlated neighbours as independent
    evidence and give an interval that is far too narrow.
    """
    if n_bootstrap <= 0:
        return (float("nan"), float("nan"))
    unique = np.unique(groups)
    index_of = {g.item(): np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(int(n_bootstrap)):
        picked = rng.integers(0, unique.size, size=unique.size)
        idx = np.concatenate([index_of[unique[i].item()] for i in picked])
        value = roc_auc(y[idx], scores[idx])
        if not np.isnan(value):
            draws.append(value)
    if len(draws) < 2:
        return (float("nan"), float("nan"))
    return (
        float(np.percentile(draws, 100 * alpha / 2)),
        float(np.percentile(draws, 100 * (1 - alpha / 2))),
    )


def cross_fitted_probe(
    features: np.ndarray | torch.Tensor,
    labels: Sequence[int] | np.ndarray,
    groups: Sequence | np.ndarray,
    *,
    n_splits: int = 5,
    l2: float = 1.0,
    seed: int = 0,
    n_bootstrap: int = 1000,
) -> ProbeResult:
    """Group-disjoint cross-fitted linear probe with an out-of-fold AUC."""
    X = features.detach().cpu().numpy() if isinstance(features, torch.Tensor) else np.asarray(features)
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(labels).reshape(-1).astype(np.int64)
    g = np.asarray(groups).reshape(-1)
    if X.ndim != 2:
        raise ValueError(f"features must be 2-D [n, d], got {X.shape}.")
    if not (X.shape[0] == y.size == g.size):
        raise ValueError(
            f"length mismatch: features={X.shape[0]} labels={y.size} groups={g.size}"
        )
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be binary {0, 1}.")

    fold = grouped_kfold(g, n_splits=n_splits, seed=seed, labels=y)
    oof = np.full(y.size, np.nan, dtype=np.float64)
    per_fold: list[float] = []

    for f in range(n_splits):
        test_mask = fold == f
        train_mask = ~test_mask
        if not test_mask.any():  # pragma: no cover - grouped_kfold prevents this
            per_fold.append(float("nan"))
            continue
        if len(np.unique(y[train_mask])) < 2:
            raise ValueError(
                f"fold {f}: the training split holds a single class. Grouped folds "
                "keep whole trajectories together, so this means the positives are "
                "concentrated in too few groups. Collect more trajectories or "
                "lower n_splits."
            )
        mean, std = _standardise_params(X[train_mask])
        w = fit_logistic((X[train_mask] - mean) / std, y[train_mask], l2=l2)
        scores = np.hstack(
            [(X[test_mask] - mean) / std, np.ones((int(test_mask.sum()), 1))]
        ) @ w
        oof[test_mask] = scores
        per_fold.append(roc_auc(y[test_mask], scores))

    if np.isnan(oof).any():  # pragma: no cover - defensive
        raise RuntimeError("some samples never received an out-of-fold score.")

    auc = roc_auc(y, oof)
    ci_low, ci_high = _bootstrap_group_ci(
        y, oof, g, seed=seed, n_bootstrap=n_bootstrap
    )
    finite_folds = [v for v in per_fold if not np.isnan(v)]
    return ProbeResult(
        auc=float(auc),
        ci_low=ci_low,
        ci_high=ci_high,
        mean_fold_auc=float(np.mean(finite_folds)) if finite_folds else float("nan"),
        n_samples=int(y.size),
        n_positive=int((y == 1).sum()),
        n_groups=int(np.unique(g).size),
        n_splits=int(n_splits),
        l2=float(l2),
        seed=int(seed),
        per_fold_auc=[float(v) for v in per_fold],
        oof_scores=[float(v) for v in oof],
    )


def probe_taps(
    activations: Mapping[str, torch.Tensor | np.ndarray],
    labels: Sequence[int] | np.ndarray,
    groups: Sequence | np.ndarray,
    *,
    feature_dims: Mapping[str, int] | None = None,
    output_dim: int = DEFAULT_POOL_DIM,
    **kwargs,
) -> dict[str, ProbeResult]:
    """Pool and probe several taps with one shared configuration."""
    feature_dims = dict(feature_dims or {})
    results: dict[str, ProbeResult] = {}
    for name, activation in activations.items():
        tensor = (
            activation
            if isinstance(activation, torch.Tensor)
            else torch.as_tensor(np.asarray(activation))
        )
        pooled = pool_activation(
            tensor,
            feature_dim=feature_dims.get(name, -1),
            output_dim=output_dim,
        )
        results[name] = cross_fitted_probe(pooled, labels, groups, **kwargs)
    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _parse_feature_dims(values: Sequence[str] | None) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"--feature-dim expects NAME=INT, got {item!r}")
        name, _, raw = item.partition("=")
        try:
            parsed[name] = int(raw)
        except ValueError as exc:
            raise SystemExit(f"--feature-dim {item!r}: {raw!r} is not an int") from exc
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fastwam.diagnostics.probe",
        description=(
            "Fit a group-disjoint cross-fitted linear probe at one or more tap "
            "points. Extraction and fitting are decoupled on purpose: activations "
            "are captured once (GPU) and probed repeatedly (CPU)."
        ),
    )
    parser.add_argument("--activations", required=True, type=Path,
                        help="npz with one [N, ...] array per tap name")
    parser.add_argument("--labels", required=True, type=Path,
                        help="npz with 'y' [N] binary labels and 'groups' [N] trajectory ids")
    parser.add_argument("--tap", action="append", default=None,
                        help="restrict to this tap (repeatable); default is all")
    parser.add_argument("--feature-dim", action="append", default=None,
                        metavar="NAME=INT", help="feature axis per tap (default -1)")
    parser.add_argument("--output-dim", type=int, default=DEFAULT_POOL_DIM)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=None, help="write JSON here")
    args = parser.parse_args(argv)

    with np.load(args.activations) as bundle:
        available = list(bundle.files)
        wanted = args.tap or available
        missing = sorted(set(wanted) - set(available))
        if missing:
            parser.error(
                f"tap(s) {missing} not in {args.activations}. Available: {available}"
            )
        activations = {name: bundle[name] for name in wanted}

    with np.load(args.labels) as bundle:
        for key in ("y", "groups"):
            if key not in bundle.files:
                parser.error(
                    f"{args.labels} must contain '{key}'; found {list(bundle.files)}"
                )
        y = bundle["y"]
        groups = bundle["groups"]

    results = probe_taps(
        activations,
        y,
        groups,
        feature_dims=_parse_feature_dims(args.feature_dim),
        output_dim=args.output_dim,
        n_splits=args.folds,
        l2=args.l2,
        seed=args.seed,
        n_bootstrap=args.bootstrap,
    )

    payload = {
        "pool_layout": POOL_LAYOUT,
        "output_dim": int(args.output_dim),
        "activations_path": str(args.activations),
        "labels_path": str(args.labels),
        "taps": {name: result.to_dict() for name, result in results.items()},
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
