"""Contract tests for the E7 three-point probe harness.

Everything here runs on synthetic tensors and a toy `nn.Module`. No Wan weights,
no GPU, no simulator, no dataset -- the harness is meant to be trustworthy before
it is ever pointed at a real checkpoint.

The load-bearing test is `test_grouped_folds_prevent_the_leak_that_ungrouped_folds_cause`:
it builds data whose label is predictable *only* by memorising a trajectory
signature, and shows that per-sample folds report a near-perfect AUC while
group-disjoint folds correctly report chance. That is the exact false positive
this diagnostic exists to avoid, so it is asserted rather than assumed.
"""

import json
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from fastwam.diagnostics import (  # noqa: E402
    ActivationTaps,
    TapSpec,
    cross_fitted_probe,
    fit_logistic,
    grouped_kfold,
    pool_activation,
    probe_taps,
    resolve_module,
    roc_auc,
)


# --------------------------------------------------------------------------- #
# Toy model standing in for the three real loci
# --------------------------------------------------------------------------- #

class _FakeVAE(nn.Module):
    """Emits a [N, C, T, H, W] latent, like the real first-frame VAE output."""

    def __init__(self, channels=6):
        super().__init__()
        self.proj = nn.Linear(8, channels * 2 * 3 * 3)
        self.channels = channels

    def forward(self, x):
        flat = self.proj(x)
        return flat.view(x.shape[0], self.channels, 2, 3, 3)


class _FakeBlock(nn.Module):
    def __init__(self, dim=12):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x):
        return torch.tanh(self.lin(x))


class _FakeStack(nn.Module):
    """vae -> video blocks -> action readout, returning a dict like MoT does."""

    def __init__(self):
        super().__init__()
        self.vae = _FakeVAE()
        self.to_tokens = nn.Linear(6 * 2 * 3 * 3, 12)
        self.video = nn.ModuleList([_FakeBlock(), _FakeBlock(), _FakeBlock()])
        self.readout = nn.Linear(12, 5)

    def forward(self, x):
        latent = self.vae(x)
        tokens = self.to_tokens(latent.flatten(1)).unsqueeze(1).repeat(1, 4, 1)
        for block in self.video:
            tokens = block(tokens)
        return {"action": self.readout(tokens), "video": tokens}


def _three_point_specs():
    return [
        TapSpec("vae_latent", "vae", feature_dim=1),
        TapSpec("video_block_1", "video.1", feature_dim=-1),
        TapSpec("action_readout", "readout", site="input", feature_dim=-1),
    ]


def _hook_counts(model):
    return sum(
        len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules()
    )


# --------------------------------------------------------------------------- #
# 1. Taps: capture, cleanup, and loud failure
# --------------------------------------------------------------------------- #

def test_taps_capture_all_three_loci():
    torch.manual_seed(0)
    model = _FakeStack().eval()
    x = torch.randn(7, 8)

    with ActivationTaps(model, _three_point_specs()) as taps:
        with torch.no_grad():
            model(x)
        assert taps.active

    assert sorted(taps.names) == ["action_readout", "vae_latent", "video_block_1"]
    assert taps.counts() == {
        "vae_latent": 1,
        "video_block_1": 1,
        "action_readout": 1,
    }
    assert tuple(taps.stack("vae_latent").shape) == (7, 6, 2, 3, 3)
    assert tuple(taps.stack("video_block_1").shape) == (7, 4, 12)
    assert tuple(taps.stack("action_readout").shape) == (7, 4, 12)


def test_hooks_are_removed_on_the_normal_path():
    model = _FakeStack().eval()
    assert _hook_counts(model) == 0
    with ActivationTaps(model, _three_point_specs()) as taps:
        assert _hook_counts(model) == 3
        with torch.no_grad():
            model(torch.randn(2, 8))
    assert _hook_counts(model) == 0
    assert not taps.active


def test_hooks_are_removed_on_the_exception_path():
    model = _FakeStack().eval()
    with pytest.raises(RuntimeError, match="boom"):
        with ActivationTaps(model, _three_point_specs()):
            assert _hook_counts(model) == 3
            raise RuntimeError("boom")
    assert _hook_counts(model) == 0, "a raised exception stranded the hooks"


def test_hooks_are_rolled_back_if_registration_fails_midway():
    model = _FakeStack().eval()
    specs = [TapSpec("ok", "vae"), TapSpec("bad", "vae")]
    taps = ActivationTaps(model, specs)
    # Make the second registration blow up after the first succeeded.
    original = model.vae.register_forward_hook
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("registration failed")
        return original(*args, **kwargs)

    model.vae.register_forward_hook = flaky  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeError, match="registration failed"):
            taps.__enter__()
    finally:
        del model.vae.register_forward_hook
    assert _hook_counts(model) == 0, "partial registration was not rolled back"


def test_misspelled_module_path_fails_immediately():
    model = _FakeStack().eval()
    with pytest.raises(AttributeError) as info:
        ActivationTaps(model, [TapSpec("typo", "vidoe.1")])
    message = str(info.value)
    assert "vidoe" in message
    assert "video" in message, "the error should list what was actually available"


def test_out_of_range_block_index_fails_immediately():
    model = _FakeStack().eval()
    with pytest.raises(IndexError, match="out of range"):
        ActivationTaps(model, [TapSpec("too_deep", "video.99")])


def test_unknown_tap_name_raises_rather_than_returning_empty():
    model = _FakeStack().eval()
    with ActivationTaps(model, [TapSpec("vae_latent", "vae")]) as taps:
        with torch.no_grad():
            model(torch.randn(2, 8))
    with pytest.raises(KeyError) as info:
        taps.get("vae_latnet")
    assert "vae_latent" in str(info.value), "available names should be listed"


def test_tapping_a_module_that_never_ran_raises():
    model = _FakeStack().eval()
    with ActivationTaps(model, [TapSpec("unused", "vae")]) as taps:
        pass  # deliberately no forward pass
    with pytest.raises(RuntimeError, match="captured nothing"):
        taps.stack("unused")


def test_dict_output_requires_and_honours_a_key_selector():
    model = _FakeStack().eval()
    with pytest.raises(TypeError, match="not a Tensor"):
        with ActivationTaps(model, [TapSpec("whole", "")]):
            with torch.no_grad():
                model(torch.randn(2, 8))

    with ActivationTaps(model, [TapSpec("action", "", select="action")]) as taps:
        with torch.no_grad():
            model(torch.randn(2, 8))
    assert tuple(taps.stack("action").shape) == (2, 4, 5)


def test_duplicate_tap_names_are_rejected():
    model = _FakeStack().eval()
    with pytest.raises(ValueError, match="duplicate tap names"):
        ActivationTaps(model, [TapSpec("a", "vae"), TapSpec("a", "readout")])


def test_resolve_module_handles_root_and_indices():
    model = _FakeStack().eval()
    assert resolve_module(model, "") is model
    assert resolve_module(model, "video.2") is model.video[2]


# --------------------------------------------------------------------------- #
# 2. Pooling determinism and guards
# --------------------------------------------------------------------------- #

def test_pooling_is_bitwise_deterministic():
    torch.manual_seed(1)
    x = torch.randn(11, 6, 2, 3, 3)
    first = pool_activation(x, feature_dim=1, output_dim=16)
    second = pool_activation(x, feature_dim=1, output_dim=16)
    assert torch.equal(first, second), "pooling must be exactly reproducible"
    assert tuple(first.shape) == (11, 16)


def test_pooling_reduces_every_axis_except_batch_and_feature():
    x = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    pooled = pool_activation(x, feature_dim=-1, output_dim=4)
    expected = x.mean(dim=1)  # [2, 4]; adaptive pool to 4 is the identity
    assert torch.allclose(pooled, expected, atol=1e-6)


def test_pooling_rejects_batch_axis_and_bad_shapes():
    with pytest.raises(ValueError, match="batch axis"):
        pool_activation(torch.randn(4, 5), feature_dim=0)
    with pytest.raises(ValueError, match="at least 2-D"):
        pool_activation(torch.randn(4), feature_dim=-1)
    with pytest.raises(ValueError, match="out of range"):
        pool_activation(torch.randn(4, 5), feature_dim=7)
    with pytest.raises(ValueError, match="positive"):
        pool_activation(torch.randn(4, 5), output_dim=0)


def test_pooling_refuses_non_finite_activations():
    x = torch.zeros(3, 4)
    x[1, 2] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        pool_activation(x, feature_dim=-1, output_dim=2)


# --------------------------------------------------------------------------- #
# 3. AUC
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "y,scores,expected",
    [
        ([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 1.0),
        ([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1], 0.0),
        ([0, 1], [0.5, 0.5], 0.5),                       # a single tied pair
        ([0, 0, 1, 1], [0.0, 1.0, 0.0, 1.0], 0.5),       # fully interleaved
        ([0, 1, 1], [0.0, 0.0, 1.0], 0.75),              # one tie, one clean win
    ],
)
def test_roc_auc_known_cases_including_ties(y, scores, expected):
    assert roc_auc(y, scores) == pytest.approx(expected)


def test_roc_auc_is_nan_when_a_class_is_missing():
    assert np.isnan(roc_auc([1, 1, 1], [0.1, 0.2, 0.3]))
    assert np.isnan(roc_auc([], []))


def test_roc_auc_rejects_non_binary_labels():
    with pytest.raises(ValueError, match="binary"):
        roc_auc([0, 1, 2], [0.1, 0.2, 0.3])


# --------------------------------------------------------------------------- #
# 4. Grouped folds
# --------------------------------------------------------------------------- #

def test_grouped_folds_never_split_a_group():
    groups = np.repeat(np.arange(20), 7)
    fold = grouped_kfold(groups, n_splits=5, seed=0)
    for g in np.unique(groups):
        assert len(set(fold[groups == g])) == 1, f"group {g} was split across folds"
    assert set(np.unique(fold)) == set(range(5))


def test_grouped_folds_are_deterministic_and_seed_sensitive():
    groups = np.repeat(np.arange(30), 3)
    a = grouped_kfold(groups, n_splits=5, seed=0)
    b = grouped_kfold(groups, n_splits=5, seed=0)
    c = grouped_kfold(groups, n_splits=5, seed=1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_grouped_folds_ignore_input_ordering():
    base = np.repeat(np.arange(12), 2)
    shuffled = base[::-1]
    fold_base = grouped_kfold(base, n_splits=4, seed=3)
    fold_shuf = grouped_kfold(shuffled, n_splits=4, seed=3)
    mapping = {g: fold_base[base == g][0] for g in np.unique(base)}
    assert all(mapping[g] == f for g, f in zip(shuffled, fold_shuf))


def test_grouped_folds_reject_too_few_groups():
    with pytest.raises(ValueError, match="group-disjoint folds"):
        grouped_kfold(np.repeat(np.arange(3), 10), n_splits=5)


# --------------------------------------------------------------------------- #
# 5. Logistic fit
# --------------------------------------------------------------------------- #

def test_fit_logistic_converges_on_separable_data():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(-3, 0.3, (50, 2)), rng.normal(3, 0.3, (50, 2))])
    y = np.concatenate([np.zeros(50), np.ones(50)])
    w = fit_logistic(X, y, l2=1e-3)
    assert np.all(np.isfinite(w)), "separable data must not diverge with l2 > 0"
    scores = np.hstack([X, np.ones((X.shape[0], 1))]) @ w
    assert roc_auc(y, scores) == pytest.approx(1.0)


def test_fit_logistic_rejects_single_class_and_nonpositive_l2():
    X = np.zeros((6, 2))
    with pytest.raises(ValueError, match="both classes"):
        fit_logistic(X, np.ones(6), l2=1.0)
    with pytest.raises(ValueError, match="strictly convex"):
        fit_logistic(X, np.array([0, 1, 0, 1, 0, 1]), l2=0.0)


def test_sigmoid_path_survives_extreme_logits():
    # Reaching |eta| ~ 1e3 is routine once a fold is nearly separable; a naive
    # 1/(1+exp(-z)) overflows there.
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(-500, 1, (20, 1)), rng.normal(500, 1, (20, 1))])
    y = np.concatenate([np.zeros(20), np.ones(20)])
    with np.errstate(over="raise", invalid="raise"):
        w = fit_logistic(X, y, l2=1e-2)
    assert np.all(np.isfinite(w))


# --------------------------------------------------------------------------- #
# 6. End-to-end probe behaviour
# --------------------------------------------------------------------------- #

def _separable_dataset(n_groups=20, per_group=10, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    labels_per_group = np.tile([0, 1], n_groups // 2)
    X, y, g = [], [], []
    for gi in range(n_groups):
        label = labels_per_group[gi]
        centre = np.zeros(dim)
        centre[0] = 3.0 if label else -3.0
        X.append(rng.normal(centre, 1.0, (per_group, dim)))
        y.append(np.full(per_group, label))
        g.append(np.full(per_group, gi))
    return np.vstack(X), np.concatenate(y), np.concatenate(g)


def _pure_noise_dataset(n_groups=20, per_group=10, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    labels_per_group = np.tile([0, 1], n_groups // 2)
    X = rng.normal(0.0, 1.0, (n_groups * per_group, dim))
    y = np.repeat(labels_per_group, per_group)
    g = np.repeat(np.arange(n_groups), per_group)
    return X, y, g


def test_probe_recovers_a_real_signal():
    X, y, g = _separable_dataset()
    result = cross_fitted_probe(X, y, g, n_splits=5, seed=0, n_bootstrap=200)
    assert result.auc > 0.95, result
    assert result.ci_low <= result.auc <= result.ci_high
    assert result.n_samples == X.shape[0]
    assert result.n_groups == 20


@pytest.mark.parametrize("n_groups", [20, 40, 60])
def test_probe_reports_chance_on_pure_noise(n_groups):
    X, y, g = _pure_noise_dataset(n_groups=n_groups, seed=7)
    result = cross_fitted_probe(X, y, g, n_splits=5, seed=0, n_bootstrap=200)
    assert 0.35 < result.auc < 0.65, result
    assert result.ci_low < 0.5 < result.ci_high, "chance must lie inside the CI"


@pytest.mark.parametrize("n_groups", [20, 40, 60])
def test_stratified_folds_are_class_balanced_and_never_single_class(n_groups):
    """Regression test for a real bias, not a style preference.

    Folds are group-disjoint and the E7 label is constant within a trajectory, so
    an *unstratified* test fold is routinely class-imbalanced. Its training folds
    then carry the opposite imbalance, the intercept shifts, and every score in
    that fold moves together. Per-fold AUC is immune (a constant shift does not
    reorder), but the pooled out-of-fold AUC is not: positives concentrate in the
    down-shifted folds.

    Measured before stratification, the pooled AUC on pure noise settled at a
    *stable* ~0.34 across 20/40/60 group counts -- reproducible enough to be
    mistaken for signal rather than dismissed as variance.
    """
    _, y, g = _pure_noise_dataset(n_groups=n_groups, seed=7)
    stratified = grouped_kfold(g, n_splits=5, seed=0, labels=y)
    plain = grouped_kfold(g, n_splits=5, seed=0)

    ratios = [y[stratified == f].mean() for f in range(5)]
    assert max(ratios) - min(ratios) < 0.15, f"stratified folds drifted: {ratios}"
    assert all(len(np.unique(y[stratified == f])) == 2 for f in range(5))

    plain_ratios = [y[plain == f].mean() for f in range(5)]
    assert max(plain_ratios) - min(plain_ratios) > max(ratios) - min(ratios), (
        "the unstratified control should be visibly worse; if it is not, this "
        f"test proves nothing. stratified={ratios} plain={plain_ratios}"
    )


@pytest.mark.parametrize("n_groups", [20, 40, 60])
def test_pooled_and_per_fold_auc_agree_under_stratification(n_groups):
    """With balanced folds the pooled and averaged statistics must not diverge.

    A gap between them is the signature of the imbalance bias returning.
    """
    X, y, g = _pure_noise_dataset(n_groups=n_groups, seed=7)
    result = cross_fitted_probe(X, y, g, n_splits=5, seed=0, n_bootstrap=0)
    assert abs(result.auc - 0.5) < 0.15, f"null drifted from chance: {result.auc}"
    assert abs(result.auc - result.mean_fold_auc) < 0.10, (
        f"pooled {result.auc} and per-fold {result.mean_fold_auc} disagree"
    )
    assert not any(np.isnan(v) for v in result.per_fold_auc), (
        "stratification should leave no fold single-class"
    )


def test_stratification_refuses_a_group_with_mixed_labels():
    g = np.repeat(np.arange(10), 4)
    y = np.zeros(40, dtype=int)
    y[:20] = 1                     # groups 0-4 positive, groups 5-9 negative
    assert len(np.unique(y[g == 5])) == 1, "fixture precondition"
    y[21] = 1                      # group 5 (indices 20..23) now straddles both
    assert len(np.unique(y[g == 5])) == 2, "fixture must actually be mixed"
    with pytest.raises(ValueError, match="misaligned"):
        grouped_kfold(g, n_splits=5, seed=0, labels=y)


def test_stratification_refuses_a_class_with_too_few_groups():
    g = np.repeat(np.arange(20), 4)
    y = np.repeat(np.array([1, 1, 1] + [0] * 17), 4)  # only 3 positive groups
    with pytest.raises(ValueError, match="stratified folds"):
        grouped_kfold(g, n_splits=5, seed=0, labels=y)


def test_grouped_folds_prevent_the_leak_that_ungrouped_folds_cause():
    """The reason grouped CV is mandatory rather than a nicety.

    Each trajectory gets a random signature and a random label. The signature
    carries no transferable information -- only trajectory identity. Per-sample
    folds let the model memorise a trajectory from its own other frames and score
    near-perfectly; group-disjoint folds correctly report chance.
    """
    # dim must exceed ~n_groups/2 for random per-trajectory labels to be linearly
    # separable from the signatures at all; at dim=8 the leak is real but the
    # linear probe cannot express it (leaky AUC 0.74), which would test the wrong
    # thing. Measured: dim 8 -> 0.74, 16 -> 0.91, 32 -> 1.00.
    rng = np.random.default_rng(11)
    n_groups, per_group, dim = 20, 12, 32
    labels_per_group = np.tile([0, 1], n_groups // 2)
    signatures = rng.normal(0, 5.0, (n_groups, dim))
    X = np.repeat(signatures, per_group, axis=0) + rng.normal(
        0, 0.05, (n_groups * per_group, dim)
    )
    y = np.repeat(labels_per_group, per_group)
    trajectory = np.repeat(np.arange(n_groups), per_group)
    per_sample = np.arange(n_groups * per_group)  # "each frame is its own group"

    leaky = cross_fitted_probe(X, y, per_sample, n_splits=5, seed=0, n_bootstrap=0)
    honest = cross_fitted_probe(X, y, trajectory, n_splits=5, seed=0, n_bootstrap=0)

    assert leaky.auc > 0.95, f"the leak should be blatant, got {leaky.auc}"
    assert honest.auc < 0.75, (
        f"grouped folds still leaked: {honest.auc} (ungrouped {leaky.auc})"
    )


def test_probe_is_deterministic():
    X, y, g = _separable_dataset(seed=3)
    a = cross_fitted_probe(X, y, g, n_splits=5, seed=0, n_bootstrap=50)
    b = cross_fitted_probe(X, y, g, n_splits=5, seed=0, n_bootstrap=50)
    # NaN != NaN, and a per-fold AUC is legitimately NaN when a grouped test fold
    # lands single-class, so compare through a NaN-stable serialisation.
    dumps = lambda r: json.dumps(r.to_dict(include_scores=True), sort_keys=True)
    assert dumps(a) == dumps(b)


def test_probe_rejects_mismatched_lengths_and_labels():
    X, y, g = _separable_dataset()
    with pytest.raises(ValueError, match="length mismatch"):
        cross_fitted_probe(X, y[:-1], g)
    bad = y.copy()
    bad[0] = 5
    with pytest.raises(ValueError, match="binary"):
        cross_fitted_probe(X, bad, g)


def test_probe_taps_pools_each_locus_with_its_own_feature_axis():
    X, y, g = _separable_dataset(dim=8)
    n = X.shape[0]
    video_like = torch.zeros(n, 4, 8)
    video_like[:, :, :] = torch.as_tensor(X, dtype=torch.float32).unsqueeze(1)
    latent_like = torch.zeros(n, 8, 2, 3, 3)
    latent_like += torch.as_tensor(X, dtype=torch.float32)[:, :, None, None, None]

    results = probe_taps(
        {"video_block_1": video_like, "vae_latent": latent_like},
        y,
        g,
        feature_dims={"vae_latent": 1},
        output_dim=8,
        n_splits=5,
        seed=0,
        n_bootstrap=0,
    )
    assert set(results) == {"video_block_1", "vae_latent"}
    for name, result in results.items():
        assert result.auc > 0.95, (name, result.auc)


# --------------------------------------------------------------------------- #
# 7. CLI
# --------------------------------------------------------------------------- #

def test_cli_round_trip(tmp_path):
    X, y, g = _separable_dataset(seed=5)
    acts = tmp_path / "acts.npz"
    labels = tmp_path / "labels.npz"
    out = tmp_path / "result.json"
    np.savez(acts, vae_latent=X.astype(np.float32))
    np.savez(labels, y=y, groups=g)

    proc = subprocess.run(
        [
            sys.executable, "-m", "fastwam.diagnostics.probe",
            "--activations", str(acts),
            "--labels", str(labels),
            "--output-dim", "8",
            "--folds", "5",
            "--bootstrap", "50",
            "--out", str(out),
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(out.read_text())
    assert payload["taps"]["vae_latent"]["auc"] > 0.95
    assert "oof_scores" not in payload["taps"]["vae_latent"]
    assert payload["output_dim"] == 8


def test_cli_rejects_an_unknown_tap(tmp_path):
    X, y, g = _separable_dataset(seed=5)
    acts = tmp_path / "acts.npz"
    labels = tmp_path / "labels.npz"
    np.savez(acts, vae_latent=X.astype(np.float32))
    np.savez(labels, y=y, groups=g)

    proc = subprocess.run(
        [
            sys.executable, "-m", "fastwam.diagnostics.probe",
            "--activations", str(acts),
            "--labels", str(labels),
            "--tap", "does_not_exist",
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "does_not_exist" in proc.stderr


def test_package_cli_entry_point_is_warning_free(tmp_path):
    """`python -m fastwam.diagnostics` must not emit runpy's double-import warning."""
    X, y, g = _separable_dataset(seed=5)
    acts = tmp_path / "acts.npz"
    labels = tmp_path / "labels.npz"
    np.savez(acts, vae_latent=X.astype(np.float32))
    np.savez(labels, y=y, groups=g)

    proc = subprocess.run(
        [
            sys.executable, "-m", "fastwam.diagnostics",
            "--activations", str(acts),
            "--labels", str(labels),
            "--output-dim", "8",
            "--bootstrap", "0",
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "sys.modules" not in proc.stderr, proc.stderr
    assert json.loads(proc.stdout)["taps"]["vae_latent"]["auc"] > 0.95


def test_cli_json_goes_to_stdout_uncontaminated(tmp_path):
    """Whatever warnings land on stderr, stdout must stay parseable JSON."""
    X, y, g = _separable_dataset(seed=5)
    acts = tmp_path / "acts.npz"
    labels = tmp_path / "labels.npz"
    np.savez(acts, vae_latent=X.astype(np.float32))
    np.savez(labels, y=y, groups=g)

    proc = subprocess.run(
        [
            sys.executable, "-m", "fastwam.diagnostics.probe",
            "--activations", str(acts),
            "--labels", str(labels),
            "--output-dim", "8",
            "--bootstrap", "0",
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    json.loads(proc.stdout)  # must parse even though runpy warns on stderr


def test_cli_rejects_labels_missing_groups(tmp_path):
    X, y, _ = _separable_dataset(seed=5)
    acts = tmp_path / "acts.npz"
    labels = tmp_path / "labels.npz"
    np.savez(acts, vae_latent=X.astype(np.float32))
    np.savez(labels, y=y)  # no `groups`

    proc = subprocess.run(
        [
            sys.executable, "-m", "fastwam.diagnostics.probe",
            "--activations", str(acts),
            "--labels", str(labels),
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "groups" in proc.stderr


# --------------------------------------------------------------------------- #
# 8. The harness must not touch model source
# --------------------------------------------------------------------------- #

def test_taps_do_not_mutate_the_model():
    torch.manual_seed(0)
    model = _FakeStack().eval()
    x = torch.randn(5, 8)
    with torch.no_grad():
        before = model(x)["action"].clone()

    with ActivationTaps(model, _three_point_specs()):
        with torch.no_grad():
            during = model(x)["action"].clone()

    with torch.no_grad():
        after = model(x)["action"].clone()

    assert torch.equal(before, during), "tapping perturbed the forward pass"
    assert torch.equal(before, after), "tapping left the model changed"
