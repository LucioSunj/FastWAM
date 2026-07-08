"""Milestone-3 tests: self-supervised oracle mode labels (fastwam side).

Pure logic, no Wan weights / GPU: error reduction + cheapest-sufficient label
selection + offline relabeling + shard IO, and `compute_mode_step_errors`
exercised with a stub adapter whose per-mode action offsets are known.

Run:
    cd FastWAM
    pytest tests/test_gate_oracle.py -v
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from fastwam.adaptive_gate import (  # noqa: E402
    FULL_INDEX,
    MODE_ORDER,
    NUM_MODES,
    SHARD_DATA_KEYS,
    chunk_errors_from_steps,
    coerce_mode,
    compute_mode_step_errors,
    label_distribution,
    load_label_shards,
    per_step_errors,
    relabel_from_steps,
    resolve_shard_paths,
    select_cheapest_sufficient,
    write_label_shard,
)


# ======================================================================== #
# error reduction
# ======================================================================== #
def test_per_step_errors_known_values():
    pred = torch.tensor([[1.0, 1.0], [2.0, 0.0]])
    gt = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    l1, l2 = per_step_errors(pred, gt)
    assert torch.allclose(l1, torch.tensor([1.0, 1.0]))
    assert torch.allclose(l2, torch.tensor([1.0, (4.0 / 2) ** 0.5]))
    with pytest.raises(ValueError):
        per_step_errors(pred, gt[:1])


def test_chunk_errors_masking_and_exec_horizon():
    # 3 modes x 4 steps; mode i has constant error i+1
    step = torch.stack([torch.full((4,), float(i + 1)) for i in range(NUM_MODES)])
    valid = torch.tensor([True, True, True, False])
    chunk, has_valid = chunk_errors_from_steps(step, valid)
    assert bool(has_valid)
    assert torch.allclose(chunk, torch.tensor([1.0, 2.0, 3.0]))

    # exec_horizon truncates *before* the mask mean
    step2 = step.clone()
    step2[:, 2:] = 100.0
    chunk2, _ = chunk_errors_from_steps(step2, torch.ones(4, dtype=torch.bool), exec_horizon=2)
    assert torch.allclose(chunk2, torch.tensor([1.0, 2.0, 3.0]))

    # empty window -> +inf and has_valid False
    chunk3, has_valid3 = chunk_errors_from_steps(step, torch.zeros(4, dtype=torch.bool))
    assert not bool(has_valid3)
    assert torch.isinf(chunk3).all()

    with pytest.raises(ValueError):
        chunk_errors_from_steps(step, valid, exec_horizon=0)
    with pytest.raises(ValueError):
        chunk_errors_from_steps(step[:2], valid)  # wrong mode dim


def test_chunk_errors_batched():
    step = torch.rand(5, NUM_MODES, 8)
    valid = torch.ones(5, 8, dtype=torch.bool)
    valid[3] = False
    chunk, has_valid = chunk_errors_from_steps(step, valid)
    assert chunk.shape == (5, NUM_MODES)
    assert has_valid.tolist() == [True, True, True, False, True]
    assert torch.isinf(chunk[3]).all()
    assert torch.allclose(chunk[0], step[0].mean(dim=-1))


# ======================================================================== #
# cheapest-sufficient selection
# ======================================================================== #
def test_select_cheapest_sufficient_semantics():
    # err(FULL)=0.05; thresholds pick out each mode
    err = torch.tensor([0.30, 0.10, 0.05])
    assert select_cheapest_sufficient(err, tol_abs=0.0, tol_rel=0.0).item() == FULL_INDEX
    assert select_cheapest_sufficient(err, tol_abs=0.06, tol_rel=0.0).item() == 1  # LATENT
    assert select_cheapest_sufficient(err, tol_abs=0.30, tol_rel=0.0).item() == 0  # SKIP
    # relative tolerance: thr = 0.05 * 3 = 0.15 -> LATENT
    assert select_cheapest_sufficient(err, tol_abs=0.0, tol_rel=2.0).item() == 1


def test_select_skip_can_beat_full():
    # stochastic inference can make SKIP strictly better than FULL
    err = torch.tensor([0.01, 0.20, 0.05])
    assert select_cheapest_sufficient(err).item() == 0


def test_select_batched_and_invalid_rows():
    err = torch.tensor([
        [0.30, 0.10, 0.05],
        [float("inf"), float("inf"), float("inf")],  # empty mask row -> FULL
    ])
    labels = select_cheapest_sufficient(err, tol_abs=0.06)
    assert labels.tolist() == [1, FULL_INDEX]


def test_select_rejects_negative_tolerance_and_bad_shape():
    err = torch.tensor([0.3, 0.2, 0.1])
    with pytest.raises(ValueError):
        select_cheapest_sufficient(err, tol_abs=-0.1)
    with pytest.raises(ValueError):
        select_cheapest_sufficient(err, tol_rel=-0.1)
    with pytest.raises(ValueError):
        select_cheapest_sufficient(err[:2])


def test_relabel_from_steps_matches_composition():
    n, t = 6, 8
    step_l1 = torch.rand(n, NUM_MODES, t)
    step_l2 = torch.rand(n, NUM_MODES, t)
    valid = torch.rand(n, t) > 0.2
    labels, chunk, has_valid = relabel_from_steps(
        step_l1, step_l2, valid, metric="l2", exec_horizon=4, tol_abs=0.05
    )
    chunk_ref, has_valid_ref = chunk_errors_from_steps(step_l2, valid, exec_horizon=4)
    assert torch.equal(chunk, chunk_ref)
    assert torch.equal(has_valid, has_valid_ref)
    assert torch.equal(labels, select_cheapest_sufficient(chunk_ref, tol_abs=0.05))
    with pytest.raises(ValueError):
        relabel_from_steps(step_l1, step_l2, valid, metric="mse")


def test_label_distribution():
    labels = torch.tensor([0, 0, 1, 2])
    dist = label_distribution(labels)
    assert dist == pytest.approx({"skip": 0.5, "latent": 0.25, "full": 0.25})


# ======================================================================== #
# compute_mode_step_errors with a stub adapter
# ======================================================================== #
class _OracleStubAdapter:
    """Per-mode constant action offsets -> exactly known per-step L1 errors."""

    OFFSETS = {"skip": 0.30, "latent": 0.10, "full": 0.05}
    COSTS = {"skip": 0.1, "latent": 0.4, "full": 1.0}

    def __init__(self, seed_jitter: float = 0.0):
        self.seed_jitter = seed_jitter
        self.encode_calls = 0
        self.act_calls: list[tuple[str, int]] = []

    def encode_world_feat(self, input_image):
        self.encode_calls += 1
        return torch.ones(6)

    def act(self, *, input_image, mode, proprio=None, context=None, context_mask=None,
            prompt=None, action_horizon=None, world_feat=None, seed=None, **kw):
        mode = coerce_mode(mode)
        self.act_calls.append((mode.value, int(seed)))
        offset = self.OFFSETS[mode.value] + self.seed_jitter * int(seed)
        chunk = torch.zeros(int(action_horizon), 7) + offset
        return {"action_chunk": chunk, "world_feat": world_feat,
                "cost": self.COSTS[mode.value], "aux": {"mode": mode.value}}


def _state(horizon=4):
    return dict(
        input_image=torch.rand(1, 3, 16, 16),
        gt_action=torch.zeros(horizon, 7),
        proprio=torch.randn(1, 5),
        context=torch.randn(1, 8, 32),
        context_mask=torch.ones(1, 8, dtype=torch.bool),
    )


def test_compute_mode_step_errors_known_offsets():
    adapter = _OracleStubAdapter()
    out = compute_mode_step_errors(adapter, seeds=(0,), **_state())
    assert out["step_l1"].shape == (NUM_MODES, 4)
    for i, mode in enumerate(MODE_ORDER):
        assert torch.allclose(out["step_l1"][i], torch.full((4,), adapter.OFFSETS[mode.value]))
        assert out["costs"][i].item() == pytest.approx(adapter.COSTS[mode.value])
    # constant offsets: L2 over the action dim == L1
    assert torch.allclose(out["step_l1"], out["step_l2"])
    # world_feat encoded exactly once and reused for every mode call
    assert adapter.encode_calls == 1
    assert out["world_feat"].shape == (6,)


def test_compute_mode_step_errors_pairs_seeds_across_modes():
    adapter = _OracleStubAdapter(seed_jitter=0.01)
    out = compute_mode_step_errors(adapter, seeds=(3, 4), **_state())
    # every mode sees the SAME seed list (paired comparison)
    for mode in MODE_ORDER:
        assert [s for m, s in adapter.act_calls if m == mode.value] == [3, 4]
    # per-seed errors are averaged: offset + jitter * mean(seeds)
    expected = _OracleStubAdapter.OFFSETS["skip"] + 0.01 * 3.5
    assert torch.allclose(out["step_l1"][0], torch.full((4,), expected), atol=1e-6)


def test_compute_mode_step_errors_validates_inputs():
    adapter = _OracleStubAdapter()
    state = _state()
    with pytest.raises(ValueError):
        compute_mode_step_errors(adapter, seeds=(), **state)
    state["gt_action"] = torch.zeros(4)
    with pytest.raises(ValueError):
        compute_mode_step_errors(adapter, seeds=(0,), **state)


def test_oracle_end_to_end_with_stub():
    """stub errors (skip .30 / latent .10 / full .05) + tol_abs=.06 -> LATENT."""
    adapter = _OracleStubAdapter()
    state = _state()
    out = compute_mode_step_errors(adapter, seeds=(0,), **state)
    valid = torch.ones(4, dtype=torch.bool)
    chunk, has_valid = chunk_errors_from_steps(out["step_l1"], valid)
    assert bool(has_valid)
    assert select_cheapest_sufficient(chunk, tol_abs=0.06).item() == 1


# ======================================================================== #
# shard IO
# ======================================================================== #
def _shard_data(n=4, z=6, p=5, t=4):
    return {
        "world_feat": torch.rand(n, z),
        "proprio": torch.rand(n, p),
        "label": torch.randint(0, NUM_MODES, (n,)),
        "chunk_err": torch.rand(n, NUM_MODES),
        "step_l1": torch.rand(n, NUM_MODES, t),
        "step_l2": torch.rand(n, NUM_MODES, t),
        "valid_steps": torch.ones(n, t, dtype=torch.bool),
        "sample_idx": torch.arange(n),
    }


def test_shard_roundtrip_and_concat(tmp_path):
    d1, d2 = _shard_data(n=4), _shard_data(n=3)
    write_label_shard(str(tmp_path / "shard_0_of_2.pt"), data=d1, meta={"task": "t", "tol_abs": 0.02})
    write_label_shard(str(tmp_path / "shard_1_of_2.pt"), data=d2, meta={"task": "t", "tol_abs": 0.02})
    data, meta = load_label_shards(str(tmp_path / "shard_*.pt"))
    assert set(SHARD_DATA_KEYS) <= set(data)
    assert data["label"].shape == (7,)
    assert torch.equal(data["world_feat"][:4], d1["world_feat"])
    assert torch.equal(data["world_feat"][4:], d2["world_feat"])
    assert meta["num_shards"] == 2
    assert meta["num_samples"] == 7
    assert meta["mode_order"] == [m.value for m in MODE_ORDER]


def test_shard_write_validates(tmp_path):
    data = _shard_data()
    bad = {k: v for k, v in data.items() if k != "label"}
    with pytest.raises(ValueError):
        write_label_shard(str(tmp_path / "x.pt"), data=bad, meta={})
    data_bad = dict(data, label=torch.zeros(99, dtype=torch.long))
    with pytest.raises(ValueError):
        write_label_shard(str(tmp_path / "x.pt"), data=data_bad, meta={})


def test_shard_load_rejects_incompatible(tmp_path):
    write_label_shard(str(tmp_path / "a.pt"), data=_shard_data(t=4), meta={})
    write_label_shard(str(tmp_path / "b.pt"), data=_shard_data(t=8), meta={})
    with pytest.raises(ValueError):
        load_label_shards([str(tmp_path / "a.pt"), str(tmp_path / "b.pt")])

    # tampered mode_order must be rejected
    payload = torch.load(str(tmp_path / "a.pt"), weights_only=False)
    payload["meta"]["mode_order"] = ["full", "latent", "skip"]
    torch.save(payload, str(tmp_path / "c.pt"))
    with pytest.raises(ValueError):
        load_label_shards(str(tmp_path / "c.pt"))

    # wrong version must be rejected
    payload = torch.load(str(tmp_path / "a.pt"), weights_only=False)
    payload["version"] = 999
    torch.save(payload, str(tmp_path / "d.pt"))
    with pytest.raises(ValueError):
        load_label_shards(str(tmp_path / "d.pt"))


def test_resolve_shard_paths(tmp_path):
    for name in ("s1.pt", "s0.pt"):
        write_label_shard(str(tmp_path / name), data=_shard_data(n=2), meta={})
    paths = resolve_shard_paths(str(tmp_path / "s*.pt"))
    assert [p.split("/")[-1] for p in paths] == ["s0.pt", "s1.pt"]  # sorted
    # list of explicit paths + de-dupe
    paths2 = resolve_shard_paths([str(tmp_path / "s0.pt"), str(tmp_path / "s0.pt")])
    assert len(paths2) == 1
    with pytest.raises(FileNotFoundError):
        resolve_shard_paths(str(tmp_path / "nothing_*.pt"))
