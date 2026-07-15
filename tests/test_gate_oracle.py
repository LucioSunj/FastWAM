"""Binary oracle semantics, shared text features and strict shard IO."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from fastwam.adaptive_gate import (  # noqa: E402
    IDM_INDEX,
    MODE_ORDER,
    NUM_MODES,
    SHARD_DATA_KEYS,
    EncodedWorldState,
    chunk_errors_from_steps,
    all_mode_errors_finite,
    compose_group_id,
    coerce_mode,
    compute_mode_step_errors,
    label_distribution,
    load_label_shards,
    per_step_errors,
    pool_text_context,
    quality_metadata,
    relabel_from_steps,
    resolve_shard_paths,
    select_cheapest_near_best,
    write_label_shard,
)


def test_text_pooling_is_masked_deterministic_and_batched():
    context = torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8)
    mask = torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]], dtype=torch.bool)
    out = pool_text_context(context, mask, output_dim=4)
    assert out.shape == (2, 4) and out.dtype == torch.float32
    assert torch.equal(out, pool_text_context(context, mask, output_dim=4))
    changed = context.clone()
    changed[0, 2:] = 1e6
    assert torch.equal(out[0], pool_text_context(changed, mask, output_dim=4)[0])
    with pytest.raises(ValueError):
        pool_text_context(context, torch.zeros_like(mask))


def test_per_step_and_chunk_errors():
    pred = torch.tensor([[1.0, 1.0], [2.0, 0.0]])
    gt = torch.zeros_like(pred)
    l1, l2 = per_step_errors(pred, gt)
    assert torch.allclose(l1, torch.tensor([1.0, 1.0]))
    assert torch.allclose(l2, torch.tensor([1.0, 2.0**0.5]))

    step = torch.tensor([[1.0, 1.0, 9.0], [2.0, 2.0, 9.0]])
    chunk, valid = chunk_errors_from_steps(
        step, torch.ones(3, dtype=torch.bool), exec_horizon=2
    )
    assert bool(valid)
    assert torch.allclose(chunk, torch.tensor([1.0, 2.0]))
    empty, valid = chunk_errors_from_steps(step, torch.zeros(3, dtype=torch.bool))
    assert not bool(valid) and torch.isinf(empty).all()


def test_near_best_not_idm_relative():
    # IDM is worse: UNCOND is correctly selected even with zero tolerance.
    assert select_cheapest_near_best(torch.tensor([0.10, 0.50])).item() == 0
    # IDM is best by more than tolerance.
    assert select_cheapest_near_best(torch.tensor([0.30, 0.10]), tol_abs=0.05).item() == IDM_INDEX
    # UNCOND becomes near-best and wins on compute.
    assert select_cheapest_near_best(torch.tensor([0.30, 0.10]), tol_abs=0.21).item() == 0
    # Measured costs, rather than categorical position, drive cheapest selection.
    label = select_cheapest_near_best(
        torch.tensor([0.10, 0.10]), costs=torch.tensor([2.0, 1.0])
    )
    assert label.item() == IDM_INDEX
    invalid = select_cheapest_near_best(torch.tensor([float("inf"), float("nan")]))
    assert invalid.item() == IDM_INDEX


def test_quality_metadata_exposes_bad_idm_reference():
    quality = quality_metadata(torch.tensor([[0.1, 0.6], [0.3, 0.2]]))
    assert torch.allclose(quality["best_err"], torch.tensor([0.1, 0.2]))
    assert torch.allclose(quality["idm_err"], torch.tensor([0.6, 0.2]))
    assert torch.allclose(quality["idm_regret"], torch.tensor([0.5, 0.0]))
    invalid = quality_metadata(torch.tensor([[float("inf"), float("inf")]]))
    assert torch.isinf(invalid["idm_regret"]).all()
    assert not torch.isnan(invalid["idm_regret"]).any()
    assert all_mode_errors_finite(torch.zeros(2, 3))
    assert not all_mode_errors_finite(torch.tensor([0.0, float("nan")]))


def test_group_id_includes_dataset_identity():
    assert compose_group_id(0, 7) != compose_group_id(1, 7)
    assert compose_group_id(torch.tensor([2]), torch.tensor([9])) == (2 << 32) | 9
    assert compose_group_id(-1, 9) == -1


def test_relabel_from_steps():
    step_l1 = torch.tensor([[[0.3, 0.3], [0.1, 0.1]]])
    step_l2 = step_l1 + 0.1
    labels, chunk, valid = relabel_from_steps(
        step_l1, step_l2, torch.ones(1, 2, dtype=torch.bool), tol_abs=0.05
    )
    assert labels.tolist() == [IDM_INDEX]
    assert chunk.shape == (1, NUM_MODES) and valid.tolist() == [True]
    assert label_distribution(torch.tensor([0, 0, 1])) == pytest.approx(
        {"uncond": 2 / 3, "idm": 1 / 3}
    )


class _OracleStubAdapter:
    OFFSETS = {"uncond": 0.30, "idm": 0.05}
    COSTS = {"uncond": 0.2, "idm": 1.0}

    def __init__(self):
        self.encode_calls = 0
        self.calls = []

    def encode_world_state(self, image):
        self.encode_calls += 1
        return EncodedWorldState(torch.ones(10), torch.ones(1, 2, 1, 2, 2))

    def act(
        self,
        *,
        mode,
        generation_horizon,
        encoded_state,
        seed,
        **kwargs,
    ):
        mode = coerce_mode(mode)
        self.calls.append((mode.value, int(seed), encoded_state))
        return {
            "action_chunk": torch.full((generation_horizon, 7), self.OFFSETS[mode.value]),
            "world_feat": encoded_state.world_feat,
            "cost": self.COSTS[mode.value],
        }


def test_compute_mode_errors_pairs_seeds_and_reuses_encoding():
    adapter = _OracleStubAdapter()
    context = torch.randn(1, 8, 32)
    out = compute_mode_step_errors(
        adapter,
        input_image=torch.rand(1, 3, 16, 16),
        gt_action=torch.zeros(4, 7),
        context=context,
        context_mask=torch.ones(1, 8, dtype=torch.bool),
        seeds=(3, 4),
    )
    assert adapter.encode_calls == 1
    assert out["world_feat"].shape == (10,)
    assert out["text_feat"].shape == (64,)
    assert out["step_l1"].shape == (2, 4)
    for mode in MODE_ORDER:
        assert [seed for name, seed, _ in adapter.calls if name == mode.value] == [3, 4]
    assert len({id(state) for _, _, state in adapter.calls}) == 1


def _meta(**overrides):
    base = {
        "task": "libero",
        "backbone_kind": "idm",
        "ckpt_fingerprint": "ckpt-sha",
        "ckpt_file_sha256": "ckpt-file-sha",
        "dataset_stats_fingerprint": "stats-sha",
        "num_video_frames": 9,
        "inference_steps": 20,
        "solver_fingerprint": "solver-sha",
        "context_len": 128,
        "model_dtype": "torch.bfloat16",
        "cost_table": {"uncond": 0.2, "idm": 1.0},
        "metric": "l1",
        "exec_horizon": 10,
        "tol_rel": 0.1,
        "tol_abs": 0.02,
        "num_seeds": 1,
        "seed_base": 0,
        "stride": 20,
        "skip_padded": False,
        "max_samples": None,
        "num_shards": 1,
        "shard_index": 0,
        "world_feat_layout": "spatial_2x2_plus_channel_std_v1",
        "world_feat_dim": 10,
        "text_feat_dim": 64,
        "text_feat_layout": "masked_mean_adaptive_avg_pool_v1",
        "proprio_dim": 5,
        "action_horizon": 4,
        "group_id_layout": "dataset_index_u31_episode_index_u32_v1",
    }
    return {**base, **overrides}


def _shard_data(n=3, t=4):
    chunk = torch.rand(n, 2)
    quality = quality_metadata(chunk)
    return {
        "world_feat": torch.randn(n, 10),
        "text_feat": torch.randn(n, 64),
        "proprio": torch.randn(n, 5),
        "label": torch.randint(0, 2, (n,)),
        "chunk_err": chunk,
        "step_l1": torch.rand(n, 2, t),
        "step_l2": torch.rand(n, 2, t),
        "valid_steps": torch.ones(n, t, dtype=torch.bool),
        "sample_idx": torch.arange(n),
        "group_id": torch.arange(n),
        **quality,
    }


def test_shard_roundtrip_and_strict_fingerprint(tmp_path):
    a, b = tmp_path / "a.pt", tmp_path / "b.pt"
    write_label_shard(str(a), data=_shard_data(2), meta=_meta(num_shards=2, shard_index=0))
    write_label_shard(str(b), data=_shard_data(3), meta=_meta(num_shards=2, shard_index=1))
    data, meta = load_label_shards([str(a), str(b)])
    assert set(data) == set(SHARD_DATA_KEYS)
    assert data["label"].shape == (5,)
    assert meta["mode_order"] == ["uncond", "idm"]
    assert meta["num_shards"] == 2

    incompatible = tmp_path / "other.pt"
    write_label_shard(
        str(incompatible), data=_shard_data(1),
        meta=_meta(task="robotwin", num_shards=2, shard_index=1),
    )
    with pytest.raises(ValueError, match="fingerprint"):
        load_label_shards([str(a), str(incompatible)])

    payload = torch.load(a, weights_only=False)
    payload["meta"]["tol_abs"] = 999
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)
    with pytest.raises(ValueError, match="stale"):
        load_label_shards(str(tampered))

    payload = torch.load(a, weights_only=False)
    payload["data"]["text_feat"][0, 0] = float("inf")
    corrupted = tmp_path / "corrupted.pt"
    torch.save(payload, corrupted)
    with pytest.raises(ValueError, match="non-finite"):
        load_label_shards(str(corrupted), allow_partial=True)

    with pytest.raises(ValueError, match="incomplete"):
        load_label_shards(str(a))
    partial, partial_meta = load_label_shards(str(a), allow_partial=True)
    assert partial["label"].shape == (2,) and partial_meta["num_loaded_shards"] == 1


def test_shard_loader_rejects_duplicate_indices(tmp_path):
    for name in ("a.pt", "b.pt"):
        write_label_shard(
            str(tmp_path / name), data=_shard_data(1),
            meta=_meta(num_shards=2, shard_index=0),
        )
    with pytest.raises(ValueError, match="duplicate"):
        load_label_shards(str(tmp_path / "*.pt"), allow_partial=True)


def test_shard_validation_and_path_resolution(tmp_path):
    data = _shard_data()
    data.pop("text_feat")
    with pytest.raises(ValueError):
        write_label_shard(str(tmp_path / "bad.pt"), data=data, meta=_meta())
    for name in ("s1.pt", "s0.pt"):
        write_label_shard(str(tmp_path / name), data=_shard_data(), meta=_meta())
    paths = resolve_shard_paths(str(tmp_path / "s*.pt"))
    assert [path.split("/")[-1] for path in paths] == ["s0.pt", "s1.pt"]

    nonfinite = _shard_data()
    nonfinite["world_feat"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        write_label_shard(str(tmp_path / "nonfinite.pt"), data=nonfinite, meta=_meta())
