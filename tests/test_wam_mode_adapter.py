"""Milestone-1 tests: WAMModeAdapter (3 modes) + cost table.

Group 1 (pure logic, needs only torch + pyyaml): mode mapping, cost table, and the
adapter's mode->(branch, steps) dispatch + world_feat pooling exercised with a tiny
STUB model — no Wan weights / GPU needed.

Group 2 (gated, RUN_FASTWAM_MODEL_TESTS=1): build the real dual-regime model and
check world_feat dim, that each mode runs, and that SKIP reproduces the frozen
model's base-branch inference.

Run:
    cd FastWAM
    pytest tests/test_wam_mode_adapter.py -v
    RUN_FASTWAM_MODEL_TESTS=1 pytest tests/test_wam_mode_adapter.py -v
"""
from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from fastwam.adaptive_gate import (  # noqa: E402
    MODE_ORDER,
    NUM_MODES,
    WAMMode,
    WAMModeAdapter,
    default_cost_table,
    future_branch_for,
    load_cost_table,
    mode_from_index,
    mode_to_branch_steps,
    mode_to_index,
    normalize_cost_table,
    save_cost_table,
)


# ======================================================================== #
# modes
# ======================================================================== #
def test_mode_order_and_index_roundtrip():
    assert NUM_MODES == 3
    assert MODE_ORDER == (WAMMode.SKIP, WAMMode.LATENT, WAMMode.FULL)
    for i, m in enumerate(MODE_ORDER):
        assert mode_to_index(m) == i
        assert mode_from_index(i) == m


def test_future_branch_for():
    assert future_branch_for("joint") == "joint"
    assert future_branch_for("idm") == "idm"
    with pytest.raises(ValueError):
        future_branch_for("nope")


@pytest.mark.parametrize("backbone,future", [("joint", "joint"), ("idm", "idm")])
def test_mode_to_branch_steps(backbone, future):
    common = dict(backbone_kind=backbone, k_lo=4, k_hi=20, action_steps=20)
    assert mode_to_branch_steps(WAMMode.SKIP, **common) == ("base", 20)
    assert mode_to_branch_steps(WAMMode.LATENT, **common) == (future, 4)
    assert mode_to_branch_steps(WAMMode.FULL, **common) == (future, 20)


# ======================================================================== #
# cost table
# ======================================================================== #
def test_default_cost_table_normalized_and_monotonic():
    t = default_cost_table(k_lo=4, k_hi=20, fixed_fraction=0.15)
    assert t["full"] == pytest.approx(1.0)
    assert t["skip"] == pytest.approx(0.15)
    assert t["latent"] == pytest.approx(0.15 + 0.85 * 4 / 20)
    assert t["skip"] < t["latent"] < t["full"]


def test_default_cost_table_validates():
    with pytest.raises(ValueError):
        default_cost_table(k_lo=4, k_hi=0)
    with pytest.raises(ValueError):
        default_cost_table(k_lo=4, k_hi=20, fixed_fraction=1.0)


def test_normalize_cost_table():
    norm = normalize_cost_table({"skip": 2.0, "latent": 5.0, "full": 10.0})
    assert norm == pytest.approx({"skip": 0.2, "latent": 0.5, "full": 1.0})


def test_cost_table_save_load_roundtrip(tmp_path):
    path = str(tmp_path / "wam_cost.yaml")
    norm = {"skip": 0.14, "latent": 0.33, "full": 1.0}
    save_cost_table(path, normalized=norm, source="flops",
                    raw={"flops": {"skip": 14.0, "latent": 33.0, "full": 100.0}})
    assert load_cost_table(path) == pytest.approx(norm)
    # source override re-normalizes from raw
    assert load_cost_table(path, source="flops") == pytest.approx({"skip": 0.14, "latent": 0.33, "full": 1.0})


def test_load_cost_table_missing_returns_none():
    assert load_cost_table(None) is None
    assert load_cost_table("/no/such/file.yaml") is None


# ======================================================================== #
# adapter dispatch via a stub model (no Wan weights)
# ======================================================================== #
class _StubVAEModel:
    z_dim = 8


class _StubVAE:
    model = _StubVAEModel()


class _StubModel:
    """Minimal stand-in exposing the surface WAMModeAdapter touches."""

    def __init__(self):
        self.vae = _StubVAE()
        self.action_expert = type("A", (), {"action_dim": 7})()
        self.proprio_dim = 8
        self.calls = []
        self.encode_calls = 0

    def _encode_input_image_latents_tensor(self, input_image):
        self.encode_calls += 1
        return torch.randn(1, 8, 1, 4, 4)  # [1, z_dim, t, h, w]

    def infer_action(self, *, prompt, input_image, action_horizon, num_video_frames=None,
                     proprio=None, context=None, context_mask=None, num_inference_steps=20,
                     seed=None, force_branch=None, return_routing_info=False, **kw):
        self.calls.append(
            dict(force_branch=force_branch, num_inference_steps=num_inference_steps,
                 num_video_frames=num_video_frames, action_horizon=action_horizon)
        )
        return {"action": torch.zeros(action_horizon, 7), "_routing": {"selected_branch": force_branch}}


class _NoForceBranchModel:
    vae = _StubVAE()

    def infer_action(self, *, prompt, input_image, action_horizon, num_inference_steps=20):
        return {"action": torch.zeros(action_horizon, 7)}


def _adapter(backbone="joint", **kw):
    return WAMModeAdapter(
        _StubModel(), backbone_kind=backbone, num_video_frames=9, action_horizon=32,
        k_lo=4, k_hi=20, **kw,
    )


def test_adapter_requires_force_branch_model():
    with pytest.raises(TypeError):
        WAMModeAdapter(_NoForceBranchModel(), backbone_kind="joint",
                       num_video_frames=9, action_horizon=32)


def test_world_feat_dim_and_shape():
    a = _adapter()
    assert a.world_feat_dim == 8
    feat = a.encode_world_feat(torch.rand(1, 3, 64, 64))
    assert feat.shape == (8,)


@pytest.mark.parametrize(
    "backbone,mode,exp_branch,exp_steps",
    [
        ("joint", WAMMode.SKIP, "base", 20),
        ("joint", WAMMode.LATENT, "joint", 4),
        ("joint", WAMMode.FULL, "joint", 20),
        ("idm", WAMMode.LATENT, "idm", 4),
        ("idm", WAMMode.FULL, "idm", 20),
    ],
)
def test_act_dispatches_branch_and_steps(backbone, mode, exp_branch, exp_steps):
    a = _adapter(backbone)
    out = a.act(input_image=torch.rand(1, 3, 64, 64), mode=mode,
                context=torch.randn(1, 16, 4096), context_mask=torch.ones(1, 16, dtype=torch.bool))
    call = a.model.calls[-1]
    assert call["force_branch"] == exp_branch
    assert call["num_inference_steps"] == exp_steps
    assert call["num_video_frames"] == 9  # passed for all modes; stub keeps it
    assert out["action_chunk"].shape == (32, 7)
    assert out["world_feat"].shape == (8,)
    assert out["aux"]["mode"] == mode.value
    assert out["aux"]["branch"] == exp_branch


def test_act_cost_matches_table():
    a = _adapter()
    for mode in MODE_ORDER:
        out = a.act(input_image=torch.rand(1, 3, 64, 64), mode=mode,
                    context=torch.randn(1, 16, 4096), context_mask=torch.ones(1, 16, dtype=torch.bool))
        assert out["cost"] == pytest.approx(a.cost_table[mode.value])
    # cost(FULL) == 1 by construction
    assert a.cost_table["full"] == pytest.approx(1.0)


def test_world_feat_reused_when_passed():
    a = _adapter()
    feat = a.encode_world_feat(torch.rand(1, 3, 64, 64))
    encodes_before = a.model.encode_calls
    a.act(input_image=torch.rand(1, 3, 64, 64), mode=WAMMode.SKIP, world_feat=feat,
          context=torch.randn(1, 16, 4096), context_mask=torch.ones(1, 16, dtype=torch.bool))
    assert a.model.encode_calls == encodes_before  # not re-encoded


# ======================================================================== #
# real-model tests (gated)
# ======================================================================== #
RUN_MODEL = os.environ.get("RUN_FASTWAM_MODEL_TESTS", "0") == "1"


@pytest.mark.skipif(not RUN_MODEL, reason="set RUN_FASTWAM_MODEL_TESTS=1 (needs Wan2.2 weights + GPU)")
class TestRealModel:
    @pytest.fixture(scope="class")
    def adapter(self):
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate

        task = os.environ.get("FASTWAM_TEST_TASK", "libero_metric_adaptive_joint_2cam224_1e-4")
        backbone = os.environ.get("FASTWAM_TEST_BACKBONE", "joint")
        device = os.environ.get("FASTWAM_TEST_DEVICE", "cuda")
        configs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
        try:
            with initialize_config_dir(version_base="1.3", config_dir=configs_dir):
                cfg = compose(config_name="train", overrides=[f"task={task}"])
            model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=device)
        except Exception as exc:
            pytest.skip(f"could not build model: {type(exc).__name__}: {exc}")
        model.eval()
        model.requires_grad_(False)
        return WAMModeAdapter(model, backbone_kind=backbone, num_video_frames=9,
                              action_horizon=32, k_lo=4, k_hi=20, default_seed=0)

    def _obs(self, a):
        proprio = None if getattr(a.model, "proprio_dim", None) is None else torch.randn(1, int(a.model.proprio_dim))
        return dict(input_image=torch.rand(1, 3, 224, 448),
                    context=torch.randn(1, 16, 4096),
                    context_mask=torch.ones(1, 16, dtype=torch.bool),
                    proprio=proprio)

    def test_world_feat_dim(self, adapter):
        feat = adapter.encode_world_feat(self._obs(adapter)["input_image"])
        assert feat.shape == (adapter.world_feat_dim,)

    @pytest.mark.parametrize("mode", list(MODE_ORDER))
    def test_each_mode_runs(self, adapter, mode):
        out = adapter.act(mode=mode, **self._obs(adapter))
        assert out["action_chunk"].shape[0] == 32
        assert out["aux"]["branch"] == ("base" if mode is WAMMode.SKIP else adapter.backbone_kind)

    def test_skip_reproduces_base_branch(self, adapter):
        obs = self._obs(adapter)
        out = adapter.act(mode=WAMMode.SKIP, seed=0, **obs)
        ref = adapter.model.infer_action(
            prompt=None, input_image=obs["input_image"], action_horizon=32, num_video_frames=9,
            proprio=obs["proprio"], context=obs["context"], context_mask=obs["context_mask"],
            num_inference_steps=20, seed=0, force_branch="base",
        )
        assert torch.allclose(out["action_chunk"].float(), ref["action"].float(), atol=1e-4)
