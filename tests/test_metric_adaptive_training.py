"""Tests for MetricAdaptiveFastWAM dual-regime training.

Two groups:

1. Import-level unit tests — run as long as the `fastwam` package imports
   (needs torch + hydra; no Wan weights / GPU). They check the loss-reduction
   helper and the training-knob parsing.

2. Model integration tests — build the real adaptive model exactly as the
   trainer does (`instantiate(cfg.model, ...)`, see runtime.py:372) and run a
   real training step + inference. These are HEAVY (Wan2.2 weights + GPU) and
   are skipped unless you opt in:

       RUN_FASTWAM_MODEL_TESTS=1 pytest tests/test_metric_adaptive_training.py -v

   Optional env:
       FASTWAM_TEST_TASK     (default: libero_metric_adaptive_2cam224_1e-4)
       FASTWAM_TEST_DEVICE   (default: cuda)
       FASTWAM_CONFIGS_DIR   (default: <repo>/configs)
"""
from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("hydra")
fma = pytest.importorskip(
    "fastwam.models.wan22.fastwam_metric_adaptive",
    reason="fastwam package not importable here (install -e . with deps)",
)

MetricAdaptiveFastWAM = fma.MetricAdaptiveFastWAM
_to_plain_dict = fma._to_plain_dict
from fastwam.adaptive_gate import normalized_dual_regime_action_loss

RUN_MODEL = os.environ.get("RUN_FASTWAM_MODEL_TESTS", "0") == "1"
needs_model = pytest.mark.skipif(
    not RUN_MODEL,
    reason="set RUN_FASTWAM_MODEL_TESTS=1 (requires Wan2.2 weights + GPU)",
)


# ======================================================================== #
# Group 1 — import-level unit tests (no weights / GPU)
# ======================================================================== #
def test_to_plain_dict_none_is_empty():
    assert _to_plain_dict(None) == {}


def test_to_plain_dict_passthrough_mapping():
    assert _to_plain_dict({"a": 1, "b": 2}) == {"a": 1, "b": 2}


def test_train_knob_parsing_matches_factory_defaults():
    # Mirrors create_metric_adaptive_fastwam's parsing of the `train:` block.
    cfg = _to_plain_dict(None)
    assert float(cfg.get("action_regime_weight_uncond", 1.0)) == 1.0
    assert bool(cfg.get("share_inputs", True)) is True

    cfg2 = _to_plain_dict({"action_regime_weight_uncond": 0.5, "share_inputs": False})
    assert float(cfg2["action_regime_weight_uncond"]) == 0.5
    assert bool(cfg2["share_inputs"]) is False


def test_dual_regime_action_objective_is_normalized_and_validated():
    combined, main, uncond = normalized_dual_regime_action_loss(
        torch.tensor(2.0), torch.tensor(4.0), 1.0
    )
    assert combined.item() == pytest.approx(3.0)
    assert main.item() == pytest.approx(1.0)
    assert uncond.item() == pytest.approx(2.0)
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            normalized_dual_regime_action_loss(torch.tensor(1.0), torch.tensor(1.0), bad)


def test_action_loss_per_sample_no_pad():
    # self is unused by the helper, so call it unbound with self=None.
    pred = torch.tensor([[[0.0], [10.0]]])  # [B=1, T=2, a=1]
    target = torch.zeros_like(pred)
    loss = MetricAdaptiveFastWAM._action_loss_per_sample(None, pred, target, None)
    assert loss.shape == (1,)
    assert loss.item() == pytest.approx(50.0)  # (0 + 100) / 2


def test_action_loss_per_sample_respects_pad_mask():
    pred = torch.tensor([[[0.0], [10.0]]])
    target = torch.zeros_like(pred)
    pad = torch.tensor([[False, True]])  # mask the high-error step
    loss = MetricAdaptiveFastWAM._action_loss_per_sample(None, pred, target, pad)
    assert loss.item() == pytest.approx(0.0)


def test_action_loss_per_sample_all_pad_is_finite():
    pred = torch.ones(1, 2, 1)
    target = torch.zeros(1, 2, 1)
    pad = torch.tensor([[True, True]])  # clamp(min=1) must avoid div-by-zero
    loss = MetricAdaptiveFastWAM._action_loss_per_sample(None, pred, target, pad)
    assert torch.isfinite(loss).all()
    assert loss.item() == pytest.approx(0.0)


# ======================================================================== #
# Group 2 — model integration tests (Wan2.2 weights + GPU)
# ======================================================================== #
def _configs_dir() -> str:
    env = os.environ.get("FASTWAM_CONFIGS_DIR")
    if env:
        return env
    # fma.__file__ = <repo>/src/fastwam/models/wan22/fastwam_metric_adaptive.py
    here = os.path.dirname(os.path.abspath(fma.__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "..", "..", "configs"))


@pytest.fixture(scope="module")
def adaptive_model():
    if not RUN_MODEL:
        pytest.skip("RUN_FASTWAM_MODEL_TESTS != 1")
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    task = os.environ.get("FASTWAM_TEST_TASK", "libero_metric_adaptive_2cam224_1e-4")
    device = os.environ.get("FASTWAM_TEST_DEVICE", "cuda")
    configs_dir = _configs_dir()
    if not os.path.isdir(configs_dir):
        pytest.skip(f"configs dir not found: {configs_dir}")
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    with initialize_config_dir(version_base="1.3", config_dir=configs_dir):
        cfg = compose(config_name="train", overrides=[f"task={task}"])
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=device)

    # Replicate the trainer's dit-only train mode (trainer.py:281-295).
    model.eval()
    model.requires_grad_(False)
    model.dit.train()
    model.dit.requires_grad_(True)
    proprio = getattr(model, "proprio_encoder", None)
    if proprio is not None:
        proprio.train()
        proprio.requires_grad_(True)
    return model


def _synthetic_sample(model, batch=1, num_frames=5, height=64, width=64):
    """Small synthetic batch satisfying the model's shape constraints.

    num_frames % 4 == 1, action_horizon % (num_frames - 1) == 0, H/W % 16 == 0.
    """
    a_dim = int(model.action_expert.action_dim)
    horizon = (num_frames - 1) * 2  # divisible by (num_frames - 1)
    text_dim = 4096  # Wan text-encoder context dim
    ctx_len = 16
    sample = {
        "video": torch.rand(batch, 3, num_frames, height, width),
        "action": torch.randn(batch, horizon, a_dim),
        "context": torch.randn(batch, ctx_len, text_dim),
        "context_mask": torch.ones(batch, ctx_len, dtype=torch.bool),
        "action_is_pad": torch.zeros(batch, horizon, dtype=torch.bool),
        "image_is_pad": torch.zeros(batch, num_frames, dtype=torch.bool),
    }
    proprio_dim = getattr(model, "proprio_dim", None)
    if proprio_dim is not None:
        sample["proprio"] = torch.randn(batch, horizon, int(proprio_dim))
    return sample


@needs_model
class TestDualRegimeTraining:
    def test_training_loss_runs_backward_and_keys(self, adaptive_model):
        m = adaptive_model
        m.zero_grad(set_to_none=True)
        loss, loss_dict = m.training_loss(_synthetic_sample(m))
        assert loss.ndim == 0 and loss.requires_grad
        assert torch.isfinite(loss)
        assert set(loss_dict.keys()) == {
            "loss_video", "loss_action_idm", "loss_action_uncond",
            "loss_action_combined", "action_regime_weight_uncond",
        }
        assert loss_dict["loss_action_combined"] == pytest.approx(
            loss_dict["loss_action_idm"] + loss_dict["loss_action_uncond"]
        )
        loss.backward()  # must not raise

    def test_gradient_covers_all_trained_params(self, adaptive_model):
        m = adaptive_model
        m.zero_grad(set_to_none=True)
        loss, _ = m.training_loss(_synthetic_sample(m))
        loss.backward()
        missing = [n for n, p in m.dit.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, f"{len(missing)} trained params got no grad, e.g. {missing[:8]}"

    def test_video_loss_computed_exactly_once(self, adaptive_model):
        m = adaptive_model
        calls = {"n": 0}
        original = m._compute_video_loss_per_sample

        def spy(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        m._compute_video_loss_per_sample = spy
        try:
            m.training_loss(_synthetic_sample(m))
        finally:
            m._compute_video_loss_per_sample = original
        assert calls["n"] == 1, "video loss must come only from the idm forward"

    def test_w_base_zero_drops_base_action_term(self, adaptive_model):
        m = adaptive_model
        old = m.action_regime_weight_uncond
        m.action_regime_weight_uncond = 0.0
        try:
            _, loss_dict = m.training_loss(_synthetic_sample(m))
        finally:
            m.action_regime_weight_uncond = old
        assert loss_dict["loss_action_uncond"] == pytest.approx(0.0)

    def test_share_inputs_false_delegated_path(self, adaptive_model):
        m = adaptive_model
        old = m.train_share_inputs
        m.train_share_inputs = False
        try:
            loss, loss_dict = m.training_loss(_synthetic_sample(m))
            loss.backward()
        finally:
            m.train_share_inputs = old
        assert "loss_action_uncond" in loss_dict and "loss_action_combined" in loss_dict

    @pytest.mark.parametrize("branch", ["base", "idm"])
    def test_force_branch_inference(self, adaptive_model, branch):
        m = adaptive_model
        num_frames, height, width = 5, 64, 64
        horizon = (num_frames - 1) * 2
        kwargs = dict(
            prompt=None,
            input_image=torch.rand(1, 3, height, width),
            action_horizon=horizon,
            num_video_frames=num_frames,
            context=torch.randn(1, 16, 4096),
            context_mask=torch.ones(1, 16, dtype=torch.bool),
            num_inference_steps=2,
            force_branch=branch,
        )
        if getattr(m, "proprio_dim", None) is not None:
            kwargs["proprio"] = torch.randn(1, int(m.proprio_dim))
        out = m.infer_action(**kwargs)
        assert "action" in out
        assert out["action"].shape[0] == horizon
        assert out["_routing"]["selected_branch"] == branch

    def test_checkpoint_records_adaptive_provenance(self, adaptive_model, tmp_path):
        m = adaptive_model
        m.dataset_stats_fingerprint = "test-stats-sha256"
        path = tmp_path / "step_test.pt"
        m.save_checkpoint(str(path))
        payload = torch.load(str(path), map_location="cpu")
        assert "mot" in payload
        assert "step" in payload
        assert "torch_dtype" in payload
        assert payload["fastwam_provenance"]["adaptive_regimes"] == ["uncond", "idm"]
        assert payload["fastwam_provenance"]["adaptive_backbone_kind"] == "idm"
        assert payload["fastwam_provenance"]["action_regime_weight_uncond"] > 0
        assert payload["fastwam_provenance"]["checkpoint_id"]
        has_proprio = getattr(m, "proprio_encoder", None) is not None
        assert ("proprio_encoder" in payload) == has_proprio
