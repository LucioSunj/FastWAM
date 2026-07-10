"""Tests for the fused single-pass dual-regime training.

Three groups:

1. Standalone helper tests — exercise `dual_regime_masks.py` directly via a
   file-path import, so they run with ONLY torch installed (no fastwam deps,
   no weights, no GPU). They pin down the fused attention-mask semantics,
   including the mask-level statement of the base-regime exactness argument.

2. Package-level structure tests — run when the `fastwam` package imports.
   They assert the fused classes change ONLY `training_loss` (inference
   routing / checkpoints are byte-inherited from the metric-adaptive classes).

3. Model integration tests — HEAVY (Wan2.2 weights + GPU), opt-in:

       RUN_FASTWAM_MODEL_TESTS=1 pytest tests/test_dual_regime_fused.py -v

   Optional env:
       FASTWAM_TEST_TASK     (default: libero_dual_regime_fused_joint_2cam224_1e-4)
       FASTWAM_TEST_DEVICE   (default: cuda)
       FASTWAM_CONFIGS_DIR   (default: <repo>/configs)

   The parity test replays identical noise/timestep draws through the fused
   forward AND a reference two-forward computation (parent-style), asserting
   the per-regime predictions match numerically.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

torch = pytest.importorskip("torch")

# --------------------------------------------------------------------------- #
# Standalone import of the pure helper module (works without fastwam deps).
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_MASKS_PATH = os.path.abspath(
    os.path.join(_HERE, "..", "src", "fastwam", "models", "wan22", "dual_regime_masks.py")
)


def _load_masks_module():
    spec = importlib.util.spec_from_file_location("dual_regime_masks_standalone", _MASKS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


drm = _load_masks_module()


def _first_frame_causal_mask(seq_len: int, tokens_per_frame: int) -> torch.Tensor:
    """Reference video->video mask (mirrors WanModel.build_video_to_video_mask
    in `first_frame_causal` mode)."""
    mask = torch.ones((seq_len, seq_len), dtype=torch.bool)
    first = min(tokens_per_frame, seq_len)
    mask[:first, first:] = False
    return mask


# ======================================================================== #
# Group 1 — standalone helper tests (torch only)
# ======================================================================== #
class TestBuildMultiRegimeAttentionMask:
    def test_joint_style_layout(self):
        """[noisy_video(6, tpf=2) | draft_joint(3) | draft_base(3)]"""
        vv = _first_frame_causal_mask(6, 2)
        mask = drm.build_multi_regime_attention_mask(
            video_block_masks=[vv],
            draft_lens=[3, 3],
            draft_video_spans=[[(0, 6)], [(0, 2)]],
        )
        assert mask.shape == (12, 12) and mask.dtype == torch.bool
        # video->video equals the provided block mask
        assert torch.equal(mask[:6, :6], vv)
        # video rows NEVER attend action drafts (video outputs unaffected by drafts)
        assert not mask[:6, 6:].any()
        # joint draft: self block + full video
        assert mask[6:9, 6:9].all() and mask[6:9, :6].all()
        assert not mask[6:9, 9:12].any()  # drafts mutually invisible
        # base draft: self block + first-frame columns ONLY
        assert mask[9:12, 9:12].all() and mask[9:12, 0:2].all()
        assert not mask[9:12, 2:6].any()
        assert not mask[9:12, 6:9].any()

    def test_joint_draft_submask_matches_fastwam_joint_mask(self):
        """Restricting the fused mask to [video + joint draft] must reproduce the
        FastWAMJoint training mask (video vv / action->action / action->full video)."""
        vv = _first_frame_causal_mask(6, 2)
        fused = drm.build_multi_regime_attention_mask(
            video_block_masks=[vv], draft_lens=[3, 3], draft_video_spans=[[(0, 6)], [(0, 2)]]
        )
        idx = list(range(6)) + list(range(6, 9))
        sub = fused[idx][:, idx]
        expected = torch.zeros((9, 9), dtype=torch.bool)
        expected[:6, :6] = vv
        expected[6:, 6:] = True
        expected[6:, :6] = True
        assert torch.equal(sub, expected)

    def test_base_draft_submask_matches_fastwam_base_mask(self):
        """Restricting the fused mask to [first-frame cols + base draft] must
        reproduce the FastWAM (base) mask over a 1-frame video: ff tokens attend
        exactly the ff block (mask-level exactness of the base-regime replica),
        and the draft attends ff + itself."""
        tpf = 2
        vv = _first_frame_causal_mask(6, tpf)
        fused = drm.build_multi_regime_attention_mask(
            video_block_masks=[vv], draft_lens=[3, 3], draft_video_spans=[[(0, 6)], [(0, tpf)]]
        )
        # ff ROWS see exactly ff COLUMNS in the whole fused sequence
        ff_rows = fused[:tpf]
        visible = ff_rows.any(dim=0).nonzero().flatten().tolist()
        assert visible == list(range(tpf))
        # restricted [ff + base draft] submask == base mask on a 1-frame video
        idx = list(range(tpf)) + list(range(9, 12))
        sub = fused[idx][:, idx]
        expected = torch.zeros((tpf + 3, tpf + 3), dtype=torch.bool)
        expected[:tpf, :tpf] = True  # 1-frame video: full self-attention
        expected[tpf:, tpf:] = True  # action->action
        expected[tpf:, :tpf] = True  # action->first-frame video
        assert torch.equal(sub, expected)

    def test_idm_style_two_block_layout(self):
        """[noisy(4, tpf=2) | cond(4, tpf=2) | draft_idm(3) | draft_base(3)]"""
        m_noisy = _first_frame_causal_mask(4, 2)
        m_cond = _first_frame_causal_mask(4, 2)
        fused = drm.build_multi_regime_attention_mask(
            video_block_masks=[m_noisy, m_cond],
            draft_lens=[3, 3],
            draft_video_spans=[[(4, 8)], [(4, 6)]],
        )
        assert fused.shape == (14, 14)
        # blocks are internally intact and mutually invisible
        assert torch.equal(fused[:4, :4], m_noisy)
        assert torch.equal(fused[4:8, 4:8], m_cond)
        assert not fused[:4, 4:8].any() and not fused[4:8, :4].any()
        # idm draft attends cond block only; base draft attends cond ff only
        assert fused[8:11, 4:8].all() and not fused[8:11, :4].any()
        assert fused[11:14, 4:6].all() and not fused[11:14, 6:8].any()
        assert not fused[11:14, :4].any()
        # video rows never attend drafts; drafts mutually invisible
        assert not fused[:8, 8:].any()
        assert not fused[8:11, 11:14].any() and not fused[11:14, 8:11].any()

    def test_idm_submask_matches_teacher_forcing_mask(self):
        """[noisy | cond | idm draft] restriction reproduces the FastWAMIDM
        teacher-forcing training mask."""
        m_noisy = _first_frame_causal_mask(4, 2)
        m_cond = _first_frame_causal_mask(4, 2)
        fused = drm.build_multi_regime_attention_mask(
            video_block_masks=[m_noisy, m_cond],
            draft_lens=[3, 3],
            draft_video_spans=[[(4, 8)], [(4, 6)]],
        )
        idx = list(range(8)) + list(range(8, 11))
        sub = fused[idx][:, idx]
        expected = torch.zeros((11, 11), dtype=torch.bool)
        expected[:4, :4] = m_noisy
        expected[4:8, 4:8] = m_cond
        expected[8:, 8:] = True   # action->action
        expected[8:, 4:8] = True  # action->cond video only
        assert torch.equal(sub, expected)

    def test_span_out_of_range_raises(self):
        vv = _first_frame_causal_mask(4, 2)
        with pytest.raises(ValueError, match="out of range"):
            drm.build_multi_regime_attention_mask(
                video_block_masks=[vv], draft_lens=[2], draft_video_spans=[[(0, 5)]]
            )

    def test_mismatched_draft_args_raise(self):
        vv = _first_frame_causal_mask(4, 2)
        with pytest.raises(ValueError, match="equal length"):
            drm.build_multi_regime_attention_mask(
                video_block_masks=[vv], draft_lens=[2, 2], draft_video_spans=[[(0, 4)]]
            )

    def test_non_square_block_mask_raises(self):
        with pytest.raises(ValueError, match="square"):
            drm.build_multi_regime_attention_mask(
                video_block_masks=[torch.ones(3, 4, dtype=torch.bool)],
                draft_lens=[2],
                draft_video_spans=[[(0, 3)]],
            )


class TestMergeActionDraftPayloads:
    @staticmethod
    def _draft(batch=2, seq=3, dim=8, ctx_len=5, rope=4, t_mod_fill=0.0, token_wise=False):
        t_mod = torch.full((batch, 6, dim), t_mod_fill)
        if token_wise:
            t_mod = t_mod.unsqueeze(1).expand(batch, seq, 6, dim).clone()
        return {
            "tokens": torch.randn(batch, seq, dim),
            "freqs": torch.randn(seq, 1, rope),
            "t_mod": t_mod,
            "context": torch.randn(batch, ctx_len, dim),
            "context_mask": torch.ones(batch, seq, ctx_len, dtype=torch.bool),
        }

    def test_shapes_slices_and_t_mod_expansion(self):
        d0 = self._draft(t_mod_fill=1.0)
        d1 = self._draft(t_mod_fill=2.0)
        merged = drm.merge_action_draft_payloads([d0, d1])
        assert merged["tokens"].shape == (2, 6, 8)
        assert merged["freqs"].shape == (6, 1, 4)
        assert merged["t_mod"].shape == (2, 6, 6, 8)  # token-wise [B, S_sum, 6, D]
        assert merged["context_mask"].shape == (2, 6, 5)
        assert merged["draft_slices"] == [(0, 3), (3, 6)]
        # each draft's timestep modulation covers exactly its own token span
        assert torch.all(merged["t_mod"][:, :3] == 1.0)
        assert torch.all(merged["t_mod"][:, 3:] == 2.0)
        # tokens/freqs are plain concatenations
        assert torch.equal(merged["tokens"][:, :3], d0["tokens"])
        assert torch.equal(merged["tokens"][:, 3:], d1["tokens"])
        assert torch.equal(merged["freqs"][:3], d0["freqs"])
        assert torch.equal(merged["freqs"][3:], d1["freqs"])
        assert merged["context"] is d0["context"]

    def test_token_wise_t_mod_passthrough(self):
        d0 = self._draft(t_mod_fill=3.0, token_wise=True)
        d1 = self._draft(t_mod_fill=4.0)
        merged = drm.merge_action_draft_payloads([d0, d1])
        assert merged["t_mod"].shape == (2, 6, 6, 8)
        assert torch.all(merged["t_mod"][:, :3] == 3.0)
        assert torch.all(merged["t_mod"][:, 3:] == 4.0)

    def test_bad_freqs_length_raises(self):
        d0 = self._draft()
        d0["freqs"] = torch.randn(2, 1, 4)  # seq mismatch
        with pytest.raises(ValueError, match="freqs"):
            drm.merge_action_draft_payloads([d0])

    def test_bad_t_mod_ndim_raises(self):
        d0 = self._draft()
        d0["t_mod"] = torch.randn(2, 8)
        with pytest.raises(ValueError, match="t_mod"):
            drm.merge_action_draft_payloads([d0])

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            drm.merge_action_draft_payloads([])


# ======================================================================== #
# Group 2 — package-level structure tests (need importable fastwam)
# ======================================================================== #
# NOTE: imported lazily inside each test (NOT via a module-level importorskip)
# so that the standalone Group-1 tests above still run on machines where the
# fastwam dependency stack is unavailable.
def _import_fdr():
    return pytest.importorskip(
        "fastwam.models.wan22.fastwam_dual_regime_fused",
        reason="fastwam package not importable here (install -e . with deps)",
    )


def test_only_training_loss_is_overridden():
    """Inference routing must be byte-inherited from the metric-adaptive classes."""
    fdr = _import_fdr()
    from fastwam.models.wan22.fastwam_metric_adaptive import MetricAdaptiveFastWAM
    from fastwam.models.wan22.fastwam_metric_adaptive_joint import MetricAdaptiveFastWAMJoint

    joint_cls = fdr.FusedDualRegimeFastWAMJoint
    idm_cls = fdr.FusedDualRegimeFastWAM

    assert issubclass(joint_cls, MetricAdaptiveFastWAMJoint)
    assert issubclass(idm_cls, MetricAdaptiveFastWAM)

    for name in ("infer_action", "infer_joint", "infer", "_route_inherited", "configure_routing"):
        assert getattr(joint_cls, name) is getattr(MetricAdaptiveFastWAMJoint, name)
        assert getattr(idm_cls, name) is getattr(MetricAdaptiveFastWAM, name)

    assert joint_cls.training_loss is not MetricAdaptiveFastWAMJoint.training_loss
    assert idm_cls.training_loss is not MetricAdaptiveFastWAM.training_loss
    assert joint_cls.main_regime_name == "joint"
    assert idm_cls.main_regime_name == "idm"


def test_first_frame_isolation_guard():
    fdr = _import_fdr()
    mixin = fdr._FusedDualRegimeTrainingMixin
    ok = _first_frame_causal_mask(6, 2)
    mixin._require_first_frame_isolated(ok, 2)  # must not raise
    bad = torch.ones((6, 6), dtype=torch.bool)  # bidirectional
    with pytest.raises(ValueError, match="first-frame"):
        mixin._require_first_frame_isolated(bad, 2)


def test_token_wise_t_mod_guard():
    fdr = _import_fdr()
    mixin = fdr._FusedDualRegimeTrainingMixin
    mixin._require_token_wise_t_mod(torch.zeros(1, 8, 6, 4))  # must not raise
    with pytest.raises(ValueError, match="token-wise"):
        mixin._require_token_wise_t_mod(torch.zeros(1, 6, 4))


# ======================================================================== #
# Group 3 — model integration tests (Wan2.2 weights + GPU)
# ======================================================================== #
RUN_MODEL = os.environ.get("RUN_FASTWAM_MODEL_TESTS", "0") == "1"
needs_model = pytest.mark.skipif(
    not RUN_MODEL,
    reason="set RUN_FASTWAM_MODEL_TESTS=1 (requires Wan2.2 weights + GPU)",
)


def _configs_dir() -> str:
    env = os.environ.get("FASTWAM_CONFIGS_DIR")
    if env:
        return env
    return os.path.abspath(os.path.join(_HERE, "..", "configs"))


@pytest.fixture(scope="module")
def fused_model():
    if not RUN_MODEL:
        pytest.skip("RUN_FASTWAM_MODEL_TESTS != 1")
    _import_fdr()
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    task = os.environ.get("FASTWAM_TEST_TASK", "libero_dual_regime_fused_joint_2cam224_1e-4")
    device = os.environ.get("FASTWAM_TEST_DEVICE", "cuda")
    configs_dir = _configs_dir()
    if not os.path.isdir(configs_dir):
        pytest.skip(f"configs dir not found: {configs_dir}")
    try:
        with initialize_config_dir(version_base="1.3", config_dir=configs_dir):
            cfg = compose(config_name="train", overrides=[f"task={task}"])
        model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=device)
    except Exception as exc:  # missing weights / GPU / data stats etc.
        pytest.skip(f"could not build fused model: {type(exc).__name__}: {exc}")

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
    """Small synthetic batch satisfying the model's shape constraints."""
    a_dim = int(model.action_expert.action_dim)
    horizon = (num_frames - 1) * 2
    text_dim = 4096
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
class TestFusedDualRegimeTraining:
    def test_training_loss_runs_backward_and_keys(self, fused_model):
        m = fused_model
        m.zero_grad(set_to_none=True)
        loss, loss_dict = m.training_loss(_synthetic_sample(m))
        assert loss.ndim == 0 and loss.requires_grad
        assert torch.isfinite(loss)
        expected_keys = {"loss_video", f"loss_action_{m.main_regime_name}", "loss_action_base"}
        assert set(loss_dict.keys()) == expected_keys
        loss.backward()

    def test_gradient_covers_all_trained_params(self, fused_model):
        m = fused_model
        m.zero_grad(set_to_none=True)
        loss, _ = m.training_loss(_synthetic_sample(m))
        loss.backward()
        missing = [n for n, p in m.dit.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, f"{len(missing)} trained params got no grad, e.g. {missing[:8]}"

    def test_single_mot_forward_and_video_loss_once(self, fused_model):
        m = fused_model
        counts = {"mot": 0, "video_loss": 0}
        orig_mot_forward = m.mot.forward
        orig_video_loss = m._compute_video_loss_per_sample

        def mot_spy(*args, **kwargs):
            counts["mot"] += 1
            return orig_mot_forward(*args, **kwargs)

        def loss_spy(*args, **kwargs):
            counts["video_loss"] += 1
            return orig_video_loss(*args, **kwargs)

        m.mot.forward = mot_spy
        m._compute_video_loss_per_sample = loss_spy
        try:
            m.training_loss(_synthetic_sample(m))
        finally:
            m.mot.forward = orig_mot_forward
            m._compute_video_loss_per_sample = orig_video_loss
        assert counts["mot"] == 1, "fused training must run exactly ONE MoT forward"
        assert counts["video_loss"] == 1, "video loss must be computed exactly once"

    def test_w_base_zero_drops_base_action_term(self, fused_model):
        m = fused_model
        old = m.action_regime_weight_base
        m.action_regime_weight_base = 0.0
        try:
            _, loss_dict = m.training_loss(_synthetic_sample(m))
        finally:
            m.action_regime_weight_base = old
        assert loss_dict["loss_action_base"] == pytest.approx(0.0)

    def test_fused_matches_two_forward_reference(self, fused_model):
        """Replay identical draws through the fused forward and through
        parent-style separate forwards; per-regime predictions must match.

        Tolerances are loose-ish because bf16 attention kernels tile
        differently for different sequence lengths; tighten after a first GPU
        run if headroom allows.
        """
        from fastwam.models.wan22.fastwam import FastWAM

        m = fused_model
        was_training = m.dit.training
        m.dit.eval()  # disable grad checkpointing for the comparison
        try:
            with torch.no_grad():
                inputs = m.build_inputs(_synthetic_sample(m))
                draws = m._sample_dual_regime_draws(inputs)
                out = m._fused_dual_regime_forward(inputs, draws)

                context, context_mask = inputs["context"], inputs["context_mask"]
                action = inputs["action"]
                fuse_flag = inputs["fuse_vae_embedding_in_latents"]

                # ---- reference for the BASE draft: standalone ff forward ----
                base_regime = draws["action_regimes"][1]
                assert base_regime["name"] == "base"
                noisy_action = m.train_action_scheduler.add_noise(
                    action, base_regime["noise"], base_regime["timestep"]
                )
                ff = inputs["first_frame_latents"]
                timestep_video0 = torch.zeros(
                    (ff.shape[0],), dtype=ff.dtype, device=m.device
                )
                video_pre_ff = m.video_expert.pre_dit(
                    x=ff, timestep=timestep_video0, context=context,
                    context_mask=context_mask, action=None,
                    fuse_vae_embedding_in_latents=fuse_flag,
                )
                action_pre_b = m.action_expert.pre_dit(
                    action_tokens=noisy_action, timestep=base_regime["timestep"],
                    context=context, context_mask=context_mask,
                )
                mask_b = FastWAM._build_mot_attention_mask(
                    m,
                    video_seq_len=int(video_pre_ff["tokens"].shape[1]),
                    action_seq_len=int(action_pre_b["tokens"].shape[1]),
                    video_tokens_per_frame=int(video_pre_ff["meta"]["tokens_per_frame"]),
                    device=video_pre_ff["tokens"].device,
                )
                ref_b = m.mot(
                    embeds_all={"video": video_pre_ff["tokens"], "action": action_pre_b["tokens"]},
                    attention_mask=mask_b,
                    freqs_all={"video": video_pre_ff["freqs"], "action": action_pre_b["freqs"]},
                    context_all={
                        "video": {"context": video_pre_ff["context"], "mask": video_pre_ff["context_mask"]},
                        "action": {"context": action_pre_b["context"], "mask": action_pre_b["context_mask"]},
                    },
                    t_mod_all={"video": video_pre_ff["t_mod"], "action": action_pre_b["t_mod"]},
                )
                ref_pred_base = m.action_expert.post_dit(ref_b["action"], action_pre_b)

                fused_pred_base = out["action_drafts"][1]["pred"]
                assert torch.allclose(
                    fused_pred_base.float(), ref_pred_base.float(), atol=5e-2, rtol=5e-2
                ), f"base-draft mismatch: max abs diff {(fused_pred_base - ref_pred_base).abs().max()}"

                # ---- reference for the MAIN draft + video ----
                if m.main_regime_name == "joint":
                    main_regime = draws["action_regimes"][0]
                    latents = m.train_video_scheduler.add_noise(
                        inputs["input_latents"], draws["noise_video"], draws["timestep_video"]
                    )
                    latents[:, :, 0:1] = inputs["first_frame_latents"]
                    video_pre = m.video_expert.pre_dit(
                        x=latents, timestep=draws["timestep_video"], context=context,
                        context_mask=context_mask, action=None,
                        fuse_vae_embedding_in_latents=fuse_flag,
                    )
                    noisy_action_m = m.train_action_scheduler.add_noise(
                        action, main_regime["noise"], main_regime["timestep"]
                    )
                    action_pre_m = m.action_expert.pre_dit(
                        action_tokens=noisy_action_m, timestep=main_regime["timestep"],
                        context=context, context_mask=context_mask,
                    )
                    mask_m = m._build_mot_attention_mask(  # FastWAMJoint mask via MRO
                        video_seq_len=int(video_pre["tokens"].shape[1]),
                        action_seq_len=int(action_pre_m["tokens"].shape[1]),
                        video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                        device=video_pre["tokens"].device,
                    )
                    ref_m = m.mot(
                        embeds_all={"video": video_pre["tokens"], "action": action_pre_m["tokens"]},
                        attention_mask=mask_m,
                        freqs_all={"video": video_pre["freqs"], "action": action_pre_m["freqs"]},
                        context_all={
                            "video": {"context": video_pre["context"], "mask": video_pre["context_mask"]},
                            "action": {"context": action_pre_m["context"], "mask": action_pre_m["context_mask"]},
                        },
                        t_mod_all={"video": video_pre["t_mod"], "action": action_pre_m["t_mod"]},
                    )
                    ref_pred_video = m.video_expert.post_dit(ref_m["video"], video_pre)
                    ref_pred_main = m.action_expert.post_dit(ref_m["action"], action_pre_m)

                    fused_pred_main = out["action_drafts"][0]["pred"]
                    assert torch.allclose(
                        fused_pred_main.float(), ref_pred_main.float(), atol=5e-2, rtol=5e-2
                    ), f"main-draft mismatch: max abs diff {(fused_pred_main - ref_pred_main).abs().max()}"
                    assert torch.allclose(
                        out["pred_video"].float(), ref_pred_video.float(), atol=5e-2, rtol=5e-2
                    ), f"video mismatch: max abs diff {(out['pred_video'] - ref_pred_video).abs().max()}"
        finally:
            if was_training:
                m.dit.train()

    @pytest.mark.parametrize("branch_attr", ["low", "high"])
    def test_force_branch_inference_inherited(self, fused_model, branch_attr):
        m = fused_model
        branch = str(getattr(m.routing_selector, f"{branch_attr}_branch"))
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

    def test_checkpoint_format_unchanged(self, fused_model, tmp_path):
        m = fused_model
        path = tmp_path / "step_test.pt"
        m.save_checkpoint(str(path))
        payload = torch.load(str(path), map_location="cpu")
        assert "mot" in payload
        assert "step" in payload
        assert "torch_dtype" in payload
        has_proprio = getattr(m, "proprio_encoder", None) is not None
        assert ("proprio_encoder" in payload) == has_proprio
