"""Contract tests for the stage-2 sampler interface (W8).

Covers, per the work-order acceptance list:

* seed <-> init_noise round-trip identity on the REAL `infer_action` solver
  loop (run on CPU via tiny fake video/VAE components + a real MoT and real
  flow scheduler — no Wan weights needed);
* velocity_hook call count, exact timestep-grid delivery, zero-delta and
  None-delta transparency, and no-grad isolation;
* predicted_clean_action / sigma_from_timestep analytic identities;
* cfm_loss bit-equality with an inline replica of the training-loss reduction,
  including the pad-mask branch and both reductions;
* training_loss (t, eps) injection/exposure round-trip (build_inputs faked);
* fail-closed validation everywhere, including the metric-adaptive dispatch
  guard that refuses to silently drop sampler kwargs on the IDM branch.
"""
from __future__ import annotations

import types

import pytest
import torch

fastwam_module = pytest.importorskip(
    "fastwam.models.wan22.fastwam",
    reason="fastwam model deps unavailable (imageio/safetensors chain)",
)
action_dit_module = pytest.importorskip("fastwam.models.wan22.action_dit")
mot_module = pytest.importorskip("fastwam.models.wan22.mot")
flow_objective = pytest.importorskip("fastwam.adaptive_gate.flow_objective")

import torch.nn as nn  # noqa: E402  (after importorskip by design)
import torch.nn.functional as F  # noqa: E402

from fastwam.models.wan22.schedulers.scheduler_continuous import (  # noqa: E402
    WanContinuousFlowMatchScheduler,
)

FastWAM = fastwam_module.FastWAM
ActionDiT = action_dit_module.ActionDiT
MoT = mot_module.MoT
cfm_loss = flow_objective.cfm_loss
predicted_clean_action = flow_objective.predicted_clean_action
sigma_from_timestep = flow_objective.sigma_from_timestep

HID, CTX_LEN, ACT_DIM, HORIZON = 8, 5, 2, 3
Z_DIM, LAT_HW, UPS = 2, 2, 16  # first_frame_latents [1,2,1,2,2]; image 32x32


def _tiny_dit(seed: int) -> ActionDiT:
    torch.manual_seed(seed)
    return ActionDiT(
        hidden_dim=HID, action_dim=ACT_DIM, ffn_dim=16, text_dim=HID, freq_dim=8,
        eps=1e-6, num_heads=2, attn_head_dim=4, num_layers=2,
    )


class TinyVideoExpert(nn.Module):
    """Duck-typed video expert reusing an ActionDiT's pre/post over latents."""

    video_attention_mask_mode = "first_frame_causal"
    fuse_vae_embedding_in_latents = False

    def __init__(self, inner: ActionDiT):
        super().__init__()
        self.inner = inner

    def pre_dit(self, *, x, timestep, context, context_mask, action=None,
                fuse_vae_embedding_in_latents=False):
        b, c, t, h, w = x.shape
        tokens = x.reshape(b, c, t * h * w).permute(0, 2, 1).contiguous()  # [B,Sv,ACT_DIM]
        pre = self.inner.pre_dit(
            action_tokens=tokens, timestep=timestep, context=context, context_mask=context_mask
        )
        pre["meta"]["tokens_per_frame"] = tokens.shape[1]
        pre["meta"]["latent_shape"] = (b, c, t, h, w)
        return pre

    def post_dit(self, tokens, pre_state):
        out = self.inner.head(tokens)  # [B, Sv, ACT_DIM]
        b, c, t, h, w = pre_state["meta"]["latent_shape"]
        return out.permute(0, 2, 1).reshape(b, c, t, h, w).contiguous()

    def build_video_to_video_mask(self, *, video_seq_len, video_tokens_per_frame, device):
        return torch.ones((video_seq_len, video_seq_len), dtype=torch.bool, device=device)


class _FakeVAE:
    """Minimal VAE stand-in: attribute surface only, never encodes."""

    upsampling_factor = UPS

    def __init__(self):
        self.model = types.SimpleNamespace(z_dim=Z_DIM)

    def to(self, *args, **kwargs):
        return self


def _build_model() -> FastWAM:
    video_inner = _tiny_dit(11)
    action_expert = _tiny_dit(22)
    torch.manual_seed(33)
    mot = MoT({"video": video_inner, "action": action_expert}, mot_checkpoint_mixed_attn=False)
    vae = _FakeVAE()
    model = FastWAM(
        video_expert=TinyVideoExpert(video_inner), action_expert=action_expert, mot=mot,
        vae=vae, text_encoder=None, tokenizer=None, text_dim=HID, proprio_dim=None,
        device="cpu", torch_dtype=torch.float32,
    )
    model.eval()
    return model


def _inputs() -> dict:
    torch.manual_seed(55)
    return dict(
        input_image=torch.rand(1, 3, LAT_HW * UPS, LAT_HW * UPS),
        first_frame_latents=torch.randn(1, Z_DIM, 1, LAT_HW, LAT_HW),
        context=torch.randn(1, CTX_LEN, HID),
        context_mask=torch.ones(1, CTX_LEN, dtype=torch.bool),
    )


def _run(model, *, steps: int = 4, **extra) -> dict:
    inp = _inputs()
    return model.infer_action(
        prompt=None, input_image=inp["input_image"], action_horizon=HORIZON,
        first_frame_latents=inp["first_frame_latents"], context=inp["context"],
        context_mask=inp["context_mask"], num_inference_steps=steps, **extra,
    )


@pytest.fixture(scope="module")
def model() -> FastWAM:
    return _build_model()


# --------------------------------------------------------------------------- #
# 1. init_noise <-> seed round trip on the real solver loop
# --------------------------------------------------------------------------- #
class TestInitNoise:
    def test_seed_only_is_deterministic(self, model):
        assert torch.equal(_run(model, seed=123)["action"], _run(model, seed=123)["action"])

    def test_returned_noise_replays_bitwise(self, model):
        ref = _run(model, seed=123, return_init_noise=True)
        assert ref["init_noise"].shape == (1, HORIZON, ACT_DIM)
        assert ref["init_noise"].dtype == torch.float32
        replay = _run(model, init_noise=ref["init_noise"])
        assert torch.equal(replay["action"], ref["action"])

    def test_2d_noise_is_accepted_and_equivalent(self, model):
        ref = _run(model, seed=123, return_init_noise=True)
        replay = _run(model, init_noise=ref["init_noise"][0])
        assert torch.equal(replay["action"], ref["action"])

    def test_manual_generator_matches_internal_draw(self, model):
        # The internal draw is torch.randn(shape, generator=cpu_gen(seed), f32);
        # reproducing it externally and injecting must be bit-identical.
        gen = torch.Generator(device="cpu").manual_seed(123)
        noise = torch.randn((1, HORIZON, ACT_DIM), generator=gen, dtype=torch.float32)
        assert torch.equal(
            _run(model, init_noise=noise)["action"], _run(model, seed=123)["action"]
        )

    def test_different_noise_changes_the_action(self, model):
        a = _run(model, seed=123)["action"]
        b = _run(model, init_noise=torch.randn(1, HORIZON, ACT_DIM) * 3.0)["action"]
        assert not torch.equal(a, b)

    def test_default_path_returns_no_noise_key(self, model):
        assert "init_noise" not in _run(model, seed=123)

    @pytest.mark.parametrize(
        "bad, match",
        (
            (torch.randn(2, HORIZON, ACT_DIM), "shape"),
            (torch.randn(1, HORIZON + 1, ACT_DIM), "shape"),
            (torch.randn(1, HORIZON, ACT_DIM + 1), "shape"),
            (torch.randint(0, 2, (1, HORIZON, ACT_DIM)), "floating"),
            ("not a tensor", "torch.Tensor"),
        ),
    )
    def test_invalid_noise_fails_closed(self, model, bad, match):
        with pytest.raises((ValueError, TypeError), match=match):
            _run(model, init_noise=bad)

    def test_noise_and_seed_are_mutually_exclusive(self, model):
        with pytest.raises(ValueError, match="mutually exclusive"):
            _run(model, seed=1, init_noise=torch.randn(1, HORIZON, ACT_DIM))


# --------------------------------------------------------------------------- #
# 2. velocity_hook
# --------------------------------------------------------------------------- #
class TestVelocityHook:
    def test_hook_sees_every_step_and_the_exact_timestep_grid(self, model):
        steps = 6
        calls: list[tuple[torch.Tensor, torch.Tensor, int]] = []

        def hook(x_t, timestep, step_index):
            calls.append((x_t.detach().clone(), timestep.detach().clone(), step_index))
            return None

        out = _run(model, seed=123, steps=steps, velocity_hook=hook, return_init_noise=True)
        assert len(calls) == steps
        assert [c[2] for c in calls] == list(range(steps))
        expected_t, _ = model.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=steps, device=torch.device("cpu"), dtype=torch.float32
        )
        for (_, seen_t, idx) in calls:
            assert torch.equal(seen_t, expected_t[idx]), idx
        # Step 0 sees the initial latent, i.e. the converted init noise.
        assert torch.equal(calls[0][0], out["init_noise"].to(torch.float32))

    def test_none_returning_hook_is_bitwise_transparent(self, model):
        ref = _run(model, seed=123)["action"]
        out = _run(model, seed=123, velocity_hook=lambda x, t, i: None)["action"]
        assert torch.equal(out, ref)

    def test_zero_delta_hook_is_value_transparent(self, model):
        ref = _run(model, seed=123)["action"]
        out = _run(model, seed=123, velocity_hook=lambda x, t, i: torch.zeros(1, HORIZON, ACT_DIM))[
            "action"
        ]
        assert torch.equal(out, ref)

    def test_nonzero_delta_changes_the_action(self, model):
        ref = _run(model, seed=123)["action"]
        out = _run(
            model, seed=123, velocity_hook=lambda x, t, i: torch.full((1, HORIZON, ACT_DIM), 0.5)
        )["action"]
        assert not torch.equal(out, ref)

    def test_hook_grad_does_not_leak_into_the_sampler(self, model):
        def hook(x_t, timestep, step_index):
            with torch.enable_grad():
                probe = x_t.detach().requires_grad_(True)
                (energy,) = torch.autograd.grad(probe.square().sum(), probe)
            return energy  # requires_grad tensor path is exercised via detach in validator

        out = _run(model, seed=123, velocity_hook=hook)["action"]
        assert out.requires_grad is False

    @pytest.mark.parametrize(
        "delta, match",
        (
            (torch.zeros(1, HORIZON + 1, ACT_DIM), "shape"),
            (torch.zeros(1, HORIZON, ACT_DIM, dtype=torch.float64), "dtype/device"),
            (0.0, "torch.Tensor"),
        ),
    )
    def test_bad_delta_fails_closed(self, model, delta, match):
        with pytest.raises((ValueError, TypeError), match=match):
            _run(model, seed=123, velocity_hook=lambda x, t, i: delta)


# --------------------------------------------------------------------------- #
# 3. flow_objective pure functions
# --------------------------------------------------------------------------- #
class TestFlowObjective:
    def test_predicted_clean_action_recovers_x0_analytically(self):
        torch.manual_seed(0)
        x0 = torch.randn(4, HORIZON, ACT_DIM, dtype=torch.float64)
        eps = torch.randn_like(x0)
        sigma = torch.rand(4, dtype=torch.float64)
        x_sigma = (1 - sigma.view(-1, 1, 1)) * x0 + sigma.view(-1, 1, 1) * eps
        v = eps - x0
        torch.testing.assert_close(
            predicted_clean_action(x_sigma, sigma, v), x0, atol=1e-12, rtol=1e-12
        )
        # Scalar sigma broadcast.
        torch.testing.assert_close(
            predicted_clean_action(x_sigma[:1], sigma[:1].reshape(()), v[:1]),
            x0[:1],
            atol=1e-12,
            rtol=1e-12,
        )

    def test_sigma_from_timestep_inverts_the_grid(self):
        scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
        timesteps, _ = scheduler.build_inference_schedule(
            num_inference_steps=7, device=torch.device("cpu"), dtype=torch.float32
        )
        sigma = sigma_from_timestep(timesteps, 1000)
        assert torch.all((sigma >= 0) & (sigma <= 1))
        torch.testing.assert_close(sigma * 1000.0, timesteps)

    def test_predicted_clean_action_validation(self):
        x = torch.randn(2, 3, 2)
        with pytest.raises(ValueError, match="shapes must match"):
            predicted_clean_action(x, torch.rand(2), torch.randn(2, 3, 3))
        with pytest.raises(ValueError, match="1 or batch"):
            predicted_clean_action(x, torch.rand(3), torch.randn(2, 3, 2))
        with pytest.raises(TypeError, match="torch.Tensor"):
            predicted_clean_action(x, 0.5, torch.randn(2, 3, 2))

    @staticmethod
    def _reference_loss(pred, target, timestep, scheduler, action_is_pad):
        """Inline replica of the FastWAM.training_loss action reduction."""
        token = F.mse_loss(pred.float(), target.float(), reduction="none").mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=token.device, dtype=token.dtype)
            per_sample = (token * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        else:
            per_sample = token.mean(dim=1)
        weight = scheduler.training_weight(timestep).to(per_sample.device, dtype=per_sample.dtype)
        return (per_sample * weight).mean()

    @pytest.mark.parametrize("with_mask", (False, True))
    def test_cfm_loss_matches_training_reduction_bitwise(self, with_mask):
        torch.manual_seed(1)
        scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
        b, t, d = 4, HORIZON, ACT_DIM
        x0, eps, pred = torch.randn(b, t, d), torch.randn(b, t, d), torch.randn(b, t, d)
        timestep = scheduler.sample_training_t(batch_size=b, device=torch.device("cpu"), dtype=torch.float32)
        mask = None
        if with_mask:
            mask = torch.zeros(b, t, dtype=torch.bool)
            mask[:, -1] = True
        got = cfm_loss(pred, x0, eps, timestep, scheduler=scheduler, action_is_pad=mask)
        want = self._reference_loss(
            pred, scheduler.training_target(x0, eps, timestep), timestep, scheduler, mask
        )
        assert torch.equal(got, want)

    def test_cfm_loss_replay_is_exact(self):
        # The property FPO ratios rely on: same (t, eps) -> float-identical loss.
        scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
        torch.manual_seed(2)
        x0, eps, pred = (torch.randn(2, HORIZON, ACT_DIM) for _ in range(3))
        timestep = torch.tensor([137.5, 802.25])
        first = cfm_loss(pred, x0, eps, timestep, scheduler=scheduler)
        second = cfm_loss(pred.clone(), x0.clone(), eps.clone(), timestep.clone(), scheduler=scheduler)
        assert torch.equal(first, second)

    def test_cfm_loss_reduction_none_and_validation(self):
        scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
        x0, eps, pred = (torch.randn(3, HORIZON, ACT_DIM) for _ in range(3))
        timestep = torch.tensor([1.0, 500.0, 999.0])
        vec = cfm_loss(pred, x0, eps, timestep, scheduler=scheduler, reduction="none")
        assert vec.shape == (3,)
        assert torch.equal(vec.mean(), cfm_loss(pred, x0, eps, timestep, scheduler=scheduler))
        with pytest.raises(ValueError, match="shapes must match"):
            cfm_loss(pred, x0, eps[:2], timestep, scheduler=scheduler)
        with pytest.raises(ValueError, match="1D"):
            cfm_loss(pred, x0, eps, timestep[:2], scheduler=scheduler)
        with pytest.raises(ValueError, match="reduction"):
            cfm_loss(pred, x0, eps, timestep, scheduler=scheduler, reduction="sum")
        with pytest.raises(ValueError, match="action_is_pad"):
            cfm_loss(pred, x0, eps, timestep, scheduler=scheduler, action_is_pad=torch.zeros(3, 99, dtype=torch.bool))


# --------------------------------------------------------------------------- #
# 4. training_loss (t, eps) injection / exposure
# --------------------------------------------------------------------------- #
class TestTrainingDraws:
    @staticmethod
    def _fake_inputs(batch=2):
        torch.manual_seed(77)
        return {
            "input_latents": torch.randn(batch, Z_DIM, 1, LAT_HW, LAT_HW),
            "context": torch.randn(batch, CTX_LEN, HID),
            "context_mask": torch.ones(batch, CTX_LEN, dtype=torch.bool),
            "action": torch.randn(batch, HORIZON, ACT_DIM),
            "action_is_pad": None,
            "image_is_pad": None,
            "first_frame_latents": None,
            "fuse_vae_embedding_in_latents": False,
        }

    def _model_with_faked_inputs(self, monkeypatch):
        model = _build_model()
        # eval() keeps the forward free of any train-mode RNG so the replay
        # equality below is a pure statement about the injected (t, eps) pair.
        model.eval()
        inputs = self._fake_inputs()
        monkeypatch.setattr(model, "build_inputs", lambda sample, tiled=False: dict(inputs))
        return model

    def test_draws_out_exposes_the_pair_and_replay_reproduces_the_loss(self, monkeypatch):
        model = self._model_with_faked_inputs(monkeypatch)
        draws: dict = {}
        torch.manual_seed(9)
        loss_a, _ = model.training_loss(sample=None, draws_out=draws)
        assert set(draws) == {"action_noise", "action_timestep"}
        assert draws["action_noise"].shape == (2, HORIZON, ACT_DIM)
        assert draws["action_timestep"].shape == (2,)
        # Replay: reseed so the VIDEO draws (consumed before the action draws)
        # align; the injected action pair replaces the remaining consumptions.
        torch.manual_seed(9)
        loss_b, _ = model.training_loss(
            sample=None,
            action_noise=draws["action_noise"],
            action_timestep=draws["action_timestep"],
        )
        assert torch.equal(loss_a.detach(), loss_b.detach())

    def test_partial_injection_fails_closed(self, monkeypatch):
        model = self._model_with_faked_inputs(monkeypatch)
        with pytest.raises(ValueError, match="together"):
            model.training_loss(sample=None, action_noise=torch.randn(2, HORIZON, ACT_DIM))
        with pytest.raises(ValueError, match="together"):
            model.training_loss(sample=None, action_timestep=torch.rand(2))

    def test_bad_injected_shapes_fail_closed(self, monkeypatch):
        model = self._model_with_faked_inputs(monkeypatch)
        with pytest.raises(ValueError, match="`action_noise`"):
            model.training_loss(
                sample=None,
                action_noise=torch.randn(2, HORIZON + 1, ACT_DIM),
                action_timestep=torch.rand(2),
            )
        with pytest.raises(ValueError, match="`action_timestep`"):
            model.training_loss(
                sample=None,
                action_noise=torch.randn(2, HORIZON, ACT_DIM),
                action_timestep=torch.rand(3),
            )


# --------------------------------------------------------------------------- #
# 5. dispatch-layer fail-closed guard (no silent kwarg drop on IDM)
# --------------------------------------------------------------------------- #
class TestDispatchGuard:
    def test_idm_branch_refuses_sampler_kwargs(self):
        metric_adaptive = pytest.importorskip("fastwam.models.wan22.fastwam_metric_adaptive")
        cls = metric_adaptive.MetricAdaptiveFastWAM
        dummy = object.__new__(cls)
        kwargs = {
            "prompt": None,
            "input_image": torch.rand(1, 3, 32, 32),
            "action_horizon": HORIZON,
            "init_noise": torch.randn(1, HORIZON, ACT_DIM),
        }
        with pytest.raises(ValueError, match="does not implement[\\s\\S]*init_noise"):
            cls._call_inherited_branch(dummy, "idm", "infer_action", kwargs)

    def test_base_branch_accepts_sampler_kwargs_in_signature(self):
        import inspect

        params = inspect.signature(FastWAM.infer_action).parameters
        assert {"init_noise", "velocity_hook", "return_init_noise"} <= set(params)
