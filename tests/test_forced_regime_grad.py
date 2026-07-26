"""Contract tests for the gradient-carrying forced-regime forward (W17).

Covers, per the work-order acceptance list:

* default-path identity: with the grad entry unused, the public
  ``@torch.no_grad()`` ``infer_action`` and the undecorated
  ``_infer_action_impl`` produce bitwise-identical outputs for a fixed seed,
  and the grad entry's forward VALUES are bitwise identical to the default
  path (autograd recording must not change numerics);
* grad flow: ``infer_action_with_grad`` returns a graph-carrying action and
  ``backward()`` populates ``.grad`` on the action-expert parameters actually
  used; the default entry keeps returning detached tensors;
* no-grad decorators NOT removed: the public ``infer_action`` (base and
  metric-adaptive dispatch) stays grad-free even when the caller opens
  ``torch.enable_grad()``;
* velocity_hook under grad: a hook delta enters the autograd graph when grad
  is enabled (checked via a trainable steering parameter receiving grad) and
  is still detached-safe on the no-grad path;
* metric-adaptive dispatch: ``force_branch="base"`` routes to the grad entry,
  IDM / missing force_branch fail closed, and W8 sampler kwargs compose;
* WAMModeAdapter ``act_with_grad``: kwarg plumbing on a recording stub,
  fail-closed on models without the W17 API, and one real tiny end-to-end
  rollout bitwise-matching ``act(mode=UNCOND)``.

Fixtures mirror tests/test_flow_sampler_interface.py (tiny real ActionDiT +
real MoT + fake VAE on CPU; no Wan weights).
"""
from __future__ import annotations

import inspect
import types

import pytest
import torch

fastwam_module = pytest.importorskip(
    "fastwam.models.wan22.fastwam",
    reason="fastwam model deps unavailable (imageio/safetensors chain)",
)
action_dit_module = pytest.importorskip("fastwam.models.wan22.action_dit")
mot_module = pytest.importorskip("fastwam.models.wan22.mot")
metric_adaptive_module = pytest.importorskip(
    "fastwam.models.wan22.fastwam_metric_adaptive"
)

import torch.nn as nn  # noqa: E402  (after importorskip by design)

FastWAM = fastwam_module.FastWAM
ActionDiT = action_dit_module.ActionDiT
MoT = mot_module.MoT
MetricAdaptiveFastWAM = metric_adaptive_module.MetricAdaptiveFastWAM
_validate_velocity_delta = fastwam_module._validate_velocity_delta

HID, CTX_LEN, ACT_DIM, HORIZON = 8, 5, 2, 3
Z_DIM, LAT_HW, UPS = 2, 2, 16  # first_frame_latents [1,2,1,2,2]; image 32x32
STEPS = 4


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
        tokens = x.reshape(b, c, t * h * w).permute(0, 2, 1).contiguous()
        pre = self.inner.pre_dit(
            action_tokens=tokens, timestep=timestep, context=context, context_mask=context_mask
        )
        pre["meta"]["tokens_per_frame"] = tokens.shape[1]
        pre["meta"]["latent_shape"] = (b, c, t, h, w)
        return pre

    def post_dit(self, tokens, pre_state):
        out = self.inner.head(tokens)
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


def _model_kwargs() -> dict:
    video_inner = _tiny_dit(11)
    action_expert = _tiny_dit(22)
    torch.manual_seed(33)
    mot = MoT({"video": video_inner, "action": action_expert}, mot_checkpoint_mixed_attn=False)
    return dict(
        video_expert=TinyVideoExpert(video_inner), action_expert=action_expert, mot=mot,
        vae=_FakeVAE(), text_encoder=None, tokenizer=None, text_dim=HID, proprio_dim=None,
        device="cpu", torch_dtype=torch.float32,
    )


def _build_model() -> FastWAM:
    model = FastWAM(**_model_kwargs())
    model.eval()
    return model


def _build_adaptive() -> MetricAdaptiveFastWAM:
    model = MetricAdaptiveFastWAM(**_model_kwargs())
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


def _call_kwargs(*, steps: int = STEPS, **extra) -> dict:
    inp = _inputs()
    return dict(
        prompt=None, input_image=inp["input_image"], action_horizon=HORIZON,
        first_frame_latents=inp["first_frame_latents"], context=inp["context"],
        context_mask=inp["context_mask"], num_inference_steps=steps, **extra,
    )


def _run(model, **extra) -> dict:
    return model.infer_action(**_call_kwargs(**extra))


def _run_grad(model, **extra) -> dict:
    return model.infer_action_with_grad(**_call_kwargs(**extra))


@pytest.fixture(scope="module")
def model() -> FastWAM:
    return _build_model()


# --------------------------------------------------------------------------- #
# 1. Default-path identity: wrapper == impl == grad-entry values, bitwise
# --------------------------------------------------------------------------- #
class TestDefaultPathIdentity:
    def test_public_method_and_impl_are_bitwise_identical(self, model):
        public = _run(model, seed=123, return_init_noise=True)
        with torch.no_grad():
            latents, noise = model._infer_action_impl(**_call_kwargs(seed=123))
        assert torch.equal(
            latents[0].detach().to(device="cpu", dtype=torch.float32), public["action"]
        )
        assert torch.equal(
            noise.detach().to(device="cpu", dtype=torch.float32), public["init_noise"]
        )

    def test_grad_entry_forward_values_match_default_bitwise(self, model):
        ref = _run(model, seed=123, return_init_noise=True)
        out = _run_grad(model, seed=123, return_init_noise=True)
        assert torch.equal(
            out["action"].detach().to(device="cpu", dtype=torch.float32), ref["action"]
        )
        assert torch.equal(out["init_noise"], ref["init_noise"])

    def test_cache_helper_wrapper_and_impl_share_one_body(self, model):
        # The decorated wrapper must be a pure delegate: same underlying
        # statements, so a fixed-seed action stays deterministic across both.
        assert torch.equal(_run(model, seed=7)["action"], _run(model, seed=7)["action"])

    def test_grad_entry_signature_mirrors_infer_action(self, model):
        grad_params = inspect.signature(FastWAM.infer_action_with_grad).parameters
        base_params = inspect.signature(FastWAM.infer_action).parameters
        assert list(grad_params) == list(base_params)


# --------------------------------------------------------------------------- #
# 2. no_grad decorators NOT removed on the public default entries
# --------------------------------------------------------------------------- #
class TestNoGradContractPreserved:
    def test_infer_action_stays_grad_free_inside_enable_grad_caller(self, model):
        with torch.enable_grad():
            out = _run(model, seed=123)
        assert out["action"].requires_grad is False
        assert out["action"].grad_fn is None

    def test_solver_interior_runs_grad_disabled_on_default_path(self, model):
        observed: list[bool] = []

        def hook(x_t, v, timestep, step_index):
            observed.append(torch.is_grad_enabled() or x_t.requires_grad)
            return None

        with torch.enable_grad():
            _run(model, seed=123, velocity_hook=hook)
        assert observed == [False] * STEPS

    def test_solver_interior_runs_grad_enabled_on_grad_path(self, model):
        observed: list[bool] = []

        def hook(x_t, v, timestep, step_index):
            observed.append(torch.is_grad_enabled())
            return None

        _run_grad(model, seed=123, velocity_hook=hook)
        assert observed == [True] * STEPS

    def test_metric_adaptive_forced_base_default_entry_stays_grad_free(self):
        adaptive = _build_adaptive()
        with torch.enable_grad():
            out = adaptive.infer_action(**_call_kwargs(seed=123, force_branch="base"))
        assert out["action"].requires_grad is False


# --------------------------------------------------------------------------- #
# 3. Grad flows through the forced-regime rollout
# --------------------------------------------------------------------------- #
class TestGradFlows:
    def test_output_carries_graph_and_backward_populates_action_expert(self):
        fresh = _build_model()
        out = _run_grad(fresh, seed=123)
        action = out["action"]
        assert action.shape == (HORIZON, ACT_DIM)
        assert action.requires_grad is True
        assert action.grad_fn is not None
        action.float().square().sum().backward()
        grads = [p.grad for p in fresh.action_expert.parameters() if p.grad is not None]
        assert grads, "backward populated no action-expert grads"
        assert any(float(g.abs().sum()) > 0.0 for g in grads)

    def test_grad_entry_overrides_a_no_grad_caller_context(self):
        fresh = _build_model()
        with torch.no_grad():
            out = _run_grad(fresh, seed=123)
        assert out["action"].requires_grad is True

    def test_all_frozen_parameters_yield_no_graph(self):
        fresh = _build_model()
        fresh.requires_grad_(False)
        out = _run_grad(fresh, seed=123)
        # No parameter requires grad and the noise is a leaf: nothing to track.
        assert out["action"].requires_grad is False

    def test_default_entry_still_detached_after_a_grad_rollout(self):
        fresh = _build_model()
        _run_grad(fresh, seed=123)["action"].sum().backward()
        out = _run(fresh, seed=123)
        assert out["action"].requires_grad is False

    def test_init_noise_replay_composes_with_grad(self):
        # W8 x W17: the (noise -> action) replay contract holds on the grad
        # path, bitwise, and still returns a graph-carrying action.
        fresh = _build_model()
        ref = _run(fresh, seed=123, return_init_noise=True)
        replay = _run_grad(fresh, init_noise=ref["init_noise"])
        assert replay["action"].requires_grad is True
        assert torch.equal(
            replay["action"].detach().to(device="cpu", dtype=torch.float32), ref["action"]
        )


# --------------------------------------------------------------------------- #
# 4. velocity_hook under grad
# --------------------------------------------------------------------------- #
class TestVelocityHookUnderGrad:
    def test_validator_keeps_graph_when_grad_enabled(self):
        reference = torch.zeros(1, HORIZON, ACT_DIM)
        delta = torch.zeros(1, HORIZON, ACT_DIM, requires_grad=True) * 1.0
        with torch.enable_grad():
            kept = _validate_velocity_delta(delta, reference=reference)
        assert kept.requires_grad is True
        assert kept.grad_fn is not None

    def test_validator_detaches_when_grad_disabled(self):
        reference = torch.zeros(1, HORIZON, ACT_DIM)
        delta = torch.nn.Parameter(torch.zeros(1, HORIZON, ACT_DIM))
        with torch.no_grad():
            detached = _validate_velocity_delta(delta, reference=reference)
        assert detached.requires_grad is False

    def test_hook_parameter_receives_grad_on_the_grad_path(self):
        # Decisive graph-through-the-delta check: a steering parameter used
        # ONLY inside the hook accumulates grad iff the delta stays attached.
        fresh = _build_model()
        steer = torch.nn.Parameter(torch.zeros(1, HORIZON, ACT_DIM))
        out = _run_grad(fresh, seed=123, velocity_hook=lambda x, v, t, i: steer * 1.0)
        out["action"].sum().backward()
        assert steer.grad is not None
        assert float(steer.grad.abs().sum()) > 0.0

    def test_x_dependent_hook_keeps_the_graph_intact(self):
        fresh = _build_model()

        def hook(x_t, v, timestep, step_index):
            return 0.1 * x_t

        out = _run_grad(fresh, seed=123, velocity_hook=hook)
        assert out["action"].requires_grad is True
        out["action"].float().square().sum().backward()
        assert any(
            p.grad is not None and float(p.grad.abs().sum()) > 0.0
            for p in fresh.action_expert.parameters()
        )

    def test_hook_remains_detach_safe_on_the_default_path(self, model):
        steer = torch.nn.Parameter(torch.zeros(1, HORIZON, ACT_DIM))
        out = _run(model, seed=123, velocity_hook=lambda x, v, t, i: steer)
        assert out["action"].requires_grad is False
        # Zero steering is also value-transparent, so the default numbers hold.
        assert torch.equal(out["action"], _run(model, seed=123)["action"])


# --------------------------------------------------------------------------- #
# 5. Metric-adaptive dispatch of the grad entry
# --------------------------------------------------------------------------- #
class TestMetricAdaptiveDispatch:
    def test_forced_base_routes_and_carries_grad(self):
        adaptive = _build_adaptive()
        out = adaptive.infer_action_with_grad(
            **_call_kwargs(seed=123, force_branch="base", return_routing_info=True)
        )
        assert out["action"].requires_grad is True
        assert out["_routing"]["selected_branch"] == "base"
        assert out["_routing"]["method_name"] == "infer_action_with_grad"
        out["action"].sum().backward()
        assert any(
            p.grad is not None and float(p.grad.abs().sum()) > 0.0
            for p in adaptive.action_expert.parameters()
        )

    def test_forced_base_values_match_default_dispatch_bitwise(self):
        adaptive = _build_adaptive()
        ref = adaptive.infer_action(**_call_kwargs(seed=123, force_branch="base"))
        out = adaptive.infer_action_with_grad(**_call_kwargs(seed=123, force_branch="base"))
        assert torch.equal(
            out["action"].detach().to(device="cpu", dtype=torch.float32), ref["action"]
        )

    def test_idm_branch_fails_closed(self):
        adaptive = _build_adaptive()
        with pytest.raises(ValueError, match="only force_branch='base'"):
            adaptive.infer_action_with_grad(**_call_kwargs(seed=1, force_branch="idm"))

    def test_missing_force_branch_fails_closed(self):
        adaptive = _build_adaptive()
        with pytest.raises(ValueError, match="explicit force_branch='base'"):
            adaptive.infer_action_with_grad(**_call_kwargs(seed=1))

    def test_idm_only_control_kwargs_raise_instead_of_dropping(self):
        adaptive = _build_adaptive()
        with pytest.raises(TypeError, match="idm_control"):
            adaptive.infer_action_with_grad(
                **_call_kwargs(seed=1, force_branch="base", idm_control="no_read")
            )

    def test_sampler_kwargs_compose_through_the_dispatch(self):
        adaptive = _build_adaptive()
        ref = adaptive.infer_action(
            **_call_kwargs(seed=123, force_branch="base", return_init_noise=True)
        )
        replay = adaptive.infer_action_with_grad(
            **_call_kwargs(force_branch="base", init_noise=ref["init_noise"])
        )
        assert replay["action"].requires_grad is True
        assert torch.equal(
            replay["action"].detach().to(device="cpu", dtype=torch.float32), ref["action"]
        )


# --------------------------------------------------------------------------- #
# 6. WAMModeAdapter.act_with_grad
# --------------------------------------------------------------------------- #
class _AdapterStubVAE:
    def __init__(self):
        self.model = types.SimpleNamespace(z_dim=8)

    def to(self, *args, **kwargs):
        return self


class _AdapterStubModel:
    """Records exactly which kwargs the adapter forwards to each entry."""

    adaptive_regimes = ("uncond", "idm")
    adaptive_backbone_kind = "idm"

    def __init__(self):
        self.vae = _AdapterStubVAE()
        self.action_expert = types.SimpleNamespace(action_dim=7)
        self.torch_dtype = torch.float32
        self.device = torch.device("cpu")
        self.proprio_dim = None
        self.infer_video_scheduler = types.SimpleNamespace(shift=5.0, num_train_timesteps=1000)
        self.infer_action_scheduler = types.SimpleNamespace(shift=5.0, num_train_timesteps=1000)
        self.calls: list[dict] = []

    def _encode_input_image_latents_tensor(self, image):
        return torch.arange(8 * 4 * 4, dtype=torch.float32).reshape(1, 8, 1, 4, 4)

    def infer_action(self, *, force_branch=None, first_frame_latents=None, **kwargs):
        call = {**kwargs, "force_branch": force_branch, "entry": "infer_action"}
        self.calls.append(call)
        out = {
            "action": torch.zeros(call["action_horizon"], 7),
            "_routing": {"selected_branch": force_branch},
        }
        if call.get("return_init_noise"):
            out["init_noise"] = torch.full((1, call["action_horizon"], 7), 0.25)
        return out

    def infer_action_with_grad(self, *, force_branch=None, first_frame_latents=None, **kwargs):
        call = {**kwargs, "force_branch": force_branch, "entry": "infer_action_with_grad"}
        self.calls.append(call)
        seedling = torch.zeros(call["action_horizon"], 7, requires_grad=True)
        out = {
            "action": seedling * 2.0,  # graph-carrying, grad_fn is not None
            "_routing": {"selected_branch": force_branch},
        }
        if call.get("return_init_noise"):
            out["init_noise"] = torch.full((1, call["action_horizon"], 7), 0.25)
        return out


class _StubModelWithoutGradEntry(_AdapterStubModel):
    """Simulates a pre-W17 model: the grad entry does not exist at all."""

    def __getattribute__(self, name):
        if name == "infer_action_with_grad":
            raise AttributeError(name)
        return super().__getattribute__(name)


class TestAdapterActWithGrad:
    @staticmethod
    def _adapter(model=None):
        adapter_module = pytest.importorskip("fastwam.adaptive_gate.wam_mode_adapter")
        model = _AdapterStubModel() if model is None else model
        adapter = adapter_module.WAMModeAdapter(
            model,
            backbone_kind="idm",
            num_video_frames=9,
            generation_horizon=32,
            inference_steps=20,
            context_len=8,
            default_seed=77,
            allow_unloaded_model=True,
        )
        return adapter, model

    @staticmethod
    def _image():
        return torch.rand(1, 3, 64, 64)

    def test_forwards_forced_base_to_the_grad_entry(self):
        adapter, model = self._adapter()
        result = adapter.act_with_grad(input_image=self._image())
        (call,) = model.calls
        assert call["entry"] == "infer_action_with_grad"
        assert call["force_branch"] == "base"
        assert call["seed"] == 77
        assert "init_noise" not in call
        assert "velocity_hook" not in call
        assert "return_init_noise" not in call
        assert result["action_chunk"].grad_fn is not None
        assert result["aux"]["mode"] == "uncond"
        assert result["aux"]["grad_enabled"] is True

    def test_injected_noise_suppresses_default_seed(self):
        adapter, model = self._adapter()
        noise = torch.randn(1, 32, 7)
        adapter.act_with_grad(input_image=self._image(), init_noise=noise)
        (call,) = model.calls
        assert call["seed"] is None
        assert call["init_noise"] is noise

    def test_noise_and_seed_are_mutually_exclusive_before_any_model_call(self):
        adapter, model = self._adapter()
        with pytest.raises(ValueError, match="mutually exclusive"):
            adapter.act_with_grad(
                input_image=self._image(), seed=5, init_noise=torch.randn(1, 32, 7)
            )
        assert model.calls == []

    def test_return_init_noise_plumbs_the_tensor_out(self):
        adapter, model = self._adapter()
        result = adapter.act_with_grad(input_image=self._image(), return_init_noise=True)
        assert torch.equal(result["init_noise"], torch.full((1, 32, 7), 0.25))

    def test_pre_w17_model_fails_closed(self):
        adapter, model = self._adapter(model=_StubModelWithoutGradEntry())
        with pytest.raises(TypeError, match="infer_action_with_grad"):
            adapter.act_with_grad(input_image=self._image())
        assert model.calls == []

    def test_act_and_act_with_grad_share_the_validation_prefix(self):
        adapter, _ = self._adapter()
        modes = pytest.importorskip("fastwam.adaptive_gate.modes")
        bad_context = torch.randn(1, 3, HID)  # wrong context length (8 expected)
        with pytest.raises(ValueError, match="context length"):
            adapter.act(
                input_image=self._image(), mode=modes.WAMMode.UNCOND,
                context=bad_context, context_mask=torch.ones(1, 3, dtype=torch.bool),
            )
        with pytest.raises(ValueError, match="context length"):
            adapter.act_with_grad(
                input_image=self._image(),
                context=bad_context, context_mask=torch.ones(1, 3, dtype=torch.bool),
            )

    def test_end_to_end_real_tiny_model_matches_act_bitwise(self):
        adapter_module = pytest.importorskip("fastwam.adaptive_gate.wam_mode_adapter")
        modes = pytest.importorskip("fastwam.adaptive_gate.modes")
        adaptive = _build_adaptive()
        adapter = adapter_module.WAMModeAdapter(
            adaptive,
            backbone_kind="idm",
            num_video_frames=9,
            generation_horizon=HORIZON,
            inference_steps=STEPS,
            context_len=CTX_LEN,
            allow_unloaded_model=True,
        )
        inp = _inputs()
        encoded = adapter_module.EncodedWorldState(
            world_feat=torch.zeros(5 * Z_DIM),
            first_frame_latents=inp["first_frame_latents"],
            image_shape=(LAT_HW * UPS, LAT_HW * UPS),
        )
        common = dict(
            input_image=inp["input_image"],
            context=inp["context"],
            context_mask=inp["context_mask"],
            encoded_state=encoded,
            seed=123,
        )
        ref = adapter.act(mode=modes.WAMMode.UNCOND, **common)
        out = adapter.act_with_grad(**common)
        assert out["action_chunk"].requires_grad is True
        assert torch.equal(
            out["action_chunk"].detach().to(device="cpu", dtype=torch.float32),
            ref["action_chunk"],
        )
        assert out["cost"] == ref["cost"]
        out["action_chunk"].sum().backward()
        assert any(
            p.grad is not None and float(p.grad.abs().sum()) > 0.0
            for p in adaptive.action_expert.parameters()
        )
