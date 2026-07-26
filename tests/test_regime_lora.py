"""Contract tests for the regime-gated LoRA adapter (stage-2 W16).

Pins the four contracts downstream work orders (W17, WS5, WS7) rely on:

1. Gating: the LoRA delta exists ONLY on the ``"uncond"`` regime delivered by
   the W9 explicit-regime protocol. Known non-uncond regimes are an exact
   passthrough with no LoRA compute and no autograd edges; a missing or
   unknown regime fails closed.
2. Additivity: zero-initialised ``B`` makes a fresh injection bitwise
   invisible; the delta scales exactly as ``alpha / r``; base weights are
   frozen and never mutated; there is no merge API anywhere.
3. Reversibility: ``handle.remove()`` restores the original modules
   bit-identically (same objects, same state_dict, same requires_grad flags).
4. Sidecar: ``fastwam-regime-lora-v1`` round-trips bitwise and refuses
   schema / parent-sha / path-set / r / alpha / key / shape / dtype
   mismatches.
"""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from fastwam.adapters.regime_lora import (
    REGIME_LORA_SIDECAR_SCHEMA,
    RegimeGatedLoRALinear,
    RegimeLoRAHandle,
    inject_regime_lora,
    load_regime_lora_sidecar,
    save_regime_lora_sidecar,
)
from fastwam.models.wan22.regime import (
    REGIME_IDM,
    REGIME_JOINT,
    REGIME_UNCOND,
    regime_call,
)

SHA_A = "ab" * 32
SHA_B = "cd" * 32

DTYPES = (torch.float32, torch.bfloat16)
DTYPE_IDS = ("float32", "bfloat16")


# --------------------------------------------------------------------------- #
# Minimal net that delivers the regime the same way MoT does: regime_call at
# every wrapped-submodule call site (mirrors mot.py's regime_call sites).
# --------------------------------------------------------------------------- #
class _Attn(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)


class _Block(nn.Module):
    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.self_attn = _Attn(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim)
        )


class _Net(nn.Module):
    def __init__(self, dim: int = 8, ffn_dim: int = 16, num_layers: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(dim, ffn_dim) for _ in range(num_layers)])
        self.head = nn.Linear(dim, 3)

    def forward(self, x: torch.Tensor, *, regime=None) -> torch.Tensor:
        for block in self.blocks:
            attn = regime_call(block.self_attn.q, x, regime=regime)
            x = x + regime_call(block.self_attn.o, attn, regime=regime)
            x = x + regime_call(block.ffn, x, regime=regime)
        return regime_call(self.head, x, regime=regime)


TARGETS = ("blocks.*.self_attn.q", "head")
TARGET_PATHS = ("blocks.0.self_attn.q", "blocks.1.self_attn.q", "head")


def _make_net(seed: int = 0, dtype: torch.dtype = torch.float32) -> _Net:
    torch.manual_seed(seed)
    net = _Net()
    if dtype is not torch.float32:
        net = net.to(dtype)
    return net


def _x(seed: int = 1, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(4, 8).to(dtype)


def _randomize_lora(handle: RegimeLoRAHandle, seed: int = 11) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        for param in handle.lora_parameters():
            param.normal_()


def _randomize_lora_module(wrapper: RegimeGatedLoRALinear, seed: int = 10) -> None:
    torch.manual_seed(seed)
    with torch.no_grad():
        wrapper.lora_A.weight.normal_()
        wrapper.lora_B.weight.normal_()


def _integer_fill(tensor: torch.Tensor, seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        tensor.copy_(
            torch.randint(-3, 4, tensor.shape, generator=generator).to(tensor.dtype)
        )


# --------------------------------------------------------------------------- #
# 1. RegimeGatedLoRALinear unit contracts
# --------------------------------------------------------------------------- #
class TestRegimeGatedLoRALinear:
    def test_constructor_validation(self):
        base = nn.Linear(4, 4)
        with pytest.raises(ValueError, match="positive integer"):
            RegimeGatedLoRALinear(base, r=0, alpha=1.0)
        with pytest.raises(ValueError, match="positive integer"):
            RegimeGatedLoRALinear(base, r=True, alpha=1.0)
        with pytest.raises(ValueError, match="finite positive"):
            RegimeGatedLoRALinear(base, r=2, alpha=0.0)
        with pytest.raises(ValueError, match="finite positive"):
            RegimeGatedLoRALinear(base, r=2, alpha=float("nan"))
        with pytest.raises(TypeError, match="nn.Linear"):
            RegimeGatedLoRALinear(nn.Conv1d(2, 2, 1), r=2, alpha=1.0)
        wrapper = RegimeGatedLoRALinear(nn.Linear(4, 4), r=2, alpha=1.0)
        with pytest.raises(ValueError, match="already injected"):
            RegimeGatedLoRALinear(wrapper, r=2, alpha=1.0)

    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    def test_zero_init_uncond_identity(self, dtype):
        torch.manual_seed(2)
        base = nn.Linear(8, 5).to(dtype)
        pristine = copy.deepcopy(base)
        wrapper = RegimeGatedLoRALinear(base, r=4, alpha=8.0)
        x = _x(seed=3, dtype=dtype)
        with torch.no_grad():
            ref = pristine(x)
            assert torch.equal(wrapper(x, regime=REGIME_UNCOND), ref)
            assert torch.equal(wrapper(x, regime=REGIME_IDM), ref)
        assert wrapper.lora_B.weight.abs().sum().item() == 0.0
        assert wrapper.lora_A.weight.dtype == dtype
        assert wrapper.lora_B.weight.dtype == dtype

    @pytest.mark.parametrize("regime", (REGIME_IDM, REGIME_JOINT))
    def test_non_uncond_passthrough_no_lora_autograd(self, regime):
        torch.manual_seed(4)
        base = nn.Linear(8, 5)
        pristine = copy.deepcopy(base)
        wrapper = RegimeGatedLoRALinear(base, r=4, alpha=8.0)
        _randomize_lora_module(wrapper)

        x = _x(seed=5).requires_grad_(True)
        out = wrapper(x, regime=regime)
        with torch.no_grad():
            assert torch.equal(out, pristine(x))
        out.square().mean().backward()
        # No autograd edges may touch the LoRA (or frozen base) parameters.
        assert wrapper.lora_A.weight.grad is None
        assert wrapper.lora_B.weight.grad is None
        assert base.weight.grad is None
        assert base.bias.grad is None
        assert x.grad is not None

    def test_uncond_nonzero_b_only_lora_gets_grads(self):
        torch.manual_seed(6)
        base = nn.Linear(8, 5)
        pristine = copy.deepcopy(base)
        wrapper = RegimeGatedLoRALinear(base, r=4, alpha=8.0)
        _randomize_lora_module(wrapper)

        x = _x(seed=7).requires_grad_(True)
        out = wrapper(x, regime=REGIME_UNCOND)
        with torch.no_grad():
            assert not torch.equal(out, pristine(x))
        out.square().mean().backward()
        assert wrapper.lora_A.weight.grad is not None
        assert wrapper.lora_B.weight.grad is not None
        assert base.weight.grad is None
        assert base.bias.grad is None

    def test_missing_regime_fails_closed(self):
        wrapper = RegimeGatedLoRALinear(nn.Linear(4, 4), r=2, alpha=4.0)
        with pytest.raises(ValueError, match="without a regime"):
            wrapper(torch.randn(2, 4))
        with pytest.raises(ValueError, match="without a regime"):
            wrapper(torch.randn(2, 4), regime=None)

    def test_unknown_regime_fails_closed(self):
        wrapper = RegimeGatedLoRALinear(nn.Linear(4, 4), r=2, alpha=4.0)
        with pytest.raises(ValueError, match="unknown regime 'banana'"):
            wrapper(torch.randn(2, 4), regime="banana")

    def test_base_never_mutated_by_lora_training(self):
        torch.manual_seed(8)
        base = nn.Linear(8, 5)
        weight_before = base.weight.detach().clone()
        bias_before = base.bias.detach().clone()
        wrapper = RegimeGatedLoRALinear(base, r=4, alpha=8.0)
        _randomize_lora_module(wrapper)

        x = _x(seed=9).requires_grad_(True)
        wrapper(x, regime=REGIME_UNCOND).square().mean().backward()
        with torch.no_grad():
            wrapper.lora_A.weight -= 0.1 * wrapper.lora_A.weight.grad
            wrapper.lora_B.weight -= 0.1 * wrapper.lora_B.weight.grad
        assert torch.equal(base.weight, weight_before)
        assert torch.equal(base.bias, bias_before)
        assert base.weight.requires_grad is False
        assert base.bias.requires_grad is False

    def test_alpha_over_r_scaling_doubles_delta_exactly(self):
        # Integer-valued tensors make every product/sum exactly representable
        # in float32, so "doubling alpha doubles the delta" is bitwise.
        base = nn.Linear(6, 5)
        _integer_fill(base.weight, seed=20)
        _integer_fill(base.bias, seed=21)
        single = RegimeGatedLoRALinear(copy.deepcopy(base), r=2, alpha=2.0)
        double = RegimeGatedLoRALinear(copy.deepcopy(base), r=2, alpha=4.0)
        for wrapper in (single, double):
            _integer_fill(wrapper.lora_A.weight, seed=22)
            _integer_fill(wrapper.lora_B.weight, seed=23)

        generator = torch.Generator().manual_seed(24)
        x = torch.randint(-3, 4, (4, 6), generator=generator).float()
        with torch.no_grad():
            ref = base(x)
            delta_single = single(x, regime=REGIME_UNCOND) - ref
            delta_double = double(x, regime=REGIME_UNCOND) - ref
        assert not torch.equal(delta_single, torch.zeros_like(delta_single))
        assert torch.equal(delta_double, 2.0 * delta_single)

    def test_no_merge_api_anywhere(self):
        # Merging the delta into base weights would silently change the IDM
        # branch and is permanently forbidden; assert no such API surface.
        forbidden = ("merge", "fold", "absorb")
        names = [
            name
            for name in list(dir(RegimeGatedLoRALinear)) + list(dir(RegimeLoRAHandle))
            if any(token in name.lower() for token in forbidden)
        ]
        assert names == []


# --------------------------------------------------------------------------- #
# 2. inject_regime_lora mechanics through the regime_call delivery path
# --------------------------------------------------------------------------- #
class TestInjectRegimeLora:
    @pytest.mark.parametrize("dtype", DTYPES, ids=DTYPE_IDS)
    def test_zero_init_injection_is_bitwise_invisible(self, dtype):
        net = _make_net(seed=0, dtype=dtype)
        x = _x(seed=1, dtype=dtype)
        with torch.no_grad():
            ref = net(x)  # pre-injection: plain Linears ignore the regime
        handle = inject_regime_lora(net, TARGETS, r=4, alpha=8.0)
        assert handle.wrapped_paths == TARGET_PATHS
        with torch.no_grad():
            assert torch.equal(net(x, regime=REGIME_UNCOND), ref)
            assert torch.equal(net(x, regime=REGIME_IDM), ref)

    def test_gating_follows_the_delivered_regime(self):
        net = _make_net(seed=0)
        x = _x(seed=1)
        with torch.no_grad():
            ref = net(x)
        handle = inject_regime_lora(net, TARGETS, r=4, alpha=8.0)
        _randomize_lora(handle)
        with torch.no_grad():
            assert torch.equal(net(x, regime=REGIME_IDM), ref)
            assert not torch.equal(net(x, regime=REGIME_UNCOND), ref)
        with pytest.raises(ValueError, match="without a regime"):
            net(x)  # regime never threaded => fail closed, not silent no-op
        with pytest.raises(ValueError, match="unknown regime 'banana'"):
            net(x, regime="banana")

    def test_uncond_backward_reaches_only_lora_in_wrapped_layers(self):
        net = _make_net(seed=0)
        handle = inject_regime_lora(net, TARGETS, r=4, alpha=8.0)
        _randomize_lora(handle)
        x = _x(seed=1)
        net(x, regime=REGIME_UNCOND).square().mean().backward()
        for param in handle.lora_parameters():
            assert param.grad is not None
        for path in handle.wrapped_paths:
            base = handle.wrapper(path).base
            for param in base.parameters():
                assert param.requires_grad is False
                assert param.grad is None

    def test_pattern_with_no_match_raises(self):
        net = _make_net()
        with pytest.raises(ValueError, match="matched no module"):
            inject_regime_lora(net, ("blocks.*.self_attn.qq",), r=2, alpha=4.0)

    def test_pattern_matching_non_linear_raises(self):
        net = _make_net()
        with pytest.raises(TypeError, match="not\\s+nn.Linear"):
            inject_regime_lora(net, ("blocks.0.self_attn",), r=2, alpha=4.0)

    def test_double_injection_raises(self):
        net = _make_net()
        inject_regime_lora(net, TARGETS, r=2, alpha=4.0)
        with pytest.raises(ValueError, match="already a RegimeGatedLoRALinear"):
            inject_regime_lora(net, TARGETS, r=2, alpha=4.0)

    def test_injecting_inside_existing_wrapper_raises(self):
        net = _make_net()
        inject_regime_lora(net, ("head",), r=2, alpha=4.0)
        with pytest.raises(ValueError, match="lives inside"):
            inject_regime_lora(net, ("head.base",), r=2, alpha=4.0)

    def test_empty_or_invalid_target_spec_raises(self):
        net = _make_net()
        with pytest.raises(ValueError, match="non-empty"):
            inject_regime_lora(net, (), r=2, alpha=4.0)
        with pytest.raises(ValueError, match="non-empty"):
            inject_regime_lora(net, ("",), r=2, alpha=4.0)

    def test_modulelist_and_sequential_children_wrap_and_restore(self):
        net = _make_net()
        original = net.get_submodule("blocks.0.ffn.0")
        handle = inject_regime_lora(net, ("blocks.0.ffn.0",), r=2, alpha=4.0)
        assert isinstance(net.get_submodule("blocks.0.ffn.0"), RegimeGatedLoRALinear)
        handle.remove()
        assert net.get_submodule("blocks.0.ffn.0") is original

    def test_remove_restores_bitwise_and_kills_handle(self):
        net = _make_net(seed=5)
        state_before = {k: v.detach().clone() for k, v in net.state_dict().items()}
        flags_before = {k: p.requires_grad for k, p in net.named_parameters()}
        originals = {path: net.get_submodule(path) for path in TARGET_PATHS}

        handle = inject_regime_lora(net, TARGETS, r=4, alpha=8.0)
        _randomize_lora(handle)
        handle.remove()

        state_after = net.state_dict()
        assert set(state_after) == set(state_before)
        for key, value in state_before.items():
            assert torch.equal(state_after[key], value), key
        assert {k: p.requires_grad for k, p in net.named_parameters()} == flags_before
        for path, module in originals.items():
            assert net.get_submodule(path) is module
        # The regime-free forward works again: no gated module remains.
        with torch.no_grad():
            net(_x())
        # Dead handle: every entry point fails closed.
        with pytest.raises(RuntimeError, match="handle is dead"):
            handle.remove()
        with pytest.raises(RuntimeError, match="handle is dead"):
            list(handle.lora_parameters())
        with pytest.raises(RuntimeError, match="handle is dead"):
            handle.lora_state_dict()

    def test_lora_parameter_enumeration(self):
        net = _make_net()
        handle = inject_regime_lora(net, TARGETS, r=4, alpha=8.0)
        named = list(handle.named_lora_parameters())
        assert [name for name, _ in named] == [
            "blocks.0.self_attn.q.lora_A.weight",
            "blocks.0.self_attn.q.lora_B.weight",
            "blocks.1.self_attn.q.lora_A.weight",
            "blocks.1.self_attn.q.lora_B.weight",
            "head.lora_A.weight",
            "head.lora_B.weight",
        ]
        params = list(handle.lora_parameters())
        assert len(params) == 6
        assert all(param.requires_grad for param in params)


# --------------------------------------------------------------------------- #
# 3. Sidecar I/O
# --------------------------------------------------------------------------- #
def _saved_sidecar(tmp_path, *, r: int = 4, alpha: float = 8.0):
    net = _make_net(seed=3)
    handle = inject_regime_lora(net, TARGETS, r=r, alpha=alpha)
    _randomize_lora(handle)
    path = tmp_path / "regime_lora.pt"
    save_regime_lora_sidecar(handle, path, SHA_A, step=123)
    return net, handle, path


def _tampered(src, dst, mutate):
    payload = torch.load(src, map_location="cpu", weights_only=True)
    mutate(payload)
    torch.save(payload, dst)
    return dst


class TestSidecar:
    def test_roundtrip_bitwise(self, tmp_path):
        net1, _, path = _saved_sidecar(tmp_path)
        x = _x(seed=7)
        with torch.no_grad():
            out1 = net1(x, regime=REGIME_UNCOND)

        net2 = _make_net(seed=3)
        handle2 = inject_regime_lora(net2, TARGETS, r=4, alpha=8.0)
        meta = load_regime_lora_sidecar(handle2, path, SHA_A)
        with torch.no_grad():
            out2 = net2(x, regime=REGIME_UNCOND)
        assert torch.equal(out1, out2)
        assert meta["schema"] == REGIME_LORA_SIDECAR_SCHEMA
        assert meta["step"] == 123
        assert meta["parent_checkpoint_sha256"] == SHA_A
        assert meta["torch_dtype"] == "torch.float32"
        assert meta["r"] == 4 and meta["alpha"] == 8.0
        assert "state_dict" not in meta

    def test_save_refuses_invalid_parent_sha(self, tmp_path):
        net = _make_net()
        handle = inject_regime_lora(net, TARGETS, r=2, alpha=4.0)
        for bad in (None, "abc", "AB" * 32, "zz" * 32, 123):
            with pytest.raises(ValueError, match="64-character lowercase hex"):
                save_regime_lora_sidecar(handle, tmp_path / "x.pt", bad)

    def test_load_refuses_wrong_parent_sha(self, tmp_path):
        _, _, path = _saved_sidecar(tmp_path)
        net2 = _make_net(seed=3)
        handle2 = inject_regime_lora(net2, TARGETS, r=4, alpha=8.0)
        with pytest.raises(ValueError, match="parent_checkpoint_sha256 mismatch"):
            load_regime_lora_sidecar(handle2, path, SHA_B)
        with pytest.raises(ValueError, match="64-character lowercase hex"):
            load_regime_lora_sidecar(handle2, path, "ZZ" * 32)

    def test_load_refuses_tampered_schema(self, tmp_path):
        _, _, path = _saved_sidecar(tmp_path)
        bad = _tampered(
            path, tmp_path / "schema.pt", lambda p: p.update(schema="fastwam-regime-lora-v2")
        )
        net2 = _make_net(seed=3)
        handle2 = inject_regime_lora(net2, TARGETS, r=4, alpha=8.0)
        with pytest.raises(ValueError, match="schema mismatch"):
            load_regime_lora_sidecar(handle2, bad, SHA_A)

    def test_load_refuses_wrong_shape(self, tmp_path):
        _, _, path = _saved_sidecar(tmp_path)

        def mutate(payload):
            payload["state_dict"]["head.lora_B.weight"] = torch.zeros(1, 1)

        bad = _tampered(path, tmp_path / "shape.pt", mutate)
        net2 = _make_net(seed=3)
        handle2 = inject_regime_lora(net2, TARGETS, r=4, alpha=8.0)
        with pytest.raises(ValueError, match="shape mismatch"):
            load_regime_lora_sidecar(handle2, bad, SHA_A)

    def test_load_refuses_wrong_dtype(self, tmp_path):
        _, _, path = _saved_sidecar(tmp_path)

        def mutate(payload):
            key = "head.lora_B.weight"
            payload["state_dict"][key] = payload["state_dict"][key].to(torch.float64)

        bad = _tampered(path, tmp_path / "dtype.pt", mutate)
        net2 = _make_net(seed=3)
        handle2 = inject_regime_lora(net2, TARGETS, r=4, alpha=8.0)
        with pytest.raises(ValueError, match="dtype mismatch"):
            load_regime_lora_sidecar(handle2, bad, SHA_A)

    def test_load_refuses_missing_or_unexpected_keys(self, tmp_path):
        _, _, path = _saved_sidecar(tmp_path)
        net2 = _make_net(seed=3)
        handle2 = inject_regime_lora(net2, TARGETS, r=4, alpha=8.0)

        missing = _tampered(
            path,
            tmp_path / "missing.pt",
            lambda p: p["state_dict"].pop("head.lora_A.weight"),
        )
        with pytest.raises(ValueError, match="state keys mismatch"):
            load_regime_lora_sidecar(handle2, missing, SHA_A)

        extra = _tampered(
            path,
            tmp_path / "extra.pt",
            lambda p: p["state_dict"].update(bogus=torch.zeros(1)),
        )
        with pytest.raises(ValueError, match="state keys mismatch"):
            load_regime_lora_sidecar(handle2, extra, SHA_A)

    def test_load_refuses_wrapped_path_set_mismatch(self, tmp_path):
        _, _, path = _saved_sidecar(tmp_path)
        net2 = _make_net(seed=3)
        handle2 = inject_regime_lora(net2, ("head",), r=4, alpha=8.0)
        with pytest.raises(ValueError, match="wrapped-path set mismatch"):
            load_regime_lora_sidecar(handle2, path, SHA_A)

    def test_load_refuses_r_and_alpha_mismatch(self, tmp_path):
        _, _, path = _saved_sidecar(tmp_path)
        net_r = _make_net(seed=3)
        handle_r = inject_regime_lora(net_r, TARGETS, r=2, alpha=8.0)
        with pytest.raises(ValueError, match="rank mismatch"):
            load_regime_lora_sidecar(handle_r, path, SHA_A)

        net_a = _make_net(seed=3)
        handle_a = inject_regime_lora(net_a, TARGETS, r=4, alpha=16.0)
        with pytest.raises(ValueError, match="alpha mismatch"):
            load_regime_lora_sidecar(handle_a, path, SHA_A)

    def test_dead_handle_refuses_io(self, tmp_path):
        net, handle, path = _saved_sidecar(tmp_path)
        handle.remove()
        with pytest.raises(RuntimeError, match="handle is dead"):
            save_regime_lora_sidecar(handle, tmp_path / "again.pt", SHA_A)
        with pytest.raises(RuntimeError, match="handle is dead"):
            load_regime_lora_sidecar(handle, path, SHA_A)


# --------------------------------------------------------------------------- #
# 4. Integration with the real W9 delivery mechanism (MoT grouped forwards)
# --------------------------------------------------------------------------- #
VIDEO_LEN = 4
FIRST_FRAME_LEN = 2
ACTION_LEN = 3
HIDDEN_DIM = 8
CONTEXT_LEN = 5

MOT_LORA_TARGET = "mixtures.action.blocks.*.self_attn.q"


def _build_mot_case():
    """Mirror tests/test_regime_protocol.py: a tiny MoT with tagged groups."""
    mot_module = pytest.importorskip(
        "fastwam.models.wan22.mot", reason="fastwam model deps unavailable (imageio chain)"
    )
    action_dit_module = pytest.importorskip(
        "fastwam.models.wan22.action_dit",
        reason="fastwam model deps unavailable (safetensors chain)",
    )
    MoT = mot_module.MoT
    ActionDiT = action_dit_module.ActionDiT
    MoTExpertSpan = mot_module.MoTExpertSpan
    MoTAttentionGroup = mot_module.MoTAttentionGroup

    def expert():
        return ActionDiT(
            hidden_dim=HIDDEN_DIM,
            action_dim=2,
            ffn_dim=16,
            text_dim=HIDDEN_DIM,
            freq_dim=8,
            eps=1e-6,
            num_heads=2,
            attn_head_dim=4,
            num_layers=2,
        )

    torch.manual_seed(7)
    model = MoT({"video": expert(), "action": expert()})
    model.eval()

    total = VIDEO_LEN + 2 * ACTION_LEN
    mask = torch.zeros((total, total), dtype=torch.bool)
    video_mask = torch.ones((VIDEO_LEN, VIDEO_LEN), dtype=torch.bool)
    video_mask[:FIRST_FRAME_LEN, FIRST_FRAME_LEN:] = False
    mask[:VIDEO_LEN, :VIDEO_LEN] = video_mask
    main_start, main_end = VIDEO_LEN, VIDEO_LEN + ACTION_LEN
    mask[main_start:main_end, :VIDEO_LEN] = True
    mask[main_start:main_end, main_start:main_end] = True
    base_start, base_end = main_end, main_end + ACTION_LEN
    mask[base_start:base_end, :FIRST_FRAME_LEN] = True
    mask[base_start:base_end, base_start:base_end] = True

    torch.manual_seed(99)
    batch = 2
    video = torch.randn(batch, VIDEO_LEN, HIDDEN_DIM)
    action = torch.randn(batch, 2 * ACTION_LEN, HIDDEN_DIM)
    video_freqs = model.mixtures["video"].freqs[:VIDEO_LEN].view(VIDEO_LEN, 1, -1)
    draft_freqs = model.mixtures["action"].freqs[:ACTION_LEN].view(ACTION_LEN, 1, -1)
    payload = {
        "embeds_all": {"video": video, "action": action},
        "attention_mask": mask,
        "freqs_all": {
            "video": video_freqs,
            "action": torch.cat((draft_freqs, draft_freqs), dim=0),
        },
        "context_all": {
            "video": {
                "context": torch.randn(batch, CONTEXT_LEN, HIDDEN_DIM),
                "mask": torch.ones(batch, VIDEO_LEN, CONTEXT_LEN, dtype=torch.bool),
            },
            "action": {
                "context": torch.randn(batch, CONTEXT_LEN, HIDDEN_DIM),
                "mask": torch.ones(batch, 2 * ACTION_LEN, CONTEXT_LEN, dtype=torch.bool),
            },
        },
        "t_mod_all": {
            "video": torch.randn(batch, VIDEO_LEN, 6, HIDDEN_DIM) * 0.1,
            "action": torch.randn(batch, 2 * ACTION_LEN, 6, HIDDEN_DIM) * 0.1,
        },
    }

    groups = (
        MoTAttentionGroup(
            "idm",
            (
                MoTExpertSpan("video", 0, VIDEO_LEN),
                MoTExpertSpan("action", 0, ACTION_LEN),
            ),
            regime=REGIME_IDM,
        ),
        MoTAttentionGroup(
            "base",
            (
                MoTExpertSpan("video", 0, FIRST_FRAME_LEN, write=False),
                MoTExpertSpan("action", ACTION_LEN, 2 * ACTION_LEN),
            ),
            regime=REGIME_UNCOND,
        ),
    )
    return model, payload, groups


class TestMoTIntegration:
    def test_zero_init_grouped_forward_bitwise_identical(self):
        model, payload, groups = _build_mot_case()
        plain = copy.deepcopy(model)
        handle = inject_regime_lora(model, MOT_LORA_TARGET, r=4, alpha=8.0)
        assert handle.wrapped_paths == (
            "mixtures.action.blocks.0.self_attn.q",
            "mixtures.action.blocks.1.self_attn.q",
        )
        with torch.no_grad():
            out_plain = plain(**payload, attention_groups=groups)
            out_lora = model(**payload, attention_groups=groups)
        assert torch.equal(out_lora["video"], out_plain["video"])
        assert torch.equal(out_lora["action"], out_plain["action"])

    def test_nonzero_lora_touches_only_uncond_action_rows(self):
        model, payload, groups = _build_mot_case()
        plain = copy.deepcopy(model)
        handle = inject_regime_lora(model, MOT_LORA_TARGET, r=4, alpha=8.0)
        _randomize_lora(handle, seed=13)
        with torch.no_grad():
            out_plain = plain(**payload, attention_groups=groups)
            out_lora = model(**payload, attention_groups=groups)
        # IDM group (video + first action span) is bitwise untouched even
        # with a nonzero adapter; only the uncond action span moves.
        assert torch.equal(out_lora["video"], out_plain["video"])
        assert torch.equal(
            out_lora["action"][:, :ACTION_LEN], out_plain["action"][:, :ACTION_LEN]
        )
        assert not torch.equal(
            out_lora["action"][:, ACTION_LEN:], out_plain["action"][:, ACTION_LEN:]
        )

    def test_dense_idm_forward_bitwise_identical_with_nonzero_lora(self):
        model, payload, _ = _build_mot_case()
        plain = copy.deepcopy(model)
        handle = inject_regime_lora(model, MOT_LORA_TARGET, r=4, alpha=8.0)
        _randomize_lora(handle, seed=13)
        with torch.no_grad():
            out_plain = plain(**payload, active_regime=REGIME_IDM)
            out_lora = model(**payload, active_regime=REGIME_IDM)
        assert torch.equal(out_lora["video"], out_plain["video"])
        assert torch.equal(out_lora["action"], out_plain["action"])

    def test_regime_free_forward_fails_closed(self):
        model, payload, _ = _build_mot_case()
        inject_regime_lora(model, MOT_LORA_TARGET, r=4, alpha=8.0)
        with pytest.raises(ValueError, match="without a regime"):
            with torch.no_grad():
                model(**payload)
