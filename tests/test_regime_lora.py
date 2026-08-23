import copy

import pytest
import torch
from fastwam.adapters import (
    LORA_MASTER_DTYPE,
    ActionLoRATargetGroup,
    PolicyRegime,
    RegimeContext,
    RegimeLoRAConfig,
    RegimeLoRALinear,
    discover_action_dit_lora_targets,
    inject_action_dit_lora,
    sha256_file,
)
from torch import nn


class TinyAttention(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.o = nn.Linear(hidden_dim, hidden_dim)


class TinyBlock(nn.Module):
    def __init__(self, hidden_dim: int, ffn_dim: int) -> None:
        super().__init__()
        self.self_attn = TinyAttention(hidden_dim)
        self.cross_attn = TinyAttention(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )


class TinyActionDiT(nn.Module):
    def __init__(
        self, *, num_layers: int = 2, hidden_dim: int = 8, ffn_dim: int = 12
    ) -> None:
        super().__init__()
        self.action_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [TinyBlock(hidden_dim, ffn_dim) for _ in range(num_layers)]
        )
        self.head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = self.action_encoder(action)
        for block in self.blocks:
            self_mix = (
                block.self_attn.q(x) + block.self_attn.k(x) + block.self_attn.v(x)
            )
            x = x + block.self_attn.o(torch.tanh(self_mix))
            cross_mix = (
                block.cross_attn.q(x)
                + block.cross_attn.k(context)
                + block.cross_attn.v(context)
            )
            x = x + block.cross_attn.o(torch.tanh(cross_mix))
            x = x + block.ffn(x)
        return self.head(x)


class FSDPStyleWrapper(nn.Module):
    """Minimal forwarding wrapper with classic FSDP's child attribute."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self._fsdp_wrapped_module = module

    def forward(self, *args, **kwargs):
        return self._fsdp_wrapped_module(*args, **kwargs)


def test_regime_context_is_nested_instance_scoped_and_exception_safe() -> None:
    context = RegimeContext()
    other = RegimeContext(PolicyRegime.UNCOND)

    assert context.current is PolicyRegime.IDM
    assert other.current is PolicyRegime.UNCOND
    with context.use(PolicyRegime.UNCOND):
        assert context.current is PolicyRegime.UNCOND
        assert other.current is PolicyRegime.UNCOND
        with context.use(PolicyRegime.IDM):
            assert context.current is PolicyRegime.IDM
        assert context.current is PolicyRegime.UNCOND
    assert context.current is PolicyRegime.IDM

    with pytest.raises(RuntimeError, match="forward failed"):
        with context.use(PolicyRegime.UNCOND):
            raise RuntimeError("forward failed")
    assert context.current is PolicyRegime.IDM


def test_default_target_discovery_matches_action_dit_block_contract() -> None:
    model = TinyActionDiT(num_layers=2)

    targets = discover_action_dit_lora_targets(model)

    assert len(targets) == 20
    assert "blocks.0.self_attn.q" in targets
    assert "blocks.1.self_attn.o" in targets
    assert "blocks.0.cross_attn.k" in targets
    assert "blocks.1.cross_attn.v" in targets
    assert "blocks.0.ffn.0" in targets
    assert "blocks.1.ffn.2" in targets
    assert all("action_encoder" not in name and "head" not in name for name in targets)

    ffn_only = discover_action_dit_lora_targets(
        model,
        (ActionLoRATargetGroup.FFN,),
    )
    assert ffn_only == (
        "blocks.0.ffn.0",
        "blocks.0.ffn.2",
        "blocks.1.ffn.0",
        "blocks.1.ffn.2",
    )


def test_lora_delta_uses_alpha_over_rank_scaling() -> None:
    base = nn.Linear(3, 2, bias=False)
    context = RegimeContext()
    layer = RegimeLoRALinear(
        base,
        regime_context=context,
        rank=2,
        alpha=4.0,
        dropout=0.0,
    )
    with torch.no_grad():
        layer.weight.zero_()
        layer.lora_A.fill_(1.0)
        layer.lora_B.fill_(1.0)
    inputs = torch.tensor([[1.0, 2.0, 3.0]])

    with context.use(PolicyRegime.IDM):
        idm_output = layer(inputs)
    with context.use(PolicyRegime.UNCOND):
        uncond_output = layer(inputs)

    assert torch.equal(idm_output, torch.zeros_like(idm_output))
    # A produces [6, 6], B sums it to 12, then alpha/rank scales by 2.
    assert torch.equal(uncond_output, torch.full_like(uncond_output, 24.0))


def test_zero_delta_is_exact_and_idm_never_uses_nonzero_adapter() -> None:
    torch.manual_seed(7)
    model = TinyActionDiT()
    baseline = copy.deepcopy(model)
    original_base_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if ".self_attn." in name or ".cross_attn." in name or ".ffn." in name
    }
    adapter = inject_action_dit_lora(
        model,
        RegimeLoRAConfig(rank=2, alpha=4.0),
    )
    action = torch.randn(2, 3, 8)
    context = torch.randn(2, 3, 8)
    expected = baseline(action, context)

    with adapter.use_regime(PolicyRegime.IDM):
        idm_zero = model(action, context)
    with adapter.use_regime(PolicyRegime.UNCOND):
        uncond_zero = model(action, context)

    assert torch.equal(idm_zero, expected)
    assert torch.equal(uncond_zero, expected)
    for name, original in original_base_parameters.items():
        assert model.get_parameter(name) is original

    with torch.no_grad():
        for _, layer in adapter.iter_adapted_linears():
            layer.lora_B.fill_(0.125)

    with adapter.use_regime(PolicyRegime.IDM):
        idm_nonzero_adapter = model(action, context)
    with adapter.use_regime(PolicyRegime.UNCOND):
        uncond_nonzero_adapter = model(action, context)

    assert torch.equal(idm_nonzero_adapter, expected)
    assert not torch.equal(uncond_nonzero_adapter, expected)
    assert adapter.regime_context.current is PolicyRegime.IDM
    assert not hasattr(adapter, "merge")
    assert all(
        isinstance(model.get_submodule(name), RegimeLoRALinear)
        for name in adapter.target_names
    )
    copied_model = copy.deepcopy(model)
    copied_contexts = {
        id(copied_model.get_submodule(name).regime_context)
        for name in adapter.target_names
    }
    assert len(copied_contexts) == 1


def test_adapter_state_is_stable_through_fsdp_style_projection_wrapper() -> None:
    model = TinyActionDiT(num_layers=1)
    adapter = inject_action_dit_lora(
        model,
        RegimeLoRAConfig(rank=2, alpha=2.0),
    )
    target = adapter.target_names[0]
    parent_path, _, child_name = target.rpartition(".")
    parent = model.get_submodule(parent_path)
    adapted = model.get_submodule(target)
    setattr(parent, child_name, FSDPStyleWrapper(adapted))

    resolved = dict(adapter.iter_adapted_linears())
    state = adapter.lora_state_dict()

    assert resolved[target] is adapted
    assert set(state) == {name for name, _ in adapter.named_lora_parameters()}
    assert all(name.endswith((".lora_A", ".lora_B")) for name in state)


def test_uncond_backward_updates_only_lora_and_idm_does_not_touch_it() -> None:
    torch.manual_seed(11)
    model = TinyActionDiT()
    adapter = inject_action_dit_lora(
        model,
        RegimeLoRAConfig(rank=2, alpha=2.0),
    )
    audit = adapter.audit_freeze()
    audit.assert_valid()
    assert audit.trainable_base == ()

    action = torch.randn(2, 3, 8, requires_grad=True)
    context = torch.randn(2, 3, 8)
    with adapter.use_regime(PolicyRegime.IDM):
        model(action, context).sum().backward()
    assert all(parameter.grad is None for parameter in adapter.lora_parameters())
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.endswith((".lora_A", ".lora_B"))
    )

    model.zero_grad(set_to_none=True)
    action.grad = None
    with adapter.use_regime(PolicyRegime.UNCOND):
        model(action, context).square().mean().backward()

    lora_gradients = {
        name: parameter.grad for name, parameter in adapter.named_lora_parameters()
    }
    assert all(gradient is not None for gradient in lora_gradients.values())
    assert any(
        torch.count_nonzero(gradient).item() > 0
        for name, gradient in lora_gradients.items()
        if name.endswith(".lora_B")
    )
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.endswith((".lora_A", ".lora_B"))
    )


def test_sidecar_round_trip_contains_only_lora_and_checks_parent_hash(tmp_path) -> None:
    torch.manual_seed(13)
    config = RegimeLoRAConfig(rank=2, alpha=3.0)
    model = TinyActionDiT()
    adapter = inject_action_dit_lora(model, config)
    with torch.no_grad():
        for _, parameter in adapter.named_lora_parameters():
            parameter.normal_()

    parent_checkpoint = tmp_path / "parent.pt"
    parent_checkpoint.write_bytes(b"frozen-fastwam-parent")
    parent_hash = sha256_file(parent_checkpoint)
    sidecar = tmp_path / "uncond_lora.pt"
    adapter.save_sidecar(
        sidecar,
        parent_checkpoint_sha256=parent_hash,
        extra_metadata={"training_step": 17},
    )

    payload = torch.load(sidecar, map_location="cpu", weights_only=True)
    assert set(payload) == {"metadata", "state_dict"}
    assert payload["metadata"]["parent_checkpoint_sha256"] == parent_hash
    assert payload["metadata"]["active_regime"] == PolicyRegime.UNCOND.value
    assert payload["metadata"]["extra"] == {"training_step": 17}
    assert payload["state_dict"]
    assert all(name.endswith((".lora_A", ".lora_B")) for name in payload["state_dict"])
    assert not any(
        name.endswith((".weight", ".bias")) for name in payload["state_dict"]
    )

    restored_model = TinyActionDiT()
    restored = inject_action_dit_lora(restored_model, config)
    metadata = restored.load_sidecar(
        sidecar,
        expected_parent_checkpoint_sha256=parent_hash,
    )
    assert metadata["schema"] == "fastwam-regime-lora-v1"
    for name, value in adapter.lora_state_dict().items():
        assert torch.equal(restored.lora_state_dict()[name], value)

    with pytest.raises(ValueError, match="parent checkpoint mismatch"):
        restored.load_sidecar(
            sidecar,
            expected_parent_checkpoint_sha256="0" * 64,
        )


def test_behavior_replay_reference_restores_live_lora_after_scope() -> None:
    model = TinyActionDiT()
    adapter = inject_action_dit_lora(
        model,
        RegimeLoRAConfig(rank=2, alpha=2.0),
    )
    with torch.no_grad():
        for _, parameter in adapter.named_lora_parameters():
            parameter.fill_(1.0)
    adapter.capture_replay_reference(actor_version=4)

    with torch.no_grad():
        for _, parameter in adapter.named_lora_parameters():
            parameter.fill_(2.0)
    live_state = adapter.lora_state_dict()

    with adapter.use_replay_reference(actor_version=4):
        assert all(
            torch.equal(value, torch.ones_like(value))
            for value in adapter.lora_state_dict().values()
        )

    assert all(
        torch.equal(adapter.lora_state_dict()[name], value)
        for name, value in live_state.items()
    )
    with pytest.raises(ValueError, match="actor version mismatch"):
        with adapter.use_replay_reference(actor_version=5):
            pass


def test_lora_factors_are_fp32_master_weights_under_a_bf16_base():
    """A BF16 factor would discard every optimizer step below half its ULP."""

    base = nn.Linear(256, 128, dtype=torch.bfloat16)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    adapted = RegimeLoRALinear(
        base,
        regime_context=RegimeContext(),
        rank=16,
        alpha=16,
    )

    assert adapted.weight.dtype is torch.bfloat16
    assert adapted.lora_A.dtype is LORA_MASTER_DTYPE
    assert adapted.lora_B.dtype is LORA_MASTER_DTYPE


def test_lora_delta_is_computed_in_the_frozen_base_dtype():
    """FP32 storage must not change the adapter's arithmetic precision."""

    base = nn.Linear(256, 128, dtype=torch.bfloat16)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    context = RegimeContext()
    adapted = RegimeLoRALinear(base, regime_context=context, rank=16, alpha=16)
    inputs = torch.randn(2, 4, 256, dtype=torch.bfloat16)

    with context.use(PolicyRegime.UNCOND):
        uncond = adapted(inputs)
    with context.use(PolicyRegime.IDM):
        idm = adapted(inputs)

    assert uncond.dtype is torch.bfloat16
    # `lora_B` is zero-initialized, so the adapter is an exact no-op at step 0.
    assert torch.equal(uncond, idm)


def test_small_optimizer_step_moves_every_lora_parameter():
    """The defect this guards: a BF16 factor never left its initialization."""

    base = nn.Linear(3072, 1024, dtype=torch.bfloat16)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    context = RegimeContext()
    adapted = RegimeLoRALinear(base, regime_context=context, rank=16, alpha=16)
    optimizer = torch.optim.AdamW(
        [adapted.lora_A, adapted.lora_B],
        lr=1e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    inputs = torch.randn(2, 8, 3072, dtype=torch.bfloat16)
    before = adapted.lora_A.detach().clone()

    with context.use(PolicyRegime.UNCOND):
        adapted(inputs).float().square().mean().backward()
    optimizer.step()

    assert torch.all(adapted.lora_A != before)
