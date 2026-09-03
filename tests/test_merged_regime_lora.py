"""Tests for offline ActionDiT LoRA materialization."""

from __future__ import annotations

import copy

import pytest
import torch
from fastwam.adapters import (
    ActionDiTLoRAAdapter,
    PolicyRegime,
    RegimeContext,
    RegimeLoRAConfig,
    RegimeLoRALinear,
    inject_action_dit_lora,
    load_frozen_uncond_action_artifact,
    merge_action_dit_lora_,
    save_frozen_uncond_action_artifact,
)
from torch import nn


class TinyAttention(nn.Module):
    """Small attention-shaped module matching ActionDiT target names."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.o = nn.Linear(hidden_dim, hidden_dim)


class TinyBlock(nn.Module):
    """Small ActionDiT-shaped block."""

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
    """Minimal forward-compatible ActionDiT stand-in."""

    def __init__(self, hidden_dim: int = 4) -> None:
        super().__init__()
        self.action_encoder = nn.Linear(hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList([TinyBlock(hidden_dim, hidden_dim * 2)])
        self.head = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Run all target projections so dynamic/static outputs can be compared."""

        value = self.action_encoder(action)
        for block in self.blocks:
            self_mix = block.self_attn.q(value)
            self_mix = self_mix + block.self_attn.k(value)
            self_mix = self_mix + block.self_attn.v(value)
            value = value + block.self_attn.o(torch.tanh(self_mix))
            cross_mix = block.cross_attn.q(value)
            cross_mix = cross_mix + block.cross_attn.k(context)
            cross_mix = cross_mix + block.cross_attn.v(context)
            value = value + block.cross_attn.o(torch.tanh(cross_mix))
            value = value + block.ffn(value)
        return self.head(value)


def _loaded_adapter(*, dropout: float = 0.0) -> ActionDiTLoRAAdapter:
    torch.manual_seed(11)
    model = TinyActionDiT()
    adapter = inject_action_dit_lora(
        model,
        RegimeLoRAConfig(rank=2, alpha=4.0, dropout=dropout),
    )
    with torch.no_grad():
        for index, (_, layer) in enumerate(adapter.iter_adapted_linears(), start=1):
            layer.lora_A.fill_(0.01 * index)
            layer.lora_B.fill_(0.02 * index)
    return adapter


def test_fp32_dynamic_uncond_matches_plain_merged_action_dit() -> None:
    adapter = _loaded_adapter()
    action = torch.randn(2, 3, 4)
    context = torch.randn(2, 3, 4)
    with adapter.use_regime(PolicyRegime.UNCOND):
        expected = adapter.action_dit(action, context)

    audit = merge_action_dit_lora_(adapter)
    actual = adapter.action_dit(action, context)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert audit.target_count == 10
    assert audit.maximum_absolute_delta > 0.0
    assert audit.mean_absolute_delta > 0.0
    assert audit.output_dtype == "torch.float32"
    assert all(
        not parameter.requires_grad for parameter in adapter.action_dit.parameters()
    )
    assert not any(
        isinstance(module, RegimeLoRALinear) for module in adapter.action_dit.modules()
    )
    assert not any(
        name.endswith((".lora_A", ".lora_B"))
        for name, _ in adapter.action_dit.named_parameters()
    )


def test_merge_applies_alpha_over_rank_once_and_preserves_bias() -> None:
    adapter = _loaded_adapter()
    target_name, layer = next(adapter.iter_adapted_linears())
    original_bias = layer.bias.detach().clone()
    original_weight = layer.weight.detach().clone()
    expected_delta = (layer.lora_B @ layer.lora_A) * (layer.alpha / layer.rank)

    merge_action_dit_lora_(adapter)
    merged = adapter.action_dit.get_submodule(target_name)

    assert type(merged) is nn.Linear
    assert torch.equal(merged.bias, original_bias)
    torch.testing.assert_close(
        merged.weight,
        original_weight + expected_delta,
        rtol=0.0,
        atol=0.0,
    )


def test_merge_rejects_nonzero_dropout() -> None:
    adapter = _loaded_adapter(dropout=0.1)

    with pytest.raises(ValueError, match="dropout 0"):
        merge_action_dit_lora_(adapter)


def test_merge_rejects_changed_factor_shape() -> None:
    adapter = _loaded_adapter()
    _, layer = next(adapter.iter_adapted_linears())
    layer.lora_A = nn.Parameter(torch.zeros(layer.rank, layer.in_features + 1))

    with pytest.raises(ValueError, match="LoRA A shape mismatch"):
        merge_action_dit_lora_(adapter)


def test_merge_rejects_incomplete_target_coverage() -> None:
    adapter = _loaded_adapter()
    incomplete = ActionDiTLoRAAdapter(
        adapter.action_dit,
        config=adapter.config,
        regime_context=RegimeContext(),
        target_names=adapter.target_names[:-1],
    )

    with pytest.raises(ValueError, match="target coverage changed"):
        merge_action_dit_lora_(incomplete)


def test_merge_does_not_mutate_independent_idm_action_dit() -> None:
    torch.manual_seed(23)
    idm = TinyActionDiT()
    idm_reference = copy.deepcopy(idm.state_dict())
    uncond_copy = copy.deepcopy(idm)
    adapter = inject_action_dit_lora(
        uncond_copy,
        RegimeLoRAConfig(rank=2, alpha=2.0),
    )
    with torch.no_grad():
        for _, layer in adapter.iter_adapted_linears():
            layer.lora_A.fill_(0.25)
            layer.lora_B.fill_(0.5)

    merge_action_dit_lora_(adapter)

    assert all(
        torch.equal(value, idm_reference[name])
        for name, value in idm.state_dict().items()
    )
    assert any(
        not torch.equal(value, idm_reference[name])
        for name, value in uncond_copy.state_dict().items()
        if name in idm_reference
    )


def test_frozen_uncond_action_artifact_strict_round_trip(tmp_path) -> None:
    adapter = _loaded_adapter()
    merge_audit = merge_action_dit_lora_(adapter)
    action_config = {"kind": "tiny", "hidden_dim": 4}
    parent_hash = "1" * 64
    sidecar_hash = "2" * 64
    sidecar_metadata = {
        "schema": "fastwam-regime-lora-v1",
        "parent_checkpoint_sha256": parent_hash,
        "rank": 2,
        "alpha": 4.0,
        "dropout": 0.0,
        "target_groups": [
            "self_attention_qkvo",
            "cross_attention_qkvo",
            "ffn",
        ],
        "target_names": list(merge_audit.target_names),
    }
    artifact = tmp_path / "warm_uncond_action.pt"
    save_frozen_uncond_action_artifact(
        artifact,
        action_dit=adapter.action_dit,
        action_dit_config=action_config,
        parent_checkpoint_sha256=parent_hash,
        source_lora_sidecar_sha256=sidecar_hash,
        source_lora_metadata=sidecar_metadata,
        merge_audit=merge_audit,
    )
    restored = TinyActionDiT()
    metadata = load_frozen_uncond_action_artifact(
        artifact,
        action_dit=restored,
        expected_action_dit_config=action_config,
        expected_parent_checkpoint_sha256=parent_hash,
        expected_source_lora_sidecar_sha256=sidecar_hash,
    )

    assert metadata["schema"] == "fastwam-frozen-uncond-action-v1"
    assert all(
        torch.equal(value, restored.state_dict()[name])
        for name, value in adapter.action_dit.state_dict().items()
    )
    assert all(not parameter.requires_grad for parameter in restored.parameters())
    assert not restored.training


def test_frozen_uncond_action_artifact_rejects_wrong_parent(tmp_path) -> None:
    adapter = _loaded_adapter()
    merge_audit = merge_action_dit_lora_(adapter)
    artifact = tmp_path / "warm_uncond_action.pt"
    parent_hash = "1" * 64
    save_frozen_uncond_action_artifact(
        artifact,
        action_dit=adapter.action_dit,
        action_dit_config={"kind": "tiny"},
        parent_checkpoint_sha256=parent_hash,
        source_lora_sidecar_sha256="2" * 64,
        source_lora_metadata={
            "schema": "fastwam-regime-lora-v1",
            "parent_checkpoint_sha256": parent_hash,
            "rank": 2,
            "alpha": 4.0,
            "target_groups": [
                "self_attention_qkvo",
                "cross_attention_qkvo",
                "ffn",
            ],
            "target_names": list(merge_audit.target_names),
        },
        merge_audit=merge_audit,
    )

    with pytest.raises(ValueError, match="parent mismatch"):
        load_frozen_uncond_action_artifact(
            artifact,
            action_dit=TinyActionDiT(),
            expected_action_dit_config={"kind": "tiny"},
            expected_parent_checkpoint_sha256="3" * 64,
        )
