from __future__ import annotations

import math

import pytest
import torch

from fastwam.adapters import PolicyRegime
from fastwam.models.wan22.gate_transformer import (
    GateTransformer,
    GateTransformerConfig,
    LayerTapConfig,
    deterministic_idm_route,
    epsilon_mixture_bernoulli,
)
from fastwam.models.wan22.kv_tap import (
    GateKVSnapshot,
    GateLayerKV,
    KVSource,
    KeyValueBank,
)


def _bank(
    source: KVSource,
    tensor: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> KeyValueBank:
    if valid_mask is None:
        valid_mask = torch.ones(tensor.shape[:2], dtype=torch.bool)
    value = (tensor.detach() * 0.7 + 0.1).requires_grad_(tensor.requires_grad)
    return KeyValueBank(
        source=source,
        key=tensor,
        value=value,
        valid_mask=valid_mask,
    )


def _snapshot(
    *,
    seed: int,
    layer_indices: tuple[int, ...] = (0, 1, 2),
    timestep: float = 0.5,
    requires_grad: bool = False,
    video_mask: torch.Tensor | None = None,
) -> GateKVSnapshot:
    generator = torch.Generator().manual_seed(seed)
    layers = []
    for layer_index in layer_indices:
        video = torch.randn(2, 3, 4, generator=generator, requires_grad=requires_grad)
        action = torch.randn(2, 2, 4, generator=generator, requires_grad=requires_grad)
        context = torch.randn(2, 5, 4, generator=generator, requires_grad=requires_grad)
        layers.append(
            GateLayerKV(
                layer_index=layer_index,
                denoise_timestep=torch.full((2,), timestep),
                current_mode=(PolicyRegime.IDM, PolicyRegime.UNCOND),
                current_frame_video=_bank(
                    KVSource.CURRENT_FRAME_VIDEO,
                    video,
                    video_mask,
                ),
                action=_bank(KVSource.ACTION, action),
                context=_bank(KVSource.TEXT_STATE_CONTEXT, context),
                actor_version=7,
            )
        )
    return GateKVSnapshot(tuple(layers))


def _config(
    *,
    layer_taps: LayerTapConfig | None = None,
    denoise_last_n: int = 1,
    share_blocks: bool = False,
) -> GateTransformerConfig:
    return GateTransformerConfig(
        num_mot_layers=3,
        source_num_heads=2,
        source_head_dim=2,
        hidden_dim=8,
        num_query_tokens=2,
        ffn_multiplier=2,
        denoise_last_n=denoise_last_n,
        share_blocks=share_blocks,
        layer_taps=layer_taps or LayerTapConfig(),
    )


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (LayerTapConfig(mode="all"), (0, 1, 2, 3)),
        (LayerTapConfig(mode="last_n", last_n=2), (2, 3)),
        (LayerTapConfig(mode="indices", indices=(0, 3)), (0, 3)),
    ],
)
def test_layer_selector_modes(config: LayerTapConfig, expected: tuple[int, ...]) -> None:
    assert config.resolve(4) == expected


@pytest.mark.parametrize(
    "config",
    [
        LayerTapConfig(mode="unknown"),
        LayerTapConfig(mode="last_n", last_n=0),
        LayerTapConfig(mode="last_n", last_n=5),
        LayerTapConfig(mode="indices", indices=()),
        LayerTapConfig(mode="indices", indices=(2, 1)),
        LayerTapConfig(mode="indices", indices=(1, 1)),
        LayerTapConfig(mode="indices", indices=(4,)),
    ],
)
def test_invalid_layer_selectors_fail_closed(config: LayerTapConfig) -> None:
    with pytest.raises(ValueError):
        config.resolve(4)


def test_gate_masks_invalid_source_tokens() -> None:
    torch.manual_seed(9)
    gate = GateTransformer(_config(layer_taps=LayerTapConfig(mode="indices", indices=(0,))))
    video_mask = torch.tensor([[True, False, True], [True, False, True]])
    original = _snapshot(seed=4, layer_indices=(0,), video_mask=video_mask)

    original_layer = original.layers[0]
    changed_key = original_layer.current_frame_video.key.detach().clone()
    changed_value = original_layer.current_frame_video.value.detach().clone()
    changed_key[:, 1] = 1_000_000
    changed_value[:, 1] = -1_000_000
    changed_video = KeyValueBank(
        source=KVSource.CURRENT_FRAME_VIDEO,
        key=changed_key,
        value=changed_value,
        valid_mask=video_mask,
    )
    changed = GateKVSnapshot(
        (
            GateLayerKV(
                layer_index=0,
                denoise_timestep=original_layer.denoise_timestep,
                current_mode=original_layer.current_mode,
                current_frame_video=changed_video,
                action=original_layer.action,
                context=original_layer.context,
                actor_version=original_layer.actor_version,
            ),
        )
    )

    assert torch.equal(gate([original]), gate([changed]))


def test_current_frame_bank_rejects_direct_future_video() -> None:
    tensor = torch.zeros(1, 2, 4)
    with pytest.raises(ValueError, match="cannot contain generated future"):
        KeyValueBank(
            source=KVSource.CURRENT_FRAME_VIDEO,
            key=tensor,
            value=tensor,
            valid_mask=torch.ones(1, 2, dtype=torch.bool),
            contains_generated_future_video=True,
        )


def test_gate_detaches_every_source_bank() -> None:
    torch.manual_seed(3)
    snapshot = _snapshot(seed=5, requires_grad=True)
    source_leaves = [
        tensor
        for layer in snapshot.layers
        for bank in (layer.current_frame_video, layer.action, layer.context)
        for tensor in (bank.key, bank.value)
    ]
    gate = GateTransformer(_config())

    gate([snapshot]).sum().backward()

    assert all(tensor.grad is None for tensor in source_leaves)
    assert gate.query_tokens.grad is not None
    assert torch.isfinite(gate.query_tokens.grad).all()


def test_denoising_taps_average_logits_not_probabilities() -> None:
    logits = torch.tensor([[-4.0, 1.0], [2.0, 3.0]])
    aggregated = GateTransformer.aggregate_denoise_logits(logits)

    assert torch.equal(aggregated, torch.tensor([-1.0, 2.0]))
    assert not torch.allclose(torch.sigmoid(aggregated), torch.sigmoid(logits).mean(dim=0))


def test_gate_uses_only_configured_last_denoising_taps() -> None:
    torch.manual_seed(11)
    gate = GateTransformer(_config(denoise_last_n=2))
    snapshots = [
        _snapshot(seed=1, timestep=0.9),
        _snapshot(seed=2, timestep=0.5),
        _snapshot(seed=3, timestep=0.1),
    ]

    expected = torch.stack(
        [gate.forward_snapshot(snapshot) for snapshot in snapshots[-2:]],
        dim=0,
    ).mean(dim=0)

    assert torch.allclose(gate(snapshots), expected)
    with pytest.raises(ValueError, match="requires the last 2"):
        gate(snapshots[:1])


def test_gate_supports_shared_blocks_and_mixed_mode_metadata() -> None:
    gate = GateTransformer(_config(share_blocks=True))
    snapshot = _snapshot(seed=12)

    logits = gate([snapshot])

    assert logits.shape == (2,)
    assert len(gate.blocks) == 1


def test_epsilon_mixture_distribution_and_log_prob_are_exact() -> None:
    logits = torch.tensor([math.log(4.0), 0.0])
    behavior = epsilon_mixture_bernoulli(
        logits,
        temperature=1.0,
        epsilon=0.25,
    )

    expected_base = torch.tensor([0.8, 0.5])
    expected_behavior = 0.75 * expected_base + 0.125
    assert torch.allclose(behavior.base_idm_probability, expected_base)
    assert torch.allclose(behavior.behavior_idm_probability, expected_behavior)

    route = torch.tensor([1, 0])
    expected_log_prob = torch.stack(
        [expected_behavior[0].log(), (1 - expected_behavior[1]).log()]
    )
    assert torch.allclose(behavior.log_prob(route), expected_log_prob)

    generator = torch.Generator().manual_seed(42)
    sample = behavior.sample(generator=generator)
    assert sample.shape == logits.shape
    assert set(sample.tolist()) <= {0, 1}


def test_gate_probability_validators_and_deterministic_threshold() -> None:
    logits = torch.tensor([-2.0, 0.0, 2.0])
    assert torch.equal(
        deterministic_idm_route(logits, threshold=0.5),
        torch.tensor([0, 1, 1]),
    )

    with pytest.raises(ValueError, match="temperature"):
        epsilon_mixture_bernoulli(logits, temperature=0.0, epsilon=0.1)
    with pytest.raises(ValueError, match="epsilon"):
        epsilon_mixture_bernoulli(logits, temperature=1.0, epsilon=1.1)
    with pytest.raises(ValueError, match="threshold"):
        deterministic_idm_route(logits, threshold=1.1)


def test_snapshot_to_preserves_metadata_and_moves_dtype() -> None:
    snapshot = _snapshot(seed=22)

    moved = snapshot.to(device="cpu", dtype=torch.float64)

    assert moved.layers[0].action.key.dtype == torch.float64
    assert moved.layers[0].denoise_timestep.dtype == torch.float64
    assert moved.layers[0].current_mode == snapshot.layers[0].current_mode
    assert moved.layers[0].actor_version == 7
    assert moved.nbytes > 0


def test_tensor_temperature_and_epsilon_are_applied_per_batch_item() -> None:
    logits = torch.tensor([-2.0, 2.0])
    temperature = torch.tensor([2.0, 0.5])
    epsilon = torch.tensor([0.0, 1.0])

    behavior = epsilon_mixture_bernoulli(
        logits,
        temperature=temperature,
        epsilon=epsilon,
    )

    torch.testing.assert_close(
        behavior.base_idm_probability,
        torch.sigmoid(torch.tensor([-1.0, 4.0])),
    )
    torch.testing.assert_close(
        behavior.behavior_idm_probability,
        torch.tensor([torch.sigmoid(torch.tensor(-1.0)), 0.5]),
    )
    assert torch.equal(
        deterministic_idm_route(
            logits,
            temperature=temperature,
            threshold=0.75,
        ),
        torch.tensor([0, 1]),
    )


def test_policy_parameters_cannot_accidentally_expand_logits() -> None:
    logits = torch.zeros(2, 1)
    with pytest.raises(ValueError, match="expand logits"):
        epsilon_mixture_bernoulli(
            logits,
            temperature=torch.ones(2),
            epsilon=0.1,
        )


def test_bfloat16_stored_kv_runs_with_float32_gate_parameters() -> None:
    torch.manual_seed(31)
    gate = GateTransformer(_config())
    snapshot = _snapshot(seed=32)
    stored_snapshot = snapshot.detached().to(device="cpu", dtype=torch.bfloat16)

    reference = gate([snapshot])
    replayed = gate([stored_snapshot])

    assert replayed.dtype == torch.float32
    assert torch.isfinite(replayed).all()
    torch.testing.assert_close(replayed, reference, atol=2e-2, rtol=2e-2)


def test_key_value_bank_rejects_mixed_dtypes() -> None:
    with pytest.raises(TypeError, match="dtypes must match"):
        KeyValueBank(
            source=KVSource.ACTION,
            key=torch.zeros(1, 2, 4, dtype=torch.float32),
            value=torch.zeros(1, 2, 4, dtype=torch.bfloat16),
            valid_mask=torch.ones(1, 2, dtype=torch.bool),
        )
