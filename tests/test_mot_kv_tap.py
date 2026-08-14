from __future__ import annotations

import importlib.util
import sys
import types

import pytest
import torch
from torch import nn

if importlib.util.find_spec("safetensors") is None:
    safetensors_stub = types.ModuleType("safetensors")
    safetensors_stub.safe_open = None
    sys.modules["safetensors"] = safetensors_stub
if importlib.util.find_spec("imageio") is None:
    imageio_stub = types.ModuleType("imageio")
    imageio_stub.get_writer = None
    sys.modules["imageio"] = imageio_stub

from fastwam.adapters import (
    ActionLoRATargetGroup,
    PolicyRegime,
    RegimeLoRAConfig,
    inject_action_dit_lora,
)
from fastwam.models.wan22.adaptive_action import (
    CachedActionCondition,
    CachedActionVelocity,
    ModalityKeepMask,
    VisualReadCondition,
)
from fastwam.models.wan22.kv_tap import GateKVTapRequest, KVSource
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.visual_contracts import (
    ActionLayerReadContext,
    ActionVisualReader,
    NativePatchMemory,
    VisualResidual,
)
from fastwam.models.wan22.wan_video_dit import DiTBlock, modulate


class _TinyExpert(nn.Module):
    def __init__(self, num_layers: int = 2) -> None:
        super().__init__()
        self.num_heads = 2
        self.attn_head_dim = 4
        self.use_gradient_checkpointing = False
        self.blocks = nn.ModuleList(
            DiTBlock(
                hidden_dim=8,
                attn_head_dim=self.attn_head_dim,
                num_heads=self.num_heads,
                ffn_dim=16,
            )
            for _ in range(num_layers)
        )


def _mot() -> MoT:
    torch.manual_seed(5)
    return MoT(
        mixtures={"video": _TinyExpert(), "action": _TinyExpert()},
        mot_checkpoint_mixed_attn=False,
    ).eval()


def _freqs(sequence_length: int) -> torch.Tensor:
    return torch.ones(sequence_length, 1, 2, dtype=torch.complex128)


def _visual_memory(batch_size: int = 2) -> NativePatchMemory:
    tokens = torch.randn(batch_size, 2, 196, 384)
    camera_valid = torch.ones(batch_size, 2, dtype=torch.bool)
    return NativePatchMemory(
        tokens=tokens,
        patch_valid_mask=camera_valid.unsqueeze(-1).expand(-1, -1, 196),
        camera_valid_mask=camera_valid,
        camera_ids=("main", "wrist"),
        grid=(14, 14),
        source_revision="revision",
        weights_sha256="1" * 64,
        input_contract_sha256="2" * 64,
        preprocess_sha256="3" * 64,
        output_contract_sha256="4" * 64,
        memory_contract_sha256="5" * 64,
    )


class _RecordingReader(ActionVisualReader):
    reader_kind = "recording-reader-v1"
    reader_contract_sha256 = "6" * 64

    def __init__(
        self,
        *,
        layer_indices: tuple[int, ...],
        scale: float,
    ) -> None:
        super().__init__()
        self._layer_indices = layer_indices
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.contexts: list[ActionLayerReadContext] = []

    @property
    def injection_layer_indices(self) -> tuple[int, ...]:
        return self._layer_indices

    def forward_layer(
        self,
        context: ActionLayerReadContext,
        memory: NativePatchMemory,
        *,
        dino_keep_mask: torch.Tensor | None = None,
    ) -> VisualResidual:
        del memory
        self.contexts.append(context)
        pattern = torch.arange(
            1,
            context.post_block_hidden.shape[-1] + 1,
            dtype=context.post_block_hidden.dtype,
            device=context.post_block_hidden.device,
        ).view(1, 1, -1)
        tensor = pattern.expand_as(context.post_block_hidden) * self.scale
        if dino_keep_mask is not None:
            tensor = tensor * dino_keep_mask[:, None, None]
        return VisualResidual(
            tensor=tensor,
            layer_index=context.layer_index,
            branch_kinds=("recording-branch",),
        )


class _TinyCachedActionExpert(_TinyExpert):
    def __init__(self) -> None:
        super().__init__(num_layers=1)
        self.use_gradient_checkpointing = True

    def pre_dit(self, *, action_tokens, timestep, context, context_mask):
        return {
            "tokens": action_tokens,
            "freqs": _freqs(action_tokens.shape[1]),
            "t": timestep[:, None].expand(-1, 8),
            "t_mod": torch.zeros(action_tokens.shape[0], 6, 8),
            "context": context,
            "context_mask": context_mask[:, None, :].expand(
                -1, action_tokens.shape[1], -1
            ),
        }

    def post_dit(self, tokens, _pre):
        return tokens


def test_cached_uncond_checkpoint_recomputation_keeps_lora_regime() -> None:
    action = _TinyCachedActionExpert()
    adapter = inject_action_dit_lora(
        action,
        RegimeLoRAConfig(
            rank=2,
            alpha=2.0,
            target_groups=(ActionLoRATargetGroup.FFN,),
        ),
    )
    mot = MoT(
        mixtures={"video": _TinyExpert(num_layers=1), "action": action},
        mot_checkpoint_mixed_attn=True,
    ).train()
    condition = CachedActionCondition(
        context=torch.randn(2, 5, 8),
        context_mask=torch.ones(2, 5, dtype=torch.bool),
        video_kv_cache=[{"k": torch.randn(2, 4, 8), "v": torch.randn(2, 4, 8)}],
        attention_mask=torch.ones(7, 7, dtype=torch.bool),
        video_seq_len=4,
        current_frame_video_tokens=4,
    )
    velocity = CachedActionVelocity(
        action_expert=action,
        mot=mot,
        condition=condition,
        regime=PolicyRegime.UNCOND,
        regime_context=adapter.regime_context,
    )

    velocity(torch.randn(2, 3, 8), torch.tensor([0.2, 0.8])).velocity.sum().backward()

    assert adapter.regime_context.current is PolicyRegime.IDM
    assert any(parameter.grad is not None for parameter in adapter.lora_parameters())


def test_cached_visual_reader_stays_outside_base_checkpoint_recomputation() -> None:
    action = _TinyCachedActionExpert()
    adapter = inject_action_dit_lora(
        action,
        RegimeLoRAConfig(
            rank=2,
            alpha=2.0,
            target_groups=(ActionLoRATargetGroup.FFN,),
        ),
    )
    mot = MoT(
        mixtures={"video": _TinyExpert(num_layers=1), "action": action},
        mot_checkpoint_mixed_attn=True,
    ).train()
    visual = VisualReadCondition(
        memory=_visual_memory(),
        proprio=torch.randn(2, 3),
        video_layout_metadata={"grid": (1, 2, 2)},
    )
    condition = CachedActionCondition(
        context=torch.randn(2, 5, 8),
        context_mask=torch.ones(2, 5, dtype=torch.bool),
        video_kv_cache=[{"k": torch.randn(2, 4, 8), "v": torch.randn(2, 4, 8)}],
        attention_mask=torch.ones(7, 7, dtype=torch.bool),
        video_seq_len=4,
        current_frame_video_tokens=4,
        visual=visual,
    )
    reader = _RecordingReader(layer_indices=(0,), scale=0.1)
    velocity = CachedActionVelocity(
        action_expert=action,
        mot=mot,
        condition=condition,
        regime=PolicyRegime.UNCOND,
        regime_context=adapter.regime_context,
        visual_reader=reader,
    )

    velocity(torch.randn(2, 3, 8), torch.tensor([0.2, 0.8])).velocity.sum().backward()

    assert len(reader.contexts) == 1
    assert reader.scale.grad is not None
    assert torch.count_nonzero(reader.scale.grad) > 0
    assert adapter.regime_context.current is PolicyRegime.IDM


def test_idm_cached_velocity_rejects_visual_graph_construction() -> None:
    action = _TinyCachedActionExpert()
    condition = CachedActionCondition(
        context=torch.randn(2, 5, 8),
        context_mask=torch.ones(2, 5, dtype=torch.bool),
        video_kv_cache=[{"k": torch.randn(2, 4, 8), "v": torch.randn(2, 4, 8)}],
        attention_mask=torch.ones(7, 7, dtype=torch.bool),
        video_seq_len=4,
        current_frame_video_tokens=4,
        visual=VisualReadCondition(
            memory=_visual_memory(),
            proprio=torch.randn(2, 3),
        ),
    )

    with pytest.raises(ValueError, match="IDM action velocity must bypass"):
        CachedActionVelocity(
            action_expert=action,
            mot=MoT(
                mixtures={"video": _TinyExpert(num_layers=1), "action": action},
                mot_checkpoint_mixed_attn=False,
            ),
            condition=condition,
            regime=PolicyRegime.IDM,
            visual_reader=_RecordingReader(layer_indices=(0,), scale=0.0),
        )


def _joint_inputs() -> dict:
    generator = torch.Generator().manual_seed(8)
    video = torch.randn(2, 4, 8, generator=generator)
    action = torch.randn(2, 3, 8, generator=generator)
    video_context = torch.randn(2, 5, 8, generator=generator)
    action_context = torch.randn(2, 5, 8, generator=generator)
    action_context_mask = torch.tensor(
        [
            [
                [True, True, True, True, False],
                [True, True, True, True, False],
                [True, True, True, True, False],
            ],
            [
                [True, True, False, False, False],
                [True, True, True, False, False],
                [True, True, False, False, False],
            ],
        ]
    )
    attention_mask = torch.ones(7, 7, dtype=torch.bool)
    attention_mask[:4, 4:] = False
    attention_mask[:2, 2:4] = False
    return {
        "embeds_all": {"video": video, "action": action},
        "attention_mask": attention_mask,
        "freqs_all": {"video": _freqs(4), "action": _freqs(3)},
        "context_all": {
            "video": {
                "context": video_context,
                "mask": torch.ones(2, 4, 5, dtype=torch.bool),
            },
            "action": {
                "context": action_context,
                "mask": action_context_mask,
            },
        },
        "t_mod_all": {
            "video": torch.zeros(2, 6, 8),
            "action": torch.zeros(2, 6, 8),
        },
    }


def _cached_action_inputs() -> dict:
    generator = torch.Generator().manual_seed(37)
    video_cache = [
        {
            "k": torch.randn(2, 4, 8, generator=generator),
            "v": torch.randn(2, 4, 8, generator=generator),
            "_gate_current_frame_video_tokens": 2,
        }
        for _ in range(2)
    ]
    return {
        "action_tokens": torch.randn(2, 3, 8, generator=generator),
        "action_freqs": _freqs(3),
        "action_t_mod": torch.randn(2, 6, 8, generator=generator),
        "action_context_payload": {
            "context": torch.randn(2, 5, 8, generator=generator),
            "mask": torch.ones(2, 3, 5, dtype=torch.bool),
        },
        "video_kv_cache": video_cache,
        "attention_mask": torch.ones(7, 7, dtype=torch.bool),
        "video_seq_len": 4,
    }


def test_joint_forward_tap_is_read_only_and_typed() -> None:
    mot = _mot()
    inputs = _joint_inputs()
    baseline = mot(**inputs)
    tap = GateKVTapRequest(
        current_mode=(PolicyRegime.IDM, PolicyRegime.UNCOND),
        denoise_timestep=torch.tensor([0.8, 0.2]),
        current_frame_video_tokens=2,
        actor_version=4,
    )

    tapped = mot(**inputs, kv_tap=tap)
    snapshot = tap.snapshot()

    assert torch.equal(tapped["video"], baseline["video"])
    assert torch.equal(tapped["action"], baseline["action"])
    assert snapshot.layer_indices == (0, 1)
    for layer in snapshot.layers:
        assert layer.current_frame_video.source is KVSource.CURRENT_FRAME_VIDEO
        assert layer.current_frame_video.sequence_length == 2
        assert layer.action.source is KVSource.ACTION
        assert layer.action.sequence_length == 3
        assert layer.context.source is KVSource.TEXT_STATE_CONTEXT
        assert layer.context.sequence_length == 5
        assert not layer.current_frame_video.contains_generated_future_video
        assert not layer.current_frame_video.key.requires_grad
        assert not layer.action.key.requires_grad
        assert not layer.context.key.requires_grad
        assert layer.current_mode == (PolicyRegime.IDM, PolicyRegime.UNCOND)
        assert layer.actor_version == 4

    expected_context_mask = torch.tensor(
        [
            [True, True, True, True, False],
            [True, True, True, False, False],
        ]
    )
    assert torch.equal(snapshot.layers[0].context.valid_mask, expected_context_mask)


def test_joint_forward_captures_only_selected_layers() -> None:
    mot = _mot()
    tap = GateKVTapRequest(
        current_mode="uncond",
        denoise_timestep=0.25,
        current_frame_video_tokens=2,
        layer_indices=(1,),
    )

    mot(**_joint_inputs(), kv_tap=tap)

    assert tap.snapshot().layer_indices == (1,)


def test_cached_visual_hook_uses_exact_modulated_input_and_zero_delta() -> None:
    mot = _mot()
    inputs = _cached_action_inputs()
    baseline = mot.forward_action_with_video_cache(**inputs)
    reader = _RecordingReader(layer_indices=(0,), scale=0.0)
    visual_memory = _visual_memory()

    hooked = mot.forward_action_with_video_cache(
        **inputs,
        visual_reader=reader,
        visual_memory=visual_memory,
        visual_proprio=torch.randn(2, 3),
        action_time_embedding=torch.randn(2, 8),
        current_frame_video_tokens=2,
        video_layout_metadata={"grid": (1, 2, 2)},
    )

    assert torch.equal(hooked, baseline)
    assert len(reader.contexts) == 1
    block = mot.mixtures["action"].blocks[0]
    shift_msa, scale_msa, *_rest = mot._split_modulation(
        block,
        inputs["action_t_mod"],
    )
    expected = modulate(
        block.norm1(inputs["action_tokens"]),
        shift_msa,
        scale_msa,
    )
    assert torch.equal(reader.contexts[0].modulated_attn_input, expected)
    assert reader.contexts[0].video.layout_metadata == {"grid": (1, 2, 2)}


def test_cached_wan_dropout_masks_keys_and_zeroes_cache_per_sample(monkeypatch) -> None:
    mot = _mot()
    inputs = _cached_action_inputs()
    observed = []
    original = mot._mixed_attention

    def capture(q_cat, k_cat, v_cat, attention_mask):
        observed.append((k_cat.detach(), v_cat.detach(), attention_mask.detach()))
        return original(q_cat, k_cat, v_cat, attention_mask)

    monkeypatch.setattr(mot, "_mixed_attention", capture)
    mot.forward_action_with_video_cache(
        **inputs,
        current_frame_video_tokens=4,
        wan_keep_mask=torch.tensor([False, True]),
    )

    key, value, mask = observed[0]
    video_tokens = inputs["video_seq_len"]
    assert mask.shape == (2, 1, 3, 7)
    assert not mask[0, ..., :video_tokens].any()
    assert mask[0, ..., video_tokens:].all()
    assert mask[1].all()
    assert torch.count_nonzero(key[0, :video_tokens]) == 0
    assert torch.count_nonzero(value[0, :video_tokens]) == 0
    assert torch.equal(key[1, :video_tokens], inputs["video_kv_cache"][0]["k"][1])
    assert torch.equal(value[1, :video_tokens], inputs["video_kv_cache"][0]["v"][1])
    assert torch.count_nonzero(inputs["video_kv_cache"][0]["k"][0]) > 0


def test_cached_condition_reuses_one_modality_mask_across_velocity_calls() -> None:
    action = _TinyCachedActionExpert()
    adapter = inject_action_dit_lora(
        action,
        RegimeLoRAConfig(
            rank=2,
            alpha=2.0,
            target_groups=(ActionLoRATargetGroup.FFN,),
        ),
    )
    mot = MoT(
        mixtures={"video": _TinyExpert(num_layers=1), "action": action},
        mot_checkpoint_mixed_attn=False,
    ).eval()
    keep = ModalityKeepMask(
        wan=torch.tensor([False, True]),
        dino=torch.ones(2, dtype=torch.bool),
    )
    condition = CachedActionCondition(
        context=torch.randn(2, 5, 8),
        context_mask=torch.ones(2, 5, dtype=torch.bool),
        video_kv_cache=[{"k": torch.randn(2, 4, 8), "v": torch.randn(2, 4, 8)}],
        attention_mask=torch.ones(7, 7, dtype=torch.bool),
        video_seq_len=4,
        current_frame_video_tokens=4,
        modality_keep_mask=keep,
    )
    velocity = CachedActionVelocity(
        action_expert=action,
        mot=mot,
        condition=condition,
        regime=PolicyRegime.UNCOND,
        regime_context=adapter.regime_context,
    )

    first = velocity(torch.randn(2, 3, 8), torch.tensor([0.2, 0.8])).velocity
    second = velocity(torch.randn(2, 3, 8), torch.tensor([0.4, 0.6])).velocity

    assert first.shape == second.shape == (2, 3, 8)
    assert velocity.condition.modality_keep_mask is keep


def test_visual_residual_affects_only_later_gate_action_kv() -> None:
    mot = _mot()
    inputs = _cached_action_inputs()
    baseline_tap = GateKVTapRequest(
        current_mode="uncond",
        denoise_timestep=torch.tensor([0.6, 0.4]),
        current_frame_video_tokens=2,
        layer_indices=(0, 1),
    )
    visual_tap = GateKVTapRequest(
        current_mode="uncond",
        denoise_timestep=torch.tensor([0.6, 0.4]),
        current_frame_video_tokens=2,
        layer_indices=(0, 1),
    )
    baseline = mot.forward_action_with_video_cache(**inputs, kv_tap=baseline_tap)
    reader = _RecordingReader(layer_indices=(0,), scale=0.2)

    adapted = mot.forward_action_with_video_cache(
        **inputs,
        kv_tap=visual_tap,
        visual_reader=reader,
        visual_memory=_visual_memory(),
        visual_proprio=torch.randn(2, 3),
        action_time_embedding=torch.randn(2, 8),
        current_frame_video_tokens=2,
    )

    baseline_layers = baseline_tap.snapshot().layers
    visual_layers = visual_tap.snapshot().layers
    assert torch.equal(visual_layers[0].action.key, baseline_layers[0].action.key)
    assert not torch.equal(visual_layers[1].action.key, baseline_layers[1].action.key)
    assert not visual_layers[0].action.key.requires_grad
    assert not visual_layers[1].action.key.requires_grad
    assert not torch.equal(adapted, baseline)


def test_cached_action_forward_tap_excludes_generated_future_video() -> None:
    mot = _mot()
    generator = torch.Generator().manual_seed(17)
    action_tokens = torch.randn(2, 3, 8, generator=generator)
    action_context = torch.randn(2, 5, 8, generator=generator)
    video_cache = []
    for _ in range(2):
        key = torch.randn(2, 6, 8, generator=generator)
        value = torch.randn(2, 6, 8, generator=generator)
        key[:, 2:] = 10_000
        value[:, 2:] = -10_000
        video_cache.append(
            {
                "k": key,
                "v": value,
                "_gate_current_frame_video_tokens": 2,
            }
        )

    kwargs = {
        "action_tokens": action_tokens,
        "action_freqs": _freqs(3),
        "action_t_mod": torch.zeros(2, 6, 8),
        "action_context_payload": {
            "context": action_context,
            "mask": torch.ones(2, 3, 5, dtype=torch.bool),
        },
        "video_kv_cache": video_cache,
        "attention_mask": torch.ones(9, 9, dtype=torch.bool),
        "video_seq_len": 6,
    }
    kwargs["attention_mask"][:2, 2:6] = False
    baseline = mot.forward_action_with_video_cache(**kwargs)
    tap = GateKVTapRequest(
        current_mode="idm",
        denoise_timestep=torch.tensor([0.4, 0.4]),
        current_frame_video_tokens=2,
    )

    tapped = mot.forward_action_with_video_cache(**kwargs, kv_tap=tap)

    assert torch.equal(tapped, baseline)
    for layer_index, layer in enumerate(tap.snapshot().layers):
        assert torch.equal(
            layer.current_frame_video.key,
            video_cache[layer_index]["k"][:, :2],
        )
        assert torch.equal(
            layer.current_frame_video.value,
            video_cache[layer_index]["v"][:, :2],
        )
        assert layer.current_frame_video.sequence_length == 2


def test_tap_rejects_direct_future_video_leak_after_layer_zero() -> None:
    mot = _mot()
    inputs = _joint_inputs()
    inputs["attention_mask"][:2, 2:4] = True
    tap = GateKVTapRequest(
        current_mode="idm",
        denoise_timestep=0.1,
        current_frame_video_tokens=2,
        layer_indices=(1,),
    )

    with pytest.raises(ValueError, match="causal video mask"):
        mot(**inputs, kv_tap=tap)


def test_cached_tap_rejects_noncausal_or_unprovenanced_prefill(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda _message: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)
    mot = _mot()
    generator = torch.Generator().manual_seed(29)
    video_tokens = torch.randn(2, 6, 8, generator=generator)
    video_mask = torch.ones(6, 6, dtype=torch.bool)

    with pytest.raises(ValueError, match="causal video mask"):
        mot.prefill_video_cache(
            video_tokens=video_tokens,
            video_freqs=_freqs(6),
            video_t_mod=torch.zeros(2, 6, 8),
            video_context_payload=None,
            video_attention_mask=video_mask,
            gate_current_frame_video_tokens=2,
        )

    unprovenanced_cache = mot.prefill_video_cache(
        video_tokens=video_tokens,
        video_freqs=_freqs(6),
        video_t_mod=torch.zeros(2, 6, 8),
        video_context_payload=None,
        video_attention_mask=video_mask,
    )
    causal_joint_mask = torch.ones(9, 9, dtype=torch.bool)
    causal_joint_mask[:2, 2:6] = False
    tap = GateKVTapRequest(
        current_mode="idm",
        denoise_timestep=0.1,
        current_frame_video_tokens=2,
        layer_indices=(1,),
    )

    with pytest.raises(ValueError, match="cache provenance"):
        mot.forward_action_with_video_cache(
            action_tokens=torch.randn(2, 3, 8, generator=generator),
            action_freqs=_freqs(3),
            action_t_mod=torch.zeros(2, 6, 8),
            action_context_payload={
                "context": torch.randn(2, 5, 8, generator=generator),
                "mask": torch.ones(2, 3, 5, dtype=torch.bool),
            },
            video_kv_cache=unprovenanced_cache,
            attention_mask=causal_joint_mask,
            video_seq_len=6,
            kv_tap=tap,
        )


def test_layer_zero_tap_is_safe_before_video_tokens_mix() -> None:
    mot = _mot()
    inputs = _joint_inputs()
    inputs["attention_mask"][:2, 2:4] = True
    tap = GateKVTapRequest(
        current_mode="idm",
        denoise_timestep=0.1,
        current_frame_video_tokens=2,
        layer_indices=(0,),
    )

    mot(**inputs, kv_tap=tap)
    assert tap.snapshot().layer_indices == (0,)


def test_causal_mask_keeps_later_current_frame_kv_independent_of_future() -> None:
    mot = _mot()
    baseline_inputs = _joint_inputs()
    perturbed_inputs = dict(baseline_inputs)
    perturbed_video = baseline_inputs["embeds_all"]["video"].clone()
    perturbed_video[:, 2:] += 10_000
    perturbed_inputs["embeds_all"] = {
        **baseline_inputs["embeds_all"],
        "video": perturbed_video,
    }
    baseline_tap = GateKVTapRequest(
        current_mode="idm",
        denoise_timestep=0.1,
        current_frame_video_tokens=2,
        layer_indices=(1,),
    )
    perturbed_tap = GateKVTapRequest(
        current_mode="idm",
        denoise_timestep=0.1,
        current_frame_video_tokens=2,
        layer_indices=(1,),
    )

    mot(**baseline_inputs, kv_tap=baseline_tap)
    mot(**perturbed_inputs, kv_tap=perturbed_tap)

    baseline_bank = baseline_tap.snapshot().layers[0].current_frame_video
    perturbed_bank = perturbed_tap.snapshot().layers[0].current_frame_video
    assert torch.equal(baseline_bank.key, perturbed_bank.key)
    assert torch.equal(baseline_bank.value, perturbed_bank.value)


def test_tap_rejects_invalid_video_or_layer_extent_before_forward() -> None:
    mot = _mot()
    invalid_video = GateKVTapRequest(
        current_mode="idm",
        denoise_timestep=0.1,
        current_frame_video_tokens=5,
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        mot(**_joint_inputs(), kv_tap=invalid_video)

    invalid_layer = GateKVTapRequest(
        current_mode="idm",
        denoise_timestep=0.1,
        current_frame_video_tokens=2,
        layer_indices=(2,),
    )
    with pytest.raises(ValueError, match="outside"):
        mot(**_joint_inputs(), kv_tap=invalid_layer)
