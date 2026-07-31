from __future__ import annotations

import importlib.util
import sys
import types

import pytest
import torch
import torch.nn as nn

if importlib.util.find_spec("safetensors") is None:
    safetensors_stub = types.ModuleType("safetensors")
    safetensors_stub.safe_open = None
    sys.modules["safetensors"] = safetensors_stub
if importlib.util.find_spec("imageio") is None:
    imageio_stub = types.ModuleType("imageio")
    imageio_stub.get_writer = None
    sys.modules["imageio"] = imageio_stub

from fastwam.adapters import PolicyRegime
from fastwam.models.wan22.kv_tap import GateKVTapRequest, KVSource
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import DiTBlock


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
    return {
        "embeds_all": {"video": video, "action": action},
        "attention_mask": torch.ones(7, 7, dtype=torch.bool),
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
        video_cache.append({"k": key, "v": value})

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
