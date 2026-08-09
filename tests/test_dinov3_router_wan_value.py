from __future__ import annotations

import hashlib

import pytest
import torch

from fastwam.models.wan22.dinov3_memory import (
    DINO_V3_OUTPUT_CONTRACT_SHA256,
    DINO_V3_PREPROCESS_SHA256,
    PINNED_DINOV3_SOURCE_REVISION,
)
from fastwam.models.wan22.visual_contracts import (
    WAN_FLATTEN_ORDER,
    WAN_VIDEO_VALUE_LAYOUT,
    ActionLayerReadContext,
    LayerVideoKVView,
    NativePatchMemory,
    WanValueSpatialMetadata,
    build_area_overlap_dino_wan_transport,
)
from fastwam.models.wan22.visual_sidecar import (
    DinoWanValueReaderConfig,
    ProjectionSpec,
    build_dino_wan_value_reader,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _metadata() -> WanValueSpatialMetadata:
    return WanValueSpatialMetadata(
        wan_grid_f=1,
        wan_grid_h=2,
        wan_grid_w=4,
        current_frame_video_tokens=8,
        wan_flatten_order=WAN_FLATTEN_ORDER,
        vae_model_type="WanVideoVAE38",
        vae_weights_sha256=_hash("vae"),
        vae_spatial_downsample_factor=16,
        video_dit_weights_sha256=_hash("video-dit"),
        video_dit_patch_size=(1, 2, 2),
        video_attention_num_heads=2,
        video_attention_head_dim=4,
        video_value_layout=WAN_VIDEO_VALUE_LAYOUT,
        video_value_rope_applied=False,
        camera_concat_mode="horizontal",
        camera_order=("main", "wrist"),
        per_camera_post_crop_hw=((224, 224), (224, 224)),
        per_camera_combined_rgb_box=((0, 0, 224, 224), (0, 224, 224, 448)),
        per_camera_wan_grid_support=((0, 2, 0, 2), (0, 2, 2, 4)),
        dino_patch_grid=(14, 14),
        dino_preprocess_sha256=DINO_V3_PREPROCESS_SHA256,
        invalid_mask_policy="renormalize_active_or_fail_closed",
    )


def _memory(*, wrist_valid: bool = True) -> NativePatchMemory:
    generator = torch.Generator().manual_seed(7)
    tokens = torch.randn(1, 2, 196, 384, generator=generator)
    camera_valid = torch.tensor([[True, wrist_valid]])
    if not wrist_valid:
        tokens[:, 1].zero_()
    patch_valid = camera_valid.unsqueeze(-1).expand(-1, -1, 196)
    return NativePatchMemory(
        tokens=tokens,
        patch_valid_mask=patch_valid,
        camera_valid_mask=camera_valid,
        camera_ids=("main", "wrist"),
        grid=(14, 14),
        source_revision=PINNED_DINOV3_SOURCE_REVISION,
        weights_sha256=_hash("dino"),
        input_contract_sha256=_hash("input"),
        preprocess_sha256=DINO_V3_PREPROCESS_SHA256,
        output_contract_sha256=DINO_V3_OUTPUT_CONTRACT_SHA256,
        memory_contract_sha256=_hash("memory"),
    )


def _reader(memory: NativePatchMemory):
    metadata = _metadata()
    transport = build_area_overlap_dino_wan_transport(metadata)
    return build_dino_wan_value_reader(
        DinoWanValueReaderConfig(
            action_hidden_dim=6,
            camera_ids=memory.camera_ids,
            layer_indices=(0,),
            temperature=0.2,
            beta_max=0.5,
            query_projection=ProjectionSpec(kind="low_rank", rank=3),
            memory_contract_sha256=memory.memory_contract_sha256,
            spatial_metadata=metadata,
            transport=transport,
            camera_mass=(1.0, 1.0),
        )
    )


def _context(
    *,
    value_requires_grad: bool = False,
    output_weight_requires_grad: bool = False,
    gate_requires_grad: bool = False,
) -> ActionLayerReadContext:
    generator = torch.Generator().manual_seed(13)
    value = torch.randn(1, 10, 8, generator=generator)
    value.requires_grad_(value_requires_grad)
    output_weight = torch.randn(6, 8, generator=generator)
    output_weight.requires_grad_(output_weight_requires_grad)
    gate = torch.ones(1, 1, 6)
    gate.requires_grad_(gate_requires_grad)
    return ActionLayerReadContext(
        layer_index=0,
        pre_block_hidden=torch.randn(1, 3, 6, generator=generator),
        modulated_attn_input=torch.randn(1, 3, 6, generator=generator),
        post_block_hidden=torch.randn(1, 3, 6, generator=generator),
        base_gate_msa=gate,
        timestep_embedding=torch.randn(1, 6, generator=generator),
        proprio=torch.randn(1, 4, generator=generator),
        video=LayerVideoKVView(
            key=torch.randn(1, 10, 8, generator=generator),
            value=value,
            current_frame_tokens=8,
            layout_metadata=_metadata(),
        ),
        base_action_output_weight=output_weight,
    )


def test_area_overlap_transport_preserves_mass_and_camera_placement() -> None:
    metadata = _metadata()
    first = build_area_overlap_dino_wan_transport(metadata)
    second = build_area_overlap_dino_wan_transport(metadata)

    assert first.transport_sha256 == second.transport_sha256
    torch.testing.assert_close(first.matrix.sum(dim=-1), torch.ones(2, 196))
    assert torch.count_nonzero(first.matrix[0, :, [2, 3, 6, 7]]) == 0
    assert torch.count_nonzero(first.matrix[1, :, [0, 1, 4, 5]]) == 0
    assert first.matrix[0, 0, 0] > 0
    assert first.matrix[1, 0, 2] > 0


@pytest.mark.parametrize(
    ("concat", "grid", "boxes", "supports", "expected"),
    [
        (
            "vertical",
            (4, 2),
            ((0, 0, 224, 224), (224, 0, 448, 224)),
            ((0, 2, 0, 2), (2, 4, 0, 2)),
            ((0, 0), (1, 4)),
        ),
        (
            "main_only",
            (2, 2),
            ((0, 0, 224, 224),),
            ((0, 2, 0, 2),),
            ((0, 0),),
        ),
    ],
)
def test_area_overlap_supports_vertical_and_main_only_layouts(
    concat,
    grid,
    boxes,
    supports,
    expected,
) -> None:
    camera_order = ("main", "wrist") if concat == "vertical" else ("main",)
    metadata = WanValueSpatialMetadata(
        **{
            **_metadata().__dict__,
            "wan_grid_h": grid[0],
            "wan_grid_w": grid[1],
            "current_frame_video_tokens": grid[0] * grid[1],
            "camera_concat_mode": concat,
            "camera_order": camera_order,
            "per_camera_post_crop_hw": tuple((224, 224) for _ in camera_order),
            "per_camera_combined_rgb_box": boxes,
            "per_camera_wan_grid_support": supports,
            "spatial_transport_contract_sha256": None,
        }
    )
    transport = build_area_overlap_dino_wan_transport(metadata)
    torch.testing.assert_close(
        transport.matrix.sum(dim=-1),
        torch.ones(len(camera_order), 196),
    )
    for camera_index, target_index in expected:
        assert transport.matrix[camera_index, 0, target_index] > 0


def test_area_overlap_handles_odd_borders_and_rejects_gaps() -> None:
    odd = WanValueSpatialMetadata(
        **{
            **_metadata().__dict__,
            "wan_grid_h": 3,
            "wan_grid_w": 3,
            "current_frame_video_tokens": 9,
            "camera_concat_mode": "main_only",
            "camera_order": ("main",),
            "per_camera_post_crop_hw": ((225, 227),),
            "per_camera_combined_rgb_box": ((0, 0, 225, 227),),
            "per_camera_wan_grid_support": ((0, 3, 0, 3),),
            "spatial_transport_contract_sha256": None,
        }
    )
    transport = build_area_overlap_dino_wan_transport(odd)
    torch.testing.assert_close(transport.matrix.sum(dim=-1), torch.ones(1, 196))
    assert torch.count_nonzero(transport.matrix[0, -1]) > 0

    gap = WanValueSpatialMetadata(
        **{
            **odd.__dict__,
            "per_camera_wan_grid_support": ((0, 3, 0, 1),),
            "spatial_transport_contract_sha256": None,
        }
    )
    with pytest.raises(ValueError, match="no positive Wan target support"):
        build_area_overlap_dino_wan_transport(gap)
    with pytest.raises(ValueError, match="Unsupported camera concatenation"):
        WanValueSpatialMetadata(
            **{
                **odd.__dict__,
                "camera_concat_mode": "interleaved",
                "spatial_transport_contract_sha256": None,
            }
        )


def test_spatial_contract_fails_closed_on_hash_or_camera_mismatch() -> None:
    metadata = _metadata()
    with pytest.raises(ValueError, match="contract SHA256 mismatch"):
        WanValueSpatialMetadata(
            **{
                **metadata.__dict__,
                "spatial_transport_contract_sha256": _hash("wrong"),
            }
        )
    transport = build_area_overlap_dino_wan_transport(metadata)
    memory = _memory()
    with pytest.raises(ValueError, match="camera order"):
        DinoWanValueReaderConfig(
            action_hidden_dim=6,
            camera_ids=("wrist", "main"),
            layer_indices=(0,),
            temperature=0.2,
            beta_max=0.5,
            query_projection=ProjectionSpec(kind="full_linear", rank=None),
            memory_contract_sha256=memory.memory_contract_sha256,
            spatial_metadata=metadata,
            transport=transport,
            camera_mass=(1.0, 1.0),
        )


def test_p6_zero_delta_then_unlocks_router_on_second_step() -> None:
    memory = _memory()
    context = _context()
    reader = _reader(memory)
    router = reader.routers["0"]
    branch = reader.branches["0"][0]

    initial = reader.forward_layer(context, memory)
    assert torch.count_nonzero(initial.tensor) == 0
    initial.tensor.sum().backward()
    assert branch.raw_beta.grad is not None
    assert torch.count_nonzero(branch.raw_beta.grad) == 1
    assert router.query_projection.down.weight.grad is not None
    assert torch.count_nonzero(router.query_projection.down.weight.grad) == 0

    reader.zero_grad(set_to_none=True)
    with torch.no_grad():
        branch.raw_beta.fill_(0.2)
    reader.forward_layer(context, memory).tensor.sum().backward()
    assert torch.count_nonzero(router.query_projection.down.weight.grad) > 0


def test_p6_uses_current_prefix_frozen_weight_only_and_detached_wan_values() -> None:
    memory = _memory(wrist_valid=False)
    context = _context(value_requires_grad=True, gate_requires_grad=True)
    reader = _reader(memory)
    branch = reader.branches["0"][0]
    with torch.no_grad():
        branch.raw_beta.fill_(0.25)

    routing = reader.routers["0"](context, memory)
    wan_weights = branch._transport_routing(memory, routing)
    current = context.video.value[:, :8].detach()
    retrieved = torch.einsum("baw,bwd->bad", wan_weights, current)
    expected = (
        branch.beta
        * context.base_gate_msa
        * torch.nn.functional.linear(
            retrieved,
            context.base_action_output_weight,
            bias=None,
        )
    )
    actual = reader.forward_layer(context, memory).tensor

    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    assert context.video.value.grad is None
    assert context.base_gate_msa.grad is None
    assert branch.raw_beta.grad is not None


def test_p6_rejects_trainable_base_output_weight() -> None:
    memory = _memory()
    reader = _reader(memory)
    with pytest.raises(ValueError, match="output weight must be frozen"):
        reader.forward_layer(
            _context(output_weight_requires_grad=True),
            memory,
        )


def test_p6_active_camera_mass_renormalizes_and_all_invalid_fails() -> None:
    memory = _memory(wrist_valid=False)
    reader = _reader(memory)
    context = _context()
    branch = reader.branches["0"][0]
    routing = reader.routers["0"](context, memory)

    wan_weights = branch._transport_routing(memory, routing)

    assert torch.count_nonzero(wan_weights[..., [2, 3, 6, 7]]) == 0
    torch.testing.assert_close(wan_weights.sum(dim=-1), torch.ones(1, 3))
    with pytest.raises(ValueError, match="at least one active camera"):
        branch.transport.effective_sha256(
            camera_valid_mask=torch.zeros(1, 2, dtype=torch.bool),
            patch_valid_mask=torch.zeros(1, 2, 196, dtype=torch.bool),
        )


def test_visual_reader_behavior_snapshot_is_version_strict() -> None:
    reader = _reader(_memory())
    assert set(reader.trainable_parameter_manifest()) == {"visual_router"}
    before = reader.export_trainable_state()
    reader.capture_replay_reference(actor_version=4)
    with torch.no_grad():
        for parameter in reader.parameters():
            if parameter.requires_grad:
                parameter.add_(1)
    changed = reader.export_trainable_state()

    with reader.use_replay_reference(actor_version=4):
        restored = reader.export_trainable_state()
        for name, tensor in before["state"].items():
            assert torch.equal(restored["state"][name], tensor)
    after = reader.export_trainable_state()
    for name, tensor in changed["state"].items():
        assert torch.equal(after["state"][name], tensor)
    with pytest.raises(ValueError, match="actor version mismatch"):
        with reader.use_replay_reference(actor_version=5):
            pass
