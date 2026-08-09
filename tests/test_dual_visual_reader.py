from __future__ import annotations

import hashlib
from types import MethodType

import pytest
import torch
import torch.nn.functional as F

from fastwam.models.wan22.dual_visual_reader import (
    DINO_RETRIEVAL_BRANCH_KIND,
    DUAL_VISUAL_READER_PARAMETER_FAMILY,
    WAN_RETRIEVAL_BRANCH_KIND,
    DinoWanTransportGeometry,
    DualDinoWanReaderConfig,
    build_dual_dino_wan_reader,
)
from fastwam.models.wan22.visual_contracts import (
    DINO_V3_NATIVE_DIM,
    DINO_V3_PATCH_COUNT,
    ActionLayerReadContext,
    LayerVideoKVView,
    NativePatchMemory,
)
from fastwam.models.wan22.visual_sidecar import ProjectionSpec


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _geometry() -> DinoWanTransportGeometry:
    transport = torch.zeros(2, DINO_V3_PATCH_COUNT, 4)
    patch_ids = torch.arange(DINO_V3_PATCH_COUNT)
    transport[:, patch_ids, patch_ids % 4] = 1
    return DinoWanTransportGeometry.from_tensors(
        transport=transport,
        target_valid_mask=torch.ones(2, 4, dtype=torch.bool),
        camera_prior=torch.tensor([0.75, 0.25]),
        camera_ids=("main", "wrist"),
        wan_grid=(1, 2, 2),
        asset_sha256=_hash("transport-asset"),
    )


def _memory(*, second_view_valid: bool = True) -> NativePatchMemory:
    generator = torch.Generator().manual_seed(31)
    tokens = torch.randn(
        2,
        2,
        DINO_V3_PATCH_COUNT,
        DINO_V3_NATIVE_DIM,
        generator=generator,
    )
    camera_valid = torch.ones(2, 2, dtype=torch.bool)
    if not second_view_valid:
        camera_valid[0, 1] = False
        tokens[0, 1].zero_()
    return NativePatchMemory(
        tokens=tokens,
        patch_valid_mask=camera_valid[:, :, None].expand(-1, -1, 196),
        camera_valid_mask=camera_valid,
        camera_ids=("main", "wrist"),
        grid=(14, 14),
        source_revision="test-revision",
        weights_sha256=_hash("weights"),
        input_contract_sha256=_hash("input"),
        preprocess_sha256=_hash("preprocess"),
        output_contract_sha256=_hash("output"),
        memory_contract_sha256=_hash("memory"),
    )


def _context(*, zero_wan_values: bool = False) -> ActionLayerReadContext:
    generator = torch.Generator().manual_seed(37)
    hidden = torch.randn(2, 3, 8, generator=generator)
    values = torch.randn(2, 4, 8, generator=generator)
    if zero_wan_values:
        values.zero_()
    return ActionLayerReadContext(
        layer_index=0,
        pre_block_hidden=hidden,
        modulated_attn_input=torch.randn(2, 3, 8, generator=generator),
        post_block_hidden=torch.randn(2, 3, 8, generator=generator),
        base_gate_msa=torch.ones(2, 1, 8),
        timestep_embedding=torch.randn(2, 8, generator=generator),
        proprio=torch.randn(2, 3, generator=generator),
        video=LayerVideoKVView(
            key=torch.randn(2, 4, 8, generator=generator),
            value=values,
            current_frame_tokens=4,
            layout_metadata={
                "wan_grid": (1, 2, 2),
                "camera_ids": ("main", "wrist"),
            },
        ),
        base_action_output_weight=torch.eye(8),
    )


def _config(
    memory: NativePatchMemory, geometry: DinoWanTransportGeometry
) -> DualDinoWanReaderConfig:
    return DualDinoWanReaderConfig(
        action_hidden_dim=8,
        camera_ids=("main", "wrist"),
        layer_indices=(0,),
        temperature=0.1,
        query_projection=ProjectionSpec(kind="low_rank", rank=4),
        dino_output_projection=ProjectionSpec(kind="low_rank", rank=4),
        gamma_dino_max=0.5,
        gamma_wan_max=0.25,
        memory_contract_sha256=memory.memory_contract_sha256,
        transport_sha256=geometry.transport_sha256,
        geometry_contract_sha256=geometry.geometry_contract_sha256,
    )


def test_dual_reader_is_exact_zero_and_unlocks_only_through_gammas() -> None:
    memory = _memory(second_view_valid=False)
    geometry = _geometry()
    reader = build_dual_dino_wan_reader(_config(memory, geometry), geometry)

    initial = reader.forward_layer(_context(), memory)
    assert initial.branch_kinds == (
        DINO_RETRIEVAL_BRANCH_KIND,
        WAN_RETRIEVAL_BRANCH_KIND,
    )
    assert torch.count_nonzero(initial.tensor) == 0
    initial.tensor.sum().backward()

    dino_branch, wan_branch = reader.branches["0"]
    assert dino_branch.gamma_raw.grad is not None
    assert torch.count_nonzero(dino_branch.gamma_raw.grad) > 0
    assert wan_branch.gamma_raw.grad is not None
    assert torch.count_nonzero(wan_branch.gamma_raw.grad) > 0
    nongamma = [
        parameter.grad
        for name, parameter in reader.named_parameters()
        if not name.endswith("gamma_raw")
    ]
    assert all(gradient is not None for gradient in nongamma)
    assert all(torch.count_nonzero(gradient) == 0 for gradient in nongamma)

    reader.zero_grad(set_to_none=True)
    with torch.no_grad():
        dino_branch.gamma_raw.fill_(0.2)
        wan_branch.gamma_raw.fill_(-0.15)
    unlocked = reader.forward_layer(_context(), memory)
    unlocked.tensor.square().sum().backward()
    router_grad = reader.routers["0"].query_projection.down.weight.grad
    output_grad = dino_branch.output_projection.down.weight.grad
    assert router_grad is not None and torch.count_nonzero(router_grad) > 0
    assert output_grad is not None and torch.count_nonzero(output_grad) > 0


def test_p7_branches_receive_the_identical_routing_object() -> None:
    memory = _memory()
    geometry = _geometry()
    reader = build_dual_dino_wan_reader(_config(memory, geometry), geometry)
    routing_ids = {}
    for branch in reader.branches["0"]:
        original = branch.forward_branch

        def recording_forward(
            self,
            context,
            native_memory,
            routing,
            *,
            _original=original,
        ):
            routing_ids[self.branch_kind] = id(routing)
            return _original(context, native_memory, routing)

        object.__setattr__(
            branch,
            "forward_branch",
            MethodType(recording_forward, branch),
        )

    reader.forward_layer(_context(), memory)

    assert set(routing_ids) == {
        DINO_RETRIEVAL_BRANCH_KIND,
        WAN_RETRIEVAL_BRANCH_KIND,
    }
    assert len(set(routing_ids.values())) == 1


def test_wan_branch_uses_current_prefix_and_zero_values_give_exact_zero() -> None:
    memory = _memory()
    geometry = _geometry()
    reader = build_dual_dino_wan_reader(_config(memory, geometry), geometry)
    dino_branch, wan_branch = reader.branches["0"]
    with torch.no_grad():
        dino_branch.gamma_raw.zero_()
        wan_branch.gamma_raw.fill_(1.0)

    residual = reader.forward_layer(_context(zero_wan_values=True), memory)

    assert torch.count_nonzero(residual.tensor) == 0


def test_wan_branch_is_exact_frozen_action_o_weight_only_and_ignores_future() -> None:
    memory = _memory()
    geometry = _geometry()
    reader = build_dual_dino_wan_reader(_config(memory, geometry), geometry)
    dino_branch, wan_branch = reader.branches["0"]
    with torch.no_grad():
        dino_branch.gamma_raw.zero_()
        wan_branch.gamma_raw.fill_(0.4)
    context = _context()
    future_context = ActionLayerReadContext(
        **{
            **context.__dict__,
            "video": LayerVideoKVView(
                key=torch.cat((context.video.key, torch.randn(2, 2, 8)), dim=1),
                value=torch.cat(
                    (context.video.value, torch.full((2, 2, 8), 1_000.0)), dim=1
                ),
                current_frame_tokens=4,
                layout_metadata=context.video.layout_metadata,
            ),
        }
    )
    routing = reader.routers["0"](context, memory)
    effective, _ = geometry.effective_transport(
        batch_size=2,
        camera_valid_mask=memory.camera_valid_mask,
        target_valid_mask=None,
        device=routing.weights.device,
        dtype=routing.weights.dtype,
    )
    routed_to_wan = torch.einsum("bvan,bvnk->bvak", routing.weights, effective)
    per_view = torch.einsum(
        "bvak,bkd->bvad",
        routed_to_wan,
        context.video.value[:, :4],
    )
    prior = geometry.camera_prior[None] * memory.camera_valid_mask
    prior = prior / prior.sum(dim=-1, keepdim=True)
    retrieved = torch.einsum("bv,bvad->bad", prior, per_view)
    expected = (
        wan_branch.gamma_max
        * torch.tanh(wan_branch.gamma_raw)
        * context.base_gate_msa
        * F.linear(retrieved, context.base_action_output_weight, bias=None)
    )

    actual = reader.forward_layer(future_context, memory).tensor

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_transport_masks_and_camera_activity_fail_closed() -> None:
    memory = _memory()
    geometry = _geometry()
    reader = build_dual_dino_wan_reader(_config(memory, geometry), geometry)
    context = _context()
    context = ActionLayerReadContext(
        **{
            **context.__dict__,
            "video": LayerVideoKVView(
                **{
                    **context.video.__dict__,
                    "layout_metadata": {
                        "wan_grid": (1, 2, 2),
                        "camera_ids": ("main", "wrist"),
                        "target_valid_mask": torch.zeros(2, 2, 4, dtype=torch.bool),
                    },
                }
            ),
        }
    )
    with pytest.raises(ValueError, match="no positive support"):
        reader.forward_layer(context, memory)

    with pytest.raises(ValueError, match="at least one active"):
        geometry.effective_transport(
            batch_size=2,
            camera_valid_mask=torch.tensor([[False, False], [True, False]]),
            target_valid_mask=None,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_geometry_and_reader_state_are_hash_bound_and_reader_only() -> None:
    memory = _memory()
    geometry = _geometry()
    config = _config(memory, geometry)
    reader = build_dual_dino_wan_reader(config, geometry)
    manifest = reader.trainable_parameter_manifest()
    state = reader.export_trainable_state()

    assert tuple(manifest) == (DUAL_VISUAL_READER_PARAMETER_FAMILY,)
    assert all("transport" not in name for name in state["state"])
    assert all("base_action_output" not in name for name in state["state"])

    bad_transport = geometry.transport.clone()
    bad_transport[0, 0].zero_()
    with pytest.raises(ValueError, match="row must sum to one"):
        DinoWanTransportGeometry(**{**geometry.__dict__, "transport": bad_transport})
    with pytest.raises(ValueError, match="detached and permanently frozen"):
        DinoWanTransportGeometry(
            **{
                **geometry.__dict__,
                "transport": geometry.transport.clone().requires_grad_(),
            }
        )
    with pytest.raises(ValueError, match="transport SHA256 mismatch"):
        build_dual_dino_wan_reader(
            DualDinoWanReaderConfig(
                **{**config.__dict__, "transport_sha256": _hash("wrong")}
            ),
            geometry,
        )


def test_wan_branch_rejects_trainable_base_output_weight() -> None:
    memory = _memory()
    geometry = _geometry()
    reader = build_dual_dino_wan_reader(_config(memory, geometry), geometry)
    context = _context()
    context = ActionLayerReadContext(
        **{
            **context.__dict__,
            "base_action_output_weight": torch.nn.Parameter(torch.eye(8)),
        }
    )

    with pytest.raises(ValueError, match="must remain frozen"):
        reader.forward_layer(context, memory)


def test_reader_behavior_reference_is_actor_version_bound_and_restored() -> None:
    memory = _memory()
    geometry = _geometry()
    reader = build_dual_dino_wan_reader(_config(memory, geometry), geometry)
    reader.capture_replay_reference(actor_version=11)
    reference = {
        name: parameter.detach().clone()
        for name, parameter in reader.named_parameters()
    }
    with torch.no_grad():
        for parameter in reader.parameters():
            parameter.add_(1)
    live = {
        name: parameter.detach().clone()
        for name, parameter in reader.named_parameters()
    }

    with reader.use_replay_reference(actor_version=11):
        for name, parameter in reader.named_parameters():
            torch.testing.assert_close(parameter, reference[name])

    for name, parameter in reader.named_parameters():
        torch.testing.assert_close(parameter, live[name])
    with pytest.raises(ValueError, match="actor version mismatch"):
        with reader.use_replay_reference(actor_version=12):
            pass
