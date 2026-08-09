from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import torch
from torch import nn

from fastwam.adapters import PolicyRegime
from fastwam.models.wan22.dinov3_memory import (
    DINO_V3_OUTPUT_CONTRACT_SHA256,
    DINO_V3_PREPROCESS_SHA256,
    PINNED_DINOV3_MODEL_NAME,
    PINNED_DINOV3_SOURCE_REVISION,
    DinoV3AssetSpec,
    FrozenDinoV3Encoder,
    native_memory_contract_sha256,
)
from fastwam.models.wan22.visual_contracts import (
    DINO_V3_NATIVE_DIM,
    DINO_V3_PATCH_COUNT,
    ActionLayerReadContext,
    LayerVideoKVView,
    NativePatchMemory,
    PreparedCameraBatch,
    RoutingWeights,
    contract_sha256,
    native_patch_layout_contract,
)
from fastwam.models.wan22.visual_sidecar import (
    DINO_SEMANTIC_READER_KIND,
    DinoSemanticReaderConfig,
    NativeDinoRouter,
    ProjectionSpec,
    RoutedVisualReader,
    VisualValueBranch,
    build_dino_semantic_reader,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _asset(*, compute_dtype: str = "float32") -> DinoV3AssetSpec:
    return DinoV3AssetSpec(
        source_root="/does/not/need/to/exist/in-unit-tests",
        source_revision=PINNED_DINOV3_SOURCE_REVISION,
        model_name=PINNED_DINOV3_MODEL_NAME,
        weights_path="/does/not/need/to/exist/in-unit-tests/model.pth",
        weights_sha256=_hash("weights"),
        preprocess_sha256=DINO_V3_PREPROCESS_SHA256,
        output_contract_sha256=DINO_V3_OUTPUT_CONTRACT_SHA256,
        compute_dtype=compute_dtype,
        license_id="DINOv3 License",
    )


class _FakeDino(nn.Module):
    def __init__(self, *, output_key: str = "x_norm_patchtokens") -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.output_key = output_key
        self.calls = 0

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        self.calls += 1
        basis = torch.linspace(
            0.25,
            1.25,
            DINO_V3_NATIVE_DIM,
            dtype=images.dtype,
            device=images.device,
        )
        image_signal = images.mean(dim=(1, 2, 3), keepdim=False)[:, None, None]
        tokens = basis[None, None, :].expand(images.shape[0], DINO_V3_PATCH_COUNT, -1)
        return {self.output_key: tokens + image_signal}


def _camera_batch() -> PreparedCameraBatch:
    pixels = torch.full((2, 2, 3, 224, 224), 127, dtype=torch.uint8)
    pixels[1, 1].fill_(191)
    return PreparedCameraBatch(
        pixels=pixels,
        camera_ids=("main", "wrist"),
        camera_valid_mask=torch.tensor([[True, False], [True, True]]),
        input_contract_sha256=_hash("libero-224-geometry"),
    )


def _memory(*, batch: int = 2, second_view_valid: bool = True) -> NativePatchMemory:
    generator = torch.Generator().manual_seed(11)
    tokens = torch.randn(
        batch,
        2,
        DINO_V3_PATCH_COUNT,
        DINO_V3_NATIVE_DIM,
        generator=generator,
    )
    camera_valid = torch.ones(batch, 2, dtype=torch.bool)
    if not second_view_valid:
        camera_valid[0, 1] = False
        tokens[0, 1].zero_()
    patch_valid = camera_valid.unsqueeze(-1).expand(-1, -1, DINO_V3_PATCH_COUNT)
    return NativePatchMemory(
        tokens=tokens,
        patch_valid_mask=patch_valid,
        camera_valid_mask=camera_valid,
        camera_ids=("main", "wrist"),
        grid=(14, 14),
        source_revision=PINNED_DINOV3_SOURCE_REVISION,
        weights_sha256=_hash("weights"),
        input_contract_sha256=_hash("input"),
        preprocess_sha256=DINO_V3_PREPROCESS_SHA256,
        output_contract_sha256=DINO_V3_OUTPUT_CONTRACT_SHA256,
        memory_contract_sha256=_hash("memory-contract"),
    )


def _context(*, batch: int = 2, action_dim: int = 8) -> ActionLayerReadContext:
    generator = torch.Generator().manual_seed(19)
    pre = torch.randn(batch, 3, action_dim, generator=generator)
    modulated = torch.randn(batch, 3, action_dim, generator=generator)
    post = torch.randn(batch, 3, action_dim, generator=generator)
    return ActionLayerReadContext(
        layer_index=0,
        pre_block_hidden=pre,
        modulated_attn_input=modulated,
        post_block_hidden=post,
        base_gate_msa=torch.zeros(batch, 1, action_dim),
        timestep_embedding=torch.randn(batch, action_dim, generator=generator),
        proprio=torch.randn(batch, 3, generator=generator),
        video=LayerVideoKVView(
            key=torch.randn(batch, 4, action_dim, generator=generator),
            value=torch.randn(batch, 4, action_dim, generator=generator),
            current_frame_tokens=2,
            layout_metadata={"grid": (1, 2, 2)},
        ),
    )


def _reader_config(
    memory: NativePatchMemory,
    *,
    query_kind: str = "full_linear",
    output_kind: str = "full_linear",
) -> DinoSemanticReaderConfig:
    return DinoSemanticReaderConfig(
        action_hidden_dim=8,
        timestep_dim=8,
        proprio_dim=3,
        camera_ids=memory.camera_ids,
        layer_indices=(0,),
        temperature=0.07,
        residual_scale=1.0,
        query_projection=ProjectionSpec(
            kind=query_kind,
            rank=None if query_kind == "full_linear" else 4,
        ),
        output_projection=ProjectionSpec(
            kind=output_kind,
            rank=None if output_kind == "full_linear" else 4,
        ),
        memory_contract_sha256=memory.memory_contract_sha256,
    )


def test_frozen_encoder_routes_before_preprocess_and_scatter_valid_views() -> None:
    model = _FakeDino()
    encoder = FrozenDinoV3Encoder._from_preloaded_model_for_tests(
        model=model,
        asset=_asset(),
    )
    cameras = _camera_batch()

    assert encoder.prepare_memory(PolicyRegime.IDM, cameras) is None
    assert model.calls == 0
    encoder.train()
    assert not encoder.training
    assert not model.training

    memory = encoder.prepare_memory(PolicyRegime.UNCOND, cameras)

    assert memory is not None
    assert model.calls == 1
    assert memory.tokens.shape == (2, 2, 196, 384)
    assert torch.count_nonzero(memory.tokens[0, 1]) == 0
    assert memory.camera_ids == cameras.camera_ids
    assert not memory.tokens.requires_grad
    assert not memory.tokens.is_inference()
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    assert memory.memory_contract_sha256 == native_memory_contract_sha256(
        encoder.asset,
        camera_ids=cameras.camera_ids,
        input_contract_sha256=cameras.input_contract_sha256,
    )

    probe = nn.Linear(DINO_V3_NATIVE_DIM, 1, bias=False)
    probe(memory.tokens).sum().backward()
    assert probe.weight.grad is not None
    assert torch.count_nonzero(probe.weight.grad) > 0


def test_native_memory_hash_binds_camera_layout_and_crop_orientation() -> None:
    asset = _asset()
    input_hash = _hash("camera-crop-orientation")
    expected_layout = {
        "patch_grid": [14, 14],
        "flatten_order": "row_major",
        "origin": "top_left",
        "x_direction": "right",
        "y_direction": "down",
        "coordinate": "normalized_patch_center",
        "camera_order": ["main", "wrist"],
    }

    assert native_patch_layout_contract(("main", "wrist")) == expected_layout
    assert _memory().layout_contract == expected_layout
    reference = native_memory_contract_sha256(
        asset,
        camera_ids=("main", "wrist"),
        input_contract_sha256=input_hash,
    )
    assert reference != native_memory_contract_sha256(
        asset,
        camera_ids=("wrist", "main"),
        input_contract_sha256=input_hash,
    )
    assert reference != native_memory_contract_sha256(
        asset,
        camera_ids=("main", "wrist"),
        input_contract_sha256=_hash("different-crop-orientation"),
    )


def test_frozen_encoder_rejects_wrong_native_output_contract() -> None:
    encoder = FrozenDinoV3Encoder._from_preloaded_model_for_tests(
        model=_FakeDino(output_key="x_storage_tokens"),
        asset=_asset(),
    )

    with pytest.raises(ValueError, match="x_norm_patchtokens"):
        encoder.encode(_camera_batch())


def test_asset_and_reader_configs_fail_closed() -> None:
    asset_payload = {
        "source_root": "/tmp/source",
        "source_revision": PINNED_DINOV3_SOURCE_REVISION,
        "model_name": PINNED_DINOV3_MODEL_NAME,
        "weights_path": "/tmp/model.pth",
        "weights_sha256": _hash("weights"),
        "preprocess_sha256": DINO_V3_PREPROCESS_SHA256,
        "output_contract_sha256": DINO_V3_OUTPUT_CONTRACT_SHA256,
        "compute_dtype": "float32",
        "license_id": "DINOv3 License",
    }
    assert (
        DinoV3AssetSpec.from_mapping(asset_payload).model_name
        == PINNED_DINOV3_MODEL_NAME
    )
    with pytest.raises(ValueError, match="unknown"):
        DinoV3AssetSpec.from_mapping({**asset_payload, "download": True})
    with pytest.raises(ValueError, match="preprocessing contract"):
        DinoV3AssetSpec.from_mapping(
            {**asset_payload, "preprocess_sha256": _hash("wrong")}
        )

    memory = _memory()
    config = _reader_config(memory)
    payload = config.contract_payload()
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="unknown"):
        DinoSemanticReaderConfig.from_mapping(payload)
    with pytest.raises(ValueError, match="temperature"):
        DinoSemanticReaderConfig(
            **{**config.__dict__, "temperature": 0.0},
        )


def test_native_router_matches_reference_and_preserves_view_slots() -> None:
    memory = _memory(second_view_valid=False)
    context = _context()
    router = NativeDinoRouter(
        action_hidden_dim=8,
        temperature=0.07,
        projection=ProjectionSpec(kind="full_linear", rank=None),
        camera_ids=memory.camera_ids,
    )

    routed = router(context, memory)
    queries = router.query_projection(context.modulated_attn_input).float()
    queries = torch.nn.functional.normalize(queries, dim=-1)
    keys = torch.nn.functional.normalize(memory.tokens.float(), dim=-1)
    reference_scores = torch.einsum("bad,bvnd->bvan", queries, keys) / 0.07
    reference = torch.softmax(reference_scores, dim=-1)

    torch.testing.assert_close(routed.weights[0, 0], reference[0, 0])
    assert torch.count_nonzero(routed.weights[0, 1]) == 0
    torch.testing.assert_close(
        routed.weights[memory.camera_valid_mask].sum(dim=-1),
        torch.ones_like(routed.weights[memory.camera_valid_mask].sum(dim=-1)),
    )

    permutation = torch.randperm(DINO_V3_PATCH_COUNT)
    permuted_memory = NativePatchMemory(
        **{
            **memory.__dict__,
            "tokens": memory.tokens[:, :, permutation],
            "patch_valid_mask": memory.patch_valid_mask[:, :, permutation],
        }
    )
    permuted = router(context, permuted_memory)
    torch.testing.assert_close(permuted.weights, routed.weights[..., permutation])


def test_native_memory_rejects_zero_norm_and_non_finite_patches() -> None:
    memory = _memory()
    zeroed = memory.tokens.clone()
    zeroed[0, 0, 0].zero_()
    with pytest.raises(ValueError, match="non-zero row norms"):
        NativePatchMemory(**{**memory.__dict__, "tokens": zeroed})

    non_finite = memory.tokens.clone()
    non_finite[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="non-finite"):
        NativePatchMemory(**{**memory.__dict__, "tokens": non_finite})


def test_p1_zero_init_and_gradient_unlock_order() -> None:
    memory = _memory()
    context = _context()
    reader = build_dino_semantic_reader(_reader_config(memory))
    router = reader.routers["0"]
    branch = reader.branches["0"][0]

    initial = reader.forward_layer(context, memory)
    assert torch.count_nonzero(initial.tensor) == 0
    initial.tensor.sum().backward()

    output_weight = branch.output_projection.weight
    query_weight = router.query_projection.weight
    assert output_weight.grad is not None
    assert torch.count_nonzero(output_weight.grad) > 0
    assert query_weight.grad is not None
    assert torch.count_nonzero(query_weight.grad) == 0
    assert branch.semantic_gate.weight.grad is not None
    assert torch.count_nonzero(branch.semantic_gate.weight.grad) == 0

    reader.zero_grad(set_to_none=True)
    with torch.no_grad():
        output_weight.fill_(0.01)
    unlocked = reader.forward_layer(context, memory)
    unlocked.tensor.sum().backward()

    assert torch.count_nonzero(query_weight.grad) > 0
    assert torch.count_nonzero(branch.semantic_gate.weight.grad) > 0


def test_low_rank_reader_state_is_strict_and_reader_only() -> None:
    memory = _memory()
    config = _reader_config(
        memory,
        query_kind="low_rank",
        output_kind="low_rank",
    )
    reader = build_dino_semantic_reader(config)
    payload = reader.export_trainable_state()
    restored = build_dino_semantic_reader(config)
    restored.load_trainable_state(payload)

    assert payload["reader_kind"] == DINO_SEMANTIC_READER_KIND
    assert all(
        "model" not in name and "encoder" not in name for name in payload["state"]
    )
    for first, second in zip(reader.parameters(), restored.parameters()):
        assert torch.equal(first, second)

    incompatible = build_dino_semantic_reader(_reader_config(memory))
    with pytest.raises(ValueError, match="contract hash"):
        incompatible.load_trainable_state(payload)


class _RecordingBranch(VisualValueBranch):
    def __init__(self, branch_kind: str) -> None:
        super().__init__()
        self.branch_kind = branch_kind
        self.branch_contract_sha256 = contract_sha256({"kind": branch_kind})
        self.scale = nn.Parameter(torch.zeros(()))
        self.routing_object_id: int | None = None

    def forward_branch(
        self,
        context: ActionLayerReadContext,
        memory: NativePatchMemory,
        routing: RoutingWeights,
    ) -> torch.Tensor:
        del memory
        self.routing_object_id = id(routing)
        return torch.zeros_like(context.post_block_hidden) + self.scale * 0


def test_routed_reader_shares_one_routing_map_across_pluggable_branches() -> None:
    memory = _memory()
    context = _context()
    router = NativeDinoRouter(
        action_hidden_dim=8,
        temperature=0.07,
        projection=ProjectionSpec(kind="full_linear", rank=None),
        camera_ids=memory.camera_ids,
    )
    semantic = _RecordingBranch("semantic-test")
    wan = _RecordingBranch("wan-test")
    reader = RoutedVisualReader(
        routers={0: router},
        branches={0: (semantic, wan)},
        memory_contract_sha256=memory.memory_contract_sha256,
        reader_kind="dual-test-reader",
    )

    residual = reader.forward_layer(context, memory)

    assert semantic.routing_object_id == wan.routing_object_id
    assert residual.branch_kinds == ("semantic-test", "wan-test")
    assert torch.count_nonzero(residual.tensor) == 0


@pytest.mark.skipif(
    not all(
        os.environ.get(name)
        for name in (
            "FASTWAM_DINOV3_SOURCE_ROOT",
            "FASTWAM_DINOV3_WEIGHTS",
            "FASTWAM_DINOV3_WEIGHTS_SHA256",
        )
    ),
    reason="real DINOv3 asset was not supplied",
)
def test_real_dinov3_native_output_contract() -> None:
    asset = DinoV3AssetSpec(
        source_root=Path(os.environ["FASTWAM_DINOV3_SOURCE_ROOT"]),
        source_revision=PINNED_DINOV3_SOURCE_REVISION,
        model_name=PINNED_DINOV3_MODEL_NAME,
        weights_path=Path(os.environ["FASTWAM_DINOV3_WEIGHTS"]),
        weights_sha256=os.environ["FASTWAM_DINOV3_WEIGHTS_SHA256"],
        preprocess_sha256=DINO_V3_PREPROCESS_SHA256,
        output_contract_sha256=DINO_V3_OUTPUT_CONTRACT_SHA256,
        compute_dtype="float32",
        license_id="DINOv3 License",
    )
    encoder = FrozenDinoV3Encoder.from_local_asset(asset, device="cpu")
    memory = encoder.encode(_camera_batch())
    assert memory.tokens.shape == (2, 2, 196, 384)
    assert not memory.tokens.is_inference()
    assert torch.isfinite(memory.tokens).all()
    assert torch.count_nonzero(memory.tokens[memory.patch_valid_mask]) > 0
    assert torch.count_nonzero(memory.tokens[~memory.patch_valid_mask]) == 0

    probe = nn.Linear(DINO_V3_NATIVE_DIM, 1, bias=False)
    probe(memory.tokens).sum().backward()
    assert probe.weight.grad is not None
    assert torch.isfinite(probe.weight.grad).all()
    assert torch.count_nonzero(probe.weight.grad) > 0
