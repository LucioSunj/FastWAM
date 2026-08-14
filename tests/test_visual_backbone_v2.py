from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

from fastwam.adapters import (
    ActionLoRATargetGroup,
    PolicyRegime,
    RegimeLoRAConfig,
    inject_action_dit_lora,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import (
    prepare_visual_camera_from_raw,
)
from fastwam.models.wan22.visual_backbone import (
    FrozenVisualPatchEncoder,
    VisualBackboneAssetSpec,
    get_visual_backbone_preset,
    verify_visual_backbone_asset,
    visual_output_contract,
    visual_preprocess_contract,
)
from fastwam.models.wan22.visual_contracts import (
    ActionLayerReadContext,
    LayerVideoKVView,
    PreparedVisualCameraBatch,
    SpatialPatchMemory,
    VISUAL_READER_STATE_SCHEMA_V2,
    contract_sha256,
)
from fastwam.models.wan22.visual_sidecar import (
    DinoContributionDiagnosticsCollector,
    ProjectionSpec,
    VisualPatchReaderConfig,
    build_visual_patch_reader,
)
from fastwam.p1_visual_bc_checkpoint import (
    inspect_p1_visual_bc_checkpoint,
    load_p1_visual_bc_checkpoint,
    save_p1_visual_bc_checkpoint,
)
from fastwam.p1_visual_bc_full_checkpoint import (
    inspect_p1_visual_bc_full_checkpoint,
    load_p1_visual_bc_full_checkpoint,
    load_p1_visual_bc_full_weights_for_evaluation,
    save_p1_visual_bc_full_checkpoint,
)
from fastwam.uncond_bc_checkpoint import capture_rng_state


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _asset(
    family: str,
    variant: str,
    size: int,
    *,
    microbatch: int = 2,
) -> VisualBackboneAssetSpec:
    preset = get_visual_backbone_preset(family, variant, input_size=size)
    filename = "model.safetensors" if family == "dinov3" else "model.pt"
    return VisualBackboneAssetSpec(
        family=family,
        variant=variant,
        input_size=size,
        source_root="/unit-test/source",
        source_revision=preset.source_revision,
        weights_revision=preset.weights_revision,
        weights_path=f"/unit-test/{variant}/{filename}",
        weights_sha256=preset.weights_sha256,
        preprocess_sha256=contract_sha256(
            visual_preprocess_contract(preset, input_size=size)
        ),
        output_contract_sha256=contract_sha256(
            visual_output_contract(preset, input_size=size)
        ),
        compute_dtype="float32",
        encode_microbatch_size=microbatch,
        license_id=preset.license_id,
    )


class _FakePatchModel(nn.Module):
    def __init__(self, patches: int, dimension: int) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.patches = patches
        self.dimension = dimension
        self.chunk_sizes: list[int] = []


class _FakePatchEncoder(FrozenVisualPatchEncoder):
    def _forward_patch_tokens(self, normalized: torch.Tensor) -> torch.Tensor:
        model = self.model
        model.chunk_sizes.append(normalized.shape[0])
        basis = torch.linspace(
            0.25,
            1.25,
            model.dimension,
            dtype=normalized.dtype,
            device=normalized.device,
        )
        patch = torch.linspace(
            0.0,
            0.5,
            model.patches,
            dtype=normalized.dtype,
            device=normalized.device,
        )
        signal = normalized.mean(dim=(1, 2, 3))
        return basis[None, None] + patch[None, :, None] + signal[:, None, None]


def _camera_batch(size: int) -> PreparedVisualCameraBatch:
    pixels = (
        torch.arange(
            2 * 2 * 3 * size * size,
            dtype=torch.int64,
        )
        .remainder(256)
        .to(torch.uint8)
        .reshape(2, 2, 3, size, size)
    )
    pixels[0, 1].zero_()
    return PreparedVisualCameraBatch(
        pixels=pixels,
        camera_ids=("main", "wrist"),
        camera_valid_mask=torch.tensor([[True, False], [True, True]]),
        input_size=size,
        input_contract_sha256=_hash(f"visual-input-{size}"),
        source_resolution=torch.tensor(
            [[[512, 512], [0, 0]], [[512, 512], [256, 256]]],
            dtype=torch.int32,
        ),
    )


@pytest.mark.parametrize(
    ("family", "variant", "size", "dimension", "grid"),
    [
        ("dinov3", "vits16", 224, 384, (14, 14)),
        ("dinov3", "vitb16", 224, 768, (14, 14)),
        ("dinov3", "vitl16", 224, 1024, (14, 14)),
        ("lingbot_vision", "small", 224, 384, (14, 14)),
        ("lingbot_vision", "small", 512, 384, (32, 32)),
        ("lingbot_vision", "base", 224, 768, (14, 14)),
        ("lingbot_vision", "base", 512, 768, (32, 32)),
        ("lingbot_vision", "large", 224, 1024, (14, 14)),
        ("lingbot_vision", "large", 512, 1024, (32, 32)),
    ],
)
def test_all_nine_registered_presets_have_fixed_dimensions_and_grids(
    family: str,
    variant: str,
    size: int,
    dimension: int,
    grid: tuple[int, int],
) -> None:
    asset = _asset(family, variant, size)
    assert asset.preset.native_dim == dimension
    assert asset.grid == grid
    assert asset.patch_count == grid[0] * grid[1]


def test_registry_rejects_giant_arbitrary_sizes_and_wrong_contracts() -> None:
    with pytest.raises(ValueError, match="Unsupported visual backbone"):
        get_visual_backbone_preset("dinov3", "vitg16", input_size=224)
    with pytest.raises(ValueError, match="unsupported"):
        get_visual_backbone_preset("dinov3", "vits16", input_size=512)
    with pytest.raises(ValueError, match="unsupported"):
        get_visual_backbone_preset("lingbot_vision", "small", input_size=384)
    with pytest.raises(ValueError, match="preprocessing contract"):
        replace(
            _asset("dinov3", "vits16", 224),
            preprocess_sha256=_hash("wrong"),
        )
    payload = dict(_asset("dinov3", "vits16", 224).__dict__)
    payload["native_dim"] = 384
    with pytest.raises(ValueError, match="unknown"):
        VisualBackboneAssetSpec.from_mapping(payload)


def test_asset_verifier_rejects_wrong_weight_before_model_allocation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    bad_weight = tmp_path / "model.safetensors"
    bad_weight.write_bytes(b"not-a-checkpoint")
    asset = replace(
        _asset("dinov3", "vits16", 224),
        source_root=source,
        weights_path=bad_weight,
    )
    with pytest.raises(ValueError, match="size"):
        verify_visual_backbone_asset(asset)


def test_encoder_microbatches_valid_views_and_idm_bypasses_sidecar() -> None:
    asset = _asset("lingbot_vision", "small", 512, microbatch=2)
    model = _FakePatchModel(asset.patch_count, asset.preset.native_dim)
    encoder = _FakePatchEncoder(model=model, asset=asset)
    cameras = _camera_batch(512)

    assert encoder.prepare_memory(PolicyRegime.IDM, cameras) is None
    assert model.chunk_sizes == []
    encoder.train(True)
    assert not encoder.training and not model.training
    memory = encoder.prepare_memory(PolicyRegime.UNCOND, cameras)

    assert memory is not None
    assert model.chunk_sizes == [2, 1]
    assert memory.tokens.shape == (2, 2, 1024, 384)
    assert torch.count_nonzero(memory.tokens[0, 1]) == 0
    assert memory.grid == (32, 32)
    assert memory.source_resolution.tolist() == cameras.source_resolution.tolist()
    assert not memory.tokens.requires_grad and not memory.tokens.is_inference()
    assert all(not parameter.requires_grad for parameter in encoder.parameters())


def test_lingbot_512_is_built_from_raw_rgb_and_records_source_resolution() -> None:
    raw = torch.zeros(3, 300, 500, dtype=torch.uint8)
    raw[:, :, 200:300] = 255
    pixels, resolution = prepare_visual_camera_from_raw(raw, input_size=512)

    assert pixels.shape == (3, 512, 512)
    assert pixels.dtype == torch.uint8
    assert resolution.tolist() == [300, 500]
    assert torch.count_nonzero(pixels) > 0


def _context(batch: int = 2, hidden: int = 8) -> ActionLayerReadContext:
    generator = torch.Generator().manual_seed(8)
    value = torch.randn(batch, 4, hidden, generator=generator)
    return ActionLayerReadContext(
        layer_index=0,
        pre_block_hidden=value.clone(),
        modulated_attn_input=value.clone(),
        post_block_hidden=value.clone(),
        base_gate_msa=torch.ones(batch, 1, hidden),
        timestep_embedding=torch.randn(batch, hidden, generator=generator),
        proprio=torch.randn(batch, 3, generator=generator),
        video=LayerVideoKVView(
            key=torch.randn(batch, 2, hidden, generator=generator),
            value=torch.randn(batch, 2, hidden, generator=generator),
            current_frame_tokens=2,
        ),
    )


def _reader(memory: SpatialPatchMemory):
    return build_visual_patch_reader(
        VisualPatchReaderConfig(
            action_hidden_dim=8,
            timestep_dim=8,
            proprio_dim=3,
            memory_dim=memory.native_dim,
            camera_ids=memory.camera_ids,
            layer_indices=(0,),
            temperature=0.07,
            residual_scale=1.0,
            query_projection=ProjectionSpec(kind="full_linear", rank=None),
            output_projection=ProjectionSpec(kind="full_linear", rank=None),
            memory_contract_sha256=memory.memory_contract_sha256,
            semantic_gate_floor=0.05,
            semantic_gate_temperature=1.25,
        )
    )


def test_dynamic_reader_normalizes_routes_masks_missing_camera_and_owns_gradients() -> (
    None
):
    asset = _asset("lingbot_vision", "base", 224)
    encoder = _FakePatchEncoder(
        model=_FakePatchModel(asset.patch_count, asset.preset.native_dim),
        asset=asset,
    )
    memory = encoder.encode(_camera_batch(224))
    reader = _reader(memory)
    context = _context()
    routing = reader.routers["0"](context, memory)

    assert torch.count_nonzero(routing.weights[0, 1]) == 0
    torch.testing.assert_close(
        routing.weights[memory.camera_valid_mask].sum(dim=-1),
        torch.ones_like(routing.weights[memory.camera_valid_mask].sum(dim=-1)),
    )
    initial = reader.forward_layer(context, memory)
    assert torch.count_nonzero(initial.tensor) == 0
    initial.tensor.sum().backward()
    branch = reader.branches["0"][0]
    assert branch.output_projection.weight.grad is not None
    assert torch.count_nonzero(branch.output_projection.weight.grad) > 0
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert reader.export_trainable_state()["schema"] == VISUAL_READER_STATE_SCHEMA_V2


def test_v2_dino_dropout_is_per_sample_exact_zero_and_diagnostics_are_complete() -> (
    None
):
    asset = _asset("dinov3", "vits16", 224)
    encoder = _FakePatchEncoder(
        model=_FakePatchModel(asset.patch_count, asset.preset.native_dim),
        asset=asset,
    )
    memory = encoder.encode(_camera_batch(224))
    reader = _reader(memory)
    branch = reader.branches["0"][0]
    with torch.no_grad():
        branch.output_projection.weight.fill_(0.01)
    context = _context()
    keep = torch.tensor([False, True])
    collector = DinoContributionDiagnosticsCollector()

    with reader.capture_diagnostics(collector):
        output = reader.forward_layer(
            context,
            memory,
            dino_keep_mask=keep,
        ).tensor

    assert torch.count_nonzero(output[0]) == 0
    assert torch.count_nonzero(output[1]) > 0
    assert torch.isfinite(output).all()
    record = collector.records[0]
    assert record["effective_gate"].shape == (2, 1)
    assert record["projected_norm"].shape == (2, 4)
    assert record["effective_residual_norm"].shape == (2, 4)
    assert record["effective_residual_sum"].shape == (4, 8)
    assert record["effective_residual_square_sum"].shape == (4, 8)
    assert record["sample_count"] == 2
    assert branch._diagnostics_collector is None


class _LoRABlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ffn = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8))


class _LoRAAction(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_LoRABlock()])


def _adapter():
    return inject_action_dit_lora(
        _LoRAAction(),
        RegimeLoRAConfig(
            rank=2,
            alpha=2.0,
            dropout=0.0,
            target_groups=(ActionLoRATargetGroup.FFN,),
        ),
    )


def test_v2_checkpoint_round_trip_and_cross_backbone_resume_fail_closed(
    tmp_path: Path,
) -> None:
    asset = _asset("dinov3", "vits16", 224)
    encoder = _FakePatchEncoder(
        model=_FakePatchModel(asset.patch_count, asset.preset.native_dim),
        asset=asset,
    )
    memory = encoder.encode(_camera_batch(224))
    reader = _reader(memory)
    adapter = _adapter()
    metadata = asset.checkpoint_metadata(
        camera_ids=memory.camera_ids,
        input_contract_sha256=memory.input_contract_sha256,
    )
    contract = {"resolved_config_sha256": _hash("v2-config")}
    path = tmp_path / "visual.pt"
    save_p1_visual_bc_checkpoint(
        path,
        adapter=adapter,
        reader=reader,
        global_step=2,
        stage="canary",
        arm="a3_joint",
        parent_checkpoint_sha256=_hash("parent"),
        visual_backbone=metadata,
        memory_contract_sha256=memory.memory_contract_sha256,
        contract=contract,
        provenance={"source": "unit-test"},
        trainer_state={
            "last_loss_action_bc": 1.0,
            "best_dev_loss_action_bc": None,
            "nonzero_update_count": 2,
        },
    )
    assert inspect_p1_visual_bc_checkpoint(path)["result"] == "PASS"
    load_p1_visual_bc_checkpoint(
        path,
        adapter=adapter,
        reader=reader,
        expected_parent_checkpoint_sha256=_hash("parent"),
        expected_visual_backbone=metadata,
        expected_contract=contract,
    )
    lingbot = _asset("lingbot_vision", "small", 224).checkpoint_metadata(
        camera_ids=memory.camera_ids,
        input_contract_sha256=memory.input_contract_sha256,
    )
    with pytest.raises(ValueError, match="contract mismatch"):
        load_p1_visual_bc_checkpoint(
            path,
            adapter=adapter,
            reader=reader,
            expected_parent_checkpoint_sha256=_hash("parent"),
            expected_visual_backbone=lingbot,
            expected_contract=contract,
        )


def test_v2_full_checkpoint_restores_optimizer_scheduler_and_rng_manifest(
    tmp_path: Path,
) -> None:
    asset = _asset("lingbot_vision", "base", 512)
    encoder = _FakePatchEncoder(
        model=_FakePatchModel(asset.patch_count, asset.preset.native_dim),
        asset=asset,
    )
    memory = encoder.encode(_camera_batch(512))
    reader = _reader(memory)
    adapter = _adapter()
    parameters = [
        *adapter.lora_parameters(),
        *reader.parameters(),
    ]
    optimizer = torch.optim.AdamW(parameters, lr=1.0e-3)
    for parameter in parameters:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda _step: 1.0,
    )
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    metadata = asset.checkpoint_metadata(
        camera_ids=memory.camera_ids,
        input_contract_sha256=memory.input_contract_sha256,
    )
    contract = {"resolved_config_sha256": _hash("full-config")}
    path = tmp_path / "full.pt"
    save_p1_visual_bc_full_checkpoint(
        path,
        adapter=adapter,
        reader=reader,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        grad_scaler=scaler,
        global_step=1,
        epoch=0,
        sampler_offset=4,
        rng_by_rank=[capture_rng_state()],
        parent_checkpoint_sha256=_hash("parent"),
        visual_backbone=metadata,
        memory_contract_sha256=memory.memory_contract_sha256,
        contract=contract,
        provenance={"source": "unit-test"},
        trainer_state={
            "best_validation_loss_action_bc": None,
            "best_step": None,
            "epochs_without_improvement": 0,
            "nonzero_update_count": 1,
        },
    )
    report = inspect_p1_visual_bc_full_checkpoint(path)
    assert report["result"] == "PASS"
    assert report["optimizer_tensor_count"] > 0
    restored_optimizer = torch.optim.AdamW(parameters, lr=1.0e-3)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer,
        lr_lambda=lambda _step: 1.0,
    )
    payload = load_p1_visual_bc_full_checkpoint(
        path,
        adapter=adapter,
        reader=reader,
        optimizer=restored_optimizer,
        lr_scheduler=restored_scheduler,
        grad_scaler=scaler,
        expected_parent_checkpoint_sha256=_hash("parent"),
        expected_visual_backbone=metadata,
        expected_contract=contract,
    )
    assert payload["global_step"] == 1
    assert len(payload["rng_by_rank"]) == 1
    evaluation_payload = load_p1_visual_bc_full_weights_for_evaluation(
        path,
        adapter=adapter,
        reader=reader,
        expected_parent_checkpoint_sha256=_hash("parent"),
        expected_visual_backbone=metadata,
        expected_contract=contract,
    )
    assert evaluation_payload["schema"] == report["schema"]


def test_hydra_exposes_all_nine_presets_with_matching_camera_contracts() -> None:
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    names = (
        "dinov3_vits16_224",
        "dinov3_vitb16_224",
        "dinov3_vitl16_224",
        "lingbot_small_224",
        "lingbot_small_512",
        "lingbot_base_224",
        "lingbot_base_512",
        "lingbot_large_224",
        "lingbot_large_512",
    )
    for name in names:
        with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
            cfg = compose(
                config_name="p1_visual_bc_full",
                overrides=[f"visual_backbone={name}"],
            )
        payload = OmegaConf.to_container(
            cfg.p1.visual_camera_input_contract,
            resolve=True,
        )
        assert contract_sha256(payload) == str(
            cfg.p1.visual_camera_input_contract_sha256
        )
        assert int(cfg.data.train.video_size[0]) == 224
        assert int(cfg.data.train.visual_camera_input_size) in {224, 512}
