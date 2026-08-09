from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fastwam.adapters import PolicyRegime
from fastwam.models.wan22.adaptive_action import (
    CachedActionCondition,
    CachedActionVelocity,
)
from fastwam.models.wan22.kv_tap import GateKVTapRequest
from fastwam.models.wan22.mot import MoT
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
    NativePatchMemory,
    PreparedCameraBatch,
)
from fastwam.models.wan22.wan_current_refiner import (
    ActionVideoKVView,
    WanCurrentKVRefiner,
    WanCurrentRefinerConfig,
    WanCurrentSourceCaptureRequest,
)
from fastwam.models.wan22.wan_video_dit import DiTBlock
from fastwam.runtime import (
    create_wan_current_refinement_sidecar,
    validate_wan_current_refinement_config,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


class _TinyExpert(nn.Module):
    def __init__(self, *, layers: int = 2) -> None:
        super().__init__()
        self.num_heads = 2
        self.attn_head_dim = 4
        self.use_gradient_checkpointing = False
        self.blocks = nn.ModuleList(
            DiTBlock(
                hidden_dim=8,
                attn_head_dim=4,
                num_heads=2,
                ffn_dim=16,
            )
            for _ in range(layers)
        )


class _TinyAction(_TinyExpert):
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


class _TinyDino(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        basis = torch.linspace(
            0.25,
            1.25,
            DINO_V3_NATIVE_DIM,
            dtype=images.dtype,
            device=images.device,
        )
        signal = images.mean(dim=(1, 2, 3))[:, None, None]
        tokens = basis[None, None].expand(images.shape[0], DINO_V3_PATCH_COUNT, -1)
        return {"x_norm_patchtokens": tokens + signal}


def _freqs(tokens: int) -> torch.Tensor:
    return torch.ones(tokens, 1, 2, dtype=torch.complex128)


def _memory(batch: int = 2) -> NativePatchMemory:
    generator = torch.Generator().manual_seed(41)
    tokens = torch.randn(batch, 2, 196, 384, generator=generator)
    camera_valid = torch.ones(batch, 2, dtype=torch.bool)
    return NativePatchMemory(
        tokens=tokens,
        patch_valid_mask=camera_valid.unsqueeze(-1).expand(-1, -1, 196),
        camera_valid_mask=camera_valid,
        camera_ids=("main", "wrist"),
        grid=(14, 14),
        source_revision="test",
        weights_sha256=_hash("weights"),
        input_contract_sha256=_hash("input"),
        preprocess_sha256=_hash("preprocess"),
        output_contract_sha256=_hash("output"),
        memory_contract_sha256=_hash("memory"),
    )


def _mot_and_sources():
    torch.manual_seed(17)
    video = _TinyExpert()
    action = _TinyAction()
    mot = MoT(
        mixtures={"video": video, "action": action},
        mot_checkpoint_mixed_attn=False,
    ).eval()
    mot.requires_grad_(False)
    request = WanCurrentSourceCaptureRequest.create(
        layer_indices=(0, 1),
        current_frame_video_tokens=4,
        camera_index_current=torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]),
        source_contract_sha256=_hash("wan-source"),
    )
    video_tokens = torch.randn(2, 4, 8)
    mask = torch.ones(4, 4, dtype=torch.bool)
    base = mot.prefill_video_cache(
        video_tokens=video_tokens,
        video_freqs=_freqs(4),
        video_t_mod=torch.zeros(2, 6, 8),
        video_context_payload=None,
        video_attention_mask=mask,
        gate_current_frame_video_tokens=4,
        wan_current_source_capture=request,
    )
    return mot, action, base, request.snapshot()


def _refiner() -> WanCurrentKVRefiner:
    return WanCurrentKVRefiner(
        WanCurrentRefinerConfig(
            wan_hidden_dim=8,
            native_dim=384,
            layer_indices=(0, 1),
            query_rank=4,
            output_rank=4,
            temperature=0.2,
            alpha=0.5,
            memory_contract_sha256=_hash("memory"),
            source_contract_sha256=_hash("wan-source"),
        )
    )


def _action_kwargs(base):
    return {
        "action_tokens": torch.randn(2, 3, 8),
        "action_freqs": _freqs(3),
        "action_t_mod": torch.zeros(2, 6, 8),
        "action_context_payload": {
            "context": torch.randn(2, 5, 8),
            "mask": torch.ones(2, 3, 5, dtype=torch.bool),
        },
        "video_kv_cache": base,
        "attention_mask": torch.ones(7, 7, dtype=torch.bool),
        "video_seq_len": 4,
    }


def test_runtime_factory_is_default_off_and_compile_fails_closed() -> None:
    disabled = {
        "type": "dinov3_guided_wan_current_refinement",
        "enabled": False,
        "dino": object(),
    }
    assert validate_wan_current_refinement_config(disabled) == {
        "type": "dinov3_guided_wan_current_refinement",
        "enabled": False,
    }
    assert (
        create_wan_current_refinement_sidecar(
            disabled,
            actor=object(),
            device="cpu",
            dtype=torch.bfloat16,
        )
        is None
    )

    enabled = {
        "type": "dinov3_guided_wan_current_refinement",
        "enabled": True,
        "compile": True,
        "enabled_regimes": ["uncond"],
        "dino": {},
        "refiner": {},
        "camera_ids": ["main", "wrist"],
        "camera_input_contract_sha256": _hash("input"),
        "license_record_sha256": _hash("license"),
    }
    with pytest.raises(ValueError, match="compiled execution is not implemented"):
        validate_wan_current_refinement_config(enabled)


def test_runtime_factory_constructs_hash_bound_enabled_sidecar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dino = {
        "source_root": ".",
        "source_revision": PINNED_DINOV3_SOURCE_REVISION,
        "model_name": PINNED_DINOV3_MODEL_NAME,
        "weights_path": "/unused/dinov3.pt",
        "weights_sha256": _hash("weights"),
        "preprocess_sha256": DINO_V3_PREPROCESS_SHA256,
        "output_contract_sha256": DINO_V3_OUTPUT_CONTRACT_SHA256,
        "compute_dtype": "bfloat16",
        "license_id": "DINOv3-test-license",
    }
    camera_hash = _hash("camera-input")
    memory_hash = native_memory_contract_sha256(
        DinoV3AssetSpec.from_mapping(dino),
        camera_ids=("main", "wrist"),
        input_contract_sha256=camera_hash,
    )
    config = {
        "type": "dinov3_guided_wan_current_refinement",
        "enabled": True,
        "compile": False,
        "enabled_regimes": ["uncond"],
        "dino": dino,
        "refiner": {
            "wan_hidden_dim": 8,
            "native_dim": 384,
            "layer_indices": [0],
            "query_rank": 2,
            "output_rank": 2,
            "temperature": 0.2,
            "alpha": 1.0,
            "memory_contract_sha256": memory_hash,
            "source_contract_sha256": _hash("source"),
        },
        "camera_ids": ["main", "wrist"],
        "camera_input_contract_sha256": camera_hash,
        "license_record_sha256": _hash("license-record"),
    }
    fake_encoder = SimpleNamespace(asset=DinoV3AssetSpec.from_mapping(dino))
    monkeypatch.setattr(
        FrozenDinoV3Encoder,
        "from_local_asset",
        staticmethod(lambda _asset, *, device: fake_encoder),
    )

    build = create_wan_current_refinement_sidecar(
        config,
        actor=SimpleNamespace(
            video_expert=SimpleNamespace(hidden_dim=8),
            mot=SimpleNamespace(num_layers=2),
        ),
        device="cpu",
        dtype=torch.bfloat16,
    )

    assert build is not None
    assert build.encoder is fake_encoder
    assert build.refiner.config.memory_contract_sha256 == memory_hash


def test_selected_sources_are_detached_current_only_and_future_fails_closed() -> None:
    mot, _action, _base, sources = _mot_and_sources()

    assert tuple(source.layer_index for source in sources) == (0, 1)
    assert all(source.hidden_current.shape == (2, 4, 8) for source in sources)
    assert all(not source.hidden_current.requires_grad for source in sources)
    assert all(not source.attention_input_current.requires_grad for source in sources)
    assert all(not source.key_pre_norm_current.requires_grad for source in sources)

    request = WanCurrentSourceCaptureRequest.create(
        layer_indices=(0,),
        current_frame_video_tokens=2,
        camera_index_current=torch.zeros(2, 2, dtype=torch.long),
        source_contract_sha256=_hash("wan-source"),
    )
    with pytest.raises(ValueError, match="generated future video is forbidden"):
        mot.prefill_video_cache(
            video_tokens=torch.randn(2, 4, 8),
            video_freqs=_freqs(4),
            video_t_mod=torch.zeros(2, 6, 8),
            video_context_payload=None,
            video_attention_mask=torch.ones(4, 4, dtype=torch.bool),
            wan_current_source_capture=request,
        )


def test_disabled_alias_and_active_zero_are_eager_exact_with_live_gradient() -> None:
    mot, _action, base, sources = _mot_and_sources()
    refiner = _refiner().train()
    memory = _memory()
    base_alias = ActionVideoKVView.base_alias(base, actor_version=3)

    assert base_alias.layer(0)[0] is base[0]["k"]
    assert base_alias.layer(0)[1] is base[0]["v"]

    shadow = refiner.build_action_view(
        base_video_kv_cache=base,
        sources=sources,
        memory=memory,
        video_blocks=mot.mixtures["video"].blocks,
        actor_version=3,
    )
    for layer_index in (0, 1):
        key, value = shadow.layer(layer_index)
        assert torch.equal(key, base[layer_index]["k"])
        assert torch.equal(value, base[layer_index]["v"])

    kwargs = _action_kwargs(base)
    baseline = mot.forward_action_with_video_cache(**kwargs)
    active_zero = mot.forward_action_with_video_cache(
        **kwargs,
        action_video_kv_view=shadow,
    )
    assert torch.equal(active_zero, baseline)

    active_zero.sum().backward()
    up_gradients = [module.output_up.weight.grad for module in refiner.layers.values()]
    assert all(gradient is not None for gradient in up_gradients)
    assert all(torch.count_nonzero(gradient) > 0 for gradient in up_gradients)
    assert all(parameter.grad is None for parameter in mot.parameters())


def test_p8_encoder_memory_leaves_inference_mode_for_live_refiner_gradient() -> None:
    camera_batch = PreparedCameraBatch(
        pixels=torch.full((2, 2, 3, 224, 224), 127, dtype=torch.uint8),
        camera_ids=("main", "wrist"),
        camera_valid_mask=torch.ones(2, 2, dtype=torch.bool),
        input_contract_sha256=_hash("camera-input"),
    )
    asset = DinoV3AssetSpec(
        source_root="/unused/dinov3",
        source_revision=PINNED_DINOV3_SOURCE_REVISION,
        model_name=PINNED_DINOV3_MODEL_NAME,
        weights_path="/unused/model.safetensors",
        weights_sha256=_hash("weights"),
        preprocess_sha256=DINO_V3_PREPROCESS_SHA256,
        output_contract_sha256=DINO_V3_OUTPUT_CONTRACT_SHA256,
        compute_dtype="float32",
        license_id="DINOv3-test-license",
    )
    encoder = FrozenDinoV3Encoder._from_preloaded_model_for_tests(
        model=_TinyDino(),
        asset=asset,
    )
    memory = encoder.encode(camera_batch)
    assert not memory.tokens.is_inference()
    assert not memory.tokens.requires_grad

    mot, _action, base, sources = _mot_and_sources()
    refiner = WanCurrentKVRefiner(
        WanCurrentRefinerConfig(
            wan_hidden_dim=8,
            native_dim=384,
            layer_indices=(0, 1),
            query_rank=4,
            output_rank=4,
            temperature=0.2,
            alpha=0.5,
            memory_contract_sha256=memory.memory_contract_sha256,
            source_contract_sha256=_hash("wan-source"),
        )
    ).train()
    shadow = refiner.build_action_view(
        base_video_kv_cache=base,
        sources=sources,
        memory=memory,
        video_blocks=mot.mixtures["video"].blocks,
        actor_version=0,
    )
    output = mot.forward_action_with_video_cache(
        **_action_kwargs(base),
        action_video_kv_view=shadow,
    )
    output.sum().backward()

    gradients = [module.output_up.weight.grad for module in refiner.layers.values()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(torch.count_nonzero(gradient) > 0 for gradient in gradients)
    assert all(parameter.grad is None for parameter in encoder.parameters())


def test_nonzero_shadow_changes_action_but_gate_direct_video_reads_base() -> None:
    mot, _action, base, sources = _mot_and_sources()
    refiner = _refiner().eval()
    for module in refiner.layers.values():
        nn.init.normal_(module.output_up.weight, std=0.1)
    shadow = refiner.build_action_view(
        base_video_kv_cache=base,
        sources=sources,
        memory=_memory(),
        video_blocks=mot.mixtures["video"].blocks,
        actor_version=7,
    )
    kwargs = _action_kwargs(base)
    baseline = mot.forward_action_with_video_cache(**kwargs)
    tap = GateKVTapRequest(
        current_mode=PolicyRegime.UNCOND,
        denoise_timestep=0.5,
        current_frame_video_tokens=4,
        actor_version=7,
    )
    refined = mot.forward_action_with_video_cache(
        **kwargs,
        action_video_kv_view=shadow,
        kv_tap=tap,
    )

    assert not torch.equal(refined, baseline)
    for index, layer in enumerate(tap.snapshot().layers):
        assert torch.equal(layer.current_frame_video.key, base[index]["k"])
        assert torch.equal(layer.current_frame_video.value, base[index]["v"])
        assert not torch.equal(shadow.layer(index)[0], base[index]["k"])


def test_idm_cached_velocity_rejects_p8_shadow() -> None:
    mot, action, base, sources = _mot_and_sources()
    refiner = _refiner().eval()
    shadow = refiner.build_action_view(
        base_video_kv_cache=base,
        sources=sources,
        memory=_memory(),
        video_blocks=mot.mixtures["video"].blocks,
        actor_version=0,
    )
    condition = CachedActionCondition(
        context=torch.randn(2, 5, 8),
        context_mask=torch.ones(2, 5, dtype=torch.bool),
        video_kv_cache=base,
        attention_mask=torch.ones(7, 7, dtype=torch.bool),
        video_seq_len=4,
        current_frame_video_tokens=4,
        action_video_kv_view=shadow,
    )

    with pytest.raises(ValueError, match="IDM action velocity must reject"):
        CachedActionVelocity(
            action_expert=action,
            mot=mot,
            condition=condition,
            regime=PolicyRegime.IDM,
        )


def test_refiner_checkpoint_and_behavior_snapshot_are_strict_and_trainable_only() -> (
    None
):
    refiner = _refiner()
    payload = refiner.checkpoint_state()
    assert payload["schema"] == "fastwam-wan-current-refiner-v1"
    assert set(payload["state"]) == set(refiner.state_dict())
    assert all("self_attn" not in name for name in payload["state"])
    manifest = refiner.trainable_parameter_manifest()
    assert {entry.name for entry in manifest} == {
        name for name, _parameter in refiner.named_parameters()
    }
    assert {id(entry.parameter) for entry in manifest} == {
        id(parameter) for parameter in refiner.parameters()
    }

    snapshot = refiner.capture_behavior_snapshot(actor_version=4)
    original = {name: tensor.clone() for name, tensor in refiner.state_dict().items()}
    with torch.no_grad():
        next(refiner.parameters()).add_(1)
    with refiner.use_behavior_snapshot(snapshot, actor_version=4):
        assert all(
            torch.equal(refiner.state_dict()[name].cpu(), snapshot.state[name])
            for name in snapshot.state
        )
    assert any(
        not torch.equal(refiner.state_dict()[name], original[name]) for name in original
    )

    with pytest.raises(ValueError, match="actor version mismatch"):
        with refiner.use_behavior_snapshot(snapshot, actor_version=5):
            pass
