from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from torch import nn

from fastwam.adapters import ActionLoRATargetGroup, PolicyRegime, RegimeLoRAConfig
from fastwam.datasets.lerobot.robot_video_dataset import _p1_camera_float_to_uint8
from fastwam.models.wan22.dinov3_memory import (
    DINO_V3_OUTPUT_CONTRACT_SHA256,
    DINO_V3_PREPROCESS_SHA256,
    PINNED_DINOV3_MODEL_NAME,
    PINNED_DINOV3_SOURCE_REVISION,
    DinoV3AssetSpec,
    FrozenDinoV3Encoder,
    native_memory_contract_sha256,
)
from fastwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)
from fastwam.models.wan22.visual_contracts import (
    DINO_V3_NATIVE_DIM,
    DINO_V3_PATCH_COUNT,
    ActionLayerReadContext,
    LayerVideoKVView,
    PreparedCameraBatch,
    contract_sha256,
)
from fastwam.models.wan22.visual_sidecar import (
    DinoSemanticReaderConfig,
    ProjectionSpec,
    build_dino_semantic_reader,
)
from fastwam.modality_dropout_bc import sample_modality_keep_mask
from fastwam.models.wan22.adaptive_action import ModalityKeepMask
from fastwam.p1_dino_bc import (
    P1_LORA_PARAMETER_FAMILY,
    FastWAMP1DinoBCConfig,
    FastWAMP1DinoBCPolicy,
    _intervene_memory,
    build_p1_optimizer,
)
from fastwam.p1_dino_bc_checkpoint import (
    inspect_p1_dino_bc_checkpoint,
    load_p1_dino_bc_checkpoint,
    save_p1_dino_bc_checkpoint,
)
from fastwam.p1_dino_bc_full_checkpoint import (
    inspect_p1_dino_bc_full_checkpoint,
    inspect_p1_dino_bc_full_checkpoint_v2,
    load_p1_dino_bc_full_checkpoint,
    load_p1_dino_bc_full_checkpoint_v2,
    save_p1_dino_bc_full_checkpoint,
    save_p1_dino_bc_full_checkpoint_v2,
)
from fastwam.p1_dino_contribution_v2 import (
    CausalCheckpointSelector,
    CausalSelectionThresholds,
    DependencyWarmupController,
    NegativeModeCycle,
    TaskPairedDistributedBatchSampler,
)
from fastwam.p1_dino_bc_full_trainer import (
    DINO_CONTRIBUTION_PROFILE,
    _memory_dependency_active,
    _memory_dependency_settings,
    _run_acceptance_passed,
    _suppress_parameter_family_gradients,
    _validate_full_config,
)
from fastwam.uncond_bc_checkpoint import capture_rng_state
from fastwam.uncond_bc import FastWAMUncondBCConfig


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_p1_camera_uint8_allows_only_declared_resize_roundoff() -> None:
    source = torch.tensor([0.0, 0.5, 1.0 + 2.4e-7], dtype=torch.float32)
    converted = _p1_camera_float_to_uint8(source, range_tolerance=1.0e-6)
    assert torch.equal(converted, torch.tensor([0, 128, 255], dtype=torch.uint8))

    with pytest.raises(ValueError, match="exceeds the allowed"):
        _p1_camera_float_to_uint8(
            torch.tensor([1.0 + 2.0e-6], dtype=torch.float32),
            range_tolerance=1.0e-6,
        )
    with pytest.raises(ValueError, match="finite values"):
        _p1_camera_float_to_uint8(
            torch.tensor([float("nan")], dtype=torch.float32),
            range_tolerance=1.0e-6,
        )


def _asset() -> DinoV3AssetSpec:
    return DinoV3AssetSpec(
        source_root="/unused/local/dinov3",
        source_revision=PINNED_DINOV3_SOURCE_REVISION,
        model_name=PINNED_DINOV3_MODEL_NAME,
        weights_path="/unused/local/dinov3_vits16.pth",
        weights_sha256=_hash("dino-vits16-weights"),
        preprocess_sha256=DINO_V3_PREPROCESS_SHA256,
        output_contract_sha256=DINO_V3_OUTPUT_CONTRACT_SHA256,
        compute_dtype="float32",
        license_id="DINOv3 License",
    )


class _FakeDino(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.calls = 0

    def forward_features(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        self.calls += 1
        patch = torch.arange(
            DINO_V3_PATCH_COUNT,
            dtype=images.dtype,
            device=images.device,
        ).view(1, -1, 1)
        channel = torch.linspace(
            0.25,
            1.25,
            DINO_V3_NATIVE_DIM,
            dtype=images.dtype,
            device=images.device,
        ).view(1, 1, -1)
        image_signal = images.mean(dim=(1, 2, 3)).view(-1, 1, 1)
        return {"x_norm_patchtokens": patch / 196.0 + channel + image_signal}


def _camera_batch(batch: int = 2) -> PreparedCameraBatch:
    pixels = (
        torch.arange(
            batch * 2 * 3 * 224 * 224,
            dtype=torch.int64,
        )
        .remainder(256)
        .to(torch.uint8)
        .reshape(batch, 2, 3, 224, 224)
    )
    return PreparedCameraBatch(
        pixels=pixels,
        camera_ids=("main", "wrist"),
        camera_valid_mask=torch.ones(batch, 2, dtype=torch.bool),
        input_contract_sha256=_hash("official-pil-crop-rgb-orientation-v1"),
    )


def _reader(memory_hash: str, *, hidden: int, timestep: int, proprio: int):
    return build_dino_semantic_reader(
        DinoSemanticReaderConfig(
            action_hidden_dim=hidden,
            timestep_dim=timestep,
            proprio_dim=proprio,
            camera_ids=("main", "wrist"),
            layer_indices=(0,),
            temperature=0.07,
            residual_scale=1.0,
            query_projection=ProjectionSpec(kind="full_linear", rank=None),
            output_projection=ProjectionSpec(kind="full_linear", rank=None),
            memory_contract_sha256=memory_hash,
        )
    )


def _read_context(*, batch: int = 2, hidden: int = 4) -> ActionLayerReadContext:
    generator = torch.Generator().manual_seed(123)
    pre = torch.randn(batch, 4, hidden, generator=generator)
    modulated = torch.randn(batch, 4, hidden, generator=generator)
    post = torch.randn(batch, 4, hidden, generator=generator)
    return ActionLayerReadContext(
        layer_index=0,
        pre_block_hidden=pre,
        modulated_attn_input=modulated,
        post_block_hidden=post,
        base_gate_msa=torch.ones(batch, 1, hidden),
        timestep_embedding=torch.randn(batch, hidden, generator=generator),
        proprio=torch.randn(batch, 2, generator=generator),
        video=LayerVideoKVView(
            key=torch.randn(batch, 1, hidden, generator=generator),
            value=torch.randn(batch, 1, hidden, generator=generator),
            current_frame_tokens=1,
        ),
    )


def test_encoder_materializes_ordinary_memory_and_binds_complete_layout() -> None:
    encoder = FrozenDinoV3Encoder._from_preloaded_model_for_tests(
        model=_FakeDino(),
        asset=_asset(),
    )
    cameras = _camera_batch()
    memory = encoder.encode(cameras)

    assert memory.tokens.shape == (2, 2, 196, 384)
    assert not memory.tokens.is_inference()
    assert not memory.tokens.requires_grad
    assert memory.layout_contract == {
        "patch_grid": [14, 14],
        "flatten_order": "row_major",
        "origin": "top_left",
        "x_direction": "right",
        "y_direction": "down",
        "coordinate": "normalized_patch_center",
        "camera_order": ["main", "wrist"],
    }
    assert memory.memory_contract_sha256 == native_memory_contract_sha256(
        encoder.asset,
        camera_ids=cameras.camera_ids,
        input_contract_sha256=cameras.input_contract_sha256,
    )
    changed_crop = native_memory_contract_sha256(
        encoder.asset,
        camera_ids=cameras.camera_ids,
        input_contract_sha256=_hash("different-crop-orientation"),
    )
    assert changed_crop != memory.memory_contract_sha256

    reader = _reader(memory.memory_contract_sha256, hidden=4, timestep=4, proprio=2)
    first = reader.forward_layer(_read_context(), memory)
    first.tensor.sum().backward()
    output = reader.branches["0"][0].output_projection.weight
    assert output.grad is not None
    assert torch.isfinite(output.grad).all()
    assert torch.count_nonzero(output.grad) > 0
    assert all(parameter.grad is None for parameter in encoder.parameters())


class _TinyBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, hidden),
        )


class _TinyActionExpert(nn.Module):
    def __init__(self, action_dim: int, hidden: int) -> None:
        super().__init__()
        self.input = nn.Linear(action_dim, hidden)
        self.blocks = nn.ModuleList([_TinyBlock(hidden)])
        self.output = nn.Linear(hidden, action_dim)

    def pre_dit(
        self,
        *,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> dict:
        hidden = self.input(action_tokens)
        time = timestep[:, None].expand(-1, hidden.shape[-1]) / 1000.0
        return {
            "tokens": hidden,
            "freqs": torch.empty(0),
            "t_mod": torch.empty(0),
            "t": time,
            "context": context,
            "context_mask": context_mask,
        }

    def post_dit(self, tokens: torch.Tensor, payload: dict) -> torch.Tensor:
        del payload
        return self.output(tokens)


class _TinyVideoExpert(nn.Module):
    fuse_vae_embedding_in_latents = True

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.arange(1, hidden + 1).float())

    def pre_dit(self, *, x: torch.Tensor, context: torch.Tensor, **kwargs) -> dict:
        del kwargs
        pooled = x.mean(dim=(1, 2, 3, 4)).view(-1, 1, 1)
        tokens = pooled * self.scale.view(1, 1, -1)
        return {
            "tokens": tokens,
            "freqs": torch.empty(0),
            "t_mod": torch.empty(0),
            "context": context,
            "context_mask": torch.ones(context.shape[:2], dtype=torch.bool),
            "meta": {"tokens_per_frame": 1},
        }


class _TinyMoT(nn.Module):
    def __init__(self, action_expert: _TinyActionExpert) -> None:
        super().__init__()
        self.action_expert = action_expert
        self.num_layers = 1
        self.keep_masks: list[tuple[torch.Tensor | None, torch.Tensor | None]] = []

    def prefill_video_cache(self, *, video_tokens: torch.Tensor, **kwargs):
        del kwargs
        return [{"k": video_tokens.detach(), "v": video_tokens.detach()}]

    def forward_action_with_video_cache(
        self,
        *,
        action_tokens: torch.Tensor,
        video_kv_cache: list[dict],
        visual_reader=None,
        visual_memory=None,
        visual_proprio=None,
        action_time_embedding=None,
        **kwargs,
    ) -> torch.Tensor:
        self.keep_masks.append(
            (
                None
                if kwargs.get("wan_keep_mask") is None
                else kwargs["wan_keep_mask"].detach().clone(),
                None
                if kwargs.get("dino_keep_mask") is None
                else kwargs["dino_keep_mask"].detach().clone(),
            )
        )
        block = self.action_expert.blocks[0]
        pre = action_tokens
        mixed = pre + video_kv_cache[0]["v"].mean(dim=1, keepdim=True)
        post = mixed + block.ffn(mixed)
        if visual_reader is None:
            return post
        context = ActionLayerReadContext(
            layer_index=0,
            pre_block_hidden=pre,
            modulated_attn_input=pre,
            post_block_hidden=post,
            base_gate_msa=torch.ones(pre.shape[0], 1, pre.shape[-1]),
            timestep_embedding=action_time_embedding,
            proprio=visual_proprio,
            video=LayerVideoKVView(
                key=video_kv_cache[0]["k"],
                value=video_kv_cache[0]["v"],
                current_frame_tokens=1,
            ),
        )
        return post + visual_reader.forward_layer(context, visual_memory).tensor


class _TinyActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_expert = _TinyActionExpert(action_dim=3, hidden=4)
        self.video_expert = _TinyVideoExpert(hidden=4)
        self.mot = _TinyMoT(self.action_expert)
        self.proprio_encoder = nn.Linear(2, 4)
        self.train_action_scheduler = WanContinuousFlowMatchScheduler()

    def _encode_video_latents(self, video: torch.Tensor, *, tiled: bool):
        del tiled
        return video

    def _append_proprio_to_context(self, *, context, context_mask, proprio):
        token = self.proprio_encoder(proprio).unsqueeze(1)
        return (
            torch.cat((context, token), dim=1),
            torch.cat(
                (
                    context_mask,
                    torch.ones(context.shape[0], 1, dtype=torch.bool),
                ),
                dim=1,
            ),
        )

    @staticmethod
    def _build_mot_attention_mask(
        *, video_seq_len: int, action_seq_len: int, device, **kwargs
    ):
        del kwargs
        total = video_seq_len + action_seq_len
        return torch.ones(total, total, dtype=torch.bool, device=device)


def _policy_and_batch():
    torch.manual_seed(7)
    actor = _TinyActor()
    encoder = FrozenDinoV3Encoder._from_preloaded_model_for_tests(
        model=_FakeDino(),
        asset=_asset(),
    )
    input_hash = _hash("official-pil-crop-rgb-orientation-v1")
    memory_hash = native_memory_contract_sha256(
        encoder.asset,
        camera_ids=("main", "wrist"),
        input_contract_sha256=input_hash,
    )
    policy = FastWAMP1DinoBCPolicy(
        actor=actor,
        lora_config=RegimeLoRAConfig(
            rank=2,
            alpha=2.0,
            dropout=0.0,
            target_groups=(ActionLoRATargetGroup.FFN,),
        ),
        visual_encoder=encoder,
        visual_reader=_reader(memory_hash, hidden=4, timestep=4, proprio=2),
        config=FastWAMP1DinoBCConfig(
            action=FastWAMUncondBCConfig(
                action_horizon=4,
                action_dim=3,
                proprio_dim=2,
                expected_video_frames=3,
                expected_video_height=2,
                expected_video_width=2,
                gripper_dimension=2,
                timestep_bins=4,
            ),
            camera_ids=("main", "wrist"),
            camera_input_contract_sha256=input_hash,
        ),
    )
    camera = _camera_batch()
    batch = {
        "video": torch.randn(2, 3, 3, 2, 2),
        "context": torch.randn(2, 2, 4),
        "context_mask": torch.ones(2, 2, dtype=torch.bool),
        "proprio": torch.randn(2, 4, 2),
        "action": torch.randn(2, 4, 3),
        "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
        "p1_camera_pixels": camera.pixels,
        "p1_camera_valid_mask": camera.camera_valid_mask,
    }
    return policy, batch


def test_t1_two_reader_steps_unlock_gradients_and_preserve_bypasses() -> None:
    policy, batch = _policy_and_batch()
    timestep = torch.tensor([200.0, 700.0])
    noise = torch.randn_like(batch["action"])
    noisy = policy.actor.train_action_scheduler.add_noise(
        batch["action"],
        noise,
        timestep,
    )
    calls = policy.visual_encoder.model.calls
    sidecar_off = policy.predict_velocity(
        batch,
        noisy_action=noisy,
        timestep=timestep,
        include_visual=False,
    )
    zero_a3 = policy.predict_velocity(
        batch,
        noisy_action=noisy,
        timestep=timestep,
        include_visual=True,
    )
    assert torch.equal(sidecar_off, zero_a3)
    idm = policy.predict_velocity(
        batch,
        noisy_action=noisy,
        timestep=timestep,
        regime=PolicyRegime.IDM,
        include_visual=False,
    )
    assert idm.shape == sidecar_off.shape
    assert policy.visual_encoder.model.calls == calls + 1

    optimizer = build_p1_optimizer(
        policy,
        lora_learning_rate=1.0e-4,
        reader_learning_rate=1.0e-2,
        train_lora=False,
        weight_decay=0.0,
    )
    router = policy.visual_reader.routers["0"]
    branch = policy.visual_reader.branches["0"][0]

    first = policy(batch, timestep=timestep, noise=noise)
    first["loss_action_bc"].backward()
    assert branch.output_projection.weight.grad is not None
    assert torch.isfinite(branch.output_projection.weight.grad).all()
    assert torch.count_nonzero(branch.output_projection.weight.grad) > 0
    assert all(
        parameter.grad is None for parameter in policy.visual_encoder.parameters()
    )
    optimizer.step()


    optimizer.zero_grad(set_to_none=True)

    second = policy(batch, timestep=timestep, noise=noise)
    second["loss_action_bc"].backward()
    for parameter in (
        router.query_projection.weight,
        branch.semantic_gate.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
    frozen_ids = {
        id(parameter) for parameter in policy.lora_adapter.lora_parameters()
    } | {id(parameter) for parameter in policy.visual_reader.parameters()}
    assert all(
        parameter.grad is None
        for parameter in policy.parameters()
        if id(parameter) not in frozen_ids
    )
    optimizer.step()


def test_zero_probability_modality_dropout_has_exact_prediction_and_loss_parity(
) -> None:
    policy, batch = _policy_and_batch()
    timestep = torch.tensor([200.0, 700.0])
    noise = torch.randn_like(batch["action"])
    baseline_condition = policy.prepare_action_condition(batch)
    baseline = policy.loss_from_prepared_condition(
        batch,
        condition=baseline_condition,
        timestep=timestep,
        noise=noise,
        return_prediction=True,
    )
    keep = sample_modality_keep_mask(
        sample_identities=("left", "right"),
        p_wan=0.0,
        p_dino=0.0,
        seed=42,
        step=0,
        device=policy.device,
    )
    candidate_condition = policy.prepare_action_condition(
        batch,
        modality_keep_mask=keep,
    )
    candidate = policy.loss_from_prepared_condition(
        batch,
        condition=candidate_condition,
        timestep=timestep,
        noise=noise,
        return_prediction=True,
    )

    assert keep is None
    assert torch.equal(baseline["prediction"], candidate["prediction"])
    assert torch.equal(baseline["loss_action_bc"], candidate["loss_action_bc"])


def test_one_chunk_reuses_one_wan_mask_across_repeated_denoising_calls() -> None:
    policy, batch = _policy_and_batch()
    keep = ModalityKeepMask(
        wan=torch.tensor([False, True]),
        dino=torch.ones(2, dtype=torch.bool),
    )
    condition = policy.prepare_action_condition(
        batch,
        modality_keep_mask=keep,
    )
    timestep = torch.tensor([200.0, 700.0])
    noise = torch.randn_like(batch["action"])
    policy.loss_from_prepared_condition(
        batch,
        condition=condition,
        timestep=timestep,
        noise=noise,
    )
    policy.loss_from_prepared_condition(
        batch,
        condition=condition,
        timestep=timestep,
        noise=noise,
    )

    observed = policy.actor.mot.keep_masks
    assert len(observed) == 2
    assert observed[0][1] is None and observed[1][1] is None
    assert torch.equal(observed[0][0], keep.wan)
    assert torch.equal(observed[1][0], keep.wan)


def test_p1_optimizer_groups_are_explicit_and_tiny_stage_freezes_lora() -> None:
    policy, _ = _policy_and_batch()
    optimizer = build_p1_optimizer(
        policy,
        lora_learning_rate=3.0e-4,
        reader_learning_rate=1.0e-3,
        train_lora=False,
    )
    assert [group["name"] for group in optimizer.param_groups] == ["reader"]
    assert all(
        not parameter.requires_grad
        for parameter in policy.lora_adapter.lora_parameters()
    )
    assert all(
        parameter.requires_grad for parameter in policy.visual_reader.parameters()
    )

    baseline, _ = _policy_and_batch()
    baseline_optimizer = build_p1_optimizer(
        baseline,
        lora_learning_rate=3.0e-4,
        reader_learning_rate=1.0e-3,
        train_lora=True,
        train_reader=False,
    )
    assert [group["name"] for group in baseline_optimizer.param_groups] == ["lora"]
    assert all(
        parameter.requires_grad for parameter in baseline.lora_adapter.lora_parameters()
    )
    assert all(
        not parameter.requires_grad for parameter in baseline.visual_reader.parameters()
    )


def test_prepared_condition_reuses_fixed_cache_for_memory_interventions() -> None:
    policy, batch = _policy_and_batch()
    timestep = torch.tensor([200.0, 700.0])
    noise = torch.randn_like(batch["action"])
    condition = policy.prepare_action_condition(batch, include_visual=True)
    calls_after_prepare = policy.visual_encoder.model.calls

    correct = policy.loss_from_prepared_condition(
        batch,
        condition=condition,
        timestep=timestep,
        noise=noise,
        memory_mode="correct",
    )
    repeated = policy.loss_from_prepared_condition(
        batch,
        condition=condition,
        timestep=timestep,
        noise=noise,
        memory_mode="correct",
    )
    zero = policy.loss_from_prepared_condition(
        batch,
        condition=condition,
        timestep=timestep,
        noise=noise,
        memory_mode="zero",
    )
    shuffled = policy.loss_from_prepared_condition(
        batch,
        condition=condition,
        timestep=timestep,
        noise=noise,
        memory_mode="shuffled",
    )
    sidecar_off_condition = policy.prepare_action_condition(
        batch,
        include_visual=False,
    )
    sidecar_off = policy.loss_from_prepared_condition(
        batch,
        condition=sidecar_off_condition,
        timestep=timestep,
        noise=noise,
        memory_mode="off",
    )

    assert torch.equal(correct["loss_action_bc"], repeated["loss_action_bc"])
    assert torch.isfinite(zero["loss_action_bc"])
    assert torch.isfinite(shuffled["loss_action_bc"])
    assert torch.isfinite(sidecar_off["loss_action_bc"])
    assert policy.visual_encoder.model.calls == calls_after_prepare


def test_counterfactual_memory_margin_uses_one_dino_memory_and_shared_flow() -> None:
    policy, batch = _policy_and_batch()
    timestep = torch.tensor([200.0, 700.0])
    noise = torch.randn_like(batch["action"])
    calls_before = policy.visual_encoder.model.calls

    output = policy(
        batch,
        timestep=timestep,
        noise=noise,
        memory_dependency_weight=0.1,
        memory_dependency_relative_margin=0.05,
        memory_dependency_negative_mode="shuffled",
    )

    assert policy.visual_encoder.model.calls == calls_before + 1
    assert torch.equal(
        output["loss_action_bc_negative_memory"],
        output["loss_action_bc"],
    )
    expected_dependency = 0.05 * output["loss_action_bc"].detach()
    torch.testing.assert_close(
        output["loss_memory_dependency"],
        expected_dependency,
    )
    torch.testing.assert_close(
        output["loss_total"],
        output["loss_action_bc"] + 0.1 * expected_dependency,
    )
    output["loss_total"].backward()
    output_projection = policy.visual_reader.branches["0"][0].output_projection
    assert output_projection.weight.grad is not None
    assert torch.count_nonzero(output_projection.weight.grad) > 0

    with pytest.raises(ValueError, match="explicit shared timestep"):
        policy(
            batch,
            memory_dependency_weight=0.1,
            memory_dependency_relative_margin=0.05,
        )


def test_task_paired_and_camera_drop_interventions_are_strict() -> None:
    policy, batch = _policy_and_batch()
    condition = policy.prepare_action_condition(batch, include_visual=True)
    memory = condition.visual.memory
    permutation = torch.tensor([1, 0], dtype=torch.int64)

    paired = _intervene_memory(
        memory,
        "task_paired",
        permutation=permutation,
    )
    assert torch.equal(paired.tokens, memory.tokens[permutation])
    assert paired.memory_contract_sha256 == memory.memory_contract_sha256

    main_dropped = _intervene_memory(memory, "drop_main")
    wrist_dropped = _intervene_memory(memory, "drop_wrist")
    assert torch.count_nonzero(main_dropped.tokens[:, 0]) == 0
    assert not bool(main_dropped.camera_valid_mask[:, 0].any())
    assert torch.count_nonzero(wrist_dropped.tokens[:, 1]) == 0
    assert not bool(wrist_dropped.camera_valid_mask[:, 1].any())

    timestep = torch.tensor([200.0, 700.0])
    noise = torch.randn_like(batch["action"])
    calls_before = policy.visual_encoder.model.calls
    output = policy(
        batch,
        timestep=timestep,
        noise=noise,
        memory_dependency_weight=0.25,
        memory_dependency_relative_margin=0.10,
        memory_dependency_negative_mode="task_paired",
        memory_dependency_permutation=permutation,
    )
    assert torch.isfinite(output["loss_total"])
    assert policy.visual_encoder.model.calls == calls_before + 1

    with pytest.raises(ValueError, match="fixed points"):
        _intervene_memory(
            memory,
            "task_paired",
            permutation=torch.tensor([0, 1]),
        )
    with pytest.raises(ValueError, match="explicit permutation"):
        _intervene_memory(memory, "task_paired")


def test_reader_only_warmup_suppresses_only_lora_optimizer_gradients() -> None:
    policy, _ = _policy_and_batch()
    optimizer = build_p1_optimizer(
        policy,
        lora_learning_rate=3.0e-4,
        reader_learning_rate=2.0e-4,
        train_lora=True,
        train_reader=True,
        weight_decay=0.0,
    )
    families = policy.parameter_families()
    before_lora = [
        parameter.detach().clone() for parameter in families[P1_LORA_PARAMETER_FAMILY]
    ]
    before_reader = [parameter.detach().clone() for parameter in families["reader"]]
    for parameters in families.values():
        for parameter in parameters:
            parameter.grad = torch.ones_like(parameter)

    cleared = _suppress_parameter_family_gradients(
        policy,
        P1_LORA_PARAMETER_FAMILY,
    )

    assert cleared == len(families[P1_LORA_PARAMETER_FAMILY])
    assert all(
        parameter.grad is None for parameter in families[P1_LORA_PARAMETER_FAMILY]
    )
    assert all(parameter.grad is not None for parameter in families["reader"])
    optimizer.step()
    assert all(
        torch.equal(before, parameter)
        for before, parameter in zip(
            before_lora,
            families[P1_LORA_PARAMETER_FAMILY],
            strict=True,
        )
    )
    assert all(
        parameter not in optimizer.state
        for parameter in families[P1_LORA_PARAMETER_FAMILY]
    )
    assert any(
        not torch.equal(before, parameter)
        for before, parameter in zip(
            before_reader,
            families["reader"],
            strict=True,
        )
    )


def test_position_mode_fails_closed_before_coordinate_score_exists() -> None:
    action = FastWAMUncondBCConfig()
    try:
        FastWAMP1DinoBCConfig(
            action=action,
            camera_ids=("main", "wrist"),
            camera_input_contract_sha256=contract_sha256({"crop": "official"}),
            position_mode="coordinate_score",
        )
    except ValueError as error:
        assert "forbids a coordinate-score branch" in str(error)
    else:
        raise AssertionError("Coordinate-score must remain unavailable before T2.")


def test_v2_policy_runs_minimal_reader_lora_updates_and_freezes_encoder() -> None:
    from fastwam.models.wan22.visual_backbone import (
        FrozenVisualPatchEncoder,
        VisualBackboneAssetSpec,
        get_visual_backbone_preset,
        visual_output_contract,
        visual_preprocess_contract,
    )
    from fastwam.models.wan22.visual_sidecar import (
        VisualPatchReaderConfig,
        build_visual_patch_reader,
    )
    from fastwam.p1_dino_bc import FastWAMP1VisualBCPolicy

    preset = get_visual_backbone_preset("dinov3", "vits16", input_size=224)
    asset = VisualBackboneAssetSpec(
        family="dinov3",
        variant="vits16",
        input_size=224,
        source_root="/unit-test/dinov3",
        source_revision=preset.source_revision,
        weights_revision=preset.weights_revision,
        weights_path="/unit-test/model.safetensors",
        weights_sha256=preset.weights_sha256,
        preprocess_sha256=contract_sha256(
            visual_preprocess_contract(preset, input_size=224)
        ),
        output_contract_sha256=contract_sha256(
            visual_output_contract(preset, input_size=224)
        ),
        compute_dtype="float32",
        encode_microbatch_size=2,
        license_id=preset.license_id,
    )

    class FakeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.ones(()))

    class FakeEncoder(FrozenVisualPatchEncoder):
        def _forward_patch_tokens(self, normalized: torch.Tensor) -> torch.Tensor:
            basis = torch.linspace(
                0.25,
                1.25,
                self.asset.preset.native_dim,
                device=normalized.device,
                dtype=normalized.dtype,
            )
            signal = normalized.mean(dim=(1, 2, 3))[:, None, None]
            return (
                basis[None, None].expand(
                    normalized.shape[0],
                    self.asset.patch_count,
                    -1,
                )
                + signal
            )

    encoder = FakeEncoder(model=FakeModel(), asset=asset)
    input_hash = _hash("v2-raw-camera-contract")
    memory_hash = encoder.memory_contract_sha256(
        camera_ids=("main", "wrist"),
        input_contract_sha256=input_hash,
    )
    reader = build_visual_patch_reader(
        VisualPatchReaderConfig(
            action_hidden_dim=4,
            timestep_dim=4,
            proprio_dim=2,
            memory_dim=384,
            camera_ids=("main", "wrist"),
            layer_indices=(0,),
            temperature=0.07,
            residual_scale=1.0,
            query_projection=ProjectionSpec(kind="full_linear", rank=None),
            output_projection=ProjectionSpec(kind="full_linear", rank=None),
            memory_contract_sha256=memory_hash,
        )
    )
    policy = FastWAMP1VisualBCPolicy(
        actor=_TinyActor(),
        lora_config=RegimeLoRAConfig(
            rank=2,
            alpha=2.0,
            dropout=0.0,
            target_groups=(ActionLoRATargetGroup.FFN,),
        ),
        visual_encoder=encoder,
        visual_reader=reader,
        config=FastWAMP1DinoBCConfig(
            action=FastWAMUncondBCConfig(
                action_horizon=4,
                action_dim=3,
                proprio_dim=2,
                expected_video_frames=3,
                expected_video_height=2,
                expected_video_width=2,
                gripper_dimension=2,
                timestep_bins=4,
            ),
            camera_ids=("main", "wrist"),
            camera_input_contract_sha256=input_hash,
        ),
    )
    batch = {
        "video": torch.randn(2, 3, 3, 2, 2),
        "context": torch.randn(2, 2, 4),
        "context_mask": torch.ones(2, 2, dtype=torch.bool),
        "proprio": torch.randn(2, 4, 2),
        "action": torch.randn(2, 4, 3),
        "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
        "visual_camera_pixels": torch.randint(
            0,
            256,
            (2, 2, 3, 224, 224),
            dtype=torch.uint8,
        ),
        "visual_camera_valid_mask": torch.ones(2, 2, dtype=torch.bool),
        "visual_camera_source_resolution": torch.full(
            (2, 2, 2),
            512,
            dtype=torch.int32,
        ),
    }
    optimizer = build_p1_optimizer(
        policy,
        lora_learning_rate=1.0e-3,
        reader_learning_rate=1.0e-2,
        train_lora=True,
        weight_decay=0.0,
    )
    timestep = torch.tensor([200.0, 700.0])
    noise = torch.randn_like(batch["action"])
    first = policy(batch, timestep=timestep, noise=noise)
    first["loss_total"].backward()
    assert (
        torch.count_nonzero(reader.branches["0"][0].output_projection.weight.grad) > 0
    )
    assert all(parameter.grad is None for parameter in encoder.parameters())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    second = policy(batch, timestep=timestep, noise=noise)
    second["loss_total"].backward()
    assert torch.count_nonzero(reader.routers["0"].query_projection.weight.grad) > 0
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in policy.lora_adapter.lora_parameters()
    )
    assert all(parameter.grad is None for parameter in encoder.parameters())


def test_p1_checkpoint_round_trip_is_lora_reader_only(tmp_path: Path) -> None:
    policy, _ = _policy_and_batch()
    contract = {
        "resolved_config_sha256": _hash("resolved-p1-config"),
        "layout": policy.p1_config.layout_contract,
        "reader_contract_sha256": policy.visual_reader.reader_contract_sha256,
    }
    target = tmp_path / "reader_lora_checkpoint.pt"
    save_p1_dino_bc_checkpoint(
        target,
        adapter=policy.lora_adapter,
        reader=policy.visual_reader,
        global_step=2,
        stage="t1_smoke",
        arm="a3_joint",
        parent_checkpoint_sha256=_hash("fastwam-parent"),
        dinov3_weights_sha256=policy.visual_encoder.asset.weights_sha256,
        memory_contract_sha256=policy.expected_memory_contract,
        contract=contract,
        provenance={"dataset_manifest_sha256": _hash("dataset-manifest")},
        trainer_state={
            "last_loss_action_bc": 1.25,
            "best_dev_loss_action_bc": None,
            "nonzero_update_count": 2,
        },
    )
    report = inspect_p1_dino_bc_checkpoint(target)
    assert report["result"] == "PASS"
    assert report["lora_tensor_count"] > 0
    assert report["reader_tensor_count"] > 0
    assert not report["contains_frozen_fastwam_tensors"]
    assert not report["contains_dinov3_tensors"]

    restored, _ = _policy_and_batch()
    payload = load_p1_dino_bc_checkpoint(
        target,
        adapter=restored.lora_adapter,
        reader=restored.visual_reader,
        expected_parent_checkpoint_sha256=_hash("fastwam-parent"),
        expected_dinov3_weights_sha256=policy.visual_encoder.asset.weights_sha256,
        expected_contract=contract,
    )
    assert payload["global_step"] == 2
    for first, second in zip(
        policy.visual_reader.parameters(),
        restored.visual_reader.parameters(),
        strict=True,
    ):
        assert torch.equal(first, second)


def test_p1_checkpoint_inspector_rejects_forbidden_payload(tmp_path: Path) -> None:
    policy, _ = _policy_and_batch()
    target = tmp_path / "valid.pt"
    contract = {"resolved_config_sha256": _hash("config")}
    save_p1_dino_bc_checkpoint(
        target,
        adapter=policy.lora_adapter,
        reader=policy.visual_reader,
        global_step=0,
        stage="t1_smoke",
        arm="a3_joint",
        parent_checkpoint_sha256=_hash("parent"),
        dinov3_weights_sha256=policy.visual_encoder.asset.weights_sha256,
        memory_contract_sha256=policy.expected_memory_contract,
        contract=contract,
        provenance={"dataset_manifest_sha256": _hash("data")},
        trainer_state={
            "last_loss_action_bc": None,
            "best_dev_loss_action_bc": None,
            "nonzero_update_count": 0,
        },
    )
    payload = torch.load(target, map_location="cpu", weights_only=False)
    payload["provenance"]["observation"] = torch.zeros(1)
    forbidden = tmp_path / "forbidden.pt"
    torch.save(payload, forbidden)
    with pytest.raises(ValueError, match="outside LoRA/reader state"):
        inspect_p1_dino_bc_checkpoint(forbidden)


def test_p1_full_checkpoint_round_trip_excludes_frozen_models(tmp_path: Path) -> None:
    policy, _ = _policy_and_batch()
    optimizer = build_p1_optimizer(
        policy,
        lora_learning_rate=3.0e-4,
        reader_learning_rate=1.0e-4,
        train_lora=True,
        train_reader=True,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    contract = {"resolved_config_sha256": _hash("p1-full-config")}
    target = tmp_path / "p1_full_step_000002.pt"
    save_p1_dino_bc_full_checkpoint(
        target,
        adapter=policy.lora_adapter,
        reader=policy.visual_reader,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        grad_scaler=scaler,
        global_step=2,
        epoch=0,
        sampler_offset=8,
        rng_by_rank=[capture_rng_state()],
        parent_checkpoint_sha256=_hash("parent"),
        dinov3_weights_sha256=policy.visual_encoder.asset.weights_sha256,
        memory_contract_sha256=policy.expected_memory_contract,
        contract=contract,
        provenance={"dataset_manifest_sha256": _hash("dataset")},
        trainer_state={
            "best_validation_loss_action_bc": None,
            "best_step": None,
            "epochs_without_improvement": 0,
            "nonzero_update_count": 2,
        },
    )
    report = inspect_p1_dino_bc_full_checkpoint(target)
    assert report["result"] == "PASS"
    assert report["adapter_rank"] == 2
    assert report["reader_tensor_count"] > 0
    assert not report["contains_frozen_fastwam_tensors"]
    assert not report["contains_dinov3_tensors"]

    restored, _ = _policy_and_batch()
    restored_optimizer = build_p1_optimizer(
        restored,
        lora_learning_rate=3.0e-4,
        reader_learning_rate=1.0e-4,
        train_lora=True,
        train_reader=True,
    )
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer,
        lambda _: 1.0,
    )
    restored_scaler = torch.amp.GradScaler("cpu", enabled=False)
    payload = load_p1_dino_bc_full_checkpoint(
        target,
        adapter=restored.lora_adapter,
        reader=restored.visual_reader,
        optimizer=restored_optimizer,
        lr_scheduler=restored_scheduler,
        grad_scaler=restored_scaler,
        expected_parent_checkpoint_sha256=_hash("parent"),
        expected_dinov3_weights_sha256=policy.visual_encoder.asset.weights_sha256,
        expected_contract=contract,
    )
    assert payload["global_step"] == 2


def test_p1_v2_full_checkpoint_round_trip_restores_control_state(
    tmp_path: Path,
) -> None:
    policy, _ = _policy_and_batch()
    optimizer = build_p1_optimizer(
        policy,
        lora_learning_rate=3.0e-4,
        reader_learning_rate=2.0e-4,
        train_lora=True,
        train_reader=True,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    sampler = TaskPairedDistributedBatchSampler(
        task_keys=[(0, 0)] * 32,
        episode_indices=[index % 4 for index in range(32)],
        batch_size=32,
        rank=0,
        world_size=1,
        seed=42,
    )
    warmup = DependencyWarmupController()
    cycle = NegativeModeCycle()
    selector = CausalCheckpointSelector(
        CausalSelectionThresholds(
            validation_loss_max=0.05,
            pose_mse_max=0.04,
            gripper_mse_max=0.09,
        )
    )
    v2_state = {
        "profile": "dino_contribution_v2",
        "warmup": warmup.state_dict(),
        "negative_cycle": cycle.state_dict(),
        "task_paired_sampler_by_rank": [sampler.state_dict()],
        "causal_selector": selector.state_dict(),
    }
    contract = {"resolved_config_sha256": _hash("p1-v2-full-config")}
    target = tmp_path / "p1_v2_step_000000.pt"
    save_p1_dino_bc_full_checkpoint_v2(
        target,
        adapter=policy.lora_adapter,
        reader=policy.visual_reader,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        grad_scaler=scaler,
        global_step=0,
        epoch=0,
        sampler_offset=0,
        rng_by_rank=[capture_rng_state()],
        parent_checkpoint_sha256=_hash("parent"),
        dinov3_weights_sha256=policy.visual_encoder.asset.weights_sha256,
        memory_contract_sha256=policy.expected_memory_contract,
        contract=contract,
        provenance={"lineage": "dino_contribution_v2"},
        trainer_state={
            "best_validation_loss_action_bc": None,
            "best_step": None,
            "epochs_without_improvement": 0,
            "nonzero_update_count": 0,
        },
        v2_state=v2_state,
    )
    report = inspect_p1_dino_bc_full_checkpoint_v2(target)
    assert report["result"] == "PASS"
    assert report["phase"] == "reader_warmup"
    assert report["sampler_rank_count"] == 1

    restored, _ = _policy_and_batch()
    restored_optimizer = build_p1_optimizer(
        restored,
        lora_learning_rate=3.0e-4,
        reader_learning_rate=2.0e-4,
        train_lora=True,
        train_reader=True,
    )
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer,
        lambda _: 1.0,
    )
    restored_scaler = torch.amp.GradScaler("cpu", enabled=False)
    payload = load_p1_dino_bc_full_checkpoint_v2(
        target,
        adapter=restored.lora_adapter,
        reader=restored.visual_reader,
        optimizer=restored_optimizer,
        lr_scheduler=restored_scheduler,
        grad_scaler=restored_scaler,
        expected_parent_checkpoint_sha256=_hash("parent"),
        expected_dinov3_weights_sha256=policy.visual_encoder.asset.weights_sha256,
        expected_contract=contract,
    )
    assert payload["v2_state"] == v2_state
    sampler.validate_state_dict(payload["v2_state"]["task_paired_sampler_by_rank"][0])


def test_p1_full_rank16_and_rank32_configs_preserve_matched_contract() -> None:
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        rank16 = compose(config_name="p1_dino_bc_full")
        rank32 = compose(
            config_name="p1_dino_bc_full",
            overrides=["lora.rank=32", "lora.alpha=32"],
        )
    _validate_full_config(rank16, world_size=4)
    _validate_full_config(rank32, world_size=4)
    assert rank16.training.microbatch_size == 32
    assert rank16.training.gradient_accumulation_steps == 1
    assert rank16.data.num_workers == 3
    assert rank16.data.prefetch_factor == 1
    assert list(rank16.training.early_checkpoint_steps) == [10]
    assert rank16.training.save_every_steps == 100
    assert rank16.training.global_batch_size == rank32.training.global_batch_size
    assert rank16.optimizer == rank32.optimizer


def test_p1_dino_contribution_profile_is_separate_and_strict() -> None:
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="p1_dino_bc_dino_contribution")
    _validate_full_config(cfg, world_size=4)

    settings = _memory_dependency_settings(cfg)
    assert cfg.training.dino_contribution_profile == DINO_CONTRIBUTION_PROFILE
    assert cfg.lora.rank == 32
    assert cfg.p1.reader.semantic_gate_floor == 0.05
    assert cfg.p1.reader.semantic_gate_temperature == 1.25
    assert cfg.training.reader_only_warmup_steps == 256
    assert settings == {
        "enabled": True,
        "negative_mode": "shuffled",
        "weight": 0.1,
        "relative_margin": 0.05,
        "every_n_steps": 4,
    }
    assert _memory_dependency_active(global_step=0, settings=settings)
    assert not _memory_dependency_active(global_step=1, settings=settings)
    assert _memory_dependency_active(global_step=4, settings=settings)

    cfg.p1.reader.semantic_gate_floor = 0.0
    with pytest.raises(ValueError, match="semantic-gate controls"):
        _validate_full_config(cfg, world_size=4)


def test_p1_full_formal_acceptance_allows_preregistered_early_stop() -> None:
    checkpoint = {"result": "PASS"}
    common = {
        "global_step": 5826,
        "stop_after_steps": 19420,
        "nonzero_update_count": 5826,
        "last_checkpoint": checkpoint,
        "best_step": 3884,
        "controlled_stop": False,
    }
    assert _run_acceptance_passed(**common, early_stopped=True)
    assert not _run_acceptance_passed(**common, early_stopped=False)
    assert not _run_acceptance_passed(
        **(common | {"controlled_stop": True}),
        early_stopped=True,
    )
