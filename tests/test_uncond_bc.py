import copy

import pytest
import torch
from torch import nn

from fastwam.adapters import (
    ActionLoRATargetGroup,
    PolicyRegime,
    RegimeLoRAConfig,
)
from fastwam.models.wan22.adaptive_action import CachedActionVelocity
from fastwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)
from fastwam.uncond_bc import (
    FastWAMUncondBCConfig,
    FastWAMUncondBCPolicy,
    compute_action_flow_matching_bc_loss,
    stateless_validation_flow_inputs,
)


class _WeightedScheduler:
    num_train_timesteps = 1000

    @staticmethod
    def training_weight(timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return torch.tensor([1.0, 2.0])


class _TinyBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )


class _TinyActionExpert(nn.Module):
    def __init__(self, *, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.input = nn.Linear(action_dim, hidden_dim)
        self.blocks = nn.ModuleList([_TinyBlock(hidden_dim), _TinyBlock(hidden_dim)])
        self.output = nn.Linear(hidden_dim, action_dim)

    def pre_dit(
        self,
        *,
        action_tokens: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> dict:
        del timestep
        return {
            "tokens": self.input(action_tokens),
            "freqs": torch.empty(0, device=action_tokens.device),
            "t_mod": torch.empty(0, device=action_tokens.device),
            "context": context,
            "context_mask": context_mask,
        }

    def post_dit(self, tokens: torch.Tensor, payload: dict) -> torch.Tensor:
        del payload
        return self.output(tokens)


class _TinyVideoExpert(nn.Module):
    fuse_vae_embedding_in_latents = True

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.arange(1, hidden_dim + 1).float())

    def pre_dit(
        self,
        *,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        action,
        fuse_vae_embedding_in_latents: bool,
    ) -> dict:
        del timestep, action, fuse_vae_embedding_in_latents
        pooled = x.float().mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(1)
        tokens = pooled.unsqueeze(2) * self.scale.view(1, 1, -1)
        return {
            "tokens": tokens,
            "freqs": torch.empty(0, device=x.device),
            "t_mod": torch.empty(0, device=x.device),
            "context": context,
            "context_mask": context_mask,
            "meta": {"tokens_per_frame": 1},
        }

    def post_dit(self, *args, **kwargs):
        raise AssertionError("future-video post-DiT must not run in action-only BC")


class _TinyMoT(nn.Module):
    def __init__(self, action_expert: _TinyActionExpert) -> None:
        super().__init__()
        self.action_expert = action_expert
        self.num_layers = len(action_expert.blocks)
        self.prefill_calls = 0

    def prefill_video_cache(
        self,
        *,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: dict,
        video_attention_mask: torch.Tensor,
        gate_current_frame_video_tokens: int,
    ) -> list[dict]:
        del (
            video_freqs,
            video_t_mod,
            video_context_payload,
            video_attention_mask,
            gate_current_frame_video_tokens,
        )
        self.prefill_calls += 1
        return [
            {"k": video_tokens.detach().clone(), "v": video_tokens.detach().clone()}
            for _ in range(self.num_layers)
        ]

    def forward_action_with_video_cache(
        self,
        *,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: dict,
        video_kv_cache: list[dict],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        kv_tap,
        checkpoint_context_fn,
    ) -> torch.Tensor:
        del (
            action_freqs,
            action_t_mod,
            action_context_payload,
            attention_mask,
            video_seq_len,
            kv_tap,
            checkpoint_context_fn,
        )
        x = action_tokens + video_kv_cache[0]["v"].mean(dim=1, keepdim=True)
        for block in self.action_expert.blocks:
            x = x + block.ffn(x)
        return x


class _TinyActor(nn.Module):
    def __init__(self, *, action_dim: int = 3, hidden_dim: int = 4) -> None:
        super().__init__()
        self.action_expert = _TinyActionExpert(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        )
        self.video_expert = _TinyVideoExpert(hidden_dim)
        self.mot = _TinyMoT(self.action_expert)
        self.proprio_encoder = nn.Linear(2, hidden_dim)
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=1000,
            shift=5.0,
        )
        self.encoded_shapes: list[tuple[int, ...]] = []
        self.online_encoded_shapes: list[tuple[int, ...]] = []

    def _encode_video_latents(self, video: torch.Tensor, *, tiled: bool):
        del tiled
        self.encoded_shapes.append(tuple(video.shape))
        return video

    def _encode_input_image_latents_tensor(
        self,
        image: torch.Tensor,
        *,
        tiled: bool,
    ) -> torch.Tensor:
        del tiled
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("RLinf reference image must be one [1,3,H,W] sample.")
        self.online_encoded_shapes.append(tuple(image.shape))
        return image.unsqueeze(2)

    def _append_proprio_to_context(
        self,
        *,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token = self.proprio_encoder(proprio).unsqueeze(1)
        mask = torch.ones(
            (context.shape[0], 1),
            dtype=torch.bool,
            device=context.device,
        )
        return torch.cat([context, token], dim=1), torch.cat(
            [context_mask, mask], dim=1
        )

    @staticmethod
    def _build_mot_attention_mask(
        *,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        del video_tokens_per_frame
        size = video_seq_len + action_seq_len
        return torch.ones((size, size), dtype=torch.bool, device=device)

    def _video_denoise_step_compiled(self, *args, **kwargs):
        raise AssertionError("future-video denoise must not run in action-only BC")

    def training_loss_from_inputs(self, *args, **kwargs):
        raise AssertionError("joint prediction loss must not run in action-only BC")


def _policy_and_batch(batch_size: int = 2):
    torch.manual_seed(7)
    actor = _TinyActor()
    policy = FastWAMUncondBCPolicy(
        actor=actor,
        lora_config=RegimeLoRAConfig(
            rank=2,
            alpha=2.0,
            dropout=0.0,
            target_groups=(ActionLoRATargetGroup.FFN,),
        ),
        config=FastWAMUncondBCConfig(
            action_horizon=4,
            action_dim=3,
            proprio_dim=2,
            expected_video_frames=3,
            expected_video_height=2,
            expected_video_width=2,
            gripper_dimension=2,
            timestep_bins=4,
        ),
    )
    batch = {
        "video": torch.randn(batch_size, 3, 3, 2, 2),
        "context": torch.randn(batch_size, 2, 4),
        "context_mask": torch.ones(batch_size, 2, dtype=torch.bool),
        "proprio": torch.randn(batch_size, 4, 2),
        "action": torch.randn(batch_size, 4, 3),
        "action_is_pad": torch.zeros(batch_size, 4, dtype=torch.bool),
    }
    return policy, batch


def test_action_flow_matching_loss_matches_manual_padding_reduction() -> None:
    prediction = torch.zeros(2, 2, 3)
    target = torch.tensor(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[2.0, 0.0, 1.0], [9.0, 9.0, 9.0]],
        ]
    )
    is_pad = torch.tensor([[False, False], [False, True]])

    result = compute_action_flow_matching_bc_loss(
        prediction=prediction,
        target=target,
        timestep=torch.tensor([100.0, 900.0]),
        action_is_pad=is_pad,
        scheduler=_WeightedScheduler(),
        gripper_dimension=2,
        timestep_bins=2,
    )

    first = torch.tensor([(1 + 4 + 9) / 3, (16 + 25 + 36) / 3]).mean()
    second = torch.tensor((4 + 0 + 1) / 3)
    assert torch.allclose(result.loss_action_bc, (first + 2 * second) / 2)
    assert torch.allclose(
        result.mse_per_dimension,
        torch.tensor([(1 + 16 + 4) / 3, (4 + 25 + 0) / 3, (9 + 36 + 1) / 3]),
    )
    assert torch.allclose(result.mse_pose, result.mse_per_dimension[:2].mean())
    assert torch.equal(result.mse_gripper, result.mse_per_dimension[2])
    assert result.valid_action_count.item() == 3
    assert torch.equal(result.timestep_bin_count, torch.tensor([1, 1]))


def test_stateless_validation_noise_is_order_and_batch_independent() -> None:
    scheduler = WanContinuousFlowMatchScheduler()
    identities = ["suite:episode:3", "suite:episode:9", "other:4"]
    timestep, noise = stateless_validation_flow_inputs(
        sample_identities=identities,
        action_shape=(3, 4, 3),
        scheduler=scheduler,
        seed=42,
        device="cpu",
        dtype=torch.float32,
    )
    reordered_timestep, reordered_noise = stateless_validation_flow_inputs(
        sample_identities=[identities[2], identities[0]],
        action_shape=(2, 4, 3),
        scheduler=scheduler,
        seed=42,
        device="cpu",
        dtype=torch.float32,
    )

    assert torch.equal(reordered_timestep[0], timestep[2])
    assert torch.equal(reordered_noise[0], noise[2])
    assert torch.equal(reordered_timestep[1], timestep[0])
    assert torch.equal(reordered_noise[1], noise[0])


def test_future_frames_cannot_change_bc_loss_or_gradients() -> None:
    policy, batch = _policy_and_batch()
    altered = copy.deepcopy(batch)
    altered["video"][:, :, 1:] = torch.randn_like(altered["video"][:, :, 1:]) * 100
    timestep = torch.tensor([250.0, 750.0])
    noise = torch.randn_like(batch["action"])

    first = policy(batch, timestep=timestep, noise=noise)
    first["loss_action_bc"].backward()
    first_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in policy.lora_adapter.named_lora_parameters()
    }
    policy.zero_grad(set_to_none=True)
    second = policy(altered, timestep=timestep, noise=noise)
    second["loss_action_bc"].backward()

    assert torch.equal(first["loss_action_bc"], second["loss_action_bc"])
    assert policy.actor.encoded_shapes == [(2, 3, 1, 2, 2), (2, 3, 1, 2, 2)]
    for name, parameter in policy.lora_adapter.named_lora_parameters():
        assert torch.equal(first_gradients[name], parameter.grad)


def test_bc_forward_has_no_video_loss_and_only_lora_gradients() -> None:
    policy, batch = _policy_and_batch()
    output = policy(
        batch,
        timestep=torch.tensor([100.0, 800.0]),
        noise=torch.randn_like(batch["action"]),
    )
    output["loss_action_bc"].backward()

    assert set(output) == {
        "loss_action_bc",
        "mse_per_dimension",
        "mse_pose",
        "mse_gripper",
        "mse_by_timestep_bin",
        "timestep_bin_count",
        "valid_action_count",
    }
    assert not any("video" in name and "loss" in name for name in output)
    lora_ids = {id(parameter) for parameter in policy.lora_adapter.lora_parameters()}
    assert all(
        parameter.grad is None
        for parameter in policy.actor.parameters()
        if id(parameter) not in lora_ids
    )
    assert all(
        parameter.grad is not None
        for parameter in policy.lora_adapter.lora_parameters()
    )
    assert any(
        torch.count_nonzero(parameter.grad).item() > 0
        for name, parameter in policy.lora_adapter.named_lora_parameters()
        if name.endswith("lora_B")
    )


def _rlinf_uncond_condition_reference(policy, batch, index):
    """Mirror LiberoFastWAMRuntime._prepare_action_condition for UNCOND."""

    actor = policy.actor
    image = batch["video"][index : index + 1, :, 0]
    first_frame = actor._encode_input_image_latents_tensor(image, tiled=False)
    context = batch["context"][index : index + 1]
    context_mask = batch["context_mask"][index : index + 1]
    context, context_mask = actor._append_proprio_to_context(
        context=context,
        context_mask=context_mask,
        proprio=batch["proprio"][index : index + 1, 0],
    )
    video_pre = actor.video_expert.pre_dit(
        x=first_frame,
        timestep=torch.zeros(1, dtype=first_frame.dtype),
        context=context,
        context_mask=context_mask,
        action=None,
        fuse_vae_embedding_in_latents=True,
    )
    video_seq_len = int(video_pre["tokens"].shape[1])
    tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
    attention_mask = actor._build_mot_attention_mask(
        video_seq_len=video_seq_len,
        action_seq_len=policy.config.action_horizon,
        video_tokens_per_frame=tokens_per_frame,
        device=video_pre["tokens"].device,
    )
    video_cache = actor.mot.prefill_video_cache(
        video_tokens=video_pre["tokens"],
        video_freqs=video_pre["freqs"],
        video_t_mod=video_pre["t_mod"],
        video_context_payload={
            "context": video_pre["context"],
            "mask": video_pre["context_mask"],
        },
        video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        gate_current_frame_video_tokens=tokens_per_frame,
    )
    return {
        "context": context,
        "context_mask": context_mask,
        "video_kv_cache": video_cache,
        "attention_mask": attention_mask,
        "video_seq_len": video_seq_len,
        "current_frame_video_tokens": tokens_per_frame,
    }


def test_batched_current_condition_matches_rlinf_uncond_reference() -> None:
    policy, batch = _policy_and_batch()
    batched = policy.prepare_action_condition(batch)
    individual = []
    for index in range(2):
        individual.append(_rlinf_uncond_condition_reference(policy, batch, index))

    torch.testing.assert_close(
        batched.context,
        torch.cat([condition["context"] for condition in individual]),
        rtol=0,
        atol=1e-7,
    )
    assert torch.equal(
        batched.context_mask,
        torch.cat([condition["context_mask"] for condition in individual]),
    )
    for layer_index in range(len(batched.video_kv_cache)):
        for key in ("k", "v"):
            torch.testing.assert_close(
                batched.video_kv_cache[layer_index][key],
                torch.cat(
                    [
                        condition["video_kv_cache"][layer_index][key]
                        for condition in individual
                    ]
                ),
                rtol=0,
                atol=1e-7,
            )
    assert policy.actor.online_encoded_shapes == [(1, 3, 2, 2)] * 2
    assert all(
        condition["video_seq_len"] == batched.video_seq_len
        and condition["current_frame_video_tokens"]
        == batched.current_frame_video_tokens
        and torch.equal(condition["attention_mask"], batched.attention_mask)
        for condition in individual
    )


def test_zero_lora_matches_idm_base_and_idm_stays_exact_after_bc_update() -> None:
    policy, batch = _policy_and_batch()
    condition = policy.prepare_action_condition(batch)
    action = batch["action"]
    timestep = torch.tensor([300.0, 600.0])

    def velocity(regime: PolicyRegime) -> torch.Tensor:
        return CachedActionVelocity(
            action_expert=policy.actor.action_expert,
            mot=policy.actor.mot,
            condition=condition,
            regime=regime,
            regime_context=policy.lora_adapter.regime_context,
        )(action, timestep).velocity

    idm_before = velocity(PolicyRegime.IDM)
    uncond_zero = velocity(PolicyRegime.UNCOND)
    assert torch.equal(idm_before, uncond_zero)

    optimizer = torch.optim.AdamW(policy.lora_adapter.lora_parameters(), lr=0.1)
    loss = policy(batch, timestep=timestep, noise=torch.randn_like(action))[
        "loss_action_bc"
    ]
    loss.backward()
    optimizer.step()
    idm_after = velocity(PolicyRegime.IDM)
    uncond_after = velocity(PolicyRegime.UNCOND)

    assert torch.equal(idm_before, idm_after)
    assert not torch.equal(uncond_after, idm_after)


def test_batch_contract_rejects_dtype_padding_and_context_shape_mismatch() -> None:
    policy, batch = _policy_and_batch()

    bad_video = copy.deepcopy(batch)
    bad_video["video"] = torch.zeros_like(batch["video"], dtype=torch.uint8)
    with pytest.raises(TypeError, match="video tensor must be floating"):
        policy.prepare_action_condition(bad_video)

    bad_padding = copy.deepcopy(batch)
    bad_padding["action_is_pad"] = torch.zeros(2, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="action_is_pad"):
        policy.prepare_action_condition(bad_padding)

    bad_context = copy.deepcopy(batch)
    bad_context["context_mask"] = torch.ones(2, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="context/mask"):
        policy.prepare_action_condition(bad_context)
