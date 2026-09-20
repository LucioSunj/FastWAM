"""CPU coverage of both rank-128 BC graphs, DDP and resumable artifacts."""

from __future__ import annotations

import copy
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from fastwam.adapters import PolicyRegime, RegimeLoRAConfig, RegimeLoRALinear
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)
from fastwam.models.wan22.wan_video_dit import DiTBlock
from fastwam.uncond_bc import FastWAMUncondBCConfig, FastWAMUncondBCPolicy
from fastwam.uncond_bc_checkpoint import (
    DUAL_UNCOND_BC_TRAINING_SCHEMA,
    UNCOND_BC_TRAINING_SCHEMA,
    capture_rng_state,
    compare_uncond_bc_checkpoints,
    inspect_uncond_bc_checkpoint,
    load_uncond_bc_adapter_checkpoint,
    load_uncond_bc_checkpoint,
    load_uncond_bc_sidecar,
    restore_rng_state,
    save_uncond_bc_checkpoint,
    save_uncond_bc_sidecar,
)
from fastwam.uncond_bc_trainer import (
    _bc0_parity_and_action_report,
    _frozen_versions,
    _lora_update_norms,
    _snapshot_lora,
    _strict_reload_best_sidecar,
    _validate_training_config,
)


class _Expert(nn.Module):
    def __init__(self, *, video: bool, checkpoint: bool) -> None:
        super().__init__()
        self.video = video
        self.hidden_dim = 8
        self.num_heads = 2
        self.attn_head_dim = 4
        self.action_dim = 7
        self.use_gradient_checkpointing = checkpoint
        self.fuse_vae_embedding_in_latents = False
        self.input = nn.Linear(3 if video else 7, 8)
        self.output = nn.Linear(8, 7)
        self.blocks = nn.ModuleList(
            DiTBlock(hidden_dim=8, attn_head_dim=4, num_heads=2, ffn_dim=16)
            for _ in range(2)
        )

    def pre_dit(self, *, context, context_mask, timestep, **kwargs):
        del timestep
        if self.video:
            pixels = kwargs["x"][:, :, 0].flatten(2).transpose(1, 2)
            tokens = self.input(pixels)
        else:
            tokens = self.input(kwargs["action_tokens"])
        return {
            "tokens": tokens,
            "freqs": torch.ones(tokens.shape[1], 1, 2, dtype=torch.complex128),
            "t_mod": torch.zeros(tokens.shape[0], 6, 8),
            "context": context,
            "context_mask": context_mask[:, None, :].expand(-1, tokens.shape[1], -1),
            "meta": {"tokens_per_frame": tokens.shape[1]},
        }

    def post_dit(self, tokens, _pre):
        if self.video:
            raise AssertionError("Action BC must not call Video post_dit.")
        return self.output(tokens)


class _Actor(nn.Module):
    def __init__(self, *, checkpoint: bool) -> None:
        super().__init__()
        self.video_expert = _Expert(video=True, checkpoint=checkpoint)
        self.action_expert = _Expert(video=False, checkpoint=checkpoint)
        self.proprio_encoder = nn.Linear(8, 8)
        self.mot = MoT(
            mixtures={"video": self.video_expert, "action": self.action_expert},
            mot_checkpoint_mixed_attn=checkpoint,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=1000, shift=5.0
        )

    def _encode_video_latents(self, video, *, tiled):
        del tiled
        assert video.shape[2] == 1
        return video

    def _append_proprio_to_context(self, *, context, context_mask, proprio):
        return (
            torch.cat([context, self.proprio_encoder(proprio)[:, None]], dim=1),
            torch.cat(
                [context_mask, torch.ones(context.shape[0], 1, dtype=torch.bool)], dim=1
            ),
        )

    def _build_mot_attention_mask(self, *, video_seq_len, action_seq_len, **kwargs):
        del kwargs
        mask = torch.ones(
            video_seq_len + action_seq_len,
            video_seq_len + action_seq_len,
            dtype=torch.bool,
        )
        mask[:video_seq_len, video_seq_len:] = False
        return mask


def _policy(*, dual: bool, checkpoint: bool = True) -> FastWAMUncondBCPolicy:
    torch.manual_seed(42)
    return FastWAMUncondBCPolicy(
        actor=_Actor(checkpoint=checkpoint),
        lora_config=RegimeLoRAConfig(rank=128, alpha=128),
        video_lora_config=RegimeLoRAConfig(rank=128, alpha=128) if dual else None,
        config=FastWAMUncondBCConfig(
            action_horizon=3,
            expected_video_frames=3,
            expected_video_height=2,
            expected_video_width=2,
        ),
    ).train()


def _batch() -> dict[str, torch.Tensor | list[str]]:
    return {
        "video": torch.randn(2, 3, 3, 2, 2),
        "action": torch.randn(2, 3, 7),
        "proprio": torch.randn(2, 1, 8),
        "context": torch.randn(2, 4, 8),
        "context_mask": torch.ones(2, 4, dtype=torch.bool),
        "action_is_pad": torch.tensor([[False, False, True], [False, False, False]]),
        "sample_identity": ["tiny:0", "tiny:1"],
    }


@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("world_size", [4, 8])
def test_rank128_presets_and_contract(dual, world_size) -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    variant = "video_action" if dual else "action"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="uncond_bc",
            overrides=[
                f"task=libero_uncond_lora_bc_{variant}_rank128",
                f"training.gradient_accumulation_steps={128 // world_size}",
            ],
        )
    _validate_training_config(cfg, world_size=world_size)
    assert cfg.lora.rank == cfg.lora.alpha == 128
    assert cfg.optimizer.learning_rate == 1e-4
    assert (
        cfg.training.microbatch_size
        * cfg.training.gradient_accumulation_steps
        * world_size
        == 128
    )
    assert cfg.data.train.current_frame_only and cfg.data.validation.current_frame_only
    assert cfg.bc_policy.expected_video_frames == 1
    assert (cfg.video_lora is not None) == dual
    assert len(cfg.provenance.dataset_paths) == 4
    assert OmegaConf.to_container(cfg, resolve=True)["runner"]["stage"] == "formal"
    cfg.training.gradient_accumulation_steps *= 2
    with pytest.raises(ValueError, match="requires accumulation"):
        _validate_training_config(cfg, world_size=world_size)
    cfg.training.gradient_accumulation_steps //= 2
    with pytest.raises(ValueError, match="4 or 8 GPUs"):
        _validate_training_config(cfg, world_size=6)
    cfg.optimizer.learning_rate = 3e-4
    with pytest.raises(ValueError, match="fixed LR"):
        _validate_training_config(cfg, world_size=world_size)


def test_rank128_eight_worker_throughput_config() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="uncond_bc",
            overrides=[
                "task=libero_uncond_lora_bc_video_action_rank128",
                "training.microbatch_size=8",
                "training.gradient_accumulation_steps=2",
                "data.num_workers=8",
                "data.prefetch_factor=2",
                "data.multiprocessing_context=spawn",
                "data.persistent_workers=true",
                "model.mot_checkpoint_mixed_attn=false",
                "model.video_dit_config.use_gradient_checkpointing=false",
                "model.action_dit_config.use_gradient_checkpointing=false",
            ],
        )
    _validate_training_config(cfg, world_size=8)
    assert cfg.training.global_batch_size == 128
    assert cfg.data.persistent_workers
    cfg.data.multiprocessing_context = None
    with pytest.raises(ValueError, match="eight spawn workers"):
        _validate_training_config(cfg, world_size=8)


@pytest.mark.parametrize("dual", [False, True])
def test_rank128_gradients_frozen_parent_and_future_invariance(dual) -> None:
    policy = _policy(dual=dual)
    batch = _batch()
    noise = torch.randn_like(batch["action"])
    timestep = torch.tensor([100.0, 800.0])
    base = {
        n: p.detach().clone()
        for n, p in policy.actor.named_parameters()
        if not p.requires_grad
    }
    initial_versions = _frozen_versions(policy)
    assert len(policy.lora_adapter.target_names) == 20
    if dual:
        assert len(policy.video_lora_adapter.target_names) == 12
        assert [
            n
            for n in policy.video_lora_adapter.target_names
            if n.startswith("blocks.1.")
        ] == ["blocks.1.self_attn.k", "blocks.1.self_attn.v"]
    for parameter in policy.lora_parameters():
        assert parameter.dtype == torch.float32
    condition = policy.prepare_action_condition(batch)
    assert all(layer["k"].requires_grad == dual for layer in condition.video_kv_cache)
    parity = _bc0_parity_and_action_report(policy, batch, seed=42)
    assert parity["zero_lora_idm_uncond_exact"]
    loss = policy(batch, noise=noise, timestep=timestep)["loss_action_bc"]
    loss.backward()
    first_grads = {
        n: p.grad.clone() for n, p in policy.named_parameters() if p.requires_grad
    }
    for adapter in policy.lora_adapters.values():
        for name, parameter in adapter.named_lora_parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert bool(torch.count_nonzero(parameter.grad)) == name.endswith("lora_B")
    changed = copy.deepcopy(batch)
    changed["video"][:, :, 1:] += 1000
    policy.zero_grad(set_to_none=True)
    other = policy(changed, noise=noise, timestep=timestep)["loss_action_bc"]
    other.backward()
    assert torch.equal(loss, other)
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad:
            assert torch.equal(first_grads[name], parameter.grad)
    optimizer = torch.optim.AdamW(policy.lora_parameters(), lr=1e-4)
    before = _snapshot_lora(policy)
    optimizer.step()
    updates = _lora_update_norms(policy, before)
    assert set(updates) == ({"action", "video"} if dual else {"action"})
    assert all(torch.isfinite(norm) and norm > 0 for norm in updates.values())
    optimizer.zero_grad(set_to_none=True)
    policy(batch, noise=noise, timestep=timestep)["loss_action_bc"].backward()
    assert all(
        p.grad is not None and bool(torch.count_nonzero(p.grad))
        for p in policy.lora_parameters()
    )
    optimizer.step()
    assert _frozen_versions(policy) == initial_versions
    for name, parameter in policy.actor.named_parameters():
        if name in base:
            assert parameter.grad is None and torch.equal(parameter, base[name])


def test_dual_checkpoint_recomputation_matches_eager() -> None:
    checkpointed = _policy(dual=True, checkpoint=True)
    eager = _policy(dual=True, checkpoint=False)
    batch = _batch()
    noise = torch.randn_like(batch["action"])
    for policy in (checkpointed, eager):
        loss = policy(batch, timestep=torch.tensor([100.0, 800.0]), noise=noise)[
            "loss_action_bc"
        ]
        loss.backward()
        assert policy.lora_adapter.regime_context.current == PolicyRegime.IDM
    for left, right in zip(
        checkpointed.lora_parameters(), eager.lora_parameters(), strict=True
    ):
        torch.testing.assert_close(left.grad, right.grad, atol=0, rtol=0)


def test_rank128_nonzero_delta_scaling() -> None:
    policy = _policy(dual=False)
    layer = policy.actor.action_expert.blocks[0].ffn[0]
    assert isinstance(layer, RegimeLoRALinear)
    with torch.no_grad():
        layer.lora_B.normal_()
    inputs = torch.randn(2, 3, 8)
    expected = nn.functional.linear(
        inputs, layer.weight, layer.bias
    ) + nn.functional.linear(nn.functional.linear(inputs, layer.lora_A), layer.lora_B)
    with policy.lora_adapter.use_regime(PolicyRegime.UNCOND):
        torch.testing.assert_close(layer(inputs), expected, atol=0, rtol=0)


def _optimizer_state(policy):
    optimizer = torch.optim.AdamW(policy.lora_parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: 1.0 / (step + 1)
    )
    return optimizer, scheduler, torch.amp.GradScaler("cpu", enabled=False)


def _update(policy, optimizer, scheduler):
    optimizer.zero_grad(set_to_none=True)
    policy(_batch())["loss_action_bc"].backward()
    optimizer.step()
    scheduler.step()


def _save(path, policy, optimizer, scheduler, scaler, step):
    save_uncond_bc_checkpoint(
        path,
        adapter=policy.lora_adapter,
        video_adapter=policy.video_lora_adapter,
        parent_checkpoint_sha256="a" * 64,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        grad_scaler=scaler,
        global_step=step,
        epoch=0,
        sampler_offset=step,
        rng_by_rank=[capture_rng_state()],
        contract={"resolved_config_sha256": "b" * 64},
        provenance={},
    )


@pytest.mark.parametrize("dual", [False, True])
def test_rank128_checkpoint_exact_resume_and_best_sidecar(tmp_path, dual) -> None:
    policy = _policy(dual=dual)
    optimizer, scheduler, scaler = _optimizer_state(policy)
    _update(policy, optimizer, scheduler)
    checkpoint = tmp_path / "step1.pt"
    _save(checkpoint, policy, optimizer, scheduler, scaler, 1)
    report = inspect_uncond_bc_checkpoint(checkpoint)
    assert report["schema"] == (
        DUAL_UNCOND_BC_TRAINING_SCHEMA if dual else UNCOND_BC_TRAINING_SCHEMA
    )
    assert report["lora_tensor_count"] == (64 if dual else 40)
    _update(policy, optimizer, scheduler)
    uninterrupted = tmp_path / "step2.pt"
    _save(uninterrupted, policy, optimizer, scheduler, scaler, 2)

    restored = _policy(dual=dual)
    optimizer2, scheduler2, scaler2 = _optimizer_state(restored)
    payload = load_uncond_bc_checkpoint(
        checkpoint,
        adapter=restored.lora_adapter,
        video_adapter=restored.video_lora_adapter,
        expected_parent_checkpoint_sha256="a" * 64,
        expected_contract={"resolved_config_sha256": "b" * 64},
        optimizer=optimizer2,
        lr_scheduler=scheduler2,
        grad_scaler=scaler2,
    )
    restore_rng_state(payload["rng_by_rank"][0])
    _update(restored, optimizer2, scheduler2)
    resumed = tmp_path / "resumed2.pt"
    _save(resumed, restored, optimizer2, scheduler2, scaler2, 2)
    assert compare_uncond_bc_checkpoints(uninterrupted, resumed)["exact_training_state"]
    load_uncond_bc_adapter_checkpoint(
        checkpoint,
        adapter=restored.lora_adapter,
        video_adapter=restored.video_lora_adapter,
        expected_parent_checkpoint_sha256="a" * 64,
    )
    sidecar = tmp_path / "best.pt"
    extra = {"bc_step": 1, "bc_config_sha256": "b" * 64}
    save_uncond_bc_sidecar(
        sidecar,
        adapter=restored.lora_adapter,
        video_adapter=restored.video_lora_adapter,
        parent_checkpoint_sha256="a" * 64,
        extra_metadata=extra,
    )
    assert _strict_reload_best_sidecar(
        restored, sidecar, parent_sha256="a" * 64, expected_extra=extra
    )["tensor_exact"]
    if dual:
        with pytest.raises(ValueError, match="keys changed"):
            load_uncond_bc_sidecar(
                sidecar,
                adapter=restored.lora_adapter,
                expected_parent_checkpoint_sha256="a" * 64,
            )
        broken = torch.load(checkpoint, weights_only=False)
        broken["video_adapter"]["state_dict"].pop(
            next(iter(broken["video_adapter"]["state_dict"]))
        )
        torch.save(broken, tmp_path / "broken.pt")
        with pytest.raises(ValueError, match="targets"):
            inspect_uncond_bc_checkpoint(tmp_path / "broken.pt")


def _ddp_worker(rank, init_path, dual):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        policy = _policy(dual=dual)
        model = DistributedDataParallel(
            policy, broadcast_buffers=False, find_unused_parameters=False
        )
        optimizer = torch.optim.AdamW(policy.lora_parameters(), lr=1e-4)
        torch.manual_seed(100 + rank)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with model.no_sync():
                (model(_batch())["loss_action_bc"] / 2).backward()
            (model(_batch())["loss_action_bc"] / 2).backward()
            assert all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in policy.lora_parameters()
            )
            optimizer.step()
        flat = torch.nn.utils.parameters_to_vector(
            list(policy.lora_parameters())
        ).detach()
        gathered = [torch.empty_like(flat) for _ in range(2)]
        dist.all_gather(gathered, flat)
        assert torch.equal(gathered[0], gathered[1])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("dual", [False, True])
def test_rank128_two_rank_ddp_two_updates(tmp_path, dual):
    mp.spawn(
        _ddp_worker, args=(str(tmp_path / "rendezvous"), dual), nprocs=2, join=True
    )
