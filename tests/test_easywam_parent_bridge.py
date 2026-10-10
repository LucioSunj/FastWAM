"""CPU coverage of masked FastWAM parents, text caches and dual-LoRA BC."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

from fastwam.adapters import RegimeLoRAConfig, sha256_file
from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.fastwam_idm import FastWAMIDM
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import WanVideoDiT
from fastwam.uncond_bc import FastWAMUncondBCConfig, FastWAMUncondBCPolicy
from fastwam.uncond_bc_checkpoint import load_uncond_bc_sidecar, save_uncond_bc_sidecar
from fastwam.uncond_bc_trainer import (
    _validate_training_config,
    load_strict_fastwam_parent,
)
from fastwam.utils.text_cache import load_text_context


class _VAE(nn.Module):
    def encode(self, videos, device=None, **kwargs):
        if isinstance(videos, list):
            videos = torch.stack(videos)
        return F.avg_pool3d(videos[:, :, ::4], (1, 8, 8))


class _TextEncoder(nn.Embedding):
    def forward(self, ids, mask):
        return super().forward(ids)


def _tokenize(prompts, **kwargs):
    ids = torch.tensor([[1, 2, 0, 0] for _ in prompts])
    return ids, ids != 0


def _actor(text_padding="masked"):
    video = WanVideoDiT(
        hidden_dim=16,
        in_dim=3,
        out_dim=3,
        ffn_dim=32,
        text_dim=12,
        freq_dim=8,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=2,
        attn_head_dim=8,
        num_layers=2,
        has_image_input=False,
        seperated_timestep=True,
        video_attention_mask_mode="first_frame_causal",
    )
    action = ActionDiT(
        hidden_dim=8,
        action_dim=3,
        ffn_dim=16,
        text_dim=12,
        freq_dim=8,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=8,
        num_layers=2,
    )
    return FastWAMIDM(
        video_expert=video,
        action_expert=action,
        mot=MoT({"video": video, "action": action}, mot_checkpoint_mixed_attn=False),
        vae=_VAE(),
        text_encoder=_TextEncoder(3, 12),
        tokenizer=_tokenize,
        proprio_dim=4,
        text_dim=12,
        text_padding=text_padding,
        device="cpu",
        torch_dtype=torch.float32,
    )


@pytest.fixture(autouse=True)
def _cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_online_padding_and_checkpoint_semantics(tmp_path):
    masked = _actor()
    context, mask = masked.encode_prompt(["pick"])
    assert mask.tolist() == [[True, True, False, False]]
    legacy = _actor("legacy_visible")
    legacy.text_encoder.load_state_dict(masked.text_encoder.state_dict())
    old_context, old_mask = legacy.encode_prompt(["pick"])
    assert old_mask.all()
    torch.testing.assert_close(context[:, :2], old_context[:, :2])
    assert not old_context[:, 2:].any()

    checkpoint = tmp_path / "masked.pt"
    masked.save_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="text_padding"):
        legacy.load_checkpoint(checkpoint)
    with pytest.raises(ValueError, match="text_padding"):
        load_strict_fastwam_parent(legacy, str(checkpoint))
    restored = _actor()
    restored.load_checkpoint(checkpoint)
    for name, value in masked.mot.state_dict().items():
        torch.testing.assert_close(
            restored.mot.state_dict()[name], value, atol=0, rtol=0
        )

    legacy.save_checkpoint(tmp_path / "legacy.pt")
    old_payload = torch.load(tmp_path / "legacy.pt", weights_only=True)
    old_payload.pop("text_padding")
    torch.save(old_payload, tmp_path / "historical.pt")
    legacy.load_checkpoint(tmp_path / "historical.pt")
    with pytest.raises(ValueError, match="text_padding"):
        masked.load_checkpoint(tmp_path / "historical.pt")


@pytest.mark.parametrize("variant", ["fastwam", "fastwam_joint", "fastwam_idm"])
def test_runtime_factories_forward_masked_mode(monkeypatch, variant):
    from fastwam import runtime
    from fastwam.models.wan22 import fastwam

    reference = _actor()
    components = SimpleNamespace(
        dit=reference.video_expert,
        vae=_VAE(),
        text_encoder=None,
        tokenizer=None,
        dit_path="synthetic",
        vae_path="synthetic",
        text_encoder_path=None,
        tokenizer_path=None,
    )
    monkeypatch.setattr(
        fastwam, "load_wan22_ti2v_5b_components", lambda **kw: components
    )
    monkeypatch.setattr(
        ActionDiT, "from_pretrained", lambda **kw: reference.action_expert
    )
    result = getattr(runtime, f"create_{variant}")(
        model_id="synthetic",
        tokenizer_model_id="synthetic",
        video_dit_config={"text_dim": 12},
        action_dit_config={},
        action_scheduler={
            "train_shift": 5,
            "infer_shift": 5,
            "num_train_timesteps": 1000,
        },
        proprio_dim=4,
        text_padding="masked",
        model_dtype=torch.float32,
        device="cpu",
    )
    assert result.text_padding == "masked"


def _cache(tmp_path, *, text_padding="masked", metadata=None):
    prompt = "A video prompt"
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    family = "text" if text_padding == "masked" else "t5"
    context = torch.arange(48, dtype=torch.bfloat16).reshape(4, 12)
    mask = torch.tensor([True, True, False, False])
    payload = {"context": context, "mask": mask}
    if text_padding == "masked":
        payload.update(
            format_version=3,
            encoder_id="wan22ti2v5b",
            context_len=4,
            prompt_hash=digest,
        )
    payload.update(metadata or {})
    torch.save(payload, tmp_path / f"{digest}.{family}_len4.wan22ti2v5b.pt")
    return prompt, context, mask


@pytest.mark.parametrize("text_padding", ["masked", "legacy_visible"])
def test_dataset_and_shared_loader_agree_on_padding(tmp_path, text_padding):
    prompt, context, mask = _cache(tmp_path, text_padding=text_padding)
    dataset = RobotVideoDataset.__new__(RobotVideoDataset)
    dataset.text_padding = text_padding
    dataset.context_len = 4
    dataset.text_embedding_cache_dir = str(tmp_path)
    dataset._text_context_cache = {}
    expected = load_text_context(tmp_path, prompt, 4, 12, text_padding)
    actual = dataset._get_cached_text_context(prompt)
    for left, right in zip(expected, actual):
        torch.testing.assert_close(left, right, atol=0, rtol=0)
    if text_padding == "masked":
        torch.testing.assert_close(actual[0], context, atol=0, rtol=0)
        assert torch.equal(actual[1], mask)
    else:
        assert not actual[0][~mask].any()
        assert actual[1].all()
    assert (
        dataset._get_cached_text_context(prompt)[0].data_ptr() == actual[0].data_ptr()
    )


def test_masked_cache_cannot_accidentally_read_legacy_file_or_wrong_encoder(tmp_path):
    prompt, _, _ = _cache(tmp_path, text_padding="legacy_visible")
    with pytest.raises(FileNotFoundError, match="text_len4"):
        load_text_context(tmp_path, prompt, 4, text_padding="masked")
    _cache(tmp_path, metadata={"encoder_id": "cosmos25"})
    with pytest.raises(ValueError, match="encoder_id"):
        load_text_context(tmp_path, prompt, 4, text_padding="masked")


def _bc_config(tmp_path):
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="uncond_bc", overrides=["task=libero_uncond_lora_bc_easywam"]
        )
    cfg.parent.checkpoint = str(tmp_path / "parent.pt")
    cfg.parent.checkpoint_sha256 = "a" * 64
    cfg.parent.statistics = str(tmp_path / "dataset_stats.json")
    cfg.parent.statistics_sha256 = "b" * 64
    cfg.provenance.dataset_paths = [str(tmp_path / "libero_spatial_no_noops_lerobot")]
    cfg.provenance.text_cache_path = str(tmp_path / "text_cache")
    cfg.data.expected_train_episodes = 90
    cfg.data.expected_validation_episodes = 10
    cfg.data.expected_source_episodes = 100
    cfg.data.expected_source_transitions = 2000
    return cfg


def test_new_bc_recipe_uses_new_parent_counts_and_both_masked_datasets(tmp_path):
    cfg = _bc_config(tmp_path)
    _validate_training_config(cfg, world_size=4)
    assert cfg.lora.rank == cfg.video_lora.rank == 128
    assert cfg.data.train.text_padding == cfg.data.validation.text_padding == "masked"
    assert OmegaConf.to_container(cfg, resolve=True)["parent"]["checkpoint"] == str(
        tmp_path / "parent.pt"
    )
    cfg.data.validation.text_padding = "legacy_visible"
    with pytest.raises(ValueError, match="same text_padding"):
        _validate_training_config(cfg, world_size=4)
    cfg.data.validation.text_padding = "masked"
    cfg.data.expected_source_episodes += 1
    with pytest.raises(ValueError, match="source/split counts"):
        _validate_training_config(cfg, world_size=4)


def _policy(actor):
    return FastWAMUncondBCPolicy(
        actor=actor,
        lora_config=RegimeLoRAConfig(rank=2, alpha=2),
        video_lora_config=RegimeLoRAConfig(rank=2, alpha=2),
        config=FastWAMUncondBCConfig(
            action_horizon=4,
            action_dim=3,
            proprio_dim=4,
            expected_video_frames=1,
            expected_video_height=16,
            expected_video_width=32,
            gripper_dimension=2,
        ),
    )


def test_masked_parent_dual_bc_update_and_sidecar_roundtrip(tmp_path):
    torch.manual_seed(61)
    checkpoint = tmp_path / "parent.pt"
    _actor().save_checkpoint(checkpoint)
    actor = _actor()
    load_strict_fastwam_parent(actor, str(checkpoint))
    policy = _policy(actor)
    frozen = {
        name: p.detach().clone()
        for name, p in policy.named_parameters()
        if not p.requires_grad
    }
    batch = {
        "video": torch.randn(2, 3, 1, 16, 32),
        "action": torch.randn(2, 4, 3),
        "proprio": torch.randn(2, 1, 4),
        "context": torch.randn(2, 4, 12),
        "context_mask": torch.tensor([[True, True, False, False]] * 2),
        "action_is_pad": torch.zeros(2, 4, dtype=torch.bool),
    }
    noise, timestep = torch.randn_like(batch["action"]), torch.tensor([100.0, 800.0])
    optimizer = torch.optim.AdamW(policy.lora_parameters(), lr=1e-2)
    loss = policy(batch, noise=noise, timestep=timestep)["loss_action_bc"]
    changed = dict(batch, context=batch["context"].clone())
    changed["context"][~batch["context_mask"]] += 1000
    torch.testing.assert_close(
        policy(changed, noise=noise, timestep=timestep)["loss_action_bc"], loss
    )
    loss.backward()
    for adapter in policy.lora_adapters.values():
        assert (
            sum(
                float(p.grad.abs().sum())
                for _, p in adapter.named_lora_parameters()
                if p.grad is not None
            )
            > 0
        )
    optimizer.step()
    for name, p in policy.named_parameters():
        if name in frozen:
            torch.testing.assert_close(p, frozen[name], atol=0, rtol=0)

    parent_digest = sha256_file(checkpoint)
    sidecar = tmp_path / "uncond_dual.pt"
    save_uncond_bc_sidecar(
        sidecar,
        adapter=policy.lora_adapter,
        video_adapter=policy.video_lora_adapter,
        parent_checkpoint_sha256=parent_digest,
        extra_metadata={"step": 1},
    )
    restored_actor = _actor()
    load_strict_fastwam_parent(restored_actor, str(checkpoint))
    restored = _policy(restored_actor)
    load_uncond_bc_sidecar(
        sidecar,
        adapter=restored.lora_adapter,
        video_adapter=restored.video_lora_adapter,
        expected_parent_checkpoint_sha256=parent_digest,
    )
    torch.testing.assert_close(
        policy(batch, noise=noise, timestep=timestep)["loss_action_bc"],
        restored(batch, noise=noise, timestep=timestep)["loss_action_bc"],
        atol=0,
        rtol=0,
    )
