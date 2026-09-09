"""CPU test backbone for real-robot integration, never a robot policy asset.

Uses the production cached velocity, regime LoRA, Flow-SDE and BC objectives.
The small networks replace only the expensive image/language/action backbones.
"""

from __future__ import annotations

import torch
from torch import nn

from fastwam.adapters import PolicyRegime, RegimeLoRAConfig, inject_action_dit_lora
from fastwam.models.wan22.adaptive_action import (
    CachedActionCondition,
    CachedActionVelocity,
)
from fastwam.models.wan22.condition_kv import ConditionLayerKV
from fastwam.models.wan22.kv_tap import KeyValueBank, KVSource
from fastwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
)
from fastwam.uncond_bc import (
    FastWAMUncondBCPolicy,
    compute_action_flow_matching_bc_loss,
)


class TinyAttention(nn.Module):
    def __init__(self, dim=16):
        super().__init__()
        for name in ("q", "k", "v", "o"):
            setattr(self, name, nn.Linear(dim, dim))

    def forward(self, x, condition):
        return self.o(
            torch.tanh(self.q(x) + self.k(condition)) * torch.sigmoid(self.v(condition))
        )


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttention()
        self.cross_attn = TinyAttention()
        self.ffn = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))

    def forward(self, x, context):
        x = x + self.self_attn(x, x.mean(1, keepdim=True))
        x = x + self.cross_attn(x, context)
        return x + self.ffn(x)


class TinyActionExpert(nn.Module):
    action_dim = 7

    def __init__(self):
        super().__init__()
        self.input = nn.Linear(7, 16)
        self.time = nn.Linear(1, 16)
        self.text_embedding = nn.Linear(16, 16)
        self.blocks = nn.ModuleList([TinyBlock()])
        self.output = nn.Linear(16, 7)

    def pre_dit(self, *, action_tokens, timestep, context, context_mask):
        return {
            "tokens": self.input(action_tokens)
            + self.time(timestep[:, None] / 1000)[:, None],
            "freqs": None,
            "t_mod": None,
            "context": self.text_embedding(context),
            "context_mask": context_mask,
        }

    def post_dit(self, tokens, pre):
        return self.output(tokens)


class TinyMoT(nn.Module):
    num_layers = 20
    num_heads = 1
    attn_head_dim = 16

    def forward_action_with_video_cache(
        self,
        *,
        action_tokens,
        action_context_payload,
        video_kv_cache,
        action_expert,
        **kwargs,
    ):
        context = action_context_payload["context"].mean(
            1, keepdim=True
        ) + video_kv_cache[-1]["v"].mean(1, keepdim=True)
        return action_expert.blocks[0](action_tokens, context)

    def read_condition_layer_kv(
        self,
        *,
        layer_index,
        video_kv_cache,
        current_frame_video_tokens,
        context,
        context_mask,
        **kwargs,
    ):
        cache = video_kv_cache[layer_index]
        key, value = (
            cache["k"][:, :current_frame_video_tokens],
            cache["v"][:, :current_frame_video_tokens],
        )
        return ConditionLayerKV(
            layer_index=layer_index,
            current_frame_video=KeyValueBank(
                source=KVSource.CURRENT_FRAME_VIDEO,
                key=key,
                value=value,
                valid_mask=torch.ones(key.shape[:2], dtype=torch.bool),
            ),
            context=KeyValueBank(
                source=KVSource.TEXT_STATE_CONTEXT,
                key=context,
                value=context,
                valid_mask=context_mask,
            ),
        )


class TinyFastWAM(nn.Module):
    text_dim = 16
    proprio_dim = 8

    def __init__(self, shift=5.0):
        super().__init__()
        self.action_expert = TinyActionExpert()
        self.mot = TinyMoT()
        self.proprio_encoder = nn.Linear(8, 16)
        self.text_encoder = nn.Embedding(256, 16)
        self.video_encoder = nn.Linear(3, 16)
        self.video_layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(20)])
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(shift=shift)
        self.infer_action_scheduler = self.train_action_scheduler

    def encode_prompt(self, prompts):
        tokens = torch.zeros(len(prompts), 16, dtype=torch.long)
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        for i, prompt in enumerate(prompts):
            values = list(prompt.encode("utf-8"))[:16]
            tokens[i, : len(values)] = torch.tensor(values)
            mask[i, : len(values)] = True
        return self.text_encoder(tokens), mask

    def _append_proprio_to_context(self, *, context, context_mask, proprio):
        return torch.cat(
            [context, self.proprio_encoder(proprio)[:, None]], 1
        ), torch.cat([context_mask, torch.ones(len(context), 1, dtype=torch.bool)], 1)

    def make_condition(self, image, context, context_mask, *, future_noise=None):
        tokens = self.video_encoder(image.mean((-3, -2, -1))[:, None])
        if future_noise is not None:
            tokens = torch.cat([tokens, tokens + 0.1 * future_noise], 1)
        cache = []
        for layer in self.video_layers:
            tokens = torch.tanh(layer(tokens)) + 0.5 * tokens
            cache.append({"k": tokens, "v": torch.tanh(tokens)})
        size = tokens.shape[1] + 32
        return CachedActionCondition(
            context=context,
            context_mask=context_mask,
            video_kv_cache=cache,
            attention_mask=torch.ones(size, size, dtype=torch.bool),
            video_seq_len=tokens.shape[1],
            current_frame_video_tokens=1,
        )


class TinyUncondBCPolicy(FastWAMUncondBCPolicy):
    """Use the production BC forward; replace only the expensive condition encoder."""

    def prepare_action_condition(self, batch):
        context, mask = self.actor._append_proprio_to_context(
            context=batch["context"],
            context_mask=batch["context_mask"],
            proprio=batch["proprio"][:, 0],
        )
        return self.actor.make_condition(batch["video"][:, :, :1], context, mask)


def adapt_tiny_fixture(actor: TinyFastWAM, *, batch=None, steps=8):
    """Actually fit parent velocities, then UNCOND LoRA, on a synthetic CPU fixture.

    A caller may supply normalized samples from the real-data reader. Without
    them, the fixture has varied commands and conditions and is marked synthetic.
    """
    if batch is None:
        generator = torch.Generator().manual_seed(187)
        proprio = torch.rand(8, 32, 8, generator=generator)
        context, mask = actor.encode_prompt(
            [f"Move to fixture target {i}." for i in range(8)]
        )
        action = torch.randn(8, 32, 7, generator=generator) * 0.2
        action[:, :, 0] += proprio[:, :1, 0] - 0.5
        batch = {
            "video": torch.rand(8, 3, 9, 32, 32, generator=generator) * 2 - 1,
            "action": action,
            "proprio": proprio,
            "context": context.detach(),
            "context_mask": mask,
        }
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.002)
    parent_losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        context, mask = actor._append_proprio_to_context(
            context=batch["context"],
            context_mask=batch["context_mask"],
            proprio=batch["proprio"][:, 0],
        )
        future = actor.video_encoder(batch["video"][:, :, -1].mean((-2, -1)))[:, None]
        condition = actor.make_condition(
            batch["video"][:, :, :1], context, mask, future_noise=future
        )
        action = batch["action"]
        timestep = actor.train_action_scheduler.sample_training_t(
            len(action), action.device, action.dtype
        )
        noise = torch.randn_like(action)
        prediction = CachedActionVelocity(
            action_expert=actor.action_expert,
            mot=actor.mot,
            condition=condition,
            regime=PolicyRegime.IDM,
        )(
            actor.train_action_scheduler.add_noise(action, noise, timestep), timestep
        ).velocity
        loss = compute_action_flow_matching_bc_loss(
            prediction=prediction,
            target=noise - action,
            timestep=timestep,
            action_is_pad=None,
            scheduler=actor.train_action_scheduler,
        ).loss_action_bc
        loss.backward()
        optimizer.step()
        parent_losses.append(float(loss.detach()))
    parent_state = {
        key: value.detach().clone() for key, value in actor.state_dict().items()
    }
    optimizer.zero_grad(set_to_none=True)
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    adapter = inject_action_dit_lora(
        actor.action_expert, RegimeLoRAConfig(rank=16, alpha=16)
    )
    policy = TinyUncondBCPolicy(
        actor=actor, lora_adapter=adapter, lora_config=adapter.config
    )
    optimizer = torch.optim.Adam(adapter.lora_parameters(), lr=0.002)
    bc_losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = policy(batch)["loss_action_bc"]
        loss.backward()
        optimizer.step()
        bc_losses.append(float(loss.detach()))
    optimizer.zero_grad(set_to_none=True)
    actor.eval()
    return (
        adapter,
        {
            "kind": "tiny_cpu_fixture",
            "parent_losses": parent_losses,
            "bc_losses": bc_losses,
        },
        parent_state,
    )
