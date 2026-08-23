"""Differentiable ActionDiT calls for the always-active causal LoRA."""

from __future__ import annotations

import torch
from torch import nn

from fastwam.models.wan22.adaptive_action import CachedActionCondition
from fastwam.models.wan22.adaptive_sampler import VelocityOutput

from .shared_lora import SharedLoRALinear


class SharedCachedActionVelocity:
    """Bind cached video conditioning to one shared-LoRA ActionDiT.

    No route context is accepted: the same LoRA Parameter objects are active for
    every causal compute mode by construction.
    """

    def __init__(
        self,
        *,
        action_expert: nn.Module,
        mot: nn.Module,
        condition: CachedActionCondition,
    ) -> None:
        self.action_expert = action_expert
        self.mot = mot
        self.condition = condition
        if not any(
            isinstance(module, SharedLoRALinear)
            for module in self.action_expert.modules()
        ):
            raise ValueError(
                "Causal action velocity requires an injected shared ActionDiT LoRA."
            )

    def __call__(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
    ) -> VelocityOutput:
        """Predict one action velocity without exposing future tensors to a Gate."""

        if latents_action.ndim != 3:
            raise ValueError(
                "`latents_action` must be [B, horizon, action_dim], got "
                f"{tuple(latents_action.shape)}."
            )
        batch_size = latents_action.shape[0]
        if timestep_action.shape != (batch_size,):
            raise ValueError(
                f"`timestep_action` must be [{batch_size}], got "
                f"{tuple(timestep_action.shape)}."
            )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=self.condition.context,
            context_mask=self.condition.context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=self.condition.video_kv_cache,
            attention_mask=self.condition.attention_mask,
            video_seq_len=self.condition.video_seq_len,
            kv_tap=None,
        )
        return VelocityOutput(
            velocity=self.action_expert.post_dit(action_tokens, action_pre)
        )


def assert_current_prefix_cache_equal(
    current_condition: CachedActionCondition,
    full_condition: CachedActionCondition,
) -> None:
    """Require exact equality of every current-frame video K/V prefix.

    This is the executable acceptance gate that makes future availability the
    only action-conditioning difference between C0 and C2.
    """

    current_tokens = current_condition.current_frame_video_tokens
    if full_condition.current_frame_video_tokens != current_tokens:
        raise AssertionError("C0/C2 current-frame token counts differ.")
    if len(current_condition.video_kv_cache) != len(full_condition.video_kv_cache):
        raise AssertionError("C0/C2 video K/V layer counts differ.")
    for layer_index, (current, full) in enumerate(
        zip(current_condition.video_kv_cache, full_condition.video_kv_cache)
    ):
        for bank_name in ("k", "v"):
            current_bank = current[bank_name]
            full_prefix = full[bank_name][:, :current_tokens]
            if current_bank.shape != full_prefix.shape or not torch.equal(
                current_bank, full_prefix
            ):
                raise AssertionError(
                    "C0/C2 current-frame K/V prefix mismatch at "
                    f"layer={layer_index}, bank={bank_name}."
                )


def compact_current_condition(
    full_condition: CachedActionCondition,
) -> CachedActionCondition:
    """Copy current-frame K/V out of a full-shape video prefill.

    C0 deliberately performs its frozen video prefill at the same logical
    temporal shape as C2. Compacting *after* that prefill retains the exact
    parent-kernel prefix while ensuring ActionDiT can read only current-frame
    K/V and does not retain a view into the future cache storage.
    """

    current_tokens = int(full_condition.current_frame_video_tokens)
    video_tokens = int(full_condition.video_seq_len)
    if video_tokens <= current_tokens:
        raise ValueError("Current-condition compaction requires full-shape video K/V.")
    action_tokens = int(full_condition.attention_mask.shape[0]) - video_tokens
    if action_tokens < 1:
        raise ValueError("The full condition does not contain an action mask suffix.")

    compact_cache = []
    for layer_index, full_layer in enumerate(full_condition.video_kv_cache):
        compact_layer = dict(full_layer)
        for bank_name in ("k", "v"):
            bank = full_layer[bank_name]
            if bank.ndim < 2 or bank.shape[1] != video_tokens:
                raise ValueError(
                    "Full video K/V cannot be compacted at "
                    f"layer={layer_index}, bank={bank_name}."
                )
            compact_layer[bank_name] = bank[:, :current_tokens].clone(
                memory_format=torch.contiguous_format
            )
        compact_cache.append(compact_layer)

    device = full_condition.attention_mask.device
    selected = torch.cat(
        (
            torch.arange(current_tokens, device=device),
            torch.arange(
                video_tokens,
                video_tokens + action_tokens,
                device=device,
            ),
        )
    )
    compact_mask = full_condition.attention_mask.index_select(0, selected).index_select(
        1, selected
    )
    return CachedActionCondition(
        context=full_condition.context,
        context_mask=full_condition.context_mask,
        video_kv_cache=compact_cache,
        attention_mask=compact_mask,
        video_seq_len=current_tokens,
        current_frame_video_tokens=current_tokens,
    )


def splice_exact_current_prefix(
    current_condition: CachedActionCondition,
    future_condition: CachedActionCondition,
) -> CachedActionCondition:
    """Replace a full condition's current prefix with the compact C0 cache.

    A compact-shape prefill can differ from a full-shape prefix under the
    production kernels. This diagnostic helper preserves every future token
    from ``future_condition`` while installing an independently computed
    compact prefix. It is not the accepted C0/C2 production route because that
    splice changes raw-parent C2 actions; production C0 instead compacts only
    after a full-shape null prefill.
    """

    current_tokens = int(current_condition.current_frame_video_tokens)
    if current_condition.video_seq_len != current_tokens:
        raise ValueError("The exact-prefix donor must be a compact C0 condition.")
    if future_condition.current_frame_video_tokens != current_tokens:
        raise ValueError("Current and future conditions use different prefix sizes.")
    if future_condition.video_seq_len <= current_tokens:
        raise ValueError("The future condition does not contain future video tokens.")
    if len(current_condition.video_kv_cache) != len(future_condition.video_kv_cache):
        raise ValueError("Current and future conditions use different layer counts.")

    spliced_cache = []
    for layer_index, (current_layer, future_layer) in enumerate(
        zip(current_condition.video_kv_cache, future_condition.video_kv_cache)
    ):
        spliced_layer = dict(future_layer)
        for bank_name in ("k", "v"):
            current_bank = current_layer[bank_name]
            future_bank = future_layer[bank_name]
            if current_bank.ndim != future_bank.ndim or (
                current_bank.shape[0] != future_bank.shape[0]
                or current_bank.shape[2:] != future_bank.shape[2:]
                or current_bank.shape[1] != current_tokens
            ):
                raise ValueError(
                    "Current/future K/V shapes cannot be spliced at "
                    f"layer={layer_index}, bank={bank_name}."
                )
            spliced_layer[bank_name] = torch.cat(
                (current_bank, future_bank[:, current_tokens:]),
                dim=1,
            )
        spliced_cache.append(spliced_layer)

    result = CachedActionCondition(
        context=future_condition.context,
        context_mask=future_condition.context_mask,
        video_kv_cache=spliced_cache,
        attention_mask=future_condition.attention_mask,
        video_seq_len=future_condition.video_seq_len,
        current_frame_video_tokens=current_tokens,
    )
    assert_current_prefix_cache_equal(current_condition, result)
    return result
