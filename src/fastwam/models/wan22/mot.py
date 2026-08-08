from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Any

import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda import nvtx

from fastwam.utils.logging_config import get_logger

from .kv_tap import (
    GateKVTapRequest,
    GateLayerKV,
    KeyValueBank,
    KVSource,
    context_token_mask,
)
from .wan_video_dit import flash_attention, modulate, rope_apply

logger = get_logger(__name__)

_GATE_CURRENT_FRAME_PROVENANCE_KEY = "_gate_current_frame_video_tokens"
CheckpointContextFn = Callable[
    [], tuple[AbstractContextManager[Any], AbstractContextManager[Any]]
]


class MoT(nn.Module):
    def __init__(
        self,
        mixtures: dict[str, nn.Module],
        mot_checkpoint_mixed_attn: bool = True,
    ):
        super().__init__()
        if not mixtures:
            raise ValueError("`mixtures` cannot be empty.")
        if "video" not in mixtures or "action" not in mixtures:
            raise ValueError(
                "`mixtures` must include both 'video' and 'action' experts."
            )

        self.mixtures = nn.ModuleDict(mixtures)
        self.expert_order = list(self.mixtures.keys())
        self.mot_checkpoint_mixed_attn = mot_checkpoint_mixed_attn
        if mot_checkpoint_mixed_attn:
            logger.info(
                "Using gradient checkpointing for mixture attention. This will save memory but use more computation."
            )

        first_expert = self.mixtures[self.expert_order[0]]
        self.num_layers = len(first_expert.blocks)
        self.num_heads = first_expert.num_heads
        self.attn_head_dim = first_expert.attn_head_dim

        for name in self.expert_order[1:]:
            expert = self.mixtures[name]
            if len(expert.blocks) != self.num_layers:
                raise ValueError(
                    f"All experts must have same number of layers; got {self.num_layers} and {len(expert.blocks)}"
                )
            if expert.num_heads != self.num_heads:
                raise ValueError(
                    f"All experts must have same num_heads; got {self.num_heads} and {expert.num_heads}"
                )
            if expert.attn_head_dim != self.attn_head_dim:
                raise ValueError(
                    "All experts must have same attn_head_dim; "
                    f"got {self.attn_head_dim} and {expert.attn_head_dim}"
                )

        logger.info(
            f"Initialized MoT with experts: {self.expert_order}, num_layers={self.num_layers}"
        )
        for name in self.expert_order:
            expert = self.mixtures[name]
            logger.info(
                f"  Expert '{name}': num_params={sum(p.numel() for p in expert.parameters()) / 1e9:.2f} B"
            )

    @staticmethod
    def _split_modulation(block, t_mod: torch.Tensor):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1

        base_mod = block.modulation.to(dtype=t_mod.dtype, device=t_mod.device)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            base_mod + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            # means t_mod has separate modulation for each token, otherwise same modulation for all tokens in the block
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2),
                scale_msa.squeeze(2),
                gate_msa.squeeze(2),
                shift_mlp.squeeze(2),
                scale_mlp.squeeze(2),
                gate_mlp.squeeze(2),
            )
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def _mixed_attention(
        self,
        q_cat: torch.Tensor,
        k_cat: torch.Tensor,
        v_cat: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_mask = attention_mask.to(device=q_cat.device)

        def _forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            return flash_attention(
                q=q, k=k, v=v, num_heads=self.num_heads, ctx_mask=attn_mask
            )

        if self.mot_checkpoint_mixed_attn and self.training:
            return torch.utils.checkpoint.checkpoint(
                _forward,
                q_cat,
                k_cat,
                v_cat,
                use_reentrant=False,
            )
        return _forward(q_cat, k_cat, v_cat)

    def _validate_gate_current_frame_causality(
        self,
        *,
        kv_tap: GateKVTapRequest,
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> None:
        """Reject a tap whose current-frame K/V can contain direct future video."""

        selected_layers = kv_tap.selected_layers(self.num_layers)
        if not any(layer_index > 0 for layer_index in selected_layers):
            return
        self._validate_current_frame_video_mask(
            attention_mask=attention_mask,
            current_frame_video_tokens=kv_tap.current_frame_video_tokens,
            video_seq_len=video_seq_len,
        )

    @staticmethod
    def _validate_current_frame_video_mask(
        *,
        attention_mask: torch.Tensor,
        current_frame_video_tokens: int,
        video_seq_len: int,
    ) -> None:
        if attention_mask.dtype != torch.bool:
            raise TypeError("Gate video-mask provenance requires a boolean mask.")
        current_len = int(current_frame_video_tokens)
        future_video_access = attention_mask[
            :current_len,
            current_len:video_seq_len,
        ]
        if future_video_access.numel() and bool(future_video_access.any().item()):
            raise ValueError(
                "Gate current-frame K/V capture after layer 0 requires a causal "
                "video mask: current-frame video queries may not attend generated "
                "future-video tokens. Action K/V may still carry indirect future "
                "information."
            )

    def _validate_gate_cache_provenance(
        self,
        *,
        kv_tap: GateKVTapRequest,
        video_kv_cache: list[dict[str, Any]],
    ) -> None:
        if not any(
            layer_index > 0 for layer_index in kv_tap.selected_layers(self.num_layers)
        ):
            return
        expected = kv_tap.current_frame_video_tokens
        provenance = [
            layer_cache.get(_GATE_CURRENT_FRAME_PROVENANCE_KEY)
            for layer_cache in video_kv_cache
        ]
        if any(value != expected for value in provenance):
            raise ValueError(
                "Gate capture after layer 0 requires video K/V cache provenance "
                f"for exactly {expected} causal current-frame tokens."
            )

    @staticmethod
    def _apply_expert_post_block(
        block,
        residual_x: torch.Tensor,
        mixed_attn_out: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        context_payload: dict | None,
    ) -> torch.Tensor:
        x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))

        if context_payload is not None:
            context = context_payload.get("context")
            if context is not None:
                context_mask = context_payload.get("mask")
                if context_mask is not None and context_mask.dim() == 3:
                    context_mask = context_mask.unsqueeze(1)
                x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)

        mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
        x = block.gate(x, gate_mlp, block.ffn(mlp_input))
        return x

    def _build_expert_attention_io(
        self,
        expert,
        block,
        x: torch.Tensor,
        freqs: torch.Tensor,
        t_mod: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        bool,
    ]:
        """Build per-expert attention tensors and post-block states.

        Args:
            expert: Expert module that owns this `block`; only used to read
                `use_gradient_checkpointing`.
            block: Transformer block for current layer (`expert.blocks[layer_idx]`).
            x: Current expert tokens, shape [B, S, D].
            freqs: RoPE frequencies aligned with token sequence, shape [S, 1, rope_dim].
            t_mod: Time modulation tensor for this expert/layer.

        Returns:
            q: Query after q-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            k: Key after k-proj, RMSNorm, and RoPE, shape [B, S, H*Dh].
            v: Value after v-proj, shape [B, S, H*Dh].
            residual_x: Original input `x` for residual path in post block.
            gate_msa: Gating tensor for self-attention residual branch.
            shift_mlp: Shift tensor for MLP modulation.
            scale_mlp: Scale tensor for MLP modulation.
            gate_mlp: Gating tensor for MLP residual branch.
            use_gradient_checkpointing: Whether this expert enables checkpointing.
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self._split_modulation(block, t_mod)
        )
        attn_input = modulate(block.norm1(x), shift_msa, scale_msa)

        q = block.self_attn.norm_q(block.self_attn.q(attn_input))
        k = block.self_attn.norm_k(block.self_attn.k(attn_input))
        v = block.self_attn.v(attn_input)

        q = rope_apply(q, freqs, block.num_heads)
        k = rope_apply(k, freqs, block.num_heads)

        use_gradient_checkpointing = bool(
            getattr(expert, "use_gradient_checkpointing", False)
        )
        return (
            q,
            k,
            v,
            x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        )

    def _apply_post_with_optional_checkpoint(
        self,
        block,
        residual_x: torch.Tensor,
        gate_msa: torch.Tensor,
        shift_mlp: torch.Tensor,
        scale_mlp: torch.Tensor,
        gate_mlp: torch.Tensor,
        use_gradient_checkpointing: bool,
        mixed_slice: torch.Tensor,
        context_payload: dict | None,
        checkpoint_context_fn: CheckpointContextFn | None = None,
    ) -> torch.Tensor:
        """Apply post-attention computations, with optional checkpointing.

        Args:
            block: Transformer block for current layer.
            residual_x: Residual input tokens before attention update, shape [B, S, D].
            gate_msa: Gating tensor used after mixed self-attention.
            shift_mlp: Shift tensor for MLP input modulation.
            scale_mlp: Scale tensor for MLP input modulation.
            gate_mlp: Gating tensor used after MLP.
            use_gradient_checkpointing: If True and training, checkpoint this post block.
            mixed_slice: Mixed-attention output for this expert, shape [B, S, H*Dh].
            context_payload: Optional dict for cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, S, L] or [B, 1, S, L]
            checkpoint_context_fn: Optional contexts for the original forward and
                backward recomputation.

        Returns:
            Updated expert tokens after self-attn residual, optional cross-attn, and MLP.
        """

        def _post_fn(
            _mixed_slice: torch.Tensor,
            _x: torch.Tensor,
            _gate_msa: torch.Tensor,
            _shift_mlp: torch.Tensor,
            _scale_mlp: torch.Tensor,
            _gate_mlp: torch.Tensor,
            _block=block,
            _context_payload=context_payload,
        ) -> torch.Tensor:
            return self._apply_expert_post_block(
                block=_block,
                residual_x=_x,
                mixed_attn_out=_mixed_slice,
                gate_msa=_gate_msa,
                shift_mlp=_shift_mlp,
                scale_mlp=_scale_mlp,
                gate_mlp=_gate_mlp,
                context_payload=_context_payload,
            )

        if use_gradient_checkpointing and self.training:
            checkpoint_kwargs: dict[str, Any] = {}
            if checkpoint_context_fn is not None:
                checkpoint_kwargs["context_fn"] = checkpoint_context_fn
            return torch.utils.checkpoint.checkpoint(
                _post_fn,
                mixed_slice,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_reentrant=False,
                **checkpoint_kwargs,
            )
        return _post_fn(
            mixed_slice,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        )

    def prefill_video_cache(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: dict | None,
        video_attention_mask: torch.Tensor,
        gate_current_frame_video_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """Prefill video branch once and cache per-layer K/V for action denoising.

        Args:
            video_tokens: Video tokens before layer 0, shape [B, Sv, D].
            video_freqs: Video RoPE frequencies, shape [Sv, 1, rope_dim].
            video_t_mod: Video time modulation tensor.
            video_context_payload: Optional dict for video cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sv, L] or [B, 1, Sv, L]
            video_attention_mask: Video self-attention mask, shape [Sv, Sv].
            gate_current_frame_video_tokens: If provided, validate and record
                the causal current-frame prefix required by later Gate taps.

        Returns:
            Layer-wise cache list with length `num_layers`.
            Each entry contains:
                - `k`: video key tensor [B, Sv, H*Dh]
                - `v`: video value tensor [B, Sv, H*Dh]
        """
        if "video" not in self.mixtures:
            raise ValueError("MoT requires `video` expert for `prefill_video_cache`.")
        if video_attention_mask.ndim != 2:
            raise ValueError(
                f"`video_attention_mask` must be 2D [S,S], got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_attention_mask.shape[1]:
            raise ValueError(
                f"`video_attention_mask` must be square, got shape {tuple(video_attention_mask.shape)}"
            )
        if video_attention_mask.shape[0] != video_tokens.shape[1]:
            raise ValueError(
                "`video_attention_mask` seq length mismatch: "
                f"mask={video_attention_mask.shape[0]} vs tokens={video_tokens.shape[1]}"
            )
        if gate_current_frame_video_tokens is not None:
            current_len = int(gate_current_frame_video_tokens)
            if current_len < 1 or current_len > int(video_tokens.shape[1]):
                raise ValueError(
                    "`gate_current_frame_video_tokens` must lie in the video "
                    f"sequence, got {current_len} for {video_tokens.shape[1]} tokens."
                )
            self._validate_current_frame_video_mask(
                attention_mask=video_attention_mask,
                current_frame_video_tokens=current_len,
                video_seq_len=int(video_tokens.shape[1]),
            )

        expert = self.mixtures["video"]
        x = video_tokens
        kv_cache: list[dict[str, Any]] = []
        for layer_idx in range(self.num_layers):
            nvtx.range_push(f"prefill_video_layer_{layer_idx}")
            block = expert.blocks[layer_idx]
            # Build video Q/K/V from current layer input tokens.
            (
                q,
                k,
                v,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            # Video prefill uses only video self-attention mask.
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=k,
                v_cat=v,
                attention_mask=video_attention_mask,
            )
            # Update video tokens for the next layer and persist current layer K/V.
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=video_context_payload,
            )
            layer_cache: dict[str, Any] = {"k": k, "v": v}
            if gate_current_frame_video_tokens is not None:
                layer_cache[_GATE_CURRENT_FRAME_PROVENANCE_KEY] = int(
                    gate_current_frame_video_tokens
                )
            kv_cache.append(layer_cache)
            nvtx.range_pop()  # prefill_video_layer_N
        return kv_cache

    def _prefill_video_cache_inner(
        self,
        video_tokens: torch.Tensor,
        video_freqs: torch.Tensor,
        video_t_mod: torch.Tensor,
        video_context_payload: dict | None,
        video_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Core loop of prefill_video_cache without NVTX or validation.

        Designed to be torch.compile-friendly. Returns flat K/V lists
        instead of list[dict] to avoid graph breaks.

        NOTE: This method skips gradient checkpointing and is intended
        for inference only. Do not call during training.

        Returns:
            (x, cache_k_list, cache_v_list) where each list has num_layers entries.
        """
        expert = self.mixtures["video"]
        x = video_tokens
        cache_k_list: list[torch.Tensor] = []
        cache_v_list: list[torch.Tensor] = []
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            (
                q,
                k,
                v,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                _use_ckpt,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=video_freqs,
                t_mod=video_t_mod,
            )
            mixed = self._mixed_attention(
                q_cat=q,
                k_cat=k,
                v_cat=v,
                attention_mask=video_attention_mask,
            )
            x = self._apply_expert_post_block(
                block=block,
                residual_x=residual_x,
                mixed_attn_out=mixed,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                context_payload=video_context_payload,
            )
            cache_k_list.append(k)
            cache_v_list.append(v)
        return x, cache_k_list, cache_v_list

    def _forward_action_with_video_cache_inner(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: dict | None,
        video_cache_k: list[torch.Tensor],
        video_cache_v: list[torch.Tensor],
        action_attention_mask: torch.Tensor,
        kv_tap: GateKVTapRequest | None = None,
        checkpoint_context_fn: CheckpointContextFn | None = None,
    ) -> torch.Tensor:
        """Core loop of forward_action_with_video_cache without NVTX or validation.

        This method is designed to be torch.compile-friendly. All validation
        and mask slicing should be done by the caller.

        This method supports expert-requested gradient checkpointing while
        retaining cached video K/V.

        Args:
            action_tokens: Action tokens before layer 0, shape [B, Sa, D].
            action_freqs: Action RoPE frequencies, shape [Sa, 1, rope_dim].
            action_t_mod: Action time modulation tensor.
            action_context_payload: Optional dict for action cross-attention.
            video_cache_k: Per-layer cached video keys, length num_layers.
            video_cache_v: Per-layer cached video values, length num_layers.
            action_attention_mask: Pre-sliced action rows of the joint mask,
                shape [Sa, Sv+Sa].
            checkpoint_context_fn: Optional contexts for checkpoint forward and
                recomputation.

        Returns:
            Updated action tokens after all layers, shape [B, Sa, D].
        """
        expert = self.mixtures["action"]
        x = action_tokens
        for layer_idx in range(self.num_layers):
            block = expert.blocks[layer_idx]
            (
                q_action,
                k_action,
                v_action,
                residual_x,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                use_gradient_checkpointing,
            ) = self._build_expert_attention_io(
                expert=expert,
                block=block,
                x=x,
                freqs=action_freqs,
                t_mod=action_t_mod,
            )
            k_video = video_cache_k[layer_idx]
            v_video = video_cache_v[layer_idx]

            if kv_tap is not None and kv_tap.should_capture(layer_idx, self.num_layers):
                self._capture_gate_layer_kv(
                    kv_tap=kv_tap,
                    layer_idx=layer_idx,
                    video_k=k_video,
                    video_v=v_video,
                    action_k=k_action,
                    action_v=v_action,
                    action_block=block,
                    action_context_payload=action_context_payload,
                )

            k_cat = torch.cat([k_video, k_action], dim=1)
            v_cat = torch.cat([v_video, v_action], dim=1)
            mixed = self._mixed_attention(
                q_cat=q_action,
                k_cat=k_cat,
                v_cat=v_cat,
                attention_mask=action_attention_mask,
            )
            x = self._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=residual_x,
                gate_msa=gate_msa,
                shift_mlp=shift_mlp,
                scale_mlp=scale_mlp,
                gate_mlp=gate_mlp,
                use_gradient_checkpointing=use_gradient_checkpointing,
                mixed_slice=mixed,
                context_payload=action_context_payload,
                checkpoint_context_fn=checkpoint_context_fn,
            )
        return x

    def _capture_gate_layer_kv(
        self,
        *,
        kv_tap: GateKVTapRequest,
        layer_idx: int,
        video_k: torch.Tensor,
        video_v: torch.Tensor,
        action_k: torch.Tensor,
        action_v: torch.Tensor,
        action_block,
        action_context_payload: dict | None,
    ) -> None:
        """Capture detached current-frame/action/context K/V for the Gate sidecar."""

        if (
            action_context_payload is None
            or action_context_payload.get("context") is None
        ):
            raise ValueError("Gate K/V capture requires the action text/state context.")
        context = action_context_payload["context"]
        context_mask = context_token_mask(
            action_context_payload.get("mask"),
            context=context,
        )
        with torch.no_grad():
            context_k = action_block.cross_attn.norm_k(
                action_block.cross_attn.k(context)
            )
            context_v = action_block.cross_attn.v(context)

        batch_size = action_k.shape[0]
        timestep = kv_tap.normalized_timestep(
            batch_size,
            device=action_k.device,
            dtype=action_k.dtype,
        )
        current_video_len = kv_tap.current_frame_video_tokens
        kv_tap.collector.append(
            GateLayerKV(
                layer_index=layer_idx,
                denoise_timestep=timestep,
                current_mode=kv_tap.normalized_modes(batch_size),
                current_frame_video=KeyValueBank(
                    source=KVSource.CURRENT_FRAME_VIDEO,
                    key=video_k[:, :current_video_len],
                    value=video_v[:, :current_video_len],
                    valid_mask=torch.ones(
                        (batch_size, current_video_len),
                        dtype=torch.bool,
                        device=video_k.device,
                    ),
                    contains_generated_future_video=False,
                ),
                action=KeyValueBank(
                    source=KVSource.ACTION,
                    key=action_k,
                    value=action_v,
                    valid_mask=torch.ones(
                        action_k.shape[:2],
                        dtype=torch.bool,
                        device=action_k.device,
                    ),
                ),
                context=KeyValueBank(
                    source=KVSource.TEXT_STATE_CONTEXT,
                    key=context_k,
                    value=context_v,
                    valid_mask=context_mask,
                ),
                actor_version=kv_tap.actor_version,
            )
        )

    def forward_action_with_video_cache(
        self,
        action_tokens: torch.Tensor,
        action_freqs: torch.Tensor,
        action_t_mod: torch.Tensor,
        action_context_payload: dict | None,
        video_kv_cache: list[dict[str, Any]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        kv_tap: GateKVTapRequest | None = None,
        checkpoint_context_fn: CheckpointContextFn | None = None,
    ) -> torch.Tensor:
        """Run action branch with cached video K/V instead of recomputing video tokens.

        Args:
            action_tokens: Action tokens before layer 0, shape [B, Sa, D].
            action_freqs: Action RoPE frequencies, shape [Sa, 1, rope_dim].
            action_t_mod: Action time modulation tensor.
            action_context_payload: Optional dict for action cross-attention.
                - `context`: encoder states [B, L, D]
                - `mask`: attention mask [B, Sa, L] or [B, 1, Sa, L]
            video_kv_cache: Layer-wise cached video K/V from `prefill_video_cache`.
            attention_mask: Joint [video+action] mask, shape [Sv+Sa, Sv+Sa].
            video_seq_len: Video token count `Sv` in the joint sequence prefix.
            checkpoint_context_fn: Optional contexts for checkpoint forward and
                recomputation.

        Returns:
            Updated action tokens after all layers, shape [B, Sa, D].
        """
        if "action" not in self.mixtures:
            raise ValueError(
                "MoT requires `action` expert for `forward_action_with_video_cache`."
            )
        if len(video_kv_cache) != self.num_layers:
            raise ValueError(
                f"`video_kv_cache` must contain {self.num_layers} layers, got {len(video_kv_cache)}."
            )
        if attention_mask.ndim != 2:
            raise ValueError(
                f"`attention_mask` must be 2D [S,S], got shape {tuple(attention_mask.shape)}"
            )
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(
                f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}"
            )

        action_seq_len = int(action_tokens.shape[1])
        total_seq_len = int(video_seq_len) + action_seq_len
        if attention_mask.shape[0] != total_seq_len:
            raise ValueError(
                "`attention_mask` seq length mismatch: "
                f"mask={attention_mask.shape[0]} vs expected_total={total_seq_len}"
            )

        for layer_idx in range(self.num_layers):
            layer_cache = video_kv_cache[layer_idx]
            if "k" not in layer_cache or "v" not in layer_cache:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` must contain `k` and `v`."
                )
            k_video = layer_cache["k"]
            v_video = layer_cache["v"]
            if k_video.shape[1] != video_seq_len or v_video.shape[1] != video_seq_len:
                raise ValueError(
                    f"`video_kv_cache[{layer_idx}]` seq len mismatch, expected {video_seq_len}."
                )

        if kv_tap is not None:
            kv_tap.validate(
                num_layers=self.num_layers,
                video_seq_len=video_seq_len,
                batch_size=int(action_tokens.shape[0]),
            )
            self._validate_gate_cache_provenance(
                kv_tap=kv_tap,
                video_kv_cache=video_kv_cache,
            )

        # Pre-slice mask outside compiled region
        action_attention_mask = attention_mask[
            video_seq_len:total_seq_len, :total_seq_len
        ]

        # Extract flat lists for compile-friendly inner method
        video_cache_k = [layer_cache["k"] for layer_cache in video_kv_cache]
        video_cache_v = [layer_cache["v"] for layer_cache in video_kv_cache]

        return self._forward_action_with_video_cache_inner(
            action_tokens=action_tokens,
            action_freqs=action_freqs,
            action_t_mod=action_t_mod,
            action_context_payload=action_context_payload,
            video_cache_k=video_cache_k,
            video_cache_v=video_cache_v,
            action_attention_mask=action_attention_mask,
            kv_tap=kv_tap,
            checkpoint_context_fn=checkpoint_context_fn,
        )

    def forward(
        self,
        embeds_all: dict[str, torch.Tensor],
        attention_mask: torch.Tensor,
        freqs_all: dict[str, torch.Tensor],
        context_all: dict[str, dict | None],
        t_mod_all: dict[str, torch.Tensor],
        kv_tap: GateKVTapRequest | None = None,
    ):
        missing = [k for k in self.expert_order if k not in embeds_all]
        if missing:
            raise ValueError(f"Missing expert tokens for {missing}")
        missing = [k for k in self.expert_order if k not in freqs_all]
        if missing:
            raise ValueError(f"Missing expert freqs for {missing}")
        missing = [k for k in self.expert_order if k not in t_mod_all]
        if missing:
            raise ValueError(f"Missing expert t_mod for {missing}")

        if attention_mask.ndim != 2:
            raise ValueError(
                f"`attention_mask` must be 2D [S, S], got shape {tuple(attention_mask.shape)}"
            )
        if attention_mask.shape[0] != attention_mask.shape[1]:
            raise ValueError(
                f"`attention_mask` must be square, got shape {tuple(attention_mask.shape)}"
            )

        tokens_all = {k: v for k, v in embeds_all.items()}
        if kv_tap is not None:
            kv_tap.validate(
                num_layers=self.num_layers,
                video_seq_len=int(tokens_all["video"].shape[1]),
                batch_size=int(tokens_all["action"].shape[0]),
            )
            self._validate_gate_current_frame_causality(
                kv_tap=kv_tap,
                attention_mask=attention_mask,
                video_seq_len=int(tokens_all["video"].shape[1]),
            )

        for layer_idx in range(self.num_layers):
            q_chunks = []
            k_chunks = []
            v_chunks = []
            cached = {}
            seq_lens = []
            layer_kv = {}

            for name in self.expert_order:
                expert = self.mixtures[name]
                block = expert.blocks[layer_idx]
                x = tokens_all[name]
                freqs = freqs_all[name]
                t_mod = t_mod_all[name]

                (
                    q,
                    k,
                    v,
                    residual_x,
                    gate_msa,
                    shift_mlp,
                    scale_mlp,
                    gate_mlp,
                    use_gradient_checkpointing,
                ) = self._build_expert_attention_io(
                    expert=expert,
                    block=block,
                    x=x,
                    freqs=freqs,
                    t_mod=t_mod,
                )

                q_chunks.append(q)
                k_chunks.append(k)
                v_chunks.append(v)
                seq_lens.append(x.shape[1])
                cached[name] = {
                    "block": block,
                    "residual_x": residual_x,
                    "gate_msa": gate_msa,
                    "shift_mlp": shift_mlp,
                    "scale_mlp": scale_mlp,
                    "gate_mlp": gate_mlp,
                    "use_gradient_checkpointing": use_gradient_checkpointing,
                }
                layer_kv[name] = (k, v, block)

            if kv_tap is not None and kv_tap.should_capture(layer_idx, self.num_layers):
                video_k, video_v, _video_block = layer_kv["video"]
                action_k, action_v, action_block = layer_kv["action"]
                self._capture_gate_layer_kv(
                    kv_tap=kv_tap,
                    layer_idx=layer_idx,
                    video_k=video_k,
                    video_v=video_v,
                    action_k=action_k,
                    action_v=action_v,
                    action_block=action_block,
                    action_context_payload=context_all.get("action"),
                )

            # 3. concat all tokens for mixed attention
            q_cat = torch.cat(q_chunks, dim=1)
            k_cat = torch.cat(k_chunks, dim=1)
            v_cat = torch.cat(v_chunks, dim=1)

            total_seq = q_cat.shape[1]
            if attention_mask.shape[0] != total_seq:
                raise ValueError(
                    "Attention mask seq length mismatch: "
                    f"mask={attention_mask.shape[0]} vs tokens={total_seq}"
                )

            mixed = self._mixed_attention(
                q_cat=q_cat, k_cat=k_cat, v_cat=v_cat, attention_mask=attention_mask
            )

            start = 0
            for name, seq_len in zip(self.expert_order, seq_lens):
                # 4. split mixed attention output and apply post-attention blocks for each expert
                end = start + seq_len
                mixed_slice = mixed[:, start:end, :]
                cached_expert = cached[name]
                block = cached_expert["block"]
                context_payload = context_all.get(name)

                updated_tokens = self._apply_post_with_optional_checkpoint(
                    block=block,
                    residual_x=cached_expert["residual_x"],
                    gate_msa=cached_expert["gate_msa"],
                    shift_mlp=cached_expert["shift_mlp"],
                    scale_mlp=cached_expert["scale_mlp"],
                    gate_mlp=cached_expert["gate_mlp"],
                    use_gradient_checkpointing=cached_expert[
                        "use_gradient_checkpointing"
                    ],
                    mixed_slice=mixed_slice,
                    context_payload=context_payload,
                )

                tokens_all[name] = updated_tokens
                start = end

        return tokens_all
