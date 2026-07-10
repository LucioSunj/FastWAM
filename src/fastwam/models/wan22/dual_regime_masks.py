"""Pure attention-mask / payload-merge helpers for fused multi-regime training.

This module intentionally has NO fastwam-internal imports (torch only) so it can
be imported standalone — e.g. by unit tests on machines where the full
`fastwam` dependency stack (diffsynth, imageio, ...) is not installed.

Coordinate convention
---------------------
The MoT mixed-attention sequence produced by ``MoT.forward`` is the expert-order
concatenation ``[ video | action ]``. Inside those, the fused dual-regime
training further concatenates:

    [ video block 0 | video block 1 | ... || action draft 0 | action draft 1 | ... ]

- A *video block* is one independently pre-processed video sequence (e.g. the
  noisy-video branch and the teacher-forcing cond-video branch of FastWAMIDM).
  Video blocks never attend across blocks and never attend action tokens,
  exactly as in the existing FastWAM/FastWAMJoint/FastWAMIDM masks.
- An *action draft* is one independently noised copy of the action chunk, i.e.
  one training regime. Drafts attend themselves plus a caller-specified span of
  video columns, and never attend other drafts. Because video tokens never
  attend drafts, appending drafts provably leaves every video-token output
  unchanged.

All video spans below are expressed in CONCATENATED VIDEO coordinates
(offsets relative to the start of video block 0).
"""
from __future__ import annotations

from typing import Sequence

import torch


def build_multi_regime_attention_mask(
    video_block_masks: Sequence[torch.Tensor],
    draft_lens: Sequence[int],
    draft_video_spans: Sequence[Sequence[tuple[int, int]]],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Build the fused [video blocks | action drafts] boolean attention mask.

    Args:
        video_block_masks: One square bool mask per video block (the block's
            internal video->video visibility, e.g. from
            ``video_expert.build_video_to_video_mask``). Blocks are laid out in
            order and are mutually invisible.
        draft_lens: Token length of each action draft, in order.
        draft_video_spans: For each draft, the list of half-open video-column
            spans ``(start, end)`` (concatenated video coordinates) that the
            draft may attend. Every draft always attends its own tokens.
        device: Device of the returned mask. Defaults to the first block mask's
            device.

    Returns:
        Bool mask of shape ``[S_total, S_total]`` with
        ``S_total = sum(block sizes) + sum(draft_lens)``, laid out as
        ``[video block 0 | ... | draft 0 | ...]``. ``True`` = may attend.
    """
    if len(video_block_masks) == 0:
        raise ValueError("`video_block_masks` must contain at least one block mask.")
    if len(draft_lens) != len(draft_video_spans):
        raise ValueError(
            "`draft_lens` and `draft_video_spans` must have equal length, "
            f"got {len(draft_lens)} and {len(draft_video_spans)}."
        )

    block_sizes: list[int] = []
    for i, block_mask in enumerate(video_block_masks):
        if block_mask.ndim != 2 or block_mask.shape[0] != block_mask.shape[1]:
            raise ValueError(
                f"`video_block_masks[{i}]` must be a square 2D mask, got shape {tuple(block_mask.shape)}."
            )
        if block_mask.dtype != torch.bool:
            raise ValueError(f"`video_block_masks[{i}]` must be bool, got {block_mask.dtype}.")
        block_sizes.append(int(block_mask.shape[0]))

    video_total = sum(block_sizes)
    for i, draft_len in enumerate(draft_lens):
        if int(draft_len) <= 0:
            raise ValueError(f"`draft_lens[{i}]` must be positive, got {draft_len}.")
    for i, spans in enumerate(draft_video_spans):
        for start, end in spans:
            if not (0 <= int(start) < int(end) <= video_total):
                raise ValueError(
                    f"`draft_video_spans[{i}]` span ({start}, {end}) out of range for "
                    f"video_total={video_total}."
                )

    if device is None:
        device = video_block_masks[0].device
    total = video_total + sum(int(n) for n in draft_lens)
    mask = torch.zeros((total, total), dtype=torch.bool, device=device)

    # Video blocks: block-diagonal internal visibility; no cross-block, no
    # video->draft attention (so draft columns cannot perturb video outputs).
    offset = 0
    for block_mask, size in zip(video_block_masks, block_sizes):
        mask[offset : offset + size, offset : offset + size] = block_mask.to(device=device)
        offset += size

    # Action drafts: self-block plus the requested video spans; drafts are
    # mutually invisible.
    draft_offset = video_total
    for draft_len, spans in zip(draft_lens, draft_video_spans):
        draft_len = int(draft_len)
        rows = slice(draft_offset, draft_offset + draft_len)
        mask[rows, rows] = True
        for start, end in spans:
            mask[rows, int(start) : int(end)] = True
        draft_offset += draft_len

    return mask


def merge_action_draft_payloads(drafts: Sequence[dict]) -> dict:
    """Merge per-draft ``ActionDiT.pre_dit`` payloads into one action payload.

    Each draft dict must provide:
        - ``tokens``:       [B, S, D]
        - ``freqs``:        [S, 1, R]   (RoPE, sequence-major)
        - ``t_mod``:        [B, 6, D] (per-sample) or [B, S, 6, D] (token-wise)
        - ``context``:      [B, L, D]  (must be shared conditioning across drafts)
        - ``context_mask``: [B, S, L]  (per-query-token cross-attn mask)

    Per-sample ``t_mod`` is expanded to token-wise form so that drafts with
    different diffusion timesteps coexist in one sequence
    (``MoT._split_modulation`` natively supports 4D token-wise modulation).
    Drafts reuse identical RoPE positions; this is unambiguous because the
    fused attention mask never lets drafts attend each other (same trick as the
    ``[noisy_video, cond_video]`` concat in FastWAMIDM training).

    Returns:
        dict with merged ``tokens`` [B, S_sum, D], ``freqs`` [S_sum, 1, R],
        ``t_mod`` [B, S_sum, 6, D], ``context`` (draft 0's tensor),
        ``context_mask`` [B, S_sum, L], and ``draft_slices`` — one half-open
        ``(start, end)`` token span per draft, in merged-action coordinates.
    """
    if len(drafts) == 0:
        raise ValueError("`drafts` must contain at least one draft payload.")

    tokens_list, freqs_list, t_mod_list, ctx_mask_list = [], [], [], []
    draft_slices: list[tuple[int, int]] = []
    offset = 0
    for i, draft in enumerate(drafts):
        tokens = draft["tokens"]
        freqs = draft["freqs"]
        t_mod = draft["t_mod"]
        context_mask = draft["context_mask"]
        if tokens.ndim != 3:
            raise ValueError(f"draft[{i}]['tokens'] must be 3D [B,S,D], got {tuple(tokens.shape)}.")
        batch_size, seq_len, _ = tokens.shape
        if freqs.ndim != 3 or freqs.shape[0] != seq_len:
            raise ValueError(
                f"draft[{i}]['freqs'] must be [S,1,R] with S={seq_len}, got {tuple(freqs.shape)}."
            )
        if context_mask.ndim != 3 or context_mask.shape[:2] != (batch_size, seq_len):
            raise ValueError(
                f"draft[{i}]['context_mask'] must be [B,S,L] with B={batch_size}, S={seq_len}, "
                f"got {tuple(context_mask.shape)}."
            )
        if t_mod.ndim == 3:
            # [B, 6, D] per-sample -> token-wise [B, S, 6, D]
            if t_mod.shape[0] != batch_size:
                raise ValueError(
                    f"draft[{i}]['t_mod'] batch mismatch: {t_mod.shape[0]} vs {batch_size}."
                )
            t_mod = t_mod.unsqueeze(1).expand(batch_size, seq_len, *t_mod.shape[1:])
        elif t_mod.ndim == 4:
            if t_mod.shape[0] != batch_size or t_mod.shape[1] != seq_len:
                raise ValueError(
                    f"draft[{i}]['t_mod'] must be [B,S,6,D] with B={batch_size}, S={seq_len}, "
                    f"got {tuple(t_mod.shape)}."
                )
        else:
            raise ValueError(
                f"draft[{i}]['t_mod'] must be 3D per-sample or 4D token-wise, got ndim={t_mod.ndim}."
            )

        tokens_list.append(tokens)
        freqs_list.append(freqs)
        t_mod_list.append(t_mod)
        ctx_mask_list.append(context_mask)
        draft_slices.append((offset, offset + seq_len))
        offset += seq_len

    return {
        "tokens": torch.cat(tokens_list, dim=1),
        "freqs": torch.cat(freqs_list, dim=0),
        "t_mod": torch.cat(t_mod_list, dim=1),
        "context": drafts[0]["context"],
        "context_mask": torch.cat(ctx_mask_list, dim=1),
        "draft_slices": draft_slices,
    }
