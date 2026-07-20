from __future__ import annotations

import copy

import pytest
import torch

from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.mot import MoT, MoTAttentionGroup, MoTExpertSpan


VIDEO_LEN = 4
FIRST_FRAME_LEN = 2
ACTION_LEN = 3
HIDDEN_DIM = 8
CONTEXT_LEN = 5


def _expert(*, checkpoint: bool = False) -> ActionDiT:
    return ActionDiT(
        hidden_dim=HIDDEN_DIM,
        action_dim=2,
        ffn_dim=16,
        text_dim=HIDDEN_DIM,
        freq_dim=8,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=4,
        num_layers=2,
        use_gradient_checkpointing=checkpoint,
    )


def _mot(*, checkpoint: bool = False) -> MoT:
    return MoT(
        {"video": _expert(checkpoint=checkpoint), "action": _expert(checkpoint=checkpoint)},
        mot_checkpoint_mixed_attn=checkpoint,
    )


def _attention_mask() -> torch.Tensor:
    total = VIDEO_LEN + 2 * ACTION_LEN
    mask = torch.zeros((total, total), dtype=torch.bool)
    video_mask = torch.ones((VIDEO_LEN, VIDEO_LEN), dtype=torch.bool)
    video_mask[:FIRST_FRAME_LEN, FIRST_FRAME_LEN:] = False
    mask[:VIDEO_LEN, :VIDEO_LEN] = video_mask

    main_start = VIDEO_LEN
    main_end = main_start + ACTION_LEN
    mask[main_start:main_end, :VIDEO_LEN] = True
    mask[main_start:main_end, main_start:main_end] = True

    base_start = main_end
    base_end = base_start + ACTION_LEN
    mask[base_start:base_end, :FIRST_FRAME_LEN] = True
    mask[base_start:base_end, base_start:base_end] = True
    return mask


def _groups() -> tuple[MoTAttentionGroup, ...]:
    return (
        MoTAttentionGroup(
            "main",
            (
                MoTExpertSpan("video", 0, VIDEO_LEN),
                MoTExpertSpan("action", 0, ACTION_LEN),
            ),
        ),
        MoTAttentionGroup(
            "base",
            (
                MoTExpertSpan("video", 0, FIRST_FRAME_LEN, write=False),
                MoTExpertSpan("action", ACTION_LEN, 2 * ACTION_LEN),
            ),
        ),
    )


def _payload(model: MoT) -> dict:
    torch.manual_seed(0)
    batch = 2
    video = torch.randn(batch, VIDEO_LEN, HIDDEN_DIM)
    main_action = torch.randn(batch, ACTION_LEN, HIDDEN_DIM)
    base_action = torch.randn(batch, ACTION_LEN, HIDDEN_DIM)
    action = torch.cat((main_action, base_action), dim=1)

    video_freqs = model.mixtures["video"].freqs[:VIDEO_LEN].view(
        VIDEO_LEN, 1, -1
    )
    draft_freqs = model.mixtures["action"].freqs[:ACTION_LEN].view(
        ACTION_LEN, 1, -1
    )
    action_freqs = torch.cat((draft_freqs, draft_freqs), dim=0)

    video_context = torch.randn(batch, CONTEXT_LEN, HIDDEN_DIM)
    action_context = torch.randn(batch, CONTEXT_LEN, HIDDEN_DIM)
    return {
        "embeds_all": {"video": video, "action": action},
        "attention_mask": _attention_mask(),
        "freqs_all": {"video": video_freqs, "action": action_freqs},
        "context_all": {
            "video": {
                "context": video_context,
                "mask": torch.ones(
                    batch, VIDEO_LEN, CONTEXT_LEN, dtype=torch.bool
                ),
            },
            "action": {
                "context": action_context,
                "mask": torch.ones(
                    batch, 2 * ACTION_LEN, CONTEXT_LEN, dtype=torch.bool
                ),
            },
        },
        "t_mod_all": {
            "video": torch.randn(batch, VIDEO_LEN, 6, HIDDEN_DIM) * 0.1,
            "action": torch.randn(batch, 2 * ACTION_LEN, 6, HIDDEN_DIM) * 0.1,
        },
    }


def _submask(mask: torch.Tensor, indices: list[int]) -> torch.Tensor:
    index = torch.tensor(indices, dtype=torch.long)
    return mask.index_select(0, index).index_select(1, index)


def _reference(model: MoT, payload: dict) -> dict[str, torch.Tensor]:
    main_indices = list(range(VIDEO_LEN)) + list(
        range(VIDEO_LEN, VIDEO_LEN + ACTION_LEN)
    )
    base_indices = list(range(FIRST_FRAME_LEN)) + list(
        range(VIDEO_LEN + ACTION_LEN, VIDEO_LEN + 2 * ACTION_LEN)
    )
    main = model(
        embeds_all={
            "video": payload["embeds_all"]["video"],
            "action": payload["embeds_all"]["action"][:, :ACTION_LEN],
        },
        attention_mask=_submask(payload["attention_mask"], main_indices),
        freqs_all={
            "video": payload["freqs_all"]["video"],
            "action": payload["freqs_all"]["action"][:ACTION_LEN],
        },
        context_all={
            "video": payload["context_all"]["video"],
            "action": {
                "context": payload["context_all"]["action"]["context"],
                "mask": payload["context_all"]["action"]["mask"][:, :ACTION_LEN],
            },
        },
        t_mod_all={
            "video": payload["t_mod_all"]["video"],
            "action": payload["t_mod_all"]["action"][:, :ACTION_LEN],
        },
    )
    base = model(
        embeds_all={
            "video": payload["embeds_all"]["video"][:, :FIRST_FRAME_LEN],
            "action": payload["embeds_all"]["action"][:, ACTION_LEN:],
        },
        attention_mask=_submask(payload["attention_mask"], base_indices),
        freqs_all={
            "video": payload["freqs_all"]["video"][:FIRST_FRAME_LEN],
            "action": payload["freqs_all"]["action"][ACTION_LEN:],
        },
        context_all={
            "video": {
                "context": payload["context_all"]["video"]["context"],
                "mask": payload["context_all"]["video"]["mask"][
                    :, :FIRST_FRAME_LEN
                ],
            },
            "action": {
                "context": payload["context_all"]["action"]["context"],
                "mask": payload["context_all"]["action"]["mask"][:, ACTION_LEN:],
            },
        },
        t_mod_all={
            "video": payload["t_mod_all"]["video"][:, :FIRST_FRAME_LEN],
            "action": payload["t_mod_all"]["action"][:, ACTION_LEN:],
        },
    )
    return {
        "video": main["video"],
        "action": torch.cat((main["action"], base["action"]), dim=1),
    }


def test_group_submasks_match_reference_shapes_and_values():
    full = _attention_mask()
    main_indices = list(range(VIDEO_LEN)) + list(
        range(VIDEO_LEN, VIDEO_LEN + ACTION_LEN)
    )
    base_indices = list(range(FIRST_FRAME_LEN)) + list(
        range(VIDEO_LEN + ACTION_LEN, VIDEO_LEN + 2 * ACTION_LEN)
    )
    assert _submask(full, main_indices).shape == (
        VIDEO_LEN + ACTION_LEN,
        VIDEO_LEN + ACTION_LEN,
    )
    assert _submask(full, base_indices).shape == (
        FIRST_FRAME_LEN + ACTION_LEN,
        FIRST_FRAME_LEN + ACTION_LEN,
    )
    base = _submask(full, base_indices)
    expected_base = torch.zeros_like(base)
    expected_base[:FIRST_FRAME_LEN, :FIRST_FRAME_LEN] = True
    expected_base[FIRST_FRAME_LEN:, :FIRST_FRAME_LEN] = True
    expected_base[FIRST_FRAME_LEN:, FIRST_FRAME_LEN:] = True
    assert torch.equal(base, expected_base)


def test_grouped_single_forward_matches_two_reference_forwards_fp32():
    grouped_model = _mot()
    reference_model = copy.deepcopy(grouped_model)
    grouped_model.eval()
    reference_model.eval()
    payload = _payload(grouped_model)

    grouped = grouped_model(**payload, attention_groups=_groups())
    reference = _reference(reference_model, payload)

    torch.testing.assert_close(grouped["video"], reference["video"])
    torch.testing.assert_close(
        grouped["action"], reference["action"], atol=1e-5, rtol=1e-5
    )


def test_grouped_gradients_match_two_reference_forwards_fp32():
    grouped_model = _mot()
    reference_model = copy.deepcopy(grouped_model)
    grouped_model.train()
    reference_model.train()
    payload = _payload(grouped_model)

    grouped = grouped_model(**payload, attention_groups=_groups())
    grouped_loss = grouped["video"].square().mean() + grouped["action"].square().mean()
    grouped_loss.backward()

    reference = _reference(reference_model, payload)
    reference_loss = (
        reference["video"].square().mean() + reference["action"].square().mean()
    )
    reference_loss.backward()

    grouped_grads = dict(grouped_model.named_parameters())
    for name, reference_param in reference_model.named_parameters():
        if reference_param.grad is None:
            continue
        assert grouped_grads[name].grad is not None, name
        torch.testing.assert_close(
            grouped_grads[name].grad,
            reference_param.grad,
            atol=2e-5,
            rtol=2e-4,
            msg=lambda message, name=name: f"{name}: {message}",
        )


def test_grouped_checkpoint_backward_covers_block_parameters():
    model = _mot(checkpoint=True)
    model.train()
    output = model(**_payload(model), attention_groups=_groups())
    (output["video"].mean() + output["action"].mean()).backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if ".blocks." in name and parameter.requires_grad and parameter.grad is None
    ]
    assert not missing


def test_attention_groups_none_preserves_dense_path():
    model = _mot()
    model.eval()
    payload = _payload(model)
    omitted = model(**payload)
    explicit = model(**payload, attention_groups=None)
    torch.testing.assert_close(omitted["video"], explicit["video"], rtol=0, atol=0)
    torch.testing.assert_close(omitted["action"], explicit["action"], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("groups", "message"),
    (
        ((), "cannot be empty"),
        (
            (
                MoTAttentionGroup(
                    "bad",
                    (
                        MoTExpertSpan("video", 0, VIDEO_LEN),
                        MoTExpertSpan("action", 0, 2 * ACTION_LEN),
                    ),
                ),
                MoTAttentionGroup(
                    "duplicate",
                    (MoTExpertSpan("action", 0, ACTION_LEN),),
                ),
            ),
            "exactly once",
        ),
        (
            (
                MoTAttentionGroup(
                    "missing",
                    (
                        MoTExpertSpan("video", 0, VIDEO_LEN),
                        MoTExpertSpan("action", 0, ACTION_LEN),
                    ),
                ),
            ),
            "exactly once",
        ),
        (
            (
                MoTAttentionGroup(
                    "range",
                    (
                        MoTExpertSpan("video", 0, VIDEO_LEN + 1),
                        MoTExpertSpan("action", 0, 2 * ACTION_LEN),
                    ),
                ),
            ),
            "out of range",
        ),
    ),
)
def test_invalid_group_plans_fail_closed(groups, message):
    model = _mot()
    payload = _payload(model)
    with pytest.raises(ValueError, match=message):
        model._validate_attention_groups(groups, payload["embeds_all"])
