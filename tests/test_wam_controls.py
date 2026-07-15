"""Future-content intervention contracts."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from fastwam.adaptive_gate import (  # noqa: E402
    IDMControl,
    ShuffledFutureBank,
    ShuffledFutureDonor,
    WAMModeAdapter,
    block_action_future_reads,
    coerce_idm_control,
    intervene_video_latents,
)


def test_control_enum_is_separate_and_strict():
    assert coerce_idm_control("valid_idm") is IDMControl.VALID_IDM
    assert coerce_idm_control(IDMControl.NO_READ) is IDMControl.NO_READ
    with pytest.raises(ValueError, match="Unknown IDM control"):
        coerce_idm_control("idm")


def test_repeat_current_pays_generation_then_replaces_only_future():
    generated = torch.arange(1 * 2 * 3 * 2 * 2, dtype=torch.float32).reshape(1, 2, 3, 2, 2)
    current = torch.full((1, 2, 1, 2, 2), 7.0)
    result = intervene_video_latents(
        generated, control=IDMControl.REPEAT_CURRENT, first_frame_latents=current
    )
    assert torch.equal(result[:, :, 0:1], current)
    assert torch.equal(result[:, :, 1:], current.expand(-1, -1, 2, -1, -1))
    assert not torch.equal(result, generated)


def test_shuffled_future_keeps_recipient_current_and_checks_cell():
    recipient = torch.zeros(1, 2, 3, 2, 2)
    current = torch.ones(1, 2, 1, 2, 2)
    donor = ShuffledFutureDonor(
        latents=torch.full_like(recipient, 9.0),
        metadata={"task": "pick", "factor": "camera", "level": 3, "phase": "contact"},
    )
    expected = {"task": "pick", "factor": "camera", "level": 3, "phase": "contact"}
    result = intervene_video_latents(
        recipient,
        control="shuffled",
        first_frame_latents=current,
        donor=donor,
        expected_donor_metadata=expected,
    )
    assert torch.equal(result[:, :, 0:1], current)
    assert torch.equal(result[:, :, 1:], torch.full_like(result[:, :, 1:], 9.0))
    with pytest.raises(ValueError, match="metadata"):
        intervene_video_latents(
            recipient,
            control="shuffled",
            first_frame_latents=current,
            donor=donor,
            expected_donor_metadata={**expected, "level": 5},
        )


def test_no_read_mask_preserves_full_shape_and_action_self_attention():
    # 6 video tokens (2/frame) + 3 action tokens, initially full joint access.
    mask = torch.ones(9, 9, dtype=torch.bool)
    result = block_action_future_reads(
        mask, video_seq_len=6, video_tokens_per_frame=2
    )
    assert result.shape == mask.shape
    assert result[6:, :2].all()
    assert not result[6:, 2:6].any()
    assert result[6:, 6:].all()


def test_invalid_control_inputs_fail_closed():
    video = torch.zeros(1, 2, 3, 2, 2)
    current = torch.zeros(1, 2, 1, 2, 2)
    with pytest.raises(ValueError, match="requires"):
        intervene_video_latents(video, control="shuffled", first_frame_latents=current)
    with pytest.raises(ValueError, match="UNCOND"):
        intervene_video_latents(video, control="extra_compute", first_frame_latents=current)


def test_donor_bank_selection_is_matched_deterministic_and_excludes_recipient():
    metadata = {
        "task": "pick",
        "factor": "camera",
        "level": 3,
        "phase": "contact",
        "wam_seed": 11,
    }
    records = [
        {"state_id": state_id, "latents": torch.full((1, 2, 3, 1, 1), value), "metadata": metadata}
        for state_id, value in (("a", 1.0), ("b", 2.0), ("c", 3.0))
    ]
    bank = ShuffledFutureBank(
        records, metadata={"checkpoint": "abc", "wam_seed": 11}
    )
    first = bank.select(recipient_state_id="a", recipient_metadata=metadata, seed=7)
    second = bank.select(recipient_state_id="a", recipient_metadata=metadata, seed=7)
    assert torch.equal(first.latents, second.latents)
    assert not torch.equal(first.latents, records[0]["latents"])
    restored = ShuffledFutureBank.from_payload(bank.to_payload())
    assert torch.equal(
        restored.select(recipient_state_id="a", recipient_metadata=metadata, seed=7).latents,
        first.latents,
    )


@pytest.mark.parametrize("value", [None, True, -1])
def test_donor_bank_requires_valid_global_wam_seed(value):
    metadata = {"task": "pick", "factor": "camera", "level": 3, "phase": "contact"}
    records = [
        {
            "state_id": "a",
            "latents": torch.zeros(1, 2, 3, 1, 1),
            "metadata": metadata,
        }
    ]
    with pytest.raises(ValueError, match="wam_seed"):
        ShuffledFutureBank(records, metadata={"wam_seed": value})


def test_donor_bank_rejects_row_global_wam_seed_mismatch():
    metadata = {
        "task": "pick",
        "factor": "camera",
        "level": 3,
        "phase": "contact",
        "wam_seed": 12,
    }
    records = [
        {
            "state_id": "a",
            "latents": torch.zeros(1, 2, 3, 1, 1),
            "metadata": metadata,
        }
    ]
    with pytest.raises(ValueError, match="does not match"):
        ShuffledFutureBank(records, metadata={"wam_seed": 11})


class _ControlStubModel:
    adaptive_regimes = ("uncond", "idm")
    adaptive_backbone_kind = "idm"
    proprio_dim = None
    torch_dtype = torch.float32
    device = torch.device("cpu")

    def __init__(self):
        self.vae = type("VAE", (), {"model": type("V", (), {"z_dim": 2})()})()
        self.action_expert = type("A", (), {"action_dim": 3})()
        self.last_call = None
        scheduler = type("Scheduler", (), {"shift": 5.0, "num_train_timesteps": 1000})
        self.infer_video_scheduler = scheduler()
        self.infer_action_scheduler = scheduler()

    def _encode_input_image_latents_tensor(self, image):
        return torch.zeros(1, 2, 1, 2, 2)

    def infer_action(
        self,
        *,
        prompt,
        input_image,
        first_frame_latents,
        action_horizon,
        num_video_frames,
        proprio=None,
        context=None,
        context_mask=None,
        num_inference_steps=20,
        action_inference_steps=None,
        sigma_shift=None,
        seed=None,
        force_branch=None,
        return_routing_info=False,
        return_video_latents=False,
        idm_control="valid_idm",
        shuffled_future_donor=None,
        expected_donor_metadata=None,
    ):
        self.last_call = {
            "branch": force_branch,
            "control": str(getattr(idm_control, "value", idm_control)),
            "num_steps": num_inference_steps,
            "action_steps": action_inference_steps,
        }
        result = {
            "action": torch.zeros(action_horizon, 3),
            "_routing": {"selected_branch": force_branch},
        }
        if return_video_latents:
            result["video_latents"] = torch.zeros(1, 2, 3, 2, 2)
        return result


def test_adapter_control_routes_without_expanding_production_modes():
    model = _ControlStubModel()
    adapter = WAMModeAdapter(
        model,
        backbone_kind="idm",
        num_video_frames=9,
        generation_horizon=32,
        context_len=8,
        allow_unloaded_model=True,
    )
    image = torch.zeros(1, 3, 32, 32)
    context = torch.zeros(1, 8, 4)
    context_mask = torch.ones(1, 8, dtype=torch.bool)
    adapter.act_control(
        input_image=image,
        control="no_read",
        context=context,
        context_mask=context_mask,
    )
    assert model.last_call == {
        "branch": "idm",
        "control": "no_read",
        "num_steps": 20,
        "action_steps": 20,
    }
    adapter.act_control(
        input_image=image,
        control="extra_compute",
        context=context,
        context_mask=context_mask,
        extra_action_steps=37,
    )
    assert model.last_call == {
        "branch": "base",
        "control": "extra_compute",
        "num_steps": 37,
        "action_steps": 37,
    }
