"""Contract tests for the explicit regime protocol (stage-2 W9).

The protocol threads a conditioning-regime label ("uncond" / "idm" / "joint")
from ``MoTAttentionGroup.regime`` (or ``MoT.forward(active_regime=...)``) down
to regime-aware wrapped submodules as an EXPLICIT argument. These tests pin:

1. per-span regime visibility inside grouped forwards (idm spans see "idm",
   base spans see "uncond", including the read-only ``write=False`` video span);
2. bitwise numerical transparency of the probe (outputs AND gradients);
3. checkpoint-replay correctness: entries recorded during backward-time
   recomputation carry the same regime as the original forward;
4. "no regime" (``None``) everywhere outside a regime-scoped forward;
5. fail-closed validation of unknown regimes and ambiguous regime sources.
"""
from __future__ import annotations

import copy

import pytest
import torch

mot_module = pytest.importorskip(
    "fastwam.models.wan22.mot", reason="fastwam model deps unavailable (imageio chain)"
)
action_dit_module = pytest.importorskip(
    "fastwam.models.wan22.action_dit",
    reason="fastwam model deps unavailable (safetensors chain)",
)
regime_module = pytest.importorskip("fastwam.models.wan22.regime")

MoT = mot_module.MoT
MoTAttentionGroup = mot_module.MoTAttentionGroup
MoTExpertSpan = mot_module.MoTExpertSpan
ActionDiT = action_dit_module.ActionDiT
REGIME_IDM = regime_module.REGIME_IDM
REGIME_UNCOND = regime_module.REGIME_UNCOND
RegimeRecorderProbe = regime_module.RegimeRecorderProbe
regime_call = regime_module.regime_call
validate_regime = regime_module.validate_regime

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


def _mot(*, checkpoint: bool = False, seed: int = 7) -> MoT:
    torch.manual_seed(seed)
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


def _payload(model: MoT, *, seed: int = 99) -> dict:
    torch.manual_seed(seed)
    batch = 2
    video = torch.randn(batch, VIDEO_LEN, HIDDEN_DIM)
    action = torch.randn(batch, 2 * ACTION_LEN, HIDDEN_DIM)
    video_freqs = model.mixtures["video"].freqs[:VIDEO_LEN].view(VIDEO_LEN, 1, -1)
    draft_freqs = model.mixtures["action"].freqs[:ACTION_LEN].view(ACTION_LEN, 1, -1)
    return {
        "embeds_all": {"video": video, "action": action},
        "attention_mask": _attention_mask(),
        "freqs_all": {
            "video": video_freqs,
            "action": torch.cat((draft_freqs, draft_freqs), dim=0),
        },
        "context_all": {
            "video": {
                "context": torch.randn(batch, CONTEXT_LEN, HIDDEN_DIM),
                "mask": torch.ones(batch, VIDEO_LEN, CONTEXT_LEN, dtype=torch.bool),
            },
            "action": {
                "context": torch.randn(batch, CONTEXT_LEN, HIDDEN_DIM),
                "mask": torch.ones(batch, 2 * ACTION_LEN, CONTEXT_LEN, dtype=torch.bool),
            },
        },
        "t_mod_all": {
            "video": torch.randn(batch, VIDEO_LEN, 6, HIDDEN_DIM) * 0.1,
            "action": torch.randn(batch, 2 * ACTION_LEN, 6, HIDDEN_DIM) * 0.1,
        },
    }


def _spans() -> dict[str, tuple[MoTExpertSpan, ...]]:
    return {
        "main": (
            MoTExpertSpan("video", 0, VIDEO_LEN),
            MoTExpertSpan("action", 0, ACTION_LEN),
        ),
        "base": (
            MoTExpertSpan("video", 0, FIRST_FRAME_LEN, write=False),
            MoTExpertSpan("action", ACTION_LEN, 2 * ACTION_LEN),
        ),
    }


def _tagged_groups() -> tuple[MoTAttentionGroup, ...]:
    spans = _spans()
    return (
        MoTAttentionGroup("idm", spans["main"], regime=REGIME_IDM),
        MoTAttentionGroup("base", spans["base"], regime=REGIME_UNCOND),
    )


def _untagged_groups() -> tuple[MoTAttentionGroup, ...]:
    spans = _spans()
    return (
        MoTAttentionGroup("main", spans["main"]),
        MoTAttentionGroup("base", spans["base"]),
    )


def _wrap_probes(model: MoT, records: list, *, targets: tuple[str, ...] = ("q", "ffn")) -> None:
    """Wrap selected block submodules in-place with recording probes.

    ``q`` observes the Q/K/V path (outside gradient checkpointing);
    ``ffn`` observes the post-attention path (inside the checkpointed region).
    """
    for expert_name in ("video", "action"):
        expert = model.mixtures[expert_name]
        for layer_idx, block in enumerate(expert.blocks):
            if "q" in targets:
                block.self_attn.q = RegimeRecorderProbe(
                    block.self_attn.q, tag=f"{expert_name}.b{layer_idx}.q", records=records
                )
            if "ffn" in targets:
                block.ffn = RegimeRecorderProbe(
                    block.ffn, tag=f"{expert_name}.b{layer_idx}.ffn", records=records
                )


def _normalized_grads(model: MoT) -> dict[str, torch.Tensor]:
    """Parameter grads keyed by probe-transparent names (`.inner` stripped)."""
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name.replace(".inner.", ".")] = param.grad.detach().clone()
    return grads


# --------------------------------------------------------------------------- #
# 1. Regime visibility inside grouped forwards
# --------------------------------------------------------------------------- #
class TestGroupedRegimeVisibility:
    def test_spans_see_their_group_regime_in_order(self):
        model = _mot()
        model.eval()
        records: list = []
        _wrap_probes(model, records)
        payload = _payload(model)
        with torch.no_grad():
            model(**payload, attention_groups=_tagged_groups())

        q_records = [entry for entry in records if entry[0].endswith(".q")]
        ffn_records = [entry for entry in records if entry[0].endswith(".ffn")]

        # Per layer, groups run in declaration order (idm then base) and spans
        # in expert order (video then action). QKV is computed for EVERY span,
        # including the read-only base video span — which must see "uncond".
        expected_q = []
        for layer_idx in range(model.num_layers):
            expected_q.extend(
                [
                    (f"video.b{layer_idx}.q", REGIME_IDM),
                    (f"action.b{layer_idx}.q", REGIME_IDM),
                    (f"video.b{layer_idx}.q", REGIME_UNCOND),
                    (f"action.b{layer_idx}.q", REGIME_UNCOND),
                ]
            )
        assert q_records == expected_q

        # Post blocks run only for write=True spans: the base group's video
        # span is read-only, so no base-video ffn call may appear.
        expected_ffn = []
        for layer_idx in range(model.num_layers):
            expected_ffn.extend(
                [
                    (f"video.b{layer_idx}.ffn", REGIME_IDM),
                    (f"action.b{layer_idx}.ffn", REGIME_IDM),
                    (f"action.b{layer_idx}.ffn", REGIME_UNCOND),
                ]
            )
        assert ffn_records == expected_ffn

    def test_untagged_groups_record_no_regime(self):
        model = _mot()
        model.eval()
        records: list = []
        _wrap_probes(model, records)
        with torch.no_grad():
            model(**_payload(model), attention_groups=_untagged_groups())
        assert records, "probes must have observed calls"
        assert {entry[1] for entry in records} == {None}

    def test_positional_group_construction_defaults_to_no_regime(self):
        group = MoTAttentionGroup("main", _spans()["main"])
        assert group.regime is None


# --------------------------------------------------------------------------- #
# 2. Bitwise numerical transparency
# --------------------------------------------------------------------------- #
class TestProbeTransparency:
    def test_outputs_and_grads_bitwise_equal_with_probes(self):
        plain = _mot()
        wrapped = copy.deepcopy(plain)
        records: list = []
        _wrap_probes(wrapped, records)

        plain.train()
        wrapped.train()
        payload = _payload(plain)

        plain_out = plain(**payload, attention_groups=_tagged_groups())
        wrapped_out = wrapped(**payload, attention_groups=_tagged_groups())
        assert torch.equal(plain_out["video"], wrapped_out["video"])
        assert torch.equal(plain_out["action"], wrapped_out["action"])

        (plain_out["video"].square().mean() + plain_out["action"].square().mean()).backward()
        (wrapped_out["video"].square().mean() + wrapped_out["action"].square().mean()).backward()
        plain_grads = _normalized_grads(plain)
        wrapped_grads = _normalized_grads(wrapped)
        assert set(plain_grads) == set(wrapped_grads)
        for name, grad in plain_grads.items():
            assert torch.equal(grad, wrapped_grads[name]), name

    def test_tagging_groups_does_not_change_outputs(self):
        model = _mot()
        model.eval()
        payload = _payload(model)
        with torch.no_grad():
            tagged = model(**payload, attention_groups=_tagged_groups())
            untagged = model(**payload, attention_groups=_untagged_groups())
        assert torch.equal(tagged["video"], untagged["video"])
        assert torch.equal(tagged["action"], untagged["action"])


# --------------------------------------------------------------------------- #
# 3. Gradient-checkpoint replay correctness
# --------------------------------------------------------------------------- #
class TestCheckpointReplay:
    def test_replay_records_carry_the_same_regime(self):
        model = _mot(checkpoint=True)
        model.train()
        records: list = []
        # ffn lives inside the checkpointed post block; q does not.
        _wrap_probes(model, records, targets=("ffn",))
        output = model(**_payload(model), attention_groups=_tagged_groups())
        forward_records = list(records)
        assert forward_records, "forward must record ffn calls"

        (output["video"].square().mean() + output["action"].square().mean()).backward()
        replay_records = records[len(forward_records):]
        assert replay_records, (
            "checkpointed post blocks must re-execute during backward; if this "
            "is empty the probe is not inside the checkpointed region"
        )
        assert None not in {entry[1] for entry in replay_records}
        # Replay recomputes each checkpointed post block exactly once, with the
        # identical regime captured in the checkpoint closure.
        assert sorted(replay_records) == sorted(forward_records)

    def test_checkpoint_grads_bitwise_equal_with_probes(self):
        plain = _mot(checkpoint=True)
        wrapped = copy.deepcopy(plain)
        _wrap_probes(wrapped, records=[])
        plain.train()
        wrapped.train()
        payload = _payload(plain)

        plain_out = plain(**payload, attention_groups=_tagged_groups())
        (plain_out["video"].square().mean() + plain_out["action"].square().mean()).backward()
        wrapped_out = wrapped(**payload, attention_groups=_tagged_groups())
        (wrapped_out["video"].square().mean() + wrapped_out["action"].square().mean()).backward()

        plain_grads = _normalized_grads(plain)
        wrapped_grads = _normalized_grads(wrapped)
        assert set(plain_grads) == set(wrapped_grads)
        for name, grad in plain_grads.items():
            assert torch.equal(grad, wrapped_grads[name]), name


# --------------------------------------------------------------------------- #
# 4. No regime outside regime-scoped forwards
# --------------------------------------------------------------------------- #
class TestNoRegimeOutsideScope:
    def test_dense_forward_without_active_regime_records_none(self):
        model = _mot()
        model.eval()
        records: list = []
        _wrap_probes(model, records)
        with torch.no_grad():
            model(**_payload(model))
        assert records
        assert {entry[1] for entry in records} == {None}

    def test_dense_forward_with_active_regime_records_it_everywhere(self):
        model = _mot()
        model.eval()
        records: list = []
        _wrap_probes(model, records)
        with torch.no_grad():
            model(**_payload(model), active_regime=REGIME_UNCOND)
        assert records
        assert {entry[1] for entry in records} == {REGIME_UNCOND}

    def test_direct_module_call_records_no_regime(self):
        records: list = []
        probe = RegimeRecorderProbe(torch.nn.Linear(4, 4), tag="direct", records=records)
        probe(torch.randn(2, 4))
        assert records == [("direct", None)]

    @pytest.mark.parametrize("regime", (REGIME_UNCOND, REGIME_IDM, None))
    def test_kv_cache_paths_forward_the_given_regime(self, regime):
        # Parametrized over BOTH known regimes and the no-regime default so a
        # hardcoded regime inside the cache path (mutation M1) cannot pass: a
        # single-value test cannot distinguish "forwards the argument" from
        # "ignores it".
        model = _mot()
        model.eval()
        records: list = []
        _wrap_probes(model, records)
        payload = _payload(model)
        mask = payload["attention_mask"]
        video_mask = mask[:VIDEO_LEN, :VIDEO_LEN]
        joint_len = VIDEO_LEN + ACTION_LEN
        regime_kwargs = {} if regime is None else {"regime": regime}
        with torch.no_grad():
            cache = model.prefill_video_cache(
                video_tokens=payload["embeds_all"]["video"],
                video_freqs=payload["freqs_all"]["video"],
                video_t_mod=payload["t_mod_all"]["video"],
                video_context_payload=payload["context_all"]["video"],
                video_attention_mask=video_mask,
                **regime_kwargs,
            )
            model.forward_action_with_video_cache(
                action_tokens=payload["embeds_all"]["action"][:, :ACTION_LEN],
                action_freqs=payload["freqs_all"]["action"][:ACTION_LEN],
                action_t_mod=payload["t_mod_all"]["action"][:, :ACTION_LEN],
                action_context_payload={
                    "context": payload["context_all"]["action"]["context"],
                    "mask": payload["context_all"]["action"]["mask"][:, :ACTION_LEN],
                },
                video_kv_cache=cache,
                attention_mask=mask[:joint_len, :joint_len],
                video_seq_len=VIDEO_LEN,
                **regime_kwargs,
            )
        assert records
        assert {entry[1] for entry in records} == {regime}


# --------------------------------------------------------------------------- #
# 5. Fail-closed validation
# --------------------------------------------------------------------------- #
class TestValidation:
    def test_unknown_group_regime_rejected(self):
        model = _mot()
        spans = _spans()
        groups = (
            MoTAttentionGroup("idm", spans["main"], regime="banana"),
            MoTAttentionGroup("base", spans["base"], regime=REGIME_UNCOND),
        )
        with pytest.raises(ValueError, match="unknown regime 'banana'"):
            model(**_payload(model), attention_groups=groups)

    def test_unknown_active_regime_rejected(self):
        model = _mot()
        with pytest.raises(ValueError, match="unknown regime 'banana'"):
            model(**_payload(model), active_regime="banana")

    def test_groups_and_active_regime_are_mutually_exclusive(self):
        model = _mot()
        with pytest.raises(ValueError, match="both `attention_groups` and `active_regime`"):
            model(
                **_payload(model),
                attention_groups=_tagged_groups(),
                active_regime=REGIME_UNCOND,
            )

    def test_validate_regime_allows_none_and_known_values(self):
        assert validate_regime(None, context="t") is None
        assert validate_regime(REGIME_IDM, context="t") == REGIME_IDM

    def test_regime_call_plain_module_ignores_regime(self):
        linear = torch.nn.Linear(4, 4)
        x = torch.randn(2, 4)
        assert torch.equal(regime_call(linear, x, regime=REGIME_IDM), linear(x))
