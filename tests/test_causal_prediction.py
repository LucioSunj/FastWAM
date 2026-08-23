from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from fastwam.causal_prediction import (
    CAUSAL_POLICY_CHECKPOINT_SCHEMA,
    CAUSAL_POLICY_CHECKPOINT_SCHEMA_V2,
    C1IntervalKVFusion,
    CausalComputeMode,
    CausalControlKind,
    CausalCostV2,
    CausalDomain,
    CausalGateFeatureRecordV1,
    CausalInterventionSpecV2,
    CausalOutcomeV2,
    CausalPhase,
    CausalSamplingMetadataV2,
    CausalSamplingStratum,
    CausalStateIdentityV2,
    CausalTerminationType,
    CausalUpliftGate,
    UpliftGateConfig,
    UpliftGateInputs,
    apply_temperature_calibration,
    build_causal_policy_checkpoint,
    build_gate_training_example,
    build_proposal_gate_feature,
    causal_uplift_gate_loss,
    checkpoint_passes_dual_mode_selection,
    compact_current_condition,
    compose_controlled_video_latents,
    derive_generic_proposal_seed,
    deterministic_dual_mode_sequence,
    deterministic_tri_mode_sequence,
    fit_temperature_per_head,
    inject_shared_action_dit_lora,
    inspect_uplift_gate_checkpoint_payload,
    load_causal_policy_checkpoint,
    normalized_action_medoid,
    pool_current_kv_layers,
    save_causal_policy_checkpoint,
    save_uplift_gate_checkpoint,
    select_budgeted_mode,
    select_latency_matched_proposal_count,
    select_proposal_variant,
    splice_exact_current_prefix,
    validate_fold_ownership,
)
from fastwam.causal_prediction.shared_lora import SharedLoRAConfig
from fastwam.causal_prediction_trainer import _expected_global_mode_counts
from fastwam.models.wan22.adaptive_action import CachedActionCondition


class _Attention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.o = nn.Linear(width, width)


class _Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.self_attn = _Attention(width)
        self.cross_attn = _Attention(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width)
        )


class _ActionDiT(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_Block(width), _Block(width)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            value = block.self_attn.o(block.self_attn.q(value))
            value = block.cross_attn.o(block.cross_attn.q(value))
            value = block.ffn(value)
        return value


def test_shared_lora_is_one_always_active_parameter_set() -> None:
    torch.manual_seed(4)
    model = _ActionDiT()
    value = torch.randn(2, 3, 8)
    baseline = model(value)
    adapter = inject_shared_action_dit_lora(
        model,
        SharedLoRAConfig(rank=2, alpha=2.0),
    )
    assert torch.equal(model(value), baseline)
    adapter.audit_freeze().assert_valid()
    identities = tuple(id(parameter) for parameter in adapter.lora_parameters())
    assert identities == tuple(id(parameter) for parameter in adapter.lora_parameters())

    with torch.no_grad():
        for _, layer in adapter.iter_adapted_linears():
            layer.lora_B.fill_(0.1)
    adapted = model(value)
    assert not torch.equal(adapted, baseline)
    with adapter.base_only():
        assert torch.equal(model(value), baseline)
    assert torch.equal(model(value), adapted)
    metadata = adapter.metadata(
        parent_checkpoint_sha256="a" * 64,
        statistics_sha256="b" * 64,
    )
    assert metadata["schema"] == CAUSAL_POLICY_CHECKPOINT_SCHEMA
    assert metadata["active_modes"] == ["c0_current", "c2_full"]
    assert set(adapter.lora_state_dict()) == {
        name for name, _ in adapter.named_lora_parameters()
    }
    parsed = SharedLoRAConfig(
        target_groups=(
            "self_attention_qkvo",
            "cross_attention_qkvo",
            "ffn",
        )
    )
    assert [group.value for group in parsed.target_groups] == [
        "self_attention_qkvo",
        "cross_attention_qkvo",
        "ffn",
    ]


def test_compute_modes_and_deterministic_half_assignment() -> None:
    assert not CausalComputeMode.C0_CURRENT.runs_future_prediction
    assert CausalComputeMode.C2_FULL.reads_future_condition
    assert CausalComputeMode.G_NO_READ.runs_future_prediction
    assert not CausalComputeMode.G_NO_READ.reads_future_condition
    assert not CausalComputeMode.G_NO_READ.is_routable
    sequence = deterministic_dual_mode_sequence(
        128,
        optimizer_step=3,
        accumulation_index=2,
    )
    assert sequence.count(CausalComputeMode.C0_CURRENT) == 64
    assert sequence.count(CausalComputeMode.C2_FULL) == 64
    assert CausalComputeMode.C1_ONE_PASS.runs_future_prediction
    assert CausalComputeMode.C1_ONE_PASS.reads_future_condition


def test_tri_mode_sequence_has_exact_rotating_global_quotas() -> None:
    totals = {mode: 0 for mode in ("c0_current", "c1_one_pass", "c2_full")}
    per_step = []
    for step in range(3):
        step_modes = []
        for rank in range(4):
            for accumulation in range(4):
                modes = deterministic_tri_mode_sequence(
                    8,
                    optimizer_step=step,
                    accumulation_index=accumulation,
                    rank=rank,
                    world_size=4,
                )
                assert set(modes) == set(totals)
                step_modes.extend(modes)
        counts = tuple(step_modes.count(mode) for mode in totals)
        per_step.append(counts)
        for mode in totals:
            totals[mode] += step_modes.count(mode)
    assert per_step == [(43, 43, 42), (42, 43, 43), (43, 42, 43)]
    assert set(totals.values()) == {128}


def test_optimizer_step_mode_quota_audit_matches_preregistration() -> None:
    assert _expected_global_mode_counts(exposure="dual_mode", optimizer_step=17) == (
        64,
        0,
        64,
    )
    assert _expected_global_mode_counts(exposure="current_only", optimizer_step=17) == (
        128,
        0,
        0,
    )
    assert [
        _expected_global_mode_counts(exposure="tri_mode", optimizer_step=step)
        for step in range(3)
    ] == [(43, 43, 42), (42, 43, 43), (43, 42, 43)]


def test_c1_interval_fusion_changes_only_future_rows() -> None:
    fusion = C1IntervalKVFusion()
    caches = [
        {
            "k": torch.arange(12, dtype=torch.float32).reshape(1, 6, 2),
            "v": torch.arange(12, dtype=torch.float32).reshape(1, 6, 2) + 1,
            "provenance": layer,
        }
        for layer in range(30)
    ]
    fused = fusion(caches, current_token_count=2)
    for layer, (before, after) in enumerate(zip(caches, fused)):
        assert torch.equal(before["k"][:, :2], after["k"][:, :2])
        assert torch.equal(before["v"][:, :2], after["v"][:, :2])
        if layer in range(0, 30, 4):
            assert not torch.equal(before["k"][:, 2:], after["k"][:, 2:])
        else:
            assert torch.equal(before["k"], after["k"])
    sum(item["k"].sum() + item["v"].sum() for item in fused).backward()
    assert fusion.future_mix_logits.grad is not None


def test_exact_current_prefix_splice_preserves_only_future_rows() -> None:
    context = torch.zeros(1, 2, 3)
    context_mask = torch.ones(1, 2, dtype=torch.bool)
    current = CachedActionCondition(
        context=context,
        context_mask=context_mask,
        video_kv_cache=[
            {"k": torch.full((1, 2, 3), 2.0), "v": torch.full((1, 2, 3), 3.0)}
        ],
        attention_mask=torch.zeros(3, 3),
        video_seq_len=2,
        current_frame_video_tokens=2,
    )
    full = CachedActionCondition(
        context=context,
        context_mask=context_mask,
        video_kv_cache=[
            {
                "k": torch.arange(15, dtype=torch.float32).reshape(1, 5, 3),
                "v": torch.arange(15, dtype=torch.float32).reshape(1, 5, 3) + 20,
                "provenance": "full",
            }
        ],
        attention_mask=torch.zeros(6, 6),
        video_seq_len=5,
        current_frame_video_tokens=2,
    )

    spliced = splice_exact_current_prefix(current, full)

    assert torch.equal(
        spliced.video_kv_cache[0]["k"][:, :2], current.video_kv_cache[0]["k"]
    )
    assert torch.equal(
        spliced.video_kv_cache[0]["v"][:, :2], current.video_kv_cache[0]["v"]
    )
    assert torch.equal(
        spliced.video_kv_cache[0]["k"][:, 2:], full.video_kv_cache[0]["k"][:, 2:]
    )
    assert torch.equal(
        spliced.video_kv_cache[0]["v"][:, 2:], full.video_kv_cache[0]["v"][:, 2:]
    )
    assert spliced.video_kv_cache[0]["provenance"] == "full"


def test_full_shape_condition_compacts_without_retaining_future_storage() -> None:
    context = torch.zeros(1, 2, 3)
    context_mask = torch.ones(1, 2, dtype=torch.bool)
    full_k = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3)
    full_v = full_k + 20
    full_mask = torch.arange(49).reshape(7, 7) % 2 == 0
    full = CachedActionCondition(
        context=context,
        context_mask=context_mask,
        video_kv_cache=[{"k": full_k, "v": full_v, "provenance": "full-shape-prefill"}],
        attention_mask=full_mask,
        video_seq_len=5,
        current_frame_video_tokens=2,
    )

    compact = compact_current_condition(full)

    selected = torch.tensor([0, 1, 5, 6])
    assert compact.video_seq_len == 2
    assert compact.current_frame_video_tokens == 2
    assert torch.equal(compact.video_kv_cache[0]["k"], full_k[:, :2])
    assert torch.equal(compact.video_kv_cache[0]["v"], full_v[:, :2])
    assert compact.video_kv_cache[0]["k"].untyped_storage().data_ptr() != (
        full_k.untyped_storage().data_ptr()
    )
    assert compact.video_kv_cache[0]["provenance"] == "full-shape-prefill"
    assert torch.equal(
        compact.attention_mask,
        full_mask.index_select(0, selected).index_select(1, selected),
    )


def _gate_inputs(batch: int = 3) -> UpliftGateInputs:
    return UpliftGateInputs(
        current_video_kv=torch.randn(batch, 5, 12, requires_grad=True),
        current_video_mask=torch.ones(batch, 5, dtype=torch.bool),
        language=torch.randn(batch, 3, 10, requires_grad=True),
        language_mask=torch.ones(batch, 3, dtype=torch.bool),
        proprio=torch.randn(batch, 8, requires_grad=True),
        history=torch.randn(batch, 4, 6, requires_grad=True),
        history_mask=torch.ones(batch, 4, dtype=torch.bool),
        action_proposal=torch.randn(batch, 7, requires_grad=True),
        remaining_budget=torch.ones(batch, 1),
        previous_mode=torch.tensor([[1.0, 0.0, 0.0]]).expand(batch, -1).clone(),
        steps_to_go=torch.ones(batch, 1),
    )


def test_uplift_gate_detaches_policy_inputs_and_enforces_budget(tmp_path) -> None:
    config = UpliftGateConfig(
        current_kv_dim=12,
        language_dim=10,
        history_dim=6,
        proposal_dim=7,
    )
    gate = CausalUpliftGate(config)
    inputs = _gate_inputs()
    output = gate(inputs, normalized_cost=torch.tensor([0.0, 1.0]))
    losses = causal_uplift_gate_loss(
        output,
        empirical_outcomes=torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.5, 1.0]]),
        inclusion_weights=torch.tensor([1.0, 2.0, 3.0]),
    )
    losses["loss"].backward()
    assert inputs.current_video_kv.grad is None
    assert inputs.language.grad is None
    assert inputs.proprio.grad is None
    assert all(torch.isfinite(value) for value in losses.values())
    calibrated = apply_temperature_calibration(output, torch.tensor([1.0, 2.0]))
    assert calibrated.q_values.shape == output.q_values.shape
    selected, remaining = select_budgeted_mode(
        output,
        remaining_budget=torch.tensor([0.0, 0.5, 1.0]),
        beta=0.5,
        cost_weight=0.0,
    )
    assert selected[:2].tolist() == [0, 0]
    assert bool((remaining >= 0).all())
    checkpoint = tmp_path / "gate.pt"
    save_uplift_gate_checkpoint(
        checkpoint,
        gate=gate,
        calibration={"temperatures": [1.0, 1.0]},
        training_state={"best_epoch": 3},
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert inspect_uplift_gate_checkpoint_payload(payload, gate=gate)["schema"] == (
        "causal-uplift-gate-v1"
    )


def test_checkpoint_selection_uses_frozen_two_percent_guard() -> None:
    assert checkpoint_passes_dual_mode_selection(
        {
            "loss_c0_action": 0.2,
            "loss_c2_action": 0.101,
            "loss_c2_teacher": 0.1,
        }
    )
    assert not checkpoint_passes_dual_mode_selection(
        {
            "loss_c0_action": 0.2,
            "loss_c2_action": 0.103,
            "loss_c2_teacher": 0.1,
        }
    )


def test_causal_checkpoint_contains_only_shared_model_tensors() -> None:
    model = _ActionDiT()
    adapter = inject_shared_action_dit_lora(model, SharedLoRAConfig(rank=2, alpha=2))
    payload = build_causal_policy_checkpoint(
        adapter=adapter,
        parent_checkpoint_sha256="a" * 64,
        statistics_sha256="b" * 64,
        global_step=3,
        epoch=1,
        optimizer_state={"state": {}, "param_groups": []},
        lr_scheduler_state={"last_epoch": 3},
        grad_scaler_state={},
        rng_by_rank=[{"torch_cpu": torch.get_rng_state()}],
        trainer_state={"best_validation": 0.1, "sampler_offset": 8},
        config={"image_height": 224},
    )
    assert set(payload["adapter_state_dict"]) == {
        name for name, _ in adapter.named_lora_parameters()
    }
    with pytest.raises(ValueError, match="forbidden field"):
        build_causal_policy_checkpoint(
            adapter=adapter,
            parent_checkpoint_sha256="a" * 64,
            statistics_sha256="b" * 64,
            global_step=3,
            epoch=1,
            optimizer_state={},
            lr_scheduler_state={},
            grad_scaler_state={},
            rng_by_rank=[{"torch_cpu": torch.get_rng_state()}],
            trainer_state={"observation": torch.ones(1)},
            config={},
        )


def test_causal_checkpoint_round_trip_restores_numpy_rng_payload(tmp_path) -> None:
    model = _ActionDiT()
    config = SharedLoRAConfig(rank=2, alpha=2)
    adapter = inject_shared_action_dit_lora(model, config)
    path = tmp_path / "causal.pt"
    save_causal_policy_checkpoint(
        path,
        adapter=adapter,
        parent_checkpoint_sha256="a" * 64,
        statistics_sha256="b" * 64,
        global_step=3,
        epoch=1,
        optimizer_state={"state": {}, "param_groups": []},
        lr_scheduler_state={"last_epoch": 3},
        grad_scaler_state={},
        rng_by_rank=[
            {
                "numpy": np.random.RandomState(42).get_state(),
                "torch_cpu": torch.get_rng_state(),
            }
        ],
        trainer_state={"best_validation": 0.1},
        config={"modes": ["c0_current", "c2_full"]},
    )
    restored_model = _ActionDiT()
    restored = inject_shared_action_dit_lora(restored_model, config)
    payload = load_causal_policy_checkpoint(
        path,
        adapter=restored,
        expected_parent_checkpoint_sha256="a" * 64,
        expected_statistics_sha256="b" * 64,
    )
    assert isinstance(payload["rng_by_rank"][0]["numpy"][1], np.ndarray)
    assert set(restored.lora_state_dict()) == set(adapter.lora_state_dict())


def test_tri_checkpoint_keeps_v1_immutable_and_saves_only_fusion() -> None:
    model = _ActionDiT()
    adapter = inject_shared_action_dit_lora(model, SharedLoRAConfig(rank=2, alpha=2))
    fusion = C1IntervalKVFusion()
    payload = build_causal_policy_checkpoint(
        adapter=adapter,
        fusion=fusion,
        parent_checkpoint_sha256="a" * 64,
        statistics_sha256="b" * 64,
        global_step=3,
        epoch=1,
        optimizer_state={"state": {}, "param_groups": []},
        lr_scheduler_state={"last_epoch": 3},
        grad_scaler_state={},
        rng_by_rank=[{"torch_cpu": torch.get_rng_state()}],
        trainer_state={"best_validation": 0.1},
        config={"modes": ["c0_current", "c1_one_pass", "c2_full"]},
        checkpoint_schema=CAUSAL_POLICY_CHECKPOINT_SCHEMA_V2,
    )
    assert payload["schema"] == CAUSAL_POLICY_CHECKPOINT_SCHEMA_V2
    assert set(payload["fusion_state_dict"]) == {"future_mix_logits"}
    assert payload["metadata"]["adapter_contract"]["active_modes"] == [
        "c0_current",
        "c1_one_pass",
        "c2_full",
    ]
    with pytest.raises(ValueError, match="Only tri-mode"):
        build_causal_policy_checkpoint(
            adapter=adapter,
            fusion=fusion,
            parent_checkpoint_sha256="a" * 64,
            statistics_sha256="b" * 64,
            global_step=0,
            epoch=0,
            optimizer_state={},
            lr_scheduler_state={},
            grad_scaler_state={},
            rng_by_rank=[{}],
            trainer_state={},
            config={},
        )


def test_gate_selection_uses_only_validation_tasks_and_prefers_one_on_tie() -> None:
    validate_fold_ownership(
        {
            "train": ["a", "b"],
            "validation": ["c"],
            "calibration": ["c"],
            "beta_selection": ["c"],
            "checkpoint_selection": ["c"],
            "test": ["d"],
        },
        train_tasks=["a", "b"],
        validation_tasks=["c"],
        test_tasks=["d"],
    )
    temperatures = fit_temperature_per_head(
        torch.tensor([[0.2, 0.8], [0.8, 0.2]]),
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    )
    assert temperatures.shape == (2,)
    assert bool((temperatures > 0).all())
    assert (
        select_proposal_variant(
            one_proposal_value=0.50,
            one_proposal_latency_ms=10.0,
            two_proposal_value=0.509,
            two_proposal_latency_ms=10.0,
        )
        == "one_proposal"
    )
    proposal = torch.ones(2, 7, requires_grad=True)
    disagreement = torch.ones(2, 1, requires_grad=True)
    one = build_proposal_gate_feature(
        proposal,
        disagreement,
        proposal_variant="one_proposal",
    )
    two = build_proposal_gate_feature(
        proposal,
        disagreement,
        proposal_variant="two_proposal",
    )
    assert one.shape == (2, 7) and two.shape == (2, 8)
    assert not one.requires_grad and not two.requires_grad


def _identity(*, suite: str = "spatial") -> CausalStateIdentityV2:
    return CausalStateIdentityV2(
        domain=CausalDomain.CLEAN,
        suite=suite,
        local_task_id=3,
        global_task_uid=f"{suite}:3",
        task_name="test task",
        clean_base_task_uid=f"{suite}:3",
        trial_id=2,
        reset_id=152,
        source_episode_id="episode-2",
        chunk_index=4,
        policy_seed=42,
        model_seed=42,
    )


def _sampling() -> CausalSamplingMetadataV2:
    components = {
        "gripper_transition": 0.1,
        "action_curvature": 0.2,
        "contact_proximity": 0.3,
        "predicate_transition": 0.4,
        "action_precision": 0.5,
    }
    return CausalSamplingMetadataV2(
        source_policy="always_c0",
        source_final_success=False,
        sampling_stratum=CausalSamplingStratum.UNIFORM,
        criticality_components=components,
        criticality_percentiles=components,
        criticality_score=0.35,
        eligible_chunk_count=10,
        conditional_selection_probability=0.1,
        joint_inclusion_probability=0.05,
        phase=CausalPhase.TRANSIT,
    )


def _record(
    mode: CausalComputeMode,
    success: bool,
    replicate: int,
    *,
    continuation_mode: CausalComputeMode = CausalComputeMode.C2_FULL,
):
    from fastwam.causal_prediction import PairedInterventionRecordV2

    latency = {"critical_path": 1.0, "action_dit": 1.0}
    return PairedInterventionRecordV2(
        identity=_identity(),
        sampling=_sampling(),
        intervention=CausalInterventionSpecV2(
            mode=mode,
            control=CausalControlKind.STANDARD,
            treatment_chunks=1,
            continuation_mode=continuation_mode,
            replicate=replicate,
            action_seed=42 + replicate,
            video_seed=100 + replicate if mode.runs_future_prediction else None,
        ),
        outcome=CausalOutcomeV2(
            predicate_before=(False,),
            predicate_after_treatment=(success,),
            predicate_terminal=(success,),
            progress_before=0.0,
            progress_after_treatment=float(success),
            progress_terminal=float(success),
            final_success=success,
            final_return=float(success),
            first_success_step=10 if success else None,
            completion_step=10,
            termination_type=(
                CausalTerminationType.SUCCESS
                if success
                else CausalTerminationType.TIME_LIMIT
            ),
            contact_events={"target": 1},
            treatment_submitted_action_count=1,
            continuation_submitted_action_count=0,
            treatment_action_audit={"status": "PASS"},
            continuation_action_audit={},
        ),
        cost=CausalCostV2(
            treatment_latency_ms=latency,
            continuation_latency_ms={"critical_path": 0.0, "action_dit": 0.0},
            total_latency_ms=latency,
            treatment_calls={"proposal_calls": 1},
            continuation_calls={},
            episode_gpu_seconds=0.001,
        ),
        treatment_submitted_actions=((0.0,) * 7,),
        continuation_submitted_actions=(),
        secondary_outcomes={},
    )


def test_v2_contracts_keep_suite_identity_and_segmented_costs() -> None:
    spatial = _identity(suite="spatial")
    goal = _identity(suite="goal")
    assert spatial.snapshot_id != goal.snapshot_id
    artifact = _record(CausalComputeMode.C2_FULL, True, 0).to_artifact()
    assert artifact["schema"] == "paired-intervention-record-v2"
    assert artifact["identity"]["global_task_uid"] == "spatial:3"
    assert artifact["cost"]["treatment_latency_ms"]["critical_path"] == 1.0
    with pytest.raises(ValueError, match="single-chunk"):
        CausalInterventionSpecV2(
            mode=CausalComputeMode.C2_FULL,
            control=CausalControlKind.NO_READ,
            treatment_chunks=2,
            continuation_mode=CausalComputeMode.C2_FULL,
            replicate=0,
            action_seed=1,
            video_seed=2,
        )


def test_generic_medoid_seed_and_latency_selection_are_deterministic() -> None:
    proposals = torch.tensor(
        [
            [[0.0, 0.0]],
            [[2.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )
    medoid, distances = normalized_action_medoid(proposals)
    assert medoid == 2
    assert distances.shape == (3,)
    assert derive_generic_proposal_seed(7, 0) == 7
    result = select_latency_matched_proposal_count(
        c2_latency_ms=[10.0, 10.0],
        proposal_latency_ms={
            2: [4.0],
            3: [6.0],
            4: [8.0],
            5: [10.0],
            6: [12.0],
            8: [16.0],
        },
    )
    assert result["proposal_count"] == 5


def test_wrong_future_controls_preserve_recipient_prefix() -> None:
    first = torch.zeros(1, 2, 1, 2, 2)
    generated = torch.cat((first, torch.ones(1, 2, 2, 2, 2)), dim=2)
    donor = torch.full((1, 2, 2, 2, 2), 3.0)
    repeated = compose_controlled_video_latents(
        first_frame=first,
        generated_full=generated,
        control=CausalControlKind.REPEAT_CURRENT,
    )
    shuffled = compose_controlled_video_latents(
        first_frame=first,
        generated_full=generated,
        control=CausalControlKind.SHUFFLED_WRONG_STATE,
        donor_future=donor,
    )
    assert torch.equal(repeated[:, :, :1], first)
    assert torch.equal(repeated[:, :, 1:], torch.zeros_like(donor))
    assert torch.equal(shuffled[:, :, :1], first)
    assert torch.equal(shuffled[:, :, 1:], donor)


def test_gate_kv_pooling_and_replicate_label_builder() -> None:
    layers = [
        {"k": torch.ones(1, 3, 2, requires_grad=True) * index, "v": torch.ones(1, 3, 2)}
        for index in range(30)
    ]
    pooled = pool_current_kv_layers(layers, current_token_count=2)
    assert pooled.shape == (1, 30, 4)
    assert not pooled.requires_grad
    feature = CausalGateFeatureRecordV1(
        state=_identity(),
        tensor_shard="gate_features/shard-000.pt",
        tensor_row=0,
        proposal_variant="one_proposal",
        feature_names=(
            "current_video_kv",
            "language",
            "proprio",
            "history",
            "action_proposal",
            "remaining_budget",
            "previous_mode",
            "steps_to_go",
        ),
    )
    records = [
        _record(mode, success, replicate)
        for replicate in range(2)
        for mode, success in (
            (CausalComputeMode.C0_CURRENT, replicate == 1),
            (CausalComputeMode.C2_FULL, True),
        )
    ]
    records.extend(
        _record(
            mode,
            not success,
            replicate,
            continuation_mode=CausalComputeMode.C0_CURRENT,
        )
        for replicate in range(2)
        for mode, success in (
            (CausalComputeMode.C0_CURRENT, replicate == 1),
            (CausalComputeMode.C2_FULL, True),
        )
    )
    example = build_gate_training_example(
        feature=feature,
        records=records,
        modes=(CausalComputeMode.C0_CURRENT, CausalComputeMode.C2_FULL),
        fold=0,
        split="test",
    )
    assert example.outcomes[0].success_mean == 0.5
    assert example.outcomes[0].success_variance == 0.5
    assert example.empirical_uplift["c2_full"] == 0.5
