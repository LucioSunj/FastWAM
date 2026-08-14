from __future__ import annotations

import hashlib

import pytest
import torch

from fastwam.modality_dropout_bc import (
    aggregate_dino_diagnostics,
    baseline_plateau_decision,
    fixed_gaussian_patch_memory,
    forced_modality_keep_mask,
    random_patch_kill_test,
    sample_modality_keep_mask,
    summarize_heldout_losses,
)
from fastwam.modality_dropout_bc_decision import decide_modality_dropout_pilot
from fastwam.models.wan22.visual_contracts import SpatialPatchMemory


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _memory(
    *, batch: int, views: int, patches: int, dimension: int
) -> SpatialPatchMemory:
    valid = torch.ones(batch, views, patches, dtype=torch.bool)
    if views > 1:
        valid[0, -1] = False
    camera_valid = valid.any(dim=-1)
    tokens = torch.ones(batch, views, patches, dimension)
    tokens = tokens * valid.unsqueeze(-1)
    return SpatialPatchMemory(
        tokens=tokens,
        patch_valid_mask=valid,
        camera_valid_mask=camera_valid,
        camera_ids=tuple(f"camera-{index}" for index in range(views)),
        grid=(1, patches),
        patch_size=16,
        backbone_family="test",
        backbone_variant="dynamic",
        native_dim=dimension,
        source_revision="test-revision",
        weights_sha256=_hash("weights"),
        asset_contract_sha256=_hash("asset"),
        input_contract_sha256=_hash("input"),
        preprocess_sha256=_hash("preprocess"),
        output_contract_sha256=_hash("output"),
        memory_contract_sha256=_hash("memory"),
    )


def test_stateless_masks_are_per_sample_independent_and_zero_arm_is_none() -> None:
    identities = [f"sample-{index}" for index in range(512)]
    first = sample_modality_keep_mask(
        sample_identities=identities,
        p_wan=0.5,
        p_dino=0.5,
        seed=42,
        step=17,
        device="cpu",
    )
    reordered = sample_modality_keep_mask(
        sample_identities=list(reversed(identities)),
        p_wan=0.5,
        p_dino=0.5,
        seed=42,
        step=17,
        device="cpu",
    )

    assert first is not None and reordered is not None
    combinations = set(zip(first.wan.tolist(), first.dino.tolist(), strict=True))
    assert combinations == {(False, False), (False, True), (True, False), (True, True)}
    assert torch.equal(first.wan, reordered.wan.flip(0))
    assert torch.equal(first.dino, reordered.dino.flip(0))
    assert not torch.equal(first.wan, first.dino)
    assert (
        sample_modality_keep_mask(
            sample_identities=identities,
            p_wan=0.0,
            p_dino=0.0,
            seed=42,
            step=17,
            device="cpu",
        )
        is None
    )


def test_forced_masks_share_chunk_shape_and_cover_four_conditions() -> None:
    assert forced_modality_keep_mask("clean", batch_size=3, device="cpu") is None
    wan = forced_modality_keep_mask("wan-drop", batch_size=3, device="cpu")
    dino = forced_modality_keep_mask("dino_drop", batch_size=3, device="cpu")
    both = forced_modality_keep_mask("both_drop", batch_size=3, device="cpu")
    assert wan is not None and dino is not None and both is not None
    assert not wan.wan.any() and wan.dino.all()
    assert dino.wan.all() and not dino.dino.any()
    assert not both.wan.any() and not both.dino.any()


@pytest.mark.parametrize(
    ("batch", "views", "patches", "dimension"),
    [(2, 2, 6, 4), (5, 3, 15, 7)],
)
def test_fixed_random_bank_audits_runtime_shape_and_is_shared(
    batch: int,
    views: int,
    patches: int,
    dimension: int,
) -> None:
    memory = _memory(
        batch=batch,
        views=views,
        patches=patches,
        dimension=dimension,
    )
    replaced, metadata = fixed_gaussian_patch_memory(memory)
    repeated, repeated_metadata = fixed_gaussian_patch_memory(memory)

    assert metadata == repeated_metadata
    assert metadata["shape"] == [views, patches, dimension]
    assert len(metadata["sha256"]) == 64
    torch.testing.assert_close(replaced.tokens, repeated.tokens, rtol=0, atol=0)
    for sample in range(1, batch):
        common = memory.patch_valid_mask[0] & memory.patch_valid_mask[sample]
        torch.testing.assert_close(
            replaced.tokens[0][common],
            replaced.tokens[sample][common],
            rtol=0,
            atol=0,
        )
    assert torch.count_nonzero(replaced.tokens[~memory.patch_valid_mask]) == 0
    assert torch.equal(memory.tokens, _memory(
        batch=batch,
        views=views,
        patches=patches,
        dimension=dimension,
    ).tokens)


def test_heldout_summary_plateau_and_random_patch_kill_are_preregistered() -> None:
    clean = torch.ones(64)
    history = [
        {
            "step": 0,
            "losses": {
                "clean": clean,
                "dino_drop": clean * 1.10,
            },
        },
        {
            "step": 500,
            "losses": {
                "clean": clean,
                "dino_drop": clean * 1.15,
            },
        },
        {
            "step": 1000,
            "losses": {
                "clean": clean,
                "dino_drop": clean * 1.30,
            },
        },
        {
            "step": 1500,
            "losses": {
                "clean": clean,
                "dino_drop": clean * 1.30,
            },
        },
    ]
    decision = baseline_plateau_decision(history, draws=200, seed=9)
    assert decision["platform"]
    assert decision["endpoint"] == 1500
    assert decision["reason"] == "regular_plateau"

    conditions = {
        "clean": clean,
        "wan_drop": clean * 1.4,
        "dino_drop": clean * 1.3,
        "both_drop": clean * 1.7,
    }
    summary = summarize_heldout_losses(conditions)
    assert summary["d_dino_loss"] == pytest.approx(0.3)
    assert summary["d_wan_loss"] == pytest.approx(0.4)

    kill = random_patch_kill_test(
        baseline={"clean": clean, "dino_drop": clean * 1.10},
        semantic={"clean": clean, "dino_drop": clean * 1.30},
        random_patch={"clean": clean, "dino_drop": clean * 1.27},
        draws=200,
        seed=11,
    )
    assert kill["outcome"] == "REPRODUCED"


def test_diagnostics_report_gate_projection_residual_and_cross_sample_variance(
) -> None:
    residual = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[3.0, 4.0], [5.0, 6.0]],
        ]
    )
    record = {
        "layer_index": 6,
        "effective_gate": torch.tensor([[0.25], [0.75]]),
        "projected_norm": torch.tensor([[2.0, 4.0], [6.0, 8.0]]),
        "effective_residual_norm": torch.tensor([[1.0, 3.0], [5.0, 7.0]]),
        "effective_residual_sum": residual.sum(dim=0),
        "effective_residual_square_sum": residual.square().sum(dim=0),
        "sample_count": 2,
    }

    summary = aggregate_dino_diagnostics([record])
    overall = summary["overall"]
    assert overall["gate_mean"] == pytest.approx(0.5)
    assert overall["projected_norm"] == pytest.approx(5.0)
    assert overall["residual_norm"] == pytest.approx(4.0)
    assert overall["residual_cross_sample_variance"] == pytest.approx(1.0)


def test_cross_arm_decision_is_evidence_insufficient_without_rollout_and_language(
) -> None:
    clean = [1.0] * 256
    diagnostics = {
        "overall": {
            "gate_mean": 0.5,
            "projected_norm": 1.0,
            "residual_norm": 0.2,
            "residual_cross_sample_variance": 0.03,
        }
    }
    evidence = {}
    for arm, dino in {
        "A": 1.10,
        "B15": 1.12,
        "B30": 1.30,
        "B50": 1.35,
        "C": 1.15,
        "D": 1.20,
    }.items():
        losses = {
            "clean": clean,
            "wan_drop": [1.4] * 256,
            "dino_drop": [dino] * 256,
            "both_drop": [1.7] * 256,
        }
        result = {
            "status": "COMPLETE",
            "arm": {"name": arm},
            "global_step": 1500,
            "initial_trainable_sha256": {"lora": "a", "reader": "b"},
            "initial_frozen_parent_and_dino_sha256": "frozen",
            "final_frozen_parent_and_dino_sha256": "frozen",
            "frozen_parameter_versions_unchanged": True,
            "heldout_indices": list(range(256)),
            "heldout_split_contract": "episode-disjoint-from-training",
            "endpoint": {"status": "PLATFORMED"},
        }
        evidence[arm] = {
            "result": result,
            "step_zero": {
                "arm": arm,
                "step": 0,
                "losses": losses,
                "diagnostics": diagnostics,
            },
            "final": {
                "arm": arm,
                "step": 1500,
                "loss": {"clean": 1.0},
                "d_dino_loss": dino - 1.0,
                "d_wan_loss": 0.4,
                "losses": losses,
                "diagnostics": diagnostics,
            },
        }

    result = decide_modality_dropout_pilot(evidence, bootstrap_draws=100, seed=5)
    assert result["decision"] == "EVIDENCE_INSUFFICIENT"
    assert result["audit"]["passed"]
    assert result["rollout"] == {"status": "NOT-RUN", "arms": {}}
    assert result["language_canary"] == "NOT-RUN"
    assert len(result["table"]) == 6
