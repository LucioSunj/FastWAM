"""CPU checks for the required dual-regime training infrastructure."""
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from fastwam.adaptive_gate.training import (  # noqa: E402
    advance_successful_optimizer_steps,
    build_optimizer_parameter_groups,
    canonicalize_uncond_weight_schedule,
    classify_training_resume_source,
    raw_loss_gradient_statistics,
    uncond_weight_at_step,
    validate_dual_regime_trainer_state,
)
from fastwam.adaptive_gate.warm_start import (  # noqa: E402
    sha256_file,
    strict_standalone_idm_warm_start,
    warm_start_is_enabled,
)


def test_adaptive_resume_requires_full_state_directory(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"weights")

    assert (
        classify_training_resume_source(state_dir, is_dual_regime=True)
        == "full_state"
    )
    assert (
        classify_training_resume_source(weights, is_dual_regime=False)
        == "weights_only"
    )
    with pytest.raises(ValueError, match="weights-only file"):
        classify_training_resume_source(weights, is_dual_regime=True)
    with pytest.raises(FileNotFoundError, match="Resume checkpoint not found"):
        classify_training_resume_source(
            tmp_path / "missing.pt", is_dual_regime=True
        )


def test_adaptive_trainer_state_binds_stats_contract_and_successful_step():
    contract = {
        "uncond_weight_schedule": [[0.0, 0.05], [1.0, 1.0]],
        "total_optimizer_steps": 100,
    }
    stats_sha = "a" * 64
    payload = {
        "global_step": 37,
        "dual_regime_optimizer_steps": 37,
        "dual_regime_training_contract": contract,
        "dataset_stats_fingerprint": stats_sha,
    }
    assert validate_dual_regime_trainer_state(
        payload,
        expected_contract=contract,
        expected_dataset_stats_fingerprint=stats_sha,
    ) == 37

    wrong_stats = dict(payload, dataset_stats_fingerprint="b" * 64)
    with pytest.raises(ValueError, match="dataset-stats SHA"):
        validate_dual_regime_trainer_state(
            wrong_stats,
            expected_contract=contract,
            expected_dataset_stats_fingerprint=stats_sha,
        )
    mismatched_clock = dict(payload, global_step=38)
    with pytest.raises(ValueError, match="global_step must equal"):
        validate_dual_regime_trainer_state(
            mismatched_clock,
            expected_contract=contract,
            expected_dataset_stats_fingerprint=stats_sha,
        )
    with pytest.raises(ValueError, match="contract does not match"):
        validate_dual_regime_trainer_state(
            payload,
            expected_contract={**contract, "total_optimizer_steps": 101},
            expected_dataset_stats_fingerprint=stats_sha,
        )


def test_skipped_optimizer_updates_do_not_advance_schedule_clock():
    global_step = 0
    dual_steps = 0
    outcomes = iter([True, False, True, False, False])
    attempts = 0
    while global_step < 3:
        global_step, dual_steps = advance_successful_optimizer_steps(
            global_step=global_step,
            dual_regime_optimizer_steps=dual_steps,
            is_dual_regime=True,
            optimizer_step_was_skipped=next(outcomes),
        )
        attempts += 1

    assert attempts == 5
    assert global_step == dual_steps == 3
    schedule = canonicalize_uncond_weight_schedule([(0.0, 0.05), (1.0, 1.0)])
    assert uncond_weight_at_step(
        schedule, optimizer_step=dual_steps, total_optimizer_steps=3
    ) == 1.0


def test_piecewise_uncond_schedule_is_optimizer_step_deterministic():
    schedule = canonicalize_uncond_weight_schedule(
        [
            {"fraction": 0.0, "weight": 0.05},
            {"fraction": 0.1, "weight": 0.05},
            {"fraction": 0.4, "weight": 0.5},
            {"fraction": 1.0, "weight": 1.0},
        ]
    )
    assert uncond_weight_at_step(schedule, optimizer_step=0, total_optimizer_steps=100) == 0.05
    assert uncond_weight_at_step(schedule, optimizer_step=10, total_optimizer_steps=100) == 0.05
    assert uncond_weight_at_step(schedule, optimizer_step=25, total_optimizer_steps=100) == pytest.approx(0.275)
    assert uncond_weight_at_step(schedule, optimizer_step=40, total_optimizer_steps=100) == 0.5
    assert uncond_weight_at_step(schedule, optimizer_step=100, total_optimizer_steps=100) == 1.0

    with pytest.raises(ValueError, match="start at 0.0"):
        canonicalize_uncond_weight_schedule([(0.1, 0.1), (1.0, 1.0)])
    with pytest.raises(ValueError, match="strictly increasing"):
        canonicalize_uncond_weight_schedule([(0.0, 0.1), (0.0, 0.2), (1.0, 1.0)])
    with pytest.raises(ValueError, match="> 0"):
        canonicalize_uncond_weight_schedule([(0.0, 0.0), (1.0, 1.0)])


def test_raw_gradient_statistics_do_not_pollute_training_gradients():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    auxiliary = torch.nn.Parameter(torch.tensor([3.0]))
    main = parameter.square().sum() + auxiliary.square().sum()
    uncond = -parameter.square().sum() + auxiliary.square().sum()
    stats = raw_loss_gradient_statistics(
        main,
        uncond,
        {"opposed": [parameter], "mixed": [parameter, auxiliary]},
    )
    dot, main_sq, uncond_sq, used, count = stats["opposed"]
    assert float(dot / torch.sqrt(main_sq * uncond_sq)) == pytest.approx(-1.0)
    assert used.item() == 1
    assert count.item() == 1
    assert parameter.grad is None
    assert auxiliary.grad is None

    (main + uncond).backward()
    assert parameter.grad is not None


class _TinyGroupedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action_expert = torch.nn.Linear(3, 2)
        self.video_expert = torch.nn.Linear(3, 2)
        self.proprio_encoder = torch.nn.Linear(2, 2)
        self.mot = torch.nn.ModuleDict(
            {"action": self.action_expert, "video": self.video_expert}
        )
        self.dit = self.mot


def test_optimizer_groups_are_named_scaled_and_identity_deduplicated():
    model = _TinyGroupedModel()
    groups = build_optimizer_parameter_groups(
        model,
        base_learning_rate=2e-5,
        action_lr_scale=1.0,
        proprio_lr_scale=0.5,
        video_lr_scale=0.1,
    )
    assert [group["name"] for group in groups] == ["action", "proprio", "video"]
    assert [group["lr"] for group in groups] == pytest.approx([2e-5, 1e-5, 2e-6])
    ids = [id(param) for group in groups for param in group["params"]]
    assert len(ids) == len(set(ids))

    frozen = _TinyGroupedModel()
    frozen_groups = build_optimizer_parameter_groups(
        frozen,
        base_learning_rate=2e-5,
        action_lr_scale=1.0,
        proprio_lr_scale=1.0,
        video_lr_scale=0.0,
    )
    assert [group["name"] for group in frozen_groups] == ["action", "proprio"]
    assert not any(param.requires_grad for param in frozen.video_expert.parameters())

    suffix_model = _TinyGroupedModel()
    suffix_model.video_expert = torch.nn.Module()
    suffix_model.video_expert.blocks = torch.nn.ModuleList(
        [torch.nn.Linear(3, 3) for _ in range(4)]
    )
    suffix_model.mot["video"] = suffix_model.video_expert
    suffix_groups = build_optimizer_parameter_groups(
        suffix_model,
        base_learning_rate=2e-5,
        action_lr_scale=1.0,
        proprio_lr_scale=1.0,
        video_lr_scale=0.1,
        video_final_blocks=2,
    )
    video_group = next(group for group in suffix_groups if group["name"] == "video")
    expected = {
        id(param)
        for block in suffix_model.video_expert.blocks[-2:]
        for param in block.parameters()
    }
    assert {id(param) for param in video_group["params"]} == expected
    assert not any(
        param.requires_grad
        for block in suffix_model.video_expert.blocks[:2]
        for param in block.parameters()
    )


class _VideoExpert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(2, 2)
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        self.video_attention_mask_mode = "first_frame_causal"
        self.action_conditioned = False
        self.patch_size = (1, 2, 2)


class _ActionExpert(torch.nn.Module):
    action_dim = 2
    hidden_dim = 2
    num_heads = 1
    attn_head_dim = 2

    def __init__(self):
        super().__init__()
        self.action_encoder = torch.nn.Linear(2, 2)
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2)])


class _VAEModel:
    z_dim = 4


class _VAE:
    model = _VAEModel()


class _WarmTargetBase(torch.nn.Module):
    adaptive_regimes = ("uncond", "idm")
    adaptive_backbone_kind = "idm"

    def __init__(self):
        super().__init__()
        self.video_expert = _VideoExpert()
        self.action_expert = _ActionExpert()
        self.mot = torch.nn.ModuleDict(
            {"video": self.video_expert, "action": self.action_expert}
        )
        self.proprio_encoder = torch.nn.Linear(2, 2)
        self.proprio_dim = 2
        self.vae = _VAE()
        self.checkpoint_task = "libero_dual_regime_fused_2cam224_1e-4"


FusedDualRegimeFastWAM = type(
    "FusedDualRegimeFastWAM", (_WarmTargetBase,), {}
)


def _architecture_config(target: str, task: str) -> dict:
    return {
        "_target_": target,
        "checkpoint_task": task,
        "model_id": "wan-test",
        "tokenizer_model_id": "text-test",
        "tokenizer_max_len": 8,
        "proprio_dim": 2,
        "video_dit_config": {"hidden_dim": 2},
        "action_dit_config": {"hidden_dim": 2},
        "video_scheduler": {"train_shift": 5},
        "action_scheduler": {"train_shift": 5},
    }


def test_strict_standalone_idm_warm_start_imports_only_verified_weights(tmp_path):
    from omegaconf import OmegaConf
    from fastwam.adaptive_gate.provenance import checkpoint_model_contract

    target = FusedDualRegimeFastWAM()
    source_mot = copy.deepcopy(target.mot.state_dict())
    source_proprio = copy.deepcopy(target.proprio_encoder.state_dict())
    for tensor in source_mot.values():
        tensor.fill_(0.25)
    for tensor in source_proprio.values():
        tensor.fill_(-0.5)

    stats_path = tmp_path / "dataset_stats.json"
    stats_path.write_text('{"action": {"mean": [0, 0]}}', encoding="utf-8")
    source_task = "libero_idm_2cam224_1e-4"
    source_config = _architecture_config(
        "fastwam.runtime.create_fastwam_idm", source_task
    )
    source_config_path = tmp_path / "source_config.yaml"
    OmegaConf.save(OmegaConf.create({"model": source_config}), source_config_path)
    checkpoint_path = tmp_path / "standalone_idm.pt"
    torch.save(
        {
            "mot": source_mot,
            "proprio_encoder": source_proprio,
            "optimizer": {"must": "not be restored"},
            "step": 77,
            "fastwam_provenance": {
                "schema_version": 2,
                "checkpoint_id": "parent-id",
                "model_class": "FastWAMIDM",
                "adaptive_regimes": [],
                "adaptive_backbone_kind": None,
                "task": source_task,
                "action_dim": 2,
                "proprio_dim": 2,
                "video_latent_dim": 4,
                "dataset_stats_fingerprint": sha256_file(stats_path),
                "model_contract": checkpoint_model_contract(target),
            },
        },
        checkpoint_path,
    )
    warm_config = {
        "kind": "standalone_idm",
        "checkpoint": str(checkpoint_path),
        "expected_checkpoint_sha256": sha256_file(checkpoint_path),
        "source_task": source_task,
        "source_config": str(source_config_path),
        "source_dataset_stats": str(stats_path),
    }
    target_config = _architecture_config("unused.target", target.checkpoint_task)
    record = strict_standalone_idm_warm_start(
        target,
        warm_config,
        target_model_config=target_config,
        target_dataset_stats=stats_path,
    )
    assert record["parent_checkpoint_id"] == "parent-id"
    assert record["parent_step"] == 77
    assert target.dual_regime_optimizer_steps == 0
    assert target.mot.state_dict().keys() == source_mot.keys()
    for key, tensor in target.mot.state_dict().items():
        torch.testing.assert_close(tensor, source_mot[key])
    for key, tensor in target.proprio_encoder.state_dict().items():
        torch.testing.assert_close(tensor, source_proprio[key])

    legacy_payload = torch.load(checkpoint_path, weights_only=False)
    legacy_payload["fastwam_provenance"]["task"] = None
    legacy_payload["fastwam_provenance"].pop("model_contract")
    legacy_path = tmp_path / "standalone_idm_pre_contract.pt"
    torch.save(legacy_payload, legacy_path)
    legacy_config = dict(
        warm_config,
        checkpoint=str(legacy_path),
        expected_checkpoint_sha256=sha256_file(legacy_path),
    )
    legacy_record = strict_standalone_idm_warm_start(
        FusedDualRegimeFastWAM(),
        legacy_config,
        target_model_config=target_config,
        target_dataset_stats=stats_path,
    )
    assert legacy_record["task_binding"] == "explicit_source_task_and_hashed_config"
    assert legacy_record["model_contract_binding"] == "hashed_config_and_strict_state_schema"

    # Released/previously trained standalone checkpoints may predate all
    # FastWAM provenance. The explicit source config/stats/SHA plus exact state
    # schema still form a strict, auditable import contract.
    pre_provenance_payload = torch.load(checkpoint_path, weights_only=False)
    pre_provenance_payload.pop("fastwam_provenance")
    pre_provenance_path = tmp_path / "standalone_idm_pre_provenance.pt"
    torch.save(pre_provenance_payload, pre_provenance_path)
    pre_provenance_config = dict(
        warm_config,
        checkpoint=str(pre_provenance_path),
        expected_checkpoint_sha256=sha256_file(pre_provenance_path),
    )
    pre_provenance_record = strict_standalone_idm_warm_start(
        FusedDualRegimeFastWAM(),
        pre_provenance_config,
        target_model_config=target_config,
        target_dataset_stats=stats_path,
    )
    assert not pre_provenance_record["source_checkpoint_provenance_present"]
    assert pre_provenance_record["parent_checkpoint_id"].startswith("standalone-idm-")
    assert pre_provenance_record["dataset_stats_binding"] == (
        "explicit_hashed_source_and_target_artifacts"
    )

    bad = dict(warm_config, expected_checkpoint_sha256="0" * 64)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        strict_standalone_idm_warm_start(
            FusedDualRegimeFastWAM(),
            bad,
            target_model_config=target_config,
            target_dataset_stats=stats_path,
        )


def test_warm_start_requires_explicit_kind_and_no_ignored_fields():
    assert not warm_start_is_enabled(
        {
            "kind": None,
            "checkpoint": None,
            "expected_checkpoint_sha256": None,
            "source_task": None,
            "source_config": None,
            "source_dataset_stats": None,
        }
    )
    with pytest.raises(ValueError, match="kind is null"):
        warm_start_is_enabled({"kind": None, "checkpoint": "weights.pt"})
