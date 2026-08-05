import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

from fastwam.uncond_bc_trainer import (
    DistributedEvalSampler,
    _canonical_config,
    _dataset_summary,
    _distributed_context,
    _instantiate_bc_dataset,
    _prune_training_checkpoints,
    _save_checkpoint,
    _validate_training_config,
    _write_resolved_config,
    _write_run_manifest,
    claim_uncond_bc_output,
    load_strict_fastwam_parent,
    record_uncond_bc_failure,
)


def _compose_config():
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name="uncond_bc")


def test_uncond_bc_hydra_preset_is_isolated_and_parent_bound() -> None:
    cfg = _compose_config()
    resolved = OmegaConf.to_container(cfg, resolve=True)

    assert cfg.parent.checkpoint.endswith(
        "fastwam-idm-libero-wan-robot-init-step_021700.pt"
    )
    assert cfg.parent.checkpoint_sha256 == (
        "e979511a2d7a1310009496c6b2f06957171bba28b96aac0d513992c6ed21ca5a"
    )
    assert cfg.parent.statistics_sha256 == (
        "30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638"
    )
    assert len(cfg.data.train.dataset_dirs) == 4
    assert cfg.data.train.val_set_proportion == 0.1
    assert cfg.data.train.is_training_set is True
    assert cfg.data.validation.is_training_set is False
    assert cfg.data.expected_train_episodes == 1539
    assert cfg.data.expected_validation_episodes == 173
    assert cfg.data.expected_source_episodes == 1712
    assert cfg.data.expected_source_transitions == 277713
    assert cfg.runner.stage == "formal"
    assert cfg.lora.rank == 16
    assert cfg.lora.alpha == 16.0
    assert cfg.lora.dropout == 0.0
    assert cfg.training.global_batch_size == 128
    assert cfg.training.deterministic_algorithms is True
    assert cfg.training.stateless_flow_inputs is True
    assert cfg.training.checkpoint_keep_last == 1
    assert cfg.model.load_text_encoder is False
    assert cfg.model.skip_dit_load_from_pretrain is True
    assert "gate" not in resolved
    assert "critic" not in resolved
    assert "value_head" not in resolved
    assert "loss_video" not in resolved


def test_training_config_enforces_4gpu_capacity_ladder_and_allows_bc0() -> None:
    cfg = _compose_config()
    _validate_training_config(cfg, world_size=4)

    for microbatch, accumulation in ((1, 32), (2, 16), (4, 8), (8, 4)):
        candidate = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        candidate.training.microbatch_size = microbatch
        candidate.training.gradient_accumulation_steps = accumulation
        _validate_training_config(candidate, world_size=4)

    invalid = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    invalid.training.microbatch_size = 8
    invalid.training.gradient_accumulation_steps = 8
    with pytest.raises(ValueError, match="requires accumulation"):
        _validate_training_config(invalid, world_size=4)

    bc0 = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    bc0.runner.stage = "bc0"
    _validate_training_config(bc0, world_size=1)


def test_distributed_context_binds_local_cuda_device_before_nccl(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "2")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: calls.append(device))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(
        torch.distributed,
        "init_process_group",
        lambda **kwargs: calls.append(kwargs),
    )

    rank, world_size, local_rank, device = _distributed_context()

    assert (rank, world_size, local_rank) == (2, 4, 2)
    assert device == torch.device("cuda", 2)
    assert calls == [device, {"backend": "nccl", "device_id": device}]


def test_resume_path_is_excluded_from_strict_training_contract_hash() -> None:
    cfg = _compose_config()
    _, launch_hash, contract_hash = _canonical_config(cfg)
    resumed = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    resumed.runner.resume = "/checkpoint/step_000500.pt"
    resumed.runner.stop_after_steps = 1
    _, resumed_launch_hash, resumed_contract_hash = _canonical_config(resumed)

    assert launch_hash != resumed_launch_hash
    assert contract_hash == resumed_contract_hash

    relocated = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    relocated.runner.output_dir = "/a/different/audit/directory"
    _, relocated_launch_hash, relocated_contract_hash = _canonical_config(relocated)
    assert launch_hash != relocated_launch_hash
    assert contract_hash == relocated_contract_hash

    changed_retention = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    changed_retention.training.checkpoint_keep_last = 7
    _, retention_launch_hash, retention_contract_hash = _canonical_config(
        changed_retention
    )
    assert launch_hash != retention_launch_hash
    assert contract_hash == retention_contract_hash


class _Actor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mot = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 3))
        self.proprio_encoder = nn.Linear(2, 3)


def test_strict_parent_loader_restores_exact_keys_and_rejects_mismatch(
    tmp_path,
) -> None:
    torch.manual_seed(5)
    source = _Actor()
    checkpoint = tmp_path / "parent.pt"
    torch.save(
        {
            "mot": source.mot.state_dict(),
            "proprio_encoder": source.proprio_encoder.state_dict(),
            "step": 21700,
            "torch_dtype": "torch.bfloat16",
        },
        checkpoint,
    )
    torch.manual_seed(9)
    restored = _Actor()

    report = load_strict_fastwam_parent(restored, str(checkpoint))

    assert report["parent_step"] == 21700
    for name, value in source.state_dict().items():
        assert torch.equal(restored.state_dict()[name], value)

    malformed = torch.load(checkpoint, weights_only=False)
    malformed["mot"].pop("0.bias")
    malformed_path = tmp_path / "malformed.pt"
    torch.save(malformed, malformed_path)
    with pytest.raises(ValueError, match="MoT key mismatch"):
        load_strict_fastwam_parent(_Actor(), str(malformed_path))


def test_distributed_validation_sampler_has_no_padding_or_duplicates() -> None:
    dataset = list(range(11))
    shards = [
        list(DistributedEvalSampler(dataset, rank=rank, world_size=4))
        for rank in range(4)
    ]
    flattened = [index for shard in shards for index in shard]

    assert sorted(flattened) == list(range(11))
    assert len(flattened) == len(set(flattened))


def _resolved_copy(cfg):
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def test_training_phases_are_explicit_and_world_size_bound() -> None:
    base = _compose_config()

    bc1 = _resolved_copy(base)
    bc1.runner.stage = "bc1"
    bc1.runner.single_gpu_diagnostic = True
    bc1.runner.stop_after_steps = 1
    bc1.training.max_steps = 2
    _validate_training_config(bc1, world_size=1)
    with pytest.raises(ValueError, match="exactly one GPU"):
        _validate_training_config(bc1, world_size=4)

    bc2 = _resolved_copy(base)
    bc2.runner.stage = "bc2"
    bc2.runner.stop_after_steps = 2
    bc2.training.max_steps = 2
    _validate_training_config(bc2, world_size=4)

    pilot = _resolved_copy(base)
    pilot.runner.stage = "pilot"
    pilot.training.max_steps = 1000
    _validate_training_config(pilot, world_size=4)

    malformed = _resolved_copy(base)
    malformed.lora.rank = 8
    with pytest.raises(ValueError, match="LoRA structure"):
        _validate_training_config(malformed, world_size=4)


def test_split_seed_is_explicit_and_fails_before_dataset_construction() -> None:
    cfg = _compose_config()
    malformed = _resolved_copy(cfg.data.train)
    malformed.split_seed = 7

    with pytest.raises(ValueError, match="episode split seed mismatch"):
        _instantiate_bc_dataset(malformed, expected_seed=42)


class _SummaryDataset:
    def __init__(self) -> None:
        multi = SimpleNamespace(
            ds_names=(
                "/data/libero_spatial_no_noops_lerobot",
                "/data/libero_10_no_noops_lerobot",
            ),
            _datasets=(
                SimpleNamespace(num_episodes=434, num_frames=53229),
                SimpleNamespace(num_episodes=388, num_frames=104280),
            ),
            num_episodes=822,
        )
        self.dataset = SimpleNamespace(
            lerobot_dataset=SimpleNamespace(multi_dataset=multi)
        )

    def __len__(self) -> int:
        return 157509


def test_dataset_summary_records_per_suite_episode_and_window_counts() -> None:
    summary = _dataset_summary(_SummaryDataset())

    assert summary["episodes"] == 822
    assert summary["windows"] == 157509
    assert summary["suites"] == [
        {
            "dataset_dir": "/data/libero_spatial_no_noops_lerobot",
            "suite": "libero_spatial_no_noops_lerobot",
            "episodes": 434,
            "windows": 53229,
        },
        {
            "dataset_dir": "/data/libero_10_no_noops_lerobot",
            "suite": "libero_10_no_noops_lerobot",
            "episodes": 388,
            "windows": 104280,
        },
    ]


def test_output_claim_and_failure_manifest_are_fail_closed(tmp_path) -> None:
    cfg = _compose_config()
    cfg.runner.output_dir = str(tmp_path / "owned")
    claimed = claim_uncond_bc_output(cfg)

    assert claim_uncond_bc_output(cfg) == claimed
    assert (claimed / ".fastwam-uncond-bc-output-v1").read_text() == (
        "fastwam-uncond-bc-output-v1\n"
    )
    failure = record_uncond_bc_failure(
        cfg,
        RuntimeError("synthetic failure"),
        traceback_text="synthetic traceback",
    )
    payload = json.loads(failure.read_text())
    assert payload["status"] == "FAIL"
    assert payload["exit_status"] == 1
    assert payload["error_type"] == "RuntimeError"
    assert payload["error"] == "synthetic failure"
    assert payload["declared_parent"]["sha256"] == cfg.parent.checkpoint_sha256

    unowned = tmp_path / "unowned"
    unowned.mkdir()
    (unowned / "user-data").write_text("keep")
    cfg.runner.output_dir = str(unowned)
    with pytest.raises(FileExistsError, match="not empty"):
        claim_uncond_bc_output(cfg)
    assert not (unowned / ".fastwam-uncond-bc-output-v1").exists()


def test_checkpoint_retention_prunes_only_older_known_training_state(
    tmp_path,
) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    older_step = checkpoints / "step_000500.pt"
    older_epoch = checkpoints / "epoch_01_step_001943.pt"
    latest = checkpoints / "step_002000.pt"
    unrelated = checkpoints / "do_not_delete.pt"
    for path in (older_step, older_epoch, latest, unrelated):
        path.write_bytes(path.name.encode())

    pruned = _prune_training_checkpoints(
        checkpoints,
        keep_path=latest,
        keep_last=1,
    )

    assert pruned == [older_epoch.name, older_step.name]
    assert latest.read_bytes() == latest.name.encode()
    assert unrelated.read_bytes() == unrelated.name.encode()
    assert not older_step.exists()
    assert not older_epoch.exists()
    with pytest.raises(ValueError, match="exactly one"):
        _prune_training_checkpoints(
            checkpoints,
            keep_path=latest,
            keep_last=2,
        )


def _patch_checkpoint_dependencies(
    monkeypatch,
    *,
    save,
    inspect,
    prune,
    barrier,
) -> None:
    monkeypatch.setattr(
        "fastwam.uncond_bc_trainer._gather_rng",
        lambda *, world_size: [{"rank": rank} for rank in range(world_size)],
    )
    monkeypatch.setattr(
        "fastwam.uncond_bc_trainer.save_uncond_bc_checkpoint",
        save,
    )
    monkeypatch.setattr(
        "fastwam.uncond_bc_trainer.inspect_uncond_bc_checkpoint",
        inspect,
    )
    monkeypatch.setattr(
        "fastwam.uncond_bc_trainer._prune_training_checkpoints",
        prune,
    )
    monkeypatch.setattr("fastwam.uncond_bc_trainer._barrier", barrier)


def _call_mock_checkpoint(path: Path):
    token = object()
    return _save_checkpoint(
        path,
        policy=SimpleNamespace(lora_adapter=token),
        optimizer=token,
        scheduler=token,
        scaler=token,
        global_step=500,
        epoch=0,
        sampler_offset=2000,
        contract={},
        provenance={},
        trainer_state={},
        parent_sha256="a" * 64,
        rank=0,
        world_size=1,
        checkpoint_keep_last=1,
    )


def test_save_checkpoint_inspects_before_pruning(monkeypatch, tmp_path) -> None:
    target = tmp_path / "checkpoints" / "step_000500.pt"
    events = []

    def save(path, **kwargs):
        del kwargs
        events.append("save")
        Path(path).parent.mkdir()
        Path(path).write_bytes(b"complete")

    def inspect(path):
        assert Path(path).read_bytes() == b"complete"
        events.append("inspect")
        return {"result": "PASS", "global_step": 500}

    def prune(directory, *, keep_path, keep_last):
        assert events == ["save", "inspect"]
        assert Path(directory) == target.parent
        assert Path(keep_path) == target
        assert keep_last == 1
        events.append("prune")
        return ["step_000001.pt"]

    def barrier(world_size):
        assert world_size == 1
        events.append("barrier")

    _patch_checkpoint_dependencies(
        monkeypatch,
        save=save,
        inspect=inspect,
        prune=prune,
        barrier=barrier,
    )

    report = _call_mock_checkpoint(target)

    assert events == ["save", "inspect", "prune", "barrier"]
    assert report["checkpoint_retention"] == {
        "keep_last": 1,
        "kept": target.name,
        "pruned": ["step_000001.pt"],
    }


def test_failed_checkpoint_inspection_retains_older_state(
    monkeypatch,
    tmp_path,
) -> None:
    target = tmp_path / "checkpoints" / "step_000500.pt"
    older = tmp_path / "checkpoints" / "step_000001.pt"
    older.parent.mkdir()
    older.write_bytes(b"recoverable")

    def save(path, **kwargs):
        del kwargs
        Path(path).write_bytes(b"malformed")

    _patch_checkpoint_dependencies(
        monkeypatch,
        save=save,
        inspect=lambda path: {"result": "FAIL", "path": str(path)},
        prune=lambda *args, **kwargs: pytest.fail("prune must not run"),
        barrier=lambda world_size: pytest.fail("barrier must not run"),
    )

    with pytest.raises(RuntimeError, match="older state was retained"):
        _call_mock_checkpoint(target)
    assert older.read_bytes() == b"recoverable"


def test_success_artifacts_preserve_each_resume_launch(tmp_path) -> None:
    first_config = _write_resolved_config(
        tmp_path,
        launch_hash="a" * 64,
        value="runner:\n  stop_after_steps: 1\n",
    )
    second_config = _write_resolved_config(
        tmp_path,
        launch_hash="b" * 64,
        value="runner:\n  stop_after_steps: 2\n",
    )
    first_manifest = _write_run_manifest(
        tmp_path,
        {"stage": "bc1", "optimizer_steps": 1, "status": "PASS"},
        launch_hash="a" * 64,
    )
    second_manifest = _write_run_manifest(
        tmp_path,
        {"stage": "bc1", "optimizer_steps": 2, "status": "PASS"},
        launch_hash="b" * 64,
    )

    assert first_config.is_file()
    assert second_config.is_file()
    assert first_manifest.is_file()
    assert second_manifest.is_file()
    latest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert latest["optimizer_steps"] == 2
    assert "stop_after_steps: 2" in (tmp_path / "resolved_config.yaml").read_text()
