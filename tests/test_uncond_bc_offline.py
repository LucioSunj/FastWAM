from pathlib import Path
from unittest.mock import Mock

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import fastwam.uncond_bc_offline as offline
from fastwam.uncond_bc_offline import (
    _validate_offline_config,
    _validate_sidecar_extra,
    claim_uncond_bc_offline_output,
)


def _config():
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name="uncond_bc_eval")


def test_offline_preset_is_isolated_and_requires_four_gpu_profile() -> None:
    cfg = _config()

    _validate_offline_config(cfg, world_size=4)

    assert cfg.runner.policy == "zero_lora"
    assert cfg.runner.sidecar is None
    assert cfg.training.microbatch_size == 8
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert "gate" not in resolved
    assert "critic" not in resolved
    assert "value_head" not in resolved
    with pytest.raises(ValueError, match="exactly 4"):
        _validate_offline_config(cfg, world_size=1)


def test_hostb_offline_preset_uses_gloo_without_loader_children() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="uncond_bc_eval",
            overrides=["task=libero_uncond_lora_bc_hostb_offline"],
        )

    _validate_offline_config(cfg, world_size=4)

    assert cfg.training.distributed_backend == "gloo_manual"
    assert cfg.data.num_workers == 0
    assert cfg.data.prefetch_factor == 1
    assert cfg.data.multiprocessing_context is None
    assert cfg.data.persistent_workers is False


def test_offline_policy_and_sidecar_pair_fail_closed() -> None:
    cfg = _config()
    cfg.runner.policy = "bc_lora"
    with pytest.raises(ValueError, match="requires sidecar"):
        _validate_offline_config(cfg, world_size=4)

    cfg.runner.sidecar = "/tmp/sidecar.pt"
    cfg.runner.sidecar_sha256 = "a" * 64
    _validate_offline_config(cfg, world_size=4)

    cfg.runner.policy = "zero_lora"
    with pytest.raises(ValueError, match="forbids"):
        _validate_offline_config(cfg, world_size=4)


def test_offline_rank32_training_checkpoint_uses_six_gpus_and_zero_training() -> None:
    cfg = _config()
    cfg.lora.rank = 32
    cfg.runner.policy = "bc_training_checkpoint"
    cfg.runner.training_checkpoint = "/tmp/step_001000.pt"
    cfg.runner.training_checkpoint_sha256 = "b" * 64
    cfg.runner.training_checkpoint_step = 1000

    _validate_offline_config(cfg, world_size=6)
    with pytest.raises(ValueError, match="exactly 6"):
        _validate_offline_config(cfg, world_size=4)

    cfg.runner.sidecar = "/tmp/rejected.pt"
    cfg.runner.sidecar_sha256 = "c" * 64
    with pytest.raises(ValueError, match="forbids a sidecar"):
        _validate_offline_config(cfg, world_size=6)


def test_offline_asset_and_normalization_contracts_fail_closed() -> None:
    cfg = _config()
    cfg.parent.checkpoint_sha256 = "f" * 64
    with pytest.raises(ValueError, match="parent path/hash"):
        _validate_offline_config(cfg, world_size=4)

    cfg = _config()
    cfg.data.validation.processor.norm_default_mode = "mean/std"
    with pytest.raises(ValueError, match="normalization contract"):
        _validate_offline_config(cfg, world_size=4)


def test_sidecar_data_provenance_is_strict() -> None:
    contract = {
        "statistics_sha256": "a" * 64,
        "dataset_sha256": {"suite": "b" * 64},
        "text_cache_sha256": "c" * 64,
    }
    extra = {
        "bc_step": 1000,
        "bc_config_sha256": "d" * 64,
        "validation_loss_action_bc": 0.25,
        **contract,
    }

    assert _validate_sidecar_extra(extra, contract=contract)["bc_step"] == 1000

    malformed = dict(extra)
    malformed["statistics_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="data provenance mismatch"):
        _validate_sidecar_extra(malformed, contract=contract)

    malformed = dict(extra)
    malformed["unexpected"] = True
    with pytest.raises(ValueError, match="provenance keys changed"):
        _validate_sidecar_extra(malformed, contract=contract)


def test_offline_output_claim_refuses_existing_content(tmp_path) -> None:
    cfg = _config()
    cfg.runner.output_dir = str(tmp_path / "accepted")
    output = claim_uncond_bc_offline_output(cfg)
    assert (output / ".fastwam-uncond-bc-offline-output-v1").is_file()

    cfg.runner.output_dir = str(tmp_path / "occupied")
    occupied = Path(cfg.runner.output_dir)
    occupied.mkdir()
    (occupied / "user-file").write_text("preserve")
    with pytest.raises(FileExistsError, match="not empty"):
        claim_uncond_bc_offline_output(cfg)


def test_gloo_offline_validation_skips_cuda_ddp_parameter_sync(monkeypatch) -> None:
    policy = Mock()
    ddp = Mock()
    monkeypatch.setattr(offline, "DistributedDataParallel", ddp)

    observed = offline._offline_evaluation_model(
        policy,
        local_rank=0,
        distributed_backend="gloo_manual",
    )

    assert observed is policy
    ddp.assert_not_called()
