from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import fastwam.causal_prediction_trainer as trainer
import fastwam.uncond_bc_trainer as bc_trainer


class _Policy:
    dtype = torch.float32

    def __init__(self) -> None:
        self.actor = SimpleNamespace(train_action_scheduler=object())
        self.evaluated_batches = 0

    def eval(self) -> None:
        return None

    def evaluate_both_modes(self, batch, *, timestep, noise):
        del timestep, noise
        self.evaluated_batches += 1
        assert batch["action"].shape[0] == 2
        return {
            "loss_c0_action": torch.tensor(1.0),
            "loss_c2_action": torch.tensor(2.0),
            "loss_c2_teacher": torch.tensor(2.0),
        }


def _compose_causal_config():
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        return compose(config_name="causal_dual_mode")


def test_causal_config_requires_spawn_without_changing_bc_default() -> None:
    cfg = _compose_causal_config()
    assert cfg.data.multiprocessing_context == "spawn"
    assert cfg.data.video_backend == "pyav"
    trainer._validate_config(cfg, world_size=4)

    changed = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    changed.data.multiprocessing_context = "fork"
    with pytest.raises(ValueError, match="must use spawn"):
        trainer._validate_config(changed, world_size=4)

    changed = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    changed.data.video_backend = "torchcodec"
    with pytest.raises(ValueError, match="select PyAV directly"):
        trainer._validate_config(changed, world_size=4)

    bc_config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(bc_config_dir)):
        bc_cfg = compose(config_name="uncond_bc")
    assert bc_cfg.data.get("multiprocessing_context") is None
    assert bc_cfg.data.get("video_backend") is None


def test_shared_loader_passes_only_explicit_multiprocessing_context(
    monkeypatch,
) -> None:
    calls = []

    class _Loader:
        def __init__(self, dataset, **kwargs) -> None:
            self.dataset = dataset
            calls.append(kwargs)

    monkeypatch.setattr(bc_trainer, "DataLoader", _Loader)
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "training": {"microbatch_size": 2},
            "validation": {"seed": 42},
            "data": {
                "num_workers": 2,
                "prefetch_factor": 1,
                "multiprocessing_context": "spawn",
            },
        }
    )
    bc_trainer._build_loaders(
        cfg,
        train_dataset=list(range(8)),
        validation_dataset=list(range(4)),
        rank=0,
        world_size=1,
    )
    assert len(calls) == 2
    assert all(call["multiprocessing_context"] == "spawn" for call in calls)

    calls.clear()
    del cfg.data.multiprocessing_context
    bc_trainer._build_loaders(
        cfg,
        train_dataset=list(range(8)),
        validation_dataset=list(range(4)),
        rank=0,
        world_size=1,
    )
    assert len(calls) == 2
    assert all("multiprocessing_context" not in call for call in calls)


def test_causal_video_backend_is_set_before_workers() -> None:
    subsets = [SimpleNamespace(video_backend="torchcodec") for _ in range(8)]
    datasets = tuple(
        SimpleNamespace(
            dataset=SimpleNamespace(
                lerobot_dataset=SimpleNamespace(
                    multi_dataset=SimpleNamespace(
                        _datasets=subsets[offset : offset + 4]
                    )
                )
            )
        )
        for offset in (0, 4)
    )

    assert trainer._set_causal_video_backend(datasets, backend="pyav") == 8
    assert all(subset.video_backend == "pyav" for subset in subsets)


def test_bounded_validation_stops_at_cap_and_records_timing(monkeypatch) -> None:
    synchronize_calls = []
    monkeypatch.setattr(
        trainer,
        "stateless_validation_flow_inputs",
        lambda **kwargs: (
            torch.zeros(kwargs["action_shape"][0]),
            torch.zeros(kwargs["action_shape"]),
        ),
    )
    monkeypatch.setattr(
        trainer.torch,
        "autocast",
        lambda **_kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        trainer.torch.cuda,
        "synchronize",
        lambda device: synchronize_calls.append(device),
    )
    loader = [
        {
            "sample_identity": [f"{batch_index}:0", f"{batch_index}:1"],
            "action": torch.zeros(2, 32, 7),
        }
        for batch_index in range(3)
    ]
    policy = _Policy()

    result = trainer._validate(
        policy,
        loader,
        cfg=SimpleNamespace(validation=SimpleNamespace(seed=42)),
        world_size=1,
        device=torch.device("cpu"),
        max_batches_per_rank=2,
        synchronize_timing=True,
    )

    assert policy.evaluated_batches == 2
    assert result["sample_count"] == 4
    assert result["batches_per_rank_min"] == 2
    assert result["batches_per_rank_max"] == 2
    assert result["batch_cap_per_rank"] == 2
    assert result["elapsed_seconds"] > 0.0
    assert result["global_samples_per_second"] > 0.0
    assert len(synchronize_calls) == 2
