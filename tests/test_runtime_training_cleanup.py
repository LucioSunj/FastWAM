"""Regression tests for production training runtime cleanup."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

runtime = pytest.importorskip("fastwam.runtime")


def test_run_training_ends_accelerator_when_train_raises(monkeypatch, tmp_path):
    events = []

    class FakeTrainer:
        def __init__(self, **_kwargs):
            self.accelerator = SimpleNamespace(
                end_training=lambda: events.append("end_training")
            )

        def train(self):
            events.append("train")
            raise RuntimeError("training failed")

    monkeypatch.setattr(runtime, "setup_logging", lambda **_kwargs: None)
    monkeypatch.setattr(runtime.misc, "register_work_dir", lambda _path: None)
    monkeypatch.setattr(runtime, "_resolve_train_device", lambda: "cpu")
    monkeypatch.setattr(runtime, "instantiate", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(runtime, "build_datasets", lambda _cfg: (object(), None))
    monkeypatch.setattr(runtime, "Wan22Trainer", FakeTrainer)
    cfg = OmegaConf.create(
        {
            "output_dir": str(tmp_path),
            "mixed_precision": "bf16",
            "model": {"_target_": "unused"},
            "data": {},
        }
    )

    with pytest.raises(RuntimeError, match="training failed"):
        runtime.run_training(cfg)
