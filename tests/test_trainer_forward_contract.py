"""Regression test for DDP/DeepSpeed forward-hook preservation."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

trainer = pytest.importorskip("fastwam.trainer")


def test_training_forward_enters_prepared_wrapper():
    class PreparedWrapper:
        def __init__(self):
            self.calls = 0

        def __call__(self, sample):
            self.calls += 1
            return sample["loss"], {"loss": sample["loss"]}

        def training_loss(self, sample):
            raise AssertionError("wrapper.training_loss must not be called directly")

    wrapper = PreparedWrapper()
    loss, metrics = trainer._forward_prepared_model(wrapper, {"loss": 2.0})
    assert wrapper.calls == 1
    assert loss == 2.0 and metrics == {"loss": 2.0}


def test_finish_training_can_skip_final_checkpoint():
    instance = object.__new__(trainer.Wan22Trainer)
    instance.global_step = 1
    instance.save_final_checkpoint = False
    instance.accelerator = SimpleNamespace(
        is_main_process=True,
    )
    instance.save_checkpoint = lambda: pytest.fail("checkpoint must not be written")

    assert instance._finish_training("smoke complete") is None


def test_finish_training_preserves_default_checkpoint_behavior():
    expected = {"weights_path": "weights.pt", "state_path": "state"}
    calls = []
    instance = object.__new__(trainer.Wan22Trainer)
    instance.global_step = 3
    instance.save_final_checkpoint = True
    instance.accelerator = SimpleNamespace(
        is_main_process=True,
    )
    instance.save_checkpoint = lambda: calls.append("save") or expected

    assert instance._finish_training("training finished") == expected
    assert calls == ["save"]
