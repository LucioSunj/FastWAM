"""Regression test for DDP/DeepSpeed forward-hook preservation."""
from __future__ import annotations

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
