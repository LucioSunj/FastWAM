"""Runtime placement helpers for modules that allocate tensors explicitly."""

from __future__ import annotations

import torch


class RuntimePlacementMixin:
    """Synchronize cached placement metadata after recursive module moves."""

    device: torch.device
    torch_dtype: torch.dtype

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        anchor = next(self.parameters(), None)
        if anchor is not None:
            self.device = anchor.device
            if anchor.is_floating_point():
                self.torch_dtype = anchor.dtype
        return result
