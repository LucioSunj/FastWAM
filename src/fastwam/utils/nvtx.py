"""Optional CUDA profiling ranges; CPU builds execute the same model graph."""

import torch


def range_push(message: str) -> None:
    """Open a profiling range only when the PyTorch build includes CUDA."""
    if torch.version.cuda is not None:
        torch.cuda.nvtx.range_push(message)


def range_pop() -> None:
    """Close a profiling range only when the PyTorch build includes CUDA."""
    if torch.version.cuda is not None:
        torch.cuda.nvtx.range_pop()
