"""Instance-scoped MB1 linear geometry for route-neutral batching."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import torch
import torch.nn.functional as F
from torch import nn


class BatchLinearContext:
    """Carry the real sample batch without changing unrelated FastWAM actors."""

    def __init__(self) -> None:
        self._batch_size: ContextVar[int | None] = ContextVar(
            f"fastwam_batch_linear_{id(self)}", default=None
        )

    @property
    def batch_size(self) -> int | None:
        return self._batch_size.get()

    @contextmanager
    def use(self, batch_size: int) -> Iterator[None]:
        if batch_size < 1:
            raise ValueError("Batch-linear sample count must be positive.")
        token = self._batch_size.set(int(batch_size))
        try:
            yield
        finally:
            self._batch_size.reset(token)


def rowwise_linear(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    batch_size: int,
    flattened_batch: bool = False,
) -> torch.Tensor:
    """Apply the original Linear kernel once per sample and concatenate rows."""

    original_shape = input_tensor.shape
    if flattened_batch:
        if input_tensor.ndim != 2 or input_tensor.shape[0] % batch_size:
            raise ValueError(
                "Flattened video-time Linear input does not match sample batch."
            )
        input_tensor = input_tensor.reshape(batch_size, -1, input_tensor.shape[-1])
    elif input_tensor.ndim < 2 or input_tensor.shape[0] != batch_size:
        raise ValueError("Batch-major Linear input does not match sample batch.")
    output = torch.cat(
        [
            F.linear(input_tensor[index : index + 1], weight, bias)
            for index in range(batch_size)
        ],
        dim=0,
    )
    if flattened_batch:
        return output.reshape(*original_shape[:-1], weight.shape[0])
    return output


class BatchInvariantLinear(nn.Module):
    """Reuse an existing Linear's parameters with opt-in MB1 execution geometry."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        context: BatchLinearContext,
        flattened_batch: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.weight = base.weight
        self.bias = base.bias
        self.context = context
        self.flattened_batch = bool(flattened_batch)
        self.train(base.training)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        batch_size = self.context.batch_size
        if batch_size is None:
            return F.linear(input_tensor, self.weight, self.bias)
        return rowwise_linear(
            input_tensor,
            self.weight,
            self.bias,
            batch_size=batch_size,
            flattened_batch=self.flattened_batch,
        )


def install_batch_invariant_linears(actor: nn.Module) -> BatchLinearContext:
    """Install route-neutral wrappers while retaining keys and Parameter objects."""

    from fastwam.adapters.regime_lora import RegimeLoRALinear

    context = BatchLinearContext()
    flattened_ids = {
        id(module)
        for module in actor.video_expert.time_embedding.modules()
        if isinstance(module, nn.Linear)
    }

    def visit(parent: nn.Module) -> None:
        for name, child in tuple(parent.named_children()):
            if isinstance(child, RegimeLoRALinear):
                if float(child.lora_dropout.p) != 0.0:
                    raise ValueError(
                        "Route-neutral batch-invariant LoRA requires dropout=0."
                    )
                child.batch_linear_context = context
                continue
            if isinstance(child, nn.Linear):
                setattr(
                    parent,
                    name,
                    BatchInvariantLinear(
                        child,
                        context=context,
                        flattened_batch=id(child) in flattened_ids,
                    ),
                )
                continue
            visit(child)

    visit(actor)
    return context


__all__ = [
    "BatchInvariantLinear",
    "BatchLinearContext",
    "install_batch_invariant_linears",
    "rowwise_linear",
]
