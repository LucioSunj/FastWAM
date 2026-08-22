"""Read-only FastWAM condition K/V payloads for state-value estimation."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .kv_tap import KeyValueBank, KVSource


@dataclass(frozen=True)
class ConditionLayerKV:
    """Current-video and prompt/state K/V captured at one MoT layer."""

    layer_index: int
    current_frame_video: KeyValueBank
    context: KeyValueBank

    def __post_init__(self) -> None:
        if self.layer_index < 0:
            raise ValueError(
                f"`layer_index` must be non-negative, got {self.layer_index}."
            )
        if self.current_frame_video.source is not KVSource.CURRENT_FRAME_VIDEO:
            raise ValueError("`current_frame_video` has the wrong K/V source identity.")
        if self.context.source is not KVSource.TEXT_STATE_CONTEXT:
            raise ValueError("`context` has the wrong K/V source identity.")
        if self.current_frame_video.contains_generated_future_video:
            raise ValueError(
                "Current-frame value features cannot contain generated future video."
            )
        if self.current_frame_video.batch_size != self.context.batch_size:
            raise ValueError("Condition K/V banks must have the same batch size.")
        if self.current_frame_video.feature_dim != self.context.feature_dim:
            raise ValueError(
                "Condition K/V banks must have the same feature dimension."
            )
        if self.current_frame_video.key.device != self.context.key.device:
            raise ValueError("Condition K/V banks must be on the same device.")

    @property
    def batch_size(self) -> int:
        """Return the common batch size."""

        return self.current_frame_video.batch_size

    @property
    def feature_dim(self) -> int:
        """Return the common projected K/V width."""

        return self.current_frame_video.feature_dim

    @property
    def nbytes(self) -> int:
        """Return the combined storage used by both sources."""

        return self.current_frame_video.nbytes + self.context.nbytes

    def detached(self) -> ConditionLayerKV:
        """Detach both sources from the frozen FastWAM feature producer."""

        return ConditionLayerKV(
            layer_index=self.layer_index,
            current_frame_video=self.current_frame_video.detached(),
            context=self.context.detached(),
        )

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> ConditionLayerKV:
        """Move one condition payload without changing source identities."""

        return ConditionLayerKV(
            layer_index=self.layer_index,
            current_frame_video=self.current_frame_video.to(
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            context=self.context.to(
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
        )
