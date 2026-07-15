"""Experimental controls for testing whether IDM gains require valid futures.

These controls are deliberately separate from :class:`WAMMode`: production
routing remains the binary UNCOND/IDM decision.  The helpers here are used only
by intervention evaluation and profiling.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
from typing import Any, Mapping

import torch


class IDMControl(str, Enum):
    """Runtime intervention applied to an otherwise frozen dual-regime WAM."""

    VALID_IDM = "valid_idm"
    NO_READ = "no_read"
    REPEAT_CURRENT = "repeat_current"
    SHUFFLED = "shuffled"
    EXTRA_COMPUTE = "extra_compute"


IDM_CONTROL_ORDER = (
    IDMControl.VALID_IDM,
    IDMControl.NO_READ,
    IDMControl.REPEAT_CURRENT,
    IDMControl.SHUFFLED,
    IDMControl.EXTRA_COMPUTE,
)


def coerce_idm_control(value: IDMControl | str) -> IDMControl:
    if isinstance(value, IDMControl):
        return value
    try:
        return IDMControl(str(value).strip().lower())
    except ValueError as exc:
        expected = [control.value for control in IDM_CONTROL_ORDER]
        raise ValueError(f"Unknown IDM control {value!r}; expected one of {expected}.") from exc


@dataclass(frozen=True)
class ShuffledFutureDonor:
    """Pre-generated matched donor used by the shuffled-future control."""

    latents: torch.Tensor
    metadata: Mapping[str, Any]


DONOR_BANK_VERSION = 1
DONOR_CELL_FIELDS = ("task", "factor", "level", "phase")


def _validated_wam_seed(metadata: Mapping[str, Any], *, source: str) -> int:
    value = metadata.get("wam_seed")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{source}.wam_seed must be a non-negative integer")
    return value


def donor_cell(metadata: Mapping[str, Any]) -> tuple[str, str, str, str]:
    missing = [field for field in DONOR_CELL_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"donor metadata is missing matched-cell fields: {missing}")
    return tuple(str(metadata[field]) for field in DONOR_CELL_FIELDS)  # type: ignore[return-value]


class ShuffledFutureBank:
    """Validated in-memory bank with deterministic, recipient-excluding selection."""

    def __init__(self, records: list[Mapping[str, Any]], *, metadata: Mapping[str, Any]):
        self.metadata = dict(metadata)
        bank_wam_seed = _validated_wam_seed(
            self.metadata, source="donor bank metadata"
        )
        self._cells: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
        for record in records:
            state_id = record.get("state_id")
            latents = record.get("latents")
            record_meta = record.get("metadata")
            if not isinstance(state_id, str) or not state_id:
                raise ValueError("every donor-bank record needs a non-empty state_id")
            if not torch.is_tensor(latents) or latents.ndim != 5:
                raise ValueError(f"donor {state_id} latents must be [B,C,T,H,W]")
            if not isinstance(record_meta, Mapping):
                raise ValueError(f"donor {state_id} metadata must be a mapping")
            if (
                "wam_seed" in record_meta
                and _validated_wam_seed(
                    record_meta, source=f"donor {state_id} metadata"
                )
                != bank_wam_seed
            ):
                raise ValueError(
                    f"donor {state_id} wam_seed does not match donor bank metadata"
                )
            self._cells.setdefault(donor_cell(record_meta), []).append(record)
        for records_in_cell in self._cells.values():
            records_in_cell.sort(key=lambda item: str(item["state_id"]))

    def select(
        self,
        *,
        recipient_state_id: str,
        recipient_metadata: Mapping[str, Any],
        seed: int,
    ) -> ShuffledFutureDonor:
        candidates = [
            record
            for record in self._cells.get(donor_cell(recipient_metadata), [])
            if record["state_id"] != recipient_state_id
        ]
        if not candidates:
            raise ValueError(
                "no matched shuffled-future donor remains after excluding recipient "
                f"{recipient_state_id!r}"
            )
        digest = hashlib.sha256(
            f"{int(seed)}:{recipient_state_id}".encode("utf-8")
        ).digest()
        record = candidates[int.from_bytes(digest[:8], "big") % len(candidates)]
        return ShuffledFutureDonor(record["latents"], record["metadata"])

    def to_payload(self) -> dict[str, Any]:
        records = [record for cell in sorted(self._cells) for record in self._cells[cell]]
        return {
            "schema_version": DONOR_BANK_VERSION,
            "kind": "fastwam_shuffled_future_bank",
            "metadata": self.metadata,
            "records": records,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "ShuffledFutureBank":
        if payload.get("schema_version") != DONOR_BANK_VERSION:
            raise ValueError(
                f"unsupported donor-bank schema {payload.get('schema_version')!r}"
            )
        if payload.get("kind") != "fastwam_shuffled_future_bank":
            raise ValueError(f"unexpected donor-bank kind {payload.get('kind')!r}")
        records = payload.get("records")
        metadata = payload.get("metadata")
        if not isinstance(records, list) or not isinstance(metadata, Mapping):
            raise ValueError("malformed shuffled-future donor bank")
        return cls(records, metadata=metadata)


def validate_donor_metadata(
    actual: Mapping[str, Any], expected: Mapping[str, Any] | None
) -> None:
    """Require every expected semantic field to match exactly."""
    if expected is None:
        return
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "Shuffled-future donor metadata does not match the recipient cell "
            f"(actual, expected): {mismatches}."
        )


def intervene_video_latents(
    latents_video: torch.Tensor,
    *,
    control: IDMControl | str,
    first_frame_latents: torch.Tensor,
    donor: ShuffledFutureDonor | None = None,
    expected_donor_metadata: Mapping[str, Any] | None = None,
) -> torch.Tensor:
    """Apply a post-generation latent intervention while preserving frame zero."""
    selected = coerce_idm_control(control)
    if selected in {IDMControl.VALID_IDM, IDMControl.NO_READ}:
        if donor is not None:
            raise ValueError(f"{selected.value} does not accept a shuffled donor.")
        return latents_video
    if selected is IDMControl.EXTRA_COMPUTE:
        raise ValueError("extra_compute is an UNCOND action-only control, not an IDM latent intervention.")
    if latents_video.ndim != 5 or first_frame_latents.ndim != 5:
        raise ValueError(
            "video and first-frame latents must be [B,C,T,H,W], got "
            f"{tuple(latents_video.shape)} and {tuple(first_frame_latents.shape)}."
        )
    if first_frame_latents.shape[:2] != latents_video.shape[:2] or (
        first_frame_latents.shape[-2:] != latents_video.shape[-2:]
    ):
        raise ValueError("first-frame latent batch/channel/spatial dimensions do not match video latents.")
    if first_frame_latents.shape[2] != 1:
        raise ValueError(
            f"first_frame_latents must contain exactly one latent frame, got {first_frame_latents.shape[2]}."
        )

    result = latents_video.clone()
    result[:, :, 0:1] = first_frame_latents.to(result)
    if selected is IDMControl.REPEAT_CURRENT:
        if result.shape[2] > 1:
            result[:, :, 1:] = first_frame_latents.to(result).expand(
                -1, -1, result.shape[2] - 1, -1, -1
            )
        return result

    if donor is None:
        raise ValueError("shuffled control requires a pre-generated ShuffledFutureDonor.")
    validate_donor_metadata(donor.metadata, expected_donor_metadata)
    donor_latents = donor.latents.to(device=result.device, dtype=result.dtype)
    if donor_latents.shape != result.shape:
        raise ValueError(
            "shuffled donor/video latent shape mismatch: "
            f"{tuple(donor_latents.shape)} vs {tuple(result.shape)}."
        )
    if result.shape[2] > 1:
        result[:, :, 1:] = donor_latents[:, :, 1:]
    return result


def block_action_future_reads(
    attention_mask: torch.Tensor,
    *,
    video_seq_len: int,
    video_tokens_per_frame: int,
) -> torch.Tensor:
    """Keep full video compute but restrict action rows to current-frame columns."""
    if attention_mask.ndim != 2 or attention_mask.shape[0] != attention_mask.shape[1]:
        raise ValueError(f"attention_mask must be square, got {tuple(attention_mask.shape)}.")
    video_seq_len = int(video_seq_len)
    first_frame_tokens = min(int(video_tokens_per_frame), video_seq_len)
    if video_seq_len <= 0 or first_frame_tokens <= 0 or video_seq_len >= attention_mask.shape[0]:
        raise ValueError(
            "invalid video/action attention partition: "
            f"video_seq_len={video_seq_len}, tokens_per_frame={video_tokens_per_frame}, "
            f"mask={tuple(attention_mask.shape)}."
        )
    result = attention_mask.clone()
    result[video_seq_len:, :video_seq_len] = False
    result[video_seq_len:, :first_frame_tokens] = True
    return result
