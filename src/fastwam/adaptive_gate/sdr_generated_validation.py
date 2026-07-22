"""Provenance and metrics for deployment-aligned generated-future validation."""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .provenance import inference_solver_fingerprint
from .sdr_contracts import artifact_record
from .controls import (
    IDMControl,
    ShuffledFutureDonor,
    coerce_idm_control,
    intervene_video_latents,
)


CACHE_SCHEMA = "fastwam-sdr-generated-future-cache-v1"
VALIDATION_SCHEMA = "fastwam-sdr-generated-future-validation-v1"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def module_state_value_sha256(module: torch.nn.Module) -> str:
    """Hash tensor names, metadata, and values without serializing a checkpoint."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().to(device="cpu").contiguous()
        header = json.dumps(
            {
                "name": name,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big"))
        digest.update(header)
        digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def build_cache_metadata(
    *,
    e_i_checkpoint: str | os.PathLike[str],
    e_i_config: str | os.PathLike[str],
    dataset_stats: str | os.PathLike[str],
    validation_manifest: str | os.PathLike[str],
    sample_id: str,
    solver_contract: Mapping[str, Any],
    video_state_sha256: str,
    proprio_state_sha256: str,
    seed: int,
    verified_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if not sample_id:
        raise ValueError("Generated-future cache requires a non-empty sample_id.")
    if not _is_sha256(video_state_sha256) or not _is_sha256(
        proprio_state_sha256
    ):
        raise ValueError("Video/proprio state fingerprints must be SHA256 values.")
    manifest_path = Path(validation_manifest).expanduser().resolve()
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = manifest_payload.get("samples") if isinstance(manifest_payload, dict) else None
    donors = (
        manifest_payload.get("shuffle_donors")
        if isinstance(manifest_payload, dict)
        else None
    )
    registered_ids = {
        str(sample.get("sample_id"))
        for sample in samples or ()
        if isinstance(sample, Mapping)
    } | {
        str(donor.get("donor_id"))
        for donor in donors or ()
        if isinstance(donor, Mapping)
    }
    if not isinstance(samples, list) or sample_id not in registered_ids:
        raise ValueError(
            f"sample_id {sample_id!r} is not present in the validation manifest."
        )
    solver_fingerprint = inference_solver_fingerprint(solver_contract)
    artifact_paths = {
        "e_i_checkpoint": e_i_checkpoint,
        "e_i_config": e_i_config,
        "dataset_stats": dataset_stats,
        "validation_manifest": manifest_path,
    }
    artifacts = {}
    for name, path in artifact_paths.items():
        if verified_artifacts is not None and name in verified_artifacts:
            record = dict(verified_artifacts[name])
            if Path(str(record.get("path", ""))).expanduser().resolve() != Path(
                path
            ).expanduser().resolve():
                raise ValueError(
                    f"Verified cache artifact path mismatch for {name}."
                )
            if not _is_sha256(record.get("sha256")):
                raise ValueError(
                    f"Verified cache artifact has invalid SHA256 for {name}."
                )
            artifacts[name] = record
        else:
            artifacts[name] = artifact_record(path)
    metadata = {
        "schema": CACHE_SCHEMA,
        "source_stage": "e_i_s0",
        "artifacts": artifacts,
        "sample_id": sample_id,
        "seed": int(seed),
        "solver_contract": dict(solver_contract),
        "solver_fingerprint": solver_fingerprint,
        "video_state_sha256": video_state_sha256,
        "proprio_state_sha256": proprio_state_sha256,
        "contents": "video_latents_only",
    }
    metadata["cache_key"] = _canonical_sha256(metadata)
    return metadata


def validate_cache_metadata(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    if actual.get("schema") != CACHE_SCHEMA or expected.get("schema") != CACHE_SCHEMA:
        raise ValueError("Unsupported generated-future cache metadata schema.")
    if actual.get("contents") != "video_latents_only":
        raise ValueError("Generated-future cache may contain only video latents.")
    expected_key = _canonical_sha256(
        {key: value for key, value in actual.items() if key != "cache_key"}
    )
    if actual.get("cache_key") != expected_key:
        raise ValueError("Generated-future cache_key is invalid.")
    if dict(actual) != dict(expected):
        differing = sorted(
            key
            for key in set(actual) | set(expected)
            if actual.get(key) != expected.get(key)
        )
        raise ValueError(
            "Generated-future cache provenance changed; cache is invalid for "
            f"fields={differing}."
        )


def write_latent_cache(
    path: str | os.PathLike[str],
    *,
    video_latents: torch.Tensor,
    metadata: Mapping[str, Any],
) -> None:
    if metadata.get("schema") != CACHE_SCHEMA:
        raise ValueError("Generated-future cache metadata is missing or unsupported.")
    if not torch.is_tensor(video_latents) or video_latents.ndim != 5:
        raise ValueError("video_latents must be a [B,C,T,H,W] tensor.")
    if not bool(torch.isfinite(video_latents).all()):
        raise ValueError("video_latents contains NaN or Inf.")
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(
        {
            "metadata": dict(metadata),
            "video_latents": video_latents.detach().to(
                device="cpu", dtype=torch.float32
            ),
        },
        temporary,
    )
    os.replace(temporary, target)


def load_latent_cache(
    path: str | os.PathLike[str],
    *,
    expected_metadata: Mapping[str, Any],
) -> torch.Tensor:
    payload = torch.load(
        Path(path).expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, Mapping) or set(payload) != {
        "metadata",
        "video_latents",
    }:
        raise ValueError(
            "Generated-future cache must contain exactly metadata and video_latents."
        )
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise ValueError("Generated-future cache metadata must be an object.")
    validate_cache_metadata(metadata, expected_metadata)
    latents = payload["video_latents"]
    if not torch.is_tensor(latents) or latents.ndim != 5:
        raise ValueError("Cached video_latents must be [B,C,T,H,W].")
    if latents.device.type != "cpu" or latents.dtype != torch.float32:
        raise ValueError("Cached video latents must be CPU FP32.")
    if not bool(torch.isfinite(latents).all()):
        raise ValueError("Cached video latents contain NaN or Inf.")
    return latents


@torch.no_grad()
def infer_action_from_cached_video_latents(
    model,
    *,
    video_latents: torch.Tensor,
    action_horizon: int,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    proprio: torch.Tensor | None,
    seed: int,
    action_inference_steps: int = 20,
    sigma_shift: float | None = None,
    control: IDMControl | str = IDMControl.VALID_IDM,
    shuffled_future_latents: torch.Tensor | None = None,
    shuffled_future_metadata: Mapping[str, Any] | None = None,
    expected_donor_metadata: Mapping[str, Any] | None = None,
    rand_device: str = "cpu",
) -> torch.Tensor:
    """Run only the IDM action stage against immutable cached video latents."""
    selected = coerce_idm_control(control)
    if selected is IDMControl.EXTRA_COMPUTE:
        raise ValueError("extra_compute is not an IDM cached-latent control.")
    if video_latents.ndim != 5 or video_latents.shape[0] != 1:
        raise ValueError("video_latents must be [1,C,T,H,W].")
    if context.ndim == 2:
        context = context.unsqueeze(0)
    if context_mask.ndim == 1:
        context_mask = context_mask.unsqueeze(0)
    if context.ndim != 3 or context_mask.ndim != 2:
        raise ValueError("context/context_mask must be [B,L,D]/[B,L].")

    latents_video = video_latents.to(
        device=model.device,
        dtype=model.torch_dtype,
    )
    first_frame_latents = latents_video[:, :, 0:1].clone()
    donor = None
    if shuffled_future_latents is not None:
        if shuffled_future_metadata is None:
            raise ValueError("Shuffled future latents require donor metadata.")
        donor = ShuffledFutureDonor(
            latents=shuffled_future_latents,
            metadata=shuffled_future_metadata,
        )
    latents_video = intervene_video_latents(
        latents_video,
        control=selected,
        first_frame_latents=first_frame_latents,
        donor=donor,
        expected_donor_metadata=expected_donor_metadata,
    )

    context = context.to(
        device=model.device,
        dtype=model.torch_dtype,
        non_blocking=True,
    )
    context_mask = context_mask.to(
        device=model.device,
        dtype=torch.bool,
        non_blocking=True,
    )
    if proprio is not None:
        if proprio.ndim == 1:
            proprio = proprio.unsqueeze(0)
        if proprio.ndim != 2 or proprio.shape[0] != 1:
            raise ValueError("proprio must be [D] or [1,D].")
        proprio = proprio.to(device=model.device, dtype=model.torch_dtype)
        context, context_mask = model._append_proprio_to_context(
            context=context,
            context_mask=context_mask,
            proprio=proprio,
        )

    generator = torch.Generator(device=rand_device).manual_seed(int(seed))
    latents_action = torch.randn(
        (1, int(action_horizon), model.action_expert.action_dim),
        generator=generator,
        device=rand_device,
        dtype=torch.float32,
    ).to(device=model.device, dtype=model.torch_dtype)
    fuse_flag = bool(
        getattr(model.video_expert, "fuse_vae_embedding_in_latents", False)
    )
    timestep_video = torch.zeros(
        (1,), dtype=latents_video.dtype, device=model.device
    )
    video_pre = model.video_expert.pre_dit(
        x=latents_video,
        timestep=timestep_video,
        context=context,
        context_mask=context_mask,
        action=None,
        fuse_vae_embedding_in_latents=fuse_flag,
    )
    video_seq_len = int(video_pre["tokens"].shape[1])
    tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
    attention_mask = model._build_mot_attention_mask(
        video_seq_len=video_seq_len,
        action_seq_len=latents_action.shape[1],
        video_tokens_per_frame=tokens_per_frame,
        device=video_pre["tokens"].device,
    )
    video_kv_cache = model.mot.prefill_video_cache(
        video_tokens=video_pre["tokens"],
        video_freqs=video_pre["freqs"],
        video_t_mod=video_pre["t_mod"],
        video_context_payload={
            "context": video_pre["context"],
            "mask": video_pre["context_mask"],
        },
        video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
    )
    if selected is IDMControl.NO_READ:
        # Pay the full generated-video cache work above, then use the same
        # single-frame action path as forced UNCOND. A masked long sequence is
        # not BF16-equivalent because the attention kernel still sees a
        # different K/V length.
        video_pre = model.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        attention_mask = model._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=tokens_per_frame,
            device=video_pre["tokens"].device,
        )
        video_kv_cache = model.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[
                :video_seq_len, :video_seq_len
            ],
        )
    timesteps, deltas = model.infer_action_scheduler.build_inference_schedule(
        num_inference_steps=int(action_inference_steps),
        device=model.device,
        dtype=latents_action.dtype,
        shift_override=sigma_shift,
    )
    for step_t, step_delta in zip(timesteps, deltas):
        timestep_action = step_t.unsqueeze(0).to(
            dtype=latents_action.dtype,
            device=model.device,
        )
        prediction = model._predict_action_noise_with_cache(
            latents_action=latents_action,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        latents_action = model.infer_action_scheduler.step(
            prediction,
            step_delta,
            latents_action,
        )
    return latents_action[0].detach().to(device="cpu", dtype=torch.float32)


def action_distance(
    left: torch.Tensor,
    right: torch.Tensor,
) -> dict[str, float]:
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Action tensors must have matching [T,A] shapes.")
    delta = left.detach().float() - right.detach().float()
    return {
        "l1": float(delta.abs().mean().item()),
        "l2": float(torch.sqrt(torch.mean(delta.square())).item()),
        "max_abs": float(delta.abs().max().item()),
    }


def generated_future_sensitivity_gate(
    records: Sequence[Mapping[str, Any]],
    *,
    threshold: float = 1e-3,
    minimum_fraction: float = 0.25,
) -> dict[str, Any]:
    if not records:
        raise ValueError("Generated-future sensitivity requires sample records.")
    values = [float(record["valid_no_read_normalized_action_l2"]) for record in records]
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("Generated-future sensitivity values must be finite.")
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    median = (
        ordered[midpoint]
        if len(ordered) % 2
        else 0.5 * (ordered[midpoint - 1] + ordered[midpoint])
    )
    fraction = sum(value >= threshold for value in values) / len(values)
    passed = median >= threshold and fraction >= minimum_fraction
    return {
        "pass": passed,
        "sample_count": len(values),
        "threshold": threshold,
        "minimum_fraction": minimum_fraction,
        "median": median,
        "fraction_at_or_above_threshold": fraction,
        "interpretation": (
            "Action sensitivity is conditioning-read evidence, not task usefulness evidence."
        ),
    }


def no_read_uncond_parity(
    no_read: torch.Tensor,
    uncond: torch.Tensor,
    *,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    distance = action_distance(no_read, uncond)
    return {
        "pass": distance["max_abs"] <= float(tolerance),
        "tolerance": float(tolerance),
        **distance,
    }
