"""Checkpoint-bound normalization provenance helpers."""
from __future__ import annotations

import hashlib
import os
import warnings


def sha256_file(path: str | os.PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset_stats_fingerprint(model, stats_path: str | os.PathLike) -> str:
    """Verify stats for modern adaptive checkpoints; leave vanilla models alone."""
    actual = sha256_file(stats_path)
    provenance = getattr(model, "_loaded_checkpoint_provenance", None)
    live_regimes = tuple(getattr(model, "adaptive_regimes", ()))
    if provenance is None:
        if live_regimes:
            warnings.warn(
                "Adaptive legacy checkpoint has no dataset-stats provenance; "
                "normalization compatibility cannot be verified.",
                RuntimeWarning,
                stacklevel=2,
            )
        return actual
    regimes = tuple(provenance.get("adaptive_regimes", ()))
    if not regimes:
        return actual
    expected = provenance.get("dataset_stats_fingerprint")
    if not isinstance(expected, str) or not expected:
        raise ValueError(
            "Adaptive checkpoint is missing dataset_stats_fingerprint and cannot "
            "be evaluated safely."
        )
    if actual != expected:
        raise ValueError(
            "Dataset stats do not match the adaptive checkpoint: "
            f"checkpoint={expected}, file={actual}, path={os.fspath(stats_path)!r}."
        )
    return actual
