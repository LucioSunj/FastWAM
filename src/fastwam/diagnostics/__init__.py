"""E7 mechanism diagnostics: read-only activation taps and linear probes.

Answers decision point DP2 — *where* information is lost under texture shift —
by fitting the same linear read-out at three loci (VAE latent, video-expert
hidden state, action-expert readout) and reporting where AUC collapses.

Extraction never modifies model source: taps are `register_forward_hook`
only. See `taps` and `probe` for the reasoning behind that and behind the
mandatory group-disjoint cross-validation.
"""

from .probe import (
    DEFAULT_POOL_DIM,
    POOL_LAYOUT,
    ProbeResult,
    cross_fitted_probe,
    fit_logistic,
    grouped_kfold,
    pool_activation,
    probe_taps,
    roc_auc,
)
from .taps import ActivationTaps, TapSpec, resolve_module

__all__ = [
    "ActivationTaps",
    "TapSpec",
    "resolve_module",
    "DEFAULT_POOL_DIM",
    "POOL_LAYOUT",
    "ProbeResult",
    "cross_fitted_probe",
    "fit_logistic",
    "grouped_kfold",
    "pool_activation",
    "probe_taps",
    "roc_auc",
]
