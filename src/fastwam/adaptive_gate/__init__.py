"""Adaptive-prediction gate support (fastwam side).

Mode-switched, FROZEN-WAM inference wrapper + per-mode compute cost. The RL gate
policy / env / configs live in the RLinf repo and import `WAMModeAdapter` from
here.
"""
from .cost import default_cost_table, load_cost_table, normalize_cost_table, save_cost_table
from .modes import (
    MODE_ORDER,
    NUM_MODES,
    WAMMode,
    future_branch_for,
    mode_from_index,
    mode_to_branch_steps,
    mode_to_index,
)
from .wam_mode_adapter import WAMModeAdapter

__all__ = [
    "WAMMode",
    "MODE_ORDER",
    "NUM_MODES",
    "mode_from_index",
    "mode_to_index",
    "future_branch_for",
    "mode_to_branch_steps",
    "WAMModeAdapter",
    "default_cost_table",
    "normalize_cost_table",
    "load_cost_table",
    "save_cost_table",
]
