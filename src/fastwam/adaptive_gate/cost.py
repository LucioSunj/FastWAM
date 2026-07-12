"""Validated per-mode compute costs for the binary adaptive gate."""
from __future__ import annotations

import math
import os
from typing import Any, Mapping, Optional

from .modes import MODE_ORDER, WAMMode


def validate_cost_table(table: Mapping[str, float]) -> dict[str, float]:
    """Validate and canonicalize a normalized ``UNCOND``/``IDM`` table."""
    expected = {m.value for m in MODE_ORDER}
    missing = expected.difference(table)
    extra = set(table).difference(expected)
    if missing or extra:
        raise ValueError(
            f"cost table keys must be exactly {sorted(expected)}; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    result = {m.value: float(table[m.value]) for m in MODE_ORDER}
    for name, value in result.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"cost({name}) must be finite and positive, got {value}.")
    if not math.isclose(result[WAMMode.IDM.value], 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"normalized cost(idm) must equal 1, got {result[WAMMode.IDM.value]}."
        )
    if result[WAMMode.UNCOND.value] >= result[WAMMode.IDM.value]:
        raise ValueError(
            "UNCOND must be cheaper than IDM; got "
            f"{result[WAMMode.UNCOND.value]} >= {result[WAMMode.IDM.value]}."
        )
    return result


def default_cost_table(
    inference_steps: int = 20,
    *,
    uncond_fraction: float = 0.15,
) -> dict[str, float]:
    """Analytical placeholder normalized to ``cost(IDM)=1``.

    ``inference_steps`` is validated and recorded by callers, but the ratio
    cannot be inferred from step count alone because both modes run the full
    action solver.  Profile the exact checkpoint/resolution for reward use.
    """
    if int(inference_steps) <= 0:
        raise ValueError(f"inference_steps must be positive, got {inference_steps}.")
    fraction = float(uncond_fraction)
    return validate_cost_table({"uncond": fraction, "idm": 1.0})


def normalize_cost_table(raw: Mapping[str, float]) -> dict[str, float]:
    """Normalize raw FLOPs/latency so ``cost(IDM)=1`` and validate it."""
    expected = {m.value for m in MODE_ORDER}
    missing = expected.difference(raw)
    extra = set(raw).difference(expected)
    if missing or extra:
        raise ValueError(
            f"raw cost keys must be exactly {sorted(expected)}; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    idm = float(raw[WAMMode.IDM.value])
    if not math.isfinite(idm) or idm <= 0.0:
        raise ValueError("IDM raw cost must be finite and positive to normalize.")
    return validate_cost_table({m.value: float(raw[m.value]) / idm for m in MODE_ORDER})


def save_cost_table(
    path: str,
    *,
    normalized: Mapping[str, float],
    source: str = "flops",
    raw: Optional[dict[str, dict[str, float]]] = None,
    meta: Optional[dict[str, Any]] = None,
) -> None:
    """Write a validated cost YAML consumed by the adapter/reward."""
    import yaml

    normalized = validate_cost_table(normalized)
    payload: dict[str, Any] = {
        "version": 3,
        "source": str(source),
        "mode_order": [m.value for m in MODE_ORDER],
        "modes": normalized,
    }
    if raw is not None:
        for raw_name, raw_table in raw.items():
            normalize_cost_table(raw_table)
        payload["raw"] = raw
    if meta is not None:
        payload["meta"] = meta
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def load_cost_table(
    path: Optional[str],
    *,
    source: Optional[str] = None,
    return_meta: bool = False,
):
    """Load a cost YAML.

    ``None`` means "use the analytical placeholder".  Supplying an invalid or
    missing path is an error rather than a silent change in the RL objective.
    ``source='latency'`` is accepted as an alias for the profiler key
    ``latency_ms``.
    """
    if path is None:
        return (None, None) if return_meta else None
    expanded = os.path.expanduser(str(path))
    if not os.path.isfile(expanded):
        raise FileNotFoundError(f"cost table does not exist: {expanded}")

    import yaml

    with open(expanded, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{expanded}: cost YAML must contain a mapping.")
    if int(payload.get("version", -1)) != 3:
        raise ValueError(
            f"{expanded}: unsupported cost profile version {payload.get('version')!r}; expected 3."
        )
    expected_order = [m.value for m in MODE_ORDER]
    if payload.get("mode_order") != expected_order:
        raise ValueError(
            f"{expanded}: mode_order {payload.get('mode_order')!r} does not match "
            f"{expected_order}."
        )

    if source is not None:
        source_key = "latency_ms" if str(source) == "latency" else str(source)
        raw = payload.get("raw")
        if not isinstance(raw, dict) or source_key not in raw:
            available = sorted(raw) if isinstance(raw, dict) else []
            raise ValueError(
                f"{expanded}: requested raw cost source {source!r} is unavailable; "
                f"available={available}."
            )
        table = normalize_cost_table(raw[source_key])
        return (table, dict(payload.get("meta") or {})) if return_meta else table

    if "modes" not in payload:
        raise ValueError(f"{expanded}: cost YAML is missing the `modes` block.")
    table = validate_cost_table(payload["modes"])
    return (table, dict(payload.get("meta") or {})) if return_meta else table
