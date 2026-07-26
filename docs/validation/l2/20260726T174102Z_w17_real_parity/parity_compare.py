#!/usr/bin/env python
"""Phase B2: cross-checkout bitwise comparison of the saved action tensors.

Criterion (W17 merge-precondition-2, verbatim: NO tolerance):
  torch.equal(main, base) is True AND max_abs == 0, for every pair.

Pairs, per seed in {0,1,2}:
  main_idm_seed{s}.pt  vs base_idm_seed{s}.pt   (public forced-IDM pipeline)
  main_base_seed{s}.pt vs base_base_seed{s}.pt  (W17-refactored UNCOND solver)

Exit codes: 0 = all pairs bitwise equal; 2 = at least one mismatch (FINDING);
3 = missing tensor file.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

TENSOR_DIR = Path("/root/autodl-tmp/.tmp/phaseB2/tensors")
OUT_PATH = Path("/root/autodl-tmp/.tmp/phaseB2/compare_result.json")
SEEDS = (0, 1, 2)


def tensor_sha256(t: torch.Tensor) -> str:
    t = t.detach().contiguous().cpu()
    return hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()


def main() -> int:
    rows = []
    all_equal = True
    for path_kind, human in (("idm", "public forced-IDM pipeline"),
                             ("base", "explicit-dispatch UNCOND solver")):
        for seed in SEEDS:
            a_path = TENSOR_DIR / f"main_{path_kind}_seed{seed}.pt"
            b_path = TENSOR_DIR / f"base_{path_kind}_seed{seed}.pt"
            if not a_path.exists() or not b_path.exists():
                print(f"MISSING: {a_path} or {b_path}", file=sys.stderr)
                return 3
            a = torch.load(str(a_path), map_location="cpu", weights_only=True)
            b = torch.load(str(b_path), map_location="cpu", weights_only=True)
            same_meta = (a.shape == b.shape) and (a.dtype == b.dtype)
            equal = bool(same_meta and torch.equal(a, b))
            max_abs = (
                float((a.double() - b.double()).abs().max().item())
                if a.shape == b.shape else float("nan")
            )
            row = {
                "pair": f"{path_kind}_seed{seed}",
                "path_kind": human,
                "seed": seed,
                "shape_main": list(a.shape),
                "shape_base": list(b.shape),
                "dtype_main": str(a.dtype),
                "dtype_base": str(b.dtype),
                "sha256_main": tensor_sha256(a),
                "sha256_base": tensor_sha256(b),
                "torch_equal": equal,
                "max_abs": max_abs,
            }
            rows.append(row)
            all_equal = all_equal and equal and max_abs == 0.0
            print(json.dumps(row, sort_keys=True))
    verdict = {
        "criterion": "torch.equal True AND max_abs == 0 for all pairs (no tolerance)",
        "all_pairs_bitwise_equal": bool(all_equal),
        "pairs": rows,
    }
    OUT_PATH.write_text(json.dumps(verdict, indent=2, sort_keys=True))
    print(f"VERDICT all_pairs_bitwise_equal={all_equal} -> {OUT_PATH}")
    return 0 if all_equal else 2


if __name__ == "__main__":
    raise SystemExit(main())
