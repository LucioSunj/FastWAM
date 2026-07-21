import hashlib
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import torch


logger = logging.getLogger(__name__)

PACKED_CACHE_BIN = "packed_cache.bin"
PACKED_CACHE_INDEX = "packed_cache.index.jsonl"


class PrecomputedTextContextCache(dict[str, tuple[torch.Tensor, torch.Tensor]]):
    """Lazy prompt cache backed by individual files or one packed cache."""

    def __init__(self, cache_dir: str | Path, context_len: int, max_size: Optional[int] = 512):
        super().__init__()
        self.cache_dir = Path(
            os.path.expanduser(os.path.expandvars(str(cache_dir)))
        ).resolve()
        self.context_len = int(context_len)
        self.max_size = None if max_size is None else int(max_size)
        self.bin_path = self.cache_dir / PACKED_CACHE_BIN
        self.index_path = self.cache_dir / PACKED_CACHE_INDEX
        self._packed_index: dict[str, tuple[int, int]] = {}
        self._packed_file = None

        if not self.cache_dir.is_dir():
            raise FileNotFoundError(
                f"Precomputed text embedding cache directory not found: {self.cache_dir}"
            )
        self._validate_manifest()
        self._load_packed_index()

        individual_entries = 0
        if not self._packed_index:
            individual_entries = sum(1 for _ in self.cache_dir.glob("*.pt"))
        total_entries = len(self._packed_index) or individual_entries
        if total_entries == 0:
            raise FileNotFoundError(
                f"No text embedding entries found under {self.cache_dir}"
            )

        logger.info(
            "Using precomputed LIBERO text embeddings: dir=%s entries=%d packed_entries=%d",
            self.cache_dir,
            total_entries,
            len(self._packed_index),
        )

    def _validate_manifest(self) -> None:
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            return
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("complete") is False:
            raise ValueError(f"Text embedding cache is incomplete: {manifest_path}")
        manifest_context_len = manifest.get("context_len")
        if (
            manifest_context_len is not None
            and int(manifest_context_len) != self.context_len
        ):
            raise ValueError(
                "Text embedding cache context length mismatch: "
                f"expected {self.context_len}, got {manifest_context_len} in {manifest_path}"
            )

    def _load_packed_index(self) -> None:
        if not self.bin_path.exists() and not self.index_path.exists():
            return
        if not self.bin_path.exists() or not self.index_path.exists():
            raise FileNotFoundError(
                "Packed text cache requires both files: "
                f"{self.bin_path} and {self.index_path}"
            )
        with self.index_path.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                self._packed_index[str(record["name"])] = (
                    int(record["offset"]),
                    int(record["length"]),
                )

    def _load_payload(self, cache_name: str) -> tuple[dict[str, Any], str]:
        span = self._packed_index.get(cache_name)
        if span is not None:
            if self._packed_file is None:
                self._packed_file = self.bin_path.open("rb")
            offset, length = span
            self._packed_file.seek(offset)
            payload_bytes = self._packed_file.read(length)
            if len(payload_bytes) != length:
                raise IOError(
                    f"Packed cache entry {cache_name} is truncated: "
                    f"expected {length} bytes, got {len(payload_bytes)}"
                )
            payload = torch.load(
                io.BytesIO(payload_bytes),
                map_location="cpu",
                weights_only=True,
            )
            return payload, f"{self.bin_path}:{offset}+{length}"

        cache_path = self.cache_dir / cache_name
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Missing precomputed text embedding cache entry {cache_name} "
                f"under {self.cache_dir}"
            )
        payload = torch.load(
            str(cache_path),
            map_location="cpu",
            weights_only=True,
        )
        return payload, str(cache_path)

    def get(
        self,
        prompt: str,
        default: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        cached = super().get(prompt)
        if cached is not None:
            return cached

        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_name = (
            f"{digest}.t5_len{self.context_len}.wan22ti2v5b.pt"
        )
        payload, source = self._load_payload(cache_name)
        context = payload["context"]
        context_mask = payload["mask"].bool()

        if context.ndim != 2:
            raise ValueError(
                f"Cached context must be [L,D], got {tuple(context.shape)} in {source}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached mask must be [L], got {tuple(context_mask.shape)} in {source}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context length mismatch: expected {self.context_len}, "
                f"got {context.shape[0]} in {source}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask length mismatch: expected {self.context_len}, "
                f"got {context_mask.shape[0]} in {source}"
            )

        context = (
            context.detach()
            .to(device="cpu", dtype=torch.bfloat16)
            .contiguous()
            .clone()
        )
        context_mask = (
            context_mask.detach()
            .to(device="cpu", dtype=torch.bool)
            .contiguous()
        )
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        cached = (context, context_mask)

        if self.max_size is not None and self.max_size > 0:
            while len(self) >= self.max_size:
                self.pop(next(iter(self)))
        self[prompt] = cached
        return cached

    def close(self) -> None:
        if self._packed_file is not None:
            self._packed_file.close()
            self._packed_file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

def build_text_context_cache(cfg):
    cache_dir = cfg.EVALUATION.get("text_embedding_cache_dir")
    if cache_dir is None or str(cache_dir).strip().lower() in {"", "none", "null"}:
        return {}
    if not bool(cfg.EVALUATION.get("cache_text_context", True)):
        raise ValueError(
            "EVALUATION.cache_text_context must be true when a precomputed "
            "text embedding cache is configured."
        )

    max_size_cfg = cfg.EVALUATION.get("text_context_cache_max_size", 512)
    max_size = None if max_size_cfg is None else int(max_size_cfg)
    cache = PrecomputedTextContextCache(
        cache_dir=cache_dir,
        context_len=int(cfg.data.train.context_len),
        max_size=max_size,
    )
    logger.info(
        "Precomputed text cache is configured; forcing model.load_text_encoder=false."
    )
    cfg.model.load_text_encoder = False
    return cache
