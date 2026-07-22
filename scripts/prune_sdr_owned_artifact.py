#!/usr/bin/env python3
"""Prune only preregistered transient artifacts inside one S-DR run root."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _directory_sha256(path: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    total_bytes = 0
    count = 0
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                total_bytes += len(chunk)
        count += 1
    if count == 0:
        raise ValueError("Refusing to prune an empty state directory.")
    return digest.hexdigest(), total_bytes, count


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--kind", choices=["delta", "state"], required=True)
    parser.add_argument("--record", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).expanduser().resolve()
    target = Path(args.path).expanduser().resolve()
    target.relative_to(run_root)
    record_path = Path(args.record).expanduser().resolve()
    record_path.parent.mkdir(parents=True, exist_ok=True)
    if args.kind == "delta":
        if not target.is_file() or not target.name.endswith(
            ".action_dit_delta.pt"
        ):
            raise ValueError("Transient delta path is missing or malformed.")
        digest = _file_sha256(target)
        size_bytes = target.stat().st_size
        file_count = 1
    else:
        if not target.is_dir() or target.parent.name != "state":
            raise ValueError("Rolling state path is missing or malformed.")
        trainer_state = target / "trainer_state.json"
        payload = json.loads(trainer_state.read_text(encoding="utf-8"))
        if int(payload.get("global_step", -1)) <= 0:
            raise ValueError("Rolling state has no valid successful step.")
        digest, size_bytes, file_count = _directory_sha256(target)
    record = {
        "schema": "fastwam-sdr-pruned-artifact-v1",
        "pruned_at_utc": datetime.now(timezone.utc).isoformat(),
        "kind": args.kind,
        "path": str(target),
        "sha256": digest,
        "size_bytes": size_bytes,
        "file_count": file_count,
    }
    with record_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
    if args.kind == "delta":
        target.unlink()
    else:
        shutil.rmtree(target)
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
