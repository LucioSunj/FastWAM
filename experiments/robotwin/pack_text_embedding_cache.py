import argparse
import hashlib
import json
from pathlib import Path

PACKED_CACHE_BIN = "packed_cache.bin"
PACKED_CACHE_INDEX = "packed_cache.index.jsonl"
ENCODER_ID = "wan22ti2v5b"


def cache_name(prompt: str, context_len: int) -> str:
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return f"{hashed}.t5_len{context_len}.{ENCODER_ID}.pt"


def load_index(index_path: Path) -> set[str]:
    names: set[str] = set()
    if not index_path.exists():
        return names
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                names.add(str(json.loads(line)["name"]))
    return names


def append_record(bin_f, index_f, name: str, payload: bytes) -> None:
    offset = bin_f.tell()
    bin_f.write(payload)
    bin_f.flush()
    index_f.write(json.dumps({"name": name, "offset": offset, "length": len(payload)}, sort_keys=True) + "\n")
    index_f.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack FastWAM text embedding .pt files into one append-only cache.")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--prompts-json", required=True)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--delete-source", action="store_true")
    parser.add_argument("--delete-unlisted", action="store_true")
    parser.add_argument("--progress-every", type=int, default=500)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    prompts_json = Path(args.prompts_json).expanduser().resolve()
    prompts = json.loads(prompts_json.read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or not all(isinstance(item, str) for item in prompts):
        raise ValueError(f"Invalid prompts JSON: {prompts_json}")

    names = [cache_name(prompt, args.context_len) for prompt in prompts]
    wanted = set(names)
    bin_path = cache_dir / PACKED_CACHE_BIN
    index_path = cache_dir / PACKED_CACHE_INDEX
    packed_names = load_index(index_path)

    packed = 0
    skipped = 0
    missing = 0
    deleted_sources = 0
    cache_dir.mkdir(parents=True, exist_ok=True)
    with bin_path.open("ab") as bin_f, index_path.open("a", encoding="utf-8") as index_f:
        for idx, name in enumerate(names, start=1):
            source_path = cache_dir / name
            if name in packed_names:
                skipped += 1
                if args.delete_source and source_path.exists():
                    source_path.unlink()
                    deleted_sources += 1
                continue
            if not source_path.exists():
                missing += 1
                continue
            payload = source_path.read_bytes()
            append_record(bin_f, index_f, name, payload)
            packed_names.add(name)
            packed += 1
            if args.delete_source:
                source_path.unlink()
                deleted_sources += 1
            if args.progress_every > 0 and idx % args.progress_every == 0:
                print(
                    f"packed_progress {idx}/{len(names)} packed={packed} skipped={skipped} "
                    f"missing={missing} deleted_sources={deleted_sources}",
                    flush=True,
                )

    deleted_unlisted = 0
    if args.delete_unlisted:
        for path in cache_dir.glob(f"*.t5_len{args.context_len}.{ENCODER_ID}.pt"):
            if path.name not in wanted:
                path.unlink()
                deleted_unlisted += 1
                if args.progress_every > 0 and deleted_unlisted % args.progress_every == 0:
                    print(f"deleted_unlisted {deleted_unlisted}", flush=True)

    print(
        "pack_done "
        f"prompts={len(prompts)} packed={packed} skipped={skipped} missing={missing} "
        f"deleted_sources={deleted_sources} deleted_unlisted={deleted_unlisted} "
        f"bin={bin_path} index={index_path}",
        flush=True,
    )
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
