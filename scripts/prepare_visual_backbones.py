#!/usr/bin/env python3
"""Prepare all registered visual backbones in one verified local directory.

Runtime model construction never calls this module and never downloads.  This
explicit command copies already verified DINOv3 S/B assets, downloads missing
pinned official releases through the existing Hugging Face login, verifies
exact size/SHA256, and atomically installs only complete files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from fastwam.models.wan22.visual_backbone import (
    PINNED_DINOV3_SOURCE_REVISION,
    PINNED_LINGBOT_VISION_SOURCE_REVISION,
    VisualBackbonePreset,
    registered_visual_backbones,
)

DEFAULT_ROOT = Path("/home/amax/data0/Checkpoints/visual-backbones")
MANIFEST_NAME = "visual_backbones_manifest.json"
MANIFEST_SCHEMA = "fastwam-visual-backbone-assets-v2"

_DIRECTORY_NAMES = {
    ("dinov3", "vits16"): "dinov3-vits16-lvd1689m",
    ("dinov3", "vitb16"): "dinov3-vitb16-lvd1689m",
    ("dinov3", "vitl16"): "dinov3-vitl16-lvd1689m",
    ("lingbot_vision", "small"): "lingbot-vision-small",
    ("lingbot_vision", "base"): "lingbot-vision-base",
    ("lingbot_vision", "large"): "lingbot-vision-large",
}
_LOCAL_COPY_CANDIDATES = {
    ("dinov3", "vits16"): (
        Path("/home/amax/data0/dinov3-vits16-pretrain-lvd1689m/model.safetensors"),
    ),
    ("dinov3", "vitb16"): (
        Path("/home/amax/data0/dinov3-vitb16-pretrain-lvd1689m/model.safetensors"),
        Path(
            "/home/amax/data0/Checkpoints/dinov3-vitb16-pretrain-lvd1689m/"
            "model.safetensors"
        ),
    ),
}


def _sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, preset: VisualBackbonePreset) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != preset.weights_size_bytes:
        raise ValueError(
            f"{path} has {size} bytes; expected {preset.weights_size_bytes}."
        )
    observed = _sha256(path)
    if observed != preset.weights_sha256:
        raise ValueError(
            f"{path} SHA256 mismatch: {observed} != {preset.weights_sha256}."
        )


def _atomic_install(
    source: Path,
    target: Path,
    preset: VisualBackbonePreset,
) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        _verify(target, preset)
        return "reused_verified"
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(
            f"Refusing to overwrite unknown temporary asset: {temporary}"
        )
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        _verify(temporary, preset)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    _verify(target, preset)
    return "installed"


def _verified_copy_source(preset: VisualBackbonePreset) -> Path | None:
    for candidate in _LOCAL_COPY_CANDIDATES.get(preset.key, ()):
        if not candidate.is_file():
            continue
        _verify(candidate, preset)
        return candidate
    return None


def _download_to_cache(preset: VisualBackbonePreset) -> Path:
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError as error:
        raise RuntimeError(
            "Asset preparation requires the `huggingface_hub` package."
        ) from error
    try:
        path = hf_hub_download(
            repo_id=preset.weights_repo_id,
            filename=preset.weights_filename,
            revision=preset.weights_revision,
        )
    except HfHubHTTPError as error:
        status = getattr(getattr(error, "response", None), "status_code", None)
        if preset.family == "dinov3" and status in {401, 403}:
            raise PermissionError(
                "DINOv3 download is gated. The current Hugging Face session "
                "does not have accepted-license/model access; no incomplete "
                "asset or manifest will be installed."
            ) from error
        raise
    candidate = Path(path).resolve()
    _verify(candidate, preset)
    return candidate


def _source_revision(path: Path, expected: str) -> str:
    if not path.is_dir():
        raise FileNotFoundError(path)
    try:
        observed = (
            subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                text=True,
            )
            .strip()
            .lower()
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"Cannot verify source revision under {path}.") from error
    if observed != expected:
        raise ValueError(f"Source revision mismatch: {observed} != {expected}.")
    return observed


def _atomic_manifest(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Refusing to overwrite unknown manifest file: {path}"
            ) from error
        if existing.get("schema") != MANIFEST_SCHEMA:
            raise ValueError(
                f"Refusing to overwrite non-{MANIFEST_SCHEMA} manifest: {path}"
            )
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(root: Path) -> dict[str, Any]:
    """Prepare all six unique weights and the nine supported preset entries."""

    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[1]
    sources = {
        "dinov3": project_root / "src/fastwam/models/dinov3",
        "lingbot_vision": project_root / "src/fastwam/models/lingbot_vision",
    }
    source_revisions = {
        "dinov3": _source_revision(
            sources["dinov3"],
            PINNED_DINOV3_SOURCE_REVISION,
        ),
        "lingbot_vision": _source_revision(
            sources["lingbot_vision"],
            PINNED_LINGBOT_VISION_SOURCE_REVISION,
        ),
    }
    assets = []
    for preset in registered_visual_backbones():
        target = root / _DIRECTORY_NAMES[preset.key] / preset.weights_filename
        if target.is_file():
            _verify(target, preset)
            source = target
            method = "reused_verified"
        else:
            source = _verified_copy_source(preset)
            source_kind = "verified_local_copy"
            if source is None:
                source = _download_to_cache(preset)
                source_kind = "pinned_huggingface_download"
            action = _atomic_install(source, target, preset)
            method = f"{source_kind}:{action}"
        assets.append(
            {
                "family": preset.family,
                "variant": preset.variant,
                "supported_input_sizes": list(preset.allowed_input_sizes),
                "native_dim": preset.native_dim,
                "depth": preset.depth,
                "patch_size": 16,
                "path": str(target),
                "sha256": preset.weights_sha256,
                "size_bytes": preset.weights_size_bytes,
                "repo_id": preset.weights_repo_id,
                "weights_revision": preset.weights_revision,
                "weights_filename": preset.weights_filename,
                "source_repository": (
                    "https://github.com/facebookresearch/dinov3"
                    if preset.family == "dinov3"
                    else "https://github.com/robbyant/lingbot-vision"
                ),
                "source_commit": source_revisions[preset.family],
                "license": preset.license_id,
                "install_method": method,
            }
        )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "PASS",
        "root": str(root),
        "created_unix_seconds": time.time(),
        "assets": assets,
        "preset_count": sum(
            len(preset.allowed_input_sizes) for preset in registered_visual_backbones()
        ),
        "runtime_downloads_allowed": False,
    }
    _atomic_manifest(root / MANIFEST_NAME, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    manifest = prepare(args.root)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "root": manifest["root"],
                "asset_count": len(manifest["assets"]),
                "preset_count": manifest["preset_count"],
                "manifest": str(Path(manifest["root"]) / MANIFEST_NAME),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
