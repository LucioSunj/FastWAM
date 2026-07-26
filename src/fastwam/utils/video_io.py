import os
from typing import Iterable

import numpy as np
from PIL import Image

from .fs import ensure_dir


def _require_imageio():
    """Import `imageio` on demand.

    `fastwam.utils.__init__` re-exports `save_mp4`, so a module-scope
    `import imageio` made *every* importer of `fastwam.utils` — including
    `fastwam.utils.logging_config`, which nearly the whole package pulls in —
    depend on a package that is only ever needed to write a preview video.
    Importing it here keeps that cost on the one code path that uses it.
    """
    try:
        import imageio
    except ImportError as exc:  # pragma: no cover - exercised via the stub test
        raise ImportError(
            "save_mp4() requires the `imageio` package and its FFMPEG plugin, "
            "which are not installed. Install them with "
            "`pip install imageio imageio-ffmpeg`."
        ) from exc
    return imageio


def _to_even_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def save_mp4(frames: Iterable[Image.Image], path: str, fps: int = 8):
    imageio = _require_imageio()
    ensure_dir(os.path.dirname(path) or ".")
    writer = imageio.get_writer(
        path,
        fps=max(fps, 1),
        codec="libx264",
        format="FFMPEG",
        pixelformat="yuv420p",
    )
    try:
        for frame in frames:
            arr = np.array(frame.convert("RGB"))
            writer.append_data(_to_even_frame(arr))
    finally:
        writer.close()
