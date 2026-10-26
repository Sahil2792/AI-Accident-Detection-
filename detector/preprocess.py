"""Shared image pre-processing helpers for memory-constrained deployments.

Render's free tier gives a single web process only 512 MB of RAM.  Before any
YOLO inference we therefore downscale oversized uploads so PyTorch never has to
letterbox a giant 12-megapixel photo into a 640x640 tensor (that temporary
buffer alone can be hundreds of MB).  Detection coordinates always stay in the
pixel space of the *saved* file, so the annotated image stays perfectly
aligned.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2

# Longest edge allowed before an upload is downscaled for inference.
# Env-tunable: MAX_INFERENCE_IMAGE_DIM=1280
MAX_INFERENCE_IMAGE_DIM = int(os.environ.get("MAX_INFERENCE_IMAGE_DIM", "1280"))


def downscale_image_if_needed(image_path: str, max_dim: Optional[int] = None) -> Tuple[str, bool]:
    """Resize ``image_path`` in place when its longest edge exceeds ``max_dim``.

    Args:
        image_path: Path to a saved image file (JPG/PNG/BMP/WebP...).
        max_dim: Longest edge threshold (defaults to MAX_INFERENCE_IMAGE_DIM).

    Returns:
        (path, changed) - the (possibly rewritten) path plus a flag that says
        whether the file was resized.
    """
    limit = int(max_dim or MAX_INFERENCE_IMAGE_DIM)
    if limit <= 0:
        return image_path, False
    if not image_path or not os.path.exists(image_path):
        return image_path, False

    try:
        img = cv2.imread(image_path)
        if img is None:
            return image_path, False
        height, width = img.shape[:2]
        longest = max(height, width)
        if longest <= limit:
            return image_path, False

        scale = limit / float(longest)
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        # Release the original buffer before writing to keep the peak low.
        del img
        ok, buf = cv2.imencode(_extension_for(image_path), resized)
        if not ok:
            return image_path, False
        del resized
        with open(image_path, "wb") as handle:
            handle.write(buf.tobytes())
        return image_path, True
    except cv2.error:
        return image_path, False
    except OSError:
        return image_path, False


def _extension_for(image_path: str) -> str:
    """Map a filename to a cv2.imencode extension (defaults to JPEG)."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        return ".jpg"
    if ext == ".png":
        return ".png"
    if ext == ".webp":
        return ".webp"
    if ext == ".bmp":
        return ".bmp"
    return ".jpg"
