"""Automatic accident screenshot capture.

Saves an annotated frame/image whenever an accident is detected and returns the
path so it can be stored in the detection history record.

The low-level writer here centralises the screenshots directory and filename
generation. Dedup logic (only save once per distinct accident event) is handled
by each caller so video/webcam decide when a new event starts, but this module
never overwrites an existing file (each filename has a unique timestamp).
"""

import os
import time
from typing import Optional

import cv2

# Directory name inside the Flask ``static/`` folder where screenshots live.
SCREENSHOT_FOLDER_NAME = "screenshots"


def screenshots_dir(project_root: str) -> str:
    """Return the absolute directory where accident screenshots are stored."""
    return os.path.join(project_root, "static", SCREENSHOT_FOLDER_NAME)


def build_screenshot_path(project_root: str, label: str = "accident", ext: str = ".jpg") -> str:
    """Build a fresh absolute path for a screenshot file.

    ``ext`` should be ``.jpg`` or ``.png``. The filename embeds a millisecond
    timestamp so distinct events never collide on disk.
    """
    directory = screenshots_dir(project_root)
    os.makedirs(directory, exist_ok=True)
    stamp = int(time.time() * 1000)
    return os.path.join(directory, f"{label}_{stamp}{ext}")


def save_frame_screenshot(
    frame,
    project_root: str,
    label: str = "accident",
    ext: str = ".jpg",
    quality: int = 90,
) -> Optional[str]:
    """Persist an annotated frame as an accident screenshot.

    Returns the absolute path on success or ``None`` if the frame could not be
    written. This never raises: any OpenCV / I/O failure degrades gracefully so
    detection itself is never interrupted by a screenshot failure.
    """
    if frame is None:
        return None

    path = build_screenshot_path(project_root, label, ext)
    try:
        if ext.lower() in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
        else:
            params = [cv2.IMWRITE_PNG_COMPRESSION, 0]
        if not cv2.imwrite(path, frame, params):
            return None
        return path
    except Exception:
        return None