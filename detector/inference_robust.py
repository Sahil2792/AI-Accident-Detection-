"""Domain-shift-robust YOLO inference helper.

Applied ONLY at the ``model(...)`` call sites so the custom trained ``best.pt``
model generalizes to unseen datasets, camera angles, cameras, and lighting
conditions WITHOUT retraining. Nothing else in the codebase is modified.

The wrapper implements four inference-side techniques:

1. **Dynamic confidence & IoU thresholds** — a lower confidence (default 0.15
   for the custom model, 0.20 for the standard COCO model) and NMS IoU 0.45 so
   the model is far less strict and emits detections it previously discarded
   on unfamiliar data.
2. **Test-Time Augmentation (TTA)** — ``augment=True`` on the model call, which
   evaluates at multiple scales plus horizontal flips and merges predictions.
3. **OpenCV enhancement (CLAHE)** — adaptive histogram equalization on the
   luminance (L) channel balances drastically different exposure/contrast
   between training and production datasets.
4. **Multi-scale inference** — additional ``imgsz`` passes (e.g. 1280) whose
   detections are merged back into a single ``Results`` object with greedy NMS.

All settings are env-overridable (e.g. ``ROBUST_CONF``, ``ROBUST_TTA``,
``ROBUST_MULTI_SCALE_SIZES``) so the app can be tuned per deployment:

    ROBUST_CONF=0.15                     # custom accident model confidence
    ROBUST_STANDARD_CONF=0.20          # standard COCO model confidence
    ROBUST_IOU=0.45                     # NMS IoU threshold
    ROBUST_TTA=1                        # enable TTA (augment=True)
    ROBUST_ENHANCE=1                    # enable CLAHE preprocessing
    ROBUST_IMGSZ=640                    # primary inference resolution
    ROBUST_MULTI_SCALE_SIZES=1280       # extra scales (comma-separated) merged
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Tunable defaults (each can be overridden with an environment variable)
# ---------------------------------------------------------------------------
ROBUST_CONF = float(os.environ.get("ROBUST_CONF", "0.15"))
ROBUST_STANDARD_CONF = float(os.environ.get("ROBUST_STANDARD_CONF", "0.20"))
ROBUST_IOU = float(os.environ.get("ROBUST_IOU", "0.45"))
ROBUST_TTA = os.environ.get("ROBUST_TTA", "1").strip().lower() in ("1", "true", "yes", "on")
ROBUST_ENHANCE = os.environ.get("ROBUST_ENHANCE", "1").strip().lower() in ("1", "true", "yes", "on")
ROBUST_IMGSZ = int(os.environ.get("ROBUST_IMGSZ", "640"))
ROBUST_MULTI_SCALE_SIZES = tuple(
    int(part.strip())
    for part in os.environ.get("ROBUST_MULTI_SCALE_SIZES", "1280").split(",")
    if part.strip().lstrip("-").isdigit()
)

# ---------------------------------------------------------------------------
# Lightweight realtime defaults (used by VIDEO + WEBCAM frame loops).
#
# These deliberately disable TTA and multi-scale and cap the resolution at 640,
# so live streams and long videos run at practical FPS instead of taking
# 10-15 minutes. Static image detection (detect.py) keeps the heavier ROBUST_*
# settings above for maximum generalisation accuracy.
# ---------------------------------------------------------------------------
REALTIME_CONF = float(os.environ.get("REALTIME_CONF", ROBUST_CONF))
REALTIME_STANDARD_CONF = float(os.environ.get("REALTIME_STANDARD_CONF", ROBUST_STANDARD_CONF))
REALTIME_IOU = float(os.environ.get("REALTIME_IOU", ROBUST_IOU))
REALTIME_IMGSZ = int(os.environ.get("REALTIME_IMGSZ", "640"))
# TTA + multi-scale are DISABLED by default for realtime streams.
REALTIME_TTA = os.environ.get("REALTIME_TTA", "0").strip().lower() in ("1", "true", "yes", "on")
REALTIME_ENHANCE = os.environ.get("REALTIME_ENHANCE", "0").strip().lower() in ("1", "true", "yes", "on")
# Multi-scale is off for realtime streams; pass the empty tuple explicitly so a
# configured ROBUST_MULTI_SCALE_SIZES env var can never leak into video paths.
REALTIME_MULTI_SCALE_SIZES: tuple = ()
# ---------------------------------------------------------------------------
# OpenCV preprocessing / enhancement (technique 3)
# ---------------------------------------------------------------------------
def enhance_image(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """Apply CLAHE (adaptive histogram equalization) to balance local contrast.

    Works for both BGR color images (L-channel of LAB) and single-channel gray
    images. The spatial dimensions are unchanged, so boxes stay pixel-aligned.
    """
    if image is None or image.size == 0:
        return image

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    try:
        if image.ndim == 2:  # grayscale
            return clahe.apply(image)
        if image.ndim == 3 and image.shape[2] == 3:  # BGR color
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            l_channel = clahe.apply(l_channel)
            enhanced = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
            return enhanced
    except cv2.error:
        # Any OpenCV failure should never take down inference; degrade gracefully.
        pass
    return image


# ---------------------------------------------------------------------------
# Multi-scale merge utilities (technique 4)
# ---------------------------------------------------------------------------
def _bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """IoU of two boxes in xyxy format."""
    inter_x1 = max(float(box_a[0]), float(box_b[0]))
    inter_y1 = max(float(box_a[1]), float(box_b[1]))
    inter_x2 = min(float(box_a[2]), float(box_b[2]))
    inter_y2 = min(float(box_a[3]), float(box_b[3]))
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union = area_a + area_b - inter_area
    return 0.0 if union <= 0.0 else inter_area / union


def _greedy_nms(candidates: List[tuple], iou_threshold: float) -> List[tuple]:
    """Greedy NMS over (score, class_id, xyxy) candidates, score descending."""
    kept: List[tuple] = []
    for score, class_id, box in sorted(candidates, key=lambda item: -item[0]):
        if any(_bbox_iou(box, kept_box) >= iou_threshold for _, _, kept_box in kept):
            continue
        kept.append((score, class_id, box))
    return kept
def _make_lightweight_result(names: Dict[int, str], data: torch.Tensor):
    """Minimal container exposing .names/.boxes(.conf|.cls|.xyxy) for merge fallback."""

    class _MergedBoxes:
        def __init__(self, arr):
            self._arr = arr

        @property
        def xyxy(self):
            return self._arr[:, :4]

        @property
        def conf(self):
            return self._arr[:, 4]

        @property
        def cls(self):
            return self._arr[:, 5]

    class _MergedResult:
        def __init__(self, box_data, names_map):
            self.names = names_map
            self.boxes = _MergedBoxes(box_data)

    return _MergedResult(data, names)


def _merge_result_groups(groups: List[List[Any]], iou_threshold: float) -> List[Any]:
    """Merge per-scale prediction groups into a single results list via NMS."""
    groups = [g for g in groups if g]
    if not groups:
        return []

    # Fast path: single group -> keep untouched so mock/legacy models work.
    if len(groups) == 1:
        return groups[0]

    candidates: List[tuple] = []
    names: Dict[int, str] = {}
    path = None
    orig_shape = None
    orig_img = None
    has_real_boxes = False

    for group in groups:
        for result in group:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = getattr(boxes, "xyxy", None)
            confs = getattr(boxes, "conf", None)
            classes = getattr(boxes, "cls", None)
            if xyxy is None or confs is None or classes is None:
                continue
            conf_values = confs.tolist() if hasattr(confs, "tolist") else list(confs)
            cls_values = classes.tolist() if hasattr(classes, "tolist") else list(classes)
            xyxy_values = xyxy.tolist() if hasattr(xyxy, "tolist") else list(xyxy)
            if len(xyxy_values) == 0:
                continue
            has_real_boxes = True
            result_names = getattr(result, "names", None)
            if result_names:
                names.update(result_names)
            if path is None:
                path = getattr(result, "path", None)
            if orig_shape is None:
                orig_shape = getattr(result, "orig_shape", None)
            if orig_img is None:
                orig_img = getattr(result, "orig_img", None)
            for idx in range(len(xyxy_values)):
                candidates.append(
                    (float(conf_values[idx]), int(cls_values[idx]), [float(v) for v in xyxy_values[idx]])
                )

    # Groups without box tensors (mocks / old API) -> keep the primary group.
    if not has_real_boxes:
        return groups[0]

    kept = _greedy_nms(candidates, iou_threshold=iou_threshold)
    if not kept:
        return groups[0]

    data = np.asarray(
        [[x1, y1, x2, y2, score, float(class_id)] for score, class_id, (x1, y1, x2, y2) in kept],
        dtype=np.float32,
    )
    if orig_shape is None and orig_img is not None:
        orig_shape = orig_img.shape[:2]

    # Prefer a real ``ultralytics.engine.results.Results`` object so every
    # downstream consumer (normalize_yolo_results, plot, .boxes.*) keeps working.
    try:
        from ultralytics.engine.results import Boxes, Results  # type: ignore

        if orig_img is None:
            height, width = orig_shape if orig_shape is not None else (640, 640)
            orig_img = np.zeros((int(height), int(width), 3), dtype=np.uint8)
        merged_boxes = Boxes(torch.tensor(data), orig_shape or orig_img.shape[:2])
        return [Results(orig_img=orig_img, path=path, names=names, boxes=merged_boxes)]
    except Exception:
        return [_make_lightweight_result(names, torch.tensor(data))]
# ---------------------------------------------------------------------------
# Main inference entrypoint
# ---------------------------------------------------------------------------
def run_robust_inference(
    model: Any,
    source: Union[str, os.PathLike, np.ndarray],
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    imgsz: Optional[int] = None,
    augment: Optional[bool] = None,
    enhance: Optional[bool] = None,
    multi_scale_sizes: Optional[Sequence[Union[int, str]]] = None,
    stream: bool = False,
) -> List[Any]:
    """Run the YOLO model with dynamic thresholds, TTA, CLAHE and multi-scale.

    Args:
        model: Ultralytics YOLO model instance.
        source: Image path or a numpy (BGR) frame.
        conf: Model confidence threshold (default ROBUST_CONF).
        iou: NMS IoU threshold (default ROBUST_IOU).
        imgsz: Primary inference resolution in pixels (default ROBUST_IMGSZ).
        augment: Enable TTA / augment=True (default ROBUST_TTA).
        enhance: Enable CLAHE preprocessing (default ROBUST_ENHANCE).
        multi_scale_sizes: Extra sizes to run in addition to ``imgsz``
            (default ROBUST_MULTI_SCALE_SIZES).
        stream: Keep the stream=False contract of the original call sites.

    Returns:
        List of ultralytics-compatible Results objects (length 1 per image),
        safe for ``normalize_yolo_results`` and friends.
    """
    conf = ROBUST_CONF if conf is None else float(conf)
    iou = ROBUST_IOU if iou is None else float(iou)
    imgsz = ROBUST_IMGSZ if imgsz is None else int(imgsz)
    augment = ROBUST_TTA if augment is None else bool(augment)
    enhance = ROBUST_ENHANCE if enhance is None else bool(enhance)

    raw_scales = [imgsz]
    if multi_scale_sizes:
        raw_scales.extend(int(size) for size in multi_scale_sizes)
    scales = tuple(sorted({size for size in raw_scales if size >= 32}))
    if not scales:
        scales = (imgsz,)

    # Load numpy input for CLAHE when a path is given (coordinates stay aligned
    # because enhancement does not resize the frame). Ultralytics patches
    # cv2.imread to raise on missing files, so guard and fall back to the path.
    path = str(source) if isinstance(source, (str, os.PathLike)) else None
    if isinstance(source, (str, os.PathLike)):
        try:
            loaded_image = cv2.imread(path)
        except Exception:
            loaded_image = None
        feed = loaded_image if loaded_image is not None else source
    else:
        feed = source

    if isinstance(feed, np.ndarray) and enhance and feed.size:
        feed = enhance_image(feed)

    groups: List[List[Any]] = []
    for size in scales:
        groups.append(
            _run_once(model, feed, conf=conf, iou=iou, imgsz=size, augment=augment, stream=stream)
        )
    return _merge_result_groups(groups, iou_threshold=iou)


def _run_once(
    model: Any,
    source: Any,
    conf: float,
    iou: float,
    imgsz: int,
    augment: bool,
    stream: bool,
) -> List[Any]:
    """Single model call with graceful fallback for models lacking new kwargs."""
    call_kwargs = dict(conf=conf, iou=iou, imgsz=imgsz, augment=augment, stream=stream, verbose=False)
    try:
        raw = model(source, **call_kwargs)
    except TypeError:
        # Old Ultralytics versions / mocked models may not support augment/verbose.
        # Drop only the unsupported kwargs and retry once so this never breaks.
        call_kwargs.pop("augment", None)
        call_kwargs.pop("verbose", None)
        raw = model(source, **call_kwargs)
    return list([] if raw is None else raw)