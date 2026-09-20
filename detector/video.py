"""Video detection pipeline using the custom trained accident model (best.pt).

The custom accident model is loaded DIRECTLY with standard Ultralytics YOLO
syntax::

    from ultralytics import YOLO
    accident_model = YOLO("runs/detect/models/accident_detector/weights/best.pt")

and is applied to every sampled video frame. The standard COCO model
(yolov8n.pt) is also applied to the same frames so everyday objects (cars,
trucks, persons) are still counted and can be used to validate low-confidence
accident boxes via collision/overlap geometry.

For each frame the pipeline:

  1. Runs the standard YOLO model -> bounding boxes for everyday objects.
  2. Runs the custom best.pt model -> "accident" bounding boxes.
  3. Validates accident boxes (confidence + collision evidence).
  4. Combines detections (accident boxes take priority).
  5. Draws bounding boxes + labels onto the frame.
  6. Writes the annotated frame to the output video and records the detections.

Only ``detector/video.py`` is changed by this logic - image and webcam
detection (``detector/image.py`` / ``detector/webcam.py``) are untouched.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import cv2
from ultralytics import YOLO

from detector.detect import (
    ACCIDENT_CLASSES,
    ACCIDENT_MIN_CONFIDENCE,
    INFERENCE_IOU,
    INFERENCE_SIZE,
    STANDARD_INFERENCE_CONFIDENCE,
    STANDARD_MODEL_PATH,
    STANDARD_OBJECT_CLASSES,
    combine_detections_with_accident_priority,
    derive_status,
    get_accident_confidence,
    get_detection_color,
    normalize_yolo_results,
    summarize_detections,
    validate_accident_detections,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from detector.inference_robust import (
    REALTIME_CONF,
    REALTIME_ENHANCE,
    REALTIME_IMGSZ,
    REALTIME_IOU,
    REALTIME_MULTI_SCALE_SIZES,
    REALTIME_STANDARD_CONF,
    REALTIME_TTA,
    run_robust_inference,
)
from detector.severity import classify_severity
from detector.screenshot import save_frame_screenshot

# ---------------------------------------------------------------------------
# Custom accident model weights.
# The EXACT path is used in the YOLO(...) call in _load_accident_model().
# ---------------------------------------------------------------------------
ACCIDENT_WEIGHTS_PATH = os.path.join(
    PROJECT_ROOT, "runs", "detect", "models", "accident_detector", "weights", "best.pt"
)

# Process every Nth frame for speed. 1 = every frame, 2 = every other frame etc.
# Both the standard model AND the custom accident model are applied to every
# sampled frame.
FRAME_SKIP = int(os.environ.get("VIDEO_FRAME_SKIP", "2"))

# Custom accident model inference confidence for video processing.
# Lowered to 0.50 because video frames suffer from motion blur, compression
# artifacts, and low-resolution CCTV footage, which makes the custom model
# output lower-confidence accident boxes than on still images. A 0.75
# threshold filtered out genuine accidents before they could reach the
# temporal confirmation logic, making best.pt appear to "miss" detections.
# Increase the video acceptance threshold to reduce false positives in motion-blurred frames.
ACCIDENT_VIDEO_CONFIDENCE = float(os.environ.get("ACCIDENT_VIDEO_CONFIDENCE", "0.65"))

# Minimum number of consecutive sampled frames an accident must appear in
# before it is confirmed as a genuine accident (temporal confirmation prevents
# a single-frame false positive from being flagged as an accident).
# Set to 1 so a genuine accident that fires on a SINGLE sampled frame is not
# dropped. The previous requirement of 2 consecutive sampled frames
# (FRAME_SKIP=2 -> only every other frame is inferred) suppressed genuine
# accidents in short video clips, which is why video uploads only showed
# standard classes (car/person/...) and never the accident class.
ACCIDENT_MIN_CONSECUTIVE_FRAMES = int(os.environ.get("ACCIDENT_MIN_CONSECUTIVE_FRAMES", "1"))

# Maximum gap (in sampled frames) allowed between accident detections while
# still counting them as the same sustained event.
ACCIDENT_MAX_FRAME_GAP = int(os.environ.get("ACCIDENT_MAX_FRAME_GAP", "2"))


class _TemporalBoxSmoother:
    """Smooth bounding box positions across frames to reduce flickering.

    Matches detections between consecutive frames using IoU and applies an
    exponential moving average (EMA) to the box coordinates. Tracks are kept
    alive for a few frames so a brief detection dropout does not make a box
    flicker on/off.
    """

    def __init__(self, alpha: float = 0.6, iou_threshold: float = 0.3, max_track_age: int = 5):
        self.alpha = alpha
        self.iou_threshold = iou_threshold
        self.max_track_age = max_track_age
        self.tracks: List[Dict[str, Any]] = []

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Match new detections to existing tracks and return smoothed boxes."""
        smoothed: List[Dict[str, Any]] = []
        unmatched_track_indices = list(range(len(self.tracks)))

        for detection in detections:
            bbox = detection.get("bbox")
            if bbox is None:
                smoothed.append(detection)
                continue

            best_track_idx = -1
            best_iou = 0.0
            for track_idx in unmatched_track_indices:
                track = self.tracks[track_idx]
                if track["class_name"] != detection["class_name"]:
                    continue
                iou = self._iou(bbox, track["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_track_idx = track_idx

            if best_track_idx >= 0 and best_iou >= self.iou_threshold:
                track = self.tracks[best_track_idx]
                # EMA smoothing of box coordinates.
                track["bbox"] = {
                    "x1": self.alpha * bbox["x1"] + (1 - self.alpha) * track["bbox"]["x1"],
                    "y1": self.alpha * bbox["y1"] + (1 - self.alpha) * track["bbox"]["y1"],
                    "x2": self.alpha * bbox["x2"] + (1 - self.alpha) * track["bbox"]["x2"],
                    "y2": self.alpha * bbox["y2"] + (1 - self.alpha) * track["bbox"]["y2"],
                }
                track["age"] = 0
                track["confidence"] = detection["confidence"]
                unmatched_track_indices.remove(best_track_idx)
                smoothed.append({**detection, "bbox": track["bbox"]})
            else:
                # New detection: start a new track.
                self.tracks.append(
                    {
                        "class_name": detection["class_name"],
                        "bbox": dict(bbox),
                        "age": 0,
                        "confidence": detection["confidence"],
                    }
                )
                smoothed.append(detection)

        # Age unmatched tracks and drop stale ones.
        for track_idx in unmatched_track_indices:
            self.tracks[track_idx]["age"] += 1
        self.tracks = [t for t in self.tracks if t["age"] <= self.max_track_age]

        return smoothed

    @staticmethod
    def _iou(box_a: Dict[str, float], box_b: Dict[str, float]) -> float:
        """Compute IoU between two bbox dicts."""
        ax1, ay1, ax2, ay2 = box_a["x1"], box_a["y1"], box_a["x2"], box_a["y2"]
        bx1, by1, bx2, by2 = box_b["x1"], box_b["y1"], box_b["x2"], box_b["y2"]

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - intersection
        if union <= 0:
            return 0.0
        return intersection / union


class _TemporalAccidentConfirmer:
    """Confirm accident detections only when they persist across multiple frames.

    A single-frame accident prediction (e.g. a normal car briefly triggering
    the accident model) is NOT confirmed. An accident is only accepted after it
    has been detected in at least ``min_consecutive_frames`` sampled frames,
    allowing small gaps (up to ``max_frame_gap``) so brief flickers do not
    reset the count.
    """

    def __init__(self, min_consecutive_frames: int = 3, max_frame_gap: int = 2):
        self.min_consecutive_frames = max(1, min_consecutive_frames)
        self.max_frame_gap = max(0, max_frame_gap)
        self._consecutive_count = 0
        self._last_seen_frame = -1

    def update(self, has_accident: bool, frame_index: int) -> bool:
        """Update the temporal state and return whether the accident is confirmed."""
        if has_accident:
            if self._last_seen_frame >= 0 and (frame_index - self._last_seen_frame) <= (self.max_frame_gap + 1):
                self._consecutive_count += 1
            else:
                self._consecutive_count = 1
            self._last_seen_frame = frame_index
        else:
            if self._last_seen_frame >= 0 and (frame_index - self._last_seen_frame) > self.max_frame_gap:
                self._consecutive_count = 0
                self._last_seen_frame = -1

        return self._consecutive_count >= self.min_consecutive_frames


def _load_yolo(model_path: str) -> YOLO:
    """Load an Ultralytics YOLO model with torch compatibility handling.

    The standard Ultralytics syntax ``YOLO(model_path)`` is used here. On newer
    PyTorch versions (>= 2.6) ``torch.load`` defaults to ``weights_only=True``,
    which can refuse YOLO checkpoints; we temporarily enable the legacy loader
    exactly like the rest of the project does.
    """
    import torch
    from ultralytics.nn.tasks import DetectionModel

    original_torch_load = torch.load

    def load_with_compatibility(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    safe_globals = getattr(torch.serialization, "safe_globals", None)
    try:
        torch.load = load_with_compatibility
        if safe_globals is not None:
            with safe_globals([DetectionModel]):
                return YOLO(model_path)
        add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
        if add_safe_globals is not None:
            add_safe_globals([DetectionModel])
        return YOLO(model_path)
    finally:
        torch.load = original_torch_load


def _load_accident_model() -> YOLO:
    """Load the CUSTOM trained accident model with standard YOLO syntax.

    Uses the EXACT custom trained weights file::

        runs/detect/models/accident_detector/weights/best.pt

    and verifies the model exposes an ``accident`` class so the standard COCO
    model can never silently replace it.
    """
    if not os.path.exists(ACCIDENT_WEIGHTS_PATH):
        raise FileNotFoundError(
            f"Custom accident model weights not found: {ACCIDENT_WEIGHTS_PATH}. "
            "Please restore 'best.pt' at 'runs/detect/models/accident_detector/weights/best.pt' "
            "before processing videos."
        )

    # Standard Ultralytics YOLO syntax with the exact custom weights path.
    accident_model = _load_yolo(ACCIDENT_WEIGHTS_PATH)

    names = getattr(accident_model, "names", {}) or {}
    normalized_names = {str(name).lower() for name in names.values()}
    if "accident" not in normalized_names:
        raise ValueError(
            f"Model at {ACCIDENT_WEIGHTS_PATH} does not expose an 'accident' class "
            f"(found: {sorted(normalized_names)}). Please provide the trained "
            "accident detection weights."
        )

    print(
        f"[VIDEO] Custom accident model loaded with Ultralytics YOLO: "
        f"{ACCIDENT_WEIGHTS_PATH} (classes: {sorted(normalized_names)})"
    )
    return accident_model


def process_video(video_path: str, output_path: str | None = None) -> Dict[str, Any]:
    """Process a video file using the custom accident model (best.pt).

    Pipeline:
    1. Load the custom accident model directly with ``YOLO(best.pt)``.
    2. Load the standard COCO model (yolov8n.pt) for object counting.
    3. Run BOTH models on every sampled frame (frame skipping keeps it fast).
    4. Validate accident boxes, combine detections, and draw bounding boxes.
    5. Return detections, summary, status, and accident confidence.

    Args:
        video_path: Path to the input video file.
        output_path: Optional path where the annotated video is written.

    Returns:
        A dictionary with the detected objects (bounding boxes), summary,
        accident confidence, and metadata.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Use the EXACT original FPS / dimensions from the source video.
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0 or width == 0 or height == 0:
        raise ValueError("Invalid video properties")

    # ---- Standard COCO YOLO model (everyday object counting) ----
    standard_model = _load_yolo(STANDARD_MODEL_PATH)

    # ---- Custom trained accident model (best.pt) ----
    # Explicitly loaded with ultralytics YOLO and applied to every frame below.
    accident_model = _load_accident_model()
    print(
        f"[VIDEO] Dual-model pipeline ACTIVE: "
        f"standard={os.path.basename(STANDARD_MODEL_PATH)} -> everyday objects, "
        f"accident={os.path.basename(ACCIDENT_WEIGHTS_PATH)} -> accident boxes. "
        f"Both models run on every sampled frame."
    )

    # Temporal smoother stabilizes bounding boxes across frames.
    box_smoother = _TemporalBoxSmoother()
    # Temporal accident confirmer ensures accidents are flagged only when they
    # persist across several sampled frames.
    accident_confirmer = _TemporalAccidentConfirmer(
        min_consecutive_frames=ACCIDENT_MIN_CONSECUTIVE_FRAMES,
        max_frame_gap=ACCIDENT_MAX_FRAME_GAP,
    )

    all_detections: List[Dict[str, Any]] = []
    summary: Dict[str, int] = {}
    frame_count = 0
    processed_frames = 0
    accident_frames: List[int] = []
    sampled_frame_index = 0
    accident_alerted = False
    accident_screenshot_saved = False
    accident_screenshot_path = None

    start_time = time.time()

    # Create the annotated output video if requested.
    output_video = None
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        for codec_code in ("avc1", "H264", "mp4v"):
            fourcc = cv2.VideoWriter_fourcc(*codec_code)
            # Strictly use the original FPS to keep 1:1 video duration.
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if writer and writer.isOpened():
                output_video = writer
                break
            if writer:
                writer.release()

    # Remember the last drawn detections so skipped frames can still render
    # the same boxes (avoids flickering on/off between inference frames).
    last_drawn_detections: List[Dict[str, Any]] = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Frame skipping: still write skipped frames, but draw the last
            # known smoothed detections to eliminate flicker.
            if frame_count % FRAME_SKIP != 0:
                if last_drawn_detections:
                    frame = _draw_detections_on_frame(frame, last_drawn_detections)
                if output_video:
                    output_video.write(frame)
                continue

            processed_frames += 1
            sampled_frame_index += 1

            # ---- Run BOTH models on the current frame ----
            # REALTIME inference: lightweight imgsz=640, TTA disabled
            # (augment=False) and NO multi-scale, so video processing stays
            # smooth and completes in a few seconds instead of 10-15 minutes.
            standard_raw = run_robust_inference(
                standard_model,
                frame,
                conf=REALTIME_STANDARD_CONF,
                iou=REALTIME_IOU,
                imgsz=REALTIME_IMGSZ,
                augment=REALTIME_TTA,
                enhance=REALTIME_ENHANCE,
                multi_scale_sizes=list(REALTIME_MULTI_SCALE_SIZES),
            )
            # Custom accident model (best.pt) applied to THIS frame.
            accident_raw = run_robust_inference(
                accident_model,
                frame,
                conf=REALTIME_CONF,
                iou=REALTIME_IOU,
                imgsz=REALTIME_IMGSZ,
                augment=REALTIME_TTA,
                enhance=REALTIME_ENHANCE,
                multi_scale_sizes=list(REALTIME_MULTI_SCALE_SIZES),
            )

            standard_results = list([] if standard_raw is None else standard_raw)
            accident_results = list([] if accident_raw is None else accident_raw)

            standard_detections = _extract_frame_detections(
                standard_results,
                allowed_classes=STANDARD_OBJECT_CLASSES,
                source_model="yolov8n.pt",
            )
            raw_accident_detections = _extract_frame_detections(
                accident_results,
                allowed_classes=ACCIDENT_CLASSES,
                source_model="best.pt",
            )

            # Keep the trained accident model's OWN detections as-is.
            # The image/webcam pipeline's validate_accident_detections() drops
            # any accident box that overlaps EXACTLY ONE standard vehicle. That
            # heuristic was designed for still images and filters out the most
            # common video case - a single crashed car - which is why video
            # uploads showed ONLY standard objects (car/person/...) and never
            # the "accident" class. The trained model's own confidence filter
            # (ACCIDENT_VIDEO_CONFIDENCE = 0.50) is the only filter applied here.
            accident_detections = [
                detection
                for detection in raw_accident_detections
                if float(detection.get("confidence", 0.0)) >= ACCIDENT_VIDEO_CONFIDENCE
            ]

            # STRICT multi-vehicle collision validation - identical rules to the
            # image pipeline: an accident box survives ONLY when it overlaps
            # >=2 DISTINCT vehicles that collide with EACH OTHER, with detector
            # confidence >= ACCIDENT_MIN_CONFIDENCE (0.65). Single moving cars,
            # roadside objects and shadows are rejected here. Pure geometry
            # math on already-computed boxes -> zero impact on realtime FPS.
            accident_detections = validate_accident_detections(
                accident_detections,
                standard_detections,
                min_confidence=ACCIDENT_MIN_CONFIDENCE,
            )

            # Combine with accident boxes taking priority.
            frame_detections = combine_detections_with_accident_priority(
                standard_detections, accident_detections
            )

            # Smooth boxes across frames to reduce flickering.
            frame_detections = box_smoother.update(frame_detections)

            # Temporal accident confirmation (multiple consecutive frames).
            has_accident_this_frame = any(
                str(d.get("class_name", "")).lower() == "accident" for d in frame_detections
            )
            accident_confirmed = accident_confirmer.update(has_accident_this_frame, sampled_frame_index)

            # Reset the per-event screenshot guard once the accident is no
            # longer confirmed, so the NEXT distinct accident event can save a
            # fresh screenshot (dedup: only one screenshot per event).
            if not accident_confirmed:
                accident_screenshot_saved = False

            # Keep all standard detections; keep accident detections only when
            # temporally confirmed.
            confirmed_detections = [
                d for d in frame_detections
                if str(d.get("class_name", "")).lower() != "accident" or accident_confirmed
            ]

            if confirmed_detections:
                # Draw the bounding boxes (red rectangles for accidents).
                frame = _draw_detections_on_frame(frame, confirmed_detections)
                last_drawn_detections = confirmed_detections
                for detection in confirmed_detections:
                    class_name = detection["class_name"]
                    summary[class_name] = summary.get(class_name, 0) + 1
                    all_detections.append({**detection, "frame": frame_count})
                    if class_name == "accident":
                        accident_frames.append(frame_count)
                        # Auto-screenshot: save the confirmed accident frame once
                        # per event (dedup) after the red box has been drawn.
                        if accident_confirmed and not accident_screenshot_saved:
                            saved = save_frame_screenshot(frame, PROJECT_ROOT)
                            if saved is not None:
                                accident_screenshot_path = saved
                                accident_screenshot_saved = True
                        if not accident_alerted:
                            print(
                                f"[VIDEO] Accident CONFIRMED on frame {frame_count} - "
                                f"custom model (best.pt) fired, red 'accident' box drawn."
                            )
                            accident_alerted = True

            if output_video:
                output_video.write(frame)

        inference_time = round(time.time() - start_time, 2)

        if output_video:
            output_video.release()

        cap.release()

        accident_confidence = get_accident_confidence(all_detections)
        avg_confidence = (
            sum(d["confidence"] for d in all_detections) / len(all_detections)
            if all_detections
            else 0.0
        )
        summary = summarize_detections(all_detections)

        # Count UNIQUE accident incidents (FPS-aware gap threshold ~1.5 sec).
        accident_incidents = _count_unique_accident_incidents(
            accident_frames, gap_threshold=max(5, int(fps * 1.5))
        )
        summary["accident"] = accident_incidents
        total_standard_objects = sum(
            1 for detection in all_detections
            if str(detection.get("class_name", "")).lower() != "accident"
        )

        return {
            "video_path": video_path,
            "output_video": output_path,
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "total_objects": total_standard_objects,
            "total_detections": len(all_detections),
            "objects": all_detections,
            "summary": summary,
            "status": derive_status(
                summary,
                accident_confidence,
                confidence_threshold=ACCIDENT_VIDEO_CONFIDENCE,
            ),
            # Include a simple vehicle-count-derived extra factor for video severity
            # (normalised to [0,1] and blended with detector confidence inside classify_severity).
            "severity": classify_severity(
                accident_confidence,
                derive_status(summary, accident_confidence, confidence_threshold=ACCIDENT_VIDEO_CONFIDENCE),
                extra={"vehicle_count": min(1.0, total_standard_objects / 10.0)}
            ),
            "screenshot_path": accident_screenshot_path,
            "average_confidence": round(avg_confidence, 4),
            "accident_confidence": round(accident_confidence, 4),
            "accident_incidents": accident_incidents,
            "inference_time": inference_time,
            "model": ACCIDENT_WEIGHTS_PATH,
            "standard_model": STANDARD_MODEL_PATH,
            "accident_model": ACCIDENT_WEIGHTS_PATH,
        }

    except Exception:
        cap.release()
        if output_video:
            output_video.release()
        raise


def _count_unique_accident_incidents(accident_frames: List[int], gap_threshold: int = 5) -> int:
    """Count unique accident incidents by grouping consecutive accident frames."""
    if not accident_frames:
        return 0

    sorted_frames = sorted(set(accident_frames))
    incidents = 1
    for prev_frame, curr_frame in zip(sorted_frames, sorted_frames[1:]):
        if curr_frame - prev_frame > gap_threshold:
            incidents += 1
    return incidents


def _extract_frame_detections(
    raw_results: List[Any],
    allowed_classes: set[str] | None = None,
    source_model: str | None = None,
) -> List[Dict[str, Any]]:
    """Extract detection data (bounding boxes + labels) from YOLO frame results."""
    return normalize_yolo_results(raw_results, allowed_classes=allowed_classes, source_model=source_model)


def _draw_detections_on_frame(frame, detections: List[Dict[str, Any]]) -> Any:
    """Draw bounding boxes and labels on a video frame.

    Accident detections from the custom ``best.pt`` model are drawn as red
    rectangles with a thicker border. Standard objects use their class color.
    Each box also gets a filled label with the class name and confidence.
    """
    for detection in detections:
        if detection.get("bbox") is None:
            continue

        bbox = detection["bbox"]
        x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
        class_name = detection["class_name"]
        confidence = detection["confidence"]

        color = get_detection_color(detection)
        thickness = 3 if class_name == "accident" else 2

        # Red/colored bounding box.
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        # Label with class name + confidence.
        label = f"{class_name} ({confidence:.2%})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_thickness = 1

        text_size = cv2.getTextSize(label, font, font_scale, text_thickness)[0]
        text_x = x1
        text_y = max(y1 - 5, text_size[1] + 5)

        cv2.rectangle(
            frame,
            (text_x, text_y - text_size[1] - 5),
            (text_x + text_size[0], text_y),
            color,
            -1,
        )
        cv2.putText(frame, label, (text_x, text_y), font, font_scale, (0, 0, 0), text_thickness)

    return frame