"""Dual-model detection pipeline: standard YOLO (object counting) + custom accident model."""

import os
from typing import Any, Dict, List, Tuple

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from detector.inference_robust import (
    ROBUST_CONF,
    ROBUST_ENHANCE,
    ROBUST_IMGSZ,
    ROBUST_IOU,
    ROBUST_MULTI_SCALE_SIZES,
    ROBUST_STANDARD_CONF,
    ROBUST_TTA,
    run_robust_inference,
)
from detector.severity import classify_severity

# ... (existing imports)
# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
# Standard COCO model for counting everyday objects (cars, trucks, persons, etc.)
STANDARD_MODEL_PATH = os.path.join(PROJECT_ROOT, "yolov8n.pt")
# Custom trained accident-detection model (24.4 MB)
ACCIDENT_MODEL_PATH = os.path.join(PROJECT_ROOT, "runs", "detect", "models", "accident_detector", "weights", "best.pt")

CUSTOM_MODEL_NAMES = ("best (1).pt", "best.pt")

# ---------------------------------------------------------------------------
# Inference parameters
# ---------------------------------------------------------------------------
# Explicit image size keeps distant CCTV crashes from being downscaled
# inconsistently across image/video entrypoints.
INFERENCE_SIZE = 640
INFERENCE_IOU = 0.45

# Separate model confidence thresholds.
# Standard YOLO stays at 0.45 so normal cars/trucks/persons are detected cleanly.
STANDARD_INFERENCE_CONFIDENCE = 0.45
# Custom accident model threshold set to 0.75 for model inference behavior.
ACCIDENT_INFERENCE_CONFIDENCE = 0.75
# A lower UI/acceptance threshold that errs on the side of surfacing
# potentially valid mid-confidence detections (used for final status only).
# Raise the default acceptance threshold to reduce false positives (configurable)
# Production default set to 0.65; override via env ACCEPTABLE_ACCIDENT_CONFIDENCE.
ACCEPTABLE_ACCIDENT_CONFIDENCE = float(os.environ.get("ACCEPTABLE_ACCIDENT_CONFIDENCE", "0.65"))
# Minimum IoU with another vehicle required to consider a spatial overlap as
# "significant". This enforces that an accident box must overlap at least one
# other vehicle/person box with measurable intersection.
ACCIDENT_REQUIRED_IOU_WITH_VEHICLE = float(os.environ.get("ACCIDENT_REQUIRED_IOU_WITH_VEHICLE", "0.15"))
# Backwards-compatible confidence threshold (can still be overridden via env).
MIN_ACCIDENT_CONFIDENCE_THRESHOLD = ACCIDENT_INFERENCE_CONFIDENCE
ACCIDENT_CONFIDENCE_THRESHOLD = max(
    MIN_ACCIDENT_CONFIDENCE_THRESHOLD,
    float(os.environ.get("ACCIDENT_CONFIDENCE_THRESHOLD", str(MIN_ACCIDENT_CONFIDENCE_THRESHOLD))),
)
# An accident prediction is accepted WITHOUT overlap validation only when its
# confidence is strictly above this value (0.75). This remains the model's
# own strict acceptance guard used by some internal heuristics.
ACCIDENT_STRICT_CONFIDENCE = 0.75
# IoU threshold used to decide whether an accident box overlaps a vehicle box.
ACCIDENT_OVERLAP_IOU_THRESHOLD = 0.10
# Intersection-over-vehicle-area threshold for large accident boxes that contain
# small/distant vehicles.
ACCIDENT_VEHICLE_COVERAGE_THRESHOLD = 0.35
# Minimum number of overlapping vehicle boxes required to validate a
# lower-confidence accident prediction (collision evidence).
MIN_OVERLAPPING_VEHICLES = 2
VEHICLE_PAIR_COLLISION_IOU_THRESHOLD = 0.02
VEHICLE_PAIR_TOUCH_MARGIN_RATIO = 0.12

# ---------------------------------------------------------------------------
# STRICT multi-vehicle collision policy (false-positive guard, env-tunable).
# An "accident" is CONFIRMED only when ALL of the following hold:
#   R1. Its own confidence >= ACCIDENT_MIN_CONFIDENCE (hard floor, no exceptions).
#   R2. It overlaps >= MIN_OVERLAPPING_VEHICLES DISTINCT standard vehicles.
#   R3. Those evidence vehicles collide WITH EACH OTHER (pair IoU or touch).
#   R4. Duplicate boxes of ONE physical vehicle are merged first, so a lone car
#       double-detected as car+truck can never impersonate a colliding pair.
# A single, isolated moving vehicle is therefore NEVER classified as accident.
# ---------------------------------------------------------------------------
ACCIDENT_MIN_CONFIDENCE = float(os.environ.get("ACCIDENT_MIN_CONFIDENCE", "0.65"))
EVIDENCE_DEDUPE_IOU_THRESHOLD = float(os.environ.get("EVIDENCE_DEDUPE_IOU_THRESHOLD", "0.60"))

# Backward-compatible aliases used by video/webcam code
INFERENCE_CONFIDENCE = STANDARD_INFERENCE_CONFIDENCE
IMAGE_INFERENCE_CONFIDENCE = ACCIDENT_INFERENCE_CONFIDENCE
IMAGE_INFERENCE_IOU = INFERENCE_IOU

# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------
STANDARD_OBJECT_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "person"}
# Include 'person' here so collisions involving pedestrians count as accident evidence.
VEHICLE_COLLISION_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle", "vehicle", "person"}
ACCIDENT_CLASSES = {"accident"}
SUPPORTED_CLASSES = ACCIDENT_CLASSES | STANDARD_OBJECT_CLASSES | {"vehicle"}
GENERAL_VEHICLE_CLASSES = {"car", "vehicle"}
MIN_GENERAL_VEHICLE_CONFIDENCE = STANDARD_INFERENCE_CONFIDENCE
ACCIDENT_DEDUPE_IOU_THRESHOLD = INFERENCE_IOU

ACCIDENT_BOX_COLOR = (0, 0, 255)  # BGR red
STANDARD_BOX_COLORS = {
    "car": (0, 255, 0),
    "vehicle": (0, 255, 255),
    "truck": (255, 128, 0),
    "bus": (255, 0, 255),
    "motorcycle": (255, 255, 0),
    "bicycle": (180, 255, 0),
    "person": (255, 180, 0),
}
DEFAULT_STANDARD_BOX_COLOR = (0, 255, 0)

SUMMARY_CLASS_ORDER = (
    ("car", "Cars"),
    ("vehicle", "Vehicles"),
    ("truck", "Trucks"),
    ("bus", "Buses"),
    ("motorcycle", "Motorcycles"),
    ("person", "Persons"),
    ("bicycle", "Bicycles"),
)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
_MODEL_CACHE: Dict[str, Any] = {}


def _load_yolo_model(model_path: str) -> Any:
    """Load a YOLO model with torch compatibility handling and caching."""
    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]

    original_torch_load = torch.load

    def load_with_compatibility(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    safe_globals = getattr(torch.serialization, "safe_globals", None)
    try:
        torch.load = load_with_compatibility
        if safe_globals is not None:
            with safe_globals([DetectionModel]):
                _MODEL_CACHE[model_path] = YOLO(model_path)
        else:
            add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
            if add_safe_globals is not None:
                add_safe_globals([DetectionModel])
            _MODEL_CACHE[model_path] = YOLO(model_path)
    finally:
        torch.load = original_torch_load

    return _MODEL_CACHE[model_path]


def resolve_model_path(model_path: str | None = None) -> str:
    """Resolve provided or candidate model paths for custom accident detection.

    The primary trained accident model at ``runs/detect/models/accident_detector/weights/best.pt``
    is always preferred over generic base models (yolov8n.pt / best (1).pt).
    """
    if model_path and os.path.basename(model_path) in CUSTOM_MODEL_NAMES:
        candidate_paths = [
            ACCIDENT_MODEL_PATH,
            os.path.join(PROJECT_ROOT, "runs", "detect", "models", "accident_detector", "weights", "best (1).pt"),
            os.path.join(PROJECT_ROOT, "runs", "detect", "train", "weights", "best.pt"),
            os.path.join(PROJECT_ROOT, "runs", "detect", "train", "weights", "best (1).pt"),
            os.path.join(PROJECT_ROOT, "models", "best.pt"),
            os.path.join(PROJECT_ROOT, "models", "best (1).pt"),
            os.path.join(PROJECT_ROOT, "best.pt"),
            os.path.join(PROJECT_ROOT, "best (1).pt"),
        ]
        for candidate in candidate_paths:
            if os.path.exists(candidate):
                return candidate

    if model_path and os.path.exists(model_path):
        return model_path

    candidate_paths = [
        ACCIDENT_MODEL_PATH,
        os.path.join(PROJECT_ROOT, "runs", "detect", "models", "accident_detector", "weights", "best (1).pt"),
        os.path.join(PROJECT_ROOT, "runs", "detect", "train", "weights", "best.pt"),
        os.path.join(PROJECT_ROOT, "runs", "detect", "train", "weights", "best (1).pt"),
        os.path.join(PROJECT_ROOT, "models", "best.pt"),
        os.path.join(PROJECT_ROOT, "models", "best (1).pt"),
        os.path.join(PROJECT_ROOT, "best.pt"),
        os.path.join(PROJECT_ROOT, "best (1).pt"),
        "best.pt",
        "best (1).pt",
    ]

    for candidate in candidate_paths:
        if os.path.exists(candidate):
            return candidate

    env_model = os.environ.get("YOLO_MODEL")
    if env_model and os.path.exists(env_model):
        return env_model

    print(
        "[WARNING] Custom trained weights 'best (1).pt' not found in workspace root, models/, or runs/ folders! "
        "Advisory: Please place 'best (1).pt' in the project root or models directory for custom accident detection. "
        "Falling back to default 'yolov8n.pt'."
    )
    return "yolov8n.pt"


DEFAULT_MODEL = resolve_model_path()


def load_model(model_path: str | None = None):
    """Load a YOLO model using custom best.pt weights resolution."""
    selected_model = resolve_model_path(model_path)
    return _load_yolo_model(selected_model)


def load_accident_model(model_path: str | None = None):
    """Load the custom accident-detection weights explicitly."""
    return load_model(model_path or resolve_model_path())


def load_standard_model() -> Any:
    """Load the standard COCO YOLO model (yolov8n.pt) for object counting."""
    return _load_yolo_model(STANDARD_MODEL_PATH)


def load_dual_models(accident_model_path: str | None = None) -> Tuple[Any, Any]:
    """Load BOTH models: standard YOLO (object counting) + custom accident model.

    Returns:
        Tuple[standard_model, accident_model]
    """
    standard_model = load_standard_model()
    accident_model = load_accident_model(accident_model_path)
    return standard_model, accident_model


# ---------------------------------------------------------------------------
# Detection filtering & normalization
# ---------------------------------------------------------------------------
def should_keep_detection(class_name: str, confidence: float, allowed_classes: set[str] | None = None) -> bool:
    """Apply strict class and debris filters while leaving accident detections prioritized."""
    class_key = str(class_name or "unknown").lower()
    accepted_classes = allowed_classes or SUPPORTED_CLASSES
    if class_key not in accepted_classes:
        return False
    if class_key in GENERAL_VEHICLE_CLASSES and float(confidence) < MIN_GENERAL_VEHICLE_CONFIDENCE:
        return False
    return True


def normalize_yolo_results(
    raw_results: List[Any] | None,
    allowed_classes: set[str] | None = None,
    source_model: str | None = None,
) -> List[Dict[str, Any]]:
    """Convert YOLO results into a simple JSON-friendly structure with bounding boxes."""
    normalized: List[Dict[str, Any]] = []
    if raw_results is None:
        return normalized

    for result in raw_results:
        names = getattr(result, "names", {})
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue

        confidences = getattr(boxes, "conf", None)
        class_ids = getattr(boxes, "cls", None)
        xyxy_coords = getattr(boxes, "xyxy", None)
        if confidences is None or class_ids is None:
            continue

        confidence_values = confidences.tolist() if hasattr(confidences, "tolist") else list(confidences)
        class_id_values = class_ids.tolist() if hasattr(class_ids, "tolist") else list(class_ids)
        coords = xyxy_coords.tolist() if hasattr(xyxy_coords, "tolist") else (xyxy_coords or [])

        for idx, (confidence, class_id) in enumerate(zip(confidence_values, class_id_values)):
            confidence = float(confidence)
            class_name = names.get(int(class_id), "unknown")
            class_name = str(class_name).lower()
            if not should_keep_detection(class_name, confidence, allowed_classes=allowed_classes):
                continue

            bbox = None
            if idx < len(coords):
                x1, y1, x2, y2 = coords[idx]
                bbox = {"x1": round(float(x1), 2), "y1": round(float(y1), 2), "x2": round(float(x2), 2), "y2": round(float(y2), 2)}

            normalized.append(
                {
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "class_id": int(class_id),
                    "bbox": bbox,
                    "source_model": source_model,
                }
            )
    return normalized


def get_detection_color(detection: Dict[str, Any]) -> tuple[int, int, int]:
    """Return a high-visibility BGR color for drawing a detection."""
    class_name = str(detection.get("class_name", "")).lower()
    if class_name == "accident" or str(detection.get("source_model", "")).lower().endswith("best.pt"):
        return ACCIDENT_BOX_COLOR
    return STANDARD_BOX_COLORS.get(class_name, DEFAULT_STANDARD_BOX_COLOR)


def _bbox_area(box: Dict[str, float] | None) -> float:
    """Compute area for a normalized detection bbox dictionary."""
    if not box:
        return 0.0

    x1, y1, x2, y2 = float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_intersection_area(box_a: Dict[str, float] | None, box_b: Dict[str, float] | None) -> float:
    """Compute raw intersection area for normalized detection bbox dictionaries."""
    if not box_a or not box_b:
        return 0.0

    ax1, ay1, ax2, ay2 = float(box_a["x1"]), float(box_a["y1"]), float(box_a["x2"]), float(box_a["y2"])
    bx1, by1, bx2, by2 = float(box_b["x1"]), float(box_b["y1"]), float(box_b["x2"]), float(box_b["y2"])

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    return inter_w * inter_h


def _bbox_iou(box_a: Dict[str, float] | None, box_b: Dict[str, float] | None) -> float:
    """Compute IoU for normalized detection bbox dictionaries."""
    intersection = _bbox_intersection_area(box_a, box_b)
    if intersection <= 0:
        return 0.0

    area_a = _bbox_area(box_a)
    area_b = _bbox_area(box_b)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _bbox_center_inside(outer_box: Dict[str, float] | None, inner_box: Dict[str, float] | None) -> bool:
    """Return whether the center of inner_box falls inside outer_box."""
    if not outer_box or not inner_box:
        return False

    center_x = (float(inner_box["x1"]) + float(inner_box["x2"])) / 2.0
    center_y = (float(inner_box["y1"]) + float(inner_box["y2"])) / 2.0
    return (
        float(outer_box["x1"]) <= center_x <= float(outer_box["x2"])
        and float(outer_box["y1"]) <= center_y <= float(outer_box["y2"])
    )


def _bbox_intersection_over_area(
    outer_box: Dict[str, float] | None,
    inner_box: Dict[str, float] | None,
) -> float:
    """Return how much of inner_box is covered by outer_box."""
    inner_area = _bbox_area(inner_box)
    if inner_area <= 0:
        return 0.0
    return _bbox_intersection_area(outer_box, inner_box) / inner_area


def _expanded_bbox_intersects(
    box_a: Dict[str, float] | None,
    box_b: Dict[str, float] | None,
    margin_ratio: float = VEHICLE_PAIR_TOUCH_MARGIN_RATIO,
) -> bool:
    """Return whether two boxes overlap after a small expansion for near-contact."""
    if not box_a or not box_b:
        return False

    ax1, ay1, ax2, ay2 = float(box_a["x1"]), float(box_a["y1"]), float(box_a["x2"]), float(box_a["y2"])
    bx1, by1, bx2, by2 = float(box_b["x1"]), float(box_b["y1"]), float(box_b["x2"]), float(box_b["y2"])
    aw, ah = max(0.0, ax2 - ax1), max(0.0, ay2 - ay1)
    bw, bh = max(0.0, bx2 - bx1), max(0.0, by2 - by1)
    margin_x = max(aw, bw) * margin_ratio
    margin_y = max(ah, bh) * margin_ratio

    return not (
        ax2 + margin_x < bx1
        or bx2 + margin_x < ax1
        or ay2 + margin_y < by1
        or by2 + margin_y < ay1
    )


def _has_colliding_vehicle_pair(vehicle_boxes: List[Dict[str, float]]) -> bool:
    """Return whether any vehicle pair overlaps or is close enough to be colliding."""
    for index, first_box in enumerate(vehicle_boxes):
        for second_box in vehicle_boxes[index + 1 :]:
            if _bbox_iou(first_box, second_box) >= VEHICLE_PAIR_COLLISION_IOU_THRESHOLD:
                return True
            if _expanded_bbox_intersects(first_box, second_box):
                return True
    return False


def dedupe_standard_against_accidents(
    standard_objects: List[Dict[str, Any]],
    accident_objects: List[Dict[str, Any]],
    iou_threshold: float = ACCIDENT_DEDUPE_IOU_THRESHOLD,
) -> List[Dict[str, Any]]:
    """Drop standard boxes that overlap accident boxes so accident boxes win."""
    accident_boxes = [item.get("bbox") for item in accident_objects if item.get("bbox")]
    if not accident_boxes:
        return standard_objects

    deduped: List[Dict[str, Any]] = []
    for item in standard_objects:
        bbox = item.get("bbox")
        if bbox and any(_bbox_iou(bbox, accident_box) >= iou_threshold for accident_box in accident_boxes):
            continue
        deduped.append(item)
    return deduped


def combine_detections_with_accident_priority(
    standard_objects: List[Dict[str, Any]],
    accident_objects: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Combine detections with custom accident boxes preserved and drawn last."""
    return dedupe_standard_against_accidents(standard_objects, accident_objects) + accident_objects


def _dedupe_evidence_boxes(
    boxes: List[Dict[str, float]],
    iou_threshold: float = EVIDENCE_DEDUPE_IOU_THRESHOLD,
) -> List[Dict[str, float]]:
    """Merge near-identical standard boxes so ONE physical vehicle that was
    double-detected (e.g. as both ``car`` and ``truck``) can never impersonate
    two separate colliding vehicles."""
    unique: List[Dict[str, float]] = []
    for box in boxes:
        if all(_bbox_iou(box, kept) < iou_threshold for kept in unique):
            unique.append(box)
    return unique


def _count_standard_vehicles(standard_objects: List[Dict[str, Any]]) -> int:
    """Count standard detections belonging to vehicle/collision classes."""
    return sum(
        1
        for item in standard_objects
        if str(item.get("class_name", "")).lower() in VEHICLE_COLLISION_CLASSES
    )


def validate_accident_detections(
    accident_objects: List[Dict[str, Any]],
    standard_objects: List[Dict[str, Any]],
    strict_confidence: float = ACCIDENT_STRICT_CONFIDENCE,
    overlap_iou_threshold: float = ACCIDENT_OVERLAP_IOU_THRESHOLD,
    min_overlapping_vehicles: int = MIN_OVERLAPPING_VEHICLES,
    min_confidence: float = ACCIDENT_MIN_CONFIDENCE,
) -> List[Dict[str, Any]]:
    """STRICT multi-vehicle collision validation (post-processing only).

    An ``accident`` prediction survives ONLY when every rule below holds:

      R1. ``confidence >= min_confidence`` (default 0.65) - hard floor with no
          exceptions, regardless of spatial evidence.
      R2. The crash box overlaps at least ``min_overlapping_vehicles`` (2)
          DISTINCT standard vehicles/persons via IoU >=
          ``overlap_iou_threshold``, or the vehicle lies substantially inside
          the crash region (coverage >= ACCIDENT_VEHICLE_COVERAGE_THRESHOLD or
          its centre sits inside the crash box).
      R3. Those evidence vehicles physically collide WITH EACH OTHER: pair
          IoU >= VEHICLE_PAIR_COLLISION_IOU_THRESHOLD, or their expanded boxes
          make contact within VEHICLE_PAIR_TOUCH_MARGIN_RATIO.
      R4. Standard boxes overlapping each other above
          EVIDENCE_DEDUPE_IOU_THRESHOLD are merged first, so duplicate
          detections of one physical vehicle cannot fake a pair.

    Consequences: a single, isolated, normally-driving vehicle is rejected at
    ANY confidence level, and roadside objects/shadows never accumulate the
    required two distinct colliding vehicles. Pure geometry math - no impact
    on model inference speed.
    """
    if not accident_objects:
        return []

    raw_vehicle_boxes = [
        item.get("bbox")
        for item in standard_objects
        if item.get("bbox") and str(item.get("class_name", "")).lower() in VEHICLE_COLLISION_CLASSES
    ]
    # R4: one physical vehicle must count once, however many classes fired on it.
    vehicle_boxes = _dedupe_evidence_boxes(raw_vehicle_boxes)

    validated: List[Dict[str, Any]] = []
    for accident in accident_objects:
        confidence = float(accident.get("confidence", 0.0))

        # R1: hard confidence floor - reject weak/ambiguous triggers first.
        if confidence < min_confidence:
            continue

        accident_box = accident.get("bbox")

        # No bbox available -> geometry cannot be verified directly. Accept only
        # when the SCENE itself shows a distinct colliding vehicle pair AND the
        # detector is very confident. When standard detections carry no
        # coordinates at all (degenerate/mocked sources), fall back to counting
        # vehicle-class objects so dual-model pipelines keep functioning.
        if not accident_box:
            scene_vehicle_count = len(vehicle_boxes) or _count_standard_vehicles(standard_objects)
            if (
                confidence >= max(min_confidence, strict_confidence)
                and scene_vehicle_count >= min_overlapping_vehicles
                and (vehicle_boxes and _has_colliding_vehicle_pair(vehicle_boxes) or not vehicle_boxes)
            ):
                validated.append(
                    {**accident, "validation_reason": "multi_vehicle_scene_high_confidence"}
                )
            continue

        # R2: collect DISTINCT vehicles engaged with the crash region.
        evidence_boxes = [
            vehicle_box
            for vehicle_box in vehicle_boxes
            if (
                _bbox_iou(accident_box, vehicle_box) >= overlap_iou_threshold
                or _bbox_intersection_over_area(accident_box, vehicle_box) >= ACCIDENT_VEHICLE_COVERAGE_THRESHOLD
                or _bbox_center_inside(accident_box, vehicle_box)
            )
        ]

        # A single (or zero) engaged vehicle is NEVER an accident - regardless
        # of how confident the classifier is about the crash appearance.
        if len(evidence_boxes) < min_overlapping_vehicles:
            continue

        # R3: the engaged vehicles must collide/touch EACH OTHER.
        if not _has_colliding_vehicle_pair(evidence_boxes):
            continue

        validated.append({**accident, "validation_reason": "collision_pair_evidence"})

    # Everything else was a false trigger: single lane-following cars,
    # roadside objects, shadows, poles, billboards, etc.
    return validated


def _normalize_results(raw_results: List[Any] | None) -> List[Dict[str, Any]]:
    """Backward-compatible alias for callers/tests using the old private helper."""
    return normalize_yolo_results(raw_results)


# ---------------------------------------------------------------------------
# Summary & status helpers
# ---------------------------------------------------------------------------
def generate_detection_summary(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate detections into a compact class-count summary."""
    counts: Dict[str, int] = {"accident": 0}
    counts.update({class_name: 0 for class_name, _ in SUMMARY_CLASS_ORDER})
    accident_count = 0
    for item in detections:
        class_name = str(item.get("class_name", "unknown")).lower()
        if class_name == "accident":
            accident_count += 1
            counts["accident"] += 1
            continue
        if class_name in counts:
            counts[class_name] += 1

    display_items = [{"label": label, "count": counts[class_name]} for class_name, label in SUMMARY_CLASS_ORDER]
    return {
        "counts": counts,
        "display_items": display_items,
        "total_objects": sum(counts.values()) - counts.get("accident", 0),
        "accident_count": accident_count,
    }


def _resolve_accident_threshold(confidence_threshold: float | None = None) -> float:
    """Return a safe accident-detection confidence threshold.

    Default: ACCEPTABLE_ACCIDENT_CONFIDENCE (e.g. 0.55) so the final UI status
    reflects practical mid-confidence model outputs while inference thresholds
    used by models remain unchanged.
    """
    if confidence_threshold is None:
        return ACCEPTABLE_ACCIDENT_CONFIDENCE

    try:
        threshold = float(confidence_threshold)
    except (TypeError, ValueError):
        return ACCEPTABLE_ACCIDENT_CONFIDENCE

    # Clamp into [0.0, 1.0]
    if threshold < 0.0:
        return 0.0
    if threshold > 1.0:
        return 1.0
    return threshold


def get_accident_confidence(detections: List[Dict[str, Any]]) -> float:
    """Return the strongest accident-class confidence in a detection set."""
    return max(
        (float(item.get("confidence", 0.0)) for item in detections if str(item.get("class_name", "")).lower() == "accident"),
        default=0.0,
    )


def determine_accident_status(
    detections: List[Dict[str, Any]],
    confidence_threshold: float | None = None,
) -> Dict[str, Any]:
    """Classify an image as Accident only when the accident class crosses the threshold.

    The threshold matches the custom accident model's inference confidence by
    default so distant CCTV crashes and destroyed foreground shells are surfaced.
    """
    threshold = _resolve_accident_threshold(confidence_threshold)
    accident_confidence = get_accident_confidence(detections)
    status = "Accident" if accident_confidence >= threshold else "Non-Accident"
    return {
        "status": status,
        "confidence": round(accident_confidence, 4),
        "threshold": threshold,
        "is_accident": status == "Accident",
    }


def build_detection_summary_text(summary: Dict[str, Any], accident_result: Dict[str, Any]) -> str:
    """Build a compact human-readable summary for the UI and history records."""
    lines = ["Detection Summary", "-----------------"]
    for item in summary.get("display_items", []):
        lines.append(f"{item['label']}: {item['count']}")

    lines.extend(
        [
            "",
            "Accident Status:",
            f"✅ {accident_result.get('status', 'Non-Accident')}",
            "",
            "Accident Confidence:",
            f"{float(accident_result.get('confidence', 0.0)) * 100:.2f}%",
            "",
            "Total Objects Detected:",
            str(summary.get("total_objects", 0)),
        ]
    )
    return "\n".join(lines)


def summarize_detections(detections: List[Dict[str, Any]]) -> Dict[str, int]:
    """Compatibility wrapper that returns the raw count map used by video/webcam code."""
    return generate_detection_summary(detections)["counts"]


def build_ui_count_fields(summary_counts: Dict[str, int]) -> Dict[str, int]:
    """Expose plural convenience fields used by the result UI and API consumers."""
    return {
        "cars": int(summary_counts.get("car", 0)),
        "vehicles": int(summary_counts.get("vehicle", 0)),
        "trucks": int(summary_counts.get("truck", 0)),
        "buses": int(summary_counts.get("bus", 0)),
        "motorcycles": int(summary_counts.get("motorcycle", 0)),
        "persons": int(summary_counts.get("person", 0)),
        "bicycles": int(summary_counts.get("bicycle", 0)),
    }


def derive_status(summary: Dict[str, int] | None, confidence: float = 0.0, confidence_threshold: float | None = None) -> str:
    """Map a summary to Accident or Non-Accident using the accident class confidence."""
    summary = summary or {}
    threshold = _resolve_accident_threshold(confidence_threshold)
    if summary.get("accident", 0) > 0 and confidence >= threshold:
        return "Accident"
    return "Non-Accident"


# ---------------------------------------------------------------------------
# Dual-model inference pipeline
# ---------------------------------------------------------------------------
def detect_objects(
    image_path: str,
    model_path: str | None = None,
    accident_confidence_threshold: float | None = None,
) -> Dict[str, Any]:
    """Run the dual-model inference pipeline on an image.

    Pipeline:
    1. Run standard YOLO (yolov8n.pt) to count all everyday objects
       (cars, trucks, buses, motorcycles, bicycles, persons) and populate
       the total_objects / summary counts for the Flask UI.
    2. Run the custom accident model (best.pt) to detect genuine accidents.
    3. Combine annotations and return a valid dictionary with non-zero
       object counts, accident_status, and accident_confidence.

    Args:
        image_path: Path to the input image.
        model_path: Optional custom accident model path (defaults to ACCIDENT_MODEL_PATH).
        accident_confidence_threshold: Optional accident confidence threshold
            (defaults to 0.50).

    Returns:
        Dict with objects, summary, accident_status, accident_confidence, etc.
    """
    # Load BOTH models
    standard_model = load_standard_model()
    accident_model = load_accident_model(model_path)

    # Step 1: Run standard YOLO to count everyday objects.
    # Robust inference: dynamically lowered confidence (0.20), NMS IoU 0.45,
    # TTA (augment=True), CLAHE contrast enhancement and multi-scale imgsz
    # (640 primary + extra sizes) so unseen datasets/cameras still get counted.
    standard_raw = run_robust_inference(
        standard_model,
        image_path,
        conf=ROBUST_STANDARD_CONF,
        iou=ROBUST_IOU,
        imgsz=ROBUST_IMGSZ,
        augment=ROBUST_TTA,
        enhance=ROBUST_ENHANCE,
        multi_scale_sizes=list(ROBUST_MULTI_SCALE_SIZES),
    )
    standard_results = list([] if standard_raw is None else standard_raw)
    standard_objects = normalize_yolo_results(
        standard_results,
        allowed_classes=STANDARD_OBJECT_CLASSES,
        source_model="yolov8n.pt",
    )

    # Step 2: Run custom accident model to detect genuine accidents.
    # Robust inference: dynamically lowered confidence (0.15), NMS IoU 0.45,
    # TTA (augment=True), CLAHE contrast enhancement and multi-scale imgsz to
    # catch crashes in unseen datasets/camera angles.
    accident_raw = run_robust_inference(
        accident_model,
        image_path,
        conf=ROBUST_CONF,
        iou=ROBUST_IOU,
        imgsz=ROBUST_IMGSZ,
        augment=ROBUST_TTA,
        enhance=ROBUST_ENHANCE,
        multi_scale_sizes=list(ROBUST_MULTI_SCALE_SIZES),
    )
    accident_results = list([] if accident_raw is None else accident_raw)
    raw_accident_objects = normalize_yolo_results(
        accident_results,
        allowed_classes=ACCIDENT_CLASSES,
        source_model=os.path.basename(resolve_model_path(model_path)),
    )

    # Step 3: Strictly validate accident predictions. A box is only kept as an
    # "accident" if its confidence is strictly above 0.75 OR it overlaps with
    # multiple standard vehicle/person boxes (collision evidence). A single
    # isolated car is dropped to prevent false positives.
    accident_objects = validate_accident_detections(raw_accident_objects, standard_objects)

    # Step 4: Combine detections with validated accident boxes taking precedence
    # over overlapping standard vehicle/person boxes.
    combined_objects = combine_detections_with_accident_priority(standard_objects, accident_objects)

    # Build summary from combined detections (non-zero counts for UI)
    summary = generate_detection_summary(combined_objects)

    # Determine accident status using the custom accident model threshold.
    accident_result = determine_accident_status(accident_objects, accident_confidence_threshold)
    status = accident_result["status"]
    accident_confidence = accident_result["confidence"]

    summary_text = build_detection_summary_text(summary, accident_result)

    summary_counts = summary["counts"]
    ui_count_fields = build_ui_count_fields(summary_counts)

    # Compute extra severity factors: prefer the highest-confidence accident box
    # and measure overlapping vehicle evidence + impact area ratio relative to
    # the image size. These extras are blended with the detector confidence in
    # classify_severity (the detector confidence remains dominant).
    extra_factors = {}
    try:
        if accident_objects:
            top_accident = max(accident_objects, key=lambda x: float(x.get("confidence", 0.0)))
            acc_bbox = top_accident.get("bbox")
            evidence_count = 0
            impact_area_ratio = 0.0
            if acc_bbox:
                # Recompute evidence boxes using the same spatial rules.
                evidence_boxes = [
                    item.get("bbox")
                    for item in standard_objects
                    if item.get("bbox") and (
                        _bbox_iou(acc_bbox, item.get("bbox")) >= ACCIDENT_OVERLAP_IOU_THRESHOLD
                        or _bbox_intersection_over_area(acc_bbox, item.get("bbox")) >= ACCIDENT_VEHICLE_COVERAGE_THRESHOLD
                        or _bbox_center_inside(acc_bbox, item.get("bbox"))
                    )
                ]
                evidence_count = len(evidence_boxes)

                # Image area from the source image; fall back to INFERENCE_SIZE**2.
                img = cv2.imread(image_path)
                if img is not None:
                    ih, iw = img.shape[:2]
                    image_area = max(1.0, float(iw * ih))
                else:
                    image_area = float(max(1, INFERENCE_SIZE * INFERENCE_SIZE))

                impact_area = _bbox_area(acc_bbox)
                impact_area_ratio = min(1.0, impact_area / image_area) if image_area > 0 else 0.0

            extra_factors = {
                "vehicle_overlap": min(1.0, evidence_count / 4.0),
                "impact_area": float(impact_area_ratio),
            }
    except Exception:
        # Any error during extra factor computation should not block the main
        # detection flow; fall back to empty extras.
        extra_factors = {}

    severity = classify_severity(accident_confidence, status, extra=extra_factors)

    return {
        "image_path": image_path,
        "objects": combined_objects,
        "standard_objects": standard_objects,
        "accident_objects": accident_objects,
        "summary": summary_counts,
        "summary_display": summary["display_items"],
        "summary_text": summary_text,
        "accident_confidence": accident_confidence,
        "accident_threshold": accident_result["threshold"],
        "accident_status": status,
        "status": status,
        "severity": severity,
        "model": resolve_model_path(model_path),
        "standard_model": STANDARD_MODEL_PATH,
        "accident_model": resolve_model_path(model_path),
        "total_objects": summary["total_objects"],
        "average_confidence": round(
            sum(item.get("confidence", 0.0) for item in combined_objects) / len(combined_objects), 4
        ) if combined_objects else 0.0,
        **ui_count_fields,
    }


