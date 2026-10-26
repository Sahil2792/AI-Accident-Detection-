"""Image inference module using the dual-model YOLO pipeline."""

import gc

from detector.detect import (
    STANDARD_MODEL_PATH,
    build_detection_summary_text,
    build_ui_count_fields,
    detect_objects,
    determine_accident_status,
    generate_detection_summary,
    resolve_model_path,
)
from detector.preprocess import downscale_image_if_needed
from detector.severity import classify_severity


def prepare_image_for_inference(image_path: str, max_dim: int | None = None):
    """Downscale an oversized upload in place so it fits in the inference
    tensor without blowing up memory on a 512 MB host.

    Returns the (possibly rewritten) path plus a change flag.  Coordinates stay
    aligned because both inference and annotation operate on the same file that
    is on disk after this call.
    """
    return downscale_image_if_needed(image_path, max_dim=max_dim)


def _empty_image_result(image_path: str, model_path: str | None = None):
    """Return the stable payload shape expected by Flask when inference is empty."""
    summary = generate_detection_summary([])
    accident_result = determine_accident_status([])
    summary_counts = summary["counts"]
    status = accident_result["status"]
    accident_model_path = resolve_model_path(model_path)
    return {
        "image_path": image_path,
        "objects": [],
        "standard_objects": [],
        "accident_objects": [],
        "summary": summary_counts,
        "summary_display": summary["display_items"],
        "summary_text": build_detection_summary_text(summary, accident_result),
        "accident_confidence": accident_result["confidence"],
        "accident_threshold": accident_result["threshold"],
        "accident_status": status,
        "status": status,
        "severity": classify_severity(accident_result["confidence"], status),
        "model": accident_model_path,
        "standard_model": STANDARD_MODEL_PATH,
        "accident_model": accident_model_path,
        "total_objects": 0,
        "average_confidence": 0.0,
        **build_ui_count_fields(summary_counts),
    }


def process_image(
    image_path: str,
    model_path: str | None = None,
    accident_confidence_threshold: float | None = None,
):
    """Process an image using the dual-model pipeline.

    The dual-model pipeline runs:
    1. Standard YOLO (yolov8n.pt) to count everyday objects
       (cars, trucks, buses, motorcycles, bicycles, persons).
    2. Custom accident model (best.pt) to detect genuine accidents.

    Both models use explicit 640px inference resolution and 0.45 NMS IoU via
    detector.detect.detect_objects. Accident detections are additionally
    validated against standard YOLO vehicle geometry so isolated normal cars
    are not promoted to accident labels.

    Returns a valid dictionary with non-zero object counts, accident_status,
    and accident_confidence for the Flask UI.
    """
    result = detect_objects(
        image_path,
        model_path,
        accident_confidence_threshold=accident_confidence_threshold,
    )
    if not isinstance(result, dict):
        result = _empty_image_result(image_path, model_path)
    gc.collect()
    return result
