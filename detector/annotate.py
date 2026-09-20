"""Image annotation helpers for drawing bounding boxes and displaying detection results."""

import os
from typing import Dict, Any, List

import cv2

from detector.detect import get_detection_color


def draw_bounding_boxes(
    image_path: str, detections: List[Dict[str, Any]], output_path: str
) -> None:
    """Draw bounding boxes on the image and save the annotated version."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    for detection in detections:
        if detection.get("bbox") is None:
            continue

        bbox = detection["bbox"]
        x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
        class_name = detection["class_name"]
        confidence = detection["confidence"]

        color = get_detection_color(detection)
        thickness = 4 if class_name == "accident" else 3
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        label = f"{class_name} ({confidence:.2%})"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8  # Increased for better readability
        text_thickness = 2

        text_size = cv2.getTextSize(label, font, font_scale, text_thickness)[0]
        text_x = x1
        text_y = max(y1 - 10, text_size[1] + 10)

        # Draw background rectangle for text
        cv2.rectangle(image, (text_x, text_y - text_size[1] - 10), (text_x + text_size[0] + 10, text_y + 5), color, -1)
        cv2.putText(image, label, (text_x + 5, text_y - 5), font, font_scale, (0, 0, 0), text_thickness)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save with high quality (for JPEG: quality=95, for PNG: compression=0)
    if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
        cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def save_detected_image(image_path: str, detections: List[Dict[str, Any]], output_path: str) -> str:
    """Draw detections on an image and return the saved output path."""
    draw_bounding_boxes(image_path, detections, output_path)
    return output_path
