"""Configuration values for the Flask application."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    """Base Flask configuration."""

    SECRET_KEY = os.environ.get("SECRET_KEY", "smart-accident-detection-secret")
    DATABASE_PATH = os.path.join(BASE_DIR, "database.db")
    YOLO_MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "models", "accident_detector", "weights", "best.pt")
    YOLO_INFERENCE_CONFIDENCE = float(os.environ.get("YOLO_INFERENCE_CONFIDENCE", "0.45"))
    ACCIDENT_CONFIDENCE_THRESHOLD = float(os.environ.get("ACCIDENT_CONFIDENCE_THRESHOLD", "0.50"))
    # Default camera source: 0 = primary webcam (DroidCam uses index 0 seamlessly),
    # or an IP stream URL like "http://<ip>:<port>/video"
    CAMERA_SOURCE = os.environ.get("CAMERA_SOURCE", 0)
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    JSON_SORT_KEYS = False