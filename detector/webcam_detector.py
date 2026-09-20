"""Modular webcam detection entrypoints for the Flask app."""

from detector.webcam import WebcamStream, get_webcam_stream

__all__ = ["WebcamStream", "get_webcam_stream"]
