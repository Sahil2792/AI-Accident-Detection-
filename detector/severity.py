"""Severity classification for detected accidents.

Computes a Low / Medium / High severity level whenever the trained model detects
an accident, and returns it alongside accident status and confidence.

Design goals
------------
* Configurable at runtime with environment variables (no code edits needed).
* Easy to improve later: ``classify_severity`` accepts optional ``factors`` so
  future versions can incorporate per-frame signals (number of vehicles
  involved, overlap area, motion, scene complexity, impact size, ...) without
  changing the call sites.
* Non-accidents always resolve to ``low`` (no accident -> no elevated severity).
"""

import os
from typing import Any, Dict, Optional

# Severity thresholds map accident confidence / combined score to a level.
# All can be overridden via environment variables, e.g.:
#   SEVERITY_MEDIUM_THRESHOLD=0.45
#   SEVERITY_HIGH_THRESHOLD=0.70
SEVERITY_MEDIUM_THRESHOLD = float(os.environ.get("SEVERITY_MEDIUM_THRESHOLD", "0.50"))
SEVERITY_HIGH_THRESHOLD = float(os.environ.get("SEVERITY_HIGH_THRESHOLD", "0.75"))

# The status strings that represent a genuine accident.
ACCIDENT_STATUSES = {"Accident", "accident", "accident_detected"}

# Human-friendly severity labels returned by this module.
SEVERITY_LEVELS = ("low", "medium", "high")


def classify_severity(
    accident_confidence: float,
    status: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Classify an accident detection into ``low`` / ``medium`` / ``high``.

    Args:
        accident_confidence: The accident detector's confidence score in [0, 1].
        status: Optional accident status. If provided and not accident-like,
            the result is always ``low``.
        extra: Optional map of additional numeric factors in [0, 1] (e.g.
            ``{"vehicle_overlap": 0.8, "impact_area": 0.6}``). When supplied,
            the confidence is blended with the factor average using a simple
            weighted combination. This is the documented extension point.

    Returns:
        One of ``"low"``, ``"medium"``, or ``"high"``.
    """
    if status is not None and not _is_accident_status(status):
        return "low"

    confidence = float_confidence(accident_confidence)

    if extra:
        confidence = _blend_factors(confidence, extra)

    if confidence >= SEVERITY_HIGH_THRESHOLD:
        return "high"
    if confidence >= SEVERITY_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def _is_accident_status(status: str) -> bool:
    """Return True when the status string represents an accident."""
    return str(status or "").strip().lower() in {s.lower() for s in ACCIDENT_STATUSES}


def _blend_factors(confidence: float, extra: Dict[str, Any]) -> float:
    """Blend detector confidence with optional external factors.

    The detector confidence stays the dominant signal (60%) while the mean of
    any supplied factors contributes the remaining 40%. This is intentionally
    simple and easy to override/extend with a more sophisticated model later.
    """
    values = []
    for value in extra.values():
        try:
            values.append(float(max(0.0, min(1.0, float(value)))))
        except (TypeError, ValueError):
            continue

    if not values:
        return confidence

    factor_mean = sum(values) / len(values)
    return (0.6 * confidence) + (0.4 * factor_mean)


def float_confidence(value: float) -> float:
    """Coerce a raw value into a normalised [0, 1] confidence score."""
    try:
        return float(max(0.0, min(1.0, float(value or 0.0))))
    except (TypeError, ValueError):
        return 0.0
