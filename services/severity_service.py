"""
Severity scoring engine.

Severity is computed from:
  - Damage type weight (crack=1.5, dent=1.0, display_lines=2.0, etc.)
  - Damage area ratio (damage pixels / appliance pixels)
  - Number of defects (multiple damages increase severity)

Condition score is derived from severity (not arbitrary):
  100 - (severity_impact * damage_type_weight)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from configs.config import LOCATION_WEIGHTS

DAMAGE_TYPE_WEIGHTS = {
    "crack": 1.5,
    "dent": 1.0,
    "display_lines": 2.0,
    "rust": 1.2,
    "scratch": 0.6,
    "body_damage": 1.1,
    "screen_crack": 1.8,
    "dead_pixels": 0.8,
    "panel_damage": 2.5,
    "unknown": 1.0,
}

SEVERITY_BANDS = [
    ("Minor", 0, 10),
    ("Moderate", 10, 30),
    ("Major", 30, 60),
    ("Severe", 60, 101),
]


def compute_damage_area(bbox: List[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, (x2 - x1) * (y2 - y1))


def compute_appliance_area(bbox: List[float]) -> float:
    return compute_damage_area(bbox)


def compute_damage_percentage(damage_area: float, appliance_area: float) -> float:
    if appliance_area <= 0:
        return 0.0
    return round(min(damage_area / appliance_area * 100.0, 100.0), 1)


def classify_severity(percentage: float) -> str:
    for label, lo, hi in SEVERITY_BANDS:
        if lo <= percentage < hi:
            return label
    return "Severe"


def assess_damage(
    damage_bbox: List[float],
    appliance_bbox: List[float],
    damage_type: str,
    damage_count: int = 1,
    location: Optional[str] = None,
) -> Dict[str, Any]:
    damage_area = compute_damage_area(damage_bbox)
    appliance_area = compute_appliance_area(appliance_bbox)
    percentage = compute_damage_percentage(damage_area, appliance_area)

    type_weight = DAMAGE_TYPE_WEIGHTS.get(damage_type.lower(), 1.0)
    location_weight = LOCATION_WEIGHTS.get(location, 1.0) if location else 1.0

    defect_multiplier = 1.0 + (max(damage_count - 1, 0) * 0.15)

    severity_pct = min(percentage * type_weight * location_weight * defect_multiplier, 100.0)
    severity = classify_severity(severity_pct)

    return {
        "damage_type": damage_type,
        "damage_type_weight": type_weight,
        "location_weight": location_weight,
        "damage_area_px": round(damage_area, 1),
        "appliance_area_px": round(appliance_area, 1),
        "damage_percentage": percentage,
        "weighted_severity_pct": round(severity_pct, 1),
        "severity": severity,
        "defect_count": damage_count,
        "defect_multiplier": round(defect_multiplier, 2),
    }


def assess_all_damages(
    damage_detections: List[Dict[str, Any]],
    appliance_bbox: Optional[List[float]] = None,
    image_shape: Optional[Tuple[int, int]] = None,
) -> List[Dict[str, Any]]:
    if not appliance_bbox and image_shape:
        h, w = image_shape[:2]
        appliance_bbox = [0.0, 0.0, float(w), float(h)]
    if not appliance_bbox:
        appliance_bbox = [0.0, 0.0, 640.0, 640.0]

    type_counts: Dict[str, int] = {}
    for det in damage_detections:
        t = det.get("class_name", "unknown")
        type_counts[t] = type_counts.get(t, 0) + 1

    results = []
    for det in damage_detections:
        dt = det.get("class_name", "unknown")
        count = type_counts.get(dt, 1)
        result = assess_damage(
            damage_bbox=det.get("bbox", [0, 0, 0, 0]),
            appliance_bbox=appliance_bbox,
            damage_type=dt,
            damage_count=count,
            location=det.get("location"),
        )
        results.append(result)
    return results


def get_overall_severity(damage_assessments: List[Dict[str, Any]]) -> str:
    if not damage_assessments:
        return "None"
    levels = {"None": -1, "Minor": 0, "Moderate": 1, "Major": 2, "Severe": 3}
    worst = max(damage_assessments, key=lambda d: levels.get(d.get("severity", "Minor"), 0))
    return worst.get("severity", "Minor")


def compute_condition_score(damage_assessments: List[Dict[str, Any]]) -> int:
    if not damage_assessments:
        return 100
    levels = {"None": 0, "Minor": 0.15, "Moderate": 0.30, "Major": 0.50, "Severe": 0.80}
    total_impact = 0.0
    for d in damage_assessments:
        sev = d.get("severity", "Minor")
        weight = d.get("damage_type_weight", 1.0)
        pct = d.get("damage_percentage", 0)
        sev_penalty = levels.get(sev, 0.15)
        impact = sev_penalty * weight * min(pct / 10.0, 1.0)
        total_impact += impact
    score = max(0, 100 - int(total_impact * 100))
    return min(score, 100)


def compute_grade(condition_score: int) -> str:
    if condition_score >= 90:
        return "A"
    elif condition_score >= 75:
        return "B"
    elif condition_score >= 50:
        return "C"
    else:
        return "D"
