"""
Repair cost estimation with severity-based multipliers.

Cost = base_cost × severity_multiplier × confidence_factor
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from configs.config import BRAND_COST_MULTIPLIERS
from services.severity_service import DAMAGE_TYPE_WEIGHTS

BASE_REPAIR_COSTS = {
    "crack": (150, 400),
    "dent": (100, 300),
    "display_lines": (500, 1500),
    "screen_crack": (300, 800),
    "rust": (200, 600),
    "scratch": (50, 150),
    "body_damage": (150, 500),
    "dead_pixels": (0, 0),
    "panel_damage": (400, 1500),
    "unknown": (100, 300),
}

SEVERITY_MULTIPLIERS = {
    "Minor": 0.5,
    "Moderate": 1.0,
    "Major": 1.8,
    "Severe": 3.0,
    "None": 0.0,
}


def estimate_damage_cost(
    damage_type: str,
    severity: str,
    confidence: float,
    brand: Optional[str] = None,
) -> Dict[str, Any]:
    base_min, base_max = BASE_REPAIR_COSTS.get(damage_type.lower(), (100, 300))
    sev_mult = SEVERITY_MULTIPLIERS.get(severity, 1.0)
    conf_mult = 0.5 + (confidence * 0.5)
    brand_mult = BRAND_COST_MULTIPLIERS.get(brand, 1.0) if brand else 1.0
    cost_min = int(base_min * sev_mult * conf_mult * brand_mult)
    cost_max = int(base_max * sev_mult * conf_mult * brand_mult)
    cost_min = max(cost_min, 0)
    cost_max = max(cost_max, cost_min)
    return {
        "damage_type": damage_type,
        "base_range": f"₹{base_min}-₹{base_max}",
        "severity_multiplier": sev_mult,
        "confidence_multiplier": round(conf_mult, 2),
        "brand_multiplier": brand_mult,
        "brand": brand or "unknown",
        "cost_min": cost_min,
        "cost_max": cost_max,
        "cost_display": f"₹{cost_min} - ₹{cost_max}",
    }


def estimate_total_repair_cost(
    damage_assessments: List[Dict[str, Any]],
    damage_detections: List[Dict[str, Any]],
    brand: Optional[str] = None,
) -> Dict[str, Any]:
    if not damage_assessments or not damage_detections:
        return {
            "total_min": 0,
            "total_max": 0,
            "total_display": "₹0 - ₹0",
            "breakdown": [],
            "recommendation": "No repair needed.",
        }

    breakdown: List[Dict] = []
    total_min = 0
    total_max = 0

    det_map: Dict[str, List[Dict]] = {}
    for det in damage_detections:
        t = det.get("class_name", "unknown")
        det_map.setdefault(t, []).append(det)

    for d in damage_assessments:
        dt = d.get("damage_type", "unknown")
        sev = d.get("severity", "Minor")
        dets = det_map.get(dt, [{"confidence": 0.5}])
        avg_conf = sum(x.get("confidence", 0.5) for x in dets) / max(len(dets), 1)
        cost = estimate_damage_cost(dt, sev, avg_conf, brand=brand)
        total_min += cost["cost_min"]
        total_max += cost["cost_max"]
        breakdown.append(cost)

    if total_min > 0 and total_max > 20000:
        rec = "High cost estimate — professional assessment recommended."
    elif total_min == 0:
        rec = "No repair needed."
    else:
        rec = "Repair recommended."

    return {
        "total_min": total_min,
        "total_max": total_max,
        "total_display": f"₹{total_min} - ₹{total_max}",
        "breakdown": breakdown,
        "recommendation": rec,
    }


def format_cost_inr(amount: int) -> str:
    s = str(amount)
    if len(s) <= 3:
        return "₹" + s
    last3 = s[-3:]
    rest = s[:-3]
    groups = []
    while len(rest) > 2:
        groups.append(rest[-2:])
        rest = rest[:-2]
    if rest:
        groups.append(rest)
    groups.reverse()
    return "₹" + ",".join(groups) + "," + last3
