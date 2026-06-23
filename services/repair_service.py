"""
Repair assessment based on severity, not monetary estimates.

Maps damage severity to qualitative repair impact, repairability,
and recommended action. No fake prices.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

SEVERITY_IMPACT_MAP = {
    "Minor": "Low",
    "Moderate": "Medium",
    "Major": "High",
    "Severe": "High",
    "None": "Low",
}

SEVERITY_REPAIRABILITY_MAP = {
    "Minor": "Repairable",
    "Moderate": "Major Repair Required",
    "Major": "Assembly/Component Replacement Recommended",
    "Severe": "Screen/Panel Replacement Recommended",
    "None": "No Repair Needed",
}

SEVERITY_ACTION_MAP = {
    "Minor": "Minor Repair — Cosmetic Only",
    "Moderate": "Professional Service-Center Assessment Recommended",
    "Major": "Service-Center Quotation Required Before Claim Approval",
    "Severe": "Manual Service-Center Inspection Required",
    "None": "No Action Required",
}


def assess_repair_impact(severity: str) -> str:
    return SEVERITY_IMPACT_MAP.get(severity, "Medium")


def assess_repairability(severity: str) -> str:
    return SEVERITY_REPAIRABILITY_MAP.get(severity, "Repairable")


def assess_recommended_action(severity: str) -> str:
    return SEVERITY_ACTION_MAP.get(severity, "Professional Assessment Recommended")


def estimate_total_repair_cost(
    damage_assessments: List[Dict[str, Any]],
    damage_detections: List[Dict[str, Any]],
    brand: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns qualitative repair assessment based on severity.
    No monetary values are generated.
    """
    if not damage_assessments or not damage_detections:
        return {
            "repair_impact": "None",
            "repairability": "No Repair Needed",
            "recommended_action": "No Action Required",
            "breakdown": [],
        }

    severities = [d.get("severity", "Minor") for d in damage_assessments]
    max_sev = _max_severity(severities)

    return {
        "repair_impact": assess_repair_impact(max_sev),
        "repairability": assess_repairability(max_sev),
        "recommended_action": assess_recommended_action(max_sev),
        "breakdown": [
            {
                "damage_type": d.get("damage_type", "unknown"),
                "severity": d.get("severity", "Minor"),
                "repair_impact": assess_repair_impact(d.get("severity", "Minor")),
                "repairability": assess_repairability(d.get("severity", "Minor")),
            }
            for d in damage_assessments
        ],
    }


_SEVERITY_ORDER = {"None": 0, "Minor": 1, "Moderate": 2, "Major": 3, "Severe": 4}


def _max_severity(severities: List[str]) -> str:
    if not severities:
        return "None"
    return max(severities, key=lambda s: _SEVERITY_ORDER.get(s, 0))
