"""
Feature 8: Claim Risk Engine

Combines damage severity, fraud score, missing parts, and appliance type
into a single business risk score (0-100) and claim risk category.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


SEVERITY_RISK_MAP = {
    "None": 0,
    "Minor": 15,
    "Moderate": 35,
    "Major": 60,
    "Severe": 85,
}

APPLIANCE_RISK_WEIGHTS = {
    "phone": 1.0,
    "television": 1.1,
    "laptop": 1.2,
    "refrigerator": 1.0,
    "tablet": 1.0,
    "monitor": 1.1,
    "microwave": 0.9,
    "washing_machine": 1.0,
    "air_conditioner": 1.2,
}

CLAIM_RISK_BANDS = [
    ("Low", 0, 35),
    ("Medium", 35, 65),
    ("High", 65, 101),
]


@dataclass
class ClaimRiskResult:
    claim_score: int
    claim_risk: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_score": self.claim_score,
            "claim_risk": self.claim_risk,
            "details": self.details,
        }


def _risk_label(score: int) -> str:
    for label, lo, hi in CLAIM_RISK_BANDS:
        if lo <= score < hi:
            return label
    return "High"


def assess_claim_risk(
    appliance: str,
    damage_severity: str,
    fraud_score: int,
    missing_part_detected: bool,
) -> ClaimRiskResult:
    severity_base = SEVERITY_RISK_MAP.get(damage_severity, 0)
    appliance_weight = APPLIANCE_RISK_WEIGHTS.get(appliance.lower(), 1.0)

    weighted_severity = severity_base * appliance_weight

    fraud_contribution = fraud_score * 0.30
    missing_contribution = 15 if missing_part_detected else 0

    claim_score = int(min(weighted_severity + fraud_contribution + missing_contribution, 100))
    claim_risk = _risk_label(claim_score)

    return ClaimRiskResult(
        claim_score=claim_score,
        claim_risk=claim_risk,
        details={
            "appliance": appliance,
            "damage_severity": damage_severity,
            "severity_base_score": severity_base,
            "appliance_weight": appliance_weight,
            "weighted_severity": round(weighted_severity, 1),
            "fraud_contribution": round(fraud_contribution, 1),
            "missing_part_contribution": missing_contribution,
        },
    )
