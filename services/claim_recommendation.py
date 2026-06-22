"""
Claim recommendation engine.

Decision engine that classifies claims into:
- APPROVE: Low risk, no issues → fast-track
- MANUAL_REVIEW: Some risk flags → human review needed
- REJECT: High risk → deny claim

Uses severity, fraud score, and condition score to compute risk.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


CLAIM_RISK_LEVELS = ["low", "medium", "high", "critical"]


def assess_claim(
    severity: str,
    fraud_score: int,
    condition_score: int,
    damage_count: int,
) -> Dict[str, Any]:
    severity_risk = {"None": 0, "Minor": 10, "Moderate": 30, "Major": 60, "Severe": 80}
    sev_risk = severity_risk.get(severity, 10)

    if condition_score >= 90:
        cond_risk = 0
    elif condition_score >= 75:
        cond_risk = 15
    elif condition_score >= 50:
        cond_risk = 40
    else:
        cond_risk = 70

    if damage_count == 0:
        count_risk = 0
    elif damage_count == 1:
        count_risk = 10
    elif damage_count <= 3:
        count_risk = 30
    else:
        count_risk = 50

    claim_score = min(
        int(sev_risk * 0.4 + fraud_score * 0.3 + cond_risk * 0.2 + count_risk * 0.1),
        100,
    )

    if claim_score < 25:
        risk = "low"
        decision = "APPROVE"
    elif claim_score < 50:
        risk = "medium"
        decision = "MANUAL_REVIEW"
    elif claim_score < 75:
        risk = "high"
        decision = "MANUAL_REVIEW"
    else:
        risk = "critical"
        decision = "REJECT"

    return {
        "claim_score": claim_score,
        "claim_risk": risk,
        "decision": decision,
        "severity_risk": sev_risk,
        "condition_risk": cond_risk,
        "fraud_risk": fraud_score,
        "damage_count_risk": count_risk,
    }


def build_justification(
    claim_result: Dict[str, Any],
    severity: str,
    fraud_score: int,
    fraud_reasons: List[str],
    condition_grade: str,
) -> str:
    parts: List[str] = []
    decision = claim_result.get("decision", "MANUAL_REVIEW")

    if decision == "APPROVE":
        parts.append("Claim qualifies for automatic approval.")
        if condition_grade in ("A", "B"):
            parts.append("Appliance condition is good.")
    elif decision == "REJECT":
        parts.append("Claim recommended for rejection.")
        if fraud_score > 60:
            parts.append(f"Fraud indicators present: {'; '.join(fraud_reasons[:2])}")
        if severity == "Severe":
            parts.append("Damage severity is at maximum level.")
    else:
        parts.append("Claim requires manual review.")
        if fraud_score > 40:
            parts.append(f"Fraud score elevated ({fraud_score}/100).")
        if severity in ("Major", "Severe"):
            parts.append(f"Damage is {severity.lower()}.")
        if condition_grade in ("C", "D"):
            parts.append("Appliance condition is below threshold.")

    return " ".join(parts)
