"""
Explainable AI (XAI) service.

Generates human-readable explanations for every inspection decision:
- Why the appliance was classified as X
- What damage was detected and where
- How much area is affected
- How severity and condition were computed
- Why the claim decision was made
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_appliance_explanation(
    appliance: str,
    confidence: float,
    top_predictions: Optional[List] = None,
) -> str:
    parts = [f"Detected appliance: **{appliance}** (confidence: {confidence:.0%})."]
    if top_predictions and len(top_predictions) > 1:
        others = ", ".join(f"{p[0]} ({p[1]:.0%})" for p in top_predictions[1:])
        parts.append(f" Alternative predictions: {others}.")
    if confidence < 0.5:
        parts.append(" Confidence is moderate — manual verification recommended.")
    if confidence < 0.35:
        parts.append(" Low confidence — appliance type may be incorrect.")
    return " ".join(parts)


def build_damage_explanation(
    damage_detections: List[Dict[str, Any]],
    severity: str,
    condition_score: int,
    grade: str,
) -> str:
    if not damage_detections:
        return "No damage detected. The appliance appears to be in good condition."

    parts = [f"Found **{len(damage_detections)}** damage(s):"]
    for i, d in enumerate(damage_detections, 1):
        dt = d.get("class_name", "unknown")
        conf = d.get("confidence", 0)
        loc = d.get("location", "unknown area")
        bbox = d.get("bbox", [0, 0, 0, 0])
        area_pct = round((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / 10000, 1)
        parts.append(
            f"  {i}. **{dt}** at **{loc}** "
            f"(confidence: {conf:.0%}, area: ~{area_pct}% of image)."
        )

    parts.append(f"")
    parts.append(f"Overall severity: **{severity}**.")
    parts.append(f"Condition score: **{condition_score}/100** (grade **{grade}**).")

    if severity == "Minor":
        parts.append(" Damage is minor — cosmetic only, no functional impact expected.")
    elif severity == "Moderate":
        parts.append(" Moderate damage — repair recommended but appliance still usable.")
    elif severity == "Major":
        parts.append(" Major damage — significant impact on function and value.")
    elif severity == "Severe":
        parts.append(" Severe damage — appliance may be beyond economical repair.")

    return " ".join(parts)


def build_fraud_explanation(
    fraud_score: int,
    risk_level: str,
    reasons: List[str],
    ela_score: float,
) -> str:
    parts = [f"Fraud analysis: score **{fraud_score}/100** (risk: **{risk_level}**)."]
    if reasons:
        parts.append(" Indicators:")
        for r in reasons:
            parts.append(f"  - {r}")
    if ela_score > 0.5:
        parts.append(" ELA score elevated — possible image manipulation.")
    if fraud_score < 30:
        parts.append(" No significant fraud indicators detected.")
    return " ".join(parts)


def build_repair_explanation(
    severity: str = "",
    breakdown: Optional[List[Dict]] = None,
) -> str:
    parts = [
        "The detected damage affects a critical component. "
        "Replacement or major repair may be required. "
        "A service-center quotation is recommended before claim approval."
    ]
    if severity == "Minor":
        parts.append("Damage is cosmetic — minor repair only.")
    elif severity == "Moderate":
        parts.append("Moderate damage — professional assessment recommended.")
    elif severity in ("Major", "Severe"):
        parts.append("Severe damage — replacement or major repair may be required.")
    if breakdown:
        parts.append(" Affected components:")
        for item in breakdown:
            dt = item.get("damage_type", "unknown")
            sev = item.get("severity", "Minor")
            parts.append(f"  - {dt} ({sev})")
    return " ".join(parts)


def build_claim_explanation(
    claim_risk: str,
    claim_score: int,
    decision: str,
    severity: str,
    fraud_score: int,
) -> str:
    parts = [f"Claim assessment: risk **{claim_risk}** (score: **{claim_score}/100**)."]
    parts.append(f" Recommended decision: **{decision}**.")

    if decision == "APPROVE":
        parts.append(" Low risk — no issues detected. Fast-track processing.")
    elif decision == "MANUAL_REVIEW":
        reasons = []
        if severity in ("Major", "Severe"):
            reasons.append("damage severity is high")
        if fraud_score > 40:
            reasons.append("fraud score is elevated")
        if reasons:
            parts.append(f" Manual review needed because {' and '.join(reasons)}.")
        else:
            parts.append(" Manual review recommended as a precaution.")
    elif decision == "REJECT":
        parts.append(" Claim rejected due to high risk indicators.")
        if fraud_score > 60:
            parts.append(" Primary reason: elevated fraud risk.")
    return " ".join(parts)


def build_full_explanation(
    appliance: str,
    appliance_conf: float,
    top_preds: Optional[List],
    damage_detections: List[Dict],
    severity: str,
    condition_score: int,
    grade: str,
    fraud_score: int,
    fraud_risk: str,
    fraud_reasons: List[str],
    ela_score: float,
    claim_risk: str,
    claim_score: int,
    decision: str,
    repair_breakdown: Optional[List[Dict]] = None,
) -> Dict[str, str]:
    return {
        "appliance": build_appliance_explanation(appliance, appliance_conf, top_preds),
        "damage": build_damage_explanation(
            damage_detections, severity, condition_score, grade,
        ),
        "fraud": build_fraud_explanation(fraud_score, fraud_risk, fraud_reasons, ela_score),
        "repair": build_repair_explanation(severity=severity, breakdown=repair_breakdown),
        "claim": build_claim_explanation(claim_risk, claim_score, decision, severity, fraud_score),
    }
