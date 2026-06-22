"""
Professional PDF Report Generation for Insurance Claims.

Generates insurer-ready PDF reports with:
- Original and annotated images (side-by-side if space permits)
- Detection results with confidence, location, area
- Severity assessment with breakdown
- Repair cost estimate with per-damage breakdown
- Fraud analysis with explanations
- Explainable AI reasoning section
- Claim recommendation with justification
- Model version, timestamp, and report metadata
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, Optional, Union

from report_engine import InspectionReport


def _get(report: Union[InspectionReport, Dict[str, Any]], key: str, default: Any = "") -> Any:
    if isinstance(report, dict):
        return report.get(key, default)
    return getattr(report, key, default)


def _safe(val: Any, default: str = "N/A") -> str:
    if val is None:
        return default
    return str(val)


def _get_assessment(report: Dict[str, Any], key: str, default: Any = "") -> Any:
    assess = _get(report, "assessment", report)
    if isinstance(assess, dict):
        return assess.get(key, default)
    return default


def generate_pdf_report(
    report: Union[InspectionReport, Dict[str, Any]],
    annotated_image_path: Optional[str] = None,
    original_image_path: Optional[str] = None,
    output_dir: str = "reports",
) -> str:
    try:
        from fpdf import FPDF
    except ImportError:
        raise ImportError("fpdf2 is required. Install: pip install fpdf2")

    os.makedirs(output_dir, exist_ok=True)
    claim_id = _get(report, "report_id", "UNKNOWN")
    output_path = os.path.join(output_dir, f"claim_report_{claim_id}.pdf")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ---- Page 1: Header & Summary ----
    pdf.add_page()
    pdf.set_fill_color(25, 55, 109)
    pdf.rect(0, 0, 210, 40, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_y(10)
    pdf.cell(0, 12, "Claim Inspection Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, f"Claim ID: {claim_id}  |  {_get(report, 'timestamp', '')[:10]}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)

    pdf.ln(10)
    decision = _get_assessment(report, "decision", _get(report, "decision", "N/A"))
    if decision == "APPROVE":
        pdf.set_fill_color(212, 237, 218)
        pdf.set_text_color(21, 87, 36)
    elif decision == "REJECT":
        pdf.set_fill_color(248, 215, 218)
        pdf.set_text_color(114, 28, 36)
    else:
        pdf.set_fill_color(255, 243, 205)
        pdf.set_text_color(133, 100, 4)
    pdf.cell(0, 12, f"  Recommendation: {decision}", fill=True, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # Appliance & Key Metrics
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "1. Appliance Details", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 7, f"  Appliance: {_get(report, 'appliance', 'N/A')}", new_x="LMARGIN", new_y="NEXT")
    conf = _get(report, "appliance_confidence", 0)
    pdf.cell(0, 7, f"  Confidence: {conf:.0%}" if isinstance(conf, (int, float)) else "  Confidence: N/A", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"  Condition: {_get_assessment(report, 'condition_score', _get(report, 'condition_score', 0))}/100 (Grade: {_get_assessment(report, 'grade', _get(report, 'grade', 'N/A'))})", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"  Source: {_get(report, 'source_type', 'image')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Damage Detection
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "2. Damage Assessment", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)

    all_detections = _get(report, "damage_detections", [])
    if hasattr(report, 'to_dict'):
        report_dict = report.to_dict() if hasattr(report, 'to_dict') else {}
    else:
        report_dict = report
    damage_dict = _get(report_dict, "damage", report_dict)
    detections = damage_dict.get("all_detections", all_detections)

    if detections:
        for i, det in enumerate(detections):
            dt = det.get("class_name", "unknown").replace("_", " ").title()
            dconf = det.get("confidence", 0)
            loc = det.get("location", "unknown").replace("_", " ").title()
            bbox = det.get("bbox", [0, 0, 0, 0])
            area_pct = round((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / 10000, 1) if bbox else 0
            pdf.cell(0, 7, f"  {i+1}. {dt} @ {loc} (conf: {dconf:.0%}, area: ~{area_pct}%)", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.cell(0, 7, "  No damage detected", new_x="LMARGIN", new_y="NEXT")

    sev = _get_assessment(report, "severity", _get(report, "severity", "None"))
    pdf.cell(0, 7, f"  Overall Severity: {sev}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Severity & Damage Breakdown
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "3. Severity Breakdown", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    repair_breakdown = _get_assessment(report, "repair_breakdown", [])
    if repair_breakdown:
        for item in repair_breakdown:
            dt = item.get("damage_type", "?").replace("_", " ").title()
            b_range = item.get("base_range", "")
            s_mult = item.get("severity_multiplier", 1)
            pdf.cell(0, 7, f"  {dt}: Base {b_range}, Severity Mult: x{s_mult}", new_x="LMARGIN", new_y="NEXT")

    cost_display = _get_assessment(report, "repair_cost_display", _get(report, "repair_cost_display", ""))
    if not cost_display:
        cost_display = f"\u20b9{_get_assessment(report, 'repair_cost', _get(report, 'repair_cost', 0)):,}"
    pdf.cell(0, 7, f"  Estimated Total: {cost_display}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Fraud Analysis
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "4. Fraud Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    fraud_dict = _get(report_dict, "fraud", report)
    fraud_score = _get_assessment(report, "fraud_score", fraud_dict.get("fraud_score", 0))
    pdf.cell(0, 7, f"  Fraud Score: {fraud_score}/100", new_x="LMARGIN", new_y="NEXT")
    fraud_risk = fraud_dict.get("fraud_risk_level", _get(report, "fraud_risk_level", "Low"))
    pdf.cell(0, 7, f"  Risk Level: {fraud_risk}", new_x="LMARGIN", new_y="NEXT")
    for reason in fraud_dict.get("fraud_reasons", _get(report, "fraud_reasons", [])):
        pdf.cell(0, 7, f"    - {reason}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Explainable AI
    explanations = _get(report_dict, "explanations", {})
    if explanations:
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "5. Explainable AI Reasoning", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        for section, label in [("appliance", "Appliance Classification"),
                                ("damage", "Damage Detection"),
                                ("fraud", "Fraud Analysis"),
                                ("repair", "Repair Estimate"),
                                ("claim", "Claim Decision")]:
            text = explanations.get(section, "")
            if text:
                pdf.set_font("Helvetica", "B", 10)
                pdf.cell(0, 7, f"  {label}:", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 9)
                pdf.multi_cell(0, 5, f"  {text}")
                pdf.ln(2)

    pdf.ln(3)

    # Images
    if original_image_path and os.path.exists(original_image_path):
        try:
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, "6. Inspection Images", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(0, 7, "Original Image:", new_x="LMARGIN", new_y="NEXT")
            pdf.image(original_image_path, x=10, w=90)
            pdf.ln(55)
        except Exception:
            pass

    if annotated_image_path and os.path.exists(annotated_image_path):
        try:
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(0, 7, "Annotated Image:", new_x="LMARGIN", new_y="NEXT")
            pdf.image(annotated_image_path, x=10, w=180)
            pdf.ln(3)
        except Exception:
            pass

    # Missing Parts
    if _get(report, "missing_part_detected", False):
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "7. Missing Parts", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 7, f"  Missing: {_get(report, 'missing_part', 'unknown')}", new_x="LMARGIN", new_y="NEXT")
        for w in _get(report, "missing_part_warnings", []):
            pdf.cell(0, 7, f"    - {w}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

    # Claim Summary
    pdf.add_page()
    pdf.set_fill_color(25, 55, 109)
    pdf.rect(0, 0, 210, 30, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_y(8)
    pdf.cell(0, 12, "Claim Summary & Recommendation", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Decision Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)

    claim_risk = _get_assessment(report, "claim_risk", _get(report, "claim_risk", "N/A"))
    claim_score = _get_assessment(report, "claim_score", _get(report, "claim_score", 0))
    condition_score = _get_assessment(report, "condition_score", _get(report, "condition_score", 100))
    grade = _get_assessment(report, "grade", _get(report, "grade", "A"))

    pdf.cell(0, 7, f"  Claim Risk: {claim_risk} (Score: {claim_score}/100)", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"  Condition: {condition_score}/100 (Grade: {grade})", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"  Severity: {sev}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"  Fraud Score: {fraud_score}/100", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"  Decision: {decision}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 7, f"  Repair Estimate: {cost_display}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # Justification
    justification = _get_assessment(report, "claim_justification", "")
    if justification:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 10, "Justification", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, f"  {justification}")
        pdf.ln(5)

    # Metadata
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 10, "Report Metadata", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    meta = _get(report, "metadata", {})
    if isinstance(meta, dict):
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                pdf.cell(0, 5, f"  {k}: {v}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, f"  Report ID: {claim_id}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, f"  Platform: AI Appliance Inspection v3.0", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(5)
    pdf.set_fill_color(230, 240, 255)
    pdf.set_text_color(0, 51, 102)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 7, "IRDAI Compliance", fill=True, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(50, 50, 100)
    pdf.multi_cell(0, 4, "This report is AI-generated as an advisory tool under IRDAI (Insurance Advertisements and Disclosure) Regulations, 2021. It does not replace a licensed insurance surveyor's assessment. All claim decisions must be verified by a qualified adjuster in accordance with IRDAI guidelines.", align="C")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, "This report is AI-generated and should be reviewed by a qualified insurance adjuster before final decisions.", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 5, f"AI Appliance Inspection & Insurance Platform | {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.output(output_path)
    return output_path
