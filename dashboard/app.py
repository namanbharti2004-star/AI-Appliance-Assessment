"""
Professional Streamlit dashboard for AI Appliance Inspection Platform.

Features:
- Side-by-side original / annotated image comparison (no overlay on original)
- Confidence bars for appliance and damage detections
- Explanation panel showing why each decision was made
- Insurance claim workflow with recommendation
- Fraud analysis with detailed breakdown
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import DASHBOARD_CONFIG, MVP_APPLIANCE_CLASSES
from scripts.inference import InspectionPipeline
from services.claim_service import get_claim_by_id, get_claim_stats, get_claims, save_claim
from services.pdf_service import generate_pdf_report
from services.explain_service import build_full_explanation
from services.repair_service import estimate_total_repair_cost, assess_repair_impact, assess_repairability, assess_recommended_action
from services.severity_service import compute_condition_score, compute_grade, assess_all_damages
from services.claim_recommendation import assess_claim, build_justification

st.set_page_config(
    page_title="AI Appliance Inspection · Insurance Platform",
    page_icon="\u2699\ufe0f",
    layout=DASHBOARD_CONFIG["layout"],
    initial_sidebar_state="expanded",
)

def inject_dashboard_theme() -> None:
    PRIMARY_BG = "#0E1117"
    CARD_BG = "#1E2633"
    CARD_BORDER = "#374151"
    TEXT_PRIMARY = "#FFFFFF"
    TEXT_SECONDARY = "#D1D5DB"
    TEXT_MUTED = "#9CA3AF"
    ACCENT = "#93C5FD"
    SUCCESS = "#10B981"
    WARNING = "#F59E0B"
    DANGER = "#EF4444"

    st.markdown(f"""
<style>
    /* ── GLOBAL TEXT VISIBILITY ── */
    * {{ opacity: 1 !important; }}
    html, body, .stApp, .main, .block-container,
    p, span, div, label, h1, h2, h3, h4, h5, h6,
    li, ul, ol, a, small, strong, em, b, i,
    code, pre, .stMarkdown, .stText, .stCaption {{
        color: {TEXT_PRIMARY} !important;
    }}
    .stApp {{
        background: {PRIMARY_BG};
    }}

    /* ── STREAMLIT METRICS ── */
    [data-testid="stMetric"] {{
        background: {CARD_BG} !important;
        border: 1px solid {CARD_BORDER} !important;
        border-radius: 12px !important;
        padding: 16px !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {TEXT_PRIMARY} !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {TEXT_SECONDARY} !important;
        font-size: 0.85rem !important;
    }}

    /* ── TABS ── */
    .stTabs [role="tabpanel"] {{
        color: {TEXT_PRIMARY} !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {TEXT_SECONDARY} !important;
    }}
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: {TEXT_PRIMARY} !important;
    }}

    /* ── EXPANDERS ── */
    .streamlit-expanderContent,
    .streamlit-expanderContent * {{
        color: {TEXT_PRIMARY} !important;
    }}
    .streamlit-expanderHeader {{
        color: {TEXT_SECONDARY} !important;
    }}

    /* ── DATAFRAMES / TABLES ── */
    table, thead, tbody, tr, td, th,
    [data-testid="StyledDataFrameColHeader"],
    [data-testid="StyledDataFrameDataCell"] {{
        color: {TEXT_PRIMARY} !important;
        background-color: transparent !important;
    }}
    thead, [data-testid="StyledDataFrameColHeader"] {{
        background: {CARD_BG} !important;
        color: {TEXT_SECONDARY} !important;
    }}
    tr:nth-child(even) td {{
        background: rgba(255,255,255,0.03) !important;
    }}

    /* ── ALERT OVERRIDES (success, warning, info, error) ── */
    .stAlert,
    [data-testid="stAlert"] {{
        background: {CARD_BG} !important;
        border: 1px solid {CARD_BORDER} !important;
        color: {TEXT_PRIMARY} !important;
    }}
    .stAlert p, .stAlert span, .stAlert div {{
        color: {TEXT_PRIMARY} !important;
    }}

    /* ── BUTTONS ── */
    .stButton button {{
        color: {TEXT_PRIMARY} !important;
        background: {CARD_BG} !important;
        border: 1px solid {CARD_BORDER} !important;
    }}
    .stButton button[kind="primary"] {{
        background: {ACCENT}22 !important;
        border-color: {ACCENT} !important;
    }}

    /* ── SELECTBOX / INPUT ── */
    [data-testid="stSelectbox"] div,
    [data-testid="stSelectbox"] span,
    [data-testid="stSelectbox"] label {{
        color: {TEXT_PRIMARY} !important;
    }}
    [data-testid="stSelectbox"] [role="listbox"] {{
        background: {CARD_BG} !important;
    }}

    /* ── SIDEBAR ── */
    .css-1d391kg, .css-1wrcr25, section[data-testid="stSidebar"] {{
        background: {PRIMARY_BG} !important;
    }}
    .css-1d391kg *, section[data-testid="stSidebar"] * {{
        color: {TEXT_PRIMARY} !important;
    }}

    /* ── CUSTOM CLASSES / BADGES ── */
    .badge {{
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 0.8rem; font-weight: 600;
    }}
    .badge-green {{ background: {SUCCESS}22; color: {SUCCESS} !important; border: 1px solid {SUCCESS}; }}
    .badge-yellow {{ background: {WARNING}22; color: {WARNING} !important; border: 1px solid {WARNING}; }}
    .badge-red {{ background: {DANGER}22; color: {DANGER} !important; border: 1px solid {DANGER}; }}
    .badge-gray {{ background: {CARD_BG}; color: {TEXT_SECONDARY} !important; border: 1px solid {CARD_BORDER}; }}

    /* ── EXPLANATION CARDS (XAI) ── */
    .explanation-box {{
        background: {CARD_BG}; color: {TEXT_PRIMARY};
        padding: 15px; border-radius: 10px; margin: 8px 0;
    }}
    .explanation-box p {{ color: {TEXT_PRIMARY}; margin: 0; }}
    .explanation-box strong {{ color: {ACCENT}; }}
    .xai-section {{ margin: 4px 0; }}
    .xai-section + .xai-section {{ margin-top: 8px; }}
    .xai-label {{ color: {ACCENT}; font-weight: 700; font-size: 0.9rem; letter-spacing: 0.03em; margin-bottom: 4px; }}

    /* ── SEVERITY INDICATOR ── */
    .severity-indicator {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 6px; }}

    /* ── DECISION BANNER ── */
    .decision-banner {{
        padding: 12px; border-radius: 8px; text-align: center;
        font-weight: 700; font-size: 1.1rem;
    }}
    .approve {{ background: {SUCCESS}22; color: {SUCCESS} !important; border: 1px solid {SUCCESS}; }}
    .review {{ background: {WARNING}22; color: {WARNING} !important; border: 1px solid {WARNING}; }}
    .reject {{ background: {DANGER}22; color: {DANGER} !important; border: 1px solid {DANGER}; }}

    /* ── SIDEBAR INFO BOX ── */
    .sidebar-info {{
        background: {CARD_BG}; border-radius: 8px; padding: 8px;
        font-size: 0.75rem; border-left: 3px solid {ACCENT};
        color: {TEXT_PRIMARY};
    }}
    .sidebar-info b {{ color: {ACCENT}; }}

    /* ── CONFIDENCE BAR ── */
    .conf-bar-bg {{
        background: {CARD_BORDER}; border-radius: 4px; height: 10px;
    }}
    .conf-bar-fill {{
        height: 10px; border-radius: 4px;
    }}
    .conf-bar-label {{
        color: {TEXT_SECONDARY};
    }}

    /* ── CONSTRAINED VIDEO (expander inside centered column) ── */
    .stVideo {{
        max-width: 100% !important;
    }}
    .stVideo video {{
        max-width: 100% !important;
        max-height: 50vh !important;
        width: auto !important;
        height: auto !important;
        border-radius: 12px !important;
        border: 1px solid {CARD_BORDER} !important;
        object-fit: contain !important;
    }}

    /* ── CONSTRAINED IMAGE PREVIEW ── */
    .media-preview img {{
        max-width: 100% !important;
        max-height: 450px !important;
        width: auto !important;
        height: auto !important;
        border-radius: 12px !important;
        border: 1px solid {CARD_BORDER} !important;
        display: block !important;
        margin: 0 auto !important;
        object-fit: contain !important;
    }}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline() -> InspectionPipeline:
    return InspectionPipeline()


def _save_upload(uploaded_file, suffix: str) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getvalue())
    tmp.flush()
    tmp.close()
    return tmp.name



def _confidence_bar(confidence: float, label: str, max_width: int = 200) -> str:
    pct = int(confidence * 100)
    color = "#10B981" if pct >= 70 else ("#F59E0B" if pct >= 40 else "#EF4444")
    return f"""
    <div style="margin: 4px 0;">
        <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
            <span style="color: #D1D5DB;">{label}</span><span style="color: #FFFFFF;">{pct}%</span>
        </div>
        <div class="conf-bar-bg" style="width: {max_width}px;">
            <div class="conf-bar-fill" style="background: {color}; width: {pct}%;"></div>
        </div>
    </div>"""


def _decision_banner(decision: str, claim_risk: str) -> str:
    if decision == "APPROVE":
        cls = "approve"
        icon = "\u2705"
    elif decision == "REJECT":
        cls = "reject"
        icon = "\u274c"
    else:
        cls = "review"
        icon = "\u26a0\ufe0f"
    return f'<div class="decision-banner {cls}">{icon} {decision} &mdash; Risk: {claim_risk.upper()}</div>'


def _constrained_image(image, caption: str = "") -> None:
    st.caption(caption)
    st.markdown(f'<div class="media-preview">', unsafe_allow_html=True)
    st.image(image, use_container_width=False, width=None)
    st.markdown('</div>', unsafe_allow_html=True)


def _render_image_comparison(original: Any, annotated: Any, report: Dict[str, Any]) -> None:
    st.subheader("Image Comparison")
    col1, col2 = st.columns(2)
    with col1:
        _constrained_image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB), caption="Original Image (no overlay)")
    with col2:
        if annotated is not None:
            _constrained_image(annotated, caption="Annotated with Detections")
        else:
            st.info("No annotations available yet. Run inspection to generate.")


def _render_key_metrics(report: Dict[str, Any]) -> None:
    assess = report.get("assessment", report)
    decision = assess.get("decision", "MANUAL_REVIEW")
    claim_risk = assess.get("claim_risk", "low")

    st.markdown(_decision_banner(decision, claim_risk), unsafe_allow_html=True)
    st.markdown("")

    cols = st.columns(5)
    cols[0].metric("Appliance", report.get("appliance", "N/A"))
    cols[1].metric("Condition", f"{assess.get('condition_score', 0)}/100", assess.get("grade", "N/A"))

    sev = assess.get("severity", "None")
    cols[2].metric("Severity", sev)

    cols[3].metric("Repair Impact", assess.get("repair_impact", "N/A"))
    cols[4].metric("Fraud Risk", f'{assess.get("fraud_score", 0)}/100')


def _render_confidence_bars(report: Dict[str, Any]) -> None:
    st.subheader("Detection Confidence")
    app_conf = report.get("appliance_confidence", report.get("confidence", 0))
    st.markdown(_confidence_bar(app_conf, f"Appliance: {report.get('appliance', 'N/A')}"), unsafe_allow_html=True)

    damage_list = report.get("damage_detections", [])
    if damage_list:
        for d in damage_list:
            st.markdown(_confidence_bar(d.get("confidence", 0), f"  {d.get('class_name', '?')} @ {d.get('location', '?')}"), unsafe_allow_html=True)
    else:
        st.markdown('<div style="color: #10B981; font-weight: 600;">\u2705 No damage detected</div>', unsafe_allow_html=True)


def _render_explanations(report: Dict[str, Any]) -> None:
    st.subheader("Explainable AI")
    explanations = report.get("explanations", {})

    sections = [
        ("appliance", "Appliance Classification"),
        ("damage", "Damage Assessment"),
        ("fraud", "Fraud Analysis"),
        ("repair", "Repair Estimate"),
        ("claim", "Claim Recommendation"),
    ]
    for section, label in sections:
        text = explanations.get(section, "")
        if text:
            st.markdown(
                f'<div class="xai-section">'
                f'<div class="xai-label">{label}</div>'
                f'<div class="explanation-box">{text}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


def _render_damage_details(report: Dict[str, Any]) -> None:
    dets = report.get("damage_detections", [])
    st.subheader("Damage Assessment")
    if dets:
        rows = []
        for d in dets:
            bbox = d.get("bbox", [0, 0, 0, 0])
            area_pct = round((bbox[2]-bbox[0]) * (bbox[3]-bbox[1]) / 10000, 1) if bbox else 0
            rows.append({
                "Type": d.get("class_name", "unknown").replace("_", " ").title(),
                "Confidence": f'{d.get("confidence", 0):.0%}',
                "Location": d.get("location", "N/A").replace("_", " ").title(),
                "Area %": area_pct,
                "Source": d.get("source", "yolo"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.success("\u2705 No damage detected. Appliance appears in good condition.")

    sev = report.get("severity", "None")
    sev_color = {"None": "#9CA3AF", "Minor": "#10B981", "Moderate": "#F59E0B", "Major": "#F97316", "Severe": "#EF4444"}.get(sev, "#9CA3AF")
    st.markdown(f"<div style='color: #D1D5DB;'>Overall Severity: <span style='color:{sev_color}; font-weight:700;'>{sev}</span></div>", unsafe_allow_html=True)


def _render_repair_details(report: Dict[str, Any]) -> None:
    st.subheader("Repair Assessment")
    impact = report.get("repair_impact", "None")
    repairability = report.get("repairability", "No Repair Needed")
    action = report.get("recommended_action", "No Action Required")

    c1, c2 = st.columns(2)
    c1.metric("Repair Impact", impact)
    c2.metric("Repairability", repairability)
    st.info(f"**Recommended Action:** {action}")

    breakdown = report.get("repair_breakdown", [])
    if breakdown:
        st.markdown("### Affected Components")
        rows = []
        for item in breakdown:
            rows.append({
                "Damage": item.get("damage_type", "?").replace("_", " ").title(),
                "Severity": item.get("severity", "Minor"),
                "Repair Impact": item.get("repair_impact", "Low"),
                "Repairability": item.get("repairability", "Repairable"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_fraud_details(report: Dict[str, Any]) -> None:
    st.subheader("Fraud Analysis")
    c1, c2, c3 = st.columns(3)
    c1.metric("Advanced Score", f'{report.get("fraud_score", 0)}/100')
    c2.metric("ELA Score", f'{report.get("ela_score", 0):.2f}')
    c3.metric("Metadata Risk", f'{report.get("metadata_risk_score", 0):.2f}')

    reasons = report.get("fraud_reasons", [])
    if reasons:
        for i, r in enumerate(reasons):
            st.markdown(f'<div class="explanation-box" style="padding: 8px 12px; margin: 4px 0;">{r}</div>', unsafe_allow_html=True)

    risk = report.get("fraud_risk_level", "Low")
    badge = {"Low": "badge-green", "Medium": "badge-yellow", "High": "badge-red", "Critical": "badge-red"}.get(risk, "badge-gray")
    st.markdown(f'Fraud Risk: <span class="badge {badge}">{risk}</span>', unsafe_allow_html=True)


def _render_claim_details(report: Dict[str, Any]) -> None:
    st.subheader("Claim Recommendation")
    decision = report.get("decision", "MANUAL_REVIEW")
    claim_risk = report.get("claim_risk", "low")
    claim_score = report.get("claim_score", 0)

    cols = st.columns(3)
    cols[0].metric("Claim Score", f"{claim_score}/100")
    cols[1].metric("Risk Level", claim_risk.upper())

    badge = {"low": "badge-green", "medium": "badge-yellow", "high": "badge-red", "critical": "badge-red"}.get(claim_risk.lower(), "badge-gray")
    st.markdown(f'Decision: <span class="badge {badge}">{decision}</span>', unsafe_allow_html=True)

    justification = report.get("claim_justification", "")
    if justification:
        st.markdown(f'<div class="explanation-box">{justification}</div>', unsafe_allow_html=True)


def _build_explanations_for_report(report: Dict[str, Any]) -> Dict[str, str]:
    detections = report.get("damage_detections", [])

    appliance_name = report.get("appliance", "unknown")
    appliance_conf = report.get("appliance_confidence", 0)
    severity = report.get("severity", "None")
    condition_score = report.get("condition_score", 100)
    grade = report.get("grade", "A")
    fraud_score = report.get("fraud_score", 0)
    fraud_risk = report.get("fraud_risk_level", "Low")
    fraud_reasons = report.get("fraud_reasons", [])
    ela_score = report.get("ela_score", 0)
    claim_risk = report.get("claim_risk", "low")
    claim_score = report.get("claim_score", 0)
    decision = report.get("decision", "MANUAL_REVIEW")
    repair_breakdown = report.get("repair_breakdown", [])

    return build_full_explanation(
        appliance=appliance_name,
        appliance_conf=appliance_conf,
        top_preds=None,
        damage_detections=detections,
        severity=severity,
        condition_score=condition_score,
        grade=grade,
        fraud_score=fraud_score,
        fraud_risk=fraud_risk,
        fraud_reasons=fraud_reasons,
        ela_score=ela_score,
        claim_risk=claim_risk,
        claim_score=claim_score,
        decision=decision,
        repair_breakdown=repair_breakdown,
    )


def _enrich_report(report: Dict[str, Any], damage_detections: List[Dict], annotated_image: Optional[Any]) -> Dict[str, Any]:
    """Add enriched fields (explanations, breakdown, etc.) to report."""
    repair_result = estimate_total_repair_cost(
        assess_all_damages(damage_detections, image_shape=(640, 640)),
        damage_detections,
    )
    report["repair_impact"] = repair_result.get("repair_impact", "Medium")
    report["repairability"] = repair_result.get("repairability", "Repairable")
    report["recommended_action"] = repair_result.get("recommended_action", "Professional Assessment Recommended")
    report["repair_breakdown"] = repair_result.get("breakdown", [])

    condition_score = compute_condition_score(
        assess_all_damages(damage_detections, image_shape=(640, 640))
    )
    report["condition_score"] = condition_score
    report["grade"] = compute_grade(condition_score)

    claim_result = assess_claim(
        severity=report.get("severity", "None"),
        fraud_score=report.get("fraud_score", 0),
        condition_score=condition_score,
        damage_count=len(damage_detections),
    )
    report["claim_score"] = claim_result["claim_score"]
    report["claim_risk"] = claim_result["claim_risk"]
    report["decision"] = claim_result["decision"]
    report["claim_justification"] = build_justification(
        claim_result, report.get("severity", "None"),
        report.get("fraud_score", 0),
        report.get("fraud_reasons", []),
        report.get("grade", "A"),
    )

    explanations = _build_explanations_for_report(report)
    report["explanations"] = explanations
    return report


def image_tab() -> None:
    st.header("Image Inspection")
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp", "heic", "heif"], key="image")
    with col2:
        appliance_override = st.selectbox("Override appliance", ["auto"] + MVP_APPLIANCE_CLASSES, index=0)

    if not uploaded:
        return

    image_path = _save_upload(uploaded, suffix=os.path.splitext(uploaded.name)[1] or ".jpg")
    from utils import read_image
    original_image = read_image(image_path)
    if original_image is None:
        st.error("Could not read image file. Try a different format.")
        return

    _render_image_comparison(original_image, None, {})

    if st.button("Run Inspection", type="primary", use_container_width=True):
        with st.spinner("Analyzing appliance, damage, fraud, and risk..."):
            pipeline = get_pipeline()
            result = pipeline.inspect_image(
                image_path=image_path,
                appliance_override=None if appliance_override == "auto" else appliance_override,
                save_visualizations=True,
                output_dir="output",
            )

        if result.get("error"):
            st.error(result["error"])
            return

        report = result["report"]
        damage_detections = report.get("damage_detections", [])

        annotated_path = result.get("annotated_image_path", "")
        report = _enrich_report(report, damage_detections, annotated_path)

        annotated = None
        if annotated_path and os.path.exists(annotated_path):
            annotated_img = read_image(annotated_path)
            if annotated_img is not None:
                annotated = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        col_img1, col_img2 = st.columns(2)
        with col_img1:
            _constrained_image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption="Original (no overlay)")
        with col_img2:
            st.caption("Annotated Inspection")
            if annotated is not None:
                _constrained_image(annotated)
            else:
                st.warning("Annotated image unavailable; showing original instead.")
                _constrained_image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

        st.markdown("---")
        _render_key_metrics(report)
        st.markdown("---")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Confidence", "Damage", "Repair Cost", "Fraud", "Claim Recommendation",
        ])
        with tab1:
            _render_confidence_bars(report)
        with tab2:
            _render_damage_details(report)
        with tab3:
            _render_repair_details(report)
        with tab4:
            _render_fraud_details(report)
        with tab5:
            _render_claim_details(report)

        st.markdown("---")
        _render_explanations(report)

        if st.button("Save to History & Download PDF", use_container_width=True, type="secondary"):
            claim_id = save_claim(report)
            st.success(f"Claim {claim_id} saved!")
            try:
                pdf_path = generate_pdf_report(
                    report,
                    annotated_image_path=annotated_path,
                    output_dir="reports",
                )
                st.session_state.pdf_path = pdf_path
                st.session_state.pdf_filename = os.path.basename(pdf_path)
            except Exception as e:
                st.warning(f"PDF generation unavailable: {e}")

        if st.session_state.get("pdf_path") and os.path.exists(st.session_state.pdf_path):
            with open(st.session_state.pdf_path, "rb") as f:
                st.download_button(
                    label="Download PDF Report",
                    data=f,
                    file_name=st.session_state.pdf_filename,
                    mime="application/pdf",
                )


def video_tab() -> None:
    st.header("Video Inspection")
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded = st.file_uploader("Upload video", type=["mp4", "mov", "avi"], key="video")
    with col2:
        st.selectbox("Override", ["auto"] + MVP_APPLIANCE_CLASSES, index=0, key="vid_override")

    if not uploaded:
        return

    video_path = _save_upload(uploaded, suffix=os.path.splitext(uploaded.name)[1] or ".mp4")
    st.caption("Uploaded Video")
    colL, colM, colR = st.columns([1, 3, 1])
    with colM:
        with st.expander("Play Video"):
            st.video(video_path)

    if st.button("Run Video Inspection", type="primary", use_container_width=True):
        with st.spinner("Processing frames..."):
            pipeline = get_pipeline()
            result = pipeline.inspect_video(
                video_path=video_path,
                appliance_override=None,
                output_dir="output",
            )

        if result.get("error"):
            st.error(result["error"])
            return

        st.success("Video analysis complete!")
        if result.get("annotated_video_path"):
            st.caption("Annotated Video")
            colL, colM, colR = st.columns([1, 3, 1])
            with colM:
                with st.expander("Play Video"):
                    st.video(result["annotated_video_path"])
        if result.get("summary"):
            summary = result["summary"]
            st.markdown("### Summary Report")
            _render_key_metrics(summary)
        st.download_button(
            label="Download JSON Summary",
            data=json.dumps(result, indent=2),
            file_name="video_summary.json",
            mime="application/json",
        )


def _render_history_tab() -> None:
    st.header("Claim History")
    stats = get_claim_stats()
    if stats["total_claims"] > 0:
        cols = st.columns(4)
        cols[0].metric("Total Claims", stats["total_claims"])
        cols[1].metric("High Risk", stats["high_risk_claims"])
        cols[2].metric("Avg Fraud", f'{stats["avg_fraud_score"]:.1f}')
        cols[3].metric("Avg Condition", f'{stats["avg_condition_score"]:.1f}')
    else:
        st.info("No claims recorded yet.")

    claims = get_claims()
    if claims:
        df = pd.DataFrame(claims)
        cols = [c for c in ["claim_id", "timestamp", "appliance", "severity", "fraud_score", "claim_risk", "decision"]
                if c in df.columns]
        if cols:
            st.dataframe(df[cols], use_container_width=True, hide_index=True)

        if "claim_id" in df.columns:
            chosen = st.selectbox("View details", [""] + list(df["claim_id"]))
            if chosen:
                claim = get_claim_by_id(chosen)
                if claim and claim.get("full_report"):
                    report = json.loads(claim["full_report"])
                    _render_key_metrics(report)
                    _render_explanations(report)
                    if st.button("Download PDF", key=f"dl_pdf_{chosen}"):
                        try:
                            pdf_path = generate_pdf_report(report, output_dir="reports")
                            st.session_state.history_pdf_path = pdf_path
                            st.session_state.history_pdf_name = os.path.basename(pdf_path)
                        except Exception as e:
                            st.error(f"PDF generation failed: {e}")
                    if st.session_state.get("history_pdf_path") and os.path.exists(st.session_state.history_pdf_path):
                        with open(st.session_state.history_pdf_path, "rb") as f:
                            st.download_button("Save PDF", data=f, file_name=st.session_state.history_pdf_name, mime="application/pdf")


def _render_analytics_tab() -> None:
    st.header("Analytics Dashboard")
    from services.claim_service import get_claims
    claims = get_claims(limit=1000)
    if not claims:
        st.info("Insufficient data for analytics. Run inspections first.")
        return

    df = pd.DataFrame(claims)
    st.subheader("Key Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Inspections", len(df))
    if "fraud_score" in df.columns:
        c2.metric("Avg Fraud Score", f"{df['fraud_score'].mean():.1f}")
    if "claim_score" in df.columns:
        c3.metric("Avg Claim Score", f"{df['claim_score'].mean():.1f}")
    if "condition_score" in df.columns:
        c4.metric("Avg Condition", f"{df['condition_score'].mean():.1f}")
    if "repair_impact" in df.columns:
        c5.metric("Avg Repair Impact", df["repair_impact"].mode().iloc[0] if not df["repair_impact"].mode().empty else "N/A")

    st.markdown("---")
    dist_col1, dist_col2, dist_col3 = st.columns(3)

    with dist_col1:
        if "appliance" in df.columns:
            st.subheader("Appliances by Type")
            app_counts = df["appliance"].value_counts()
            st.bar_chart(app_counts)
            st.caption(f"Top: {app_counts.index[0] if len(app_counts) > 0 else 'N/A'} ({app_counts.iloc[0] if len(app_counts) > 0 else 0})")

    with dist_col2:
        if "severity" in df.columns:
            st.subheader("Severity Distribution")
            sev = df["severity"].value_counts()
            st.bar_chart(sev)
            acceptable = sum(sev.get(s, 0) for s in ["Minor", "None"])
            total = sev.sum()
            pct = acceptable / total * 100 if total > 0 else 0
            st.caption(f"Minor+None: {pct:.0f}% of inspections")

    with dist_col3:
        if "claim_risk" in df.columns:
            st.subheader("Claim Risk Distribution")
            risk = df["claim_risk"].value_counts()
            st.bar_chart(risk)
            auto_approve = risk.get("low", 0)
            review = risk.get("medium", 0) + risk.get("high", 0)
            st.caption(f"Auto-approve: {auto_approve} | Review needed: {review}")

    st.markdown("---")
    trend_col1, trend_col2 = st.columns(2)

    with trend_col1:
        if "timestamp" in df.columns and "fraud_score" in df.columns:
            st.subheader("Fraud Score Over Time")
            trend = df[["timestamp", "fraud_score"]].copy()
            trend["timestamp"] = pd.to_datetime(trend["timestamp"], errors="coerce")
            trend = trend.dropna(subset=["timestamp"]).sort_values("timestamp")
            if len(trend) > 1:
                st.line_chart(trend.set_index("timestamp")["fraud_score"])
            else:
                st.info("Need more data points for trend visualization.")

    with trend_col2:
        if "timestamp" in df.columns and "condition_score" in df.columns:
            st.subheader("Condition Score Over Time")
            trend = df[["timestamp", "condition_score"]].copy()
            trend["timestamp"] = pd.to_datetime(trend["timestamp"], errors="coerce")
            trend = trend.dropna(subset=["timestamp"]).sort_values("timestamp")
            if len(trend) > 1:
                st.line_chart(trend.set_index("timestamp")["condition_score"])
            else:
                st.info("Need more data points for trend visualization.")

    st.markdown("---")
    if "decision" in df.columns:
        st.subheader("Decision Breakdown")
        dec = df["decision"].value_counts()
        cols = st.columns(len(dec))
        for i, (decision, count) in enumerate(dec.items()):
            pct = count / len(df) * 100
            badge = "badge-green" if decision == "APPROVE" else ("badge-yellow" if decision == "MANUAL_REVIEW" else "badge-red")
            cols[i].metric(decision, f"{count} ({pct:.0f}%)")
            st.markdown(f'<span class="badge {badge}">{decision}</span>', unsafe_allow_html=True)

    if "damage_detected" in df.columns:
        true_count = df["damage_detected"].sum() if df["damage_detected"].dtype == bool else (df["damage_detected"] == "True").sum()
        false_count = len(df) - true_count
        st.subheader("Damage Detection Rate")
        det_col1, det_col2 = st.columns(2)
        det_col1.metric("Damage Found", true_count)
        det_col2.metric("No Damage", false_count)


def multi_image_tab() -> None:
    st.header("Multi-Image Inspection (2-6 images)")
    st.caption("Upload multiple photos from different angles for a unified assessment.")

    uploaded_files = st.file_uploader(
        "Upload 2-6 images (front, rear, left, right, top, close-up)",
        type=["jpg", "jpeg", "png", "webp", "heic", "heif"],
        accept_multiple_files=True,
    )

    if not uploaded_files or len(uploaded_files) < 1:
        st.info("Upload at least 1 image to begin.")
        return

    if len(uploaded_files) > 6:
        st.warning("Maximum 6 images supported. First 6 will be used.")
        uploaded_files = uploaded_files[:6]

    col = st.columns(min(len(uploaded_files), 3))
    paths = []
    for i, f in enumerate(uploaded_files):
        path = _save_upload(f, suffix=os.path.splitext(f.name)[1] or ".jpg")
        paths.append(path)
        with col[i % 3]:
            from utils import read_image
            thumb = read_image(path)
            if thumb is not None:
                thumb_rgb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                _constrained_image(thumb_rgb, caption=f"{i+1}. {f.name}")
            else:
                st.image(path, caption=f"{i+1}. {f.name}", width=200)

    if st.button("Run Multi-Image Inspection", type="primary", use_container_width=True):
        from services.multi_image_service import MultiImageInspector
        with st.spinner("Analyzing all images..."):
            inspector = MultiImageInspector()
            report = inspector.inspect(paths)

        st.success(f"Analysis complete! {report.image_count} images analyzed.")
        st.markdown("---")

        decision = report.decision
        cls = {"APPROVE": "approve", "REJECT": "reject"}.get(decision, "review")
        icon = {"APPROVE": "\u2705", "REJECT": "\u274c"}.get(decision, "\u26a0\ufe0f")
        st.markdown(f'<div class="decision-banner {cls}">{icon} {decision} &mdash; Risk: {report.claim_risk.upper()}</div>', unsafe_allow_html=True)

        cols = st.columns(5)
        cols[0].metric("Appliance", report.appliance)
        cols[1].metric("Condition", f"{report.condition_score}/100", report.grade)
        cols[2].metric("Severity", report.severity)
        cols[3].metric("Repair Impact", report.repair_impact)
        cols[4].metric("Fraud Risk", f"{report.fraud_score}/100")

        if report.annotated_image_path and os.path.exists(report.annotated_image_path):
            from utils import read_image
            annotated_img = read_image(report.annotated_image_path)
            if annotated_img is not None:
                _constrained_image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Annotated Best Image")
            else:
                st.warning("Annotated image unavailable.")
        else:
            st.info("Annotated image not available for this inspection.")

        st.markdown("---")
        tab1, tab2, tab3, tab4 = st.tabs(["Merged Damages", "Per-Image Quality", "Explanations", "Repair Breakdown"])

        with tab1:
            if report.merged_damage_detections:
                rows = []
                for d in report.merged_damage_detections:
                    rows.append({
                        "Type": d.get("class_name", "?").replace("_", " ").title(),
                        "Confidence": f'{d.get("confidence", 0):.0%}',
                        "Location": d.get("location", "?").replace("_", " ").title(),
                        "Source Images": len(d.get("source_images", [])),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.success("No damage detected across all images.")

        with tab2:
            for i, q in enumerate(report.per_image_quality):
                pct = q.get("score", 0)
                passed = q.get("passed", False)
                issues = q.get("issues", [])
                icon = "\u2705" if passed else "\u274c"
                st.markdown(f"**Image {i+1}:** Quality={pct}/100 {icon}")
                if issues:
                    for issue in issues:
                        st.caption(f"  - {issue}")

        with tab3:
            for section, label in [("appliance", "Appliance Classification"), ("damage", "Damage Assessment"),
                                    ("fraud", "Fraud Analysis"), ("repair", "Repair Estimate"), ("claim", "Claim Recommendation")]:
                text = report.explanations.get(section, "")
                if text:
                    st.markdown(
                        f'<div class="xai-section">'
                        f'<div class="xai-label">{label}</div>'
                        f'<div class="explanation-box">{text}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        with tab4:
            if report.repair_breakdown:
                rows = []
                for item in report.repair_breakdown:
                    rows.append({
                        "Damage": item.get("damage_type", "?").replace("_", " ").title(),
                        "Severity": item.get("severity", "Minor"),
                        "Repair Impact": item.get("repair_impact", "Low"),
                        "Repairability": item.get("repairability", "Repairable"),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            else:
                st.info("No repair assessment available.")

        if st.button("Save Multi-Image Report", use_container_width=True):
            from services.claim_service import save_claim
            claim_id = save_claim(report.to_dict())
            st.success(f"Claim {claim_id} saved!")


def _render_monitor_tab() -> None:
    st.header("System Monitoring")
    from services.monitoring import monitor as mon

    perf = mon.get_performance_summary()
    if "error" in perf:
        st.info("No monitoring data yet. Run some inspections first.")
        return

    st.subheader("Performance Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Calls", perf.get("total_calls", 0))
    c2.metric("Total Errors", perf.get("total_errors", 0))
    error_rate = (perf.get("total_errors", 0) / max(perf.get("total_calls", 1), 1)) * 100
    c3.metric("Error Rate", f"{error_rate:.1f}%")
    c4.metric("Healthy", "\u2705" if error_rate < 5 else "\u26a0\ufe0f")

    modules = perf.get("modules", [])
    if modules:
        st.subheader("Module Timing")
        rows = []
        for m in modules:
            rows.append({
                "Module": m["module"],
                "Operation": m["operation"],
                "Calls": m["calls"],
                "Avg Duration (ms)": f'{m["avg_dur"]:.1f}',
                "Errors": m["errors"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    session = mon.get_session_stats()
    if session:
        st.subheader("Session Stats")
        rows = []
        for key, stats in session.items():
            rows.append({
                "Module": key,
                "Calls": stats.get("total_calls", 0),
                "Avg (ms)": stats.get("avg_duration_ms", 0),
                "Max (ms)": stats.get("max_duration_ms", 0),
                "Success Rate": f'{stats.get("success_rate", 0):.1f}%',
                "Avg Conf": stats.get("avg_confidence", "N/A"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    errors = mon.get_recent_errors(limit=10)
    if errors:
        st.subheader("Recent Errors")
        for e in errors:
            st.error(f"{e.get('module')}.{e.get('operation')}: {e.get('error', 'unknown')}")


def main() -> None:
    inject_dashboard_theme()

    with st.sidebar:
        st.markdown("### \u2699\ufe0f AI Inspection")
        st.caption("v3.0 \u00b7 Insurance Platform")
        st.markdown("---")
        nav = st.radio("Navigation", ["Image", "Video", "Multi-Image", "History", "Analytics", "Monitor"], label_visibility="collapsed")
        st.markdown("---")
        st.markdown("""<div class="sidebar-info">
<b>\u00a9 IRDAI Compliant</b><br>
AI advisory tool under IRDAI Regulations, 2021.<br>
Does not replace licensed surveyor assessment.
</div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.caption(f"Session: {datetime.now().strftime('%Y-%m-%d')}")
    st.caption("Models: Phone, Laptop, Fridge, TV")
    st.caption("Detection: YOLO11s + Seg + CV Heuristics")
    st.caption("Explainable AI enabled | IRDAI Compliant")
    st.markdown("---")
    st.caption("AI Appliance Inspection v3.0")

    {
        "Image": image_tab,
        "Video": video_tab,
        "Multi-Image": multi_image_tab,
        "History": _render_history_tab,
        "Analytics": _render_analytics_tab,
        "Monitor": _render_monitor_tab,
    }[nav]()


if __name__ == "__main__":
    main()
