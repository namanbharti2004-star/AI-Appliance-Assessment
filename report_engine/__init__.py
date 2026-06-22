"""
Structured report generation for the appliance inspection platform.

Integrates:
- Appliance detector (top-3 predictions)
- Damage detector (NMS, shadow filter, location inference)
- Severity service (type × area × defect count)
- Repair cost service (severity-based multipliers)
- Claim recommendation engine (APPROVE/MANUAL_REVIEW/REJECT)
- Explainable AI service (natural language explanations)
- Fraud detection (ELA + advanced 7-factor)
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from configs.config import MODEL_PATHS, SEVERITY_RANGES
from fraud_detection import FraudAnalysisResult
from missing_part_detector import MissingPartDetector
from models.appliance_detector import ApplianceDetector
from models.damage_detector import get_damage_detector
from risk_engine import RiskEngine
from services.claim_recommendation import assess_claim, build_justification
from services.explain_service import build_full_explanation
from services.fraud_service import AdvancedFraudEngine
from services.repair_service import estimate_total_repair_cost
from services.severity_service import (
    assess_all_damages,
    compute_condition_score,
    compute_grade,
    get_overall_severity,
)

try:
    from services.damage_segmentation_service import SegmentationService
except ImportError:
    SegmentationService = None


@dataclass
class InspectionReport:
    report_id: str
    timestamp: str
    source_type: str = "image"
    image_path: Optional[str] = None

    appliance: str = ""
    appliance_confidence: float = 0.0
    appliance_bbox: Optional[List[float]] = None
    all_predictions: List[Dict[str, Any]] = field(default_factory=list)

    damage_detected: bool = False
    damage_type: str = ""
    damage_confidence: float = 0.0
    damage_bbox: Optional[List[float]] = None
    damage_detections: List[Dict[str, Any]] = field(default_factory=list)

    missing_part_detected: bool = False
    missing_part: Optional[str] = None
    missing_part_confidence: float = 0.0
    missing_part_warnings: List[str] = field(default_factory=list)

    ela_score: float = 0.0
    metadata_risk_score: float = 0.0
    fraud_metadata: Dict[str, Any] = field(default_factory=dict)

    damage_percentage: int = 0
    severity: str = "None"
    condition_score: int = 100
    grade: str = "A"
    repair_cost: int = 0
    fraud_score: float = 0.0
    decision: str = "APPROVE"

    fraud_risk_level: str = "Low"
    fraud_reasons: List[str] = field(default_factory=list)
    repair_cost_min: int = 0
    repair_cost_max: int = 0
    repair_cost_display: str = ""
    repair_breakdown: List[Dict] = field(default_factory=list)
    claim_score: int = 0
    claim_risk: str = "Low"
    claim_justification: str = ""
    segmentation_used: bool = False
    explanations: Dict[str, str] = field(default_factory=dict)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, output_path: str) -> bool:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(self.to_json())
        return True


class ReportGenerator:
    def __init__(
        self,
        appliance_detector: Optional[ApplianceDetector] = None,
        risk_engine: Optional[RiskEngine] = None,
        missing_part_detector: Optional[MissingPartDetector] = None,
        use_segmentation: bool = False,
    ):
        self.appliance_detector = appliance_detector or ApplianceDetector()
        self.risk_engine = risk_engine or RiskEngine()
        self.fraud_engine = getattr(self.risk_engine, "fraud_engine", None)
        self.missing_part_detector = missing_part_detector or MissingPartDetector()
        self.advanced_fraud = AdvancedFraudEngine()
        self.segmentation_service: Optional[SegmentationService] = None
        if use_segmentation and SegmentationService is not None:
            self.segmentation_service = SegmentationService()
            self.segmentation_service.initialize()

    def _resolve_damage_model_path(self, appliance: str, override_path: Optional[str]) -> Optional[str]:
        if override_path:
            return override_path
        configured = MODEL_PATHS.get("damage_detector", {}).get(appliance)
        if configured and os.path.exists(configured):
            return configured
        return None

    def _compute_enriched_fields(self, report: InspectionReport, image: np.ndarray) -> InspectionReport:
        """Compute severity, repair cost, claim recommendation, and explanations."""
        h, w = image.shape[:2]
        appliance_bbox = report.appliance_bbox or [0.0, 0.0, float(w), float(h)]

        damage_assessments = assess_all_damages(
            report.damage_detections,
            appliance_bbox=appliance_bbox,
            image_shape=(h, w),
        )

        severity_label = get_overall_severity(damage_assessments)
        if severity_label != "None":
            report.severity = severity_label

        condition_score = compute_condition_score(damage_assessments)
        report.condition_score = condition_score
        report.grade = compute_grade(condition_score)

        brand = getattr(self, "_detected_brand", None)
        cost_result = estimate_total_repair_cost(damage_assessments, report.damage_detections,
                                                  brand=brand)
        report.repair_cost = cost_result.get("total_min", 0)
        report.repair_cost_min = cost_result.get("total_min", 0)
        report.repair_cost_max = cost_result.get("total_max", 0)
        report.repair_cost_display = cost_result.get("total_display", "")
        report.repair_breakdown = cost_result.get("breakdown", [])

        claim_result = assess_claim(
            severity=report.severity,
            fraud_score=report.fraud_score,
            condition_score=condition_score,
            damage_count=len(report.damage_detections),
        )
        report.claim_score = claim_result["claim_score"]
        report.claim_risk = claim_result["claim_risk"]
        report.decision = claim_result["decision"]
        report.claim_justification = build_justification(
            claim_result,
            report.severity,
            int(report.fraud_score),
            report.fraud_reasons,
            report.grade,
        )

        report.explanations = build_full_explanation(
            appliance=report.appliance,
            appliance_conf=report.appliance_confidence,
            top_preds=[(p.get("class_name", ""), p.get("confidence", 0))
                       for p in report.all_predictions] if report.all_predictions else None,
            damage_detections=report.damage_detections,
            severity=report.severity,
            condition_score=condition_score,
            grade=report.grade,
            fraud_score=int(report.fraud_score),
            fraud_risk=report.fraud_risk_level,
            fraud_reasons=report.fraud_reasons,
            ela_score=report.ela_score,
            cost_display=report.repair_cost_display,
            cost_min=report.repair_cost_min,
            cost_max=report.repair_cost_max,
            cost_breakdown=report.repair_breakdown,
            claim_risk=report.claim_risk,
            claim_score=report.claim_score,
            decision=report.decision,
        )

        return report

    def generate_report(
        self,
        image: np.ndarray,
        image_path: Optional[str] = None,
        appliance_override: Optional[str] = None,
        damage_model_path: Optional[str] = None,
        source_type: str = "image",
    ) -> InspectionReport:
        report = InspectionReport(
            report_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().isoformat(),
            image_path=image_path,
            source_type=source_type,
        )

        if image is None or not isinstance(image, np.ndarray):
            report.decision = "MANUAL_REVIEW"
            report.metadata = {"error": "Invalid image input (None or non-array)."}
            return report

        if appliance_override:
            appliance_detection = {
                "class_name": appliance_override,
                "confidence": 1.0,
                "bbox": [0.0, 0.0, float(image.shape[1]), float(image.shape[0])],
                "roi": image,
            }
            report.all_predictions = [{"class_name": appliance_override, "confidence": 1.0}]
        else:
            appliance_detection = self.appliance_detector.detect_single(image)
            all_preds = self.appliance_detector.detect_all(image)
            if all_preds:
                report.all_predictions = [
                    {"class_name": p.get("class_name", "?"), "confidence": p.get("confidence", 0)}
                    for p in all_preds[:5]
                ]

        if appliance_detection is None:
            report.decision = "MANUAL_REVIEW"
            report.metadata = {"error": "No appliance detected."}
            return report

        report.appliance = appliance_detection["class_name"]
        report.appliance_confidence = float(appliance_detection["confidence"])
        report.appliance_bbox = appliance_detection.get("bbox")
        roi = appliance_detection.get("roi", image)

        self._detected_brand = self.appliance_detector.detect_brand(image, image_path) if hasattr(self.appliance_detector, 'detect_brand') else None

        resolved_damage_model = self._resolve_damage_model_path(report.appliance, damage_model_path)
        damage_detections: List[Dict[str, Any]] = []

        if self.segmentation_service and self.segmentation_service.is_enabled:
            seg_detections = self.segmentation_service.detect(image=image, roi=roi)
            if seg_detections:
                damage_detections = seg_detections
                report.segmentation_used = True
                logger.info("Using segmentation damage detection ({} detections)", len(seg_detections))

        if not damage_detections:
            damage_detector = get_damage_detector(report.appliance, model_path=resolved_damage_model)
            damage_detections = damage_detector.detect(image=image, roi=roi)
            if damage_detections:
                logger.info("Using bbox damage detection ({} detections)", len(damage_detections))

        MIN_CONFIDENCE = 0.4
        damage_detections = [d for d in damage_detections if d.get("confidence", 0) >= MIN_CONFIDENCE]

        report.damage_detections = damage_detections
        report.damage_detected = bool(damage_detections)
        primary_damage = max(damage_detections, key=lambda d: d.get("confidence", 0)) if damage_detections else None
        if primary_damage:
            report.damage_type = primary_damage["class_name"]
            report.damage_confidence = float(primary_damage["confidence"])
            report.damage_bbox = primary_damage.get("bbox", [0, 0, 0, 0])

        missing_result = self.missing_part_detector.inspect(report.appliance, roi)
        report.missing_part_detected = missing_result.missing_part_detected
        report.missing_part = missing_result.missing_part
        report.missing_part_confidence = missing_result.confidence
        report.missing_part_warnings = missing_result.warnings

        if self.fraud_engine is not None:
            fraud_result: FraudAnalysisResult = self.fraud_engine.analyze(image=image, image_path=image_path)
            report.ela_score = fraud_result.ela_score
            report.metadata_risk_score = fraud_result.metadata_risk_score
            report.fraud_metadata = fraud_result.metadata or {}
        else:
            report.ela_score = 0.0
            report.metadata_risk_score = 0.0
            report.fraud_metadata = {}

        advanced_fraud = self.advanced_fraud.analyze(image, image_path,
                                                     detected_appliance=report.appliance)
        report.fraud_score = advanced_fraud.fraud_score
        report.fraud_risk_level = advanced_fraud.risk_level
        report.fraud_reasons = advanced_fraud.reasons

        report = self._compute_enriched_fields(report, image)

        report.metadata = {
            "image_shape": list(image.shape),
            "damage_detection_count": len(damage_detections),
            "supported_mvp": True,
            "segmentation_used": report.segmentation_used,
            "model_version": "3.0.0",
        }
        return report


def format_report_for_api(report: InspectionReport) -> Dict[str, Any]:
    return {
        "report_id": report.report_id,
        "timestamp": report.timestamp,
        "source_type": report.source_type,
        "appliance": report.appliance,
        "appliance_confidence": round(report.appliance_confidence, 3),
        "damage_detected": report.damage_detected,
        "damage_type": report.damage_type,
        "damage_confidence": round(report.damage_confidence, 3),
        "missing_part_detected": report.missing_part_detected,
        "missing_part": report.missing_part,
        "ela_score": round(report.ela_score, 3),
        "metadata_risk_score": round(report.metadata_risk_score, 3),
        "damage_percentage": report.damage_percentage,
        "severity": report.severity,
        "condition_score": report.condition_score,
        "grade": report.grade,
        "repair_cost": report.repair_cost,
        "fraud_score": report.fraud_score,
        "fraud_risk_level": report.fraud_risk_level,
        "claim_score": report.claim_score,
        "claim_risk": report.claim_risk,
        "repair_cost_display": report.repair_cost_display,
        "decision": report.decision,
    }


def format_report_for_dashboard(report: InspectionReport) -> Dict[str, Any]:
    return {
        "report_id": report.report_id,
        "timestamp": report.timestamp,
        "appliance": {"type": report.appliance, "confidence": report.appliance_confidence, "bbox": report.appliance_bbox},
        "damage": {
            "detected": report.damage_detected,
            "type": report.damage_type,
            "confidence": report.damage_confidence,
            "bbox": report.damage_bbox,
            "all_detections": report.damage_detections,
        },
        "missing_part": {
            "detected": report.missing_part_detected,
            "part": report.missing_part,
            "confidence": report.missing_part_confidence,
            "warnings": report.missing_part_warnings,
        },
        "fraud": {
            "ela_score": report.ela_score,
            "metadata_risk_score": report.metadata_risk_score,
            "metadata": report.fraud_metadata,
            "fraud_score": report.fraud_score,
            "fraud_risk_level": report.fraud_risk_level,
            "fraud_reasons": report.fraud_reasons,
        },
        "assessment": {
            "damage_percentage": report.damage_percentage,
            "severity": report.severity,
            "condition_score": report.condition_score,
            "grade": report.grade,
            "repair_cost": report.repair_cost,
            "repair_cost_min": report.repair_cost_min,
            "repair_cost_max": report.repair_cost_max,
            "repair_cost_display": report.repair_cost_display,
            "repair_breakdown": report.repair_breakdown,
            "fraud_score": report.fraud_score,
            "claim_score": report.claim_score,
            "claim_risk": report.claim_risk,
            "decision": report.decision,
            "claim_justification": report.claim_justification,
        },
        "explanations": report.explanations,
        "raw_report": report.to_dict(),
    }
