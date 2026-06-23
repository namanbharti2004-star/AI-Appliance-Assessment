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

from configs.config import MODEL_PATHS, MVP_APPLIANCE_CLASSES, SEVERITY_RANGES
from fraud_detection import FraudAnalysisResult
from missing_part_detector import MissingPartDetector
from models.appliance_detector import ApplianceDetector
from models.damage_detector import get_damage_detector
from risk_engine import RiskEngine
from services.claim_recommendation import assess_claim, build_justification
from services.explain_service import build_full_explanation
from services.fraud_service import AdvancedFraudEngine
from services.repair_service import estimate_total_repair_cost, assess_repair_impact, assess_repairability, assess_recommended_action
from services.severity_service import (
    assess_all_damages,
    compute_condition_score,
    compute_grade,
    get_overall_severity,
)
from utils import confidence_label

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
    repair_impact: str = "None"
    repairability: str = "No Repair Needed"
    recommended_action: str = "No Action Required"
    fraud_score: float = 0.0
    decision: str = "APPROVE"

    fraud_risk_level: str = "Low"
    fraud_reasons: List[str] = field(default_factory=list)
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

    @staticmethod
    def _normalize_prediction(p: Any) -> Dict[str, Any]:
        if isinstance(p, dict):
            return {"class_name": p.get("class_name", "?"), "confidence": p.get("confidence", 0)}
        if isinstance(p, (list, tuple)):
            logger.debug("Normalizing list/tuple prediction: type={}", type(p).__name__)
            return {
                "class_name": str(p[0]) if len(p) > 0 else "?",
                "confidence": float(p[1]) if len(p) > 1 else 0,
            }
        logger.debug("Normalizing scalar prediction: type={} value={}", type(p).__name__, p)
        return {"class_name": str(p), "confidence": 0}

    @staticmethod
    def _detect_screenshot_indicators(image: np.ndarray) -> bool:
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            top_border = gray[0:3, :].mean() > 240 if h > 3 else False
            bottom_border = gray[-3:, :].mean() > 240 if h > 3 else False
            if top_border or bottom_border:
                return True
            edges = cv2.Canny(gray, 50, 150)
            edge_density = edges.sum() / (h * w * 255)
            if edge_density < 0.01:
                return True
        except Exception:
            pass
        return False

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
        repair_result = estimate_total_repair_cost(damage_assessments, report.damage_detections,
                                                    brand=brand)
        report.repair_impact = repair_result.get("repair_impact", "Medium")
        report.repairability = repair_result.get("repairability", "Repairable")
        report.recommended_action = repair_result.get("recommended_action", "Professional Assessment Recommended")
        report.repair_breakdown = repair_result.get("breakdown", [])

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
            top_preds=[(n["class_name"], n["confidence"])
                       for n in [self._normalize_prediction(p)
                                 for p in report.all_predictions]] if report.all_predictions else None,
            damage_detections=report.damage_detections,
            severity=report.severity,
            condition_score=condition_score,
            grade=report.grade,
            fraud_score=int(report.fraud_score),
            fraud_risk=report.fraud_risk_level,
            fraud_reasons=report.fraud_reasons,
            ela_score=report.ela_score,
            claim_risk=report.claim_risk,
            claim_score=report.claim_score,
            decision=report.decision,
            repair_breakdown=report.repair_breakdown,
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
            report.metadata = {"error": "Invalid image input (None or non-array)."}
            return report

        screenshot_indicators = self._detect_screenshot_indicators(image)
        if screenshot_indicators:
            report.source_type = "screenshot"

        if appliance_override and appliance_override in MVP_APPLIANCE_CLASSES:
            logger.info("Using appliance override: {}", appliance_override)
            appliance_detection = {
                "class_name": appliance_override,
                "confidence": 1.0,
                "bbox": [0.0, 0.0, float(image.shape[1]), float(image.shape[0])],
                "roi": image,
            }
            report.all_predictions = [{"class_name": appliance_override, "confidence": 1.0}]
            roi = image
        else:
            if appliance_override:
                logger.warning("Ignoring invalid appliance_override '{}' (not in {})",
                               appliance_override, MVP_APPLIANCE_CLASSES)
            raw_dets, _ = self.appliance_detector.detect_all(image)
            logger.info("STEP 1 - detect_all(conf=default) returned {} detections: {}",
                        len(raw_dets),
                        [(d["class_name"], round(d["confidence"], 3)) for d in raw_dets])

            if not raw_dets:
                low_dets, _ = self.appliance_detector.detect_all(image, conf=0.1)
                raw_dets = low_dets
                logger.info("STEP 2 - detect_all(conf=0.1) returned {} detections: {}",
                            len(raw_dets),
                            [(d["class_name"], round(d["confidence"], 3)) for d in raw_dets])

            appliance_detection = self.appliance_detector.detect_single(image)
            logger.info("STEP 3 - detect_single returned: {}",
                        {"class_name": appliance_detection["class_name"],
                         "confidence": round(appliance_detection["confidence"], 3)}
                        if appliance_detection else None)

            all_preds = raw_dets

            if all_preds:
                report.all_predictions = [self._normalize_prediction(p) for p in all_preds[:5]]

            if appliance_detection is None and all_preds:
                best = all_preds[0]
                appliance_detection = best
                logger.info("Using sub-threshold detection: {} at {:.3f}",
                            best["class_name"], best["confidence"])

            if appliance_detection is not None:
                report.appliance = appliance_detection["class_name"]
                report.appliance_confidence = float(appliance_detection["confidence"])
                report.appliance_bbox = appliance_detection.get("bbox")
                roi = appliance_detection.get("roi", image)
                logger.info("FINAL appliance='{}' FINAL confidence={:.3f}",
                            report.appliance, report.appliance_confidence)
            else:
                fallback_objects = self.appliance_detector.detect_objects(image)
                report.appliance = "unknown"
                report.appliance_confidence = 0.0
                report.appliance_bbox = None
                roi = image
                if fallback_objects:
                    report.all_predictions = [self._normalize_prediction(o) for o in fallback_objects[:5]]
                    if not report.metadata:
                        report.metadata = {}
                    report.metadata["detected_objects"] = [
                        {"class_name": o["class_name"], "confidence": round(o["confidence"], 3), "bbox": o["bbox"]}
                        for o in fallback_objects[:10]
                    ]
                    logger.info("Fallback objects detected: {} objects", len(fallback_objects))
                else:
                    logger.info("No appliance or objects detected; continuing with appliance='unknown'")
                logger.info("FINAL appliance='{}' FINAL confidence={}",
                            report.appliance, report.appliance_confidence)

        self._detected_brand = None
        if hasattr(self.appliance_detector, 'detect_brand'):
            try:
                self._detected_brand = self.appliance_detector.detect_brand(image, image_path)
            except Exception:
                self._detected_brand = None

        resolved_damage_model = self._resolve_damage_model_path(report.appliance, damage_model_path)
        damage_detections: List[Dict[str, Any]] = []

        if self.segmentation_service and self.segmentation_service.is_enabled:
            try:
                seg_detections = self.segmentation_service.detect(image=image, roi=roi)
                if seg_detections:
                    damage_detections = seg_detections
                    report.segmentation_used = True
                    logger.info("Using segmentation damage detection ({} detections)", len(seg_detections))
            except Exception:
                logger.warning("Segmentation detection failed, falling back to bbox detection")

        if not damage_detections:
            try:
                damage_detector = get_damage_detector(report.appliance, model_path=resolved_damage_model)
                damage_detections = damage_detector.detect(image=image, roi=roi)
                if damage_detections:
                    logger.info("Using bbox damage detection ({} detections)", len(damage_detections))
            except Exception as exc:
                logger.error("Damage detection failed: {}", exc)

        damage_detections = [d for d in damage_detections if d.get("confidence", 0) >= 0.25]

        report.damage_detections = damage_detections
        report.damage_detected = bool(damage_detections)
        primary_damage = max(damage_detections, key=lambda d: d.get("confidence", 0)) if damage_detections else None
        if primary_damage:
            report.damage_type = primary_damage["class_name"]
            report.damage_confidence = float(primary_damage["confidence"])
            report.damage_bbox = primary_damage.get("bbox", [0, 0, 0, 0])

        try:
            missing_result = self.missing_part_detector.inspect(report.appliance, roi)
            report.missing_part_detected = missing_result.missing_part_detected
            report.missing_part = missing_result.missing_part
            report.missing_part_confidence = missing_result.confidence
            report.missing_part_warnings = missing_result.warnings
        except Exception:
            pass

        try:
            if self.fraud_engine is not None:
                fraud_result: FraudAnalysisResult = self.fraud_engine.analyze(image=image, image_path=image_path)
                report.ela_score = fraud_result.ela_score
                report.metadata_risk_score = fraud_result.metadata_risk_score
                report.fraud_metadata = fraud_result.metadata or {}
        except Exception:
            pass

        try:
            advanced_fraud = self.advanced_fraud.analyze(image, image_path,
                                                         detected_appliance=report.appliance)
            report.fraud_score = advanced_fraud.fraud_score
            report.fraud_risk_level = advanced_fraud.risk_level
            report.fraud_reasons = advanced_fraud.reasons
        except Exception:
            pass

        report = self._compute_enriched_fields(report, image)

        report.metadata = {
            "image_shape": list(image.shape),
            "damage_detection_count": len(damage_detections),
            "supported_mvp": True,
            "segmentation_used": report.segmentation_used,
            "model_version": "3.0.0",
            "appliance_confidence_label": confidence_label(report.appliance_confidence) if report.appliance_confidence > 0 else "Unknown",
        }
        if report.damage_detections:
            avg_conf = sum(d.get("confidence", 0) for d in report.damage_detections) / len(report.damage_detections)
            report.metadata["damage_confidence_label"] = confidence_label(avg_conf)
        logger.info(
            "REPORT FINAL: appliance='{}' conf={} damage={} damage_type='{}' all_predictions={}",
            report.appliance, report.appliance_confidence,
            report.damage_detected, report.damage_type,
            [(p.get("class_name", "?"), round(p.get("confidence", 0), 3)) for p in report.all_predictions],
        )
        return report


def format_report_for_api(report: InspectionReport) -> Dict[str, Any]:
    return {
        "report_id": report.report_id,
        "timestamp": report.timestamp,
        "source_type": report.source_type,
        "appliance": report.appliance,
        "appliance_confidence": round(report.appliance_confidence, 3),
        "appliance_confidence_label": confidence_label(report.appliance_confidence) if report.appliance_confidence > 0 else "Unknown",
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
        "repair_impact": report.repair_impact,
        "repairability": report.repairability,
        "recommended_action": report.recommended_action,
        "fraud_score": report.fraud_score,
        "fraud_risk_level": report.fraud_risk_level,
        "claim_score": report.claim_score,
        "claim_risk": report.claim_risk,
        "repair_impact": report.repair_impact,
        "repairability": report.repairability,
        "recommended_action": report.recommended_action,
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
            "repair_impact": report.repair_impact,
            "repairability": report.repairability,
            "recommended_action": report.recommended_action,
            "fraud_score": report.fraud_score,
            "claim_score": report.claim_score,
            "claim_risk": report.claim_risk,
            "decision": report.decision,
            "claim_justification": report.claim_justification,
        },
        "explanations": report.explanations,
        "raw_report": report.to_dict(),
    }
