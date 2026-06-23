"""
Multi-Image Inspection Service.

Supports uploading multiple photos of the same appliance from different angles:
- Automatically identifies view type (front, rear, left, right, top, close-up, detail)
- Runs inference on each image independently
- Merges damage detections across images (avoids double-counting)
- Selects best evidence images per damage type
- Produces a single unified inspection report
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from configs.config import MODEL_CONFIG
from models.appliance_detector import ApplianceDetector
from models.damage_detector import get_damage_detector
from services.explain_service import build_full_explanation
from services.fraud_service import AdvancedFraudEngine
from services.image_quality import check_image_quality
from services.repair_service import estimate_total_repair_cost
from services.severity_service import (
    assess_all_damages,
    compute_condition_score,
    compute_grade,
    get_overall_severity,
)
from services.claim_recommendation import assess_claim, build_justification


VIEW_LABELS = ["front", "rear", "left", "right", "top", "bottom", "close_up", "detail", "unknown"]


def classify_view(image: np.ndarray) -> str:
    h, w = image.shape[:2]
    aspect = w / max(h, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.sum() / (h * w * 255)

    if aspect > 1.8:
        return "close_up"
    if aspect < 0.5:
        return "detail"
    if edge_density < 0.01:
        return "unknown"
    gray_lower = gray[3 * h // 4:, :]
    if gray_lower.mean() > 180:
        return "top"
    return "front"


def iou(box_a: List[float], box_b: List[float]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    a2 = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def merge_detections(detection_lists: List[List[Dict[str, Any]]], iou_thresh: float = 0.3) -> List[Dict[str, Any]]:
    all_dets = []
    for img_idx, dets in enumerate(detection_lists):
        for d in dets:
            d["source_image"] = img_idx
            all_dets.append(d)

    if not all_dets:
        return []

    merged: List[Dict[str, Any]] = []
    used = set()

    for i, a in enumerate(all_dets):
        if i in used:
            continue
        group = [a]
        used.add(i)
        for j, b in enumerate(all_dets):
            if j in used:
                continue
            if a.get("class_name") == b.get("class_name"):
                if iou(a.get("bbox", [0, 0, 0, 0]), b.get("bbox", [0, 0, 0, 0])) > iou_thresh:
                    group.append(b)
                    used.add(j)

        best = max(group, key=lambda d: d.get("confidence", 0))
        avg_conf = sum(d.get("confidence", 0) for d in group) / len(group)
        avg_bbox = [
            sum(d.get("bbox", [0, 0, 0, 0])[i] for d in group) / len(group) for i in range(4)
        ]
        merged.append({
            "class_name": best["class_name"],
            "class_id": best.get("class_id", 0),
            "confidence": round(avg_conf, 3),
            "bbox": [round(v, 1) for v in avg_bbox],
            "location": best.get("location", "unknown"),
            "source": "multi_image_merged",
            "source_images": [d.get("source_image") for d in group],
            "detection_count": len(group),
        })

    return merged


@dataclass
class MultiImageReport:
    report_id: str
    timestamp: str
    image_count: int
    image_paths: List[str]
    view_labels: List[str]

    appliance: str = ""
    appliance_confidence: float = 0.0

    damage_detected: bool = False
    damage_detections: List[Dict[str, Any]] = field(default_factory=list)
    merged_damage_detections: List[Dict[str, Any]] = field(default_factory=list)

    severity: str = "None"
    condition_score: int = 100
    grade: str = "A"
    fraud_score: int = 0
    fraud_risk_level: str = "Low"
    fraud_reasons: List[str] = field(default_factory=list)

    repair_impact: str = "None"
    repairability: str = "No Repair Needed"
    recommended_action: str = "No Action Required"
    repair_breakdown: List[Dict] = field(default_factory=list)

    claim_score: int = 0
    claim_risk: str = "Low"
    decision: str = "APPROVE"
    claim_justification: str = ""
    explanations: Dict[str, str] = field(default_factory=dict)

    best_evidence_images: Dict[str, int] = field(default_factory=dict)
    per_image_quality: List[Dict[str, Any]] = field(default_factory=list)
    annotated_image_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        return dataclasses.asdict(self)


class MultiImageInspector:
    def __init__(
        self,
        appliance_detector: Optional[ApplianceDetector] = None,
        fraud_engine: Optional[AdvancedFraudEngine] = None,
    ):
        self.appliance_detector = appliance_detector or ApplianceDetector()
        self.fraud_engine = fraud_engine or AdvancedFraudEngine()

    def inspect(
        self,
        image_paths: List[str],
        appliance_override: Optional[str] = None,
    ) -> MultiImageReport:
        images = []
        quality_results = []
        view_labels = []
        valid_paths = []

        from utils import read_image

        for path in image_paths:
            img = read_image(path)
            if img is None:
                logger.warning("Could not read image: {}", path)
                continue
            quality = check_image_quality(img)
            quality_results.append(quality.to_dict())
            if not quality.passed and quality.score < 20:
                logger.warning("Image quality too low, skipping: {} (score={})", path, quality.score)
                continue
            view = classify_view(img)
            images.append(img)
            view_labels.append(view)
            valid_paths.append(path)

        if not images:
            return MultiImageReport(
                report_id=str(uuid.uuid4())[:8],
                timestamp=datetime.now().isoformat(),
                image_count=len(image_paths),
                image_paths=image_paths,
                view_labels=view_labels,
                per_image_quality=quality_results,
                decision="MANUAL_REVIEW",
            )

        appliance_name = ""
        appliance_conf = 0.0
        all_image_detections: List[List[Dict[str, Any]]] = []
        fraud_scores: List[int] = []

        for idx, (image, path) in enumerate(zip(images, valid_paths)):
            if appliance_override:
                det = {"class_name": appliance_override, "confidence": 1.0,
                       "bbox": [0.0, 0.0, float(image.shape[1]), float(image.shape[0])], "roi": image}
                appliance_name = appliance_override
                appliance_conf = 1.0
            else:
                det = self.appliance_detector.detect_single(image)
                if det and not appliance_name:
                    appliance_name = det["class_name"]
                    appliance_conf = float(det["confidence"])

            roi = det.get("roi", image) if det else image
            damage_detector = get_damage_detector(appliance_name or "phone")
            detections = damage_detector.detect(image=image, roi=roi)
            detections = [d for d in detections if d.get("confidence", 0) >= 0.4]
            all_image_detections.append(detections)

            fraud_result = self.fraud_engine.analyze(image, path)
            fraud_scores.append(fraud_result.fraud_score)

        merged_detections = merge_detections(all_image_detections)

        if merged_detections:
            best_evidence: Dict[str, int] = {}
            for d in merged_detections:
                dt = d.get("class_name", "unknown")
                source_images = d.get("source_images", [0])
                if dt not in best_evidence:
                    best_evidence[dt] = source_images[0]
        else:
            best_evidence = {}

        severity = "None"
        condition_score = 100
        grade = "A"

        if merged_detections:
            image_shape = images[0].shape[:2]
            assessments = assess_all_damages(merged_detections, image_shape=image_shape)
            severity = get_overall_severity(assessments)
            condition_score = compute_condition_score(assessments)
            grade = compute_grade(condition_score)

        avg_fraud = int(np.mean(fraud_scores)) if fraud_scores else 0
        fraud_level = "Low"
        fraud_reasons = []
        if avg_fraud > 60:
            fraud_level = "High"
            fraud_reasons.append("Elevated fraud indicators across multiple images")
        elif avg_fraud > 30:
            fraud_level = "Medium"

        repair_result = estimate_total_repair_cost(
            assess_all_damages(merged_detections, image_shape=images[0].shape[:2]) if merged_detections else [],
            merged_detections,
        ) if merged_detections else {"repair_impact": "None", "repairability": "No Repair Needed",
                                      "recommended_action": "No Action Required", "breakdown": []}

        claim_result = assess_claim(
            severity=severity,
            fraud_score=avg_fraud,
            condition_score=condition_score,
            damage_count=len(merged_detections),
        )
        claim_justification = build_justification(
            claim_result, severity, avg_fraud, fraud_reasons, grade,
        )

        explanations = build_full_explanation(
            appliance=appliance_name or "unknown",
            appliance_conf=appliance_conf,
            top_preds=None,
            damage_detections=merged_detections,
            severity=severity,
            condition_score=condition_score,
            grade=grade,
            fraud_score=avg_fraud,
            fraud_risk=fraud_level,
            fraud_reasons=fraud_reasons,
            ela_score=0.0,
            claim_risk=claim_result["claim_risk"],
            claim_score=claim_result["claim_score"],
            decision=claim_result["decision"],
            repair_breakdown=repair_result.get("breakdown", []),
        )

        report_id = str(uuid.uuid4())[:8]

        annotated_image_path = ""
        if images:
            from utils import save_image
            best_img_idx = 0
            max_dets = 0
            for i, dets in enumerate(all_image_detections):
                if len(dets) > max_dets:
                    max_dets = len(dets)
                    best_img_idx = i
            img_for_annot = images[best_img_idx].copy()
            for d in all_image_detections[best_img_idx]:
                bbox = d.get("bbox")
                if bbox and len(bbox) == 4:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    label = d.get("class_name", "damage")
                    conf = d.get("confidence", 0)
                    cv2.rectangle(img_for_annot, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    text = f"{label} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img_for_annot, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 0, 255), -1)
                    cv2.putText(img_for_annot, text, (x1 + 2, y1 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            annotated_image_path = os.path.join("output", f"annotated_{report_id}.jpg")
            os.makedirs("output", exist_ok=True)
            save_image(img_for_annot, annotated_image_path)

        return MultiImageReport(
            report_id=report_id,
            timestamp=datetime.now().isoformat(),
            image_count=len(valid_paths),
            image_paths=valid_paths,
            view_labels=view_labels,
            appliance=appliance_name or "unknown",
            appliance_confidence=appliance_conf,
            damage_detected=bool(merged_detections),
            damage_detections=[d for dets in all_image_detections for d in dets],
            merged_damage_detections=merged_detections,
            severity=severity,
            condition_score=condition_score,
            grade=grade,
            fraud_score=avg_fraud,
            fraud_risk_level=fraud_level,
            fraud_reasons=fraud_reasons,
            repair_impact=repair_result.get("repair_impact", "None"),
            repairability=repair_result.get("repairability", "No Repair Needed"),
            recommended_action=repair_result.get("recommended_action", "No Action Required"),
            repair_breakdown=repair_result.get("breakdown", []),
            claim_score=claim_result["claim_score"],
            claim_risk=claim_result["claim_risk"],
            decision=claim_result["decision"],
            claim_justification=claim_justification,
            explanations=explanations,
            best_evidence_images=best_evidence,
            per_image_quality=quality_results,
            annotated_image_path=annotated_image_path,
        )
