"""
MVP risk, condition, severity, and repair estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from configs.config import (
    CONDITION_GRADE_RANGES,
    DECISION_THRESHOLDS,
    REPAIR_COST_RULES,
    RISK_WEIGHTS,
    SEVERITY_RANGES,
)
from fraud_detection import FraudDetectionEngine, FraudAnalysisResult


@dataclass
class RiskAssessment:
    condition_score: int
    grade: str
    damage_percentage: int
    severity: str
    repair_cost: int
    fraud_score: float
    decision: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "condition_score": self.condition_score,
            "grade": self.grade,
            "damage_percentage": self.damage_percentage,
            "severity": self.severity,
            "repair_cost": self.repair_cost,
            "fraud_score": self.fraud_score,
            "decision": self.decision,
            "details": self.details,
        }


class DamageSeverityEstimator:
    def estimate(self, confidence: float, bbox_area_ratio: float, missing_part_detected: bool) -> Dict[str, Any]:
        damage_percentage = int(
            max(0.0, min(100.0, (confidence * 40.0) + (bbox_area_ratio * 50.0) + (8.0 if missing_part_detected else 0.0)))
        )
        severity = "LOW"
        for label, (start, end) in SEVERITY_RANGES.items():
            if start <= damage_percentage <= end:
                severity = label
                break
        return {"damage_percentage": damage_percentage, "severity": severity}


class RepairCostEstimator:
    def estimate(self, damage_type: Optional[str], damage_percentage: int, missing_part_detected: bool) -> int:
        total = 0
        if damage_type and damage_type in REPAIR_COST_RULES:
            rule = REPAIR_COST_RULES[damage_type]
            total += int(rule["base"] + (damage_percentage * rule["multiplier"] * 10))
        if missing_part_detected:
            rule = REPAIR_COST_RULES["missing_part"]
            total += int(rule["base"] + (damage_percentage * rule["multiplier"] * 5))
        return total


class RiskEngine:
    def __init__(self, custom_weights: Optional[Dict[str, float]] = None):
        self.weights = custom_weights or RISK_WEIGHTS
        self.fraud_engine = FraudDetectionEngine()
        self.severity_engine = DamageSeverityEstimator()
        self.repair_engine = RepairCostEstimator()

    def _grade_from_condition(self, condition_score: int) -> str:
        for grade, (start, end) in CONDITION_GRADE_RANGES.items():
            if start <= condition_score <= end:
                return grade
        return "D"

    def calculate_fraud_score(
        self,
        appliance_confidence: float,
        damage_confidence: float,
        ela_score: float,
        metadata_risk_score: float,
        missing_part_detected: bool,
    ) -> float:
        weighted_sum = (
            appliance_confidence * self.weights["appliance_confidence"]
            + damage_confidence * self.weights["damage_confidence"]
            + ela_score * self.weights["ela_score"]
            + metadata_risk_score * self.weights["metadata_risk_score"]
            + (1.0 if missing_part_detected else 0.0) * self.weights["missing_part_score"]
        )
        return round(min(max(weighted_sum, 0.0), 1.0), 3)

    def calculate_condition_score(
        self, damage_percentage: int, missing_part_detected: bool, fraud_score: float
    ) -> int:
        penalty = damage_percentage + (12 if missing_part_detected else 0) + int(fraud_score * 18)
        return max(0, min(100, 100 - penalty))

    def make_decision(self, fraud_score: float, condition_score: int) -> str:
        if fraud_score >= DECISION_THRESHOLDS["REJECT"]["fraud_score_min"]:
            return "REJECT"
        if (
            fraud_score <= DECISION_THRESHOLDS["APPROVE"]["fraud_score_max"]
            and condition_score >= DECISION_THRESHOLDS["APPROVE"]["condition_score_min"]
        ):
            return "APPROVE"
        return "MANUAL_REVIEW"

    def assess(
        self,
        appliance_confidence: float,
        damage_confidence: float,
        damage_bbox_area_ratio: float,
        fraud_result: FraudAnalysisResult,
        missing_part_detected: bool,
        damage_type: Optional[str],
    ) -> RiskAssessment:
        severity_data = self.severity_engine.estimate(
            confidence=damage_confidence,
            bbox_area_ratio=damage_bbox_area_ratio,
            missing_part_detected=missing_part_detected,
        )
        fraud_score = self.calculate_fraud_score(
            appliance_confidence=appliance_confidence,
            damage_confidence=damage_confidence,
            ela_score=fraud_result.ela_score,
            metadata_risk_score=fraud_result.metadata_risk_score,
            missing_part_detected=missing_part_detected,
        )
        condition_score = self.calculate_condition_score(
            damage_percentage=severity_data["damage_percentage"],
            missing_part_detected=missing_part_detected,
            fraud_score=fraud_score,
        )
        grade = self._grade_from_condition(condition_score)
        repair_cost = self.repair_engine.estimate(
            damage_type=damage_type,
            damage_percentage=severity_data["damage_percentage"],
            missing_part_detected=missing_part_detected,
        )
        decision = self.make_decision(fraud_score, condition_score)

        return RiskAssessment(
            condition_score=condition_score,
            grade=grade,
            damage_percentage=severity_data["damage_percentage"],
            severity=severity_data["severity"],
            repair_cost=repair_cost,
            fraud_score=fraud_score,
            decision=decision,
            details={
                "weights_used": self.weights,
                "ela_score": fraud_result.ela_score,
                "metadata_risk_score": fraud_result.metadata_risk_score,
                "missing_part_detected": missing_part_detected,
                "damage_type": damage_type,
            },
        )
