"""
Missing-part detector for the MVP.

This module is intentionally hybrid:
- model-first if you later train a missing-part detector
- rule-based fallback today so the MVP can still produce actionable warnings
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from configs.config import MISSING_PART_CLASSES


@dataclass
class MissingPartResult:
    appliance: str
    missing_part_detected: bool
    missing_part: Optional[str]
    confidence: float
    warnings: List[str]
    detections: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "appliance": self.appliance,
            "missing_part_detected": self.missing_part_detected,
            "missing_part": self.missing_part,
            "confidence": self.confidence,
            "warnings": self.warnings,
            "detections": self.detections,
        }


class MissingPartDetector:
    """Rule-based MVP missing-part detector."""

    def __init__(self) -> None:
        self.supported_classes = MISSING_PART_CLASSES

    def _score_bottom_presence(self, roi: np.ndarray) -> float:
        h, _, _ = roi.shape
        bottom = roi[int(h * 0.75) :, :]
        gray = cv2.cvtColor(bottom, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        return float(edges.mean() / 255.0)

    def _score_top_presence(self, roi: np.ndarray) -> float:
        h, _, _ = roi.shape
        top = roi[: max(1, int(h * 0.2)), :]
        gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,
            param1=60,
            param2=16,
            minRadius=4,
            maxRadius=25,
        )
        return 0.85 if circles is not None else 0.1

    def _score_side_presence(self, roi: np.ndarray) -> float:
        _, w, _ = roi.shape
        side = roi[:, max(0, int(w * 0.82)) :]
        gray = cv2.cvtColor(side, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        return float(edges.mean() / 255.0)

    def _phone_rules(self, roi: np.ndarray) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []
        camera_score = self._score_top_presence(roi)
        if camera_score < 0.25:
            detections.append(
                {
                    "part": "camera",
                    "confidence": 1.0 - camera_score,
                    "reason": "camera area lacks expected circular features",
                }
            )

        buttons_score = self._score_side_presence(roi)
        if buttons_score < 0.04:
            detections.append(
                {
                    "part": "buttons",
                    "confidence": 0.45 + (0.4 * (1.0 - min(buttons_score * 10, 1.0))),
                    "reason": "side profile lacks expected hardware edge detail",
                }
            )
        return detections

    def _television_rules(self, roi: np.ndarray) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []
        stand_score = self._score_bottom_presence(roi)
        if stand_score < 0.06:
            detections.append(
                {
                    "part": "stand",
                    "confidence": 1.0 - min(stand_score * 8, 1.0),
                    "reason": "bottom appliance region lacks expected stand geometry",
                }
            )

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        border = np.concatenate(
            [gray[:12, :].ravel(), gray[-12:, :].ravel(), gray[:, :12].ravel(), gray[:, -12:].ravel()]
        )
        border_variation = float(np.std(border) / 255.0)
        if border_variation < 0.03:
            detections.append(
                {
                    "part": "bezel",
                    "confidence": 0.7,
                    "reason": "screen border is visually indistinct; bezel may be missing or cropped out",
                }
            )
        return detections

    def _laptop_rules(self, roi: np.ndarray) -> List[Dict[str, Any]]:
        detections: List[Dict[str, Any]] = []
        h, _, _ = roi.shape
        keyboard_region = roi[int(h * 0.52) :, :]
        gray = cv2.cvtColor(keyboard_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 160)
        keyboard_density = float(edges.mean() / 255.0)
        if keyboard_density < 0.05:
            detections.append(
                {
                    "part": "keys",
                    "confidence": 0.78,
                    "reason": "lower half lacks expected keyboard texture density",
                }
            )

        hinge_strip = roi[max(0, int(h * 0.45)) : int(h * 0.55), :]
        hinge_gray = cv2.cvtColor(hinge_strip, cv2.COLOR_BGR2GRAY)
        hinge_edges = cv2.Canny(hinge_gray, 60, 160)
        hinge_score = float(hinge_edges.mean() / 255.0)
        if hinge_score < 0.04:
            detections.append(
                {
                    "part": "hinge_cover",
                    "confidence": 0.72,
                    "reason": "hinge strip lacks the expected edge continuity",
                }
            )
        return detections

    def inspect(self, appliance: str, roi: np.ndarray) -> MissingPartResult:
        if appliance not in self.supported_classes:
            return MissingPartResult(
                appliance=appliance,
                missing_part_detected=False,
                missing_part=None,
                confidence=0.0,
                warnings=[f"Missing-part rules are not defined yet for {appliance}."],
                detections=[],
            )

        if appliance == "phone":
            detections = self._phone_rules(roi)
        elif appliance == "television":
            detections = self._television_rules(roi)
        else:
            detections = self._laptop_rules(roi)

        if not detections:
            return MissingPartResult(
                appliance=appliance,
                missing_part_detected=False,
                missing_part=None,
                confidence=0.0,
                warnings=[],
                detections=[],
            )

        primary = max(detections, key=lambda item: item["confidence"])
        return MissingPartResult(
            appliance=appliance,
            missing_part_detected=True,
            missing_part=primary["part"],
            confidence=float(primary["confidence"]),
            warnings=[item["reason"] for item in detections],
            detections=detections,
        )
