"""
Unit Tests for AI Appliance Assessment Platform

Tests core functionality of the various modules.
"""

import os
import sys
import pytest
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import (
    calculate_iou,
    non_max_suppression,
    resize_image,
    normalize_image,
    denormalize_image,
    get_appliance_from_class_id,
    get_class_id_from_appliance,
    validate_image_file,
)
from configs.config import MVP_APPLIANCE_CLASSES, APPLIANCE_CLASSES


class TestUtils:
    """Tests for utility functions"""

    def test_calculate_iou(self):
        """Test IoU calculation"""
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        iou = calculate_iou(box1, box2)
        assert 0 <= iou <= 1

    def test_calculate_iou_no_overlap(self):
        """Test IoU with no overlap"""
        box1 = [0, 0, 10, 10]
        box2 = [20, 20, 30, 30]
        iou = calculate_iou(box1, box2)
        assert iou == 0

    def test_calculate_iou_identical(self):
        """Test IoU with identical boxes"""
        box = [0, 0, 10, 10]
        iou = calculate_iou(box, box)
        assert iou == 1.0

    def test_non_max_suppression(self):
        """Test NMS function"""
        boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]]
        scores = [0.9, 0.8, 0.7]
        keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
        assert len(keep) <= len(boxes)

    def test_resize_image(self):
        """Test image resizing"""
        image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        resized = resize_image(image, target_size=50)
        assert max(resized.shape[:2]) == 50

    def test_normalize_denormalize(self):
        """Test image normalization"""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        normalized = normalize_image(image)
        assert normalized.max() <= 1.0
        denormalized = denormalize_image(normalized)
        assert np.allclose(image, denormalized)

    def test_get_appliance_from_class_id(self):
        """Test class ID to appliance mapping"""
        appliance = get_appliance_from_class_id(0, MVP_APPLIANCE_CLASSES)
        assert appliance == "phone"

    def test_get_class_id_from_appliance(self):
        """Test appliance to class ID mapping"""
        class_id = get_class_id_from_appliance("phone", MVP_APPLIANCE_CLASSES)
        assert class_id == 0


class TestFraudDetection:
    """Tests for fraud detection modules"""

    def test_ela_detector_import(self):
        """Test ELA detector can be imported"""
        from fraud_detection import ELADetector
        detector = ELADetector()
        assert detector.name == "Error Level Analysis"

    def test_metadata_analyzer_import(self):
        """Test metadata analyzer can be imported"""
        from fraud_detection import MetadataAnalyzer
        analyzer = MetadataAnalyzer()
        assert analyzer.name == "Metadata Analyzer"

    def test_fraud_engine_import(self):
        """Test fraud engine can be imported"""
        from fraud_detection import FraudDetectionEngine
        engine = FraudDetectionEngine()
        assert engine.ela_detector is not None
        assert engine.metadata_analyzer is not None


class TestRiskEngine:
    """Tests for risk engine"""

    def test_risk_engine_import(self):
        """Test risk engine can be imported"""
        from risk_engine import RiskEngine
        engine = RiskEngine()
        assert engine is not None

    def test_damage_severity_estimator(self):
        """Test damage severity estimation"""
        from risk_engine import DamageSeverityEstimator
        estimator = DamageSeverityEstimator()
        result = estimator.estimate(confidence=0.9, bbox_area_ratio=0.1, missing_part_detected=False)
        assert "damage_percentage" in result
        assert "severity" in result
        assert result["severity"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def test_repair_cost_estimator(self):
        """Test repair impact estimation"""
        from risk_engine import RepairCostEstimator
        estimator = RepairCostEstimator()
        impact = estimator.estimate(damage_type="crack", damage_percentage=30, missing_part_detected=False)
        assert isinstance(impact, str)
        assert len(impact) > 0


class TestReportEngine:
    """Tests for report engine"""

    def test_inspection_report_creation(self):
        """Test report creation"""
        from report_engine import InspectionReport
        report = InspectionReport(
            report_id="test123",
            timestamp="2024-01-01T00:00:00",
            appliance="phone",
            appliance_confidence=0.95,
            damage_detected=True,
            damage_type="crack",
            damage_confidence=0.88,
            condition_score=65,
            fraud_score=0.25,
            decision="APPROVE",
        )
        assert report.appliance == "phone"
        assert report.damage_detected is True
        assert report.condition_score == 65

    def test_report_to_dict(self):
        """Test report serialization"""
        from report_engine import InspectionReport
        report = InspectionReport(
            report_id="test456",
            timestamp="2024-01-01T00:00:00",
            appliance="laptop",
            appliance_confidence=0.99,
        )
        data = report.to_dict()
        assert data["appliance"] == "laptop"
        assert data["condition_score"] == 100


class TestModels:
    """Tests for model classes"""

    def test_appliance_detector_import(self):
        """Test appliance detector can be imported"""
        from models.appliance_detector import ApplianceDetector
        detector = ApplianceDetector()
        assert detector.classes == ["phone", "television", "laptop"]

    def test_damage_detector_import(self):
        """Test damage detector can be imported"""
        from models.damage_detector import DamageDetector, get_damage_detector
        detector = get_damage_detector("phone")
        assert detector.appliance_name == "phone"
        assert "crack" in detector.DAMAGE_CLASSES


class TestMissingPartDetector:
    """Tests for missing part detector"""

    def test_missing_part_detector_import(self):
        """Test missing part detector can be imported"""
        from missing_part_detector import MissingPartDetector
        detector = MissingPartDetector()
        assert "phone" in detector.supported_classes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
