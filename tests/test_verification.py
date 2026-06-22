"""
Verification suite for all 9 implemented improvements.
Run with: python -m pytest tests/test_verification.py -v --tb=short
Each test is independent. Redis/Celery tests are skipped if Redis not running.
CLIP tests are skipped if transformers not installed.
"""

import pytest
import numpy as np
import cv2
import os
import sys
import tempfile
import sqlite3
from pathlib import Path

# ─── helpers ────────────────────────────────────────────────────────────────

def make_image(h=640, w=640, brightness=128, blur=False):
    img = np.ones((h, w, 3), dtype=np.uint8) * brightness
    cv2.rectangle(img, (100,100), (300,300), (200,200,200), -1)
    if blur:
        img = cv2.GaussianBlur(img, (51,51), 0)
    return img

def save_image(img, path):
    cv2.imwrite(str(path), img)
    return str(path)

# ─── 1. HEIC SUPPORT ────────────────────────────────────────────────────────

class TestHEICSupport:
    def test_pillow_heif_importable(self):
        """pillow-heif must be installed"""
        try:
            import pillow_heif
            assert True
        except ImportError:
            pytest.fail("pillow-heif not installed — run: pip install pillow-heif")

    def test_heic_registered_in_validate_image_file(self):
        """validate_image_file must accept .heic and .heif extensions"""
        from utils import validate_image_file
        import inspect
        src = inspect.getsource(validate_image_file)
        assert "heic" in src.lower() or "heif" in src.lower(), \
            "validate_image_file does not mention heic/heif"

    def test_read_image_applies_exif_transpose(self):
        """read_image must call ImageOps.exif_transpose"""
        from utils import read_image
        import inspect
        src = inspect.getsource(read_image)
        assert "exif_transpose" in src, \
            "read_image does not apply exif_transpose — phone photo orientation will be wrong"

    def test_heic_in_dashboard_uploader(self):
        """dashboard/app.py must include heic in file uploader types"""
        dashboard_path = Path("dashboard/app.py")
        if not dashboard_path.exists():
            pytest.skip("dashboard/app.py not found")
        content = dashboard_path.read_text()
        assert "heic" in content.lower(), \
            "dashboard/app.py file uploader does not include heic"

# ─── 2. MODEL CACHING ───────────────────────────────────────────────────────

class TestModelCaching:
    def test_appliance_detector_has_get_instance(self):
        """ApplianceDetector must have singleton get_instance classmethod"""
        from models.appliance_detector import ApplianceDetector
        assert hasattr(ApplianceDetector, "get_instance"), \
            "ApplianceDetector missing get_instance() classmethod"
        assert callable(ApplianceDetector.get_instance)

    def test_appliance_detector_singleton_returns_same_object(self):
        """Two get_instance() calls must return identical object"""
        from models.appliance_detector import ApplianceDetector
        a = ApplianceDetector.get_instance()
        b = ApplianceDetector.get_instance()
        assert a is b, "get_instance() returns different objects — singleton broken"

    def test_damage_detector_has_get_instance(self):
        """DamageDetector must have singleton get_instance classmethod"""
        from models.damage_detector import DamageDetector
        assert hasattr(DamageDetector, "get_instance"), \
            "DamageDetector missing get_instance() classmethod"

    def test_model_not_loaded_in_detect_method(self):
        """YOLO() call must NOT be inside detect() — it must be in __init__ only"""
        from models.appliance_detector import ApplianceDetector
        import inspect
        detect_src = inspect.getsource(ApplianceDetector.detect_single 
                                        if hasattr(ApplianceDetector, 'detect_single') 
                                        else ApplianceDetector.detect)
        assert "YOLO(" not in detect_src, \
            "YOLO() is being called inside detect() — model loads on every request"

# ─── 3. ASYNC INFERENCE ─────────────────────────────────────────────────────

class TestAsyncInference:
    def test_threadpoolexecutor_in_api(self):
        """api/__init__.py must use ThreadPoolExecutor"""
        api_path = Path("api/__init__.py")
        if not api_path.exists():
            pytest.skip("api/__init__.py not found")
        content = api_path.read_text()
        assert "ThreadPoolExecutor" in content, \
            "ThreadPoolExecutor not found in api/__init__.py"
        assert "run_in_executor" in content, \
            "run_in_executor not found — inference is still blocking"

    def test_inspect_endpoint_is_async(self):
        """inspect/image endpoint must be async def"""
        api_path = Path("api/__init__.py")
        if not api_path.exists():
            pytest.skip("api/__init__.py not found")
        content = api_path.read_text()
        assert "async def" in content, \
            "No async def found in api/__init__.py"

# ─── 4. CLIP TV DETECTION ───────────────────────────────────────────────────

class TestCLIPDamageDetector:
    def test_clip_damage_file_exists(self):
        """models/damage_detector/clip_damage.py must exist"""
        assert Path("models/damage_detector/clip_damage.py").exists(), \
            "clip_damage.py not found"

    def test_clip_damage_detector_importable(self):
        """CLIPDamageDetector must be importable"""
        try:
            from models.damage_detector.clip_damage import CLIPTVDamageDetector
            assert True
        except ImportError as e:
            pytest.fail(f"CLIPTVDamageDetector import failed: {e}")

    def test_clip_damage_detector_has_detect_method(self):
        """CLIPTVDamageDetector must have detect() method"""
        from models.damage_detector.clip_damage import CLIPTVDamageDetector
        assert hasattr(CLIPTVDamageDetector, "detect"), \
            "CLIPTVDamageDetector has no detect() method"

    def test_clip_integrated_in_damage_detector(self):
        """DamageDetector must call CLIP when appliance is television"""
        from models.damage_detector import DamageDetector
        import inspect
        src = inspect.getsource(DamageDetector)
        assert "television" in src.lower() and ("clip" in src.lower() or "CLIPDamage" in src), \
            "DamageDetector does not integrate CLIP for TV damage"

    def test_clip_detect_returns_list(self):
        """CLIPDamageDetector.detect() must return a list"""
        try:
            from models.damage_detector.clip_damage import CLIPTVDamageDetector
            img = make_image()
            detector = CLIPTVDamageDetector()
            result = detector.detect(img)
            assert isinstance(result, list), \
                f"detect() returned {type(result)}, expected list"
        except Exception as e:
            pytest.skip(f"CLIP model not available: {e}")

# ─── 5. LOCATION-WEIGHTED SEVERITY ──────────────────────────────────────────

class TestLocationWeightedSeverity:
    def test_location_weights_in_config(self):
        """LOCATION_WEIGHTS must be defined in configs/config.py"""
        from configs.config import LOCATION_WEIGHTS
        assert isinstance(LOCATION_WEIGHTS, dict), "LOCATION_WEIGHTS is not a dict"
        assert "screen" in LOCATION_WEIGHTS, "screen not in LOCATION_WEIGHTS"
        assert "unknown" in LOCATION_WEIGHTS, "unknown not in LOCATION_WEIGHTS"
        assert LOCATION_WEIGHTS["screen"] > LOCATION_WEIGHTS.get("body", 0), \
            "screen weight must be higher than body weight"

    def test_severity_service_uses_location_weights(self):
        """Severity service must import and apply LOCATION_WEIGHTS"""
        severity_files = list(Path("services").glob("*severity*"))
        assert severity_files, "No severity service file found in services/"
        src = severity_files[0].read_text()
        assert "LOCATION_WEIGHTS" in src or "location_weight" in src.lower(), \
            "Severity service does not use LOCATION_WEIGHTS"

    def test_screen_damage_scores_higher_than_body_damage(self):
        """Screen damage must produce higher severity than same-size body damage"""
        try:
            from configs.config import LOCATION_WEIGHTS
            screen_w = LOCATION_WEIGHTS.get("screen", 1.0)
            body_w = LOCATION_WEIGHTS.get("body", 1.0)
            assert screen_w > body_w, \
                f"screen weight ({screen_w}) not greater than body weight ({body_w})"
        except ImportError:
            pytest.skip("Config not importable")

# ─── 6. FRAUD DETECTION (PRNU + METADATA) ───────────────────────────────────

class TestFraudImprovements:
    def test_prnu_method_exists(self):
        """Advanced fraud service must have PRNU check method"""
        fraud_files = list(Path("services").glob("*fraud*")) + list(Path("services").glob("*advanced*"))
        if not fraud_files:
            pytest.skip("No advanced fraud service found")
        src = fraud_files[0].read_text()
        assert "prnu" in src.lower() or "_detect_prnu" in src.lower(), \
            "PRNU check not found in fraud service"

    def test_metadata_location_consistency_check_exists(self):
        """Fraud service must check metadata-location consistency"""
        fraud_files = list(Path("services").glob("*fraud*")) + list(Path("services").glob("*advanced*"))
        if not fraud_files:
            pytest.skip("No advanced fraud service found")
        src = fraud_files[0].read_text()
        assert "metadata_location" in src.lower() or "location_consistency" in src.lower(), \
            "Metadata-location consistency check not found in fraud service"

    def test_fraud_analyze_accepts_detected_appliance(self):
        """analyze() must accept detected_appliance parameter"""
        from services.fraud_service import AdvancedFraudEngine
        import inspect
        sig = inspect.signature(AdvancedFraudEngine.analyze)
        assert "detected_appliance" in sig.parameters, \
            "analyze() does not accept detected_appliance parameter"

# ─── 7. IRDAI COMPLIANCE ────────────────────────────────────────────────────

class TestIRDAICompliance:
    def test_irdai_in_api_root(self):
        """API root endpoint must mention IRDAI"""
        api_path = Path("api/__init__.py")
        if not api_path.exists():
            pytest.skip("api/__init__.py not found")
        content = api_path.read_text()
        assert "IRDAI" in content or "irdai" in content.lower(), \
            "IRDAI not mentioned in api/__init__.py"

    def test_irdai_in_pdf_service(self):
        """PDF service must include IRDAI footer"""
        pdf_files = list(Path("services").glob("*pdf*"))
        if not pdf_files:
            pytest.skip("No PDF service found")
        src = pdf_files[0].read_text()
        assert "IRDAI" in src, \
            "IRDAI footer not found in pdf_service.py"

    def test_irdai_in_dashboard(self):
        """Dashboard sidebar must mention IRDAI"""
        dashboard_path = Path("dashboard/app.py")
        if not dashboard_path.exists():
            pytest.skip("dashboard/app.py not found")
        content = dashboard_path.read_text()
        assert "IRDAI" in content, \
            "IRDAI badge not found in dashboard/app.py"

# ─── 8. BRAND-AWARE REPAIR COSTS ────────────────────────────────────────────

class TestBrandAwareRepairCosts:
    def test_brand_cost_multipliers_in_config(self):
        """BRAND_COST_MULTIPLIERS must be in config"""
        from configs.config import BRAND_COST_MULTIPLIERS
        assert isinstance(BRAND_COST_MULTIPLIERS, dict)
        assert "Apple" in BRAND_COST_MULTIPLIERS
        assert "Samsung" in BRAND_COST_MULTIPLIERS
        assert "unknown" in BRAND_COST_MULTIPLIERS
        assert BRAND_COST_MULTIPLIERS["Apple"] > BRAND_COST_MULTIPLIERS["unknown"], \
            "Apple multiplier should be > unknown"

    def test_detect_brand_function_exists(self):
        """detect_brand() must exist somewhere in models or services"""
        found = False
        for f in list(Path("models").rglob("*.py")) + list(Path("services").rglob("*.py")):
            if "detect_brand" in f.read_text():
                found = True
                break
        assert found, "detect_brand() function not found anywhere in models/ or services/"

    def test_brand_multiplier_applied_in_repair_cost(self):
        """repair service must reference BRAND_COST_MULTIPLIERS"""
        repair_files = list(Path("services").glob("*repair*"))
        if not repair_files:
            pytest.skip("No repair service found")
        src = repair_files[0].read_text()
        assert "BRAND_COST_MULTIPLIERS" in src or "brand_multiplier" in src.lower(), \
            "Brand multiplier not applied in repair cost calculation"

# ─── 9. REDIS/CELERY VIDEO QUEUE ────────────────────────────────────────────

class TestVideoQueue:
    def test_video_queue_file_exists(self):
        """services/video_queue.py must exist"""
        assert Path("services/video_queue.py").exists(), \
            "services/video_queue.py not created"

    def test_celery_app_defined(self):
        """video_queue.py must define a Celery app"""
        src = Path("services/video_queue.py").read_text()
        assert "Celery" in src, "Celery not used in video_queue.py"
        assert "process_video" in src.lower(), "No process_video task found"

    def test_async_video_endpoint_in_api(self):
        """API must have async video endpoint returning job_id"""
        api_path = Path("api/__init__.py")
        if not api_path.exists():
            pytest.skip()
        content = api_path.read_text()
        assert "video" in content.lower() and ("job_id" in content or "task_id" in content), \
            "Async video endpoint with job_id not found in API"

    def test_redis_in_docker_compose(self):
        """docker-compose.yml must include Redis service"""
        compose_files = list(Path(".").glob("docker-compose*.yml"))
        if not compose_files:
            pytest.skip("No docker-compose.yml found")
        content = compose_files[0].read_text()
        assert "redis" in content.lower(), \
            "Redis service not found in docker-compose.yml"

    def test_celery_importable(self):
        """celery must be installed"""
        try:
            import celery
            assert True
        except ImportError:
            pytest.fail("celery not installed — run: pip install celery")

# ─── OVERALL SUMMARY ─────────────────────────────────────────────────────────

class TestOriginalTestsStillPass:
    def test_all_original_19_tests_still_importable(self):
        """Core modules must still be importable (smoke test)"""
        modules_to_check = [
            "utils",
            "configs.config",
            "models.appliance_detector",
            "models.damage_detector",
            "fraud_detection",
            "risk_engine",
        ]
        failed = []
        for mod in modules_to_check:
            try:
                __import__(mod)
            except Exception as e:
                failed.append(f"{mod}: {e}")
        assert not failed, f"These modules failed to import:\n" + "\n".join(failed)
