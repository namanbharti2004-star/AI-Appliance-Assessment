"""
Damage detector with false-positive reduction.

Improvements:
- Non-max suppression (NMS) to remove duplicate detections
- Area-based validation (damage too small or too large = noise)
- Reflection/shadow detection (brightness gradient check)
- Confidence filtering with configurable threshold
- Damage location inference (relative position within appliance)
- Returns empty list when no valid damage, enabling healthy condition score
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from configs.config import MODEL_CONFIG, MVP_DAMAGE_CLASSES, get_device

CLIP_AVAILABLE = False
try:
    from models.damage_detector.clip_damage import CLIPTVDamageDetector
    CLIP_AVAILABLE = True
except ImportError:
    CLIPTVDamageDetector = None

CLASS_NAME_MAP = {
    "cracked": "crack",
    "screen_crack": "crack",
    "body_dent": "dent",
    "lines": "display_lines",
}

DAMAGE_LOCATIONS = {
    "phone": ["screen", "back_panel", "camera", "edges"],
    "television": ["screen", "bezel", "stand", "back_panel"],
    "laptop": ["screen", "keyboard", "hinge", "trackpad", "body"],
    "refrigerator": ["upper_door", "lower_door", "side_panel", "top", "handle"],
}

MIN_DAMAGE_CONFIDENCE = 0.35
MIN_DAMAGE_AREA_RATIO = 0.001
MAX_DAMAGE_AREA_RATIO = 0.6
NMS_IOU_THRESHOLD = 0.5


def _iou(box1: List[float], box2: List[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _nms(detections: List[Dict[str, Any]], iou_thresh: float = NMS_IOU_THRESHOLD) -> List[Dict[str, Any]]:
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if _iou(best["bbox"], d["bbox"]) < iou_thresh]
    return keep


def _is_reflection_or_shadow(image: np.ndarray, bbox: List[float]) -> bool:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(image.shape[1], x2); y2 = min(image.shape[0], y2)
    if x2 - x1 < 5 or y2 - y1 < 5:
        return False
    region = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    if mean_brightness > 220:
        return True
    if np.std(gray) < 8 and mean_brightness > 180:
        return True
    return False


def _infer_location(appliance: str, bbox: List[float], image_shape: Tuple[int, int]) -> str:
    h, w = image_shape[:2]
    cx = (bbox[0] + bbox[2]) / 2 / w
    cy = (bbox[1] + bbox[3]) / 2 / h

    locs = DAMAGE_LOCATIONS.get(appliance.lower(), ["front", "back", "side"])
    if appliance.lower() == "refrigerator":
        if cy < 0.5:
            return "upper_door"
        else:
            return "lower_door"
    elif appliance.lower() in ("television", "laptop", "phone", "monitor"):
        if cy < 0.3:
            return "top_" + locs[0] if locs else "top"
        elif cy > 0.7:
            return "bottom_" + locs[0] if locs else "bottom"
        else:
            return locs[0] if locs else "center"
    return "surface"


def _filter_by_area(detections: List[Dict[str, Any]], image_area: float) -> List[Dict[str, Any]]:
    filtered = []
    for d in detections:
        bbox = d.get("bbox", [0, 0, 0, 0])
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        ratio = area / max(image_area, 1)
        if ratio < MIN_DAMAGE_AREA_RATIO or ratio > MAX_DAMAGE_AREA_RATIO:
            continue
        filtered.append(d)
    return filtered


class DamageDetector:
    DAMAGE_CLASSES = MVP_DAMAGE_CLASSES
    _instances: Dict[str, "DamageDetector"] = {}

    @classmethod
    def get_instance(cls, appliance_name: str, **kwargs) -> "DamageDetector":
        if appliance_name not in cls._instances:
            cls._instances[appliance_name] = cls(appliance_name=appliance_name, **kwargs)
        return cls._instances[appliance_name]

    def __init__(
        self,
        appliance_name: str,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        device: Optional[str] = None,
    ):
        config = MODEL_CONFIG["damage_detector"]
        self.appliance_name = appliance_name
        self.confidence_threshold = confidence_threshold or config["confidence_threshold"]
        self.iou_threshold = iou_threshold or config["iou_threshold"]
        self.device = device or get_device()
        self.model = None
        self._model_path = None
        self.clip_detector = None

        if YOLO and model_path and os.path.exists(model_path):
            self._model_path = model_path
            if os.environ.get("LAZY_LOAD_MODELS", "true").lower() != "true":
                self.load_model(model_path)
        elif appliance_name.lower() == "television" and CLIP_AVAILABLE and CLIPTVDamageDetector is not None:
            try:
                self.clip_detector = CLIPTVDamageDetector(
                    confidence_threshold=config.get("global_confidence_filter", 0.4),
                )
                logger.info("CLIP fallback loaded for TV damage detection")
            except Exception as exc:
                logger.warning("CLIP TV detector init failed: {}", exc)

    def load_model(self, model_path: str) -> bool:
        if YOLO is None:
            return False
        try:
            self.model = YOLO(model_path)
            return True
        except Exception as exc:
            logger.error("Failed to load damage model: {}", exc)
            self.model = None
            return False

    def _ensure_model_loaded(self) -> bool:
        if self.model is not None:
            return True
        if self._model_path and os.path.exists(self._model_path):
            return self.load_model(self._model_path)
        return False

    def _detect_with_yolo(self, image: np.ndarray) -> List[Dict[str, Any]]:
        self._ensure_model_loaded()
        if self.model is None or image is None or not isinstance(image, np.ndarray):
            return []
        detections: List[Dict[str, Any]] = []
        try:
            results = self.model(image, conf=self.confidence_threshold,
                                 iou=self.iou_threshold, device=self.device, verbose=False)
            for result in results:
                if result.boxes is None:
                    continue
                names = result.names
                for box in result.boxes:
                    cls_id = int(box.cls.item())
                    if not 0 <= cls_id < max(len(self.DAMAGE_CLASSES), len(names)):
                        continue
                    raw = names.get(cls_id, self.DAMAGE_CLASSES[min(cls_id, len(self.DAMAGE_CLASSES)-1)])
                    normalized = CLASS_NAME_MAP.get(raw, raw)
                    if normalized not in self.DAMAGE_CLASSES:
                        continue
                    bbox = [float(v) for v in box.xyxy[0].tolist()]
                    confidence = float(box.conf.item())
                    detections.append({
                        "class_name": normalized,
                        "class_id": self.DAMAGE_CLASSES.index(normalized),
                        "confidence": confidence,
                        "bbox": bbox,
                        "source": "yolo",
                    })
        except Exception as exc:
            logger.error("YOLO damage detection failed: {}", exc)
        return detections

    def _detect_display_lines(self, image: np.ndarray) -> List[Dict[str, Any]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 120, minLineLength=80, maxLineGap=5)
        if lines is None or len(lines) < 6:
            return []
        xs, ys = [], []
        for line in lines[:30]:
            x1, y1, x2, y2 = line[0]
            if abs(x1 - x2) < 6 or abs(y1 - y2) < 6:
                xs.extend([x1, x2]); ys.extend([y1, y2])
        if len(xs) < 12:
            return []
        bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]
        box_area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
        img_area = image.shape[0] * image.shape[1]
        if box_area / max(img_area, 1) < MIN_DAMAGE_AREA_RATIO:
            return []
        return [{"class_name": "display_lines", "class_id": 2, "confidence": 0.45, "bbox": bbox, "source": "heuristic"}]

    def _detect_crack(self, image: np.ndarray) -> List[Dict[str, Any]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        enhanced = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(enhanced, 90, 180)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        img_area = image.shape[0] * image.shape[1]
        if area < img_area * 0.002:
            return []
        x, y, w, h = cv2.boundingRect(largest)
        if max(w, h) < 40:
            return []
        bbox = [float(x), float(y), float(x + w), float(y + h)]
        if _is_reflection_or_shadow(image, bbox):
            return []
        return [{"class_name": "crack", "class_id": 0, "confidence": 0.4, "bbox": bbox, "source": "heuristic"}]

    def _detect_dent(self, image: np.ndarray) -> List[Dict[str, Any]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        diff = cv2.absdiff(gray, blur)
        _, thresh = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        img_area = image.shape[0] * image.shape[1]
        for contour in contours:
            area = cv2.contourArea(contour)
            if img_area * 0.003 <= area <= img_area * 0.2:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 20 and h > 20:
                    bbox = [float(x), float(y), float(x + w), float(y + h)]
                    if not _is_reflection_or_shadow(image, bbox):
                        candidates.append((area, bbox))
        if not candidates:
            return []
        _, bbox = max(candidates, key=lambda item: item[0])
        return [{"class_name": "dent", "class_id": 1, "confidence": 0.35, "bbox": bbox, "source": "heuristic"}]

    def detect(self, image: np.ndarray, roi: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        detect_image = roi if roi is not None else image
        img_area = detect_image.shape[0] * detect_image.shape[1]

        if self.clip_detector is not None:
            clip_dets = self.clip_detector.detect(detect_image)
            clip_dets = _filter_by_area(clip_dets, img_area)
            clip_dets = _nms(clip_dets)
            if clip_dets:
                detections = clip_dets
            else:
                detections = []
        elif self.model is not None:
            yolo_dets = self._detect_with_yolo(detect_image)
            yolo_dets = _filter_by_area(yolo_dets, img_area)
            yolo_dets = _nms(yolo_dets)
            if yolo_dets:
                detections = yolo_dets
            else:
                detections = []
        else:
            detections = []

        if not detections:
            heuristic_dets: List[Dict[str, Any]] = []
            heuristic_dets.extend(self._detect_crack(detect_image))
            heuristic_dets.extend(self._detect_dent(detect_image))
            heuristic_dets.extend(self._detect_display_lines(detect_image))
            heuristic_dets = _filter_by_area(heuristic_dets, img_area)
            detections = _nms(heuristic_dets)

        detections = [d for d in detections if d["confidence"] >= MIN_DAMAGE_CONFIDENCE]

        for d in detections:
            d["location"] = _infer_location(self.appliance_name, d["bbox"], detect_image.shape)

        if detections:
            logger.info("Damage detection: {} | types={} | locations={}",
                        len(detections),
                        [d["class_name"] for d in detections],
                        [d.get("location") for d in detections])
        else:
            logger.info("No valid damage detected. Condition will remain healthy.")

        return detections

    def is_damaged(self, detections: List[Dict[str, Any]]) -> bool:
        return bool(detections)

    def get_primary_damage(self, detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not detections:
            return None
        return max(detections, key=lambda d: d["confidence"])

    @staticmethod
    def create_dataset_yaml(train_path: str, val_path: str, test_path: str,
                            classes: List[str], output_path: str) -> str:
        yaml_content = f"""
train: {train_path}
val: {val_path}
test: {test_path}
nc: {len(classes)}
names: {classes}
"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(yaml_content.strip() + "\n")
        return output_path


def get_damage_detector(appliance: str, model_path: Optional[str] = None, **kwargs: Any) -> DamageDetector:
    return DamageDetector.get_instance(appliance_name=appliance, model_path=model_path, **kwargs)
