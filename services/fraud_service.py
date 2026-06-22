"""
Production-grade Fraud Detection Engine.

Multi-factor fraud detection (0-100):
1.  Error Level Analysis (ELA)
2.  Metadata anomalies & EXIF analysis
3.  Screenshot detection (UI patterns, color banding)
4.  AI-generated / synthetic image detection (DCT frequency, noise)
5.  Copy-move / clone detection (block matching)
6.  Duplicate image detection (persistent perceptual hash)
7.  Resolution mismatch & compression anomalies
8.  Tampering indicators
9.  Missing / inconsistent EXIF
10. Edited image detection (Photoshop, GIMP, etc.)
"""

from __future__ import annotations

import io
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

from fraud_detection import ELADetector, MetadataAnalyzer


@dataclass
class FraudResult:
    fraud_score: int
    risk_level: str
    reasons: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fraud_score": self.fraud_score,
            "risk_level": self.risk_level,
            "reasons": self.reasons,
            "details": self.details,
            "explanation": self.explanation,
        }


HASH_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "fraud_hashes.db",
)


def _ensure_hash_db() -> None:
    os.makedirs(os.path.dirname(HASH_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(HASH_DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS image_hashes (hash TEXT PRIMARY KEY, first_seen TEXT, count INTEGER DEFAULT 1)")
    conn.commit()
    conn.close()


def _store_hash(hash_int: int) -> Tuple[bool, int]:
    _ensure_hash_db()
    hash_str = str(hash_int)
    conn = sqlite3.connect(HASH_DB_PATH)
    cur = conn.execute("SELECT count FROM image_hashes WHERE hash = ?", (hash_str,))
    row = cur.fetchone()
    if row:
        count = row[0] + 1
        conn.execute("UPDATE image_hashes SET count = ? WHERE hash = ?", (count, hash_str))
        conn.commit()
        conn.close()
        return True, count
    else:
        conn.execute("INSERT INTO image_hashes (hash, first_seen, count) VALUES (?, ?, 1)",
                     (hash_str, datetime.now().isoformat()))
        conn.commit()
        conn.close()
        return False, 1


def _perceptual_hash(image: np.ndarray, size: int = 16) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_LINEAR)
    avg = small.mean()
    diff = (small > avg).astype(np.uint8)
    bits = diff.flatten().tolist()
    return sum(b << i for i, b in enumerate(bits))


def _hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def _risk_level(score: int) -> str:
    if score <= 25:
        return "Low"
    elif score <= 50:
        return "Medium"
    elif score <= 75:
        return "High"
    return "Critical"


def _detect_screenshot(gray: np.ndarray) -> Tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []
    h, w = gray.shape
    edges = cv2.Canny(gray, 50, 150)
    edge_density = edges.sum() / (h * w * 255)

    if edge_density < 0.02:
        score += 15
        reasons.append("Very low edge density — possible screenshot or synthetic image")

    top_row = gray[0, :]
    bottom_row = gray[-1, :]
    left_col = gray[:, 0]
    right_col = gray[:, -1]
    for border, name in [(top_row, "top"), (bottom_row, "bottom"), (left_col, "left"), (right_col, "right")]:
        border_std = float(np.std(border))
        if border_std < 5 and np.mean(border) > 200:
            score += 5
            reasons.append(f"White/solid border at {name} edge — possible screenshot crop")
            break

    window = 50
    regions_of_interest = [
        gray[h // 2 - window:h // 2 + window, w // 2 - window:w // 2 + window],
        gray[0:window, 0:window],
        gray[h - window:h, w - window:w],
    ]
    uniform_count = 0
    for region in regions_of_interest:
        if float(np.std(region)) < 10:
            uniform_count += 1
    if uniform_count >= 2:
        score += 10
        reasons.append("Uniform color blocks — UI/screenshot pattern")

    return min(score, 30), reasons


def _detect_ai_generated(gray: np.ndarray) -> Tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []
    h, w = gray.shape
    gray_f = np.float32(gray)
    dct = cv2.dct(gray_f)
    dct_log = np.log(np.abs(dct) + 1)
    low_freq = dct_log[:h // 8, :w // 8].mean()
    high_freq = dct_log[7 * h // 8:, 7 * w // 8:].mean()
    if high_freq > 0 and low_freq / high_freq > 50:
        score += 15
        reasons.append("Abnormal frequency distribution — possible AI-generated image")

    noise = cv2.GaussianBlur(gray, (5, 5), 0) - gray
    noise_std = float(np.std(noise))
    if noise_std < 2.0:
        score += 10
        reasons.append("Very low noise — possible synthetic/AI-generated image")
    elif noise_std > 30:
        score += 5
        reasons.append("High noise level — possible compression artifact from editing")

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 5 and edge_density(gray) < 0.01:
        score += 10
        reasons.append("Extremely smooth image — possible AI generation")

    return min(score, 30), reasons


def edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150)
    return edges.sum() / (gray.shape[0] * gray.shape[1] * 255)


def _detect_copy_move(gray: np.ndarray) -> Tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []
    h, w = gray.shape
    block = 16
    grid_h, grid_w = h // block, w // block
    if grid_h < 4 or grid_w < 4:
        return 0, []
    blocks = []
    for y in range(grid_h):
        for x in range(grid_w):
            by, bx = y * block, x * block
            patch = gray[by:by + block, bx:bx + block]
            mean_val = float(patch.mean())
            blocks.append((mean_val, y, x))
    blocks.sort()
    similar_pairs = 0
    for i in range(len(blocks) - 1):
        if abs(blocks[i][0] - blocks[i + 1][0]) < 2:
            dy = abs(blocks[i][1] - blocks[i + 1][1])
            dx = abs(blocks[i][2] - blocks[i + 1][2])
            if dy > 2 or dx > 2:
                similar_pairs += 1
    total_pairs = len(blocks)
    if total_pairs > 0 and similar_pairs / total_pairs > 0.15:
        score = 15
        reasons.append("Suspicious repeated blocks — possible copy-move manipulation")
    return min(score, 20), reasons


def _check_single_color_pixel(image: np.ndarray) -> Tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []
    unique_colors = len(np.unique(image.reshape(-1, image.shape[2]), axis=0))
    total_pixels = image.shape[0] * image.shape[1]
    ratio = unique_colors / total_pixels
    if ratio < 0.001:
        score = 15
        reasons.append("Very few unique colors — possible synthetic or posterized image")
    elif ratio < 0.01:
        score = 8
        reasons.append("Low color diversity — possible filter or edit")
    return min(score, 15), reasons


def _detect_tampering_edges(gray: np.ndarray) -> Tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []
    blurred = cv2.GaussianBlur(gray, (0, 0), 3)
    high_pass = cv2.subtract(gray, blurred)
    _, binary = cv2.threshold(high_pass, 15, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    suspicious = [c for c in contours if 50 < cv2.contourArea(c) < gray.shape[0] * gray.shape[1] * 0.3]
    if len(suspicious) > 5:
        score = 10
        reasons.append("Multiple suspicious edge artifacts — possible tampering")
    return min(score, 10), reasons


def _detect_prnu_noise(gray: np.ndarray) -> Tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []
    h, w = gray.shape
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    noise_residual = cv2.subtract(gray.astype(np.int16), denoised.astype(np.int16)).astype(np.float32)
    h2, w2 = h // 2, w // 2
    regions = [
        noise_residual[:h2, :w2],
        noise_residual[:h2, w2:],
        noise_residual[h2:, :w2],
        noise_residual[h2:, w2:],
    ]
    region_stds = [float(np.std(r)) for r in regions]
    mean_std = float(np.mean(region_stds))
    std_of_stds = float(np.std(region_stds))
    if mean_std < 1.5:
        score += 15
        reasons.append("Very low PRNU noise — possible AI-generated or synthetic image")
    if std_of_stds > mean_std * 0.5 and mean_std > 2:
        score += 10
        reasons.append("Inconsistent noise across image regions — possible compositing/tampering")
    if mean_std > 25:
        score += 5
        reasons.append("Abnormally high noise level — possible enhancement or filter")
    return min(score, 25), reasons


def _check_metadata_location_consistency(
    image_path: Optional[str], detected_appliance: Optional[str] = None
) -> Tuple[int, List[str]]:
    score = 0
    reasons: List[str] = []
    if not image_path or not os.path.exists(image_path):
        return 0, []
    try:
        img = Image.open(image_path)
        exif = img.getexif()
        if not exif:
            return 0, []
        gps_info = {}
        for tag_id, value in exif.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == "GPSInfo":
                for gps_id, gps_value in value.items():
                    gps_tag = GPSTAGS.get(gps_id, gps_id)
                    gps_info[gps_tag] = gps_value
        if gps_info:
            if detected_appliance == "television" or detected_appliance == "refrigerator":
                score += 10
                reasons.append("GPS metadata present for indoor appliance — possible stock photo")
        if gps_info.get("GPSLatitude") and gps_info.get("GPSLongitude"):
            score += 5
            reasons.append("GPS coordinates found — location can be verified against claim address")
        if len(gps_info) == 0:
            has_other_exif = any(TAGS.get(t, t) in ("Make", "Model", "DateTime", "Software") for t in exif.keys())
            if has_other_exif and not gps_info:
                score += 5
                reasons.append("EXIF present but GPS location stripped — possible privacy scrub")
    except Exception:
        pass
    return min(score, 15), reasons


class AdvancedFraudEngine:
    def __init__(self) -> None:
        self.ela = ELADetector()
        self.metadata_analyzer = MetadataAnalyzer()
        _ensure_hash_db()
        self._session_hashes: List[int] = []

    def _compute_ela_score(self, image: np.ndarray) -> int:
        raw = self.ela.analyze(image)
        return min(int(raw * 100), 100)

    def _check_metadata_anomalies(self, image_path: Optional[str]) -> Dict[str, Any]:
        result: Dict[str, Any] = {"score": 0, "reasons": []}
        if not image_path or not os.path.exists(image_path):
            result["score"] = 25
            result["reasons"].append("No metadata / file path missing")
            return result
        try:
            img = Image.open(image_path)
            exif = img.getexif()
            if not exif:
                result["score"] = 20
                result["reasons"].append("EXIF data completely missing — possibly web-downloaded or stripped")
            else:
                has_orientation = False
                has_model = False
                has_date = False
                software = None
                for tag_id, value in exif.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag in ("Make", "Model"):
                        has_model = True
                    elif tag in ("DateTime", "DateTimeOriginal", "DateTimeDigitized"):
                        has_date = True
                    elif tag == "Orientation":
                        has_orientation = True
                    elif tag == "Software":
                        software = str(value).lower()
                if not has_model and exif:
                    result["score"] += 8
                    result["reasons"].append("No camera make/model in EXIF")
                if not has_date:
                    result["score"] += 8
                    result["reasons"].append("No timestamp in EXIF metadata")
                if not has_orientation:
                    result["score"] += 4
                    result["reasons"].append("Missing orientation data")
                if software:
                    editing_software = ["photoshop", "gimp", "stable diffusion", "midjourney", "dalle",
                                        "firefly", "canva", "pixlr", "lightroom", "affinity"]
                    if any(s in software for s in editing_software):
                        result["score"] += 25
                        result["reasons"].append(f"Image software indicates editing: {software}")
                    elif any(s in software for s in ("screenshot", "capture", "snip")):
                        result["score"] += 15
                        result["reasons"].append(f"Screenshot software: {software}")
        except Exception:
            result["score"] = 15
            result["reasons"].append("Could not read metadata (corrupted or unsupported format)")
        result["score"] = min(result["score"], 40)
        return result

    def _check_resolution_mismatch(self, image: np.ndarray) -> Dict[str, Any]:
        result: Dict[str, Any] = {"score": 0, "reasons": []}
        h, w = image.shape[:2]
        if h < 150 or w < 150:
            result["score"] = 15
            result["reasons"].append(f"Suspiciously low resolution ({w}x{h})")
        if h > 8000 or w > 8000:
            result["score"] = 8
            result["reasons"].append(f"Unusually high resolution ({w}x{h})")
        aspect = w / max(h, 1)
        if aspect < 0.3 or aspect > 3.5:
            result["score"] = 8
            result["reasons"].append(f"Unusual aspect ratio ({aspect:.2f})")
        result["score"] = min(result["score"], 20)
        return result

    def _check_compression_anomalies(self, image: np.ndarray) -> Dict[str, Any]:
        result: Dict[str, Any] = {"score": 0, "reasons": []}
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < 10:
            result["score"] = 10
            result["reasons"].append("Very low detail — over-compressed or heavily processed image")
        elif laplacian_var > 5000:
            result["score"] = 5
            result["reasons"].append("Unnatural sharpness — possible sharpening filter applied")
        block_size = 8
        h, w = gray.shape
        if h > block_size and w > block_size:
            tiles = gray[: h - h % block_size, : w - w % block_size]
            tile_grid = tiles.reshape(-1, block_size, block_size)
            block_means = tile_grid.mean(axis=(1, 2))
            block_std = float(block_means.std())
            if block_std < 5:
                result["score"] += 8
                result["reasons"].append("Uniform 8x8 blocks — JPEG compression artifacts")
        result["score"] = min(result["score"], 15)
        return result

    def _check_duplicate(self, image: np.ndarray) -> Dict[str, Any]:
        result: Dict[str, Any] = {"score": 0, "reasons": []}
        hash_int = _perceptual_hash(image)

        is_duplicate, count = _store_hash(hash_int)
        if count > 1:
            result["score"] = min(25 + (count - 1) * 5, 40)
            result["reasons"].append(f"Duplicate image (seen {count}x in database)")
            return result

        for prev_hash in self._session_hashes:
            dist = _hamming_distance(hash_int, prev_hash)
            if dist <= 2:
                result["score"] = 30
                result["reasons"].append("Near-identical image in this session — possible claim reuse")
                break
            elif dist <= 8:
                result["score"] = 10
                result["reasons"].append("Similar image in this session")
                break

        self._session_hashes.append(hash_int)
        result["score"] = min(result["score"], 40)
        return result

    def analyze(self, image: np.ndarray, image_path: Optional[str] = None,
                detected_appliance: Optional[str] = None) -> FraudResult:
        reasons: List[str] = []
        total = 0
        details: Dict[str, Any] = {}

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ela_raw = self._compute_ela_score(image)
        total += int(ela_raw * 0.15)
        details["ela_score"] = ela_raw
        if ela_raw > 50:
            reasons.append("High ELA score — possible digital manipulation")

        meta_result = self._check_metadata_anomalies(image_path)
        total += meta_result["score"]
        details["metadata_score"] = meta_result["score"]
        reasons.extend(meta_result["reasons"])

        res_result = self._check_resolution_mismatch(image)
        total += res_result["score"]
        details["resolution_score"] = res_result["score"]
        reasons.extend(res_result["reasons"])

        comp_result = self._check_compression_anomalies(image)
        total += comp_result["score"]
        details["compression_score"] = comp_result["score"]
        reasons.extend(comp_result["reasons"])

        dup_result = self._check_duplicate(image)
        total += dup_result["score"]
        details["duplicate_score"] = dup_result["score"]
        reasons.extend(dup_result["reasons"])

        screenshot_score, screenshot_reasons = _detect_screenshot(gray)
        total += screenshot_score
        details["screenshot_score"] = screenshot_score
        reasons.extend(screenshot_reasons)

        ai_score, ai_reasons = _detect_ai_generated(gray)
        total += ai_score
        details["ai_generated_score"] = ai_score
        reasons.extend(ai_reasons)

        copy_move_score, copy_move_reasons = _detect_copy_move(gray)
        total += copy_move_score
        details["copy_move_score"] = copy_move_score
        reasons.extend(copy_move_reasons)

        color_score, color_reasons = _check_single_color_pixel(image)
        total += color_score
        details["color_diversity_score"] = color_score
        reasons.extend(color_reasons)

        tamper_score, tamper_reasons = _detect_tampering_edges(gray)
        total += tamper_score
        details["tampering_score"] = tamper_score
        reasons.extend(tamper_reasons)

        prnu_score, prnu_reasons = _detect_prnu_noise(gray)
        total += prnu_score
        details["prnu_score"] = prnu_score
        reasons.extend(prnu_reasons)

        loc_score, loc_reasons = _check_metadata_location_consistency(image_path, detected_appliance)
        total += loc_score
        details["metadata_location_score"] = loc_score
        reasons.extend(loc_reasons)

        fraud_score = min(total, 100)
        level = _risk_level(fraud_score)

        explanation = self._build_explanation(fraud_score, level, reasons, details)

        return FraudResult(
            fraud_score=fraud_score,
            risk_level=level,
            reasons=list(dict.fromkeys(reasons)),
            details=details,
            explanation=explanation,
        )

    @staticmethod
    def _build_explanation(score: int, level: str, reasons: List[str], details: Dict[str, Any]) -> str:
        parts = [f"Fraud Score: **{score}/100** (Risk: **{level}**)."]
        if reasons:
            parts.append(" Indicators:")
            for r in reasons:
                parts.append(f"  - {r}")
        if score < 25:
            parts.append(" No significant fraud indicators detected.")
        elif score < 50:
            parts.append(" Some minor indicators detected. Further verification recommended.")
        elif score < 75:
            parts.append(" Multiple fraud indicators present. Manual review strongly recommended.")
        else:
            parts.append(" High-confidence fraud detection. Escalate to fraud investigation team.")
        return " ".join(parts)
