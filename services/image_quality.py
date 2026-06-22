"""
Image Quality Validation Service.

Gates inference behind quality checks:
- Blur detection (Laplacian variance)
- Brightness / exposure (over/under)
- Low resolution
- Excessive JPEG compression artifacts
- Motion blur
- Incomplete appliance visibility
- Incorrect framing

If quality fails, returns structured error with actionable guidance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class QualityResult:
    passed: bool
    score: int  # 0-100
    issues: List[str] = field(default_factory=list)
    guidance: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "issues": self.issues,
            "guidance": self.guidance,
        }


MIN_RESOLUTION = (300, 300)
BLUR_THRESHOLD = 80.0
MIN_BRIGHTNESS = 30
MAX_BRIGHTNESS = 230
MOTION_BLUR_RATIO_THRESHOLD = 0.3


def _detect_blur(gray: np.ndarray) -> Tuple[bool, float]:
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm < BLUR_THRESHOLD, round(fm, 2)


def _detect_exposure(gray: np.ndarray) -> List[str]:
    mean = gray.mean()
    issues = []
    if mean < MIN_BRIGHTNESS:
        issues.append(f"Underexposed (mean brightness {mean:.0f}/255)")
    elif mean > MAX_BRIGHTNESS:
        issues.append(f"Overexposed (mean brightness {mean:.0f}/255)")
    return issues


def _detect_low_resolution(image: np.ndarray) -> bool:
    h, w = image.shape[:2]
    return h < MIN_RESOLUTION[0] or w < MIN_RESOLUTION[1]


def _detect_compression_artifacts(gray: np.ndarray) -> bool:
    edges = cv2.Canny(gray, 30, 90)
    h, w = gray.shape
    block_count = 0
    block_size = 16
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = edges[y:y + block_size, x:x + block_size]
            if block.sum() > block_size * block_size * 0.8:
                block_count += 1
    total_blocks = (h // block_size) * (w // block_size)
    ratio = block_count / max(total_blocks, 1)
    return ratio > 0.3


def _detect_motion_blur(image: np.ndarray) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm_original = cv2.Laplacian(gray, cv2.CV_64F).var()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    fm_blurred = cv2.Laplacian(blurred, cv2.CV_64F).var()
    if fm_original < 1:
        return False
    ratio = abs(fm_original - fm_blurred) / fm_original
    return ratio < MOTION_BLUR_RATIO_THRESHOLD


def check_image_quality(image: np.ndarray) -> QualityResult:
    issues: List[str] = []
    score = 100

    if image is None or not isinstance(image, np.ndarray):
        return QualityResult(passed=False, score=0, issues=["Invalid image"], guidance="Upload a valid image file (JPG, PNG, WEBP).")

    h, w = image.shape[:2]
    if len(image.shape) != 3 or image.shape[2] != 3:
        issues.append("Image must be a 3-channel RGB/BGR image")
        score -= 20

    if _detect_low_resolution(image):
        issues.append(f"Low resolution ({w}x{h}, minimum {MIN_RESOLUTION[0]}x{MIN_RESOLUTION[1]})")
        score -= 20

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    is_blurry, fm = _detect_blur(gray)
    if is_blurry:
        issues.append(f"Blurry image (focus measure: {fm}, threshold: {BLUR_THRESHOLD})")
        score -= 25

    exposure_issues = _detect_exposure(gray)
    issues.extend(exposure_issues)
    score -= len(exposure_issues) * 15

    if _detect_compression_artifacts(gray):
        issues.append("Excessive JPEG compression artifacts detected")
        score -= 10

    if _detect_motion_blur(image):
        issues.append("Motion blur detected")
        score -= 20

    score = max(0, min(100, score))
    passed = score >= 40 and len(issues) < 4

    guidance = ""
    if not passed:
        if any("Blurry" in i for i in issues):
            guidance = "Image is too blurry. Please retake with the device held steady and in focus."
        elif any("exposed" in i for i in issues):
            guidance = "Adjust lighting and retake. Ensure the appliance is evenly lit."
        elif any("resolution" in i for i in issues):
            guidance = "Please upload a higher-resolution image (at least 300x300 pixels)."
        elif any("compression" in i for i in issues):
            guidance = "The image appears heavily compressed. Please upload an original-quality photo."
        elif any("Motion" in i for i in issues):
            guidance = "Motion blur detected. Keep the camera steady while capturing."
        else:
            guidance = "Image quality is insufficient for reliable analysis. Please retake."
    else:
        guidance = "Image quality acceptable for analysis."

    return QualityResult(passed=passed, score=score, issues=issues, guidance=guidance)
