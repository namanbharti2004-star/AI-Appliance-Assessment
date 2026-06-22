"""
Phase 10: Demo Video Generation

Generates a professional demo video from an input video file.
Processes every frame through the full pipeline:
  Frame → Appliance Detection → Damage Detection → Severity → Fraud → Cost → Overlay

Output: demo_video.mp4 with all metrics overlaid.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import API_CONFIG, get_device
from scripts.inference import InspectionPipeline

DEMO_COLORS = {
    "appliance": (0, 200, 0),
    "damage": (0, 0, 255),
    "severity": (255, 165, 0),
    "fraud": (255, 0, 0),
    "cost": (128, 0, 128),
    "claim": (0, 128, 255),
    "header_bg": (20, 20, 20),
    "panel_bg": (40, 40, 40),
    "text": (255, 255, 255),
    "approved": (0, 200, 0),
    "rejected": (0, 0, 200),
}


def _put_text_bg(
    img: np.ndarray,
    text: str,
    pos: tuple,
    font_scale: float = 0.6,
    color: tuple = (255, 255, 255),
    bg_color: tuple = (40, 40, 40),
    thickness: int = 2,
    padding: int = 4,
) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - padding, y - th - padding), (x + tw + padding, y + padding), bg_color, -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


class DemoGenerator:
    def __init__(self, device: Optional[str] = None):
        self.pipeline = InspectionPipeline(device=device)
        self.device = device or get_device()

    def _create_header(self, frame: np.ndarray, report: Dict[str, Any], fps: float, frame_idx: int) -> np.ndarray:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 48), DEMO_COLORS["header_bg"], -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        sev = report.get("severity", "N/A")
        fraud = report.get("fraud_score", 0)
        risk = report.get("claim_risk", "N/A")
        decision = report.get("decision", "N/A")
        cost = report.get("repair_cost_display", f"₹{report.get('repair_cost', 0)}")
        appliance = report.get("appliance", "?")

        left = f"AI Appliance Inspection  |  {appliance}  |  Sev:{sev}  Fraud:{fraud}  Risk:{risk}"
        right = f"Dec:{decision}  Cost:{cost}  F:{fps:.1f}fps  #{frame_idx}"

        cv2.putText(frame, left, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, DEMO_COLORS["text"], 2)
        rw, _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        cv2.putText(frame, right, (w - rw - 12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, DEMO_COLORS["text"], 2)

        return frame

    def _create_info_panel(self, frame: np.ndarray, report: Dict[str, Any]) -> np.ndarray:
        h, w = frame.shape[:2]
        panel_h = 180
        panel_y = h - panel_h - 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, panel_y), (340, panel_y + panel_h), DEMO_COLORS["panel_bg"], -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        sev = report.get("severity", "N/A")
        fraud = report.get("fraud_score", 0)
        risk = report.get("claim_risk", "N/A")
        score = report.get("claim_score", 0)
        condition = report.get("condition_score", 100)
        grade = report.get("grade", "A")
        decision = report.get("decision", "N/A")
        dec_color = DEMO_COLORS["approved"] if decision == "APPROVE" else DEMO_COLORS["rejected"]
        cost = report.get("repair_cost_display", f"₹{report.get('repair_cost', 0)}")
        damage_type = report.get("damage_type", "none")
        damage_pct = report.get("damage_percentage", 0)

        lines = [
            (f"Appliance: {report.get('appliance', '?')}", DEMO_COLORS["appliance"]),
            (f"Damage: {damage_type} ({damage_pct}%)", DEMO_COLORS["damage"]),
            (f"Severity: {sev}  |  Grade: {grade} ({condition}/100)", DEMO_COLORS["severity"]),
            (f"Fraud Score: {fraud}/100  ({report.get('fraud_risk_level', 'N/A')})", DEMO_COLORS["fraud"]),
            (f"Claim Risk: {risk}  |  Score: {score}/100", DEMO_COLORS["claim"]),
            (f"Est. Cost: {cost}", DEMO_COLORS["cost"]),
            (f"Decision: {decision}", dec_color),
        ]

        y = panel_y + 22
        for text, color in lines:
            cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            y += 22

        return frame

    def _draw_detection_boxes(self, frame: np.ndarray, report: Dict[str, Any]) -> np.ndarray:
        appliance_bbox = report.get("appliance_bbox")
        if appliance_bbox:
            x1, y1, x2, y2 = [int(v) for v in appliance_bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), DEMO_COLORS["appliance"], 2)
            label = f"{report.get('appliance', '')} {report.get('appliance_confidence', 0):.2f}"
            _put_text_bg(frame, label, (x1, max(22, y1 - 6)), 0.5, DEMO_COLORS["text"], DEMO_COLORS["appliance"])

        for det in report.get("damage_detections", []):
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), DEMO_COLORS["damage"], 2)
            label = f"{det['class_name']} {det['confidence']:.2f}"
            _put_text_bg(frame, label, (x1, y2 + 18), 0.45, DEMO_COLORS["text"], DEMO_COLORS["damage"])

            if det.get("segmentation") and det.get("mask"):
                mask_np = np.array(det["mask"], dtype=np.uint8)
                h, w = frame.shape[:2]
                if mask_np.shape[:2] != (h, w):
                    mask_np = cv2.resize(mask_np, (w, h))
                colored = np.zeros_like(frame)
                colored[mask_np > 0] = DEMO_COLORS["damage"]
                cv2.addWeighted(colored, 0.35, frame, 0.65, 0, frame)

        return frame

    def _create_watermark(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        text = "AI Appliance Inspection Platform"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x, y = w - tw - 15, h - 10
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return frame

    def generate(
        self,
        video_path: str,
        output_dir: str = "output",
        max_frames: int = 0,
        skip_frames: int = 0,
        appliance_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not os.path.exists(video_path):
            return {"error": f"Video not found: {video_path}"}

        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if max_frames > 0 and max_frames < total_frames:
            skip_frames = max(1, total_frames // max_frames)
            total_to_process = max_frames
        elif skip_frames > 0:
            total_to_process = total_frames // (skip_frames + 1)
        else:
            total_to_process = total_frames

        output_path = os.path.join(output_dir, f"demo_{os.path.basename(video_path)}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_idx = 0
        processed = 0
        timings: List[float] = []
        worst_report: Dict[str, Any] = {}

        logger.info("Generating demo video: {} frames ({} total)", total_to_process, total_frames)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if skip_frames > 0 and frame_idx % (skip_frames + 1) != 0:
                frame_idx += 1
                continue

            start = time.perf_counter()
            temp_path = f"/tmp/_demo_frame_{frame_idx}.jpg"
            cv2.imwrite(temp_path, frame)

            result = self.pipeline.inspect_image(
                image_path=temp_path,
                appliance_override=appliance_override,
                save_visualizations=False,
            )
            report = result.get("report", {})

            if os.path.exists(temp_path):
                os.remove(temp_path)

            elapsed = time.perf_counter() - start
            timings.append(elapsed)

            annotated = frame.copy()
            annotated = self._draw_detection_boxes(annotated, report)
            annotated = self._create_header(annotated, report, fps, processed + 1)
            annotated = self._create_info_panel(annotated, report)
            annotated = self._create_watermark(annotated)

            writer.write(annotated)

            if not worst_report:
                worst_report = report
            else:
                fraud = report.get("fraud_score", 0)
                damage = report.get("damage_percentage", 0)
                w_fraud = worst_report.get("fraud_score", 0)
                w_damage = worst_report.get("damage_percentage", 0)
                if fraud > w_fraud or damage > w_damage:
                    if fraud / max(w_fraud, 1) > damage / max(w_damage, 1):
                        if fraud > w_fraud:
                            worst_report = report
                    elif damage > w_damage:
                        worst_report = report

            processed += 1
            if processed % 10 == 0:
                logger.info("  Processed {}/{} frames", processed, total_to_process)

            if max_frames > 0 and processed >= max_frames:
                break

            frame_idx += 1

        cap.release()
        writer.release()

        avg_time = sum(timings) / max(len(timings), 1)
        logger.info("Demo video saved: {} ({} frames, {:.2f}s avg/frame)", output_path, processed, avg_time)

        demo_report = worst_report or {}
        demo_report["demo_metadata"] = {
            "source_video": video_path,
            "output_video": output_path,
            "frames_processed": processed,
            "total_frames": total_frames,
            "avg_inference_time_ms": round(avg_time * 1000, 1),
            "fps": fps,
        }

        report_path = os.path.join(output_dir, f"demo_report_{os.path.splitext(os.path.basename(video_path))[0]}.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(demo_report, f, indent=2, default=str)

        return {
            "demo_video_path": output_path,
            "demo_report_path": report_path,
            "frames_processed": processed,
            "avg_inference_time_ms": round(avg_time * 1000, 1),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate professional demo video")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0=all)")
    parser.add_argument("--skip-frames", type=int, default=0, help="Process every Nth frame")
    parser.add_argument("--appliance-override", type=str, default=None, help="Force appliance label")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"])
    args = parser.parse_args()

    gen = DemoGenerator(device=args.device)
    result = gen.generate(
        video_path=args.video,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        skip_frames=args.skip_frames,
        appliance_override=args.appliance_override,
    )
    if result.get("error"):
        print(f"ERROR: {result['error']}")
        sys.exit(1)

    print(json.dumps(result, indent=2, default=str))
    print(f"\nDemo video saved to: {result['demo_video_path']}")


if __name__ == "__main__":
    main()
