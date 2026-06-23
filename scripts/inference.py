"""
Inference entry point for the MVP image and video inspection platform.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import API_CONFIG, PATHS, get_device
from models.appliance_detector import ApplianceDetector
from report_engine import ReportGenerator, format_report_for_api, format_report_for_dashboard
from risk_engine import RiskEngine
from utils import read_image, save_image, validate_video_file


class InspectionPipeline:
    def __init__(self, appliance_model_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or get_device()
        self.appliance_detector = ApplianceDetector(model_path=appliance_model_path, device=self.device)
        self.report_generator = ReportGenerator(
            appliance_detector=self.appliance_detector,
            risk_engine=RiskEngine(),
        )

    def annotate_image(self, image: np.ndarray, report_dict: Dict[str, object]) -> np.ndarray:
        vis = image.copy()
        h, w = vis.shape[:2]

        appliance = report_dict.get("appliance", "unknown")
        appliance_bbox = report_dict.get("appliance_bbox")
        if appliance_bbox:
            x1, y1, x2, y2 = [int(v) for v in appliance_bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(
                vis,
                f"{appliance} {report_dict.get('appliance_confidence', 0):.2f}",
                (x1, max(25, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 0),
                2,
            )

        for detection in report_dict.get("damage_detections", []):
            bbox = detection.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                vis,
                f"{detection['class_name']} {detection['confidence']:.2f}",
                (x1, y2 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )

        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

        severity = report_dict.get("severity", "N/A")
        fraud = report_dict.get("fraud_score", 0)
        claim_risk = report_dict.get("claim_risk", "N/A")
        condition = report_dict.get("condition_score", 0)
        decision = report_dict.get("decision", "N/A")
        grade = report_dict.get("grade", "N/A")
        cost = report_dict.get("repair_cost_display") or f"Rs.{report_dict.get('repair_cost', 0)}"

        header = f"Severity: {severity}  |  Fraud: {fraud}/100  |  Claim: {claim_risk}  |  Condition: {condition}({grade})  |  Cost: {cost}  |  {decision}"
        cv2.putText(vis, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        lines = [
            f"Severity: {severity}",
            f"Fraud Score: {fraud} ({report_dict.get('fraud_risk_level', 'N/A')})",
            f"Claim Risk: {claim_risk} (Score: {report_dict.get('claim_score', 0)})",
        ]
        if report_dict.get("missing_part_detected"):
            lines.append(f"Missing: {report_dict.get('missing_part')}")
        lines.append(f"Condition: {condition}/{100} ({grade})")
        lines.append(f"Decision: {decision}")

        y = h - 24 * len(lines) - 10
        for line in lines:
            cv2.rectangle(vis, (0, y - 2), (380, y + 18), (0, 0, 0), -1)
            cv2.putText(vis, line, (8, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
            y += 24

        return vis

    def inspect_image(
        self,
        image_path: str,
        appliance_override: Optional[str] = None,
        save_visualizations: bool = True,
        output_dir: str = "output",
    ) -> Dict[str, object]:
        image = read_image(image_path)
        if image is None:
            return {"error": f"Failed to read image: {image_path}"}

        report = self.report_generator.generate_report(
            image=image,
            image_path=image_path,
            appliance_override=appliance_override,
            source_type="image",
        )
        report_dict = report.to_dict()

        visual_path = None
        os.makedirs(output_dir, exist_ok=True)
        annotated = self.annotate_image(image, report_dict)
        visual_path = os.path.join(output_dir, f"annotated_{report.report_id}.jpg")
        save_image(annotated, visual_path)

        result = {
            "report": report_dict,
            "api_format": format_report_for_api(report),
            "dashboard_format": format_report_for_dashboard(report),
            "annotated_image_path": visual_path,
        }
        return result

    def extract_frames_smart(self, video_path: str, max_frames: int) -> List[np.ndarray]:
        frames: List[np.ndarray] = []
        capture = cv2.VideoCapture(video_path)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        indices = []
        max_frames = min(max_frames, total_frames)
        if max_frames <= 3:
            indices = list(range(0, total_frames, max(1, total_frames // max(1, max_frames))))
        else:
            begin = total_frames // 4
            middle = total_frames // 2
            end = 3 * total_frames // 4
            indices = sorted(set([0, begin, middle, end, total_frames - 1] +
                                  [int(total_frames * i / max_frames) for i in range(max_frames)]))

        indices = [min(max(0, i), total_frames - 1) for i in indices]
        indices = sorted(set(indices))[:max_frames]

        for idx in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = capture.read()
            if ok:
                frames.append(frame)
        capture.release()
        return frames

    def inspect_video(
        self,
        video_path: str,
        appliance_override: Optional[str] = None,
        output_dir: str = "output",
    ) -> Dict[str, object]:
        if not validate_video_file(video_path, API_CONFIG["allowed_video_formats"]):
            return {"error": f"Invalid video file: {video_path}"}

        os.makedirs(output_dir, exist_ok=True)
        frames = self.extract_frames_smart(video_path, API_CONFIG["max_video_frames"])
        if not frames:
            return {"error": "Could not extract frames from video."}

        reports = []
        annotated_frames: List[np.ndarray] = []
        for idx, frame in enumerate(frames):
            temp_frame_path = os.path.join(PATHS["temp"], f"frame_{idx}.jpg")
            os.makedirs(PATHS["temp"], exist_ok=True)
            cv2.imwrite(temp_frame_path, frame)
            report = self.report_generator.generate_report(
                image=frame,
                image_path=temp_frame_path,
                appliance_override=appliance_override,
                source_type="video",
            )
            report_dict = report.to_dict()
            reports.append(report_dict)
            annotated_frames.append(self.annotate_image(frame, report_dict))

        damage_persistence: Dict[str, int] = {}
        frame_damages = []
        for r in reports:
            dets = r.get("damage_detections", [])
            frame_damages.append(dets)
            seen = set()
            for d in dets:
                dt = d.get("class_name", "unknown")
                if dt not in seen:
                    damage_persistence[dt] = damage_persistence.get(dt, 0) + 1
                    seen.add(dt)

        persistent_damages = {k: v for k, v in damage_persistence.items()
                              if v >= max(2, len(reports) * 0.3)}
        inconsistent_frames = [
            i for i, dets in enumerate(frame_damages)
            if not any(d.get("class_name") in persistent_damages for d in dets)
        ] if persistent_damages else []

        best_frame_idx = 0
        max_score = -1
        for i, r in enumerate(reports):
            score = r.get("damage_confidence", 0) * 0.6 + (1 - r.get("fraud_score", 0) / 100) * 0.4
            if score > max_score:
                max_score = score
                best_frame_idx = i

        summary = reports[best_frame_idx].copy()
        summary["frame_count"] = len(reports)
        summary["source_video"] = video_path
        summary["damage_persistence"] = damage_persistence
        summary["persistent_damages"] = list(persistent_damages.keys())
        summary["best_frame_index"] = best_frame_idx
        summary["inconsistent_frames"] = len(inconsistent_frames)
        if inconsistent_frames:
            summary["frame_inconsistency_detected"] = True
            summary["fraud_reasons"] = summary.get("fraud_reasons", []) + [
                f"Frame inconsistency: {len(inconsistent_frames)}/{len(reports)} frames lack persistent damages"
            ]

        capture = cv2.VideoCapture(video_path)
        fps = capture.get(cv2.CAP_PROP_FPS) or 10.0
        capture.release()
        height, width = annotated_frames[0].shape[:2]
        output_video_path = os.path.join(output_dir, f"annotated_{os.path.basename(video_path)}")
        writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        for frame in annotated_frames:
            writer.write(frame)
        writer.release()

        summary_path = os.path.join(output_dir, f"video_summary_{os.path.splitext(os.path.basename(video_path))[0]}.json")
        with open(summary_path, "w", encoding="utf-8") as file:
            json.dump({"summary": summary, "frames": reports, "damage_persistence": damage_persistence}, file, indent=2)

        best_frame_annotated_path = os.path.join(
            output_dir,
            f"annotated_best_frame_{os.path.splitext(os.path.basename(video_path))[0]}.jpg"
        )
        cv2.imwrite(best_frame_annotated_path, annotated_frames[best_frame_idx])

        return {
            "summary": summary,
            "frames": reports,
            "annotated_image_path": best_frame_annotated_path,
            "annotated_video_path": output_video_path,
            "summary_report_path": summary_path,
            "damage_persistence": damage_persistence,
            "best_frame_index": best_frame_idx,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MVP inspection on images or videos")
    parser.add_argument("--image", type=str, help="Path to an input image")
    parser.add_argument("--video", type=str, help="Path to an input video")
    parser.add_argument("--appliance-model", type=str, default=None, help="Path to appliance detector weights")
    parser.add_argument("--appliance-override", type=str, default=None, help="Force appliance label")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"])
    parser.add_argument("--save-vis", action="store_true", help="Save annotated image output")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--output-format", type=str, default="json", choices=["json", "api", "dashboard"])
    args = parser.parse_args()

    if not args.image and not args.video:
        raise SystemExit("Provide either --image or --video")

    pipeline = InspectionPipeline(appliance_model_path=args.appliance_model, device=args.device)
    if args.video:
        result = pipeline.inspect_video(args.video, appliance_override=args.appliance_override, output_dir=args.output_dir)
        print(json.dumps(result, indent=2, default=str))
        return

    result = pipeline.inspect_image(
        image_path=args.image,
        appliance_override=args.appliance_override,
        save_visualizations=args.save_vis,
        output_dir=args.output_dir,
    )
    key = {"json": "report", "api": "api_format", "dashboard": "dashboard_format"}[args.output_format]
    print(json.dumps(result[key], indent=2, default=str))


if __name__ == "__main__":
    main()
