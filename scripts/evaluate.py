"""
Evaluation Script for Appliance and Damage Detection Models

Evaluates model performance using standard metrics:
- Precision, Recall, mAP50, mAP50-95
- Per-class performance
- Confusion matrix
- F1 score curves
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
import json
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import PATHS, get_device
from utils import setup_logging, read_image, calculate_iou


def evaluate_model(
    model_path: str,
    data_yaml: str,
    device: str = None,
    split: str = "val",
    save_predictions: bool = True,
    output_dir: str = "evaluation"
) -> Dict:
    """
    Evaluate YOLOv8 model performance.

    Args:
        model_path: Path to trained model
        data_yaml: Path to dataset YAML
        device: Device to use
        split: Dataset split to evaluate (train, val, test)
        save_predictions: Save prediction images
        output_dir: Directory for evaluation outputs

    Returns:
        Dictionary of evaluation metrics
    """
    from ultralytics import YOLO

    device = device or get_device()
    model = YOLO(model_path)

    logger.info(f"Evaluating model on {split} set...")

    # Run evaluation
    metrics = model.val(
        data=data_yaml,
        device=device,
        split=split,
        verbose=True,
        save=save_predictions,
        project=output_dir,
        name="predictions"
    )

    # Compile metrics
    results = {
        "model_path": model_path,
        "data_yaml": data_yaml,
        "split": split,
        "metrics": {
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "f1": float(metrics.box.f1)
        }
    }

    # Per-class metrics if available
    if hasattr(metrics.box, 'ap_class_index'):
        class_metrics = {}
        class_indices = metrics.box.ap_class_index
        for i, idx in enumerate(class_indices):
            class_metrics[int(idx)] = {
                "AP50": float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else 0.0,
                "AP": float(metrics.box.ap[i]) if i < len(metrics.box.ap) else 0.0
            }
        results["per_class_metrics"] = class_metrics

    logger.info(f"Evaluation results: {results['metrics']}")

    return results


def compute_confusion_matrix(
    predictions: List[Dict],
    ground_truths: List[Dict],
    num_classes: int,
    iou_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute confusion matrix for object detection.

    Args:
        predictions: List of prediction dictionaries
        ground_truths: List of ground truth dictionaries
        num_classes: Number of classes
        iou_threshold: IoU threshold for matching

    Returns:
        Confusion matrix and class-wise metrics
    """
    conf_matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int32)

    # Match predictions to ground truths
    for pred, gt in zip(predictions, ground_truths):
        pred_classes = [p["class_id"] for p in pred]
        gt_classes = [g["class_id"] for g in gt]

        # Simple matching based on IoU
        matched_gt = set()
        for p in pred:
            best_iou = 0
            best_gt_idx = -1
            for i, g in enumerate(gt):
                if i in matched_gt:
                    continue
                iou = calculate_iou(p["bbox"], g["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                conf_matrix[p["class_id"], gt[best_gt_idx]["class_id"]] += 1
            else:
                # False positive
                conf_matrix[p["class_id"], num_classes] += 1

        # Unmatched ground truths are false negatives
        for i, g in enumerate(gt):
            if i not in matched_gt:
                conf_matrix[num_classes, g["class_id"]] += 1

    return conf_matrix


def calculate_precision_recall(
    conf_matrix: np.ndarray
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Calculate precision, recall, and F1 from confusion matrix.

    Args:
        conf_matrix: Confusion matrix

    Returns:
        Precision, recall, and F1 per class
    """
    num_classes = conf_matrix.shape[0] - 1

    precision = {}
    recall = {}
    f1 = {}

    for i in range(num_classes):
        tp = conf_matrix[i, i]
        fp = conf_matrix[i, :].sum() - tp
        fn = conf_matrix[:, i].sum() - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        precision[i] = prec
        recall[i] = rec
        f1[i] = f1_score

    return precision, recall, f1


def save_evaluation_report(
    results: Dict,
    output_path: str
):
    """
    Save evaluation report to JSON file.

    Args:
        results: Evaluation results
        output_path: Path to save report
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Evaluation report saved to {output_path}")


def plot_metrics(
    results: Dict,
    output_dir: str
):
    """
    Create visualization plots for metrics.

    Args:
        results: Evaluation results
        output_dir: Directory to save plots
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        metrics = results.get("metrics", {})

        # Bar chart for main metrics
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Metrics bar chart
        metric_names = ["mAP50", "mAP50-95", "precision", "recall", "f1"]
        metric_values = [
            metrics.get("mAP50", 0),
            metrics.get("mAP50-95", 0),
            metrics.get("precision", 0),
            metrics.get("recall", 0),
            metrics.get("f1", 0)
        ]

        axes[0].bar(metric_names, metric_values, color='steelblue')
        axes[0].set_ylim(0, 1)
        axes[0].set_title('Model Performance Metrics')
        axes[0].set_ylabel('Score')

        # Radar chart for metrics
        angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
        metric_values_plot = metric_values + [metric_values[0]]
        angles += angles[:1]

        axes[1].polar(angles, metric_values_plot, 'o-', linewidth=2, color='steelblue')
        axes[1].fill(angles, metric_values_plot, alpha=0.25, color='steelblue')
        axes[1].set_xticks(angles[:-1])
        axes[1].set_xticklabels(metric_names)
        axes[1].set_ylim(0, 1)
        axes[1].set_title('Metrics Overview')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "metrics_plot.png"), dpi=150)
        plt.close()

        logger.info(f"Metrics plot saved to {output_dir}/metrics_plot.png")

    except ImportError:
        logger.warning("matplotlib not available for plotting")


def main():
    """Main evaluation script"""
    parser = argparse.ArgumentParser(description="Evaluate Detection Models")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to dataset YAML"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation",
        help="Output directory"
    )
    parser.add_argument(
        "--no-save-predictions",
        action="store_true",
        help="Don't save prediction images"
    )

    args = parser.parse_args()

    # Setup logging (console only due to sandbox restrictions)
    from loguru import logger
    logger.add(sys.stderr, level="INFO")

    # Run evaluation
    results = evaluate_model(
        model_path=args.model,
        data_yaml=args.data,
        device=args.device,
        split=args.split,
        save_predictions=not args.no_save_predictions,
        output_dir=args.output_dir
    )

    # Save report
    report_path = os.path.join(args.output_dir, "evaluation_report.json")
    save_evaluation_report(results, report_path)

    # Create plots
    plot_metrics(results, args.output_dir)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"mAP50: {results['metrics']['mAP50']:.4f}")
    print(f"mAP50-95: {results['metrics']['mAP50-95']:.4f}")
    print(f"Precision: {results['metrics']['precision']:.4f}")
    print(f"Recall: {results['metrics']['recall']:.4f}")
    print(f"F1: {results['metrics']['f1']:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
