"""
Model Evaluation Script for Site-Separated Sonar Anomaly Validation.
"""

import os
import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8n Sonar Anomaly Model")
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="Model weights path")
    parser.add_argument("--data", type=str, default="ml/training/dataset.yaml", help="dataset.yaml path")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split")
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image size")
    parser.add_argument("--device", type=str, default="cpu", help="Device e.g. 0 or cpu")
    return parser.parse_args()


def evaluate():
    args = parse_args()
    print("==================================================")
    print("SONAR-INTEL: Evaluation on Site-Separated Test/Val")
    print(f"Model Weights: {args.weights}")
    print(f"Dataset      : {args.data}")
    print(f"Split        : {args.split}")
    print("==================================================")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics is not installed.")
        sys.exit(1)

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        verbose=True
    )

    print("\n--- Summary Metrics ---")
    print(f"mAP@50     : {metrics.box.map50:.4f}")
    print(f"mAP@50-95  : {metrics.box.map:.4f}")
    print(f"Precision  : {metrics.box.mp:.4f}")
    print(f"Recall     : {metrics.box.mr:.4f}")
    return metrics


if __name__ == "__main__":
    evaluate()
