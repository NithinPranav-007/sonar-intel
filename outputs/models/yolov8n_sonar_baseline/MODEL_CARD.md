# Model Card: YOLOv8n Sonar Anomaly Baseline (`yolov8n-sonar-baseline`)

**Model Name:** YOLOv8n Sonar Baseline  
**Model Identifier:** `yolov8n-sonar-baseline`  
**Task:** Side-Scan Sonar Artificial Anomaly Candidate Detection  
**Artifact Path:** `outputs/models/yolov8n_sonar_baseline/best.pt`  
**Checkpoint Selection:** Epoch 17 (Peak Validation mAP@50 = 0.0685)  
**Date:** 2026-09-01  
**Framework:** PyTorch 2.13.0+cu126, Ultralytics 8.4.137  

---

## 1. Model Overview & Primary Purpose

`yolov8n-sonar-baseline` is a single-class object detection model based on the Ultralytics YOLOv8 nano architecture (3.01M parameters). It is fine-tuned from COCO pretrained weights to propose candidate bounding boxes for anthropogenic structures, shipwreck debris, and acoustic anomalies in side-scan sonar (SSS) waterfall imagery.

> [!IMPORTANT]
> **Operational Role:** This model serves strictly as an **MVP candidate generator for human-in-the-loop triage**. It is **NOT** a production-grade autonomous wreck identification system or autonomous navigation system. All AI candidates require human hydrographic/sonar analyst review.

---

## 2. Dataset & Splitting Methodology

- **Source Dataset:** AI4Shipwrecks (Side-scan sonar waterfall swaths).
- **Class Mapping:** Single-class `0: artificial_anomaly` (merges shipwrecks, metallic debris, artificial reefs, and anomalous acoustic targets).
- **Input Dimensions:** 640 &times; 640 pixels (swaths tiled with 20% spatial overlap / 512px stride, with deterministic zero padding).
- **Preprocessing:** 1–99% swath-level percentile contrast normalization (approved baseline). No CLAHE or FFT denoising applied.
- **Partitioning:** Site-aware, track-separated splitting (zero geographic leakage across folds):
  - **Train:** 5,844 tiles (612 positive, 5,232 negative backgrounds, 1,034 GT boxes across 185 sites)
  - **Validation:** 1,256 tiles (130 positive, 1,126 negative backgrounds, 195 GT boxes across 55 sites)
  - **Test (Frozen):** 1,256 tiles (132 positive, 1,124 negative backgrounds, 271 GT boxes across 46 sites)

---

## 3. Training Configuration & Hardware Environment

- **Architecture:** YOLOv8n (130 layers, 3,011,043 parameters, 8.2 GFLOPs)
- **Base Weights:** Pretrained COCO initialization (`yolov8n.pt`)
- **Epochs:** 25 completed (Early model selection froze weights at best epoch = 17)
- **Batch Size:** 8
- **Image Size:** 640 &times; 640
- **Optimizer:** AdamW (lr = 0.002, momentum = 0.9, weight_decay = 0.0005)
- **Precision:** Automatic Mixed Precision (AMP FP16 enabled)
- **DataLoader Workers:** 2
- **Seed / Deterministic:** 42 / True
- **Hardware Target:** NVIDIA GeForce RTX 3050 Laptop GPU (4,096 MiB VRAM, Compute Capability 8.6)
- **Average VRAM Utilization:** ~1.01 GB (~25% of dedicated VRAM, >3.0 GB headroom)

---

## 4. Benchmark Performance Metrics

- **Selected Checkpoint:** Epoch 17
- **Validation Precision:** `0.1612` (16.1%)
- **Validation Recall:** `0.1077` (10.8%)
- **Validation mAP@50:** `0.0685` (6.85%)
- **Validation mAP@50-95:** `0.0202` (2.02%)

*(Detailed frozen test metrics are documented in the companion evaluation report.)*

---

## 5. Important Limitations & Failure Modes

1. **Acoustic Shadow Sensitivity:** The detector relies on contrast between specular highlight acoustic backscatter and adjacent acoustic shadows. Low-relief anomalies or sedimented wrecks with faint shadows exhibit elevated false negative rates.
2. **Geological Clutter:** High-intensity rocky ridges, sand ripples, and boulder fields can trigger false positives.
3. **Tile Seam Truncation:** Large shipwrecks split across 640&times;640 tile seams may have partial fragments with insufficient spatial context to meet confidence thresholds on an individual tile.
4. **Coordinate Accuracy:** Geographic coordinates produced during inference are derived from towfish navigation interpolation (`localization_status = ESTIMATED`), not precision acoustic positioning (USBL).
