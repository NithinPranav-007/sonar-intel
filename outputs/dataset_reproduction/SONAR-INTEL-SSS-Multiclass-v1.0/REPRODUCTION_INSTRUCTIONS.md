# SONAR-INTEL-SSS-Multiclass-v1.0: Reproduction & Deployment Guide

This guide provides exact, deterministic instructions for transferring and deploying `SONAR-INTEL-SSS-Multiclass-v1.0` to a dedicated model training PC.

---

## 1. Prerequisites & Environment Setup

On the dedicated training PC:

```bash
# 1. Ensure Python 3.10+ and PyTorch with CUDA are installed
python --version
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# 2. Install required packages
pip install ultralytics opencv-python numpy pyyaml pandas
```

---

## 2. Directory Structure on Training PC

Create the workspace directory structure:

```
SONAR-INTEL/
├── data/
│   └── dataset_v1.0/
│       ├── dataset.yaml
│       ├── preprocessing_config.yaml
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
└── outputs/
    └── dataset_reproduction/
        └── SONAR-INTEL-SSS-Multiclass-v1.0/
            ├── DATASET_SPEC.md
            ├── DATASET_MANIFEST.csv
            ├── split_manifest.csv
            ├── dataset.yaml
            ├── class_mapping.yaml
            ├── preprocessing_config.yaml
            ├── provenance.yaml
            ├── checksums.sha256
            └── REPRODUCTION_INSTRUCTIONS.md
```

---

## 3. Dataset Transfer & Verification Workflow

### Step 3.1: Transfer Dataset Archive
Transfer the pre-packaged dataset archive or `data/dataset_v1.0/` folder directly to the training PC via `rsync`, `scp`, or high-speed network share:

```bash
# Example rsync command
rsync -avz --progress user@source-pc:/path/to/SONAR-INTEL/data/dataset_v1.0/ /path/to/training-pc/SONAR-INTEL/data/dataset_v1.0/
```

### Step 3.2: Verify Checksums
Run SHA-256 validation against `checksums.sha256` to ensure zero bit rot or missing tiles:

**Linux / macOS:**
```bash
cd /path/to/SONAR-INTEL
sha256sum -c outputs/dataset_reproduction/SONAR-INTEL-SSS-Multiclass-v1.0/checksums.sha256 --quiet
```

**Windows PowerShell:**
```powershell
Set-Location "C:\path\to\SONAR-INTEL"
$failed = 0
Get-Content "outputs\dataset_reproduction\SONAR-INTEL-SSS-Multiclass-v1.0\checksums.sha256" | ForEach-Object {
    $parts = $_ -split '\s+', 2
    $expectedHash = $parts[0]
    $file = $parts[1].Trim()
    if (Test-Path $file) {
        $actualHash = (Get-FileHash $file -Algorithm SHA256).Hash.ToLower()
        if ($actualHash -ne $expectedHash) {
            Write-Error "Mismatch in $file"
            $failed++
        }
    } else {
        Write-Error "Missing file $file"
        $failed++
    }
}
if ($failed -eq 0) { Write-Host "ALL 18,392 FILES PASSED SHA-256 INTEGRITY VERIFICATION!" -ForegroundColor Green }
```

---

## 4. Verification Checklist Before Launching Training

Run the following validation script on the training machine:

```python
import os, glob

base_dir = "data/dataset_v1.0"
splits = ["train", "val", "test"]
expected_counts = {"train": 6444, "val": 1376, "test": 1376}

for s in splits:
    imgs = len(glob.glob(os.path.join(base_dir, "images", s, "*.png")))
    lbls = len(glob.glob(os.path.join(base_dir, "labels", s, "*.txt")))
    print(f"[{s.upper()}] Images: {imgs} (expected {expected_counts[s]}), Labels: {lbls}")
    assert imgs == expected_counts[s], f"Count mismatch in {s} images!"
    assert lbls == expected_counts[s], f"Count mismatch in {s} labels!"

print("100% PASS: Dataset counts are verified and ready for model training!")
```

---

## 5. Training Launch Command

Once verified, launch Ultralytics YOLOv8 training on the GPU:

```python
from ultralytics import YOLO

# 1. Initialize YOLOv8 architecture (e.g., yolov8s.pt or fresh checkpoint)
model = YOLO("yolov8s.pt")

# 2. Train using authoritative dataset specification
model.train(
    data="data/dataset_v1.0/dataset.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,          # GPU device ID
    optimizer="AdamW",
    lr0=0.001,
    patience=15,
    project="outputs/training",
    name="yolov8s_sonar_v1.0"
)
```
