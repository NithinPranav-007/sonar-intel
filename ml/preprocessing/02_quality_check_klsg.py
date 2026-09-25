"""02_quality_check_klsg.py — VALID/SUSPICIOUS/INVALID labeling, no modification of raw files."""
import glob, os, csv
from PIL import Image
import numpy as np

KLSG_DIR = "KLSG/extracted"
ship_files = sorted(set(glob.glob(f"{KLSG_DIR}/ship*.png")))
plane_files = sorted(set(glob.glob(f"{KLSG_DIR}/plane-real/*.png")))
all_files = [(f, "ship") for f in ship_files] + [(f, "plane") for f in plane_files]

rows = []
for img_path, orig_class in all_files:
    stem = os.path.splitext(os.path.basename(img_path))[0]
    status, reasons = "VALID", []
    try:
        img = Image.open(img_path)
        img.verify()
        img = Image.open(img_path)
        arr = np.array(img.convert("L"))
    except Exception as e:
        rows.append({"file": stem, "orig_class": orig_class, "status": "INVALID", "reason": f"corrupt:{e}"})
        continue

    w, h = img.size
    if w < 50 or h < 50:
        status, reasons = "SUSPICIOUS", reasons + ["very_small_image"]

    zero_frac = np.mean(arr == 0)
    sat_frac = np.mean(arr == 255)
    if zero_frac > 0.5 or sat_frac > 0.5:
        status, reasons = "SUSPICIOUS", reasons + [f"excessive_zero_or_saturated_px:{zero_frac:.2f}/{sat_frac:.2f}"]

    rows.append({"file": stem, "orig_class": orig_class, "status": status, "reason": ";".join(reasons) if reasons else "ok"})

os.makedirs("outputs", exist_ok=True)
out = "outputs/klsg_quality_check.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "orig_class", "status", "reason"])
    writer.writeheader()
    writer.writerows(rows)

from collections import Counter
print("=== KLSG Quality Check ===")
print(dict(Counter(r["status"] for r in rows)))
print(f"Report written to {out}")