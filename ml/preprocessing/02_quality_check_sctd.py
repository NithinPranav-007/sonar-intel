"""02_quality_check_sctd.py — VALID/SUSPICIOUS/INVALID labeling, no modification of raw files."""
import xml.etree.ElementTree as ET
import glob, os, csv
from PIL import Image
import numpy as np

SCTD_DIR = "SCTD/extracted/SCTD"
rows = []

for xml_path in sorted(glob.glob(f"{SCTD_DIR}/Annotations/*.xml")):
    stem = os.path.splitext(os.path.basename(xml_path))[0]
    img_path = f"{SCTD_DIR}/JPEGImages/{stem}.jpg"
    status, reasons = "VALID", []

    if not os.path.exists(img_path):
        rows.append({"file": stem, "status": "INVALID", "reason": "missing_image"})
        continue

    try:
        img = Image.open(img_path)
        img.verify()
        img = Image.open(img_path)
        arr = np.array(img.convert("L"))
    except Exception as e:
        rows.append({"file": stem, "status": "INVALID", "reason": f"corrupt:{e}"})
        continue

    w, h = img.size
    if w < 50 or h < 50:
        status, reasons = "SUSPICIOUS", reasons + ["very_small_image"]

    zero_frac = np.mean(arr == 0)
    sat_frac = np.mean(arr == 255)
    if zero_frac > 0.5 or sat_frac > 0.5:
        status, reasons = "SUSPICIOUS", reasons + [f"excessive_zero_or_saturated_px:{zero_frac:.2f}/{sat_frac:.2f}"]

    tree = ET.parse(xml_path)
    objs = tree.findall("object")
    if len(objs) == 0:
        status, reasons = "SUSPICIOUS", reasons + ["no_objects_in_annotation"]
    for obj in objs:
        bb = obj.find("bndbox")
        xmin, ymin = float(bb.find("xmin").text), float(bb.find("ymin").text)
        xmax, ymax = float(bb.find("xmax").text), float(bb.find("ymax").text)
        if xmin < 0 or ymin < 0 or xmax > w or ymax > h or xmax <= xmin or ymax <= ymin:
            status, reasons = "INVALID", reasons + ["out_of_bounds_or_degenerate_box"]

    rows.append({"file": stem, "status": status, "reason": ";".join(reasons) if reasons else "ok"})

os.makedirs("outputs", exist_ok=True)
out = "outputs/sctd_quality_check.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "status", "reason"])
    writer.writeheader()
    writer.writerows(rows)

from collections import Counter
counts = Counter(r["status"] for r in rows)
print("=== SCTD Quality Check ===")
print(dict(counts))
print(f"Report written to {out}")