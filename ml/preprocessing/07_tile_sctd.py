"""
07_tile_sctd.py — tiles SCTD's normalized images and converts boxes to YOLO format directly.
(Combines what 07_tile.py + 08_mask_to_yolo.py do separately for AI4Shipwrecks, since SCTD
has real boxes already, not masks — there's no mask-tiling step needed.)

Nadir masking: SKIPPED for SCTD (documented decision, same pattern as CLAHE/denoise).
Reason: SCTD images are small, pre-cropped chips centered on one object, not continuous
swaths — there is no distinct nadir strip to mask, and doing so risks cutting the object.
"""
import xml.etree.ElementTree as ET
import glob, os, csv
from PIL import Image
import numpy as np

SCTD_ANNOT_DIR = "SCTD/extracted/SCTD/Annotations"
NORM_IMG_DIR = "data/interim/sctd_normalized"
OUT_IMG = "data/processed/sctd/images"
OUT_LBL = "data/processed/sctd/labels"
os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

TILE = 640
STRIDE = 512

# Single-class scheme, matching the repo's actual convention (0 = artificial_anomaly)
CLASS_ID = 0

def get_tile_windows(w, h, tile=TILE, stride=STRIDE):
    if w <= tile and h <= tile:
        return [(0, 0, w, h, "pad")]
    xs = list(range(0, max(w - tile, 0) + 1, stride)) or [0]
    ys = list(range(0, max(h - tile, 0) + 1, stride)) or [0]
    return [(x, y, min(x + tile, w), min(y + tile, h), "slide") for y in ys for x in xs]

tile_manifest = []
total_tiles = 0

for xml_path in sorted(glob.glob(f"{SCTD_ANNOT_DIR}/*.xml")):
    stem = os.path.splitext(os.path.basename(xml_path))[0]
    img_path = f"{NORM_IMG_DIR}/{stem}.jpg"
    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path).convert("L")
    arr = np.array(img)
    w, h = img.size

    tree = ET.parse(xml_path)
    boxes = []
    for obj in tree.findall("object"):
        bb = obj.find("bndbox")
        xmin, ymin = float(bb.find("xmin").text), float(bb.find("ymin").text)
        xmax, ymax = float(bb.find("xmax").text), float(bb.find("ymax").text)
        boxes.append((xmin, ymin, xmax, ymax))

    for i, (x0, y0, x1, y1, mode) in enumerate(get_tile_windows(w, h)):
        tile_w, tile_h = x1 - x0, y1 - y0
        canvas = np.zeros((TILE, TILE), dtype=np.uint8)
        canvas[0:tile_h, 0:tile_w] = arr[y0:y1, x0:x1]
        tile_name = f"sctd_{stem}_t{i}"

        tile_boxes = []
        for (xmin, ymin, xmax, ymax) in boxes:
            ixmin, iymin = max(xmin, x0), max(ymin, y0)
            ixmax, iymax = min(xmax, x1), min(ymax, y1)
            if ixmax <= ixmin or iymax <= iymin:
                continue
            bxmin, bymin = ixmin - x0, iymin - y0
            bxmax, bymax = ixmax - x0, iymax - y0
            cx, cy = (bxmin + bxmax) / 2 / TILE, (bymin + bymax) / 2 / TILE
            bw, bh = (bxmax - bxmin) / TILE, (bymax - bymin) / TILE
            tile_boxes.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        Image.fromarray(canvas).save(f"{OUT_IMG}/{tile_name}.png")
        with open(f"{OUT_LBL}/{tile_name}.txt", "w") as f:
            f.write("\n".join(tile_boxes))

        tile_manifest.append({
            "tile": f"{tile_name}.png",
            "source_image": f"{stem}.jpg",
            "mode": mode,
            "num_boxes": len(tile_boxes),
        })
        total_tiles += 1

os.makedirs("outputs", exist_ok=True)
with open("outputs/sctd_tile_manifest.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(tile_manifest[0].keys()))
    writer.writeheader()
    writer.writerows(tile_manifest)

print("=== SCTD Tiling ===")
print(f"Total tiles produced: {total_tiles}")
print("Manifest: outputs/sctd_tile_manifest.csv")