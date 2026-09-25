"""
07_tile_klsg.py — tiles KLSG's normalized images.
KLSG has NO real bounding box annotations. Every label here is a WEAK,
approximated box (central 80% of each image), not a real annotation.
Flagged weak_label=True in the manifest so it's never confused with
SCTD's or AI4Shipwrecks' real annotations.
"""
import glob, os, csv
from PIL import Image
import numpy as np

NORM_DIR = "data/interim/klsg_normalized"
OUT_IMG = "data/processed/klsg/images"
OUT_LBL = "data/processed/klsg/labels"
os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

TILE = 640
STRIDE = 512
CLASS_ID = 0

def get_tile_windows(w, h, tile=TILE, stride=STRIDE):
    if w <= tile and h <= tile:
        return [(0, 0, w, h, "pad")]
    xs = list(range(0, max(w - tile, 0) + 1, stride)) or [0]
    ys = list(range(0, max(h - tile, 0) + 1, stride)) or [0]
    return [(x, y, min(x + tile, w), min(y + tile, h), "slide") for y in ys for x in xs]

tile_manifest = []
total_tiles = 0

for img_path in sorted(glob.glob(f"{NORM_DIR}/*.png")):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    orig_class = "plane" if "plane" in stem.lower() else "ship"

    img = Image.open(img_path).convert("L")
    arr = np.array(img)
    w, h = img.size

    for i, (x0, y0, x1, y1, mode) in enumerate(get_tile_windows(w, h)):
        tile_w, tile_h = x1 - x0, y1 - y0
        canvas = np.zeros((TILE, TILE), dtype=np.uint8)
        canvas[0:tile_h, 0:tile_w] = arr[y0:y1, x0:x1]
        tile_name = f"klsg_{stem}_t{i}"

        cx = (tile_w / 2) / TILE
        cy = (tile_h / 2) / TILE
        bw = (tile_w * 0.8) / TILE
        bh = (tile_h * 0.8) / TILE
        box_line = f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

        Image.fromarray(canvas).save(f"{OUT_IMG}/{tile_name}.png")
        with open(f"{OUT_LBL}/{tile_name}.txt", "w") as f:
            f.write(box_line)

        tile_manifest.append({
            "tile": f"{tile_name}.png",
            "source_image": f"{stem}.png",
            "orig_class": orig_class,
            "mode": mode,
            "weak_label": True,
        })
        total_tiles += 1

os.makedirs("outputs", exist_ok=True)
with open("outputs/klsg_tile_manifest.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(tile_manifest[0].keys()))
    writer.writeheader()
    writer.writerows(tile_manifest)

print("=== KLSG Tiling (WEAK LABELS) ===")
print(f"Total tiles produced: {total_tiles}")
print("Manifest: outputs/klsg_tile_manifest.csv")