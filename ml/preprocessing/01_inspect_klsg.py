"""01_inspect_klsg.py — reports KLSG dataset stats. Modifies nothing."""
import glob, os
from PIL import Image
from collections import Counter

KLSG_DIR = "KLSG/extracted"

ship_files = sorted(set(glob.glob(f"{KLSG_DIR}/ship*.png")))
plane_files = sorted(set(glob.glob(f"{KLSG_DIR}/plane-real/*.png")))

sizes = Counter()
formats = Counter()
for f in ship_files + plane_files:
    img = Image.open(f)
    sizes[img.size] += 1
    formats[img.mode] += 1

print("=== KLSG Inspection Report ===")
print(f"Ship images found: {len(ship_files)}")
print(f"Plane images found: {len(plane_files)}")
print(f"Total: {len(ship_files) + len(plane_files)}")
print(f"\nImage size distribution (top 5): {sizes.most_common(5)}")
print(f"Image color mode distribution: {dict(formats)}")
print("\nBounding box / mask annotations available: NO — KLSG provides no")
print("annotation files at all. Each image is a pre-cropped chip implicitly")
print("containing one object, but with no coordinates for where it sits.")
print("Site/survey metadata available: NO")
print("Navigation metadata available: NO")