"""01_inspect_sctd.py — reports SCTD dataset stats. Modifies nothing."""
import xml.etree.ElementTree as ET
import glob, os
from PIL import Image
from collections import Counter

SCTD_DIR = "SCTD/extracted/SCTD" # <-- adjust this to wherever you place SCTD locally
xmls = sorted(glob.glob(f"{SCTD_DIR}/Annotations/*.xml"))

sizes = Counter()
class_counts = Counter()
matched, unmatched = 0, 0
formats = Counter()

for xml_path in xmls:
    stem = os.path.splitext(os.path.basename(xml_path))[0]
    img_path = f"{SCTD_DIR}/JPEGImages/{stem}.jpg"
    if not os.path.exists(img_path):
        unmatched += 1
        continue
    matched += 1
    img = Image.open(img_path)
    sizes[img.size] += 1
    formats[img.mode] += 1
    tree = ET.parse(xml_path)
    for obj in tree.findall("object"):
        class_counts[obj.find("name").text] += 1

print("=== SCTD Inspection Report ===")
print(f"Total XML annotation files: {len(xmls)}")
print(f"Image-annotation matched pairs: {matched}")
print(f"Unmatched (missing image): {unmatched}")
print(f"\nImage size distribution (top 5): {sizes.most_common(5)}")
print(f"Image color mode distribution: {dict(formats)}")
print(f"\nClass distribution (raw source labels): {dict(class_counts)}")
print("\nSite/survey metadata available: NO — SCTD has no site/mission field.")
print("Navigation metadata available: NO")
print("Segmentation masks available: NO — bounding boxes only (VOC XML).")