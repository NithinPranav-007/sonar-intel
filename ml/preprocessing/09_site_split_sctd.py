"""09_site_split_sctd.py — splits SCTD tiles by SOURCE IMAGE (not by site, since SCTD has
no site/mission metadata), preventing tiles from the same original photo leaking across
train/val/test."""
import csv, random
from collections import defaultdict, Counter

random.seed(42)

manifest_path = "outputs/sctd_tile_manifest.csv"
rows = list(csv.DictReader(open(manifest_path)))

by_source = defaultdict(list)
for r in rows:
    by_source[r["source_image"]].append(r)

sources = list(by_source.keys())
random.shuffle(sources)

n = len(sources)
train_cut, val_cut = int(n * 0.7), int(n * 0.85)
train_src = set(sources[:train_cut])
val_src = set(sources[train_cut:val_cut])

for r in rows:
    if r["source_image"] in train_src:
        r["split"] = "train"
    elif r["source_image"] in val_src:
        r["split"] = "val"
    else:
        r["split"] = "test"

out = "outputs/sctd_final_manifest.csv"
with open(out, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print("=== SCTD Split (by source image) ===")
print(f"Source images: train={len(train_src)}, val={len(val_src)}, test={n - len(train_src) - len(val_src)}")
print(f"Tiles per split: {dict(Counter(r['split'] for r in rows))}")
print(f"Final manifest: {out}")