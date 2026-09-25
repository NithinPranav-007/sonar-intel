# SCTD Dataset — Provenance, Validation, and Reproduction

## Provenance
- Source: https://github.com/MingqiangNing/SCTD
- License: Freely downloadable, no formal open-source license file; citation requested by authors
- Acquired: [today's date]
- Original format: Pascal VOC XML annotations, JPEG images

## Dataset Qualification
- DISCOVERED -> PROVENANCE VERIFIED -> LICENSE VERIFIED (citation-only, not formally licensed) -> DATA FORMAT VERIFIED (VOC XML) -> ANNOTATIONS VERIFIED -> QUALITY VERIFIED -> APPROVED (with caveats noted below)

## Validation Summary
- Total source images: 357
- Quality check: 356 VALID, 1 SUSPICIOUS, 0 INVALID
- Raw source classes: ship (271), aircraft (57), human (35)
- Mapped to single canonical class (0 = artificial_anomaly), per repo convention

## Known Limitations
- No site/survey/mission metadata available — leakage-safe split performed
  at the source-image level instead of site level.
- No navigation/GPS metadata available.
- No segmentation masks available — bounding boxes only.
- No hard-negative (natural clutter, no real object) examples present in
  this dataset — every image contains at least one labeled object.
- Source "ship" label is not distinguished as wreck vs. active vessel;
  treated as-is under the single-class scheme.

## Preprocessing Applied
1. Quality check (corruption, dimension, saturation, box-bounds checks)
2. 1st-99th percentile intensity normalization (matches team's validated
   method from 03_normalize.py experiment on AI4Shipwrecks)
3. CLAHE: skipped (matches team's tested decision — no default benefit)
4. Denoising: skipped (matches team's tested decision — pattern not present)
5. Nadir masking: skipped — SCTD images are small pre-cropped chips, not
   continuous swaths; no meaningful nadir strip present
6. Tiling: 640x640, 20% overlap, 512px stride where image >= 640px in a
   dimension; zero-padding to 640x640 otherwise
7. Bounding boxes re-clipped and converted directly to YOLO format per tile
   (no mask-to-box step needed, since SCTD provides boxes natively)
8. Split by source image (train/val/test), preventing tile-level leakage

## Results
- Tiles produced: 398
- Split: 282 train / 60 val / 56 test (by source image count: 249/54/54)

## Reproduction
Scripts, in order, located in ml/preprocessing/:
1. 01_inspect_sctd.py
2. 02_quality_check_sctd.py
3. 03_normalize_sctd.py
4. 07_tile_sctd.py
5. 09_site_split_sctd.py

Run each with: `python ml\preprocessing\<script_name>.py` from the project
root, after cloning SCTD into `SCTD/extracted/SCTD/` (see script comments
for exact expected path).