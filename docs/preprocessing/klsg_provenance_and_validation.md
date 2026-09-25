# KLSG Dataset — Provenance, Validation, and Reproduction

## Provenance
- Source: https://github.com/huoguanying/SeabedObjects-Ship-and-Airplane-dataset
- License: "Currently open for academic use" per author's README; no formal
  open-source license file
- Acquired: [today's date]
- Original format: flat PNG images, no annotation files of any kind
- README states 385 ship + 62 plane; actual files found: 380 ship + 48 plane
  (11 fewer than documented — discrepancy in the source repo itself, not
  introduced by this pipeline)

## Dataset Qualification
- DISCOVERED -> PROVENANCE VERIFIED -> LICENSE VERIFIED (academic-use-only,
  not formally licensed) -> DATA FORMAT VERIFIED (flat PNG, grayscale) ->
  ANNOTATIONS VERIFIED (NONE EXIST — see limitation below) -> QUALITY
  VERIFIED -> APPROVED WITH MAJOR CAVEAT (weak labels only)

## Validation Summary
- Total source images: 428 (380 ship, 48 plane)
- Quality check: 425 VALID, 3 SUSPICIOUS, 0 INVALID
- Mapped to single canonical class (0 = artificial_anomaly), per repo convention

## Known Limitations — READ BEFORE USING FOR REPORTED ACCURACY
- **No bounding box or mask annotations exist in this dataset at all.**
  Every label produced for KLSG is a WEAK, approximated box covering the
  central 80% of each image — not a verified annotation.
- Every KLSG tile is flagged `weak_label=True` in its manifest row. This
  must not be blended into accuracy claims alongside SCTD or AI4Shipwrecks
  without disclosing the difference in label quality.
- No site/survey/mission metadata available — split performed at the
  source-image level instead.
- No navigation/GPS metadata available.
- No hard-negative (natural clutter, no object) examples present — every
  image contains exactly one implied object.
- Ship vs. plane is the only distinction in filenames; no further
  sub-classification (e.g., wreck vs. active vessel) is possible.

## Preprocessing Applied
1. Quality check (corruption, dimension, saturation checks)
2. 1st-99th percentile intensity normalization (same validated method as
   AI4Shipwrecks and SCTD)
3. CLAHE: skipped (reused team's AI4Shipwrecks ablation finding)
4. Denoising: skipped (reused team's AI4Shipwrecks ablation finding)
5. Nadir masking: skipped — pre-cropped chips, not continuous swaths
6. Tiling: 640x640, 20% overlap/512px stride where image >= 640px in a
   dimension; zero-padding otherwise
7. Weak bounding box generated per tile (central 80% region) — NOT a real
   annotation, explicitly flagged
8. Split by source image (train/val/test), preventing tile-level leakage

## Results
- Tiles produced: 474
- Split: 340 train / 65 val / 69 test (by source image count: 299/64/65)

## Reproduction
Scripts, in order, located in ml/preprocessing/:
1. 01_inspect_klsg.py
2. 02_quality_check_klsg.py
3. 03_normalize_klsg.py
4. 07_tile_klsg.py
5. 09_site_split_klsg.py

Run each with: `python ml\preprocessing\<script_name>.py` from the project
root, after cloning KLSG into `KLSG/extracted/` (all four source zips
extracted into that one folder — see script comments for exact path).