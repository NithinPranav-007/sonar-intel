"""03_normalize_klsg.py — applies the team's confirmed 1-99 percentile normalization to KLSG."""
import glob, os
import numpy as np
import cv2

KLSG_DIR = "KLSG/extracted"
OUT_DIR = "data/interim/klsg_normalized"
os.makedirs(OUT_DIR, exist_ok=True)

def apply_percentile_normalization(image, p_low=1.0, p_high=99.0):
    img_float = image.astype(np.float32)
    val_low, val_high = np.percentile(img_float, p_low), np.percentile(img_float, p_high)
    if val_high <= val_low:
        return image.copy()
    stretched = (img_float - val_low) / (val_high - val_low) * 255.0
    return np.clip(stretched, 0, 255).astype(np.uint8)

ship_files = sorted(set(glob.glob(f"{KLSG_DIR}/ship*.png")))
plane_files = sorted(set(glob.glob(f"{KLSG_DIR}/plane-real/*.png")))

count = 0
for img_path in ship_files + plane_files:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    norm_img = apply_percentile_normalization(img)
    out_name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(OUT_DIR, out_name), norm_img)
    count += 1

print("=== KLSG Normalization ===")
print(f"Normalized {count} images using 1-99 percentile stretch")
print(f"Output: {OUT_DIR}")