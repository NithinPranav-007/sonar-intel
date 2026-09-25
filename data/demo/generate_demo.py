"""
Synthetic Side-Scan Sonar (SSS) Demo Generator.

Generates realistic acoustic backscatter imagery and matching navigation track
with acoustic shadows, highlights, and nadir water columns.
"""

import os
import numpy as np
import cv2
import pandas as pd


def generate_demo_sonar_and_nav():
    os.makedirs("data/demo/sonar", exist_ok=True)
    os.makedirs("data/demo/navigation", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)

    height = 1800
    width = 1280
    mid_x = width // 2

    # 1. Base acoustic seabed texture (speckle + ambient backscatter)
    np.random.seed(42)
    base_intensity = 110
    seabed = np.random.normal(base_intensity, 18, (height, width)).astype(np.float32)

    # Add gentle along-track acoustic ripple patterns
    y_coords, x_coords = np.indices((height, width))
    ripples = 12 * np.sin(x_coords / 45.0) * np.cos(y_coords / 60.0)
    seabed += ripples

    # 2. Central Nadir / Water-Column blind zone
    # Dark strip in center where acoustic pulse traverses water before hitting bottom
    nadir_half_width = 45
    nadir_mask = (x_coords >= mid_x - nadir_half_width) & (x_coords <= mid_x + nadir_half_width)
    seabed[nadir_mask] = np.random.normal(25, 8, np.count_nonzero(nadir_mask))

    # Bright First Bottom Return (FBR) lines at edges of nadir
    fbr_left = (x_coords >= mid_x - nadir_half_width - 4) & (x_coords <= mid_x - nadir_half_width)
    fbr_right = (x_coords >= mid_x + nadir_half_width) & (x_coords <= mid_x + nadir_half_width + 4)
    seabed[fbr_left] = np.random.normal(210, 15, np.count_nonzero(fbr_left))
    seabed[fbr_right] = np.random.normal(210, 15, np.count_nonzero(fbr_right))

    # 3. Inject Target 1 (C001: Strong metallic object / container)
    # Left swath: (x=420, y=240). Highlight is bright, shadow is cast to the left (-X away from nadir)
    t1_y1, t1_y2 = 200, 280
    t1_x1, t1_x2 = 440, 520
    # Highlight
    seabed[t1_y1:t1_y2, t1_x1:t1_x2] = np.random.normal(245, 10, (t1_y2 - t1_y1, t1_x2 - t1_x1))
    # Acoustic Shadow behind highlight (away from nadir -> x < 440)
    sh1_x1, sh1_x2 = 320, 440
    seabed[t1_y1:t1_y2, sh1_x1:sh1_x2] = np.random.normal(12, 4, (t1_y2 - t1_y1, sh1_x2 - sh1_x1))

    # 4. Inject Target 2 (C002: Abandoned Net / entangled debris)
    # Right swath: (x=800, y=600). Highlight is bright, shadow is cast to the right (+X away from nadir)
    t2_y1, t2_y2 = 540, 660
    t2_x1, t2_x2 = 800, 880
    # High irregular backscatter
    net_texture = np.random.normal(230, 25, (t2_y2 - t2_y1, t2_x2 - t2_x1))
    seabed[t2_y1:t2_y2, t2_x1:t2_x2] = np.clip(net_texture, 0, 255)
    # Shadow to the right (+X)
    sh2_x1, sh2_x2 = 880, 990
    seabed[t2_y1:t2_y2, sh2_x1:sh2_x2] = np.random.normal(18, 5, (t2_y2 - t2_y1, sh2_x2 - sh2_x1))

    # 5. Inject Target 3 (C003: Subtle low-profile anomaly)
    t3_y1, t3_y2 = 1140, 1190
    t3_x1, t3_x2 = 230, 290
    seabed[t3_y1:t3_y2, t3_x1:t3_x2] = np.random.normal(195, 15, (t3_y2 - t3_y1, t3_x2 - t3_x1))
    sh3_x1, sh3_x2 = 180, 230
    seabed[t3_y1:t3_y2, sh3_x1:sh3_x2] = np.random.normal(45, 8, (t3_y2 - t3_y1, sh3_x2 - sh3_x1))

    # Clip and save image
    final_img = np.clip(seabed, 0, 255).astype(np.uint8)
    # Give it realistic sonar sepia/copper or grayscale tint
    sonar_path = "data/demo/sonar/survey_001_raw.png"
    cv2.imwrite(sonar_path, final_img)
    # Also place a copy in data/raw for immediate upload tests
    cv2.imwrite("data/raw/SURVEY_001_raw.png", final_img)
    print(f"Generated synthetic sonar waterfall at {sonar_path}")

    # Generate Navigation Track CSV
    pings = 150
    nav_rows = []
    base_lat = 11.23400
    base_lon = 76.54300
    for p in range(pings):
        lat = base_lat + (p * 0.000015)
        lon = base_lon + (p * 0.000010)
        nav_rows.append({
            "ping_id": p + 1,
            "timestamp": f"2026-08-31T14:{p//60:02d}:{p%60:02d}Z",
            "latitude": round(lat, 6),
            "longitude": round(lon, 6),
            "heading": 34.5,
            "altitude": 10.2,
            "range": 50.0
        })

    nav_df = pd.DataFrame(nav_rows)
    nav_path = "data/demo/navigation/survey_001_nav.csv"
    nav_df.to_csv(nav_path, index=False)
    nav_df.to_csv("data/raw/SURVEY_001_nav.csv", index=False)
    print(f"Generated synthetic navigation CSV at {nav_path}")


if __name__ == "__main__":
    generate_demo_sonar_and_nav()
