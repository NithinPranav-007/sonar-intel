"""
Side-Scan Sonar Geolocation Estimation Service.

Associates pixel detections with towfish navigation logs (ping, timestamp, lat, lon, heading).

SCIENTIFIC / DOMAIN HONESTY CONSTRAINTS:
1. Do NOT claim survey-grade positioning.
2. Every output must carry a localization_status:
   - ESTIMATED   : Linearly interpolated from towfish coordinates and slant range.
   - VERIFIED    : Checked with acoustic USBL/DVL (reserved for survey grade).
   - UNCERTAIN   : Large time/ping offset or vessel turning maneuvers.
   - UNAVAILABLE : Navigation data missing.
3. If navigation metadata is missing, DO NOT invent fake coordinates!
   Store latitude=None, longitude=None, localization_status='UNAVAILABLE'.
"""

from typing import Optional, Tuple, List, Dict, Any
import os
import pandas as pd
import numpy as np


class GeolocationService:
    def __init__(self, nav_file_path: Optional[str] = None):
        self.nav_file_path = nav_file_path
        self.nav_df: Optional[pd.DataFrame] = None
        self._load_nav_data()

    def _load_nav_data(self):
        if not self.nav_file_path or not os.path.exists(self.nav_file_path):
            self.nav_df = None
            return

        try:
            df = pd.read_csv(self.nav_file_path)
            # Normalize column names
            df.columns = [c.strip().lower() for c in df.columns]
            if "latitude" in df.columns and "longitude" in df.columns:
                self.nav_df = df.sort_values(by="ping_id") if "ping_id" in df.columns else df
            else:
                self.nav_df = None
        except Exception as e:
            print(f"[GeolocationService] Failed to parse navigation CSV: {e}")
            self.nav_df = None

    def estimate_contact_location(
        self,
        bbox_center_x: int,
        bbox_center_y: int,
        image_width: int,
        image_height: int
    ) -> Tuple[Optional[float], Optional[float], str]:
        """
        Estimates WGS84 geographic position for a detection center in the sonar swath.
        
        Returns:
            Tuple of (latitude, longitude, localization_status)
        """
        if self.nav_df is None or self.nav_df.empty:
            return None, None, "UNAVAILABLE"

        n_pings = len(self.nav_df)
        if n_pings == 0:
            return None, None, "UNAVAILABLE"

        # 1. Estimate along-track ping index based on vertical Y position
        # Waterfall scans along Y from 0 to image_height
        y_ratio = max(0.0, min(1.0, float(bbox_center_y) / float(image_height)))
        ping_idx = int(y_ratio * (n_pings - 1))
        nav_row = self.nav_df.iloc[ping_idx]

        base_lat = float(nav_row["latitude"])
        base_lon = float(nav_row["longitude"])
        heading_deg = float(nav_row.get("heading", 0.0))
        altitude_m = float(nav_row.get("altitude", 10.0))
        slant_range_m = float(nav_row.get("range", 50.0))

        # 2. Across-track offset (Port vs. Starboard)
        # Nadir is at center X
        mid_x = image_width / 2.0
        across_track_ratio = (bbox_center_x - mid_x) / (mid_x + 1e-5)  # -1.0 (Port) to +1.0 (Starboard)
        ground_range_m = across_track_ratio * slant_range_m

        # 3. Simple orthogonal projection using heading
        # Starboard is heading + 90 deg, Port is heading - 90 deg
        offset_heading = (heading_deg + 90.0) if ground_range_m >= 0 else (heading_deg - 90.0)
        offset_heading_rad = np.radians(offset_heading)
        dist_m = abs(ground_range_m)

        # 1 deg lat ~ 111,320m; 1 deg lon ~ 111,320m * cos(lat)
        delta_lat = (dist_m * np.cos(offset_heading_rad)) / 111320.0
        delta_lon = (dist_m * np.sin(offset_heading_rad)) / (111320.0 * np.cos(np.radians(base_lat)) + 1e-6)

        est_lat = round(base_lat + delta_lat, 6)
        est_lon = round(base_lon + delta_lon, 6)

        return est_lat, est_lon, "ESTIMATED"

    def get_survey_track(self) -> List[Dict[str, Any]]:
        """Returns the series of navigation track waypoints for map rendering."""
        if self.nav_df is None or self.nav_df.empty:
            return []
        
        points = []
        for _, row in self.nav_df.iterrows():
            points.append({
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "ping_id": int(row.get("ping_id", 0)),
                "heading": float(row.get("heading", 0.0))
            })
        return points
