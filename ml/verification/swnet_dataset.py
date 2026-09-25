"""
PyTorch Dataset & ROI Patch Loader for SW-Net Shipwreck Verification Training.

Extracts localized target ROI patches and hard-negative seabed crops from
genuine high-resolution side-scan sonar swaths and binary segmentation masks
located in data/raw/AI4Shipwrecks/.
"""

import os
import glob
from typing import List, Tuple, Dict, Any, Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SWNetShipwreckDataset(Dataset):
    """
    Constructs localized (target ROI crop, binary mask crop) pairs from AI4Shipwrecks swaths.
    Ensures legitimate training on genuine human-annotated shipwreck ground-truth masks.
    Uses memory caching of extracted patches for ultra-fast training iterations.
    """

    def __init__(
        self,
        split: str = "train",
        base_dir: str = "data/raw/AI4Shipwrecks",
        patch_size: Tuple[int, int] = (128, 128),
        context_margin: float = 0.20,
        include_negatives: bool = True,
        negatives_per_swath: int = 2
    ):
        """
        Args:
            split: 'train' or 'test'.
            base_dir: Path to AI4Shipwrecks directory.
            patch_size: Output (H, W) for network training.
            context_margin: Fractional margin around target bounding box.
            include_negatives: Whether to extract background seabed crops.
            negatives_per_swath: Number of hard-negative seabed patches per swath.
        """
        self.split = split
        self.base_dir = base_dir
        self.patch_size = patch_size
        self.context_margin = context_margin
        self.include_negatives = include_negatives
        self.negatives_per_swath = negatives_per_swath

        self.cached_samples: List[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]] = []
        self._index_and_cache_dataset()

    def _index_and_cache_dataset(self) -> None:
        img_dir = os.path.join(self.base_dir, self.split, "images")
        lbl_dir = os.path.join(self.base_dir, self.split, "labels")

        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            return

        img_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        lbl_paths = sorted(glob.glob(os.path.join(lbl_dir, "*.*")))

        lbl_map = {
            os.path.splitext(os.path.basename(p))[0]: p
            for p in lbl_paths
        }

        for ip in img_paths:
            base = os.path.splitext(os.path.basename(ip))[0]
            lp = lbl_map.get(base)
            if not lp:
                continue

            # Read mask and image once per swath
            mask = cv2.imread(lp, cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue

            img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img_h, img_w = img.shape[:2]
            fg_pixels = np.count_nonzero(mask > 0)

            if fg_pixels > 0:
                # Find connected components of shipwreck objects
                binary_mask = (mask > 0).astype(np.uint8)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)

                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area < 50:  # Filter out tiny noise specks
                        continue

                    bx = stats[i, cv2.CC_STAT_LEFT]
                    by = stats[i, cv2.CC_STAT_TOP]
                    bw = stats[i, cv2.CC_STAT_WIDTH]
                    bh = stats[i, cv2.CC_STAT_HEIGHT]

                    # Expand bbox by context margin
                    pad_x = int(bw * self.context_margin)
                    pad_y = int(bh * self.context_margin)

                    x1 = max(0, bx - pad_x)
                    y1 = max(0, by - pad_y)
                    x2 = min(img_w, bx + bw + pad_x)
                    y2 = min(img_h, by + bh + pad_y)

                    crop_img = img[y1:y2, x1:x2]
                    crop_mask = (mask[y1:y2, x1:x2] > 0).astype(np.float32)

                    crop_img_res = cv2.resize(crop_img, self.patch_size, interpolation=cv2.INTER_LINEAR)
                    crop_mask_res = cv2.resize(crop_mask, self.patch_size, interpolation=cv2.INTER_NEAREST)

                    meta = {
                        "type": "positive",
                        "swath_name": os.path.basename(ip),
                        "original_bbox": [bx, by, bx + bw, by + bh],
                        "expanded_crop": [x1, y1, x2, y2],
                        "target_area_px": int(area)
                    }

                    self.cached_samples.append((crop_img_res, crop_mask_res, meta))
            else:
                # Negative swath (pure seabed terrain)
                if self.include_negatives:
                    np.random.seed(42 + len(self.cached_samples))
                    for n_idx in range(self.negatives_per_swath):
                        rand_x = np.random.randint(0, max(1, img_w - self.patch_size[1]))
                        rand_y = np.random.randint(0, max(1, img_h - self.patch_size[0]))
                        x2 = rand_x + self.patch_size[1]
                        y2 = rand_y + self.patch_size[0]

                        crop_img = img[rand_y:y2, rand_x:x2]
                        crop_mask = np.zeros(self.patch_size, dtype=np.float32)

                        crop_img_res = cv2.resize(crop_img, self.patch_size, interpolation=cv2.INTER_LINEAR)

                        meta = {
                            "type": "negative",
                            "swath_name": os.path.basename(ip),
                            "original_bbox": [rand_x, rand_y, x2, y2],
                            "expanded_crop": [rand_x, rand_y, x2, y2],
                            "target_area_px": 0
                        }

                        self.cached_samples.append((crop_img_res, crop_mask, meta))

    def __len__(self) -> int:
        return len(self.cached_samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        crop_img_res, crop_mask_res, metadata = self.cached_samples[idx]

        # Normalize image to [0.0, 1.0] and convert to PyTorch tensors (1, H, W)
        tensor_img = torch.from_numpy(crop_img_res.astype(np.float32) / 255.0).unsqueeze(0)
        tensor_mask = torch.from_numpy(crop_mask_res).unsqueeze(0)

        return tensor_img, tensor_mask, metadata
