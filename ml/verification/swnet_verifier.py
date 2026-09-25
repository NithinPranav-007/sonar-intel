"""
SW-Net Second-Stage Verification and Structural Segmentation Interface.

Provides a model-agnostic verification interface that ingests candidate ROIs,
computes fine-grained acoustic segmentation, and extracts morphological physical priors:
- Directional orientation
- Elongation (aspect ratio along principal axis)
- Compactness (isoperimetric quotient)
- Mask-to-bounding-box area ratio
- Acoustic highlight / shadow contrast

Attribution:
SW-Net architecture based on Dai & He (2026):
"SW-Net: A Lightweight Direction-Aware Semantic Segmentation Network for Shipwrecks in Side-Scan Sonar Imagery"
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass, asdict
import os
import cv2
import numpy as np

from ml.verification.roi_extractor import ExtractedROI, ROIExtractor


@dataclass
class VerificationResult:
    """Structured result returned by second-stage verification."""
    is_verified: bool                     # Whether candidate passes structural/segmentation verification
    verification_confidence: float        # Verification confidence score [0.0, 1.0]
    mask_area_px: int                     # Total positive segmented pixel count
    mask_to_box_ratio: float              # Area of segmented mask divided by candidate bounding box area
    orientation_deg: float                # Principal axis orientation in degrees [-90.0, 90.0]
    elongation: float                     # Principal axis length / minor axis length ratio (>= 1.0)
    compactness: float                    # Isoperimetric quotient 4 * pi * Area / Perimeter^2 [0.0, 1.0]
    highlight_mean: float                 # Mean intensity of acoustic highlight region [0, 255]
    shadow_mean: float                    # Mean intensity of acoustic shadow region [0, 255]
    contrast_ratio: float                 # Highlight intensity divided by shadow intensity
    model_name: str                       # Name of verification model ("SW-Net")
    model_version: str                    # Version string ("swnet-v1.0")
    status: str                           # Detailed status string
    segmentation_mask: Optional[np.ndarray] = None  # 2D binary uint8 mask (H, W) or None

    def to_dict(self) -> Dict[str, Any]:
        """Converts result to serializable dict, omitting raw mask array."""
        d = asdict(self)
        d.pop("segmentation_mask", None)
        return d


class BaseROIVerifier(ABC):
    """Abstract base class for all candidate ROI verification models."""

    @abstractmethod
    def verify_roi(
        self,
        roi: Union[ExtractedROI, np.ndarray],
        bbox: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> VerificationResult:
        """
        Executes second-stage verification on a candidate ROI.

        Args:
            roi: ExtractedROI object or raw image array.
            bbox: Optional bounding box if roi is raw array.
            metadata: Optional upstream detector candidate metadata.

        Returns:
            VerificationResult containing segmentation metrics and verification flag.
        """
        pass


def compute_morphological_metrics(
    binary_mask: np.ndarray,
    roi_image: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Extracts geometric and acoustic morphological physical priors from a binary mask.

    Args:
        binary_mask: 2D uint8 array where 255 indicates foreground target.
        roi_image: Optional 2D/3D sonar crop for computing highlight/shadow metrics.

    Returns:
        Dictionary of computed morphological metrics.
    """
    if binary_mask is None or binary_mask.size == 0 or np.count_nonzero(binary_mask) == 0:
        return {
            "mask_area_px": 0,
            "orientation_deg": 0.0,
            "elongation": 1.0,
            "compactness": 0.0,
            "highlight_mean": 0.0,
            "shadow_mean": 0.0,
            "contrast_ratio": 1.0
        }

    mask = (binary_mask > 0).astype(np.uint8)
    area = int(np.sum(mask))

    # 1. Find contours to compute perimeter and bounding ellipse
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "mask_area_px": area,
            "orientation_deg": 0.0,
            "elongation": 1.0,
            "compactness": 0.0,
            "highlight_mean": 0.0,
            "shadow_mean": 0.0,
            "contrast_ratio": 1.0
        }

    # Use largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    perimeter = cv2.arcLength(largest_contour, True)

    # Compactness: 4 * pi * Area / Perimeter^2 (1.0 for perfect circle)
    if perimeter > 0:
        compactness = float((4.0 * np.pi * area) / (perimeter ** 2))
        compactness = min(1.0, max(0.0, compactness))
    else:
        compactness = 0.0

    # 2. Moments & Orientation via Inertia Matrix
    moments = cv2.moments(largest_contour)
    if moments["m00"] != 0:
        # Central moments
        mu20 = moments["mu20"] / moments["m00"]
        mu02 = moments["mu02"] / moments["m00"]
        mu11 = moments["mu11"] / moments["m00"]

        # Orientation angle in radians / degrees
        theta = 0.5 * np.arctan2(2.0 * mu11, (mu20 - mu02))
        orientation_deg = float(np.degrees(theta))

        # Eigenvalues of inertia tensor for elongation
        common = np.sqrt(4.0 * (mu11 ** 2) + ((mu20 - mu02) ** 2))
        major_axis = np.sqrt(max(1e-6, 2.0 * (mu20 + mu02 + common)))
        minor_axis = np.sqrt(max(1e-6, 2.0 * (mu20 + mu02 - common)))
        elongation = float(major_axis / max(1e-6, minor_axis))
    else:
        orientation_deg = 0.0
        elongation = 1.0

    # 3. Acoustic highlight & shadow intensity computation
    highlight_mean = 0.0
    shadow_mean = 0.0
    contrast_ratio = 1.0

    if roi_image is not None:
        if len(roi_image.shape) == 3:
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_image

        target_pixels = gray[mask > 0]
        bg_pixels = gray[mask == 0]

        if target_pixels.size > 0:
            highlight_mean = float(np.mean(target_pixels))
        if bg_pixels.size > 0:
            shadow_mean = float(np.mean(bg_pixels))

        if shadow_mean > 0:
            contrast_ratio = float(highlight_mean / shadow_mean)
        else:
            contrast_ratio = float(highlight_mean / 1.0)

    return {
        "mask_area_px": area,
        "orientation_deg": round(orientation_deg, 2),
        "elongation": round(elongation, 2),
        "compactness": round(compactness, 4),
        "highlight_mean": round(highlight_mean, 2),
        "shadow_mean": round(shadow_mean, 2),
        "contrast_ratio": round(contrast_ratio, 2)
    }


class SWNetVerifier(BaseROIVerifier):
    """
    SW-Net: Direction-Aware Semantic Segmentation Verifier for SSS Targets.

    Functions as the Stage-2 verification module downstream of YOLO candidate proposal.
    Evaluates candidate structural integrity, eliminating rectangular false alarms and
    non-directional geological clutter.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.50,
        min_mask_ratio: float = 0.05,
        max_mask_ratio: float = 1.00,
        min_compactness: float = 0.02,
        device: Optional[str] = None
    ):
        """
        Args:
            model_path: Path to trained SW-Net PyTorch checkpoint (.pt / .pth).
            confidence_threshold: Threshold for pixel-level semantic classification.
            min_mask_ratio: Minimum ratio of mask area to candidate bounding box area.
            max_mask_ratio: Maximum ratio of mask area to candidate bounding box area.
            min_compactness: Minimum isoperimetric compactness score.
            device: 'cuda' or 'cpu'.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.min_compactness = min_compactness
        self.device = device or ("cuda:0" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")
        self.model_name = "SW-Net"
        self.model_version = "swnet-direction-aware-v1.0"
        self.model = None

        self._initialize_model()

    def _initialize_model(self):
        """Attempts to load trained SW-Net model weights if available."""
        target_path = self.model_path
        if target_path is None:
            default_best = "outputs/models/swnet/best_swnet.pt"
            if os.path.exists(default_best):
                target_path = default_best

        if target_path and os.path.exists(target_path):
            try:
                import torch
                from ml.verification.swnet_train import SWNetArchitecture
                print(f"[SWNetVerifier] Loading SW-Net checkpoint from {target_path} onto {self.device}...")
                ckpt = torch.load(target_path, map_location=self.device)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    self.model = SWNetArchitecture(in_channels=1, num_classes=1).to(self.device)
                    self.model.load_state_dict(ckpt["model_state_dict"])
                else:
                    self.model = ckpt

                if hasattr(self.model, "eval"):
                    self.model.eval()
                self.model_path = target_path
                self.model_version = "swnet-direction-aware-v1.0"
            except Exception as e:
                print(f"[SWNetVerifier] Warning: Failed to load checkpoint {target_path}: {e}")
                self.model = None
        else:
            self.model = None

    def verify_roi(
        self,
        roi: Union[ExtractedROI, np.ndarray],
        bbox: Optional[List[int]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> VerificationResult:
        """
        Runs SW-Net directional segmentation and physical prior verification on candidate ROI.
        """
        # 1. Resolve ExtractedROI
        if isinstance(roi, ExtractedROI):
            crop_img = roi.roi_image
            target_bbox = roi.bbox_local
            meta = {**roi.metadata, **(metadata or {})}
        elif isinstance(roi, np.ndarray):
            crop_img = roi
            if bbox is not None:
                target_bbox = bbox
            else:
                target_bbox = [0, 0, roi.shape[1], roi.shape[0]]
            meta = metadata or {}
        else:
            raise ValueError(f"Invalid ROI input type: {type(roi)}")

        if crop_img is None or crop_img.size == 0:
            raise ValueError("ROI crop image is empty.")

        crop_h, crop_w = crop_img.shape[:2]
        box_area = max(1, (target_bbox[2] - target_bbox[0]) * (target_bbox[3] - target_bbox[1]))

        # 2. Segmentation Inference
        if self.model is not None:
            # Model-driven inference
            mask, seg_conf = self._infer_network(crop_img)
            status_prefix = "MODEL_INFERRED"
        else:
            # Pre-training structural contour verification fallback
            mask, seg_conf = self._heuristic_contour_segmentation(crop_img, target_bbox)
            status_prefix = "STRUCTURAL_HEURISTIC_PREVIEW"

        # 3. Morphological & Physical Prior Extraction
        metrics = compute_morphological_metrics(mask, crop_img)
        mask_area = metrics["mask_area_px"]
        mask_to_box = mask_area / float(box_area)

        # 4. Decision Logic: Verify Candidate
        is_verified = True
        status_reasons = []

        if mask_area == 0:
            is_verified = False
            status_reasons.append("ZERO_MASK_AREA")
        elif mask_to_box < self.min_mask_ratio:
            is_verified = False
            status_reasons.append(f"UNDERFILLED_BOX_RATIO_{mask_to_box:.2f}")
        elif mask_to_box > self.max_mask_ratio:
            is_verified = False
            status_reasons.append(f"OVERFILLED_BACKGROUND_RATIO_{mask_to_box:.2f}")

        if metrics["compactness"] < self.min_compactness:
            is_verified = False
            status_reasons.append(f"LOW_COMPACTNESS_{metrics['compactness']:.4f}")

        final_status = f"{status_prefix}: " + ("; ".join(status_reasons) if status_reasons else "VERIFIED_VALID_STRUCTURE")

        return VerificationResult(
            is_verified=is_verified,
            verification_confidence=round(seg_conf, 3),
            mask_area_px=mask_area,
            mask_to_box_ratio=round(mask_to_box, 3),
            orientation_deg=metrics["orientation_deg"],
            elongation=metrics["elongation"],
            compactness=metrics["compactness"],
            highlight_mean=metrics["highlight_mean"],
            shadow_mean=metrics["shadow_mean"],
            contrast_ratio=metrics["contrast_ratio"],
            model_name=self.model_name,
            model_version=self.model_version,
            status=final_status,
            segmentation_mask=mask
        )

    def _infer_network(self, crop_img: np.ndarray) -> Tuple[np.ndarray, float]:
        """Executes forward pass through PyTorch SW-Net segmentation model."""
        import torch
        orig_h, orig_w = crop_img.shape[:2]
        if len(crop_img.shape) == 3:
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop_img

        # Resize to standard network patch size
        patch_res = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_LINEAR)
        norm = (patch_res.astype(np.float32) / 255.0)
        tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tensor)
            if isinstance(output, tuple):
                output = output[0]
            prob = torch.sigmoid(output).squeeze().cpu().numpy()

        # Resize probability mask back to original crop resolution
        prob_orig = cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        binary_mask = (prob_orig >= self.confidence_threshold).astype(np.uint8) * 255
        mean_conf = float(np.mean(prob_orig[prob_orig >= self.confidence_threshold])) if np.any(prob_orig >= self.confidence_threshold) else 0.0
        return binary_mask, mean_conf

    def _heuristic_contour_segmentation(
        self,
        crop_img: np.ndarray,
        target_bbox: List[int]
    ) -> Tuple[np.ndarray, float]:
        """
        Otsu / adaptive acoustic contour segmentation within candidate bbox
        used as an unweighted structural baseline prior to network training.
        """
        if len(crop_img.shape) == 3:
            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop_img.copy()

        h, w = gray.shape[:2]
        bx1, by1, bx2, by2 = target_bbox
        bx1, bx2 = max(0, bx1), min(w, bx2)
        by1, by2 = max(0, by1), min(h, by2)

        # Focus segmentation inside candidate bounding box
        roi_patch = gray[by1:by2, bx1:bx2]
        if roi_patch.size == 0:
            return np.zeros((h, w), dtype=np.uint8), 0.0

        # Otsu thresholding on localized ROI patch
        blur = cv2.GaussianBlur(roi_patch, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Place patch mask in full crop mask
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[by1:by2, bx1:bx2] = thresh

        conf = float(np.mean(roi_patch[thresh > 0]) / 255.0) if np.any(thresh > 0) else 0.5
        return full_mask, min(1.0, max(0.0, conf))
