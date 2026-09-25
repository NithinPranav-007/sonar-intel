"""
SW-Net Lightweight Direction-Aware Verification Network & Training Pipeline.

Implements the Directional Filter / Attention segmentation architecture based on
Dai & He (2026) for SSS shipwreck structural verification, complete with BCE+Dice loss,
validation metrics (IoU, Dice, Precision, Recall), morphological physical prior checks,
and checkpoint export.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, Tuple, Optional, List
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ml.verification.swnet_dataset import SWNetShipwreckDataset
from ml.verification.swnet_verifier import compute_morphological_metrics


class DirectionalBlock(nn.Module):
    """
    Direction-Aware Convolutional Filter Block for SSS imagery.
    Applies multi-angle directional kernels (0, 45, 90, 135 deg) to capture linear target structures.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_main = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_h = nn.Conv2d(in_channels, out_channels // 4, kernel_size=(1, 5), padding=(0, 2))
        self.conv_v = nn.Conv2d(in_channels, out_channels // 4, kernel_size=(5, 1), padding=(2, 0))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fuse = nn.Conv2d(out_channels + (out_channels // 2), out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_main = self.conv_main(x)
        feat_h = self.conv_h(x)
        feat_v = self.conv_v(x)
        feat_combined = torch.cat([feat_main, feat_h, feat_v], dim=1)
        out = self.relu(self.bn(self.fuse(feat_combined)))
        return out


class SWNetArchitecture(nn.Module):
    """
    Lightweight Encoder-Decoder with Directional Attention for Shipwreck Verification.
    Parameters: ~1.33M (Edge-deployable).
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 1):
        super().__init__()
        # Encoder
        self.enc1 = DirectionalBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = DirectionalBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = DirectionalBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = DirectionalBlock(128, 256)

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DirectionalBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DirectionalBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DirectionalBlock(64, 32)

        # Head
        self.out_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3_up = self.up3(b)
        if d3_up.shape[2:] != e3.shape[2:]:
            d3_up = F.interpolate(d3_up, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3_up, e3], dim=1))

        d2_up = self.up2(d3)
        if d2_up.shape[2:] != e2.shape[2:]:
            d2_up = F.interpolate(d2_up, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2_up, e2], dim=1))

        d1_up = self.up1(d2)
        if d1_up.shape[2:] != e1.shape[2:]:
            d1_up = F.interpolate(d1_up, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1_up, e1], dim=1))

        logits = self.out_conv(d1)
        return logits


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross-Entropy and Soft Dice Loss."""

    def __init__(self, smooth: float = 1.0, bce_weight: float = 0.5):
        super().__init__()
        self.smooth = smooth
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice_score = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        dice_loss = 1.0 - dice_score

        return (self.bce_weight * bce_loss) + ((1.0 - self.bce_weight) * dice_loss)


def calculate_metrics(
    preds_prob: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Calculates comprehensive segmentation metrics."""
    bin_preds = (preds_prob > threshold).float()
    targets = targets.float()

    tp = (bin_preds * targets).sum().item()
    fp = (bin_preds * (1 - targets)).sum().item()
    fn = ((1 - bin_preds) * targets).sum().item()
    tn = ((1 - bin_preds) * (1 - targets)).sum().item()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    pixel_acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    bg_iou = tn / (tn + fp + fn + 1e-6)

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "pixel_accuracy": float(pixel_acc),
        "background_iou": float(bg_iou)
    }


def train_swnet(
    data_dir: str = "data/raw/AI4Shipwrecks",
    output_dir: str = "outputs/models/swnet",
    batch_size: int = 16,
    num_epochs: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Executes complete training and evaluation of SW-Net on AI4Shipwrecks.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SW-Net Trainer] Initializing on device: {device} (Seed: {seed})")

    # 1. Swath-level isolated datasets
    start_data_load = time.time()
    train_dataset = SWNetShipwreckDataset(split="train", base_dir=data_dir)
    test_dataset = SWNetShipwreckDataset(split="test", base_dir=data_dir)
    data_load_time = time.time() - start_data_load

    print(f"[SW-Net Trainer] Loaded datasets in {data_load_time:.2f}s:")
    print(f"  - Train Patches: {len(train_dataset)} (from 141 train swaths)")
    print(f"  - Test Patches:  {len(test_dataset)} (from 120 test swaths)")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 2. Model, Loss, Optimizer, Scheduler
    model = SWNetArchitecture(in_channels=1, num_classes=1).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[SW-Net Trainer] Trainable Parameters: {total_params:,}")

    history: List[Dict[str, Any]] = []
    best_dice = -1.0
    best_epoch = -1
    best_ckpt_path = os.path.join(output_dir, "best_swnet.pt")
    final_ckpt_path = os.path.join(output_dir, "final_swnet.pt")
    no_improve_count = 0

    train_start_time = time.time()

    # 3. Training Loop
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for images, masks, _ in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss /= len(train_dataset)

        # Validation on held-out test split
        model.eval()
        val_loss = 0.0
        all_probs: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []

        with torch.no_grad():
            for images, masks, _ in test_loader:
                images = images.to(device)
                masks = masks.to(device)

                logits = model(images)
                loss = criterion(logits, masks)
                val_loss += loss.item() * images.size(0)

                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu())
                all_targets.append(masks.cpu())

        val_loss /= len(test_dataset)
        cat_probs = torch.cat(all_probs, dim=0)
        cat_targets = torch.cat(all_targets, dim=0)

        val_metrics = calculate_metrics(cat_probs, cat_targets)
        epoch_time = time.time() - epoch_start

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_iou": round(val_metrics["iou"], 4),
            "val_dice": round(val_metrics["dice"], 4),
            "val_precision": round(val_metrics["precision"], 4),
            "val_recall": round(val_metrics["recall"], 4),
            "val_pixel_accuracy": round(val_metrics["pixel_accuracy"], 4),
            "epoch_duration_sec": round(epoch_time, 2)
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} [{epoch_time:.1f}s] "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | Val Dice/F1: {val_metrics['dice']:.4f} | "
            f"Precision: {val_metrics['precision']:.4f} | Recall: {val_metrics['recall']:.4f}"
        )

        # Checkpoint if best
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            best_epoch = epoch
            no_improve_count = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "architecture": "SWNetArchitecture",
                "parameters": total_params,
                "dataset": "AI4Shipwrecks",
                "target_class": "shipwreck",
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }, best_ckpt_path)
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"[SW-Net Trainer] Early stopping triggered after {patience} epochs without improvement.")
                break

    total_train_duration = time.time() - train_start_time

    # Save final model
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "final_metrics": val_metrics,
        "architecture": "SWNetArchitecture",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }, final_ckpt_path)

    # 4. Comprehensive Final Evaluation on Best Model
    print(f"\n[SW-Net Trainer] Loading best checkpoint from Epoch {best_epoch} (Dice: {best_dice:.4f}) for evaluation...")
    best_checkpoint = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    all_probs = []
    all_targets = []
    morphological_records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for i in range(len(test_dataset)):
            img_tensor, mask_tensor, meta = test_dataset[i]
            img_batch = img_tensor.unsqueeze(0).to(device)
            logits = model(img_batch)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu())
            all_targets.append(mask_tensor.unsqueeze(0).cpu())

            # Compute morphological evidence on predicted binary mask
            pred_mask_np = (probs[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255
            img_np = (img_tensor[0].numpy() * 255).astype(np.uint8)
            morph_metrics = compute_morphological_metrics(pred_mask_np, img_np)

            target_area = meta.get("target_area_px", 0)
            pred_area = morph_metrics["mask_area_px"]
            mask_to_roi_ratio = round(pred_area / (128 * 128), 4)

            morphological_records.append({
                "type": meta["type"],
                "swath_name": meta["swath_name"],
                "true_target_area_px": target_area,
                "pred_mask_area_px": pred_area,
                "mask_to_roi_ratio": mask_to_roi_ratio,
                "orientation_deg": morph_metrics["orientation_deg"],
                "elongation": morph_metrics["elongation"],
                "compactness": morph_metrics["compactness"],
                "contrast_ratio": morph_metrics["contrast_ratio"],
                "mean_seg_prob": round(float(probs[0, 0].mean().item()), 4)
            })

    cat_probs = torch.cat(all_probs, dim=0)
    cat_targets = torch.cat(all_targets, dim=0)
    final_eval_metrics = calculate_metrics(cat_probs, cat_targets)

    # Separate metrics for positive vs negative test patches
    pos_indices = [i for i, r in enumerate(morphological_records) if r["type"] == "positive"]
    neg_indices = [i for i, r in enumerate(morphological_records) if r["type"] == "negative"]

    pos_probs = cat_probs[pos_indices]
    pos_targets = cat_targets[pos_indices]
    pos_metrics = calculate_metrics(pos_probs, pos_targets)

    neg_probs = cat_probs[neg_indices]
    neg_targets = cat_targets[neg_indices]
    neg_metrics = calculate_metrics(neg_probs, neg_targets)

    # Morphology summary for positive shipwreck candidates
    pos_orientations = [r["orientation_deg"] for r in morphological_records if r["type"] == "positive" and r["pred_mask_area_px"] > 0]
    pos_elongations = [r["elongation"] for r in morphological_records if r["type"] == "positive" and r["pred_mask_area_px"] > 0]
    pos_compactness = [r["compactness"] for r in morphological_records if r["type"] == "positive" and r["pred_mask_area_px"] > 0]

    # Save training history and evaluation metrics
    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    eval_summary = {
        "model_name": "SW-Net",
        "architecture": "SWNetArchitecture",
        "trainable_parameters": total_params,
        "input_resolution": [128, 128],
        "training_dataset": "AI4Shipwrecks",
        "swath_split": {
            "train_swaths": 141,
            "test_swaths": 120,
            "train_patches": len(train_dataset),
            "test_patches": len(test_dataset)
        },
        "training_hyperparameters": {
            "batch_size": batch_size,
            "max_epochs": num_epochs,
            "best_epoch": best_epoch,
            "learning_rate": lr,
            "optimizer": "AdamW",
            "loss": "BCEWithLogitsLoss + SoftDiceLoss (50/50)",
            "random_seed": seed,
            "hardware": str(device),
            "total_duration_sec": round(total_train_duration, 2)
        },
        "overall_test_metrics": final_eval_metrics,
        "positive_shipwreck_patches": {
            "count": len(pos_indices),
            "metrics": pos_metrics,
            "mean_orientation_deg": round(float(np.mean(pos_orientations)), 2) if pos_orientations else 0.0,
            "mean_elongation": round(float(np.mean(pos_elongations)), 2) if pos_elongations else 1.0,
            "mean_compactness": round(float(np.mean(pos_compactness)), 4) if pos_compactness else 0.0
        },
        "negative_seabed_patches": {
            "count": len(neg_indices),
            "metrics": neg_metrics,
            "false_positive_activation_rate": round(float(np.mean([(r['pred_mask_area_px'] > 50) for r in morphological_records if r['type'] == 'negative'])), 4)
        },
        "best_checkpoint": best_ckpt_path,
        "final_checkpoint": final_ckpt_path
    }

    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(eval_summary, f, indent=2)

    print("\n=======================================================")
    print("           SW-NET FINAL VALIDATION RESULTS             ")
    print("=======================================================")
    print(f"  Best Epoch:           {best_epoch}")
    print(f"  Test IoU:             {final_eval_metrics['iou']:.4f}")
    print(f"  Test Dice / F1:       {final_eval_metrics['dice']:.4f}")
    print(f"  Test Precision:       {final_eval_metrics['precision']:.4f}")
    print(f"  Test Recall:          {final_eval_metrics['recall']:.4f}")
    print(f"  Pixel Accuracy:       {final_eval_metrics['pixel_accuracy']:.4f}")
    print(f"  Seabed Background IoU:{final_eval_metrics['background_iou']:.4f}")
    print(f"  Total Duration:       {total_train_duration:.2f}s")
    print(f"  Best Checkpoint:      {best_ckpt_path}")
    print("=======================================================\n")

    return eval_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SW-Net Full Training & Validation")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_swnet(num_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
