"""
pretrain_kp_pos.py
==================
Train a ConvNeXt or CNN backbone to estimate:
  - 4 Cube Keypoints (8 outputs)
  - 3D Position only (3 outputs)

This incorporates robust visual augmentations:
  - Variable-FOV zoom crop (with correct keypoint scaling and visibility updates)
  - Pixel shifting up to ±8 px (with zeroed borders and keypoint coordinate shifts)
  - Cutout (occlusion simulation)
  - Brightness jitter
  - Gaussian noise

Displays separate keypoints and position losses.
"""

import os
import json
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Dataset with exact geometric keypoint transformations
# ──────────────────────────────────────────────────────────────────────────────

class KeypointPositionDataset(Dataset):
    def __init__(self, items, training: bool = True, num_keypoints: int = 4):
        self.items = items
        self.training = training
        self.num_keypoints = num_keypoints

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        data_dir, item = self.items[idx]

        # 1. Load RGB
        rgb_path = os.path.join(data_dir, item["rgb"])
        img = Image.open(rgb_path).convert("RGB")
        img_arr = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).float()  # [3, 128, 128]

        # 2. Load 4 Cube Keypoints
        kps = torch.tensor(item["keypoints"][:self.num_keypoints], dtype=torch.float32) # [4, 2]
        visibility = torch.tensor(item.get("visibility", [True] * 6)[:self.num_keypoints], dtype=torch.float32)

        # 3. Load 3D Position only
        pos_3d = torch.tensor(item["pose_6d"][:3], dtype=torch.float32) # [3,]

        # ─── Training Augmentations ───
        if self.training:
            _, H, W = img_tensor.shape

            # A. Variable FOV Zoom Crop
            zoom = random.uniform(1.0, 1.25)
            if zoom > 1.0:
                crop_h = int(round(H / zoom))
                crop_w = int(round(W / zoom))
                top = (H - crop_h) // 2
                left = (W - crop_w) // 2

                # Crop image and resize back to 128x128
                img_tensor = img_tensor[:, top:top + crop_h, left:left + crop_w]
                img_tensor = F.interpolate(
                    img_tensor.unsqueeze(0),
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False
                ).squeeze(0)

                # Transform keypoints: scale coordinates relative to crop frame
                kps[:, 0] = (kps[:, 0] - left) * (float(W) / crop_w)
                kps[:, 1] = (kps[:, 1] - top) * (float(H) / crop_h)

                # Update visibility for keypoints shifted off-screen
                off_screen = (kps[:, 0] < 0) | (kps[:, 0] > W) | (kps[:, 1] < 0) | (kps[:, 1] > H)
                visibility[off_screen] = 0.0

            # B. Pixel Shifting (±8 px)
            dx = random.randint(-8, 8)
            dy = random.randint(-8, 8)
            img_tensor = torch.roll(img_tensor, shifts=(dy, dx), dims=(1, 2))
            
            # Zero-out wrapped edges to avoid periodic boundary artifacts
            if dx > 0:
                img_tensor[:, :, :dx] = 0.0
            elif dx < 0:
                img_tensor[:, :, dx:] = 0.0
            if dy > 0:
                img_tensor[:, :dy, :] = 0.0
            elif dy < 0:
                img_tensor[:, dy:, :] = 0.0

            # Shift keypoints
            kps[:, 0] = kps[:, 0] + dx
            kps[:, 1] = kps[:, 1] + dy
            off_screen = (kps[:, 0] < 0) | (kps[:, 0] > W) | (kps[:, 1] < 0) | (kps[:, 1] > H)
            visibility[off_screen] = 0.0

            # C. Cutout (occlusion simulation)
            if random.random() < 0.4:
                cx = random.randint(10, W - 10)
                cy = random.randint(10, H - 10)
                cw = random.randint(8, 25)
                ch = random.randint(8, 25)
                x1, x2 = max(0, cx - cw // 2), min(W, cx + cw // 2)
                y1, y2 = max(0, cy - ch // 2), min(H, cy + ch // 2)
                img_tensor[:, y1:y2, x1:x2] = 0.0
                
                # Mark keypoints inside cutout box as invisible
                for i in range(len(kps)):
                    u, v = kps[i, 0].item(), kps[i, 1].item()
                    if x1 < u < x2 and y1 < v < y2:
                        visibility[i] = 0.0

            # D. Brightness Jitter
            if random.random() < 0.5:
                img_tensor = img_tensor * random.uniform(0.85, 1.15)

            # E. Gaussian Noise
            if random.random() < 0.5:
                img_tensor = img_tensor + torch.randn_like(img_tensor) * random.uniform(0.005, 0.02)

        # Clamping to valid range
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

        # Flip keypoints from [u, v] (col, row) to [v, u] (row, col) for SpatialSoftmax compatibility
        kps = kps.flip(dims=(-1,))
        # Normalize keypoints to [-1, 1] range
        kps = (kps / 64.0) - 1.0

        # Construct 2D visibility mask
        visibility_2d = visibility.unsqueeze(-1).repeat(1, 2).flatten()

        return img_tensor, kps.flatten(), pos_3d, visibility_2d


# ──────────────────────────────────────────────────────────────────────────────
# Spatial Softmax
# ──────────────────────────────────────────────────────────────────────────────

class SpatialSoftmax(nn.Module):
    def __init__(self, height, width, temperature=0.5):
        super(SpatialSoftmax, self).__init__()
        self.height = height
        self.width = width
        self.temperature = temperature
        
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.height),
            torch.linspace(-1.0, 1.0, self.width),
            indexing="ij"
        )
        self.register_buffer('pos_x', pos_x.reshape(1, 1, -1))
        self.register_buffer('pos_y', pos_y.reshape(1, 1, -1))

    def forward(self, feature):
        B, C, H, W = feature.size()
        features = feature.reshape(B, C, -1)
        weights = torch.softmax(features / self.temperature, dim=-1)
        expected_x = (weights * self.pos_x).sum(dim=-1)
        expected_y = (weights * self.pos_y).sum(dim=-1)
        return torch.stack([expected_x, expected_y], dim=-1).reshape(B, C * 2)


# ──────────────────────────────────────────────────────────────────────────────
# Visual Models (Dual-Head for 4 KPs + 3D position)
# ──────────────────────────────────────────────────────────────────────────────

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

    def forward(self, x):
        return x + self.pwconv2(self.act(self.pwconv1(self.norm(self.dwconv(x)))))


class ConvNeXtKPPos(nn.Module):
    """ConvNeXt Dual-Head Model predicting 4 Keypoints (8 values) + 3D Position (3 values)."""
    def __init__(self, num_keypoints=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            LayerNorm2d(32)
        )
        self.stage1 = nn.Sequential(
            ConvNeXtBlock(32),
            ConvNeXtBlock(32)
        )
        self.downsample = nn.Sequential(
            LayerNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=2, stride=2)
        )
        self.stage2 = nn.Sequential(
            ConvNeXtBlock(64),
            ConvNeXtBlock(64),
            ConvNeXtBlock(64)
        )
        
        # 1. Keypoints Head (4 channels)
        self.head = nn.Conv2d(64, num_keypoints, kernel_size=1)
        self.spatial_softmax = SpatialSoftmax(32, 32, temperature=0.5)
        
        # 2. 3D Position Head
        self.pos_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.downsample(x)
        features = self.stage2(x)
        
        # Keypoints output [B, 8]
        kps = self.head(features)
        kps = self.spatial_softmax(kps)
        
        # 3D Position output [B, 3]
        pooled = torch.mean(features, dim=[2, 3])
        pos = self.pos_head(pooled)
        
        return torch.cat([kps, pos], dim=-1)


class PolicyCNNKPPos(nn.Module):
    """Standard CNN Dual-Head Model predicting 4 Keypoints + 3D Position."""
    def __init__(self, num_keypoints=4):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU()
        )
        
        # 1. Keypoints Head
        self.head = nn.Conv2d(64, num_keypoints, kernel_size=1, stride=1, padding=0)
        self.spatial_softmax = SpatialSoftmax(32, 32, temperature=0.5)
        
        # 2. 3D Position Head
        self.pos_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        features = self.cnn(x)
        
        kps = self.head(features)
        kps = self.spatial_softmax(kps)
        
        pooled = torch.mean(features, dim=[2, 3])
        pos = self.pos_head(pooled)
        
        return torch.cat([kps, pos], dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def train():
    parser = argparse.ArgumentParser(description="Supervised Pretraining: 4 Cube Keypoints + 3D Position")
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["cnn", "convnext"],
        default="convnext",
        help="Visual backbone architecture type (default: convnext)"
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (default: 5e-4)")
    parser.add_argument("--epochs", type=int, default=100, help="Epochs (default: 100)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda or cpu)")
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Path to save best weights (defaults: pretrained_convnext_kp_pos.pth / pretrained_backbone_kp_pos.pth)"
    )
    parser.add_argument(
        "--no_kp_mask",
        action="store_true",
        default=False,
        help="Disable visibility-masked keypoint loss and use plain MSE instead"
    )
    args = parser.parse_args()

    if args.batch_size is None:
        args.batch_size = 32 if args.backbone == "convnext" else 128

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Resolve output save path
    if args.save is None:
        save_path = f"pretrained_{args.backbone}_kp_pos.pth"
    else:
        save_path = args.save

    # Locate dataset directories
    train_dir = "dataset"
    val_dir = "dataset_val"
    resolved_dirs = []
    for d in [train_dir, val_dir]:
        if os.path.exists(d):
            resolved_dirs.append(d)
        else:
            parent_d = os.path.join("..", d)
            if os.path.exists(parent_d):
                resolved_dirs.append(parent_d)

    if not resolved_dirs:
        raise FileNotFoundError("No dataset directories ('dataset' or 'dataset_val') found.")

    all_items = []
    for data_dir in resolved_dirs:
        labels_path = os.path.join(data_dir, "labels.json")
        if os.path.exists(labels_path):
            print(f"Loading labels from: {labels_path}")
            with open(labels_path, "r") as f:
                for label in json.load(f):
                    all_items.append((data_dir, label))

    print(f"Total pool: {len(all_items)} samples")
    random.shuffle(all_items)

    train_size = min(100_000, len(all_items))
    val_size = min(10_000, len(all_items) - train_size)
    has_val = val_size > 0

    train_items = all_items[:train_size]
    val_items = all_items[train_size:train_size + val_size] if has_val else []

    print(f"Split: {len(train_items)} train, {len(val_items)} validation.")

    # DataLoaders
    train_dataset = KeypointPositionDataset(train_items, training=True, num_keypoints=4)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    if has_val:
        val_dataset = KeypointPositionDataset(val_items, training=False, num_keypoints=4)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize model
    if args.backbone == "cnn":
        model = PolicyCNNKPPos(num_keypoints=4).to(device)
    else:
        model = ConvNeXtKPPos(num_keypoints=4).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized {args.backbone.upper()} Model — {num_params:,} trainable params | device={device}")
    print(f"Target weights path: {save_path}\n")

    best_val_loss = float("inf")
    best_train_loss = float("inf")

    for epoch in range(args.epochs):
        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        running_kps_loss = 0.0
        running_pos_loss = 0.0

        for images, kps_gt, pos_gt, visibility in train_dataloader:
            images = images.to(device)
            kps_gt = kps_gt.to(device)
            pos_gt = pos_gt.to(device)
            visibility = visibility.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Split output [B, 11] -> KPs [B, 8], Pos [B, 3]
            pred_kps = outputs[:, :8]
            pred_pos = outputs[:, 8:]

            # Keypoints Loss
            if args.no_kp_mask:
                kps_loss = F.mse_loss(pred_kps, kps_gt)
            else:
                # Masked MSE (only on visible coordinates)
                diff = (pred_kps - kps_gt) * visibility
                kps_loss = (diff ** 2).sum() / torch.clamp(visibility.sum(), min=1.0)

            # Position Loss (MSE)
            pos_loss = F.mse_loss(pred_pos, pos_gt)

            loss = kps_loss + pos_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_kps_loss += kps_loss.item()
            running_pos_loss += pos_loss.item()

        avg_train_loss = running_loss / len(train_dataloader)
        avg_train_kps = running_kps_loss / len(train_dataloader)
        avg_train_pos = running_pos_loss / len(train_dataloader)
        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────────
        if has_val:
            model.eval()
            val_loss = 0.0
            val_kps = 0.0
            val_pos = 0.0

            with torch.no_grad():
                for images, kps_gt, pos_gt, visibility in val_dataloader:
                    images = images.to(device)
                    kps_gt = kps_gt.to(device)
                    pos_gt = pos_gt.to(device)
                    visibility = visibility.to(device)

                    outputs = model(images)
                    pred_kps = outputs[:, :8]
                    pred_pos = outputs[:, 8:]

                    if args.no_kp_mask:
                        kps_loss = F.mse_loss(pred_kps, kps_gt)
                    else:
                        diff = (pred_kps - kps_gt) * visibility
                        kps_loss = (diff ** 2).sum() / torch.clamp(visibility.sum(), min=1.0)
                    pos_loss = F.mse_loss(pred_pos, pos_gt)

                    val_loss += (kps_loss.item() + pos_loss.item())
                    val_kps += kps_loss.item()
                    val_pos += pos_loss.item()

            avg_val_loss = val_loss / len(val_dataloader)
            avg_val_kps = val_kps / len(val_dataloader)
            avg_val_pos = val_pos / len(val_dataloader)

            # Informative printout with physical unit approximations (RMSE in cm)
            train_pos_rmse_cm = (avg_train_pos ** 0.5) * 100.0
            val_pos_rmse_cm = (avg_val_pos ** 0.5) * 100.0

            print(
                f"Epoch {epoch+1:3d}/{args.epochs} | "
                f"Train Loss: {avg_train_loss:.6f} (KPs: {avg_train_kps:.6f}, Pos: {avg_train_pos:.6f} ~{train_pos_rmse_cm:.1f}cm) | "
                f"Val Loss: {avg_val_loss:.6f} (KPs: {avg_val_kps:.6f}, Pos: {avg_val_pos:.6f} ~{val_pos_rmse_cm:.1f}cm)"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)
                print(f"  → Saved new best model to {save_path} (Val Loss: {best_val_loss:.6f})")

        else:
            train_pos_rmse_cm = (avg_train_pos ** 0.5) * 100.0
            print(
                f"Epoch {epoch+1:3d}/{args.epochs} | "
                f"Train Loss: {avg_train_loss:.6f} (KPs: {avg_train_kps:.6f}, Pos: {avg_train_pos:.6f} ~{train_pos_rmse_cm:.1f}cm)"
            )

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                torch.save(model.state_dict(), save_path)
                print(f"  → Saved new best model to {save_path} (Train Loss: {best_train_loss:.6f})")

        if avg_train_loss < 0.00005:
            print("Early stopping: Target loss met.")
            break

    print(f"\nPretraining finished. Best model weights saved to {save_path}")


if __name__ == "__main__":
    train()
