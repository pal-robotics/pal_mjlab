"""
pretrain_pos.py
===============
Train a ConvNeXt backbone to predict the 3D object position in the robot
root frame (base_footprint).  The keypoint head and orientation outputs are
removed; the model outputs a single 3-D vector (x, y, z) in metres.

New augmentations vs. the original pretrain.py:
  - Variable-FOV zoom crop  : randomly zooms into the 128×128 image by a
                               factor drawn from [1.0, 1.25], then resizes
                               back to 128×128.  This simulates different
                               camera FOVs and teaches scale invariance.
  - Pixel shift (±8 px)     : larger random translation than the original
                               ±3 px, making the model more robust to small
                               camera-extrinsic variations.
  - Random cutout            : unchanged from original (simulate occlusion).
  - Brightness jitter        : unchanged.
  - Gaussian noise           : unchanged.
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────


class PositionDataset(Dataset):
  """Loads (image, 3D-position) pairs from a dataset directory."""

  def __init__(self, items, training: bool = True):
    self.items = items
    self.training = training

  def __len__(self):
    return len(self.items)

  def __getitem__(self, idx):
    data_dir, item = self.items[idx]

    # ── Load image ────────────────────────────────────────────────────────
    rgb_path = os.path.join(data_dir, item["rgb"])
    img = Image.open(rgb_path).convert("RGB")  # 128×128 RGB
    img_arr = np.array(img).astype(np.float32) / 255.0  # H×W×3

    img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).float()  # 3×H×W

    # ── Ground-truth position (first 3 values of pose_6d) ─────────────────
    pos_3d = torch.tensor(item["pose_6d"][:3], dtype=torch.float32)

    # ── Data augmentation (training only) ─────────────────────────────────
    if self.training:
      img_tensor = self._augment(img_tensor)

    img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
    return img_tensor, pos_3d

  # ── Augmentation helpers ──────────────────────────────────────────────────

  @staticmethod
  def _augment(img: torch.Tensor) -> torch.Tensor:
    """Apply all training augmentations to a 3×H×W float tensor."""
    _, H, W = img.shape

    # 1. Variable-FOV zoom crop ─────────────────────────────────────────
    #    Zoom factor in [1.0, 1.25]: a factor of 1.2 means we crop the
    #    central 80% of the image and resize back → simulates a wider FOV.
    zoom = random.uniform(1.0, 1.25)
    if zoom > 1.0:
      crop_h = int(round(H / zoom))
      crop_w = int(round(W / zoom))
      # Centre the crop
      top = (H - crop_h) // 2
      left = (W - crop_w) // 2
      img = img[:, top : top + crop_h, left : left + crop_w]
      # Resize back with bilinear interpolation
      img = F.interpolate(
        img.unsqueeze(0),
        size=(H, W),
        mode="bilinear",
        align_corners=False,
      ).squeeze(0)

    # 2. Pixel shift (±8 px) ────────────────────────────────────────────
    dx = random.randint(-8, 8)
    dy = random.randint(-8, 8)
    img = torch.roll(img, shifts=(dy, dx), dims=(1, 2))
    # Zero-fill the wrapped-around border instead of leaving periodic artefacts
    if dx > 0:
      img[:, :, :dx] = 0.0
    elif dx < 0:
      img[:, :, dx:] = 0.0
    if dy > 0:
      img[:, :dy, :] = 0.0
    elif dy < 0:
      img[:, dy:, :] = 0.0

    # 3. Random cutout (simulate occlusion / clutter) ───────────────────
    if random.random() < 0.4:
      _, H2, W2 = img.shape
      cx = random.randint(10, W2 - 10)
      cy = random.randint(10, H2 - 10)
      cw = random.randint(8, 25)
      ch = random.randint(8, 25)
      x1, x2 = max(0, cx - cw // 2), min(W2, cx + cw // 2)
      y1, y2 = max(0, cy - ch // 2), min(H2, cy + ch // 2)
      img[:, y1:y2, x1:x2] = 0.0

    # 4. Random brightness jitter ────────────────────────────────────────
    if random.random() < 0.5:
      img = img * random.uniform(0.85, 1.15)

    # 5. Gaussian pixel noise ────────────────────────────────────────────
    if random.random() < 0.5:
      img = img + torch.randn_like(img) * random.uniform(0.005, 0.02)

    return img


# ──────────────────────────────────────────────────────────────────────────────
# Model
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


class ConvNeXtPosition(nn.Module):
  """ConvNeXt backbone with a single 3-D position regression head.

  Output: tensor of shape (B, 3)  →  (x, y, z) in the robot root frame [m].
  """

  def __init__(self):
    super().__init__()
    self.stem = nn.Sequential(
      nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
      LayerNorm2d(32),
    )
    self.stage1 = nn.Sequential(
      ConvNeXtBlock(32),
      ConvNeXtBlock(32),
    )
    self.downsample = nn.Sequential(
      LayerNorm2d(32),
      nn.Conv2d(32, 64, kernel_size=2, stride=2),
    )
    self.stage2 = nn.Sequential(
      ConvNeXtBlock(64),
      ConvNeXtBlock(64),
      ConvNeXtBlock(64),
    )
    self.pos_head = nn.Sequential(
      nn.Linear(64, 64),
      nn.ELU(),
      nn.Linear(64, 3),  # ← 3-D position only
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.stem(x)
    x = self.stage1(x)
    x = self.downsample(x)
    x = self.stage2(x)
    pooled = x.mean(dim=[2, 3])  # global average pool  (B, 64)
    return self.pos_head(pooled)  # (B, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────


def train():
  parser = argparse.ArgumentParser(
    description="Train ConvNeXt to predict 3D object position in robot root frame"
  )
  parser.add_argument("--batch_size", type=int, default=32)
  parser.add_argument("--lr", type=float, default=5e-4)
  parser.add_argument("--epochs", type=int, default=100)
  parser.add_argument("--device", type=str, default=None)
  parser.add_argument(
    "--save",
    type=str,
    default="pretrained_convnext_pos.pth",
    help="Output weights file (default: pretrained_convnext_pos.pth)",
  )
  args = parser.parse_args()

  device = torch.device(
    args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
  )

  # ── Load dataset ──────────────────────────────────────────────────────────
  all_items = []
  for candidate in ["dataset", "dataset_val", "../dataset", "../dataset_val"]:
    labels_path = os.path.join(candidate, "labels.json")
    if os.path.exists(labels_path):
      print(f"Loading labels from: {labels_path}")
      with open(labels_path) as f:
        for label in json.load(f):
          all_items.append((candidate, label))

  if not all_items:
    raise FileNotFoundError("No dataset found. Run collect_pretrain_data.py first.")

  print(f"Total pool: {len(all_items)} samples")
  random.shuffle(all_items)

  train_size = min(100_000, len(all_items))
  val_size = min(10_000, len(all_items) - train_size)
  has_val = val_size > 0

  train_items = all_items[:train_size]
  val_items = all_items[train_size : train_size + val_size] if has_val else []

  print(f"  train={len(train_items)}  val={len(val_items)}")

  train_ds = PositionDataset(train_items, training=True)
  train_dl = DataLoader(
    train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
  )

  if has_val:
    val_ds = PositionDataset(val_items, training=False)
    val_dl = DataLoader(
      val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

  # ── Model & optimiser ─────────────────────────────────────────────────────
  model = ConvNeXtPosition().to(device)
  optimizer = optim.Adam(model.parameters(), lr=args.lr)
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

  num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"ConvNeXtPosition — {num_params:,} trainable params | device={device}")
  print(f"Saving to: {args.save}\n")

  best_loss = float("inf")

  for epoch in range(args.epochs):
    # ── Train ─────────────────────────────────────────────────────────────
    model.train()
    train_loss = 0.0
    for images, pos_gt in train_dl:
      images = images.to(device)
      pos_gt = pos_gt.to(device)

      optimizer.zero_grad()
      pred = model(images)
      loss = F.mse_loss(pred, pos_gt)
      loss.backward()
      optimizer.step()
      train_loss += loss.item()

    train_loss /= len(train_dl)
    scheduler.step()

    # ── Validation ────────────────────────────────────────────────────────
    if has_val:
      model.eval()
      val_loss = 0.0
      with torch.no_grad():
        for images, pos_gt in val_dl:
          images = images.to(device)
          pos_gt = pos_gt.to(device)
          val_loss += F.mse_loss(model(images), pos_gt).item()
      val_loss /= len(val_dl)

      # Convert MSE → approximate RMSE in cm for readability
      train_rmse_cm = (train_loss**0.5) * 100.0
      val_rmse_cm = (val_loss**0.5) * 100.0

      print(
        f"Epoch {epoch + 1:3d}/{args.epochs} | "
        f"Train MSE={train_loss:.6f} (~{train_rmse_cm:.1f} cm RMSE) | "
        f"Val MSE={val_loss:.6f} (~{val_rmse_cm:.1f} cm RMSE)"
      )

      monitor_loss = val_loss
    else:
      train_rmse_cm = (train_loss**0.5) * 100.0
      print(
        f"Epoch {epoch + 1:3d}/{args.epochs} | "
        f"Train MSE={train_loss:.6f} (~{train_rmse_cm:.1f} cm RMSE)"
      )
      monitor_loss = train_loss

    if monitor_loss < best_loss:
      best_loss = monitor_loss
      torch.save(model.state_dict(), args.save)
      print(f"  → Saved best model (loss={best_loss:.6f}) to {args.save}")

    if train_loss < 1e-5:
      print("Early stop: training loss below threshold.")
      break

  print(f"\nDone. Best weights saved to {args.save}")


if __name__ == "__main__":
  train()
