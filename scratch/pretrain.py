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


class KeypointDataset(Dataset):
    def __init__(self, items, training=True, num_keypoints=6):
        self.items = items
        self.training = training
        self.num_keypoints = num_keypoints
            
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        data_dir, item = self.items[idx]
        
        # Load RGB
        rgb_path = os.path.join(data_dir, item["rgb"])
        img = Image.open(rgb_path)
        img_arr = np.array(img).astype(np.float32) / 255.0
        
        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).float()
        
        # Load Keypoints
        kps = torch.tensor(item["keypoints"], dtype=torch.float32) # shape (num_keypoints, 2)
        
        # Load Visibility Mask
        visibility = torch.tensor(item.get("visibility", [True] * self.num_keypoints), dtype=torch.float32)
        
        # Ensure we only take the requested number of keypoints in case of mismatch
        kps = kps[:self.num_keypoints]
        visibility = visibility[:self.num_keypoints]
        
        # Load 6D Pose
        pose_6d = torch.tensor(item["pose_6d"], dtype=torch.float32) # shape (6,)
        
        # --- Data Augmentation (Only during training) ---
        if self.training:
            # 1. Random Translation (-3 to +3 pixels)
            dx = torch.randint(-3, 4, (1,)).item()
            dy = torch.randint(-3, 4, (1,)).item()
            img_tensor = torch.roll(img_tensor, shifts=(dy, dx), dims=(1, 2))
            kps[:, 0] = kps[:, 0] + dx
            kps[:, 1] = kps[:, 1] + dy
            off_screen = (kps[:, 0] < 0) | (kps[:, 0] > 128) | (kps[:, 1] < 0) | (kps[:, 1] > 128)
            visibility[off_screen] = 0.0

            # 2. Random Cutout (simulate self-occlusion or clutter)
            if random.random() < 0.4:
                cx = random.randint(10, 118)
                cy = random.randint(10, 118)
                cw = random.randint(8, 25)
                ch = random.randint(8, 25)
                x1, x2 = max(0, cx - cw // 2), min(128, cx + cw // 2)
                y1, y2 = max(0, cy - ch // 2), min(128, cy + ch // 2)
                img_tensor[:, y1:y2, x1:x2] = 0.0
                for i in range(len(kps)):
                    u, v = kps[i, 0].item(), kps[i, 1].item()
                    if x1 < u < x2 and y1 < v < y2:
                        visibility[i] = 0.0

            # 3. Random Brightness Jitter (±15%)
            if random.random() < 0.5:
                factor = random.uniform(0.85, 1.15)
                img_tensor = img_tensor * factor

            # 4. Standard i.i.d. Gaussian Noise
            if random.random() < 0.5:
                noise_std = random.uniform(0.005, 0.02)
                img_tensor = img_tensor + torch.randn_like(img_tensor) * noise_std

        # Ensure final img tensor is clamped to [0.0, 1.0] range
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
        
        # Flip coordinates from [u, v] (col, row) to [v, u] (row, col) to match SpatialSoftmax
        kps = kps.flip(dims=(-1,))
        
        # Normalize keypoints to [-1, 1]
        kps = (kps / 64.0) - 1.0
        
        # Construct 2D visibility mask
        visibility_2d = visibility.unsqueeze(-1).repeat(1, 2).flatten()
        
        return img_tensor, kps.flatten(), pose_6d, visibility_2d


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


class PolicyCNNBackbone(nn.Module):
    def __init__(self, num_keypoints=6):
        super().__init__()
        # Shared visual feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU()
        )
        # Keypoints branch
        self.head = nn.Conv2d(64, num_keypoints, kernel_size=1, stride=1, padding=0)
        self.spatial_softmax = SpatialSoftmax(32, 32, temperature=0.5)
        
        # 6D Pose branch
        self.pose_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 6)
        )

    def forward(self, x):
        features = self.cnn(x)
        
        # Keypoint branch
        kps = self.head(features)
        kps = self.spatial_softmax(kps) # B x 12
        
        # 6D Pose branch
        pooled = torch.mean(features, dim=[2, 3]) # B x 64
        pose = self.pose_head(pooled) # B x 6
        
        return torch.cat([kps, pose], dim=-1)


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
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return input + x


class ConvNeXtBackbone(nn.Module):
    def __init__(self, num_keypoints=6):
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
        # Keypoints head
        self.head = nn.Conv2d(64, num_keypoints, kernel_size=1)
        self.spatial_softmax = SpatialSoftmax(32, 32, temperature=0.5)
        
        # 6D Pose Head
        self.pose_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ELU(),
            nn.Linear(64, 6)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.downsample(x)
        features = self.stage2(x)
        
        # Keypoints
        kps = self.head(features)
        kps = self.spatial_softmax(kps)
        
        # 6D Pose
        pooled = torch.mean(features, dim=[2, 3])
        pose = self.pose_head(pooled)
        
        return torch.cat([kps, pose], dim=-1)


def train():
    parser = argparse.ArgumentParser(description="Unified Supervised Offline Keypoints & 6D Pose Pretraining")
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["cnn", "convnext"],
        default="cnn",
        help="Visual backbone architecture type (default: cnn)"
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (default: 5e-4)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train (default: 100)")
    parser.add_argument("--num_keypoints", type=int, default=6, help="Number of keypoints (default: 6)")
    parser.add_argument("--device", type=str, default=None, help="Device to train on (cuda or cpu)")
    
    args = parser.parse_args()
    
    # Defaults depending on backbone
    if args.batch_size is None:
        args.batch_size = 128 if args.backbone == "cnn" else 32
        
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    
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
                labels = json.load(f)
                for label in labels:
                    all_items.append((data_dir, label))
                    
    print(f"Total data pool size: {len(all_items)} samples.")
    
    random.shuffle(all_items)
    
    train_size = min(100000, len(all_items))
    val_size = min(10000, len(all_items) - train_size)
    
    if val_size <= 0:
        print(f"Warning: Pool size is too small ({len(all_items)}). Training with {train_size} samples and no validation.")
        train_items = all_items
        val_items = []
        has_val = False
    else:
        train_items = all_items[:train_size]
        val_items = all_items[train_size : train_size + val_size]
        has_val = True
        print(f"Split data pool: {len(train_items)} train, {len(val_items)} validation.")
        
    train_dataset = KeypointDataset(train_items, training=True, num_keypoints=args.num_keypoints)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    
    if has_val:
        val_dataset = KeypointDataset(val_items, training=False, num_keypoints=args.num_keypoints)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # Initialize Backbone
    if args.backbone == "cnn":
        model = PolicyCNNBackbone(num_keypoints=args.num_keypoints).to(device)
        save_path = "pretrained_backbone_rgb.pth"
    else:
        model = ConvNeXtBackbone(num_keypoints=args.num_keypoints).to(device)
        save_path = "pretrained_convnext_rgb.pth"
        
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    
    use_visibility_mask = False
    
    print(f"Starting supervised offline training ({args.backbone.upper()}, {args.num_keypoints} keypoints + 6D Pose) on {device}...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_kps_loss = 0.0
        running_pose_loss = 0.0
        
        for images, keypoints, poses, visibility in train_dataloader:
            images = images.to(device)
            keypoints = keypoints.to(device)
            poses = poses.to(device)
            visibility = visibility.to(device)
            
            if not use_visibility_mask:
                visibility = torch.ones_like(visibility)
                
            optimizer.zero_grad()
            outputs = model(images)
            
            # Split outputs: first 12 are keypoints, next 6 are pose_6d
            pred_kps = outputs[:, :args.num_keypoints * 2]
            pred_pose = outputs[:, args.num_keypoints * 2:]
            
            # Keypoints masked MSE loss
            diff = (pred_kps - keypoints) * visibility
            kps_loss = (diff ** 2).sum() / torch.clamp(visibility.sum(), min=1.0)
            
            # 6D Pose MSE loss
            pose_loss = F.mse_loss(pred_pose, poses)
            
            # Combined Loss
            loss = kps_loss + pose_loss
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_kps_loss += kps_loss.item()
            running_pose_loss += pose_loss.item()
            
        avg_train_loss = running_loss / len(train_dataloader)
        avg_kps_loss = running_kps_loss / len(train_dataloader)
        avg_pose_loss = running_pose_loss / len(train_dataloader)
        
        # Validation Loop
        if has_val:
            model.eval()
            val_loss = 0.0
            val_kps_loss = 0.0
            val_pose_loss = 0.0
            with torch.no_grad():
                for images, keypoints, poses, visibility in val_dataloader:
                    images = images.to(device)
                    keypoints = keypoints.to(device)
                    poses = poses.to(device)
                    visibility = visibility.to(device)
                    
                    if not use_visibility_mask:
                        visibility = torch.ones_like(visibility)
                        
                    outputs = model(images)
                    pred_kps = outputs[:, :args.num_keypoints * 2]
                    pred_pose = outputs[:, args.num_keypoints * 2:]
                    
                    diff = (pred_kps - keypoints) * visibility
                    kps_loss = (diff ** 2).sum() / torch.clamp(visibility.sum(), min=1.0)
                    pose_loss = F.mse_loss(pred_pose, poses)
                    
                    val_loss += (kps_loss.item() + pose_loss.item())
                    val_kps_loss += kps_loss.item()
                    val_pose_loss += pose_loss.item()
            
            avg_val_loss = val_loss / len(val_dataloader)
            avg_val_kps_loss = val_kps_loss / len(val_dataloader)
            avg_val_pose_loss = val_pose_loss / len(val_dataloader)
            
            print(f"Epoch {epoch+1:2d}/{args.epochs} | Train Loss: {avg_train_loss:.6f} (KPs: {avg_kps_loss:.6f}, Pose: {avg_pose_loss:.6f}) | Val Loss: {avg_val_loss:.6f} (KPs: {avg_val_kps_loss:.6f}, Pose: {avg_val_pose_loss:.6f})")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), save_path)
                print(f"  -> Saved new best model with Val Loss {best_val_loss:.6f} to {save_path}!")
        else:
            print(f"Epoch {epoch+1:2d}/{args.epochs} | Train Loss: {avg_train_loss:.6f} (KPs: {avg_kps_loss:.6f}, Pose: {avg_pose_loss:.6f})")
            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                torch.save(model.state_dict(), save_path)
                print(f"  -> Saved new best model with Train Loss {best_train_loss:.6f} to {save_path}!")
            
        if avg_train_loss < 0.00005:
            print("Target training loss achieved. Stopping early.")
            break

    print(f"Offline training complete. Best weights saved to {save_path}")


if __name__ == "__main__":
    train()
