import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class KeypointDataset(Dataset):
    def __init__(self, items, training=True):
        self.items = items
        self.training = training
            
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        data_dir, item = self.items[idx]
        
        # Load Depth
        depth_path = os.path.join(data_dir, item["depth"])
        depth_map = np.load(depth_path)
        if len(depth_map.shape) == 2:
            depth_map = depth_map[:, :, np.newaxis]
        
        depth_tensor = torch.from_numpy(depth_map).permute(2, 0, 1).float()
        
        # Baseline Normalization: matches the environment's observation function exactly!
        # Environment uses a cutoff of 1.5m, so we clamp to [0.0, 1.5] and divide by 1.5.
        depth_tensor = torch.clamp(depth_tensor, 0.0, 1.5) / 1.5
        
        # Load Keypoints
        kps = torch.tensor(item["keypoints"], dtype=torch.float32) # shape (6, 2)
        
        # Load Visibility Mask
        visibility = torch.tensor(item.get("visibility", [True] * 6), dtype=torch.float32)
        
        # --- Data Augmentation (Only during training) ---
        if self.training:
            # 1. Random Translation (-3 to +3 pixels)
            dx = torch.randint(-3, 4, (1,)).item()
            dy = torch.randint(-3, 4, (1,)).item()
            depth_tensor = torch.roll(depth_tensor, shifts=(dy, dx), dims=(1, 2))
            kps[:, 0] = kps[:, 0] + dx
            kps[:, 1] = kps[:, 1] + dy
            off_screen = (kps[:, 0] < 0) | (kps[:, 0] > 128) | (kps[:, 1] < 0) | (kps[:, 1] > 128)
            visibility[off_screen] = 0.0

            # 2. Depth Dropout — simulate missing pixels from real depth cameras (regions set to 0.0)
            if random.random() < 0.6:
                for _ in range(random.randint(2, 8)):
                    hx = random.randint(0, 127)
                    hy = random.randint(0, 127)
                    hr = random.randint(2, 8)
                    y1, y2 = max(0, hy - hr), min(128, hy + hr)
                    x1, x2 = max(0, hx - hr), min(128, hx + hr)
                    depth_tensor[:, y1:y2, x1:x2] = 0.0
                    # Mark keypoints falling inside the hole as invisible
                    for i in range(len(kps)):
                        u, v = kps[i, 0].item(), kps[i, 1].item()
                        if x1 < u < x2 and y1 < v < y2:
                            visibility[i] = 0.0

            # 3. Depth Scaling — simulate camera calibration error (±5%)
            depth_tensor = depth_tensor * random.uniform(0.95, 1.05)

            # 4. Spatially Correlated Noise — scaled appropriately to the [0.0, 1.0] depth space
            noise_lr = torch.randn(1, 1, 16, 16) * random.uniform(0.0013, 0.004)
            noise_corr = F.interpolate(noise_lr, size=(128, 128), mode='bilinear', align_corners=False)
            depth_tensor = depth_tensor + noise_corr.squeeze(0)

            # 5. Random Cutout — simulate arm/gripper self-occlusion of the scene
            if random.random() < 0.4:
                cx = random.randint(10, 118)
                cy = random.randint(10, 118)
                cw = random.randint(8, 25)
                ch = random.randint(8, 25)
                x1, x2 = max(0, cx - cw // 2), min(128, cx + cw // 2)
                y1, y2 = max(0, cy - ch // 2), min(128, cy + ch // 2)
                depth_tensor[:, y1:y2, x1:x2] = 0.0
                for i in range(len(kps)):
                    u, v = kps[i, 0].item(), kps[i, 1].item()
                    if x1 < u < x2 and y1 < v < y2:
                        visibility[i] = 0.0

            # 6. Depth Range Variation — simulate sensor-to-sensor cutoff differences
            cutoff = random.uniform(0.8, 1.2)
            depth_tensor = torch.clamp(depth_tensor, 0.0, cutoff) / cutoff

            # 7. I.i.d. Gaussian Noise — scaled appropriately to the [0.0, 1.0] depth space
            noise_std = random.uniform(0.00067, 0.002)
            depth_tensor = depth_tensor + torch.randn_like(depth_tensor) * noise_std

        # Ensure final depth tensor is clamped to [0.0, 1.0] range
        depth_tensor = torch.clamp(depth_tensor, 0.0, 1.0)
        
        # Flip coordinates from [u, v] (col, row) to [v, u] (row, col)
        # to match SpatialSoftmax's [y, x] coordinate order exactly!
        kps = kps.flip(dims=(-1,))
        
        # Normalize keypoints to [-1, 1] (Standard for Spatial Softmax)
        kps = (kps / 64.0) - 1.0
        
        # Construct 2D visibility mask
        visibility_2d = visibility.unsqueeze(-1).repeat(1, 2).flatten()
        
        return depth_tensor, kps.flatten(), visibility_2d

class SpatialSoftmax(nn.Module):
    def __init__(self, height, width, temperature=0.5):
        super(SpatialSoftmax, self).__init__()
        self.height = height
        self.width = width
        # Kept as a constant float to match the structure of mjlab's SpatialSoftmax,
        # ensuring standard load_state_dict executes without errors or coordinate shrinkage.
        self.temperature = temperature
        
        # Match mjlab's coordinate grid exactly (indexing='ij')
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
        # Efficient CNN Encoder architecture [32, 64, 64, 6] channels
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Conv2d(64, num_keypoints, kernel_size=1, stride=1, padding=0),
            nn.ELU()
        )
        # 32x32 resolution remains exactly identical
        self.spatial_softmax = SpatialSoftmax(32, 32, temperature=0.5)

    def forward(self, x):
        x = self.cnn(x)
        x = self.spatial_softmax(x) 
        return x

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Find active dataset directories
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
    
    # Shuffle and partition
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
        print(f"Split data pool: {len(train_items)} train samples, {len(val_items)} validation samples.")
        
    train_dataset = KeypointDataset(train_items, training=True)
    dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    
    if has_val:
        val_dataset = KeypointDataset(val_items, training=False)
        val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)
    
    model = PolicyCNNBackbone(num_keypoints=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    
    epochs = 100
    best_val_loss = float("inf")
    best_train_loss = float("inf")
    
    # Toggle for masked MSE loss. 
    # If False, the network is forced to hallucinate occluded keypoints.
    use_visibility_mask = False
    
    print(f"Starting supervised offline training (6 keypoints) on {device}...")
    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        for images, keypoints, visibility in dataloader:
            images, keypoints, visibility = images.to(device), keypoints.to(device), visibility.to(device)
            
            if not use_visibility_mask:
                visibility = torch.ones_like(visibility)
                
            optimizer.zero_grad()
            outputs = model(images)
            
            # Masked MSE loss: only backpropagate gradients for visible keypoints
            diff = (outputs - keypoints) * visibility
            loss = (diff ** 2).sum() / torch.clamp(visibility.sum(), min=1.0)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(dataloader)
        temp_val = model.spatial_softmax.temperature
        
        # --- Validation Phase ---
        if has_val:
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for images, keypoints, visibility in val_dataloader:
                    images, keypoints, visibility = images.to(device), keypoints.to(device), visibility.to(device)
                    
                    if not use_visibility_mask:
                        visibility = torch.ones_like(visibility)
                        
                    outputs = model(images)
                    diff = (outputs - keypoints) * visibility
                    loss = (diff ** 2).sum() / torch.clamp(visibility.sum(), min=1.0)
                    running_val_loss += loss.item()
            avg_val_loss = running_val_loss / len(val_dataloader)
            
            print(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Temp: {temp_val:.4f}")
            
            # Save the model based on validation performance (prevents overfitting)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "pretrained_backbone.pth")
                print(f"  -> Saved new best model with Val Loss {best_val_loss:.6f}!")
        else:
            print(f"Epoch {epoch+1:2d}/{epochs} | Train Loss: {avg_train_loss:.6f} | Temp: {temp_val:.4f}")
            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                torch.save(model.state_dict(), "pretrained_backbone.pth")
                print(f"  -> Saved new best model with Train Loss {best_train_loss:.6f}!")
            
        if avg_train_loss < 0.00005:
            print("Target training loss achieved. Stopping early.")
            break

    print("Offline training complete. Best weights saved to pretrained_backbone.pth")

if __name__ == "__main__":
    train()