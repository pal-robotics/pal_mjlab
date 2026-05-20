import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class KeypointDataset(Dataset):
    def __init__(self, data_dir, training=True):
        self.data_dir = data_dir
        self.training = training
        with open(os.path.join(data_dir, "labels.json"), "r") as f:
            self.labels = json.load(f)
            
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.labels[idx]
        
        # Load Depth
        depth_path = os.path.join(self.data_dir, item["depth"])
        depth_map = np.load(depth_path)
        if len(depth_map.shape) == 2:
            depth_map = depth_map[:, :, np.newaxis]
        
        depth_tensor = torch.from_numpy(depth_map).permute(2, 0, 1).float()
        
        # Load Keypoints
        kps = torch.tensor(item["keypoints"], dtype=torch.float32) # shape (6, 2)
        
        # Load Visibility Mask
        visibility = torch.tensor(item.get("visibility", [True] * 6), dtype=torch.float32)
        
        # --- Data Augmentation (Only during training) ---
        if self.training:
            # 1. Random Translation (Translation range: -3 to 3 pixels)
            dx = torch.randint(-3, 4, (1,)).item()
            dy = torch.randint(-3, 4, (1,)).item()
            
            # Translate depth map (roll shifts image, wraps edges)
            depth_tensor = torch.roll(depth_tensor, shifts=(dy, dx), dims=(1, 2))
            
            # Translate keypoints in pixel space [0, 128]
            kps[:, 0] = kps[:, 0] + dx
            kps[:, 1] = kps[:, 1] + dy
            
            # If a keypoint shifts off-screen, mark it as invisible
            off_screen = (kps[:, 0] < 0) | (kps[:, 0] > 128) | (kps[:, 1] < 0) | (kps[:, 1] > 128)
            visibility[off_screen] = 0.0
            
            # 2. Gaussian Noise (Standard dev between 1mm and 3mm)
            noise_std = torch.rand(1).item() * 0.002 + 0.001
            noise = torch.randn_like(depth_tensor) * noise_std
            depth_tensor = depth_tensor + noise

        # Normalize depth [0, 1.0m] -> [0, 1.0] and clamp to range
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
    def __init__(self, height, width, channels, temperature=0.5):
        super(SpatialSoftmax, self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        # Learnable temperature parameter initialized to 0.5
        self.temperature = nn.Parameter(torch.tensor(temperature, dtype=torch.float32))
        
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
        temp = torch.clamp(self.temperature, min=0.01)
        weights = torch.softmax(features / temp, dim=-1)
        expected_x = (weights * self.pos_x).sum(dim=-1)
        expected_y = (weights * self.pos_y).sum(dim=-1)
        return torch.stack([expected_x, expected_y], dim=-1).reshape(B, C * 2)

class PolicyCNNBackbone(nn.Module):
    def __init__(self, num_keypoints=6):
        super().__init__()
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
        self.spatial_softmax = SpatialSoftmax(32, 32, num_keypoints, temperature=0.5)

    def forward(self, x):
        x = self.cnn(x)
        x = self.spatial_softmax(x) 
        return x

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = KeypointDataset("dataset", training=True)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = PolicyCNNBackbone(num_keypoints=6).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    
    epochs = 1000
    best_loss = float("inf")
    
    # Toggle for masked MSE loss. 
    # If False, the network is forced to hallucinate occluded keypoints.
    use_visibility_mask = True
    
    print(f"Starting supervised offline training (6 keypoints)...")
    for epoch in range(epochs):
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
            
        avg_loss = running_loss / len(dataloader)
        temp_val = model.spatial_softmax.temperature.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Temp: {temp_val:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "pretrained_backbone.pth")
            print(f"  -> Saved new best model with loss {best_loss:.6f}!")
            
        if avg_loss < 0.00005:
            break

    print("Offline training complete. Best weights saved to pretrained_backbone.pth")

if __name__ == "__main__":
    train()