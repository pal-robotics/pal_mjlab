import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class KeypointDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
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
        
        # Normalize depth [0, 1.0m] -> [0, 1]
        depth_tensor = torch.from_numpy(depth_map).permute(2, 0, 1).float()
        depth_tensor = torch.clamp(depth_tensor, 0, 1.0) / 1.0 
        
        # Normalize keypoints to [-1, 1] (Standard for Spatial Softmax)
        kps = torch.tensor(item["keypoints"], dtype=torch.float32) 
        kps = (kps / 64.0) - 1.0 
        
        return depth_tensor, kps.flatten()

class SpatialSoftmax(nn.Module):
    def __init__(self, height, width, channels, temperature=1.0):
        super(SpatialSoftmax, self).__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.temperature = temperature
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self.width),
            np.linspace(-1.0, 1.0, self.height)
        )
        self.register_buffer('pos_x', torch.from_numpy(pos_x).float())
        self.register_buffer('pos_y', torch.from_numpy(pos_y).float())

    def forward(self, feature):
        B, C, H, W = feature.size()
        feature = feature.view(B, C, H * W)
        softmax_attention = F.softmax(feature / self.temperature, dim=-1)
        softmax_attention = softmax_attention.view(B, C, H, W)
        expected_x = torch.sum(softmax_attention * self.pos_x, dim=(2, 3))
        expected_y = torch.sum(softmax_attention * self.pos_y, dim=(2, 3))
        return torch.stack([expected_x, expected_y], dim=-1).view(B, C * 2)

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
        self.spatial_softmax = SpatialSoftmax(32, 32, num_keypoints)

    def forward(self, x):
        x = self.cnn(x)
        x = self.spatial_softmax(x) 
        return x

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = KeypointDataset("dataset")
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = PolicyCNNBackbone(num_keypoints=6).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    
    epochs = 1000
    print(f"Starting supervised offline training (6 keypoints)...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, keypoints in dataloader:
            images, keypoints = images.to(device), keypoints.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, keypoints)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        if avg_loss < 0.00005:
            break

    torch.save(model.state_dict(), "pretrained_backbone.pth")
    print("Offline training complete. Weights saved to pretrained_backbone.pth")

if __name__ == "__main__":
    train()