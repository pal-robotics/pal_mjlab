import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from mjlab.rl.spatial_softmax import SpatialSoftmaxCNNModel
from tensordict import TensorDict
from mjlab.tasks.registry import load_rl_cfg
import pal_mjlab.tasks.manipulation.tiago_pro  # Register task

# Configuration
TASK_ID = "Mjlab-Manipulation-Lift-Cube-Vision-Pal-Tiago-Pro-v0"
DATA_FILE = "scratch/pretrain_data.pt"
OUTPUT_MODEL = "scratch/model_pretrained.pt"
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

# Load data
data = torch.load(DATA_FILE, map_location="cuda")
images = data["images"]
targets = data["keypoints"]
dataset = TensorDataset(images, targets)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load model config
rl_cfg = load_rl_cfg(TASK_ID)
dummy_obs = TensorDict({
    "actor": torch.randn(1, 1), # Dummy
    "camera": torch.randn(1, 1, 128, 128)
}, batch_size=[1])

model = SpatialSoftmaxCNNModel(
    obs=dummy_obs,
    obs_groups=rl_cfg.obs_groups,
    obs_set="actor",
    output_dim=1, # Dummy
    cnn_cfg=rl_cfg.actor.cnn_cfg,
    hidden_dims=rl_cfg.actor.hidden_dims,
).to(DEVICE)

# Training setup
optimizer = optim.Adam(model.cnns.parameters(), lr=LR)
criterion = nn.MSELoss()

print("Starting training...")
for epoch in range(EPOCHS):
    total_loss = 0
    for batch_images, batch_targets in train_loader:
        batch_images = batch_images.to(DEVICE)
        batch_targets = batch_targets.to(DEVICE) # (B, 2)
        
        # Forward
        # Output is (B, 32 * 2) = (B, 64)
        kps = model.cnns["camera"](batch_images)
        kps = kps.reshape(-1, 32, 2) # (B, 32, 2)
        
        # We want ALL 32 channels to track the same point
        # Broadcast target: (B, 1, 2) -> (B, 32, 2)
        target_expanded = batch_targets.unsqueeze(1).expand(-1, 32, -1)
        
        loss = criterion(kps, target_expanded)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader):.6f}")

# Save state dict
# rsl_rl expects the whole model state dict or at least the actor part
torch.save({
    "actor_state_dict": model.state_dict(),
}, OUTPUT_MODEL)

print(f"Pre-training finished. Model saved to {OUTPUT_MODEL}")
