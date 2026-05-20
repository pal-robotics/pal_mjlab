import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
import pal_mjlab.tasks.manipulation.tiago_pro  # Register task
from mjlab.rl.spatial_softmax import SpatialSoftmaxCNNModel
from tensordict import TensorDict

# --- CONFIGURATION ---
# Replace with your actual checkpoint path
TASK_ID = "Mjlab-Manipulation-Lift-Cube-Vision-Curriculum-Pal-Tiago-Pro-v0"
MODEL_PATH = "/home/lorenzobarbieri/pal_mjlab_manipulation/pal_mjlab/logs/rsl_rl/lift_depth/2026-05-18_10-18-26/model_3000.pt"
OUTPUT_DIR = "scratch/keypoints"
DURATION = 2.0  # seconds
INTERVAL = 0.1  # seconds
DEVICE = "cpu" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading task {TASK_ID}...")
# Load configs
env_cfg = load_env_cfg(TASK_ID, play=True)
env_cfg.scene.num_envs = 1  # Just one for visualization
rl_cfg = load_rl_cfg(TASK_ID)

# Create environment
print("Initializing Environment...")
env = ManagerBasedRlEnv(env_cfg, device=DEVICE, render_mode="rgb_array")
obs_dict, _ = env.reset()

# Create model and load weights
print(f"Loading Model from {MODEL_PATH}...")
dummy_obs = TensorDict(obs_dict, batch_size=[1])
actor_cfg = rl_cfg.actor
model = SpatialSoftmaxCNNModel(
    obs=dummy_obs,
    obs_groups=rl_cfg.obs_groups,
    obs_set="actor",
    output_dim=env.action_manager.total_action_dim,
    cnn_cfg=actor_cfg.cnn_cfg,
    hidden_dims=actor_cfg.hidden_dims,
    activation=actor_cfg.activation,
    obs_normalization=actor_cfg.obs_normalization,
    distribution_cfg=actor_cfg.distribution_cfg,
).to(DEVICE)

# First load full checkpoint if available to get the expert MLP policy
if os.path.exists(MODEL_PATH):
    print(f"Loading full policy from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['actor_state_dict'], strict=False)

# Then load our fresh supervised pretrained CNN backbone weights specifically
if os.path.exists("pretrained_backbone.pth"):
    print("Loading pretrained CNN backbone weights from pretrained_backbone.pth...")
    backbone_sd = torch.load("pretrained_backbone.pth", map_location=DEVICE)
    model.cnns["camera"].load_state_dict(backbone_sd, strict=False)
else:
    print("Warning: pretrained_backbone.pth not found!")

model.eval()

# Simulation loop
dt = env.step_dt
num_steps = int(DURATION / dt)
save_every = max(1, int(INTERVAL / dt))

print(f"Simulating for {DURATION}s ({num_steps} steps), saving every {save_every} steps.")

for i in range(num_steps):
    # Get current observation
    current_obs = TensorDict(env.obs_buf, batch_size=[1])
    
    if i % save_every == 0:
        # Extract keypoints
        with torch.no_grad():
            camera_obs = current_obs["camera"] # (B, 1, 128, 128)
            # The CNN + SpatialSoftmax is in model.cnns["camera"]
            # It outputs (B, num_keypoints * 2)
            kps_flat = model.cnns["camera"](camera_obs) 
            keypoints = kps_flat.reshape(1, -1, 2)[0] # (6, 2)
            
            # Print numerical range to console
            kps_np = keypoints.cpu().numpy()
            print(f"Step {i} Keypoints: min={kps_np.min():.4f}, max={kps_np.max():.4f}")

        # Capture image (Depth is used in this task)
        img_data = camera_obs[0, 0].cpu().numpy()
        # Scale depth (0.0 to 1.0m) to 0-255
        img_vis = (np.clip(img_data, 0, 1.0) * 255).astype(np.uint8)
        img = Image.fromarray(img_vis).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Plot keypoints
        w, h = img.size
        for kp_idx, kp in enumerate(keypoints):
            # Spatial Softmax outputs (y, x) in [-1, 1]
            y_norm, x_norm = kp.cpu().numpy()
            
            # Map [-1, 1] to [0, size]
            px = (x_norm + 1.0) / 2.0 * w
            py = (y_norm + 1.0) / 2.0 * h
            
            # Colors for 6 keypoints:
            # 0-3: Box corners, 4-5: Gripper fingertips
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), # Box (Red, Green, Blue, Yellow)
                (255, 0, 255), (0, 255, 255) # Gripper (Magenta, Cyan)
            ]
            color = colors[kp_idx % len(colors)]
            
            r = 3
            draw.ellipse([px-r, py-r, px+r, py+r], fill=color, outline=(0,0,0))
        
        img.save(os.path.join(OUTPUT_DIR, f"step_{i:04d}.png"))
        print(f"Saved visualization to {OUTPUT_DIR}/step_{i:04d}.png")

    # Step env
    with torch.no_grad():
        # Get mean action (no noise)
        action = model(current_obs)
    env.step(action)

env.close()
print(f"Finished. Visuals saved to {OUTPUT_DIR}")
