import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
import pal_mjlab.tasks.manipulation.tiago_pro  # Register task
from mjlab.rl.spatial_softmax import SpatialSoftmaxCNNModel
from tensordict import TensorDict

# Configuration
TASK_ID = "Mjlab-Manipulation-Lift-Cube-Vision-Pal-Tiago-Pro-v0"
MODEL_PATH = "/home/lorenzobarbieri/pal_mjlab_manipulation/pal_mjlab/logs/rsl_rl/lift_depth/2026-05-14_15-47-50/model_0.pt"
OUTPUT_DIR = "/home/lorenzobarbieri/pal_mjlab_manipulation/pal_mjlab/scratch/keypoints"
DURATION = 2.0  # seconds
INTERVAL = 0.1  # seconds
DEVICE = "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load configs
env_cfg = load_env_cfg(TASK_ID, play=True)
env_cfg.scene.num_envs = 1  # Just one for visualization
rl_cfg = load_rl_cfg(TASK_ID)

# Create environment
env = ManagerBasedRlEnv(env_cfg, device=DEVICE, render_mode="rgb_array")
obs_dict, _ = env.reset()

# Create model and load weights
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

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['actor_state_dict'])
model.eval()

# Simulation loop
dt = env.step_dt
num_steps = int(DURATION / dt)
save_every = int(INTERVAL / dt)

print(f"Simulating for {DURATION}s ({num_steps} steps), saving every {save_every} steps.")

for i in range(num_steps):
    # Get current observation
    current_obs = TensorDict(env.obs_buf, batch_size=[1])
    
    if i % save_every == 0:
        # Extract keypoints
        with torch.no_grad():
            camera_obs = current_obs["camera"] # (B, 1, 128, 128)
            keypoints = model.cnns["camera"](camera_obs) # (1, 64)
            keypoints = keypoints.reshape(-1, 2) # (32, 2)
            
            # Print numerical range to console
            if i % save_every == 0:
                kps_np = keypoints.cpu().numpy()
                print(f"Step {i} Keypoints: min={kps_np.min():.4f}, max={kps_np.max():.4f}, mean={kps_np.mean():.4f}")

        
        # Capture image (Depth is used in this task)
        img_data = camera_obs[0, 0].cpu().numpy()
        # Invert or scale depth for visibility if needed. 
        # Here we just use it as is (0-1).
        img_vis = (img_data * 255).astype(np.uint8)
        img = Image.fromarray(img_vis).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Plot keypoints
        w, h = img.size
        num_kps = len(keypoints)
        for kp_idx, kp in enumerate(keypoints):
            y_norm, x_norm = kp.cpu().numpy()
            
            # Map [-1, 1] to [0, size]
            px = (x_norm + 1.0) / 2.0 * w
            py = (y_norm + 1.0) / 2.0 * h
            
            # Color coding: cycle through some distinct colors
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), 
                (255, 255, 0), (255, 0, 255), (0, 255, 255),
                (255, 128, 0), (255, 0, 128), (128, 255, 0),
                (0, 255, 128), (128, 0, 255), (0, 128, 255)
            ]
            color = colors[kp_idx % len(colors)]
            
            r = 2 # Slightly larger for visibility
            draw.ellipse([px-r, py-r, px+r, py+r], fill=color, outline=(0,0,0))
        
        img.save(os.path.join(OUTPUT_DIR, f"step_{i:04d}.png"))
        print(f"Saved step {i}")

    # Step env
    with torch.no_grad():
        action = model(current_obs)
    env.step(action)

env.close()
print(f"Finished. Visuals saved to {OUTPUT_DIR}")
