import os
import torch
import numpy as np
from PIL import Image, ImageDraw
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
import pal_mjlab.tasks.manipulation.tiago_pro  # Register task
import importlib
from tensordict import TensorDict

def load_class(class_name: str):
    """Loads a python class dynamically from its string path."""
    if ":" in class_name:
        module_path, class_attr = class_name.split(":")
    else:
        parts = class_name.split(".")
        module_path = ".".join(parts[:-1])
        class_attr = parts[-1]
    module = importlib.import_module(module_path)
    return getattr(module, class_attr)

# --- CONFIGURATION ---
# Replace with your actual checkpoint path
TASK_ID = "Mjlab-Manipulation-Lift-Cube-Vision-Curriculum-Pal-Tiago-Pro-v0"
MODEL_PATH = "/home/lorenzobarbieri/Downloads/model_7999.pt"
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

# Create model dynamically and load weights
actor_cfg = rl_cfg.actor
model_cls = load_class(actor_cfg.class_name)
is_convnext = "ConvNeXt" in actor_cfg.class_name

print(f"Instantiating model class {model_cls.__name__} (is_convnext={is_convnext})...")
dummy_obs = TensorDict(obs_dict, batch_size=[1])

model = model_cls(
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

# First load full checkpoint if available to get the PPO policy.
if os.path.exists(MODEL_PATH):
    print(f"Loading full policy from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    policy_sd = checkpoint['actor_state_dict']
    filtered_sd = {k: v for k, v in policy_sd.items() if not k.startswith("cnns.")}
    model.load_state_dict(filtered_sd, strict=False)

# Then load our fresh supervised pretrained backbone weights specifically
if is_convnext:
    backbone_path = "pretrained_convnext.pth"
    if os.path.exists(backbone_path):
        print(f"Loading pretrained ConvNeXt backbone weights from {backbone_path}...")
        backbone_sd = torch.load(backbone_path, map_location=DEVICE)
        # Map ConvNeXt weights to 'convnext.*'
        mapped_sd = {f"convnext.{k}": v for k, v in backbone_sd.items()}
        model.cnns["camera"].load_state_dict(mapped_sd, strict=False)
    else:
        print(f"Warning: {backbone_path} not found!")
else:
    backbone_path = "pretrained_backbone.pth"
    if os.path.exists(backbone_path):
        print(f"Loading pretrained CNN backbone weights from {backbone_path}...")
        backbone_sd = torch.load(backbone_path, map_location=DEVICE)
        model.cnns["camera"].load_state_dict(backbone_sd, strict=False)
    else:
        print(f"Warning: {backbone_path} not found!")


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
        
        # Save unmarked raw depth image first
        img.save(os.path.join(OUTPUT_DIR, f"step_{i:04d}_unmarked.png"))
        
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
        print(f"Saved visualization to {OUTPUT_DIR}/step_{i:04d}.png and {OUTPUT_DIR}/step_{i:04d}_unmarked.png")

    # Step env
    with torch.no_grad():
        # Get mean action (no noise)
        action = model(current_obs)
    env.step(action)

env.close()
print(f"Finished. Visuals saved to {OUTPUT_DIR}")
