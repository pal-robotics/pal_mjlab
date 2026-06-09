import argparse
import importlib
import json
import os
import random
import shutil

import cv2
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from pal_mjlab.tasks.manipulation.mdp.observations import head_camera_keypoints
from PIL import Image, ImageDraw
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


def run_dataset_vis(args):
  data_dir = args.data_dir
  output_dir = args.output_dir or "scratch/dataset_vis"
  num_samples = args.num_samples

  os.makedirs(output_dir, exist_ok=True)
  labels_path = os.path.join(data_dir, "labels.json")
  if not os.path.exists(labels_path):
    raise FileNotFoundError(f"Dataset labels file {labels_path} not found.")

  with open(labels_path, "r") as f:
    labels = json.load(f)

  print(
    f"Loaded {len(labels)} samples. Visualizing random {min(num_samples, len(labels))} samples..."
  )
  sampled_items = random.sample(labels, min(num_samples, len(labels)))

  for i, item in enumerate(sampled_items):
    rgb_path = os.path.join(data_dir, item["rgb"])
    img = Image.open(rgb_path)
    draw = ImageDraw.Draw(img)

    keypoints = item["keypoints"][: args.num_keypoints]
    for kp_idx, kp in enumerate(keypoints):
      u, v = kp
      # Color code: Box corners (Red), fingertips (Green)
      color = (255, 0, 0) if kp_idx < 4 else (0, 255, 0)
      r = 2
      draw.ellipse([u - r, v - r, u + r, v + r], fill=color, outline=(255, 255, 255))

    save_path = os.path.join(output_dir, f"vis_{i:05d}.png")
    img.save(save_path)

  print(f"Done! Dataset visualizations saved to {output_dir}")


def run_noisy_gt_vis(args):
  task_id = args.task_id
  output_dir = args.output_dir or "scratch/noisy_keypoints"
  duration = args.duration
  interval = args.interval
  noise_std = args.noise_std
  device = args.device

  os.makedirs(output_dir, exist_ok=True)
  print(f"Loading task {task_id} in play mode...")
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1

  print("Initializing Mujoco Environment...")
  env = ManagerBasedRlEnv(env_cfg, device=device, render_mode="rgb_array")
  env.reset()

  dt = env.step_dt
  num_steps = int(duration / dt)
  save_every = max(1, int(interval / dt))

  print(
    f"Simulating for {duration}s ({num_steps} steps), saving every {save_every} steps."
  )
  for i in range(num_steps):
    if i % save_every == 0:
      with torch.no_grad():
        kps_flat = head_camera_keypoints(env, noise_std=noise_std)
        keypoints = kps_flat.reshape(args.num_keypoints, 2)
        print(f"Step {i} Noisy Keypoints:\n{keypoints.cpu().numpy()}")

      camera_obs = env.obs_buf["camera"]
      if camera_obs.shape[1] == 3:
        img_data = camera_obs[0].permute(1, 2, 0).cpu().numpy()
        img_vis = (np.clip(img_data, 0, 1.0) * 255).astype(np.uint8)
        img = Image.fromarray(img_vis)
      else:
        if camera_obs.shape[1] == 4:
          img_data = camera_obs[0, 3].cpu().numpy()
        else:
          img_data = camera_obs[0, 0].cpu().numpy()
        img_vis = (np.clip(img_data, 0, 1.0) * 255).astype(np.uint8)
        img = Image.fromarray(img_vis).convert("RGB")

      # Save raw unmarked
      img.save(os.path.join(output_dir, f"step_{i:04d}_unmarked.png"))

      draw = ImageDraw.Draw(img)
      w, h = img.size

      # Plot projected keypoints
      for kp_idx, kp in enumerate(keypoints):
        y_norm, x_norm = kp.cpu().numpy()
        px = (x_norm + 1.0) / 2.0 * w
        py = (y_norm + 1.0) / 2.0 * h

        # Colors: Corners (Red, Green, Blue, Yellow), Fingertips (Magenta, Cyan)
        colors = [
          (255, 0, 0),
          (0, 255, 0),
          (0, 0, 255),
          (255, 255, 0),
          (255, 0, 255),
          (0, 255, 255),
        ]
        color = colors[kp_idx % len(colors)]
        r = 1.5
        draw.ellipse([px - r, py - r, px + r, py + r], fill=color, outline=(0, 0, 0))

      img.save(os.path.join(output_dir, f"step_{i:04d}.png"))
      print(f"Saved step {i:04d} visual to {output_dir}")

    with torch.no_grad():
      actions = torch.zeros(1, env.action_manager.total_action_dim, device=device)
    env.step(actions)

  env.close()
  print("Done! Noisy ground truth keypoints visuals finished.")


def run_model_predict_vis(args):
  task_id = args.task_id
  model_path = args.model_path or "pretrained_backbone.pth"
  output_dir = args.output_dir or "scratch/keypoints"
  duration = args.duration
  interval = args.interval
  device = args.device

  os.makedirs(output_dir, exist_ok=True)
  print(f"Loading task {task_id} in play mode...")
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1
  rl_cfg = load_rl_cfg(task_id)

  print("Initializing Environment...")
  env = ManagerBasedRlEnv(env_cfg, device=device, render_mode="rgb_array")
  obs_dict, _ = env.reset()

  # Create model
  actor_cfg = rl_cfg.actor
  model_cls = load_class(actor_cfg.class_name)
  is_convnext = "ConvNeXt" in actor_cfg.class_name

  print(f"Instantiating model {model_cls.__name__} (is_convnext={is_convnext})...")
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
  ).to(device)

  # Check if checkpoint path is a full model pt or just the pre-trained backbone path
  if os.path.exists(model_path):
    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)

    # Check if it's a full PPO checkpoint or backbone only
    if isinstance(checkpoint, dict) and "actor_state_dict" in checkpoint:
      print("Detected full PPO Checkpoint! Loading policy...")
      policy_sd = checkpoint["actor_state_dict"]
      filtered_sd = {k: v for k, v in policy_sd.items() if not k.startswith("cnns.")}
      model.load_state_dict(filtered_sd, strict=False)

      # Map visual backbone dynamically from checkpoint if present
      backbone_sd = {
        k[12:]: v for k, v in policy_sd.items() if k.startswith("cnns.camera.")
      }
      if backbone_sd:
        model.cnns["camera"].load_state_dict(backbone_sd, strict=False)
    else:
      print("Detected Backbone Weight File! Loading into Camera backbone...")
      if is_convnext:
        mapped_sd = {f"convnext.{k}": v for k, v in checkpoint.items()}
        model.cnns["camera"].load_state_dict(mapped_sd, strict=False)
      else:
        model.cnns["camera"].load_state_dict(checkpoint, strict=False)
  else:
    print(
      f"WARNING: Backbone path {model_path} not found! Model will run with random weights."
    )

  model.eval()

  dt = env.step_dt
  num_steps = int(duration / dt)
  save_every = max(1, int(interval / dt))

  print(
    f"Simulating for {duration}s ({num_steps} steps), saving every {save_every} steps."
  )
  for i in range(num_steps):
    current_obs = TensorDict(env.obs_buf, batch_size=[1])

    if i % save_every == 0:
      with torch.no_grad():
        camera_obs = current_obs["camera"]
        outputs = model.cnns["camera"](camera_obs)
        # First 12 values are keypoints
        kps_flat = outputs[:, : args.num_keypoints * 2]
        keypoints = kps_flat.reshape(1, -1, 2)[0]
        kps_np = keypoints.cpu().numpy()
        print(
          f"Step {i} Predicted Keypoints: min={kps_np.min():.4f}, max={kps_np.max():.4f}"
        )

      if camera_obs.shape[1] == 3:
        img_data = camera_obs[0].permute(1, 2, 0).cpu().numpy()
        img_vis = (np.clip(img_data, 0, 1.0) * 255).astype(np.uint8)
        img = Image.fromarray(img_vis)
      else:
        img_data = camera_obs[0, 0].cpu().numpy()
        img_vis = (np.clip(img_data, 0, 1.0) * 255).astype(np.uint8)
        img = Image.fromarray(img_vis).convert("RGB")

      # Save raw unmarked
      img.save(os.path.join(output_dir, f"step_{i:04d}_unmarked.png"))

      draw = ImageDraw.Draw(img)
      w, h = img.size

      # Plot predicted keypoints
      for kp_idx, kp in enumerate(keypoints):
        y_norm, x_norm = kp.cpu().numpy()
        px = (x_norm + 1.0) / 2.0 * w
        py = (y_norm + 1.0) / 2.0 * h

        colors = [
          (255, 0, 0),
          (0, 255, 0),
          (0, 0, 255),
          (255, 255, 0),
          (255, 0, 255),
          (0, 255, 255),
        ]
        color = colors[kp_idx % len(colors)]
        r = 3
        draw.ellipse([px - r, py - r, px + r, py + r], fill=color, outline=(0, 0, 0))

      img.save(os.path.join(output_dir, f"step_{i:04d}.png"))
      print(f"Saved step {i:04d} visual to {output_dir}")

    with torch.no_grad():
      actions = model(current_obs)
    env.step(actions)

  env.close()
  print("Done! Predicted keypoints visualization finished.")


def run_task_mlp_vis(args):
  task_id = args.task_id
  model_path = args.model_path or "logs/rsl_rl/lift/2026-05-28_17-59-19/model_0.pt"
  output_dir = args.output_dir or "scratch/keypoints_task_vis"
  duration = args.duration
  interval = args.interval
  device = args.device

  os.makedirs(output_dir, exist_ok=True)
  print(f"Loading direct keypoints task {task_id} in play mode...")
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = 1

  # Add high-resolution camera sensor specifically for video capture
  env_cfg.scene.sensors = list(env_cfg.scene.sensors or []) + [
    CameraSensorCfg(
      name="head_realsense_camera",
      height=256,
      width=256,
      data_types=("rgb", "depth"),
      camera_name="robot/head_realsense_camera",
    )
  ]

  print("Initializing Environment...")
  env = ManagerBasedRlEnv(env_cfg, device=device, render_mode="rgb_array")
  obs_dict, _ = env.reset()

  # Load RL MLP Policy Model
  print("Setting up MLP model...")
  rl_cfg = load_rl_cfg(task_id)
  actor_cfg = rl_cfg.actor
  from rsl_rl.models.mlp_model import MLPModel

  dummy_obs = TensorDict(obs_dict, batch_size=[1])
  model = MLPModel(
    obs=dummy_obs,
    obs_groups=getattr(rl_cfg, "obs_groups", None),
    obs_set="actor",
    output_dim=env.action_manager.total_action_dim,
    hidden_dims=actor_cfg.hidden_dims,
    activation=actor_cfg.activation,
    obs_normalization=actor_cfg.obs_normalization,
    distribution_cfg=actor_cfg.distribution_cfg,
  ).to(device)

  if os.path.exists(model_path):
    print(f"Loading model weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    model.eval()
  else:
    print(
      f"WARNING: Checkpoint {model_path} not found! Running model with random weights."
    )

  dt = env.step_dt
  num_steps = int(duration / dt)
  save_every = max(1, int(interval / dt))

  print(
    f"Simulating for {duration}s ({num_steps} steps), saving every {save_every} steps..."
  )
  frames = []

  for i in range(num_steps):
    current_obs = TensorDict(env.obs_buf, batch_size=[1])

    if i % save_every == 0:
      with torch.no_grad():
        kps_flat = head_camera_keypoints(env, noise_std=0.0)
        keypoints = kps_flat.reshape(-1, 2)

      sensor = env.scene.sensors["head_realsense_camera"]
      rgb_data = sensor.data.rgb[0].cpu().numpy()
      depth_data = sensor.data.depth[0, :, :, 0].cpu().numpy()

      img_rgb = Image.fromarray(rgb_data).convert("RGB")
      depth_vis = (np.clip(depth_data, 0, 1.5) / 1.5 * 255).astype(np.uint8)
      img_depth = Image.fromarray(depth_vis).convert("RGB")

      w, h = img_rgb.size

      for img in [img_rgb, img_depth]:
        draw = ImageDraw.Draw(img)
        kps_np = keypoints.cpu().numpy()

        # Draw top box face (edges connecting keypoints 0, 1, 2, 3)
        corners_px = []
        for idx in range(4):
          y_norm, x_norm = kps_np[idx]
          px = (x_norm + 1.0) / 2.0 * w
          py = (y_norm + 1.0) / 2.0 * h
          corners_px.append((px, py))

        draw.line([corners_px[0], corners_px[1]], fill=(0, 255, 0), width=2)
        draw.line([corners_px[1], corners_px[3]], fill=(0, 255, 0), width=2)
        draw.line([corners_px[3], corners_px[2]], fill=(0, 255, 0), width=2)
        draw.line([corners_px[2], corners_px[0]], fill=(0, 255, 0), width=2)

        # Draw keypoint dots
        for kp_idx, (y_norm, x_norm) in enumerate(kps_np):
          px = (x_norm + 1.0) / 2.0 * w
          py = (y_norm + 1.0) / 2.0 * h

          colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
          ]
          color = colors[kp_idx % len(colors)]
          r = 4.0
          draw.ellipse([px - r, py - r, px + r, py + r], fill=color, outline=(0, 0, 0))

      combined_w = w * 2
      combined_h = h
      combined_img = Image.new("RGB", (combined_w, combined_h))
      combined_img.paste(img_rgb, (0, 0))
      combined_img.paste(img_depth, (w, 0))

      draw_comb = ImageDraw.Draw(combined_img)
      draw_comb.text((10, 10), "Head Camera RGB", fill=(255, 255, 255))
      draw_comb.text((w + 10, 10), "Head Camera Depth", fill=(255, 255, 255))

      frame_path = os.path.join(output_dir, f"step_{i:04d}.png")
      combined_img.save(frame_path)

      bgr_frame = cv2.cvtColor(np.array(combined_img), cv2.COLOR_RGB2BGR)
      frames.append(bgr_frame)
      print(f"Step {i:04d}: saved frame to {output_dir}")

    with torch.no_grad():
      action = model(current_obs)
    env.step(action)

  env.close()

  if frames:
    video_path = os.path.join(output_dir, "robot_view.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (combined_w, combined_h))
    for frame in frames:
      out.write(frame)
    out.release()
    print(f"Video saved to {video_path}")

    shutil_path = "scratch/robot_view.mp4"
    shutil.copy(video_path, shutil_path)
    print(f"Video copied to {shutil_path}")


def main():
  parser = argparse.ArgumentParser(description="Unified Keypoints Visualization Suite")
  parser.add_argument(
    "--mode",
    type=str,
    choices=["dataset", "noisy_gt", "model_predict", "task_mlp"],
    required=True,
    help="Visualization mode: dataset, noisy_gt, model_predict, or task_mlp",
  )
  # General Parameters
  parser.add_argument(
    "--task_id",
    type=str,
    default="Mjlab-Manipulation-Lift-Cube-Vision-Pal-Tiago-Pro-v0",
    help="Task ID to visualize (default: Mjlab-Manipulation-Lift-Cube-Vision-Pal-Tiago-Pro-v0)",
  )
  parser.add_argument(
    "--model_path", type=str, default=None, help="Path to checkpoint weights/pth file"
  )
  parser.add_argument(
    "--output_dir", type=str, default=None, help="Output directory to save images/video"
  )
  parser.add_argument(
    "--duration",
    type=float,
    default=2.0,
    help="Simulation duration in seconds (default: 2.0)",
  )
  parser.add_argument(
    "--interval",
    type=float,
    default=0.1,
    help="Time interval between saved frames (default: 0.1)",
  )
  parser.add_argument(
    "--num_keypoints", type=int, default=6, help="Number of keypoints (default: 6)"
  )
  parser.add_argument(
    "--device", type=str, default="cpu", help="Computation device (default: cpu)"
  )

  # Dataset Mode Specific
  parser.add_argument(
    "--data_dir",
    type=str,
    default="dataset_val",
    help="Dataset directory to sample from",
  )
  parser.add_argument(
    "--num_samples", type=int, default=100, help="Number of dataset frames to visualize"
  )

  # Noisy GT Specific
  parser.add_argument(
    "--noise_std",
    type=float,
    default=0.015,
    help="Standard deviation of added noise (default: 0.015)",
  )

  args = parser.parse_args()

  if args.mode == "dataset":
    run_dataset_vis(args)
  elif args.mode == "noisy_gt":
    run_noisy_gt_vis(args)
  elif args.mode == "model_predict":
    run_model_predict_vis(args)
  elif args.mode == "task_mlp":
    # MLP keypoints task requires direct keypoints ID
    if args.task_id == "Mjlab-Manipulation-Lift-Cube-Vision-Pal-Tiago-Pro-v0":
      args.task_id = "Mjlab-Manipulation-Lift-Cube-Keypoints-Pal-Tiago-Pro-v0"
    run_task_mlp_vis(args)


if __name__ == "__main__":
  main()
