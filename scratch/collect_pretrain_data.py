import argparse
import json
import os

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from pal_mjlab.tasks.manipulation.mdp.observations import (
  object_pose_6d_in_robot_root_frame,
)
from pal_mjlab.tasks.manipulation.tiago_pro.env_cfgs import lift_vision_env_cfg
from PIL import Image
from rsl_rl.modules import EmpiricalNormalization

_BOX_HALF_SIZE = 0.025

# --- Expert Actor Definition with Empirical Normalization ---


class OracleExpert(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.obs_normalizer = EmpiricalNormalization(36)
    self.mlp = torch.nn.Sequential(
      torch.nn.Linear(36, 512),
      torch.nn.ELU(),
      torch.nn.Linear(512, 256),
      torch.nn.ELU(),
      torch.nn.Linear(256, 128),
      torch.nn.ELU(),
      torch.nn.Linear(128, 8),
    )

  def forward(self, obs_dict):
    # The oracle expects the 35D state:
    # joint_pos(7) + joint_vel(7) + actions(7) + object_pos(3) + object_ori(4) + target_pos(3) + gripper_pos(1) + ee_pos(3)
    # This matches exactly the first 35 features of the 'critic' observation group.
    critic_obs = obs_dict["critic"]
    oracle_obs = critic_obs[:, :36]  # Take exactly the first 35 features
    normalized_obs = self.obs_normalizer(oracle_obs)
    return self.mlp(normalized_obs)


# --- Projection Utilities ---


def project_3d_to_2d(points_3d_w, cam_pos, cam_quat, K_matrix, width, height):
  from mjlab.utils.lab_api.math import quat_apply, quat_inv

  B, N, _ = points_3d_w.shape
  cam_pos_exp = cam_pos.unsqueeze(1).expand(B, N, 3)
  cam_quat_exp = cam_quat.unsqueeze(1).expand(B, N, 4)
  points_c = quat_apply(quat_inv(cam_quat_exp), points_3d_w - cam_pos_exp)
  x = points_c[..., 0]
  y = -points_c[..., 1]
  z = -points_c[..., 2]
  fx, fy, cx, cy = K_matrix
  u = (x * fx / z) + cx
  v = (y * fy / z) + cy
  return torch.stack([u, v], dim=-1)


def get_3d_keypoints(env):
  hx, hy, hz = _BOX_HALF_SIZE, _BOX_HALF_SIZE, 1.5 * _BOX_HALF_SIZE
  local_corners = torch.tensor(
    [[hx, hy, hz], [hx, -hy, hz], [-hx, hy, hz], [-hx, -hy, hz]], device=env.device
  )
  box = env.scene["box"]
  box_pos = (
    box.data.root_pos_w
    if hasattr(box.data, "root_pos_w")
    else box.data.geom_pos_w[:, 0]
  )
  box_quat = (
    box.data.root_quat_w
    if hasattr(box.data, "root_quat_w")
    else box.data.geom_quat_w[:, 0]
  )

  num_envs = env.num_envs
  box_pos_exp = box_pos.unsqueeze(1).expand(-1, 4, -1)
  box_quat_exp = box_quat.unsqueeze(1).expand(-1, 4, -1)
  local_corners_exp = local_corners.unsqueeze(0).expand(num_envs, -1, -1)
  from mjlab.utils.lab_api.math import quat_apply

  corners_3d_w = box_pos_exp + quat_apply(box_quat_exp, local_corners_exp)
  robot = env.scene["robot"]
  fingertip_site_names = [s for s in robot.site_names if "fingertip" in s]
  fingertip_pos_w = robot.data.site_pos_w[
    :, [robot.site_names.index(name) for name in fingertip_site_names]
  ]
  return torch.cat([corners_3d_w, fingertip_pos_w], dim=1)


# --- Collection Script ---


def collect_data(args):
  num_samples = args.num_samples
  save_dir = args.save_dir
  num_envs = args.num_envs
  reset_steps = args.reset_steps
  models = [os.path.expanduser(m) for m in args.models]

  os.makedirs(save_dir, exist_ok=True)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # 1. Instantiate environment with RGB type (so RGB observations are processed)
  print(f"Initializing environment with {num_envs} envs...")
  cfg = lift_vision_env_cfg(cam_type="rgb")
  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda")

  # 2. Instantiate Expert model
  expert = OracleExpert().to(device)

  # 3. Camera Params
  camera_name = "head_realsense_camera"
  width, height = cfg.scene.sensors[-1].width, cfg.scene.sensors[-1].height
  fov_y = 60.0
  f = (height / 2.0) / np.tan(np.deg2rad(fov_y / 2.0))
  K = (f, f, width / 2.0, height / 2.0)

  dataset_labels = []
  obs, _ = env.reset()
  steps = 0

  print(
    f"Starting Expert-Guided Collection for {num_samples} samples across {len(models)} model(s)..."
  )

  # We will split samples equally between all provided models
  samples_per_model = num_samples // len(models)

  for model_idx, model_path in enumerate(models):
    print(
      f"\n--- Loading Expert Policy ({model_idx + 1}/{len(models)}) from {model_path} ---"
    )
    if not os.path.exists(model_path):
      raise FileNotFoundError(f"Expert model path {model_path} does not exist.")

    checkpoint = torch.load(model_path, map_location=device)
    # Load state dict strictly omitting standard deviation parameter but keeping normalizer
    expert.load_state_dict(checkpoint["actor_state_dict"], strict=False)
    expert.eval()

    # Reset environment states at model transition to avoid carry-over
    obs, _ = env.reset()
    steps = 0

    target_count = (model_idx + 1) * samples_per_model
    # Ensure the last model collects any remaining remainder samples
    if model_idx == len(models) - 1:
      target_count = num_samples

    while len(dataset_labels) < target_count:
      with torch.no_grad():
        camera = env.scene.sensors[camera_name]
        expert_obs = {"critic": obs["critic"]}
        actions = expert(expert_obs)
        if actions.shape[1] < env.action_manager.total_action_dim:
          pad_dim = env.action_manager.total_action_dim - actions.shape[1]
          actions = torch.cat(
            [actions, torch.zeros(actions.shape[0], pad_dim, device=actions.device)],
            dim=-1,
          )

      obs, _, _, _, _ = env.step(actions)
      steps += 1

      # Retrieve all batch items
      rgb_images = camera.data.rgb.cpu().numpy()
      depth_images = camera.data.depth.cpu().numpy()
      keypoints_3d = get_3d_keypoints(env)

      # Extract ground truth 6D pose of object in robot root frame
      poses_6d = object_pose_6d_in_robot_root_frame(env, "lift_height").cpu().numpy()

      from mjlab.utils.lab_api.math import quat_from_matrix

      sim_data = env.sim.data
      cam_pos = sim_data.cam_xpos[:, camera.camera_idx]
      cam_quat = quat_from_matrix(sim_data.cam_xmat[:, camera.camera_idx])

      keypoints_2d = project_3d_to_2d(keypoints_3d, cam_pos, cam_quat, K, width, height)
      kps_raw_batch = keypoints_2d.cpu().numpy()

      # Calculate expected local Z depths to check for occlusion
      from mjlab.utils.lab_api.math import quat_apply, quat_inv

      B, N, _ = keypoints_3d.shape
      cam_pos_exp = cam_pos.unsqueeze(1).expand(B, N, 3)
      cam_quat_exp = cam_quat.unsqueeze(1).expand(B, N, 4)
      points_c = quat_apply(quat_inv(cam_quat_exp), keypoints_3d - cam_pos_exp)
      z_expected_batch = -points_c[:, :, 2].cpu().numpy()

      for b in range(num_envs):
        if len(dataset_labels) >= target_count:
          break

        rgb_image = rgb_images[b]
        depth_image = depth_images[b]
        kps_raw = kps_raw_batch[b]
        z_expected = z_expected_batch[b]
        pose_6d_val = poses_6d[b]

        # Calculate visibility mask based on depth buffer comparison
        visibility_list = []
        for idx, (u, v) in enumerate(kps_raw):
          col = int(np.clip(u, 0, width - 1))
          row = int(np.clip(v, 0, height - 1))
          observed_z = depth_image[row, col, 0]
          expected_z = z_expected[idx]
          is_visible = bool(observed_z >= expected_z - 0.02)
          visibility_list.append(is_visible)

        # Visibility check (ensure keypoints are at least on-screen or within bounds)
        valid = True
        for u, v in kps_raw:
          if u < -10 or u > 138 or v < -10 or v > 138:
            valid = False
            break

        if valid:
          filename = f"rgb_{len(dataset_labels):05d}.png"
          # Save as PNG
          img = Image.fromarray(rgb_image.astype(np.uint8))
          img.save(os.path.join(save_dir, filename))

          dataset_labels.append(
            {
              "rgb": filename,
              "keypoints": kps_raw.tolist(),
              "visibility": visibility_list,
              "pose_6d": pose_6d_val.tolist(),
            }
          )
          if len(dataset_labels) % 100 == 0:
            print(f"Collected {len(dataset_labels)}/{num_samples}...")

      if steps % reset_steps == 0:
        env.reset()

  with open(os.path.join(save_dir, "labels.json"), "w") as f:
    json.dump(dataset_labels, f, indent=4)
  print(f"\nCollection complete. Total samples collected: {len(dataset_labels)}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Collect pretrain data from expert policy models."
  )
  parser.add_argument(
    "--set",
    type=str,
    choices=["train", "validation"],
    default="train",
    help="Whether to collect training or validation dataset.",
  )
  parser.add_argument(
    "--num_samples",
    type=int,
    default=None,
    help="Total number of samples to collect. Defaults to 100000 for train, 10000 for validation.",
  )
  parser.add_argument(
    "--save_dir",
    type=str,
    default=None,
    help="Output directory. Defaults to 'dataset' for train, 'dataset_val' for validation.",
  )
  parser.add_argument(
    "--num_envs", type=int, default=64, help="Number of parallel environments to run."
  )
  parser.add_argument(
    "--reset_steps",
    type=int,
    default=40,
    help="Number of steps after which the environments reset. Default is 40.",
  )
  parser.add_argument(
    "--models",
    type=str,
    nargs="+",
    default=None,
    help="Paths to expert models. If not specified, defaults are selected based on --set.",
  )

  args = parser.parse_args()

  # Apply defaults based on chosen set
  if args.set == "train":
    if args.num_samples is None:
      args.num_samples = 100000
    if args.save_dir is None:
      args.save_dir = "dataset_rgb"
    if args.models is None:
      args.models = [
        "/home/lorenzobarbieri/pal_mjlab_manipulation/pal_mjlab/logs/rsl_rl/lift/2026-05-20_17-53-58/model_3500.pt"
      ]
  else:  # validation
    if args.num_samples is None:
      args.num_samples = 10000
    if args.save_dir is None:
      args.save_dir = "dataset_rgb_val"
    if args.models is None:
      args.models = [
        "/home/lorenzobarbieri/pal_mjlab_manipulation/pal_mjlab/logs/rsl_rl/lift/2026-05-20_17-53-58/model_3500.pt"
      ]

  collect_data(args)
