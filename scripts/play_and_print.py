import argparse
import math
import os
import sys

import cv2
import mjlab.tasks  # noqa: F401
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply, quat_inv
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

# Sourcing the filters from the workspace
sys.path.append("/home/lorenzobarbieri/exchange/tiago_pro_sim_ws/src")
try:
  from filters import (
    AdaptiveStaticKF,
    ConstantVelocityKF,
    ExponentialMovingAverage,
    ExponentialMovingAverage1D,
    ExponentialMovingAverageYaw,
    StaticState1DKF,
    YawKF,
  )
except ImportError:

  class ConstantVelocityKF:
    def __init__(self, init_pos, init_vel=0.0, q_accel=0.05, r_pos=1e-5):
      self.x = np.array([init_pos, init_vel], dtype=np.float32)
      self.P = np.array([[1.0, 0.0], [0.0, 10.0]], dtype=np.float32)
      self.q_accel = q_accel
      self.r_pos = r_pos

    def predict(self, dt, q_scale=1.0):
      F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float32)
      q = self.q_accel * q_scale
      Q = q * np.array(
        [[(dt**3) / 3.0, (dt**2) / 2.0], [(dt**2) / 2.0, dt]], dtype=np.float32
      )
      self.x = F @ self.x
      self.P = F @ self.P @ F.T + Q

    def update(self, z_pos, r_pos=None):
      if r_pos is None:
        r_pos = self.r_pos
      H = np.array([[1.0, 0.0]], dtype=np.float32)
      R = np.array([[r_pos]], dtype=np.float32)
      innovation = z_pos - (H @ self.x)[0]
      if abs(innovation) > 0.05:
        return False
      S = H @ self.P @ H.T + R
      K = self.P @ H.T / S[0, 0]
      self.x = self.x + K * innovation
      self.P = (np.eye(2) - K @ H) @ self.P
      return True

  class AdaptiveStaticKF:
    def __init__(self, init_pos, q_base=1e-6, r_pos=1e-3, window=10):
      self.x = np.array([init_pos, 0.0], dtype=np.float32)
      self.P = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
      self.q_base = q_base
      self.r_pos = r_pos
      self.window = window
      self.innovations = []

    def predict(self, dt, q_scale=1.0):
      pass

    def update(self, z_pos, r_pos=None):
      if r_pos is None:
        r_pos = self.r_pos
      H = np.array([[1.0, 0.0]], dtype=np.float32)
      innovation = z_pos - (H @ self.x)[0]
      self.innovations.append(innovation)
      if len(self.innovations) > self.window:
        self.innovations.pop(0)
      q = self.q_base
      if len(self.innovations) >= 2:
        q = max(q, float(np.var(self.innovations) * 0.1))
      Q = np.array([[q, 0.0], [0.0, q]], dtype=np.float32)
      self.P = self.P + Q
      R = np.array([[r_pos]], dtype=np.float32)
      S = H @ self.P @ H.T + R
      K = self.P @ H.T / S[0, 0]
      self.x = self.x + K * innovation
      self.P = (np.eye(2) - K @ H) @ self.P
      return True

  class YawKF:
    def __init__(self, init_yaw, q_yaw=0.001, r_yaw=0.05):
      self.x = np.array([math.cos(init_yaw), math.sin(init_yaw)], dtype=np.float32)
      self.P = np.eye(2, dtype=np.float32) * 1.0
      self.q_yaw = q_yaw
      self.r_yaw = r_yaw

    def predict(self, dt, q_scale=1.0):
      q = self.q_yaw * q_scale
      self.P = self.P + np.eye(2, dtype=np.float32) * q

    def update(self, z_yaw, r_yaw=None):
      if r_yaw is None:
        r_yaw = self.r_yaw
      z = np.array([math.cos(z_yaw), math.sin(z_yaw)], dtype=np.float32)
      innovation = z - self.x
      R = np.eye(2, dtype=np.float32) * r_yaw
      S = self.P + R
      K = self.P @ np.linalg.inv(S)
      self.x = self.x + K @ innovation
      norm = np.linalg.norm(self.x)
      if norm > 1e-5:
        self.x = self.x / norm
      self.P = (np.eye(2) - K) @ self.P
      return True

    def get_yaw(self):
      return float(math.atan2(self.x[1], self.x[0]))

  class StaticState1DKF:
    def __init__(self, init_val, q_val=1e-5, r_val=0.01):
      self.x = float(init_val)
      self.P = 1.0
      self.q_val = q_val
      self.r_val = r_val

    def predict(self, dt, q_scale=1.0):
      self.P = self.P + self.q_val * q_scale

    def update(self, z_val, r_val=None):
      if r_val is None:
        r_val = self.r_val
      innovation = z_val - self.x
      S = self.P + r_val
      K = self.P / S
      self.x = self.x + K * innovation
      self.P = (1.0 - K) * self.P
      return True

  class ExponentialMovingAverage:
    def __init__(self, init_pos, alpha=0.40):
      self.x = np.array([init_pos, 0.0], dtype=np.float32)
      self.alpha = alpha

    def predict(self, dt, q_scale=1.0):
      pass

    def update(self, z_pos, r_pos=None):
      self.x[0] = self.alpha * z_pos + (1.0 - self.alpha) * self.x[0]
      self.x[1] = 0.0
      return True

  class ExponentialMovingAverageYaw:
    def __init__(self, init_yaw, alpha=0.20):
      self.x = np.array([math.cos(init_yaw), math.sin(init_yaw)], dtype=np.float32)
      self.alpha = alpha

    def predict(self, dt, q_scale=1.0):
      pass

    def update(self, z_yaw, r_yaw=None):
      z = np.array([math.cos(z_yaw), math.sin(z_yaw)], dtype=np.float32)
      self.x = self.alpha * z + (1.0 - self.alpha) * self.x
      norm = np.linalg.norm(self.x)
      if norm > 1e-5:
        self.x = self.x / norm
      return True

    def get_yaw(self):
      return float(math.atan2(self.x[1], self.x[0]))

  class ExponentialMovingAverage1D:
    def __init__(self, init_val, alpha=0.20):
      self.x = float(init_val)
      self.alpha = alpha

    def predict(self, dt, q_scale=1.0):
      pass

    def update(self, z_val, r_val=None):
      self.x = self.alpha * z_val + (1.0 - self.alpha) * self.x
      return True


def rotate_by_quat_np(v, q):
  v_arr = np.array(v, dtype=np.float32)
  is_single = v_arr.ndim == 1
  if is_single:
    v_arr = v_arr[np.newaxis, :]
  w, x, y, z = q[0], q[1], q[2], q[3]
  q_xyz = np.array([x, y, z], dtype=np.float32)
  cross1 = np.cross(q_xyz, v_arr)
  cross2 = np.cross(q_xyz, cross1 + w * v_arr)
  v_rot = v_arr + 2.0 * cross2
  if is_single:
    return v_rot[0]
  return v_rot


def wrap_angle(angle):
  return (angle + math.pi) % (2.0 * math.pi) - math.pi


def load_class(class_name: str):
  """Loads a python class dynamically from its string path, with fallbacks for RSL RL models."""
  import importlib

  if class_name == "MLPModel":
    from rsl_rl.models.mlp_model import MLPModel

    return MLPModel
  elif class_name == "CNNModel":
    from rsl_rl.models.cnn_model import CNNModel

    return CNNModel

  if ":" in class_name:
    module_path, class_attr = class_name.split(":")
  else:
    parts = class_name.split(".")
    if len(parts) > 1:
      module_path = ".".join(parts[:-1])
      class_attr = parts[-1]
    else:
      raise ValueError(f"Cannot resolve class name: {class_name}")

  module = importlib.import_module(module_path)
  return getattr(module, class_attr)


TASK_ID = "Mjlab-Manipulation-Lift-Cube-Pal-Tiago-Pro-v0"


def scan_box_mass(env_wrapped, num_episodes: int, device: str) -> None:
  """Reset the environment for `num_episodes` episodes, taking a single step in
  each, and record the min/max box mass encountered (e.g. from domain
  randomization applied on reset)."""
  inner_env = env_wrapped.unwrapped
  action_shape = env_wrapped.unwrapped.action_space.shape
  zero_action = torch.zeros(action_shape, device=device)

  min_mass = float("inf")
  max_mass = float("-inf")
  min_episode = -1
  max_episode = -1

  print(f"Scanning box mass over {num_episodes} episodes (1 step per episode)...")
  for episode in range(num_episodes):
    env_wrapped.reset()

    box_entity = inner_env.scene["box"]
    body_id = box_entity.indexing.body_ids[0]

    # Take a single step so the episode actually advances.
    env_wrapped.step(zero_action)

    mass = inner_env.sim.model.body_mass[0, body_id].item()

    if mass < min_mass:
      min_mass = mass
      min_episode = episode
    if mass > max_mass:
      max_mass = mass
      max_episode = episode

    if (episode + 1) % 50 == 0 or episode == num_episodes - 1:
      print(
        f"  Episode {episode + 1:4d}/{num_episodes} | mass = {mass:.5f} kg | "
        f"running min = {min_mass:.5f} kg | running max = {max_mass:.5f} kg"
      )

  print("\n" + "=" * 80)
  print(f"Mass scan complete over {num_episodes} episodes")
  print(f"  Min mass: {min_mass:.5f} kg (episode {min_episode})")
  print(f"  Max mass: {max_mass:.5f} kg (episode {max_episode})")
  print("=" * 80)


class PrintingPolicy:
  def __init__(self, action_shape, env_wrapped, args, model=None):
    self.action_shape = action_shape
    self.env_wrapped = env_wrapped
    self.inner_env = env_wrapped.unwrapped
    self.obs_manager = self.inner_env.observation_manager
    self.names = self.obs_manager.active_terms.get("actor", [])
    self.shapes = self.obs_manager.group_obs_term_dim.get("actor", [])
    self.model = model
    self.args = args

    # Initialize tracking variables (since num_envs is 1)
    self.kf = None
    self.hsv_ref = None
    self.is_grasped = False
    self.grasp_override_active = False
    self.episode_success = False
    self.printed_success_this_episode = False
    self.episode_count = 0

    # Grab scene references
    self.robot = self.inner_env.scene["robot"]
    self.box = self.inner_env.scene["box"]
    self.command = self.inner_env.command_manager.get_term("lift_height")

    # Find the grasp site
    grasp_site_idx, _ = self.robot.find_sites(
      ["gripper_right_grasping_site"], preserve_order=True
    )
    self.grasp_idx = grasp_site_idx[0]

    # Load YOLO model if enabled
    self.yolo_model = None
    if args.enable_yolo:
      print(f"Loading YOLO model from {args.yolo_model}...")
      from ultralytics import YOLO

      self.yolo_model = YOLO(args.yolo_model)
      yolo_device = "cuda:0" if torch.cuda.is_available() else "cpu"
      self.yolo_model.to(yolo_device)
      print(f"YOLO model loaded successfully on device: {yolo_device}")

  def __call__(self, obs) -> torch.Tensor:
    # Get current step and time
    step = self.inner_env.episode_length_buf[0].item()
    dt = self.inner_env.cfg.decimation * self.inner_env.cfg.sim.mujoco.timestep
    t = step * dt

    if step == 0:
      if self.episode_count > 0:
        print("\n" + "*" * 80)
        print(
          f"  EPISODE {self.episode_count} RESULT: {'SUCCESS' if self.episode_success else 'FAILED'}"
        )
        print("*" * 80 + "\n")
      self.kf = None
      self.hsv_ref = None
      self.is_grasped = False
      self.grasp_override_active = False
      self.episode_success = False
      self.printed_success_this_episode = False
      self.episode_count += 1

    # Run hybrid YOLO-based estimation mode if enabled
    if self.args.enable_yolo:
      device = self.inner_env.device

      # Get camera rgb/depth
      from mjlab.tasks.manipulation.mdp import camera_depth, camera_rgb

      with torch.no_grad():
        rgb = camera_rgb(self.inner_env, "head_realsense_camera")
        depth = camera_depth(
          self.inner_env, "head_realsense_camera", cutoff_distance=1.5
        )

        cam_idx = self.inner_env.sim.mj_model.camera("robot/head_realsense_camera").id
        cam_pos = self.inner_env.sim.data.cam_xpos[:, cam_idx]
        cam_xmat = self.inner_env.sim.data.cam_xmat[:, cam_idx]
        cam_fovy = self.inner_env.sim.mj_model.cam_fovy[cam_idx].item()

      H, W = 240, 320
      fovy_rad = math.radians(cam_fovy)
      fy = (H / 2.0) / math.tan(fovy_rad / 2.0)
      fx = fy
      cx = W / 2.0
      cy = H / 2.0

      # Convert tensors to CPU / numpy
      rgb_cpu = (
        (rgb.cpu().permute(0, 2, 3, 1) * 255.0).clip(0, 255).to(torch.uint8).numpy()[0]
      )
      depth_cpu = depth.squeeze(1).cpu().numpy()[0] * 1.5
      cam_pos_i = cam_pos.cpu().numpy()[0]
      cam_xmat_i = cam_xmat.cpu().numpy()[0]

      # Ground truth fallback (pre-computed on CPU)
      gt_pos_w_i = self.command.object_pos_w.cpu().numpy()[0]
      cube_quat = self.box.data.root_link_quat_w
      _, _, gt_yaw_w_all = euler_xyz_from_quat(cube_quat)
      gt_yaw_w_i = gt_yaw_w_all.cpu().numpy()[0]

      # Compute end-effector pose in robot base frame
      ee_pos_w = self.robot.data.site_pos_w[:, self.grasp_idx]
      ee_pos_robot = quat_apply(
        quat_inv(self.robot.data.root_link_quat_w),
        ee_pos_w - self.robot.data.root_link_pos_w,
      )
      ee_xmat = self.inner_env.sim.data.site_xmat[:, self.grasp_idx]
      _, _, robot_yaw_w = euler_xyz_from_quat(self.robot.data.root_link_quat_w)
      ee_yaw_w = torch.atan2(ee_xmat[:, 1, 0], ee_xmat[:, 0, 0])
      ee_yaw = ee_yaw_w - robot_yaw_w

      ee_pos_robot_cpu = ee_pos_robot.cpu().numpy()[0]
      ee_yaw_cpu = ee_yaw.cpu().numpy()[0]

      # Track grasped status based on the flag given in input to the policy
      cursor = 0
      contact_idx = -1
      for name, shape in zip(self.names, self.shapes, strict=False):
        dim = math.prod(shape)
        if name == "object_both__contact_fingers":
          contact_idx = cursor
          break
        cursor += dim

      if contact_idx != -1:
        contact_both_val = obs["actor"][0, contact_idx].item() > 0.5
      else:
        contact_both_val = False

      if not contact_both_val:
        self.grasp_override_active = False
        self.is_grasped = False
      else:
        if not self.grasp_override_active:
          self.is_grasped = True

      ee_pose_i = (
        ee_pos_robot_cpu[0],
        ee_pos_robot_cpu[1],
        ee_pos_robot_cpu[2],
        ee_yaw_cpu,
      )

      success_fit = False
      px, py, pz = 0.0, 0.0, 0.0
      theta = 0.0
      length, width, height = (
        self.args.cube_size[0],
        self.args.cube_size[1],
        self.args.cube_size[2],
      )

      # Robot pose for transforms
      robot_pos = self.robot.data.root_link_pos_w[0].cpu().numpy()
      robot_quat = self.robot.data.root_link_quat_w[0].cpu().numpy()
      q_inv = np.array(
        [robot_quat[0], -robot_quat[1], -robot_quat[2], -robot_quat[3]],
        dtype=np.float32,
      )

      # Run YOLO
      yolo_results = self.yolo_model([rgb_cpu], verbose=False)
      best_box = None
      best_conf = 0.0
      max_area = 0.33 * W * H
      for box_det in yolo_results[0].boxes:
        conf = float(box_det.conf[0])
        bx1, by1, bx2, by2 = map(int, box_det.xyxy[0])
        area = (bx2 - bx1) * (by2 - by1)
        if area > max_area:
          continue
        if conf > self.args.yolo_conf and conf > best_conf:
          best_box = (bx1, by1, bx2, by2)
          best_conf = conf

      if best_box is not None:
        x1, y1, x2, y2 = best_box
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(W, x2)
        y2 = min(H, y2)

        if (x2 - x1) > 1 and (y2 - y1) > 1:
          depth_crop = depth_cpu[y1:y2, x1:x2]
          valid_mask = (depth_crop > 0.1) & (depth_crop < 1.5) & np.isfinite(depth_crop)

          # 1. Run HSV color segmentation first to find the cube pixels
          rgb_mask = None
          try:
            roi = rgb_cpu[y1:y2, x1:x2]
            if roi.size > 0:
              roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
              rh, rw = roi.shape[:2]

              # Extract central 60% area (yolo_depth_pose_hsv uses 0.20 to 0.80)
              cy1, cy2 = int(rh * 0.20), int(rh * 0.80)
              cx1, cx2 = int(rw * 0.20), int(rw * 0.80)
              center_pixels = roi_hsv[cy1:cy2, cx1:cx2]

              if center_pixels.size > 0:
                hues_rad = center_pixels[:, :, 0].astype(np.float32) * (
                  2.0 * np.pi / 180.0
                )
                sin_mean = np.mean(np.sin(hues_rad))
                cos_mean = np.mean(np.cos(hues_rad))
                cand_h_rad = np.arctan2(sin_mean, cos_mean)
                if cand_h_rad < 0:
                  cand_h_rad += 2.0 * np.pi
                cand_h = (cand_h_rad * (180.0 / np.pi) / 2.0) % 180.0
                cand_s = float(np.median(center_pixels[:, :, 1]))
                cand_v = float(np.median(center_pixels[:, :, 2]))

                if self.hsv_ref is None:
                  self.hsv_ref = (cand_h, cand_s, cand_v)
                else:
                  ref_h, ref_s, ref_v = self.hsv_ref
                  h_dist = abs(cand_h - ref_h)
                  h_dist = min(h_dist, 180.0 - h_dist)

                  hsv_lock_thresh_h = 20.0
                  hsv_lock_thresh_sv = 40.0
                  hsv_lock_blend = 0.15

                  if (
                    h_dist < hsv_lock_thresh_h
                    and abs(cand_s - ref_s) < hsv_lock_thresh_sv
                    and abs(cand_v - ref_v) < hsv_lock_thresh_sv
                  ):
                    blend = hsv_lock_blend
                    signed_h_diff = ((cand_h - ref_h + 90.0) % 180.0) - 90.0
                    new_h = (ref_h + blend * signed_h_diff) % 180.0
                    new_s = ref_s + blend * (cand_s - ref_s)
                    new_v = ref_v + blend * (cand_v - ref_v)
                    self.hsv_ref = (new_h, new_s, new_v)

                dominant_h, dominant_s, dominant_v = self.hsv_ref

                diff_h = np.abs(roi_hsv[:, :, 0].astype(np.int32) - dominant_h)
                diff_h = np.minimum(diff_h, 180 - diff_h)

                diff_s = np.abs(roi_hsv[:, :, 1].astype(np.int32) - dominant_s)
                diff_v = np.abs(roi_hsv[:, :, 2].astype(np.int32) - dominant_v)

                h_mask = diff_h < self.args.hsv_h_thresh
                s_mask = (diff_s < self.args.hsv_s_thresh) & (roi_hsv[:, :, 1] > 20)
                v_mask = (diff_v < self.args.hsv_v_thresh) & (roi_hsv[:, :, 2] > 20)

                rgb_mask = (h_mask & s_mask & v_mask).astype(np.uint8) * 255

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                rgb_mask = cv2.morphologyEx(rgb_mask, cv2.MORPH_CLOSE, kernel)
                rgb_mask = cv2.morphologyEx(rgb_mask, cv2.MORPH_OPEN, kernel)
          except Exception:
            pass

          # 2. Apply depth mapping on the HSV-segmented region
          inlier_mask = None
          if rgb_mask is not None and np.sum(rgb_mask > 0) >= 5:
            rgb_mask_depth = cv2.resize(
              rgb_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST
            )
            cube_depth_mask = valid_mask & (rgb_mask_depth > 0)

            if np.sum(cube_depth_mask) >= 5:
              median_depth = np.median(depth_crop[cube_depth_mask])
              inlier_mask = cube_depth_mask & (np.abs(depth_crop - median_depth) < 0.05)

          # Fallback to pure depth filter if HSV failed entirely
          if inlier_mask is None and np.sum(valid_mask) >= 5:
            median_depth = np.median(depth_crop[valid_mask])
            inlier_mask = valid_mask & (np.abs(depth_crop - median_depth) < 0.05)

          if inlier_mask is not None and np.sum(inlier_mask) >= 5:
            # Backproject
            u_range = np.arange(x1, x2)
            v_range = np.arange(y1, y2)
            uu, vv = np.meshgrid(u_range, v_range)

            valid_depths = depth_crop[inlier_mask]
            valid_u = uu[inlier_mask]
            valid_v = vv[inlier_mask]
            if len(valid_depths) >= 5:
              X = (valid_u - cx) * valid_depths / fx
              Y = (valid_v - cy) * valid_depths / fy
              Z = valid_depths

              points_cam = np.stack([X, Y, Z], axis=-1)
              points_mujoco = points_cam * np.array([1.0, -1.0, -1.0])
              points_world = cam_pos_i + np.dot(points_mujoco, cam_xmat_i.T)
              points_robot = rotate_by_quat_np(points_world - robot_pos, q_inv)

              if rgb_mask is not None:
                try:
                  if len(points_robot) >= 5:
                    coords = points_robot[:, :2].astype(np.float32)
                    rect_3d = cv2.minAreaRect(coords)
                    box_3d = cv2.boxPoints(rect_3d)

                    px = float(rect_3d[0][0])
                    py = float(rect_3d[0][1])
                    pz = float(
                      (points_robot[:, 2].max() + points_robot[:, 2].min()) / 2.0
                    )

                    p0, p1, p2, p3 = box_3d
                    v1 = p1 - p0
                    v2 = p2 - p1

                    angle1 = math.atan2(v1[1], v1[0])
                    angle2 = math.atan2(v2[1], v2[0])

                    def wrap_to_pi_over_2(angle):
                      return (angle + math.pi / 2) % math.pi - math.pi / 2

                    wrapped_angle1 = wrap_to_pi_over_2(angle1)
                    wrapped_angle2 = wrap_to_pi_over_2(angle2)

                    if abs(wrapped_angle1) < abs(wrapped_angle2):
                      theta = wrapped_angle1
                      length = float(np.linalg.norm(v1))
                      width = float(np.linalg.norm(v2))
                    else:
                      theta = wrapped_angle2
                      length = float(np.linalg.norm(v2))
                      width = float(np.linalg.norm(v1))

                    z_vals = points_robot[:, 2]
                    height = max(float(z_vals.max() - z_vals.min()), 0.01)

                    success_fit = True
                except Exception:
                  pass

      # Check EE override if grasped
      if success_fit and self.is_grasped:
        ee_x, ee_y, ee_z, ee_yaw = ee_pose_i
        px, py, pz = ee_x, ee_y, ee_z
        theta = ee_yaw
        if self.kf is not None:
          self.kf["yaw_locked"] = False
          for kf, val in [
            (self.kf["kf_x"], px),
            (self.kf["kf_y"], py),
            (self.kf["kf_z"], pz),
          ]:
            if hasattr(kf, "x"):
              if isinstance(kf.x, np.ndarray):
                kf.x[0] = val
                if len(kf.x) > 1:
                  kf.x[1] = 0.0
              else:
                kf.x = val
          if hasattr(self.kf["kf_yaw"], "x"):
            self.kf["kf_yaw"].x = np.array(
              [math.cos(theta), math.sin(theta)], dtype=np.float32
            )

      if success_fit:
        if self.kf is None:
          if self.args.pos_filter_type == "constant_velocity":
            kf_x = ConstantVelocityKF(
              init_pos=px, init_vel=0.0, q_accel=0.0005, r_pos=1e-4
            )
            kf_y = ConstantVelocityKF(
              init_pos=py, init_vel=0.0, q_accel=0.0005, r_pos=1e-4
            )
            kf_z = ConstantVelocityKF(
              init_pos=pz, init_vel=0.0, q_accel=0.0005, r_pos=1e-4
            )
          elif self.args.pos_filter_type == "ema":
            kf_x = ExponentialMovingAverage(init_pos=px, alpha=self.args.ema_alpha)
            kf_y = ExponentialMovingAverage(init_pos=py, alpha=self.args.ema_alpha)
            kf_z = ExponentialMovingAverage(init_pos=pz, alpha=0.10)
          else:
            kf_x = AdaptiveStaticKF(init_pos=px, q_base=1e-6, r_pos=1e-3, window=10)
            kf_y = AdaptiveStaticKF(init_pos=py, q_base=1e-6, r_pos=1e-3, window=10)
            kf_z = AdaptiveStaticKF(init_pos=pz, q_base=1e-6, r_pos=1e-3, window=10)

          if self.args.pos_filter_type == "ema":
            kf_yaw = ExponentialMovingAverageYaw(init_yaw=theta, alpha=0.20)
            kf_len = ExponentialMovingAverage1D(init_val=length, alpha=0.10)
            kf_wid = ExponentialMovingAverage1D(init_val=width, alpha=0.10)
            kf_hgt = ExponentialMovingAverage1D(init_val=height, alpha=0.10)
          else:
            kf_yaw = YawKF(init_yaw=theta, q_yaw=0.0001, r_yaw=0.02)
            kf_len = StaticState1DKF(init_val=length, q_val=1e-5, r_val=0.005)
            kf_wid = StaticState1DKF(init_val=width, q_val=1e-5, r_val=0.005)
            kf_hgt = StaticState1DKF(init_val=height, q_val=1e-5, r_val=0.005)

          self.kf = {
            "kf_x": kf_x,
            "kf_y": kf_y,
            "kf_z": kf_z,
            "kf_yaw": kf_yaw,
            "kf_len": kf_len,
            "kf_wid": kf_wid,
            "kf_hgt": kf_hgt,
            "occluded_while_grasping": False,
            "yaw_locked": False,
            "yaw_lock_x": None,
            "yaw_lock_y": None,
          }
        else:
          self.kf["kf_x"].predict(dt)
          self.kf["kf_y"].predict(dt)
          self.kf["kf_z"].predict(dt)
          self.kf["kf_yaw"].predict(dt)
          self.kf["kf_len"].predict(dt)
          self.kf["kf_wid"].predict(dt)
          self.kf["kf_hgt"].predict(dt)

          if self.args.pos_filter_type == "ema":
            # Set alpha dynamically based on whether we are grasping
            if self.is_grasped:
              self.kf["kf_x"].alpha = 0.001
              self.kf["kf_y"].alpha = 0.001
              self.kf["kf_z"].alpha = 0.001
              self.kf["kf_yaw"].alpha = 0.0
            else:
              self.kf["kf_x"].alpha = self.args.ema_alpha
              self.kf["kf_y"].alpha = self.args.ema_alpha
              self.kf["kf_z"].alpha = 0.10
              self.kf["kf_yaw"].alpha = 0.20

          # --- Yaw Lock Logic ---
          last_yaw = self.kf["kf_yaw"].get_yaw()
          diff_angle = theta - last_yaw
          # Cube rotational symmetry of 90 degrees (pi/2), wrap diff to [-pi/4, pi/4]
          wrapped_diff = (diff_angle + math.pi / 4) % (math.pi / 2) - math.pi / 4

          # Check displacement to potentially unlock
          if self.kf["yaw_locked"]:
            curr_filtered_x = self.kf["kf_x"].x[0]
            curr_filtered_y = self.kf["kf_y"].x[0]
            dist_moved = math.sqrt(
              (curr_filtered_x - self.kf["yaw_lock_x"]) ** 2
              + (curr_filtered_y - self.kf["yaw_lock_y"]) ** 2
            )
            if dist_moved > 0.02:  # 2 cm
              self.kf["yaw_locked"] = False

          # Check angle update to potentially lock (only lock if not grasped/override)
          if not self.kf["yaw_locked"] and not self.is_grasped:
            if abs(wrapped_diff) > math.radians(5.0):  # 5 degrees
              self.kf["yaw_locked"] = True
              self.kf["yaw_lock_x"] = self.kf["kf_x"].x[0]
              self.kf["yaw_lock_y"] = self.kf["kf_y"].x[0]

          u_x = self.kf["kf_x"].update(px)
          u_y = self.kf["kf_y"].update(py)
          z_r_pos = 0.15 if self.kf["occluded_while_grasping"] else None
          u_z = self.kf["kf_z"].update(pz, r_pos=z_r_pos)
          self.kf["occluded_while_grasping"] = False

          if u_x and u_y and u_z:
            if not self.kf["yaw_locked"]:
              self.kf["kf_yaw"].update(theta)
            self.kf["kf_len"].update(length)
            self.kf["kf_wid"].update(width)
            self.kf["kf_hgt"].update(height)

          px = float(self.kf["kf_x"].x[0])
          py = float(self.kf["kf_y"].x[0])
          pz = float(self.kf["kf_z"].x[0])
          theta = float(self.kf["kf_yaw"].get_yaw())
          length = float(self.kf["kf_len"].x)
          width = float(self.kf["kf_wid"].x)
          height = float(self.kf["kf_hgt"].x)
      else:
        if self.kf is not None:
          z_q_scale = 500.0 if contact_both_val else 10.0
          self.kf["kf_x"].predict(dt, q_scale=10.0)
          self.kf["kf_y"].predict(dt, q_scale=10.0)
          self.kf["kf_z"].predict(dt, q_scale=z_q_scale)
          self.kf["kf_yaw"].predict(dt, q_scale=10.0)
          self.kf["kf_len"].predict(dt, q_scale=10.0)
          self.kf["kf_wid"].predict(dt, q_scale=10.0)
          self.kf["kf_hgt"].predict(dt, q_scale=10.0)

          if contact_both_val:
            self.kf["occluded_while_grasping"] = True

          px = float(self.kf["kf_x"].x[0])
          py = float(self.kf["kf_y"].x[0])
          pz = float(self.kf["kf_z"].x[0])
          theta = float(self.kf["kf_yaw"].get_yaw())
          length = float(self.kf["kf_len"].x)
          width = float(self.kf["kf_wid"].x)
          height = float(self.kf["kf_hgt"].x)
        else:
          gt_pos_r_i = rotate_by_quat_np(gt_pos_w_i - robot_pos, q_inv)
          px, py, pz = gt_pos_r_i[0], gt_pos_r_i[1], gt_pos_r_i[2]
          theta = wrap_angle(gt_yaw_w_i - robot_yaw_w.cpu().numpy()[0])
          length, width, height = (
            self.args.cube_size[0],
            self.args.cube_size[1],
            self.args.cube_size[2],
          )

      # Fallback: if depth pipeline failed but grasped is True, override to EE pose
      if not success_fit and self.is_grasped:
        px, py, pz = ee_pose_i[0], ee_pose_i[1], ee_pose_i[2]
        theta = ee_pose_i[3]
        if self.kf is not None:
          length = float(self.kf["kf_len"].x)
          width = float(self.kf["kf_wid"].x)
          height = float(self.kf["kf_hgt"].x)
        else:
          length, width, height = (
            self.args.cube_size[0],
            self.args.cube_size[1],
            self.args.cube_size[2],
          )

        if self.kf is not None:
          self.kf["yaw_locked"] = False
          for kf, val in [
            (self.kf["kf_x"], px),
            (self.kf["kf_y"], py),
            (self.kf["kf_z"], pz),
          ]:
            if hasattr(kf, "x"):
              if isinstance(kf.x, np.ndarray):
                kf.x[0] = val
                if len(kf.x) > 1:
                  kf.x[1] = 0.0
              else:
                kf.x = val
          if hasattr(self.kf["kf_yaw"], "x"):
            self.kf["kf_yaw"].x = np.array(
              [math.cos(theta), math.sin(theta)], dtype=np.float32
            )

      # Overwrite observations in obs (tensor shape: [1, 35])
      est_pos_r = torch.tensor([px, py, pz], device=device, dtype=torch.float32)
      est_yaw_r = torch.tensor(
        [math.cos(theta), math.sin(theta)], device=device, dtype=torch.float32
      )

      obs["actor"][0, 22:25] = est_pos_r
      obs["actor"][0, 25:27] = est_yaw_r
      if "critic" in obs:
        obs["critic"][0, 22:25] = est_pos_r
        obs["critic"][0, 25:27] = est_yaw_r

      # Print predicted vs ground truth pose for visual feedback
      gt_pos_r = rotate_by_quat_np(gt_pos_w_i - robot_pos, q_inv)
      gt_yaw_r = wrap_angle(gt_yaw_w_i - robot_yaw_w.cpu().numpy()[0])
      print("Estimation Info:")
      print(
        f"  YOLO Fit: {'SUCCESS' if success_fit else 'FAIL'} | Grasped: {self.is_grasped}"
      )
      print(
        f"  Pos Pred: [{px:.4f}, {py:.4f}, {pz:.4f}] | GT: [{gt_pos_r[0]:.4f}, {gt_pos_r[1]:.4f}, {gt_pos_r[2]:.4f}]"
      )
      print(
        f"  Yaw Pred: {theta:.4f} rad ({math.degrees(theta):.2f}°) | GT: {gt_yaw_r:.4f} rad ({math.degrees(gt_yaw_r):.2f}°)"
      )

    # Get actual box sizes (length, width, height), orientation (yaw), and mass
    box_entity = self.inner_env.scene["box"]
    geom_id = box_entity.indexing.geom_ids[0]
    box_half_sizes = self.inner_env.sim.model.geom_size[0, geom_id]
    box_full_sizes = box_half_sizes * 2.0

    # Get body id and mass (mass is stored per-body in MuJoCo, not per-geom)
    body_id = box_entity.indexing.body_ids[0]
    box_mass = self.inner_env.sim.model.body_mass[0, body_id]

    box_quat = box_entity.data.root_link_quat_w
    _, _, box_yaw = euler_xyz_from_quat(box_quat)
    box_yaw_val = box_yaw[0].item()
    box_yaw_deg = math.degrees(box_yaw_val)

    # Get ground truth position and yaw in both world and robot root frames
    box_pos_w = box_entity.data.root_link_pos_w[0].cpu().numpy()

    robot_entity = self.inner_env.scene["robot"]
    robot_pos = robot_entity.data.root_link_pos_w[0].cpu().numpy()
    robot_quat = robot_entity.data.root_link_quat_w[0].cpu().numpy()
    q_inv = np.array(
      [robot_quat[0], -robot_quat[1], -robot_quat[2], -robot_quat[3]], dtype=np.float32
    )
    box_pos_r = rotate_by_quat_np(box_pos_w - robot_pos, q_inv)

    _, _, robot_yaw_w = euler_xyz_from_quat(robot_entity.data.root_link_quat_w)
    box_yaw_r = wrap_angle(box_yaw_val - robot_yaw_w[0].item())
    box_yaw_r_deg = math.degrees(box_yaw_r)

    # Calculate squeeze axis angle for printing
    squeeze_axis_angle_deg = None
    fingertip_site_names = [s for s in robot_entity.site_names if "fingertip" in s]
    if len(fingertip_site_names) == 2:
      left_idx = robot_entity.site_names.index(fingertip_site_names[0])
      right_idx = robot_entity.site_names.index(fingertip_site_names[1])
      p_left = robot_entity.data.site_pos_w[0, left_idx]
      p_right = robot_entity.data.site_pos_w[0, right_idx]
      v_squeeze_w = p_left - p_right
      v_squeeze_t = v_squeeze_w.unsqueeze(0)
      box_quat_t = box_entity.data.root_link_quat_w[0].unsqueeze(0)

      v_squeeze_local = quat_apply(quat_inv(box_quat_t), v_squeeze_t)
      v_local_2d = v_squeeze_local.clone()
      v_local_2d[:, 2] = 0.0
      norm_2d = torch.norm(v_local_2d, dim=-1, keepdim=True)
      if norm_2d.item() > 1e-4:
        v_local_2d_norm = v_local_2d / norm_2d
        cos_theta = torch.max(
          torch.abs(v_local_2d_norm[:, 0]), torch.abs(v_local_2d_norm[:, 1])
        )
        cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
        squeeze_axis_angle_deg = torch.rad2deg(torch.acos(cos_theta)).item()

    print("\n" + "=" * 80)
    print(f"Step: {step:3d} | Time: {t:.2f}s")
    print(
      f"Object Length (X): {box_full_sizes[0].item():.4f} m | Width (Y): {box_full_sizes[1].item():.4f} m | Height (Z): {box_full_sizes[2].item():.4f} m"
    )
    print(
      f"Object World Pos:  [{box_pos_w[0]:.4f}, {box_pos_w[1]:.4f}, {box_pos_w[2]:.4f}] m"
    )
    print(f"Object World Yaw:  {box_yaw_val:.4f} rad ({box_yaw_deg:.2f}°)")
    print(
      f"Object Robot Pos:  [{box_pos_r[0]:.4f}, {box_pos_r[1]:.4f}, {box_pos_r[2]:.4f}] m"
    )
    print(f"Object Robot Yaw:  {box_yaw_r:.4f} rad ({box_yaw_r_deg:.2f}°)")
    if squeeze_axis_angle_deg is not None:
      print(f"Squeeze Axis Angle: {squeeze_axis_angle_deg:.2f}°")
    print(f"Object Mass: {box_mass.item():.4f} kg")

    target_pos_w = self.command.target_pos[0].cpu().numpy()
    reached = self.command.reached[0].item()
    at_goal_time = self.command.at_goal_time[0].item()
    grasped_dist = self.command.grasped_distance[0].item()
    joint_ids, _ = self.robot.find_joints("gripper_right_finger_joint")
    gripper_pos = self.robot.data.joint_pos[0, joint_ids[0]].item()
    gripper_vel = self.robot.data.joint_vel[0, joint_ids[0]].item()
    print(
      f"Goal Position:     [{target_pos_w[0]:.4f}, {target_pos_w[1]:.4f}, {target_pos_w[2]:.4f}] m"
    )
    print(f"Goal Reached:      {reached} (Time at Goal: {at_goal_time:.2f}s)")
    print(f"Grasped Distance:  {grasped_dist:.4f} m")
    print(f"Gripper Position:  {gripper_pos:.4f} m | Velocity: {gripper_vel:.4f} m/s")

    # Read fingertip contact sensors (one per gripper finger) and compute contact metrics
    try:
      contact_sensor = self.inner_env.scene["box_fingertip_contact"]
    except (KeyError, AttributeError, TypeError):
      contact_sensor = None

    dist_both = False
    combined_contact = False
    try:
      robot_entity = self.inner_env.scene["robot"]

      from pal_mjlab.robots.pal_tiago_pro.tiago_pro import TiagoProRobot

      robot_cfg = TiagoProRobot()
      site_ids, _ = robot_entity.find_sites(
        [robot_cfg.fingertip_site_pattern], preserve_order=True
      )
      site_pos_w = robot_entity.data.site_pos_w[:, site_ids]
      obj_pos_w = box_entity.data.geom_pos_w[:, 0].unsqueeze(1)
      dist_to_obj = torch.norm(site_pos_w - obj_pos_w, dim=-1)
      dist_both = (dist_to_obj < 0.05).all(dim=-1)[0].item()

      from pal_mjlab.tasks.manipulation.mdp.observations import (
        object_both__contact_fingers,
      )

      combined_contact_tensor = object_both__contact_fingers(
        env=self.inner_env,
        sensor_name="box_fingertip_contact",
        site_names=[robot_cfg.fingertip_site_pattern],
      )
      combined_contact = combined_contact_tensor[0, 0].item() > 0

      # Check if the episode is successful (reached goal, fell on floor, and released)
      on_floor = box_pos_w[2] < 0.1
      success_now = reached and on_floor and not dist_both
      if success_now:
        self.episode_success = True
        if not getattr(self, "printed_success_this_episode", False):
          print("\n" + "*" * 80)
          print(
            f"*** SUCCESS ACHIEVED AT STEP {step}! (Reached & Released on floor) ***"
          )
          print("*" * 80 + "\n")
          self.printed_success_this_episode = True
    except Exception:
      pass

    if contact_sensor is not None and contact_sensor.data.found is not None:
      found = contact_sensor.data.found[0]  # shape: [N] (N=2 for two fingertips)
      finger_contacts = [f.item() > 0 for f in found]
      both_contacts_phys = all(finger_contacts)
      print(
        f"Finger Contacts (Physical): {finger_contacts} | Both Physical: {both_contacts_phys}"
      )
      print(f"Distances to object: {dist_to_obj[0].tolist()}")
      print(
        f"Both Distance-based: {dist_both} | Both Combined (New Measure): {combined_contact}"
      )
    else:
      print("Finger Contacts: Sensor not available")
    # Read reward terms
    reward_manager = self.inner_env.reward_manager
    if (
      step > 0
      and hasattr(reward_manager, "_step_reward")
      and reward_manager._step_reward is not None
    ):
      step_rewards = reward_manager._step_reward[0]  # shape: [num_terms]
      total_step_reward = step_rewards.sum().item()
      total_scaled_reward = reward_manager._reward_buf[0].item()
      print("Reward values (weighted contribution of each term to the transition):")
      for term_name, val in zip(
        reward_manager.active_terms, step_rewards.tolist(), strict=False
      ):
        print(f"  {term_name:35s}: {val:10.4f}")
      print(f"  {'Total Step Reward (unscaled)':35s}: {total_step_reward:10.4f}")
      print(f"  {'Total Step Reward (scaled by dt)':35s}: {total_scaled_reward:10.4f}")
    else:
      print("Reward values: No transition yet (initial state)")

    print("-" * 80)

    # Extract actor observations
    if hasattr(obs, "keys") and "actor" in obs:
      actor_obs = obs["actor"]
      if actor_obs.ndim > 1:
        actor_obs = actor_obs[0]
    elif torch.is_tensor(obs):
      actor_obs = obs
      if actor_obs.ndim > 1:
        actor_obs = actor_obs[0]
    else:
      actor_obs = obs

    cursor = 0
    for name, shape in zip(self.names, self.shapes, strict=False):
      dim = math.prod(shape)
      vals = actor_obs[cursor : cursor + dim].tolist()
      formatted_vals = ", ".join([f"{v:.4f}" for v in vals])
      if name == "object_yaw" and len(vals) == 2:
        obs_yaw_rad = math.atan2(vals[1], vals[0])
        obs_yaw_deg = math.degrees(obs_yaw_rad)
        print(
          f"  {name:25s} shape={str(shape):8s} value=[{formatted_vals}] (yaw: {obs_yaw_rad:.4f} rad, {obs_yaw_deg:.2f}°)"
        )
      else:
        print(f"  {name:25s} shape={str(shape):8s} value=[{formatted_vals}]")
      cursor += dim

    if self.model is not None:
      import onnxruntime
      from tensordict import TensorDict

      if isinstance(self.model, onnxruntime.InferenceSession):
        obs_tensor = (
          obs["actor"] if isinstance(obs, dict) or isinstance(obs, TensorDict) else obs
        )
        obs_numpy = obs_tensor.cpu().numpy()
        ort_inputs = {self.model.get_inputs()[0].name: obs_numpy}
        ort_outs = self.model.run(None, ort_inputs)
        action = torch.tensor(ort_outs[0], device=self.inner_env.device)
        return action
      else:
        if not isinstance(obs, TensorDict):
          obs = TensorDict(obs, batch_size=[1])
        with torch.no_grad():
          action = self.model(obs)
        return action
    else:
      return torch.zeros(self.action_shape, device=self.inner_env.device)


def main():
  parser = argparse.ArgumentParser(
    description="Play environment with checkpoint model and print observations."
  )
  parser.add_argument(
    "--viewer",
    type=str,
    choices=["auto", "native", "viser"],
    default="auto",
    help="Viewer backend to use.",
  )
  parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to the checkpoint model weights (e.g. .pt file). If None, runs the zero policy.",
  )
  parser.add_argument(
    "--scan-mass",
    action="store_true",
    help="Instead of launching the viewer, reset the env once per episode, take a "
    "single (zero-action) step, and record the min/max box mass seen over "
    "--num-episodes episodes. Exits after printing the summary.",
  )
  parser.add_argument(
    "--num-episodes",
    type=int,
    default=500,
    help="Number of episodes to run when --scan-mass is set (default: 500).",
  )
  parser.add_argument(
    "--enable_yolo",
    action="store_true",
    default=False,
    help="Enable hybrid YOLO-based estimation mode instead of ground truth (default: False).",
  )
  parser.add_argument(
    "--yolo_model",
    type=str,
    default="/home/lorenzobarbieri/exchange/tiago_pro_sim_ws/runs/detect/tiago_single_class_yolo26/weights/best.pt",
    help="Path to YOLO model checkpoint (default: best.pt).",
  )
  parser.add_argument(
    "--yolo_conf",
    type=float,
    default=0.45,
    help="YOLO detection confidence threshold (default: 0.45).",
  )
  parser.add_argument(
    "--pos_filter_type",
    type=str,
    default="ema",
    choices=["constant_velocity", "adaptive", "ema"],
    help="Type of 3D position filter to use: 'constant_velocity', 'adaptive' (Kalman), or 'ema' (Exponential Moving Average) (default: ema).",
  )
  parser.add_argument(
    "--ema_alpha",
    type=float,
    default=0.2,
    help="EMA smoothing factor alpha in [0, 1] (only used when --pos_filter_type=ema, default: 0.2). Higher = less smoothing.",
  )
  parser.add_argument(
    "--hsv_h_thresh",
    type=int,
    default=100,
    help="Hue threshold for HSV segmentation (default: 100).",
  )
  parser.add_argument(
    "--hsv_s_thresh",
    type=int,
    default=100,
    help="Saturation threshold for HSV segmentation (default: 100).",
  )
  parser.add_argument(
    "--hsv_v_thresh",
    type=int,
    default=80,
    help="Value threshold for HSV segmentation (default: 80).",
  )
  parser.add_argument(
    "--cube_size",
    nargs=3,
    type=float,
    default=[0.035, 0.035, 0.05],
    help="Nominal cube size in meters (length, width, height) (default: [0.035, 0.035, 0.05]).",
  )
  args = parser.parse_args()

  configure_torch_backends()
  device = "cuda:0" if torch.cuda.is_available() else "cpu"

  # Load configurations
  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.scene.num_envs = 1
  agent_cfg = load_rl_cfg(TASK_ID)

  # Dynamically add the camera sensor if YOLO is enabled
  if args.enable_yolo:
    print("Dynamically adding head_realsense_camera sensor to scene config...")
    env_cfg.scene.sensors = (env_cfg.scene.sensors or ()) + (
      CameraSensorCfg(
        name="head_realsense_camera",
        height=240,
        width=320,
        data_types=("rgb", "depth"),
        camera_name="robot/head_realsense_camera",
      ),
    )

  # Initialize environment
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)

  def update_visualizers(visualizer):
    robot = env.scene["robot"]
    box = env.scene["box"]

    fingertip_site_names = [s for s in robot.site_names if "fingertip" in s]
    if len(fingertip_site_names) == 2:
      left_idx = robot.site_names.index(fingertip_site_names[0])
      right_idx = robot.site_names.index(fingertip_site_names[1])

      p_left = robot.data.site_pos_w[0, left_idx].cpu().numpy()
      p_right = robot.data.site_pos_w[0, right_idx].cpu().numpy()

      v_squeeze = p_left - p_right
      v_squeeze_len = np.linalg.norm(v_squeeze)
      if v_squeeze_len > 1e-4:
        # Squeeze axis arrow from right finger to left finger
        # Color: Orange (1.0, 0.5, 0.0, 0.8)
        visualizer.add_arrow(
          start=p_right, end=p_left, color=(1.0, 0.5, 0.0, 0.8), width=0.008
        )

        # Target face normal of the box
        box_pos = box.data.root_link_pos_w[0].cpu().numpy()
        # box_quat = box.data.root_link_quat_w[0].cpu().numpy()

        v_squeeze_t = torch.tensor(
          v_squeeze, dtype=torch.float32, device=device
        ).unsqueeze(0)
        box_quat_t = box.data.root_link_quat_w[0].unsqueeze(0)

        v_squeeze_local_t = quat_apply(quat_inv(box_quat_t), v_squeeze_t)
        v_local_2d_t = v_squeeze_local_t.clone()
        v_local_2d_t[:, 2] = 0.0
        norm_2d = torch.norm(v_local_2d_t, dim=-1, keepdim=True)
        if norm_2d.item() > 1e-4:
          v_local_2d_norm_t = v_local_2d_t / norm_2d
          v_local_2d_norm = v_local_2d_norm_t[0].cpu().numpy()

          if abs(v_local_2d_norm[0]) > abs(v_local_2d_norm[1]):
            closest_local_normal = np.array(
              [np.sign(v_local_2d_norm[0]), 0.0, 0.0], dtype=np.float32
            )
          else:
            closest_local_normal = np.array(
              [0.0, np.sign(v_local_2d_norm[1]), 0.0], dtype=np.float32
            )

          closest_local_normal_t = torch.tensor(
            closest_local_normal, dtype=torch.float32, device=device
          ).unsqueeze(0)
          closest_normal_w_t = quat_apply(box_quat_t, closest_local_normal_t)
          closest_normal_w = closest_normal_w_t[0].cpu().numpy()

          # Target face normal starting from box_pos
          arrow_len = 0.15
          normal_start = box_pos
          normal_end = box_pos + closest_normal_w * arrow_len

          # Color: Cyan (0.0, 0.8, 0.8, 0.8)
          visualizer.add_arrow(
            start=normal_start, end=normal_end, color=(0.0, 0.8, 0.8, 0.8), width=0.008
          )

  env.update_visualizers = update_visualizers
  env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  # If requested, run a headless mass scan instead of the interactive viewer.
  if args.scan_mass:
    scan_box_mass(env_wrapped, args.num_episodes, device)
    env_wrapped.close()
    return

  # Load checkpoint model if provided
  model = None
  if args.checkpoint is not None:
    checkpoint_path = args.checkpoint.strip()
    if not os.path.exists(checkpoint_path):
      raise FileNotFoundError(f"Checkpoint path '{checkpoint_path}' does not exist!")

    if checkpoint_path.endswith(".onnx"):
      import onnxruntime

      print(f"Loading ONNX model from {checkpoint_path}...")
      model = onnxruntime.InferenceSession(checkpoint_path)
      print("ONNX model loaded successfully!")
    else:
      from tensordict import TensorDict

      print("Setting up policy model...")
      actor_cfg = agent_cfg.actor
      model_cls = load_class(actor_cfg.class_name)

      # Initialize with dummy observations to build model
      obs_dict, _ = env.reset()
      dummy_obs = TensorDict(obs_dict, batch_size=[1])

      model = model_cls(
        obs=dummy_obs,
        obs_groups=getattr(agent_cfg, "obs_groups", None),
        obs_set="actor",
        output_dim=env.action_manager.total_action_dim,
        hidden_dims=actor_cfg.hidden_dims,
        activation=actor_cfg.activation,
        obs_normalization=actor_cfg.obs_normalization,
        distribution_cfg=actor_cfg.distribution_cfg,
      ).to(device)

      print(f"Loading model weights from {checkpoint_path}...")
      checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
      model.load_state_dict(checkpoint["actor_state_dict"], strict=True)
      model.eval()
      print("Model loaded successfully!")

  action_shape = env_wrapped.unwrapped.action_space.shape
  policy = PrintingPolicy(action_shape, env_wrapped, args, model=model)

  # Handle viewer selection
  if args.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
  else:
    resolved_viewer = args.viewer

  print(f"Starting simulation with viewer: {resolved_viewer}...")
  if resolved_viewer == "native":
    NativeMujocoViewer(env_wrapped, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env_wrapped, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env_wrapped.close()


if __name__ == "__main__":
  main()
