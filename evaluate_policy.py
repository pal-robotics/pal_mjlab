#!/usr/bin/env python3
"""
Script to evaluate a trained policy on 100 episodes and record key metrics:
- success rate
- number of episodes with top surface collisions
- mean fingertip angles wrt cube lateral surfaces the instant before contact_both_fingers

Supports both ground truth state feedback and a hybrid YOLO-based estimation mode.
"""

import argparse
import os
import sys
import math
import cv2
import torch
import numpy as np
from tensordict import TensorDict

# Import mjlab modules
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.sensor import CameraSensorCfg

# Import task-specific modules
from pal_mjlab.tasks.manipulation.mdp.contact_sensor import site_contact_both_fingers
from pal_mjlab.tasks.manipulation.mdp.rewards import top_surface_penetration_term
from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, euler_xyz_from_quat
from mjlab.tasks.manipulation.mdp import camera_rgb, camera_depth

# Sourcing the filters from the workspace
sys.path.append("/home/lorenzobarbieri/exchange/tiago_pro_sim_ws/src")
try:
    from filters import ConstantVelocityKF, AdaptiveStaticKF, YawKF, StaticState1DKF, ExponentialMovingAverage, ExponentialMovingAverageYaw, ExponentialMovingAverage1D
except ImportError:
    print("Warning: could not import filters from exchange workspace. Implementing fallback filters.")
    # Minimal fallback implementations for safety
    class ConstantVelocityKF:
        def __init__(self, init_pos, init_vel=0.0, q_accel=0.05, r_pos=1e-5):
            self.x = np.array([init_pos, init_vel], dtype=np.float32)
            self.P = np.array([[1.0, 0.0], [0.0, 10.0]], dtype=np.float32)
            self.q_accel = q_accel
            self.r_pos = r_pos
        def predict(self, dt, q_scale=1.0):
            F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float32)
            q = self.q_accel * q_scale
            Q = q * np.array([[(dt**3)/3.0, (dt**2)/2.0], [(dt**2)/2.0, dt]], dtype=np.float32)
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
            self.x = self.x + K.flatten() * innovation
            self.P = (np.eye(2, dtype=np.float32) - np.outer(K.flatten(), H.flatten())) @ self.P
            return True

    class AdaptiveStaticKF:
        def __init__(self, init_pos, q_base=1e-7, r_pos=1e-5, window=10):
            self.x = np.array([init_pos, 0.0], dtype=np.float32)
            self.P = np.array([[1.0, 0.0], [0.0, 10.0]], dtype=np.float32)
            self.q_base = q_base
            self.r_pos = r_pos
            self.innovations = []
            self.window = window
            self.current_alpha = 1.0
        def predict(self, dt, q_scale=1.0):
            F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float32)
            q = self.q_base * self.current_alpha * q_scale
            Q = q * np.array([[(dt**3)/3.0, (dt**2)/2.0], [(dt**2)/2.0, dt]], dtype=np.float32)
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
            self.innovations.append(innovation**2)
            if len(self.innovations) > self.window:
                self.innovations.pop(0)
            S_theoretical = (H @ self.P @ H.T + R)[0, 0]
            S_empirical   = np.mean(self.innovations)
            self.current_alpha = max(1.0, S_empirical / S_theoretical)
            S = H @ self.P @ H.T + R
            K = self.P @ H.T / S[0, 0]
            self.x = self.x + K.flatten() * innovation
            self.P = (np.eye(2, dtype=np.float32) - np.outer(K.flatten(), H.flatten())) @ self.P
            return True

    class YawKF:
        def __init__(self, init_yaw, q_yaw=0.001, r_yaw=0.02):
            self.x = np.array([math.cos(init_yaw), math.sin(init_yaw)], dtype=np.float32)
            self.P = np.eye(2, dtype=np.float32) * 1.0
            self.q_yaw = q_yaw
            self.r_yaw = r_yaw
        def predict(self, dt, q_scale=1.0):
            Q = np.eye(2, dtype=np.float32) * (self.q_yaw * q_scale * dt)
            self.P = self.P + Q
        def update(self, z_yaw, r_yaw=None):
            if r_yaw is None:
                r_yaw = self.r_yaw
            z = np.array([math.cos(z_yaw), math.sin(z_yaw)], dtype=np.float32)
            y = z - self.x
            S = self.P + np.eye(2, dtype=np.float32) * r_yaw
            K = self.P @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = (np.eye(2, dtype=np.float32) - K) @ self.P
            norm = np.linalg.norm(self.x)
            if norm > 1e-5:
                self.x = self.x / norm
        def get_yaw(self):
            return float(math.atan2(self.x[1], self.x[0]))

    class StaticState1DKF:
        def __init__(self, init_val, q_val=0.0001, r_val=0.005):
            self.x = float(init_val)
            self.P = 1.0
            self.q_val = q_val
            self.r_val = r_val
        def predict(self, dt, q_scale=1.0):
            self.P = self.P + (self.q_val * q_scale * dt)
        def update(self, z_val, r_val=None):
            if r_val is None:
                r_val = self.r_val
            y = z_val - self.x
            S = self.P + r_val
            K = self.P / S
            self.x = self.x + K * y
            self.P = (1.0 - K) * self.P

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
def rotate_by_quat_np(v, q):
    """
    Rotate vectors v (shape [N, 3] or [3]) by quaternion q (shape [4] as [w, x, y, z]) using Rodrigues' formula.
    """
    v_arr = np.array(v, dtype=np.float32)
    is_single = (v_arr.ndim == 1)
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


def compute_angle_wrt_cube_lateral_surfaces(v: torch.Tensor, cube_quat: torch.Tensor) -> torch.Tensor:
    """
    Computes the angle (in degrees) between a vector v and the closest lateral surface of the cube.
    
    Args:
        v: Tensor of shape [B, 3] representing fingertip normals or squeeze axis in world frame.
        cube_quat: Tensor of shape [B, 4] representing cube quaternion in world frame.
        
    Returns:
        Tensor of shape [B] containing the angles in degrees.
    """
    # 1. Normalize the vector
    v_norm = v / torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-6)
    
    # 2. Rotate the vector into the cube's local frame
    v_local = quat_apply(quat_inv(cube_quat), v_norm)  # [B, 3]
    
    # 3. The cube's lateral surfaces are perpendicular to the local X and Y axes.
    # The cosine of the angle wrt the closest lateral normal is the max absolute dot product with X or Y.
    cos_theta = torch.max(torch.abs(v_local[:, 0]), torch.abs(v_local[:, 1]))
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    
    # 4. Compute angle in degrees
    angle_rad = torch.acos(cos_theta)
    return torch.rad2deg(angle_rad)


def evaluate(args):
    configure_torch_backends()
    device = args.device
    if device == "cuda" and torch.cuda.is_available():
        device = "cuda:0"
    print(f"Using device: {device}")

    # 1. Load Configurations
    print(f"Loading task configuration: {args.task}...")
    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = args.num_envs
    
    # Remove early termination terms (except time_out and object_held_at_goal)
    for term_name in list(env_cfg.terminations.keys()):
        if term_name not in ["time_out", "object_held_at_goal"]:
            print(f"Removing termination condition: {term_name}")
            del env_cfg.terminations[term_name]
    
    # Re-enable the success termination (object_held_at_goal) which is disabled by play=True
    from mjlab.managers import TerminationTermCfg
    import pal_mjlab.tasks.manipulation.mdp as manipulation_mdp_pal
    print("Re-enabling success termination: object_held_at_goal")
    env_cfg.terminations["object_held_at_goal"] = TerminationTermCfg(
        func=manipulation_mdp_pal.object_held_at_goal_term,
        params={"command_name": "lift_height", "hold_time_s": 1.0},
        time_out=True,
    )
    
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
    
    print("Loading RL agent configuration...")
    rl_cfg = load_rl_cfg(args.task)

    # 2. Initialize Environment
    print("Initializing Mujoco Environment...")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    
    # 3. Instantiate Policy Model
    print("Setting up MLP model...")
    actor_cfg = rl_cfg.actor
    model_cls = load_class(actor_cfg.class_name)
    
    # Initialize with dummy observations to build model
    obs_dict, _ = env.reset()
    dummy_obs = TensorDict(obs_dict, batch_size=[args.num_envs])
    
    model = model_cls(
        obs=dummy_obs,
        obs_groups=getattr(rl_cfg, "obs_groups", None),
        obs_set="actor",
        output_dim=env.action_manager.total_action_dim,
        hidden_dims=actor_cfg.hidden_dims,
        activation=actor_cfg.activation,
        obs_normalization=actor_cfg.obs_normalization,
        distribution_cfg=actor_cfg.distribution_cfg,
    ).to(device)

    # 4. Load Checkpoint Weights
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint path '{checkpoint_path}' does not exist!", file=sys.stderr)
        sys.exit(1)
        
    print(f"Loading model weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    model.eval()

    # Load YOLO model if enabled
    yolo_model = None
    if args.enable_yolo:
        print(f"Loading YOLO model from {args.yolo_model}...")
        from ultralytics import YOLO
        yolo_model = YOLO(args.yolo_model)
        yolo_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        yolo_model.to(yolo_device)
        print(f"YOLO model loaded successfully on device: {yolo_device}")

    # 5. Tracking variables
    num_envs = env.num_envs
    episodes_collected = 0  # total episodes completed across all envs

    # Final episode metrics (appended as episodes complete)
    success_recorded = []
    top_collision_recorded = []
    grasp_and_lift_recorded = []
    min_pos_error_recorded = []  # best position error seen in each episode [m]

    # Per-episode running minimum position error
    current_episode_min_pos_error = torch.full((num_envs,), float("inf"), device=device)

    # Fingertip angle trigger tracking
    contact_both_activated = np.zeros(num_envs, dtype=bool)
    angles_trigger_squeeze = []
    angles_trigger_left = []
    angles_trigger_right = []

    # -----------------------------------------------------------------------
    # Observation noise injection (mirrors training noise from env_cfgs.py)
    # -----------------------------------------------------------------------
    # Build a mapping: obs_term_name -> (slice_start, slice_end, noise_half_range)
    # We use the obs manager to find slice positions dynamically.
    _NOISE_SPECS = {
        "joint_pos":             0.02,   # Unoise ±0.02
        "joint_vel":             0.05,   # Unoise ±0.05
        # "target_object_position": 0.01,  # Unoise ±0.01
        "ee_position":           0.01,   # Unoise ±0.01
        "gripper_pos":           0.003,  # Unoise ±0.003
        # object_position / object_yaw are handled separately (YOLO path uses
        # its own noise; ground-truth path uses the values below)
        # "object_position":       0.01,   # Unoise ±0.01  (gt-only path)
        # "object_yaw":            0.05,   # Unoise ±0.05  (gt-only path)
    }

    def _build_noise_slices(obs_manager, group="actor"):
        """Return list of (name, start, end, half_range) tuples for uniform noise injection."""
        import math as _math
        slices = []
        names = obs_manager.active_terms.get(group, [])
        shapes = obs_manager.group_obs_term_dim.get(group, [])
        cursor = 0
        for name, shape in zip(names, shapes):
            dim = _math.prod(shape)
            if name in _NOISE_SPECS:
                slices.append((name, cursor, cursor + dim, _NOISE_SPECS[name]))
            cursor += dim
        return slices

    _noise_slices = _build_noise_slices(env.observation_manager) if args.inject_noise else []



    # Running state trackers for active episodes
    current_episode_success = torch.zeros(num_envs, dtype=torch.bool, device=device)
    current_episode_top_collision = torch.zeros(num_envs, dtype=torch.bool, device=device)
    current_episode_grasp_and_lift = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # Storing previous step values for angle trigger (instant before contact)
    prev_squeeze_angles = torch.zeros(num_envs, device=device)
    prev_left_angles = torch.zeros(num_envs, device=device)
    prev_right_angles = torch.zeros(num_envs, device=device)
    prev_contact_both = torch.zeros(num_envs, dtype=torch.bool, device=device)

    robot = env.scene["robot"]
    box = env.scene["box"]
    command = env.command_manager.get_term("lift_height")

    # Grasp-and-lift: record the cube's initial Z height at episode start for each env
    LIFT_THRESHOLD_M = 0.01  # 1 cm above resting height
    initial_box_z = box.data.root_link_pos_w[:, 2].clone()  # [num_envs]

    # Get box geom ID and sizes for ground truth fallback
    geom_id = box.indexing.geom_ids[0]
    box_sizes = env.sim.model.geom_size[:, geom_id]

    # Get fingertip site names and IDs
    fingertip_site_names = [s for s in robot.site_names if "fingertip" in s]
    assert len(fingertip_site_names) == 2, f"Expected exactly 2 fingertip sites, found {len(fingertip_site_names)}"
    site_ids, _ = robot.find_sites(fingertip_site_names, preserve_order=True)
    left_idx, right_idx = site_ids[0], site_ids[1]

    # Initialize Kalman Filters list
    kfs = [None] * num_envs
    hsv_refs = [None] * num_envs
    is_grasped = np.zeros(num_envs, dtype=bool)
    grasp_override_active = np.zeros(num_envs, dtype=bool)

    # Find the grasp site
    grasp_site_idx, _ = robot.find_sites(["gripper_right_grasping_site"], preserve_order=True)
    grasp_idx = grasp_site_idx[0]

    print(f"Evaluating {args.num_episodes} total episodes with {num_envs} parallel environments...")
    step_count = 0
    
    while episodes_collected < args.num_episodes:
        with torch.no_grad():
            # A. Calculate fingertip positions and normals in world frame
            p_left = robot.data.site_pos_w[:, left_idx]   # [B, 3]
            p_right = robot.data.site_pos_w[:, right_idx] # [B, 3]
            
            # Squeeze axis
            v_squeeze = p_left - p_right
            v_squeeze_norm = v_squeeze / torch.norm(v_squeeze, dim=-1, keepdim=True).clamp(min=1e-6)
            
            # Fingertip site normals (local X axis of each site)
            xmat_left = env.sim.data.site_xmat[:, left_idx]   # [B, 3, 3]
            xmat_right = env.sim.data.site_xmat[:, right_idx] # [B, 3, 3]
            
            v_left = xmat_left[:, :, 0]
            v_right = -xmat_right[:, :, 0]
            
            # Cube orientation
            cube_quat = box.data.root_link_quat_w             # [B, 4]
            
            # B. Compute current fingertip angles wrt cube lateral surfaces
            squeeze_angles = compute_angle_wrt_cube_lateral_surfaces(v_squeeze_norm, cube_quat)
            left_angles = compute_angle_wrt_cube_lateral_surfaces(v_left, cube_quat)
            right_angles = compute_angle_wrt_cube_lateral_surfaces(v_right, cube_quat)
            
            # C. Check contact_both_fingers condition
            contact_both_float = site_contact_both_fingers(
                env=env,
                sensor_name="box_fingertip_contact",
                site_names=["gripper_right_fingertip_.*_site"]
            )
            contact_both = contact_both_float > 0.5
            
            # D. Check top surface penetration (collision)
            top_collision_float = top_surface_penetration_term(
                env=env,
                command_name="lift_height",
                threshold=0.0005
            )
            top_collision = top_collision_float > 0.5
            
            # E. Check success condition (distance to target position)
            position_error = torch.norm(command.target_pos - command.object_pos_w, dim=-1)
            success_now = position_error < command.cfg.success_threshold

            # F. Check grasp-and-lift condition: both fingers in contact AND cube ≥1 cm above resting height
            current_box_z = box.data.root_link_pos_w[:, 2]  # [num_envs]
            lifted_enough = (current_box_z - initial_box_z) >= LIFT_THRESHOLD_M
            grasp_and_lift_now = contact_both & lifted_enough

        # Track per-episode minimum position error
        current_episode_min_pos_error = torch.minimum(current_episode_min_pos_error, position_error)

        # Update running states for all active episodes
        current_episode_success |= success_now
        current_episode_top_collision |= top_collision
        current_episode_grasp_and_lift |= grasp_and_lift_now

        # Check for contact_both_fingers activation transition: False -> True
        # The instant before is the previous step.
        activated_now = contact_both & ~prev_contact_both & ~torch.tensor(contact_both_activated, device=device)
        
        for i in range(num_envs):
            if activated_now[i].item():
                contact_both_activated[i] = True
                # Record angles from the previous step (instant before contact)
                angles_trigger_squeeze.append(prev_squeeze_angles[i].item())
                angles_trigger_left.append(prev_left_angles[i].item())
                angles_trigger_right.append(prev_right_angles[i].item())

        # Update previous step values
        prev_squeeze_angles = squeeze_angles.clone()
        prev_left_angles = left_angles.clone()
        prev_right_angles = right_angles.clone()
        prev_contact_both = contact_both.clone()

        # F. Early success truncation: if an environment has achieved success,
        # we record it immediately, reset it in the simulation, and start a new episode.
        success_env_ids = success_now.nonzero(as_tuple=False).squeeze(-1)
        if len(success_env_ids) > 0:
            for idx in success_env_ids:
                i = idx.item()
                if episodes_collected < args.num_episodes:
                    success_recorded.append(current_episode_success[i].item())
                    top_collision_recorded.append(current_episode_top_collision[i].item())
                    grasp_and_lift_recorded.append(current_episode_grasp_and_lift[i].item())
                    min_pos_error_recorded.append(current_episode_min_pos_error[i].item())
                    episodes_collected += 1
                    
                    # Reset per-env state so the env can contribute another episode
                    current_episode_success[i] = False
                    current_episode_top_collision[i] = False
                    current_episode_grasp_and_lift[i] = False
                    current_episode_min_pos_error[i] = float("inf")
                    contact_both_activated[i] = False
                    prev_contact_both[i] = False
                    kfs[i] = None
                    hsv_refs[i] = None
                    is_grasped[i] = False
                    grasp_override_active[i] = False
            
            if episodes_collected >= args.num_episodes:
                break
                
            # Reset successful environments in simulation
            env.reset(env_ids=success_env_ids)
            # Refresh initial box Z for reset envs (env.reset updates sim state)
            initial_box_z[success_env_ids] = box.data.root_link_pos_w[success_env_ids, 2]

        # Extract camera observations and perform estimation
        est_pos_r = torch.zeros((num_envs, 3), device=device)
        est_yaw_r = torch.zeros((num_envs, 2), device=device)
        est_quat_r = torch.zeros((num_envs, 4), device=device)

        if args.enable_yolo:
            with torch.no_grad():
                # Extract RGB and depth images from env
                rgb = camera_rgb(env, "head_realsense_camera")
                depth = camera_depth(env, "head_realsense_camera", cutoff_distance=1.5)
                
                # Camera kinematics
                cam_idx = env.sim.mj_model.camera("robot/head_realsense_camera").id
                cam_pos = env.sim.data.cam_xpos[:, cam_idx]
                cam_xmat = env.sim.data.cam_xmat[:, cam_idx]
                cam_fovy = env.sim.mj_model.cam_fovy[cam_idx].item()

            H, W = 240, 320
            fovy_rad = math.radians(cam_fovy)
            fy = (H / 2.0) / math.tan(fovy_rad / 2.0)
            fx = fy
            cx = W / 2.0
            cy = H / 2.0

            # Convert RGB, Depth, and Kinematics tensors to CPU in single batch operations
            rgb_cpu = (rgb.cpu().permute(0, 2, 3, 1) * 255.0).clip(0, 255).to(torch.uint8).numpy()
            depth_cpu = depth.squeeze(1).cpu().numpy() * 1.5
            cam_pos_cpu = cam_pos.cpu().numpy()
            cam_xmat_cpu = cam_xmat.cpu().numpy()
            
            # Batch compute ground truth fallbacks to avoid per-environment synchronous .item() calls
            gt_pos_w_cpu = command.object_pos_w.cpu().numpy()
            _, _, gt_yaw_w_all = euler_xyz_from_quat(cube_quat)
            gt_yaw_w_cpu = gt_yaw_w_all.cpu().numpy()

            # Compute end-effector pose in robot base frame (in batch using PyTorch)
            ee_pos_w = robot.data.site_pos_w[:, grasp_idx]
            ee_pos_robot = quat_apply(quat_inv(robot.data.root_link_quat_w), ee_pos_w - robot.data.root_link_pos_w)
            ee_xmat = env.sim.data.site_xmat[:, grasp_idx]
            _, _, robot_yaw_w = euler_xyz_from_quat(robot.data.root_link_quat_w)
            ee_yaw_w = torch.atan2(ee_xmat[:, 1, 0], ee_xmat[:, 0, 0])
            ee_yaw = ee_yaw_w - robot_yaw_w
            
            ee_pos_robot_cpu = ee_pos_robot.cpu().numpy()
            ee_yaw_cpu = ee_yaw.cpu().numpy()

            rgb_list = list(rgb_cpu)

            # Run batch YOLO inference in sub-batches to avoid GPU OOM
            yolo_device = "cuda:0" if torch.cuda.is_available() else "cpu"
            sub_batch_size = 16
            yolo_results = []
            for start_idx in range(0, num_envs, sub_batch_size):
                end_idx = min(start_idx + sub_batch_size, num_envs)
                sub_batch = rgb_list[start_idx:end_idx]
                sub_results = yolo_model(sub_batch, verbose=False, device=yolo_device)
                yolo_results.extend(sub_results)
            dt = env.step_dt if hasattr(env, "step_dt") else (env.sim.model.opt.timestep * env.cfg.decimation)

            for i in range(num_envs):
                # 1. Find the best bounding box
                best_box = None
                best_conf = 0.0
                max_area = 0.33 * W * H
                for box_det in yolo_results[i].boxes:
                    conf = float(box_det.conf[0])
                    x1, y1, x2, y2 = map(int, box_det.xyxy[0])
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        continue
                    if conf > args.yolo_conf and conf > best_conf:
                        best_box = (x1, y1, x2, y2)
                        best_conf = conf

                # Ground truth fallbacks (pre-computed on CPU in batch)
                gt_pos_w_i = gt_pos_w_cpu[i]
                gt_yaw_w = gt_yaw_w_cpu[i]

                # Track grasped status based on contact_both signal
                is_grasped_signal = contact_both[i].item()
                if not is_grasped_signal:
                    grasp_override_active[i] = False
                    is_grasped[i] = False
                else:
                    if not grasp_override_active[i]:
                        is_grasped[i] = True

                ee_pose_i = (ee_pos_robot_cpu[i, 0], ee_pos_robot_cpu[i, 1], ee_pos_robot_cpu[i, 2], ee_yaw_cpu[i])

                success_fit = False
                px, py, pz = 0.0, 0.0, 0.0
                theta = 0.0
                length, width, height = args.cube_size[0], args.cube_size[1], args.cube_size[2]

                # Robot pose for transforms
                robot_pos = robot.data.root_link_pos_w[i].cpu().numpy()
                robot_quat = robot.data.root_link_quat_w[i].cpu().numpy()
                q_inv = np.array([robot_quat[0], -robot_quat[1], -robot_quat[2], -robot_quat[3]], dtype=np.float32)

                if best_box is not None:
                    x1, y1, x2, y2 = best_box
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(W, x2)
                    y2 = min(H, y2)

                    if (x2 - x1) > 1 and (y2 - y1) > 1:
                        depth_crop = depth_cpu[i, y1:y2, x1:x2]
                        valid_mask = (depth_crop > 0.1) & (depth_crop < 1.5) & np.isfinite(depth_crop)

                        # 1. Run HSV color segmentation first to find the cube pixels
                        rgb_mask = None
                        try:
                            roi = rgb_cpu[i, y1:y2, x1:x2]
                            if roi.size > 0:
                                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
                                rh, rw = roi.shape[:2]
                                
                                # Extract central 60% area (yolo_depth_pose_hsv uses 0.20 to 0.80)
                                cy1, cy2 = int(rh * 0.20), int(rh * 0.80)
                                cx1, cx2 = int(rw * 0.20), int(rw * 0.80)
                                center_pixels = roi_hsv[cy1:cy2, cx1:cx2]
                                
                                if center_pixels.size > 0:
                                    hues_rad = center_pixels[:, :, 0].astype(np.float32) * (2.0 * np.pi / 180.0)
                                    sin_mean = np.mean(np.sin(hues_rad))
                                    cos_mean = np.mean(np.cos(hues_rad))
                                    cand_h_rad = np.arctan2(sin_mean, cos_mean)
                                    if cand_h_rad < 0:
                                        cand_h_rad += 2.0 * np.pi
                                    cand_h = (cand_h_rad * (180.0 / np.pi) / 2.0) % 180.0
                                    cand_s = float(np.median(center_pixels[:, :, 1]))
                                    cand_v = float(np.median(center_pixels[:, :, 2]))
                                    
                                    if hsv_refs[i] is None:
                                        hsv_refs[i] = (cand_h, cand_s, cand_v)
                                    else:
                                        ref_h, ref_s, ref_v = hsv_refs[i]
                                        h_dist = abs(cand_h - ref_h)
                                        h_dist = min(h_dist, 180.0 - h_dist)
                                        
                                        hsv_lock_thresh_h = 20.0
                                        hsv_lock_thresh_sv = 40.0
                                        hsv_lock_blend = 0.15
                                        
                                        if (h_dist < hsv_lock_thresh_h and
                                                abs(cand_s - ref_s) < hsv_lock_thresh_sv and
                                                abs(cand_v - ref_v) < hsv_lock_thresh_sv):
                                            blend = hsv_lock_blend
                                            signed_h_diff = ((cand_h - ref_h + 90.0) % 180.0) - 90.0
                                            new_h = (ref_h + blend * signed_h_diff) % 180.0
                                            new_s = ref_s + blend * (cand_s - ref_s)
                                            new_v = ref_v + blend * (cand_v - ref_v)
                                            hsv_refs[i] = (new_h, new_s, new_v)
                                    
                                    dominant_h, dominant_s, dominant_v = hsv_refs[i]
                                    
                                    diff_h = np.abs(roi_hsv[:, :, 0].astype(np.int32) - dominant_h)
                                    diff_h = np.minimum(diff_h, 180 - diff_h)
                                    
                                    diff_s = np.abs(roi_hsv[:, :, 1].astype(np.int32) - dominant_s)
                                    diff_v = np.abs(roi_hsv[:, :, 2].astype(np.int32) - dominant_v)
                                    
                                    h_mask = diff_h < args.hsv_h_thresh
                                    s_mask = (diff_s < args.hsv_s_thresh) & (roi_hsv[:, :, 1] > 20)
                                    v_mask = (diff_v < args.hsv_v_thresh) & (roi_hsv[:, :, 2] > 20)
                                    
                                    rgb_mask = (h_mask & s_mask & v_mask).astype(np.uint8) * 255
                                    
                                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                                    rgb_mask = cv2.morphologyEx(rgb_mask, cv2.MORPH_CLOSE, kernel)
                                    rgb_mask = cv2.morphologyEx(rgb_mask, cv2.MORPH_OPEN, kernel)
                        except Exception:
                            pass

                        # 2. Apply depth mapping on the HSV-segmented region
                        inlier_mask = None
                        if rgb_mask is not None and np.sum(rgb_mask > 0) >= 5:
                            rgb_mask_depth = cv2.resize(rgb_mask, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
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

                                pos_cam = np.array([np.mean(X), np.mean(Y), np.mean(Z)])
                                points_cam = np.stack([X, Y, Z], axis=-1)
                                
                                # Convert to MuJoCo camera coordinate convention
                                points_mujoco = points_cam * np.array([1.0, -1.0, -1.0])
                                cam_pos_i = cam_pos_cpu[i]
                                cam_xmat_i = cam_xmat_cpu[i]
                                points_world = cam_pos_i + np.dot(points_mujoco, cam_xmat_i.T)
                                
                                # Transform to robot base frame
                                points_robot = rotate_by_quat_np(points_world - robot_pos, q_inv)
                                pos_world = cam_pos_i + np.dot(pos_cam * np.array([1.0, -1.0, -1.0]), cam_xmat_i.T)
                                pos_robot = rotate_by_quat_np(pos_world - robot_pos, q_inv)
                                px, py, pz = pos_robot[0], pos_robot[1], pos_robot[2]

                                if rgb_mask is not None:
                                    try:
                                        if len(points_robot) >= 5:
                                            # Use actual 3D points horizontal coordinates to avoid perspective warping
                                            coords = points_robot[:, :2].astype(np.float32)
                                            rect_3d = cv2.minAreaRect(coords)
                                            box_3d = cv2.boxPoints(rect_3d)
                                            
                                            px = float(rect_3d[0][0])
                                            py = float(rect_3d[0][1])
                                            pz = float((points_robot[:, 2].max() + points_robot[:, 2].min()) / 2.0)
                                            
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
                                            
                                            # Compute height from the 3D point cloud (camera-distance independent)
                                            # Outliers already removed by the 5 cm inlier gate above, so max-min suffices.
                                            z_vals = points_robot[:, 2]
                                            height = max(float(z_vals.max() - z_vals.min()), 0.01)
                                            
                                            success_fit = True
                                    except Exception:
                                        pass

                # Check EE override if grasped
                if success_fit and is_grasped[i]:
                    ee_x, ee_y, ee_z, ee_yaw = ee_pose_i
                    px, py, pz = ee_x, ee_y, ee_z
                    theta = ee_yaw
                    if kfs[i] is not None:
                        kfs[i]["yaw_locked"] = False
                        for kf, val in [(kfs[i]["kf_x"], px), (kfs[i]["kf_y"], py), (kfs[i]["kf_z"], pz)]:
                            if hasattr(kf, "x"):
                                if isinstance(kf.x, np.ndarray):
                                    kf.x[0] = val
                                    if len(kf.x) > 1:
                                        kf.x[1] = 0.0
                                else:
                                    kf.x = val
                        if hasattr(kfs[i]["kf_yaw"], "x"):
                            if isinstance(kfs[i]["kf_yaw"].x, np.ndarray):
                                kfs[i]["kf_yaw"].x = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
                            else:
                                kfs[i]["kf_yaw"].x = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)

                # Fallback: if depth pipeline failed but grasped is True, override to EE pose
                if not success_fit and is_grasped[i]:
                    px, py, pz = ee_pose_i[0], ee_pose_i[1], ee_pose_i[2]
                    theta = ee_pose_i[3]
                    if kfs[i] is not None:
                        length = float(kfs[i]["kf_len"].x)
                        width = float(kfs[i]["kf_wid"].x)
                        height = float(kfs[i]["kf_hgt"].x)
                    else:
                        length, width, height = args.cube_size[0], args.cube_size[1], args.cube_size[2]

                    if kfs[i] is not None:
                        kfs[i]["yaw_locked"] = False
                        for kf, val in [(kfs[i]["kf_x"], px), (kfs[i]["kf_y"], py), (kfs[i]["kf_z"], pz)]:
                            if hasattr(kf, "x"):
                                if isinstance(kf.x, np.ndarray):
                                    kf.x[0] = val
                                    if len(kf.x) > 1:
                                        kf.x[1] = 0.0
                                else:
                                    kf.x = val
                        if hasattr(kfs[i]["kf_yaw"], "x"):
                            if isinstance(kfs[i]["kf_yaw"].x, np.ndarray):
                                kfs[i]["kf_yaw"].x = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
                            else:
                                kfs[i]["kf_yaw"].x = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
                    success_fit = True

                # 2. Kalman / EMA Filter predict and update
                if success_fit:
                    if kfs[i] is None:
                        if args.pos_filter_type == "constant_velocity":
                            kf_x = ConstantVelocityKF(init_pos=px, init_vel=0.0, q_accel=0.0005, r_pos=1e-4)
                            kf_y = ConstantVelocityKF(init_pos=py, init_vel=0.0, q_accel=0.0005, r_pos=1e-4)
                            kf_z = ConstantVelocityKF(init_pos=pz, init_vel=0.0, q_accel=0.0005, r_pos=1e-4)
                        elif args.pos_filter_type == "ema":
                            kf_x = ExponentialMovingAverage(init_pos=px, alpha=args.ema_alpha)
                            kf_y = ExponentialMovingAverage(init_pos=py, alpha=args.ema_alpha)
                            kf_z = ExponentialMovingAverage(init_pos=pz, alpha=0.10)
                        else:  # "adaptive" (default)
                            kf_x = AdaptiveStaticKF(init_pos=px, q_base=1e-6, r_pos=1e-3, window=10)
                            kf_y = AdaptiveStaticKF(init_pos=py, q_base=1e-6, r_pos=1e-3, window=10)
                            kf_z = AdaptiveStaticKF(init_pos=pz, q_base=1e-6, r_pos=1e-3, window=10)

                        if args.pos_filter_type == "ema":
                            kf_yaw = ExponentialMovingAverageYaw(init_yaw=theta, alpha=0.20)
                            kf_len = ExponentialMovingAverage1D(init_val=length, alpha=0.10)
                            kf_wid = ExponentialMovingAverage1D(init_val=width, alpha=0.10)
                            kf_hgt = ExponentialMovingAverage1D(init_val=height, alpha=0.10)
                        else:
                            kf_yaw = YawKF(init_yaw=theta, q_yaw=0.0001, r_yaw=0.02)
                            kf_len = StaticState1DKF(init_val=length, q_val=1e-5, r_val=0.005)
                            kf_wid = StaticState1DKF(init_val=width, q_val=1e-5, r_val=0.005)
                            kf_hgt = StaticState1DKF(init_val=height, q_val=1e-5, r_val=0.005)

                        kfs[i] = {
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
                        kfs[i]["kf_x"].predict(dt)
                        kfs[i]["kf_y"].predict(dt)
                        kfs[i]["kf_z"].predict(dt)
                        kfs[i]["kf_yaw"].predict(dt)
                        kfs[i]["kf_len"].predict(dt)
                        kfs[i]["kf_wid"].predict(dt)
                        kfs[i]["kf_hgt"].predict(dt)

                        if args.pos_filter_type == "ema":
                            # Set alpha dynamically based on whether we are grasping
                            if is_grasped[i]:
                                kfs[i]["kf_x"].alpha = 0.001
                                kfs[i]["kf_y"].alpha = 0.001
                                kfs[i]["kf_z"].alpha = 0.001
                                kfs[i]["kf_yaw"].alpha = 0.0
                            else:
                                kfs[i]["kf_x"].alpha = args.ema_alpha
                                kfs[i]["kf_y"].alpha = args.ema_alpha
                                kfs[i]["kf_z"].alpha = 0.10
                                kfs[i]["kf_yaw"].alpha = 0.20

                        # --- Yaw Lock Logic ---
                        last_yaw = kfs[i]["kf_yaw"].get_yaw()
                        diff_angle = theta - last_yaw
                        # Cube rotational symmetry of 90 degrees (pi/2), wrap diff to [-pi/4, pi/4]
                        wrapped_diff = (diff_angle + math.pi / 4) % (math.pi / 2) - math.pi / 4

                        # Check displacement to potentially unlock
                        if kfs[i]["yaw_locked"]:
                            curr_filtered_x = kfs[i]["kf_x"].x[0]
                            curr_filtered_y = kfs[i]["kf_y"].x[0]
                            dist_moved = math.sqrt((curr_filtered_x - kfs[i]["yaw_lock_x"])**2 + (curr_filtered_y - kfs[i]["yaw_lock_y"])**2)
                            if dist_moved > 0.02:  # 2 cm
                                kfs[i]["yaw_locked"] = False

                        # Check angle update to potentially lock (only lock if not grasped/override)
                        if not kfs[i]["yaw_locked"] and not is_grasped[i]:
                            if abs(wrapped_diff) > math.radians(5.0):  # 5 degrees
                                kfs[i]["yaw_locked"] = True
                                kfs[i]["yaw_lock_x"] = kfs[i]["kf_x"].x[0]
                                kfs[i]["yaw_lock_y"] = kfs[i]["kf_y"].x[0]

                        u_x = kfs[i]["kf_x"].update(px)
                        u_y = kfs[i]["kf_y"].update(py)
                        z_r_pos = 0.15 if kfs[i]["occluded_while_grasping"] else None
                        u_z = kfs[i]["kf_z"].update(pz, r_pos=z_r_pos)
                        kfs[i]["occluded_while_grasping"] = False

                        if u_x and u_y and u_z:
                            if not kfs[i]["yaw_locked"]:
                                kfs[i]["kf_yaw"].update(theta)
                            kfs[i]["kf_len"].update(length)
                            kfs[i]["kf_wid"].update(width)
                            kfs[i]["kf_hgt"].update(height)

                        px = float(kfs[i]["kf_x"].x[0])
                        py = float(kfs[i]["kf_y"].x[0])
                        pz = float(kfs[i]["kf_z"].x[0])
                        theta = float(kfs[i]["kf_yaw"].get_yaw())
                        length = float(kfs[i]["kf_len"].x)
                        width = float(kfs[i]["kf_wid"].x)
                        height = float(kfs[i]["kf_hgt"].x)
                else:
                    if kfs[i] is not None:
                        z_q_scale = 500.0 if contact_both[i].item() else 10.0
                        kfs[i]["kf_x"].predict(dt, q_scale=10.0)
                        kfs[i]["kf_y"].predict(dt, q_scale=10.0)
                        kfs[i]["kf_z"].predict(dt, q_scale=z_q_scale)
                        kfs[i]["kf_yaw"].predict(dt, q_scale=10.0)
                        kfs[i]["kf_len"].predict(dt, q_scale=10.0)
                        kfs[i]["kf_wid"].predict(dt, q_scale=10.0)
                        kfs[i]["kf_hgt"].predict(dt, q_scale=10.0)

                        if contact_both[i].item():
                            kfs[i]["occluded_while_grasping"] = True

                        px = float(kfs[i]["kf_x"].x[0])
                        py = float(kfs[i]["kf_y"].x[0])
                        pz = float(kfs[i]["kf_z"].x[0])
                        theta = float(kfs[i]["kf_yaw"].get_yaw())
                        length = float(kfs[i]["kf_len"].x)
                        width = float(kfs[i]["kf_wid"].x)
                        height = float(kfs[i]["kf_hgt"].x)
                    else:
                        gt_pos_r_i = rotate_by_quat_np(gt_pos_w_i - robot_pos, q_inv)
                        px, py, pz = gt_pos_r_i[0], gt_pos_r_i[1], gt_pos_r_i[2]
                        
                        _, _, robot_yaw_w = euler_xyz_from_quat(torch.tensor(robot_quat, device=device).unsqueeze(0))
                        robot_yaw_w = robot_yaw_w[0].item()
                        theta = gt_yaw_w - robot_yaw_w
                        length, width, height = args.cube_size[0], args.cube_size[1], args.cube_size[2]

                if args.use_yaw_width_gt:
                    _, _, robot_yaw_w = euler_xyz_from_quat(torch.tensor(robot_quat, device=device).unsqueeze(0))
                    robot_yaw_w = robot_yaw_w[0].item()
                    relative_yaw = gt_yaw_w - robot_yaw_w
                    relative_yaw = (relative_yaw + math.pi/4) % (math.pi/2) - math.pi/4
                    theta = relative_yaw
                    length = args.cube_size[0]
                    width = args.cube_size[1]
                    height = args.cube_size[2]

                if args.exclude_width_prediction:
                    width = args.cube_size[1]

                # Set outputs in robot base frame
                est_pos_r[i] = torch.tensor([px, py, pz], device=device, dtype=torch.float32)
                est_yaw_r[i, 0] = math.cos(theta)
                est_yaw_r[i, 1] = math.sin(theta)
                est_quat_r[i] = torch.tensor([math.cos(theta/2), 0.0, 0.0, math.sin(theta/2)], device=device, dtype=torch.float32)

        # Step the policy and environment
        current_obs = TensorDict(env.obs_buf, batch_size=[num_envs])

        # Overwrite observations with YOLO estimates if enabled
        if args.enable_yolo:
            current_obs["actor"][:, 22:25] = est_pos_r    # object_position
            current_obs["actor"][:, 25:27] = est_yaw_r    # object_yaw

            if "critic" in current_obs.keys():
                current_obs["critic"][:, 22:25] = est_pos_r    # object_position
                current_obs["critic"][:, 25:29] = est_quat_r   # object_orientation
                current_obs["critic"][:, 29:31] = est_yaw_r    # object_yaw


        # Inject training-equivalent observation noise if requested
        if _noise_slices:
            actor_obs = current_obs["actor"]
            for (name, s, e, half) in _noise_slices:
                # Skip object_position / object_yaw when YOLO is active (already handled above)
                if args.enable_yolo and name in ["object_position", "object_yaw"]:
                    continue
                actor_obs[:, s:e] += (torch.rand(num_envs, e - s, device=device) * 2.0 - 1.0) * half
            current_obs["actor"] = actor_obs

        with torch.no_grad():
            action = model(current_obs)
            
        _, _, terminated, truncated, _ = env.step(action)
        dones = terminated | truncated
        step_count += 1
        if step_count % 10 == 0:
            print(f"[Step {step_count:4d}] Episode Progress: {episodes_collected:3d}/{args.num_episodes:3d} completed ({(episodes_collected / args.num_episodes) * 100.0:5.1f}%)")

        # Process ended episodes
        done_env_ids = []
        for i in range(num_envs):
            if dones[i].item() and episodes_collected < args.num_episodes:
                # Save final metrics for this episode
                success_recorded.append(current_episode_success[i].item())
                top_collision_recorded.append(current_episode_top_collision[i].item())
                grasp_and_lift_recorded.append(current_episode_grasp_and_lift[i].item())
                min_pos_error_recorded.append(current_episode_min_pos_error[i].item())
                episodes_collected += 1

                # Reset per-env state so the env can contribute another episode
                current_episode_success[i] = False
                current_episode_top_collision[i] = False
                current_episode_grasp_and_lift[i] = False
                current_episode_min_pos_error[i] = float("inf")
                contact_both_activated[i] = False
                prev_contact_both[i] = False

                # Reset Kalman Filter
                kfs[i] = None
                hsv_refs[i] = None
                is_grasped[i] = False
                grasp_override_active[i] = False
                done_env_ids.append(i)

        # Refresh initial box Z for envs that just finished (sim will have reset them)
        if done_env_ids:
            done_ids_tensor = torch.tensor(done_env_ids, device=device)
            initial_box_z[done_ids_tensor] = box.data.root_link_pos_w[done_ids_tensor, 2]

    env.close()
    
    # 6. Compute final statistics
    total_episodes = len(success_recorded)
    success_recorded_arr = np.array(success_recorded, dtype=bool)
    top_collision_recorded_arr = np.array(top_collision_recorded, dtype=bool)
    grasp_and_lift_recorded_arr = np.array(grasp_and_lift_recorded, dtype=bool)
    min_pos_error_arr = np.array(min_pos_error_recorded, dtype=np.float32)
    successful_runs = np.sum(success_recorded_arr)
    success_rate = (successful_runs / total_episodes) * 100.0

    total_top_collisions = np.sum(top_collision_recorded_arr)
    collision_rate = (total_top_collisions / total_episodes) * 100.0

    total_grasp_and_lift = np.sum(grasp_and_lift_recorded_arr)
    grasp_and_lift_rate = (total_grasp_and_lift / total_episodes) * 100.0

    # Per-threshold success rates based on minimum position error achieved
    THRESHOLDS_M = [0.01, 0.02, 0.03, 0.04, 0.05]  # 1-5 cm
    threshold_counts = [int(np.sum(min_pos_error_arr <= t)) for t in THRESHOLDS_M]
    threshold_rates  = [c / total_episodes * 100.0 for c in threshold_counts]
    
    # Compute mean and standard deviation of angles at trigger
    mean_squeeze = np.mean(angles_trigger_squeeze) if angles_trigger_squeeze else float('nan')
    std_squeeze = np.std(angles_trigger_squeeze) if angles_trigger_squeeze else float('nan')
    
    mean_left = np.mean(angles_trigger_left) if angles_trigger_left else float('nan')
    std_left = np.std(angles_trigger_left) if angles_trigger_left else float('nan')
    
    mean_right = np.mean(angles_trigger_right) if angles_trigger_right else float('nan')
    std_right = np.std(angles_trigger_right) if angles_trigger_right else float('nan')
    
    num_contacts_made = len(angles_trigger_squeeze)

    # 7. Print Results Report
    print("\n" + "=" * 80)
    print("                     POLICY EVALUATION METRICS REPORT")
    print("=" * 80)
    print(f"Task ID:               {args.task}")
    print(f"Checkpoint Path:       {args.checkpoint}")
    print(f"Parallel Environments: {num_envs}")
    print(f"Total Episodes Run:    {total_episodes}")
    print(f"Total Steps Simulated: {step_count}")
    print(f"Obs Noise Injection:   {'ENABLED (training-equivalent)' if args.inject_noise else 'disabled'}")
    print("-" * 80)
    print(f"Success Rate:          {success_rate:.2f}% ({successful_runs}/{total_episodes})")
    print(f"Grasp & Lift (≥1cm):   {total_grasp_and_lift} episodes ({grasp_and_lift_rate:.2f}%)")
    print(f"Top Surface Collisions: {total_top_collisions} episodes ({collision_rate:.2f}%)")
    print(f"Contacts Established:  {num_contacts_made} episodes")
    print("-" * 80)
    print("Success Rate by Distance Threshold (best pos-error reached in episode):")
    print(f"  {'Threshold':>10s} | {'Episodes':>8s} | {'Rate':>7s}")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*7}")
    for t, cnt, rate in zip(THRESHOLDS_M, threshold_counts, threshold_rates):
        print(f"  {t*100:>9.0f}cm | {cnt:>8d} | {rate:>6.2f}%")
    print("-" * 80)
    print("Fingertip Angles wrt Cube Lateral Surfaces (Instant Before Contact):")
    print(f"  Squeeze Axis Angle:  Mean = {mean_squeeze:6.2f}° | Std = {std_squeeze:5.2f}°")
    print(f"  Left Finger Normal:  Mean = {mean_left:6.2f}° | Std = {std_left:5.2f}°")
    print(f"  Right Finger Normal: Mean = {mean_right:6.2f}° | Std = {std_right:5.2f}°")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run policy evaluation on 100 episodes and record key metrics.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/lorenzobarbieri/2026-07-07_13-08-57/model_3500.pt",
        help="Path to policy checkpoint weights (default: 2026-06-25_13-06-56-checkpoints/model_39500.pt)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Mjlab-Manipulation-Lift-Cube-Pal-Tiago-Pro-v0",
        help="Task ID to evaluate (default: Mjlab-Manipulation-Lift-Cube-Pal-Tiago-Pro-v0)"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="Total number of evaluation episodes to collect (default: 100)"
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=32,
        help="Number of parallel MuJoCo environments (default: 16)"
    )
    parser.add_argument(
        "--pos_filter_type",
        type=str,
        default="ema",
        choices=["constant_velocity", "adaptive", "ema"],
        help="Type of 3D position filter to use: 'constant_velocity', 'adaptive' (Kalman), or 'ema' (Exponential Moving Average) (default: ema)."
    )
    parser.add_argument(
        "--ema_alpha",
        type=float,
        default=0.2,
        help="EMA smoothing factor alpha in [0, 1] (only used when --pos_filter_type=ema, default: 0.2). Higher = less smoothing."
    )
    parser.add_argument(
        "--yolo_conf",
        type=float,
        default=0.45,
        help="YOLO detection confidence threshold (default: 0.45)."
    )
    parser.add_argument(
        "--hsv_h_thresh",
        type=int,
        default=100,
        help="Hue threshold for HSV segmentation (default: 20)."
    )
    parser.add_argument(
        "--hsv_s_thresh",
        type=int,
        default=100,
        help="Saturation threshold for HSV segmentation (default: 40)."
    )
    parser.add_argument(
        "--hsv_v_thresh",
        type=int,
        default=80,
        help="Value threshold for HSV segmentation (default: 40)."
    )
    parser.add_argument(
        "--exclude_width_prediction",
        action="store_true",
        default=False,
        help="Exclude width prediction and use the nominal cube width instead."
    )
    parser.add_argument(
        "--cube_size",
        nargs=3,
        type=float,
        default=[0.035, 0.035, 0.05],
        help="Nominal cube size in meters (length, width, height) (default: [0.04, 0.04, 0.075])."
    )
    parser.add_argument(
        "--use_yaw_width_gt",
        action="store_true",
        default=False,
        help="Override estimated yaw and dimensions with ground truth values."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)"
    )
    parser.add_argument(
        "--enable_yolo",
        action="store_true",
        default=True,
        help="Enable hybrid YOLO-based estimation mode (default: True)"
    )
    parser.add_argument(
        "--no_yolo",
        action="store_false",
        dest="enable_yolo",
        help="Disable hybrid YOLO-based estimation mode (uses ground truth state feedback)"
    )
    parser.add_argument(
        "--inject_noise",
        action="store_true",
        default=False,
        help="Inject training-equivalent uniform observation noise into the actor obs buffer "
             "(joint_pos ±0.02, joint_vel ±0.05, ee_position ±0.01, gripper_pos ±0.003, "
             "target_object_position ±0.01; object_position ±0.01 / object_yaw ±0.05 when "
             "YOLO is disabled). Default: False."
    )
    parser.add_argument(
        "--yolo_model",
        type=str,
        default="/home/lorenzobarbieri/exchange/tiago_pro_sim_ws/runs/detect/tiago_single_class_yolo26/weights/best.pt",
        help="Path to YOLO model weights (default: /home/lorenzobarbieri/exchange/tiago_pro_sim_ws/runs/detect/tiago_single_class_yolo26/weights/best.pt)"
    )
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
