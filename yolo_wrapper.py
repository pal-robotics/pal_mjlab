import sys
import os
import math
import torch
import numpy as np
import cv2
from ultralytics import YOLO

# Sourcing the filters from the workspace
sys.path.append("/home/lorenzobarbieri/exchange/tiago_pro_sim_ws/src")
try:
    from filters import AdaptiveStaticKF, YawKF, StaticState1DKF
except ImportError:
    # Minimal fallback implementations for safety
    class AdaptiveStaticKF:
        def __init__(self, init_pos, q_base=1e-6, r_pos=1e-3, window=10):
            self.x = np.array([init_pos, 0.0], dtype=np.float32)
            self.P = np.array([[1.0, 0.0], [0.0, 10.0]], dtype=np.float32)
            self.q_base = q_base
            self.r_pos = r_pos
            self.current_alpha = 1.0
        def predict(self, dt, q_scale=1.0):
            F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float32)
            Q = (self.q_base * self.current_alpha * q_scale) * np.array([[(dt**3)/3.0, (dt**2)/2.0], [(dt**2)/2.0, dt]], dtype=np.float32)
            self.x = F @ self.x
            self.P = F @ self.P @ F.T + Q
        def update(self, z_pos, r_pos=None):
            H = np.array([[1.0, 0.0]], dtype=np.float32)
            actual_r = r_pos if r_pos is not None else self.r_pos
            R = np.array([[actual_r]], dtype=np.float32)
            innovation = z_pos - (H @ self.x)[0]
            if abs(innovation) > 0.05 and r_pos is None: return False
            S = H @ self.P @ H.T + R
            K = self.P @ H.T / S[0, 0]
            self.x = self.x + K.flatten() * innovation
            self.P = (np.eye(2, dtype=np.float32) - np.outer(K.flatten(), H.flatten())) @ self.P
            return True
            
    class YawKF:
        def __init__(self, init_yaw, q_yaw=0.0001, r_yaw=0.02):
            self.x = np.array([math.cos(init_yaw), math.sin(init_yaw)], dtype=np.float32)
            self.P = np.eye(2, dtype=np.float32) * 1.0
            self.q_yaw = q_yaw
            self.r_yaw = r_yaw
        def predict(self, dt, q_scale=1.0):
            self.P = self.P + np.eye(2, dtype=np.float32) * (self.q_yaw * q_scale * dt)
        def update(self, z_yaw):
            z = np.array([math.cos(z_yaw), math.sin(z_yaw)], dtype=np.float32)
            y = z - self.x
            S = self.P + np.eye(2, dtype=np.float32) * self.r_yaw
            K = self.P @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = (np.eye(2, dtype=np.float32) - K) @ self.P
            norm = np.linalg.norm(self.x)
            if norm > 1e-5: self.x = self.x / norm
        def get_yaw(self):
            return math.atan2(self.x[1], self.x[0])
            
    class StaticState1DKF:
        def __init__(self, init_val, q_val=1e-5, r_val=0.005):
            self.x = float(init_val)
            self.P = 1.0
            self.q_val = q_val
            self.r_val = r_val
        def predict(self, dt, q_scale=1.0):
            self.P = self.P + (self.q_val * q_scale * dt)
        def update(self, z_val):
            y = z_val - self.x
            S = self.P + self.r_val
            K = self.P / S
            self.x = self.x + K * y
            self.P = (1.0 - K) * self.P

from mjlab.tasks.manipulation.mdp import camera_rgb, camera_depth
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply, quat_inv, quat_mul
from pal_mjlab.tasks.manipulation.mdp.contact_sensor import site_contact_both_fingers

class YoloPerceptionWrapper:
    def __init__(self, env, yolo_model_path):
        self.env = env
        self.num_envs = env.num_envs
        self.device = env.device

        print(f"[YOLO Wrapper] Loading YOLO model from {yolo_model_path}...")
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.yolo_model.to(self.yolo_device)
        print(f"[YOLO Wrapper] YOLO model loaded on device: {self.yolo_device}")

        # Entities and sensing setup
        self.robot = env.scene["robot"]
        self.box = env.scene["box"]
        self.command = env.command_manager.get_term("lift_height")
        
        self.geom_id = self.box.indexing.geom_ids[0]
        
        # Get fingertip site IDs
        fingertip_site_names = [s for s in self.robot.site_names if "fingertip" in s]
        assert len(fingertip_site_names) == 2, f"Expected exactly 2 fingertip sites, found {len(fingertip_site_names)}"
        site_ids, _ = self.robot.find_sites(fingertip_site_names, preserve_order=True)
        self.left_idx, self.right_idx = site_ids[0], site_ids[1]

        # Kalman Filters tracking
        self.kfs = [None] * self.num_envs
        self.prev_contact_both = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def __getattr__(self, name):
        # Delegate all other calls/attributes to the wrapped environment
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        # Reset all filters and trackers
        self.kfs = [None] * self.num_envs
        self.prev_contact_both = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Process first observations
        obs = self._process_observations(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Reset Kalman Filters for environments that finished their episode
        dones = terminated | truncated
        for i in range(self.num_envs):
            if dones[i].item():
                self.kfs[i] = None
                self.prev_contact_both[i] = False

        # Override observations with YOLO estimates
        obs = self._process_observations(obs)
        return obs, reward, terminated, truncated, info

    def _process_observations(self, obs_dict):
        # Extract RGB, Depth and camera info
        with torch.no_grad():
            rgb = camera_rgb(self.env, "head_realsense_camera")
            depth = camera_depth(self.env, "head_realsense_camera", cutoff_distance=1.5)
            
            cam_idx = self.env.sim.mj_model.camera("robot/head_realsense_camera").id
            cam_pos = self.env.sim.data.cam_xpos[:, cam_idx]
            cam_xmat = self.env.sim.data.cam_xmat[:, cam_idx]
            cam_fovy = self.env.sim.mj_model.cam_fovy[cam_idx].item()

            # Fingertip contact state
            contact_both_float = site_contact_both_fingers(
                env=self.env,
                sensor_name="box_fingertip_contact",
                site_names=["gripper_right_fingertip_.*_site"]
            )
            contact_both = contact_both_float > 0.5

        H, W = 240, 320
        fovy_rad = math.radians(cam_fovy)
        fy = (H / 2.0) / math.tan(fovy_rad / 2.0)
        fx = fy
        cx = W / 2.0
        cy = H / 2.0

        # CPU copies for OpenCV / NumPy operations
        rgb_cpu = (rgb.cpu().permute(0, 2, 3, 1) * 255.0).clip(0, 255).to(torch.uint8).numpy()
        depth_cpu = depth.squeeze(1).cpu().numpy() * 1.5
        cam_pos_cpu = cam_pos.cpu().numpy()
        cam_xmat_cpu = cam_xmat.cpu().numpy()

        # Ground-truth fallbacks (precomputed)
        gt_pos_w_cpu = self.command.object_pos_w.cpu().numpy()
        cube_quat = self.box.data.root_link_quat_w
        _, _, gt_yaw_w_all = euler_xyz_from_quat(cube_quat)
        gt_yaw_w_cpu = gt_yaw_w_all.cpu().numpy()
        box_sizes = self.env.sim.model.geom_size[:, self.geom_id]

        rgb_list = list(rgb_cpu)

        # Sub-batched YOLO inference on GPU to protect VRAM
        sub_batch_size = 16
        yolo_results = []
        for start_idx in range(0, self.num_envs, sub_batch_size):
            end_idx = min(start_idx + sub_batch_size, self.num_envs)
            sub_batch = rgb_list[start_idx:end_idx]
            sub_results = self.yolo_model(sub_batch, verbose=False, device=self.yolo_device)
            yolo_results.extend(sub_results)

        dt = self.env.step_dt if hasattr(self.env, "step_dt") else (self.env.sim.model.opt.timestep * self.env.cfg.decimation)

        est_pos_r = torch.zeros((self.num_envs, 3), device=self.device)
        est_width_r = torch.zeros((self.num_envs, 1), device=self.device)
        est_yaw_r = torch.zeros((self.num_envs, 2), device=self.device)

        for i in range(self.num_envs):
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
                if conf > 0.25 and conf > best_conf:
                    best_box = (x1, y1, x2, y2)
                    best_conf = conf

            # Fallbacks
            gt_pos_w_i = gt_pos_w_cpu[i]
            gt_yaw_w = gt_yaw_w_cpu[i]
            gt_width = (box_sizes[i, 1] * 2.0).item()

            success_fit = False
            px, py, pz = 0.0, 0.0, 0.0
            theta = 0.0
            length, width, height = 0.05, 0.05, 0.05

            if best_box is not None:
                x1, y1, x2, y2 = best_box
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)

                if (x2 - x1) > 1 and (y2 - y1) > 1:
                    depth_crop = depth_cpu[i, y1:y2, x1:x2]
                    valid_mask = (depth_crop > 0.1) & (depth_crop < 1.5) & np.isfinite(depth_crop)

                    if np.sum(valid_mask) >= 5:
                        median_depth = np.median(depth_crop[valid_mask])
                        inlier_mask = valid_mask & (np.abs(depth_crop - median_depth) < 0.05)

                        if np.sum(inlier_mask) >= 5:
                            u_range = np.arange(x1, x2)
                            v_range = np.arange(y1, y2)
                            uu, vv = np.meshgrid(u_range, v_range)

                            valid_depths = depth_crop[inlier_mask]
                            valid_u = uu[inlier_mask]
                            valid_v = vv[inlier_mask]

                            X = (valid_u - cx) * valid_depths / fx
                            Y = (valid_v - cy) * valid_depths / fy
                            Z = valid_depths

                            points_cam = np.stack([X, Y, Z], axis=-1)
                            points_mujoco = points_cam * np.array([1.0, -1.0, -1.0])
                            cam_pos_i = cam_pos_cpu[i]
                            cam_xmat_i = cam_xmat_cpu[i]
                            points_world = cam_pos_i + np.dot(points_mujoco, cam_xmat_i.T)

                            try:
                                min_z = np.min(points_world[:, 2])
                                above_table_mask = points_world[:, 2] > (min_z + 0.008)
                                points_above = points_world[above_table_mask]

                                if len(points_above) >= 5:
                                    centroid_above = np.mean(points_above, axis=0)
                                    dists = np.linalg.norm(points_above - centroid_above, axis=1)
                                    inlier_pts_mask = dists < 0.055
                                    filtered = points_above[inlier_pts_mask]

                                    if len(filtered) >= 5:
                                        coords = filtered[:, :2].astype(np.float32)
                                        rect = cv2.minAreaRect(coords)
                                        box_pts = cv2.boxPoints(rect)

                                        px = float(rect[0][0])
                                        py = float(rect[0][1])
                                        pz = float(np.mean(filtered[:, 2]))

                                        p0, p1, p2, p3 = box_pts
                                        v1 = p1 - p0
                                        v2 = p2 - p1
                                        angle1 = math.atan2(v1[1], v1[0])
                                        angle2 = math.atan2(v2[1], v2[0])

                                        def wrap_to_pi_over_2(angle):
                                            return (angle + math.pi/2) % math.pi - math.pi/2

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

                                        height = float(np.max(filtered[:, 2]) - np.min(filtered[:, 2])) + 0.008
                                        success_fit = True
                            except Exception:
                                pass

            # 2. Kalman Filter update/predict logic
            if success_fit:
                if self.kfs[i] is None:
                    self.kfs[i] = {
                        "kf_x": AdaptiveStaticKF(px, q_base=1e-4, r_pos=1e-3, window=10),
                        "kf_y": AdaptiveStaticKF(py, q_base=1e-4, r_pos=1e-3, window=10),
                        "kf_z": AdaptiveStaticKF(pz, q_base=1e-4, r_pos=1e-3, window=10),
                        "kf_yaw": YawKF(theta, q_yaw=0.0001, r_yaw=0.02),
                        "kf_len": StaticState1DKF(length, q_val=1e-5, r_val=0.005),
                        "kf_wid": StaticState1DKF(width, q_val=1e-5, r_val=0.005),
                        "kf_hgt": StaticState1DKF(height, q_val=1e-5, r_val=0.005),
                        "occluded_while_grasping": False,
                    }
                    est_pos_w_i = np.array([px, py, pz])
                    est_yaw_w_i = theta
                    est_width_i = width
                else:
                    self.kfs[i]["kf_x"].predict(dt)
                    self.kfs[i]["kf_y"].predict(dt)
                    self.kfs[i]["kf_z"].predict(dt)
                    self.kfs[i]["kf_yaw"].predict(dt)
                    self.kfs[i]["kf_len"].predict(dt)
                    self.kfs[i]["kf_wid"].predict(dt)
                    self.kfs[i]["kf_hgt"].predict(dt)

                    u_x = self.kfs[i]["kf_x"].update(px)
                    u_y = self.kfs[i]["kf_y"].update(py)
                    z_r_pos = 0.15 if self.kfs[i]["occluded_while_grasping"] else None
                    u_z = self.kfs[i]["kf_z"].update(pz, r_pos=z_r_pos)
                    self.kfs[i]["occluded_while_grasping"] = False

                    if u_x and u_y and u_z:
                        self.kfs[i]["kf_yaw"].update(theta)
                        self.kfs[i]["kf_len"].update(length)
                        self.kfs[i]["kf_wid"].update(width)
                        self.kfs[i]["kf_hgt"].update(height)

                    est_pos_w_i = np.array([
                        self.kfs[i]["kf_x"].x[0],
                        self.kfs[i]["kf_y"].x[0],
                        self.kfs[i]["kf_z"].x[0],
                    ])
                    est_yaw_w_i = self.kfs[i]["kf_yaw"].get_yaw()
                    est_width_i = self.kfs[i]["kf_wid"].x
            else:
                if self.kfs[i] is not None:
                    z_q_scale = 500.0 if contact_both[i].item() else 10.0
                    self.kfs[i]["kf_x"].predict(dt, q_scale=10.0)
                    self.kfs[i]["kf_y"].predict(dt, q_scale=10.0)
                    self.kfs[i]["kf_z"].predict(dt, q_scale=z_q_scale)
                    self.kfs[i]["kf_yaw"].predict(dt, q_scale=10.0)
                    self.kfs[i]["kf_len"].predict(dt, q_scale=10.0)
                    self.kfs[i]["kf_wid"].predict(dt, q_scale=10.0)
                    self.kfs[i]["kf_hgt"].predict(dt, q_scale=10.0)

                    if contact_both[i].item():
                        self.kfs[i]["occluded_while_grasping"] = True

                    est_pos_w_i = np.array([
                        self.kfs[i]["kf_x"].x[0],
                        self.kfs[i]["kf_y"].x[0],
                        self.kfs[i]["kf_z"].x[0],
                    ])
                    est_yaw_w_i = self.kfs[i]["kf_yaw"].get_yaw()
                    est_width_i = self.kfs[i]["kf_wid"].x
                else:
                    est_pos_w_i = gt_pos_w_i
                    est_yaw_w_i = gt_yaw_w
                    est_width_i = gt_width

            # 3. Transform to robot root frame
            est_pos_w_i_t = torch.tensor(est_pos_w_i, device=self.device, dtype=torch.float32)
            robot_pos_w_i = self.robot.data.root_link_pos_w[i]
            robot_quat_w_i = self.robot.data.root_link_quat_w[i]

            pos_rel = quat_apply(quat_inv(robot_quat_w_i.unsqueeze(0)), (est_pos_w_i_t - robot_pos_w_i).unsqueeze(0))[0]
            est_pos_r[i] = pos_rel
            est_width_r[i, 0] = est_width_i

            # Orientation in robot frame
            est_quat_w_i = torch.tensor([math.cos(est_yaw_w_i/2), 0.0, 0.0, math.sin(est_yaw_w_i/2)], device=self.device, dtype=torch.float32)
            quat_rel = quat_mul(quat_inv(robot_quat_w_i.unsqueeze(0)), est_quat_w_i.unsqueeze(0))[0]
            _, _, yaw_rel = euler_xyz_from_quat(quat_rel.unsqueeze(0))
            yaw_rel = yaw_rel[0].item()
            est_yaw_r[i, 0] = math.cos(yaw_rel)
            est_yaw_r[i, 1] = math.sin(yaw_rel)

        # Update previous contact state
        self.prev_contact_both = contact_both.clone()

        # Mutate observations in the obs dictionary in-place
        obs_dict["actor"][:, 22:25] = est_pos_r    # object_position
        obs_dict["actor"][:, 25:26] = est_width_r  # object_width
        obs_dict["actor"][:, 26:28] = est_yaw_r    # object_yaw

        if "critic" in obs_dict:
            # We don't have est_quat_r in full 4D format but we can reconstruct it from est_yaw_w_i
            # Actually, the critic in play.py might not even be used, but let's override it for safety
            obs_dict["critic"][:, 22:25] = est_pos_r    # object_position
            # Re-construct 4D rel quat for the entire batch
            critic_quat_rel = torch.zeros((self.num_envs, 4), device=self.device)
            for i in range(self.num_envs):
                est_quat_w_i = torch.tensor([math.cos(est_yaw_w_i/2), 0.0, 0.0, math.sin(est_yaw_w_i/2)], device=self.device, dtype=torch.float32)
                critic_quat_rel[i] = quat_mul(quat_inv(self.robot.data.root_link_quat_w[i].unsqueeze(0)), est_quat_w_i.unsqueeze(0))[0]
            obs_dict["critic"][:, 25:29] = critic_quat_rel  # object_orientation
            obs_dict["critic"][:, 29:30] = est_width_r     # object_width
            obs_dict["critic"][:, 30:32] = est_yaw_r       # object_yaw

        return obs_dict
