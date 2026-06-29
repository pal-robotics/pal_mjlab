import torch
import math
import numpy as np
import cv2
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.manipulation.mdp import camera_rgb, camera_depth
from tensordict import TensorDict

# Import ML model and YOLO
from ultralytics import YOLO
import sys
import os

# Import load_class helper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate_policy import load_class

# Include our filters
sys.path.append("/home/lorenzobarbieri/exchange/tiago_pro_sim_ws/src")
try:
    from filters import AdaptiveStaticKF, YawKF, StaticState1DKF
except ImportError:
    # Inline fallbacks if not found
    class AdaptiveStaticKF:
        def __init__(self, x0, q_base=1e-6, r_pos=1e-3, window=10):
            self.x = np.array([x0, 0.0])
        def predict(self, dt, q_scale=1.0): pass
        def update(self, z): return True
    class YawKF:
        def __init__(self, yaw0, q_yaw=0.0001, r_yaw=0.02):
            self.yaw = yaw0
        def predict(self, dt, q_scale=1.0): pass
        def update(self, z): pass
        def get_yaw(self): return self.yaw
    class StaticState1DKF:
        def __init__(self, x0, q_val=1e-5, r_val=0.005):
            self.x = x0
        def predict(self, dt, q_scale=1.0): pass
        def update(self, z): pass

# Helper functions for quaternions
def quat_inv(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return torch.stack([w, -x, -y, -z], dim=-1)

def quat_apply(q, v):
    q_w = q[..., 0]
    q_xyz = q[..., 1:]
    t = 2.0 * torch.linalg.cross(q_xyz, v, dim=-1)
    return v + q_w.unsqueeze(-1) * t + torch.linalg.cross(q_xyz, t, dim=-1)

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)

def euler_xyz_from_quat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    pitch = torch.where(torch.abs(sinp) >= 1, torch.copysign(torch.tensor(math.pi / 2, device=q.device), sinp), torch.asin(sinp))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    
    # 1. Load YOLO
    yolo_path = "/home/lorenzobarbieri/exchange/tiago_pro_sim_ws/runs/detect/tiago_single_class_yolo26/weights/best.pt"
    print("Loading YOLO model from:", yolo_path)
    yolo_model = YOLO(yolo_path)
    
    # 2. Load Env Configs
    print("Loading configurations...")
    env_cfg = load_env_cfg("Mjlab-Manipulation-Lift-Cube-Pal-Tiago-Pro-v0", play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.scene.sensors = (env_cfg.scene.sensors or ()) + (
        CameraSensorCfg(
            name="head_realsense_camera",
            height=240,
            width=320,
            data_types=("rgb", "depth"),
            camera_name="robot/head_realsense_camera",
        ),
    )
    
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    rl_cfg = load_rl_cfg("Mjlab-Manipulation-Lift-Cube-Pal-Tiago-Pro-v0")
    
    # 3. Instantiate Policy Model
    print("Setting up MLP model...")
    actor_cfg = rl_cfg.actor
    model_cls = load_class(actor_cfg.class_name)
    
    obs_dict, _ = env.reset()
    dummy_obs = TensorDict(obs_dict, batch_size=[1])
    
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
    
    # Load state dict
    checkpoint_path = "2026-06-23_17-50-36-checkpoints/model_20500.pt"
    print("Loading Policy weights from:", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    model.eval()
    
    # Setup helpers
    robot = env.scene["robot"]
    box = env.scene["box"]
    command = env.command_manager.get_term("lift_height")
    
    # Get box geom ID and sizes for ground truth fallback
    geom_id = box.indexing.geom_ids[0]
    box_sizes = env.sim.model.geom_size[:, geom_id]
    
    # Kalman Filter
    kf = None
    
    # Run 50 steps
    print("\nStarting step-by-step diagnostic loop...\n")
    for step in range(50):
        print(f"==================== STEP {step:2d} ====================")
        
        with torch.no_grad():
            rgb = camera_rgb(env, "head_realsense_camera")
            depth = camera_depth(env, "head_realsense_camera", cutoff_distance=1.5)
            
            # Camera kinematics
            cam_idx = env.sim.mj_model.camera("robot/head_realsense_camera").id
            cam_pos = env.sim.data.cam_xpos[:, cam_idx]
            cam_xmat = env.sim.data.cam_xmat[:, cam_idx]
            cam_fovy = env.sim.mj_model.cam_fovy[cam_idx].item()
            
            # Ground truth from simulator
            gt_pos_w = box.data.root_link_pos_w[0]
            # Get actual cube quaternion
            cube_quat = box.data.root_link_quat_w[0]
            
            # Ground truth robot relative
            # (object_position observation is usually index 3 of actor)
            gt_pos_r = env.obs_buf["actor"][0, 22:25]
            gt_width_r = env.obs_buf["actor"][0, 25:26]
            gt_yaw_r = env.obs_buf["actor"][0, 26:28]
            
            print(f"GT World Pos:  [{gt_pos_w[0].item():.4f}, {gt_pos_w[1].item():.4f}, {gt_pos_w[2].item():.4f}]")
            print(f"GT Robot Pos:  [{gt_pos_r[0].item():.4f}, {gt_pos_r[1].item():.4f}, {gt_pos_r[2].item():.4f}]")
            print(f"GT Robot Yaw:  [{gt_yaw_r[0].item():.4f}, {gt_yaw_r[1].item():.4f}]")
            print(f"GT Width:      {gt_width_r[0].item():.4f}")
            
        # Image prep
        H, W = 240, 320
        fovy_rad = math.radians(cam_fovy)
        fy = (H / 2.0) / math.tan(fovy_rad / 2.0)
        fx = fy
        cx = W / 2.0
        cy = H / 2.0
        
        rgb_cpu = (rgb.cpu().permute(0, 2, 3, 1) * 255.0).clip(0, 255).to(torch.uint8).numpy()
        depth_cpu = depth.squeeze(1).cpu().numpy() * 1.5
        
        # Project GT cube position into camera frame and image plane
        cam_pos_i = cam_pos[0].cpu().numpy()
        cam_xmat_i = cam_xmat[0].cpu().numpy()
        gt_pos_w_i = gt_pos_w.cpu().numpy()
        
        # p_local = cam_xmat.T * (gt_pos - cam_pos)
        # Note: cam_xmat in MuJoCo is a 3x3 matrix mapping local camera to world.
        # So local = cam_xmat.T * (world - cam_pos).
        p_local = np.dot(gt_pos_w_i - cam_pos_i, cam_xmat_i)
        d_gt = -p_local[2]
        u_gt = cx + p_local[0] * fx / d_gt
        v_gt = cy - p_local[1] * fy / d_gt
        
        print(f"GT Cam Local:  [{p_local[0]:.4f}, {p_local[1]:.4f}, {p_local[2]:.4f}]")
        print(f"GT Projected:  Pixel = ({u_gt:.1f}, {v_gt:.1f}), Depth = {d_gt:.4f}m")
        if 0 <= int(u_gt) < W and 0 <= int(v_gt) < H:
            actual_depth_at_gt = depth_cpu[0, int(v_gt), int(u_gt)]
            print(f"Sensor Depth:  {actual_depth_at_gt:.4f}m (at GT pixel)")
        else:
            print("Sensor Depth:  OUT OF BOUNDS")
            
        # YOLO inference
        yolo_results = yolo_model(list(rgb_cpu), verbose=False)
        
        best_box = None
        best_conf = 0.0
        max_area = 0.33 * W * H
        for box_det in yolo_results[0].boxes:
            conf = float(box_det.conf[0])
            x1, y1, x2, y2 = map(int, box_det.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                continue
            if conf > 0.25 and conf > best_conf:
                best_box = (x1, y1, x2, y2)
                best_conf = conf
                
        if best_box is not None:
            print(f"YOLO Box:      {best_box} (Conf: {best_conf:.3f})")
        else:
            print("YOLO Box:      NONE")
            
        # Fitting
        success_fit = False
        px, py, pz = 0.0, 0.0, 0.0
        theta = 0.0
        length, width, height = 0.05, 0.05, 0.05
        
        if best_box is not None:
            x1, y1, x2, y2 = best_box
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2)
            y2 = min(H, y2)
            
            if (x2 - x1) > 1 and (y2 - y1) > 1:
                depth_crop = depth_cpu[0, y1:y2, x1:x2]
                valid_mask = (depth_crop > 0.1) & (depth_crop < 1.5) & np.isfinite(depth_crop)
                print(f"Depth Crop:    Valid Pixels = {np.sum(valid_mask)}")
                
                if np.sum(valid_mask) >= 5:
                    median_depth = np.median(depth_crop[valid_mask])
                    inlier_mask = valid_mask & (np.abs(depth_crop - median_depth) < 0.05)
                    print(f"Depth Inliers: {np.sum(inlier_mask)} (Median Depth: {median_depth:.4f}m)")
                    
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
                        points_world = cam_pos_i + np.dot(points_mujoco, cam_xmat_i.T)
                        
                        try:
                            min_z = np.min(points_world[:, 2])
                            above_table_mask = points_world[:, 2] > (min_z + 0.008)
                            points_above = points_world[above_table_mask]
                            print(f"Points Above:  {len(points_above)}")
                            
                            if len(points_above) >= 5:
                                centroid_above = np.mean(points_above, axis=0)
                                dists = np.linalg.norm(points_above - centroid_above, axis=1)
                                inlier_pts_mask = dists < 0.055
                                filtered = points_above[inlier_pts_mask]
                                print(f"Filtered Pts:  {len(filtered)}")
                                
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
                                    print(f"Fit World Pos: [{px:.4f}, {py:.4f}, {pz:.4f}]")
                                    print(f"Fit Yaw/Width: Yaw = {math.degrees(theta):.2f}°, Width = {width:.4f}")
                        except Exception as e:
                            print("Fitting Error:", e)
                            
        # Kalman Filter
        dt = env.step_dt if hasattr(env, "step_dt") else (env.sim.model.opt.timestep * env.cfg.decimation)
        
        # GT fallback variables for printout
        gt_yaw_w = float(euler_xyz_from_quat(cube_quat.unsqueeze(0))[2][0].item())
        gt_width = float(box_sizes[0, 1] * 2.0)
        
        if success_fit:
            if kf is None:
                kf = {
                    "kf_x": AdaptiveStaticKF(px, q_base=1e-6, r_pos=1e-3, window=10),
                    "kf_y": AdaptiveStaticKF(py, q_base=1e-6, r_pos=1e-3, window=10),
                    "kf_z": AdaptiveStaticKF(pz, q_base=1e-6, r_pos=1e-3, window=10),
                    "kf_yaw": YawKF(theta, q_yaw=0.0001, r_yaw=0.02),
                    "kf_len": StaticState1DKF(length, q_val=1e-5, r_val=0.005),
                    "kf_wid": StaticState1DKF(width, q_val=1e-5, r_val=0.005),
                    "kf_hgt": StaticState1DKF(height, q_val=1e-5, r_val=0.005),
                }
                est_pos_w = np.array([px, py, pz])
                est_yaw_w = theta
                est_width = width
            else:
                kf["kf_x"].predict(dt)
                kf["kf_y"].predict(dt)
                kf["kf_z"].predict(dt)
                kf["kf_yaw"].predict(dt)
                kf["kf_len"].predict(dt)
                kf["kf_wid"].predict(dt)
                kf["kf_hgt"].predict(dt)
                
                u_x = kf["kf_x"].update(px)
                u_y = kf["kf_y"].update(py)
                u_z = kf["kf_z"].update(pz)
                
                if u_x and u_y and u_z:
                    kf["kf_yaw"].update(theta)
                    kf["kf_len"].update(length)
                    kf["kf_wid"].update(width)
                    kf["kf_hgt"].update(height)
                    
                est_pos_w = np.array([kf["kf_x"].x[0], kf["kf_y"].x[0], kf["kf_z"].x[0]])
                est_yaw_w = kf["kf_yaw"].get_yaw()
                est_width = kf["kf_wid"].x
        else:
            if kf is not None:
                kf["kf_x"].predict(dt, q_scale=10.0)
                kf["kf_y"].predict(dt, q_scale=10.0)
                kf["kf_z"].predict(dt, q_scale=10.0)
                kf["kf_yaw"].predict(dt, q_scale=10.0)
                kf["kf_len"].predict(dt, q_scale=10.0)
                kf["kf_wid"].predict(dt, q_scale=10.0)
                kf["kf_hgt"].predict(dt, q_scale=10.0)
                
                est_pos_w = np.array([kf["kf_x"].x[0], kf["kf_y"].x[0], kf["kf_z"].x[0]])
                est_yaw_w = kf["kf_yaw"].get_yaw()
                est_width = kf["kf_wid"].x
            else:
                est_pos_w = gt_pos_w_i
                est_yaw_w = gt_yaw_w
                est_width = gt_width
                
        # Transform to robot root
        est_pos_w_t = torch.tensor(est_pos_w, device=device, dtype=torch.float32)
        robot_pos_w = robot.data.root_link_pos_w[0]
        robot_quat_w = robot.data.root_link_quat_w[0]
        
        pos_rel = quat_apply(quat_inv(robot_quat_w.unsqueeze(0)), (est_pos_w_t - robot_pos_w).unsqueeze(0))[0]
        est_pos_r = pos_rel
        est_width_r = est_width
        
        est_quat_w_t = torch.tensor([math.cos(est_yaw_w/2), 0.0, 0.0, math.sin(est_yaw_w/2)], device=device, dtype=torch.float32)
        quat_rel = quat_mul(quat_inv(robot_quat_w.unsqueeze(0)), est_quat_w_t.unsqueeze(0))[0]
        _, _, yaw_rel = euler_xyz_from_quat(quat_rel.unsqueeze(0))
        yaw_rel = yaw_rel[0].item()
        est_yaw_r = [math.cos(yaw_rel), math.sin(yaw_rel)]
        
        print(f"Est World Pos: [{est_pos_w[0]:.4f}, {est_pos_w[1]:.4f}, {est_pos_w[2]:.4f}] (Error: {np.linalg.norm(est_pos_w - gt_pos_w_i):.4f}m)")
        print(f"Est Robot Pos: [{est_pos_r[0].item():.4f}, {est_pos_r[1].item():.4f}, {est_pos_r[2].item():.4f}]")
        print(f"Est Robot Yaw: [{est_yaw_r[0]:.4f}, {est_yaw_r[1]:.4f}]")
        print(f"Est Width:     {est_width_r:.4f}")
        
        # Apply override to observation buffer
        current_obs = TensorDict(env.obs_buf, batch_size=[1])
        current_obs["actor"][:, 22:25] = est_pos_r.unsqueeze(0)
        current_obs["actor"][:, 25:26] = torch.tensor([[est_width_r]], device=device, dtype=torch.float32)
        current_obs["actor"][:, 26:28] = torch.tensor([est_yaw_r], device=device, dtype=torch.float32)
        
        if "critic" in current_obs.keys():
            current_obs["critic"][:, 22:25] = est_pos_r.unsqueeze(0)
            current_obs["critic"][:, 25:29] = quat_rel.unsqueeze(0)
            current_obs["critic"][:, 29:30] = torch.tensor([[est_width_r]], device=device, dtype=torch.float32)
            current_obs["critic"][:, 30:32] = torch.tensor([est_yaw_r], device=device, dtype=torch.float32)
            
        with torch.no_grad():
            action = model(current_obs)
            
        _, _, terminated, truncated, _ = env.step(action)
        
        if terminated[0] or truncated[0]:
            print("\nEpisode Terminated/Truncated early!\n")
            break
            
    env.close()

if __name__ == "__main__":
    main()
