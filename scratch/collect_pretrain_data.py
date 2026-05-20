import os
import json
import torch
import numpy as np
from mjlab.envs import ManagerBasedRlEnv
from pal_mjlab.tasks.manipulation.tiago_pro.env_cfgs import lift_vision_env_cfg, _BOX_HALF_SIZE

# --- Expert Actor Definition (to drive collection) ---

class OracleExpert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(35, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 7)
        )

    def forward(self, obs_dict):
        # The oracle expects the 35D state: 
        # joint_pos(7) + joint_vel(7) + actions(7) + object_pos(3) + object_ori(4) + target_pos(3) + gripper_pos(1) + ee_pos(3)
        # This matches exactly the first 35 features of the 'critic' observation group.
        critic_obs = obs_dict["critic"]
        oracle_obs = critic_obs[:, :-2] # Drop the last 2 features (finger_contact)
        return self.mlp(oracle_obs)

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
    local_corners = torch.tensor([[hx, hy, hz], [hx, -hy, hz], [-hx, hy, hz], [-hx, -hy, hz]], device=env.device)
    box = env.scene["box"]
    # Using generic access to position/orientation to be safe across mjlab versions
    box_pos = box.data.root_pos_w if hasattr(box.data, "root_pos_w") else box.data.geom_pos_w[:, 0]
    box_quat = box.data.root_quat_w if hasattr(box.data, "root_quat_w") else box.data.geom_quat_w[:, 0]
    
    num_envs = env.num_envs
    box_pos_exp = box_pos.unsqueeze(1).expand(-1, 4, -1)
    box_quat_exp = box_quat.unsqueeze(1).expand(-1, 4, -1)
    local_corners_exp = local_corners.unsqueeze(0).expand(num_envs, -1, -1)
    from mjlab.utils.lab_api.math import quat_apply
    corners_3d_w = box_pos_exp + quat_apply(box_quat_exp, local_corners_exp)
    robot = env.scene["robot"]
    fingertip_site_names = [s for s in robot.site_names if "fingertip" in s]
    fingertip_pos_w = robot.data.site_pos_w[:, [robot.site_names.index(name) for name in fingertip_site_names]]
    return torch.cat([corners_3d_w, fingertip_pos_w], dim=1)

# --- Collection Script ---

def collect_data(num_samples=10000, save_dir="dataset"):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = os.path.expanduser("~/Downloads/model_14200.pt")

    # 1. Instantiate environment
    print("Initializing environment...")
    cfg = lift_vision_env_cfg(cam_type="depth")
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda")
    
    # 2. Load Expert
    print(f"Loading Expert Policy from {checkpoint_path}...")
    expert = OracleExpert().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    sd = checkpoint["actor_state_dict"]
    expert.load_state_dict(sd, strict=False)
    expert.eval()

    # 3. Camera Params
    camera_name = "head_realsense_camera"
    width, height = cfg.scene.sensors[-1].width, cfg.scene.sensors[-1].height
    fov_y = 60.0
    f = (height / 2.0) / np.tan(np.deg2rad(fov_y / 2.0))
    K = (f, f, width / 2.0, height / 2.0)

    dataset_labels = []
    obs, _ = env.reset()
    
    print(f"Starting Expert-Guided Collection for {num_samples} samples...")
    while len(dataset_labels) < num_samples:
        with torch.no_grad():
            camera = env.scene.sensors[camera_name]
            expert_obs = {
                "critic": obs["critic"],
                "camera": torch.clamp(camera.data.depth.permute(0, 3, 1, 2), 0, 1.5)
            }
            actions = expert(expert_obs)
            
        obs, _, _, _, _ = env.step(actions)
        
        # Save image and keypoints
        depth_image = camera.data.depth[0].cpu().numpy()
        keypoints_3d = get_3d_keypoints(env)
        
        from mjlab.utils.lab_api.math import quat_from_matrix
        sim_data = env.sim.data
        cam_pos = sim_data.cam_xpos[:, camera.camera_idx]
        cam_quat = quat_from_matrix(sim_data.cam_xmat[:, camera.camera_idx])
        
        keypoints_2d = project_3d_to_2d(keypoints_3d, cam_pos, cam_quat, K, width, height)
        kps_raw = keypoints_2d[0].cpu().numpy()
        
        # Calculate expected local Z depths to check for occlusion
        from mjlab.utils.lab_api.math import quat_apply, quat_inv
        B, N, _ = keypoints_3d.shape
        cam_pos_exp = cam_pos.unsqueeze(1).expand(B, N, 3)
        cam_quat_exp = cam_quat.unsqueeze(1).expand(B, N, 4)
        points_c = quat_apply(quat_inv(cam_quat_exp), keypoints_3d - cam_pos_exp)
        z_expected = -points_c[0, :, 2].cpu().numpy()
        
        # Calculate visibility mask based on depth buffer comparison
        visibility_list = []
        for idx, (u, v) in enumerate(kps_raw):
            col = int(np.clip(u, 0, width - 1))
            row = int(np.clip(v, 0, height - 1))
            observed_z = depth_image[row, col, 0]
            expected_z = z_expected[idx]
            # Keypoint is visible if observed surface is not significantly closer than expected depth
            is_visible = bool(observed_z >= expected_z - 0.02)
            visibility_list.append(is_visible)
        
        # Visibility check (ensure keypoints are at least on-screen or within bounds)
        valid = True
        for u, v in kps_raw:
            if u < -10 or u > 138 or v < -10 or v > 138:
                valid = False
                break
        
        if valid:
            filename = f"depth_{len(dataset_labels):05d}.npy"
            np.save(os.path.join(save_dir, filename), depth_image)
            dataset_labels.append({
                "depth": filename,
                "keypoints": kps_raw.tolist(),
                "visibility": visibility_list
            })
            if len(dataset_labels) % 100 == 0:
                print(f"Collected {len(dataset_labels)}/{num_samples}...")

        if len(dataset_labels) % 250 == 0:
            env.reset()

    with open(os.path.join(save_dir, "labels.json"), "w") as f:
        json.dump(dataset_labels, f, indent=4)
    print("Collection complete.")

if __name__ == "__main__":
    collect_data()