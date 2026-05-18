import os
import json
import torch
import numpy as np
from mjlab.envs import ManagerBasedRlEnv
from pal_mjlab.tasks.manipulation.tiago_pro.env_cfgs import lift_vision_env_cfg, _BOX_HALF_SIZE

# --- Expert Actor Definition (to drive collection) ---

class SpatialSoftmax(torch.nn.Module):
    def __init__(self, height, width, temperature=1.0):
        super().__init__()
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height),
            torch.linspace(-1.0, 1.0, width),
            indexing="ij",
        )
        self.register_buffer("pos_x", pos_x.reshape(1, 1, -1))
        self.register_buffer("pos_y", pos_y.reshape(1, 1, -1))
        self.temperature = temperature

    def forward(self, x):
        B, C, H, W = x.shape
        features = x.reshape(B, C, -1)
        weights = torch.softmax(features / self.temperature, dim=-1)
        expected_x = (weights * self.pos_x).sum(dim=-1)
        expected_y = (weights * self.pos_y).sum(dim=-1)
        return torch.stack([expected_x, expected_y], dim=-1).reshape(B, C * 2)

class ExpertActor(torch.nn.Module):
    def __init__(self, num_keypoints=6):
        super().__init__()
        self.cnns = torch.nn.ModuleDict({
            "camera": torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
                torch.nn.ELU(),
                torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                torch.nn.ELU(),
                torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                torch.nn.ELU(),
                torch.nn.Conv2d(64, num_keypoints, kernel_size=1, stride=1, padding=0),
                torch.nn.ELU(),
                SpatialSoftmax(32, 32)
            )
        })
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(28 + (num_keypoints * 2), 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, 7)
        )

    def forward(self, obs_dict):
        kps = self.cnns["camera"](obs_dict["camera"])
        combined = torch.cat([obs_dict["actor"], kps], dim=-1)
        return self.mlp(combined)

# --- Projection Utilities ---

def project_3d_to_2d(points_3d_w, cam_pos, cam_quat, K_matrix, width, height):
    from mjlab.utils.lab_api.math import quat_apply, quat_inv
    cam_pos_exp = cam_pos.unsqueeze(1)
    cam_quat_exp = cam_quat.unsqueeze(1)
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
    checkpoint_path = "logs/rsl_rl/lift_depth/2026-05-18_10-18-26/model_1000.pt"

    # 1. Instantiate environment
    print("Initializing environment...")
    cfg = lift_vision_env_cfg(cam_type="depth")
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda")
    
    # 2. Load Expert
    print(f"Loading Expert Policy from {checkpoint_path}...")
    expert = ExpertActor(num_keypoints=6).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    sd = checkpoint["actor_state_dict"]
    if "cnns.camera.spatial_softmax.pos_x" in sd:
        sd["cnns.camera.spatial_softmax.pos_x"] = sd["cnns.camera.spatial_softmax.pos_x"].view(32, 32)
        sd["cnns.camera.spatial_softmax.pos_y"] = sd["cnns.camera.spatial_softmax.pos_y"].view(32, 32)
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
                "actor": obs["actor"],
                "camera": torch.clamp(camera.data.depth.permute(0, 3, 1, 2), 0, 1.0)
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
        
        # Visibility check
        valid = True
        for u, v in kps_raw:
            if u < -10 or u > 138 or v < -10 or v > 138:
                valid = False
                break
        
        if valid:
            filename = f"depth_{len(dataset_labels):05d}.npy"
            np.save(os.path.join(save_dir, filename), depth_image)
            dataset_labels.append({"depth": filename, "keypoints": kps_raw.tolist()})
            if len(dataset_labels) % 100 == 0:
                print(f"Collected {len(dataset_labels)}/{num_samples}...")

        if len(dataset_labels) % 500 == 0:
            env.reset()

    with open(os.path.join(save_dir, "labels.json"), "w") as f:
        json.dump(dataset_labels, f, indent=4)
    print("Collection complete.")

if __name__ == "__main__":
    collect_data()