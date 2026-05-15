import os
import torch
import numpy as np
from PIL import Image
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from pal_mjlab.tasks.manipulation.tiago_pro.env_cfgs import lift_vision_env_cfg
import mujoco

# Configuration
NUM_SAMPLES = 10
OUTPUT_FILE = "scratch/pretrain_data.pt"
DEVICE = "cuda"

# Load environment manually to ensure local code is used
env_cfg = lift_vision_env_cfg("depth", play=True)
env_cfg.scene.num_envs = 1
env_cfg.scene.num_envs = 1
env_cfg.scene.entities["robot"].init_state.pos = (-1.0, 0.0, 0.0)
env = ManagerBasedRlEnv(env_cfg, device=DEVICE)

def project_3d_to_2d(pos_3d, mj_model, mj_data, cam_id, width, height):
    cam_pos = mj_data.cam_xpos[cam_id]
    cam_mat = mj_data.cam_xmat[cam_id].reshape(3, 3)
    fovy = mj_model.cam_fovy[cam_id]
    f = 0.5 * height / np.tan(np.deg2rad(fovy) * 0.5)
    rel_pos = pos_3d - cam_pos
    pos_cam = cam_mat.T @ rel_pos
    if pos_cam[2] >= 0:
        return None
    px = f * (pos_cam[0] / -pos_cam[2]) + width / 2
    py = -f * (pos_cam[1] / -pos_cam[2]) + height / 2
    nx = (px / width) * 2.0 - 1.0
    ny = (py / height) * 2.0 - 1.0
    return np.array([ny, nx])

images = []
keypoints = []

print(f"Collecting {NUM_SAMPLES} samples...")

for i in range(NUM_SAMPLES):
    # Reset
    obs_dict, _ = env.reset()
    
    # MANUALLY RESAMPLE COMMAND
    env.command_manager.get_term("lift_height")._resample_command(torch.tensor([0], device=DEVICE))
    
    # Step a bit
    for _ in range(np.random.randint(5, 15)):
        random_action = torch.rand(1, env.action_manager.total_action_dim, device=DEVICE) * 2 - 1
        env.step(random_action)
    
    env.sim.forward()

    img = env.obs_buf["camera"].clone()
    
    obj_id = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, "box/box_object")
    box_pos = env.sim.mj_data.xpos[obj_id]
    box_mat = env.sim.mj_data.xmat[obj_id].reshape(3, 3)
    
    hx, hy, hz = 0.025, 0.025, 0.05
    offsets = np.array([
        [hx, hy, hz], [hx, hy, -hz], [hx, -hy, hz], [hx, -hy, -hz],
        [-hx, hy, hz], [-hx, hy, -hz], [-hx, -hy, hz], [-hx, -hy, -hz]
    ])
    
    cam_name = "robot/head_realsense_camera"
    cam_id = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    
    current_kps = []
    visible = True
    
    for offset in offsets:
        corner_pos = box_pos + box_mat @ offset
        kp = project_3d_to_2d(corner_pos, env.sim.mj_model, env.sim.mj_data, cam_id, 128, 128)
        if kp is None:
            print(f"Sample {i}: Corner {offset} behind camera")
            visible = False
            break
        if np.any(np.abs(kp) > 1.0):
            print(f"Sample {i}: Corner {offset} outside FOV: {kp}")
            visible = False
            break
        current_kps.append(kp)
        
    if not visible:
        continue
        
    ee_id = mujoco.mj_name2id(env.sim.mj_model, mujoco.mjtObj.mjOBJ_SITE, "robot/gripper_right_grasping_site")
    ee_pos = env.sim.mj_data.site_xpos[ee_id]
    kp_ee = project_3d_to_2d(ee_pos, env.sim.mj_model, env.sim.mj_data, cam_id, 128, 128)
    if kp_ee is None:
        print(f"Sample {i}: EE behind camera")
        visible = False
    elif np.any(np.abs(kp_ee) > 1.0):
        print(f"Sample {i}: EE outside FOV: {kp_ee}")
        visible = False
    else:
        current_kps.append(kp_ee)
        
    if visible:
        images.append(img)
        keypoints.append(torch.tensor(np.array(current_kps), dtype=torch.float32))
    
    if (i + 1) % 100 == 0:
        print(f"Processed {i+1}/{NUM_SAMPLES} samples. Valid: {len(images)}")

if len(images) > 0:
    torch.save({
        "images": torch.cat(images, dim=0),
        "keypoints": torch.stack(keypoints, dim=0)
    }, OUTPUT_FILE)
    print(f"Saved {len(images)} valid samples to {OUTPUT_FILE}")
else:
    print("Error: No valid samples collected. Images list is empty.")
env.close()
