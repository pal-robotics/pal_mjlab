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
from ultralytics import YOLO
import sys
import os

# Import load_class helper
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate_policy import load_class

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    
    # Load YOLO
    yolo_path = "/home/lorenzobarbieri/exchange/tiago_pro_sim_ws/runs/detect/tiago_single_class_yolo26/weights/best.pt"
    yolo_model = YOLO(yolo_path)
    
    # Load Env
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
    env.reset()
    
    robot = env.scene["robot"]
    box = env.scene["box"]
    
    # Capture camera data
    with torch.no_grad():
        rgb = camera_rgb(env, "head_realsense_camera")
        depth = camera_depth(env, "head_realsense_camera", cutoff_distance=1.5)
        
        cam_idx = env.sim.mj_model.camera("robot/head_realsense_camera").id
        cam_pos = env.sim.data.cam_xpos[:, cam_idx]
        cam_xmat = env.sim.data.cam_xmat[:, cam_idx]
        cam_fovy = env.sim.mj_model.cam_fovy[cam_idx].item()
        
        gt_pos_w = box.data.root_link_pos_w[0]
        
    H, W = 240, 320
    fovy_rad = math.radians(cam_fovy)
    fy = (H / 2.0) / math.tan(fovy_rad / 2.0)
    fx = fy
    cx = W / 2.0
    cy = H / 2.0
    
    rgb_cpu = (rgb.cpu().permute(0, 2, 3, 1) * 255.0).clip(0, 255).to(torch.uint8).numpy()[0]
    depth_cpu = depth.squeeze(1).cpu().numpy()[0]
    
    # Project GT
    cam_pos_i = cam_pos[0].cpu().numpy()
    cam_xmat_i = cam_xmat[0].cpu().numpy()
    gt_pos_w_i = gt_pos_w.cpu().numpy()
    
    p_local = np.dot(gt_pos_w_i - cam_pos_i, cam_xmat_i)
    d_gt = -p_local[2]
    u_gt = cx + p_local[0] * fx / d_gt
    v_gt = cy - p_local[1] * fy / d_gt
    
    print(f"GT World Pos:  {gt_pos_w_i}")
    print(f"Cam Pos:       {cam_pos_i}")
    print(f"GT Cam Local:  {p_local}")
    print(f"GT Projected:  Pixel = ({u_gt:.2f}, {v_gt:.2f}), Depth = {d_gt:.4f}m")
    
    # Draw on image
    img_draw = rgb_cpu.copy()
    
    # Draw GT projected dot
    if 0 <= int(u_gt) < W and 0 <= int(v_gt) < H:
        cv2.circle(img_draw, (int(u_gt), int(v_gt)), 4, (0, 255, 0), -1) # Green dot
        
        # Print a 10x10 depth grid around the GT pixel
        u_start = max(0, int(u_gt) - 5)
        u_end = min(W, int(u_gt) + 5)
        v_start = max(0, int(v_gt) - 5)
        v_end = min(H, int(v_gt) + 5)
        print("\n--- 10x10 Depth Grid around GT pixel ---")
        for v in range(v_start, v_end):
            row_str = " ".join(f"{depth_cpu[v, u]:.4f}" for u in range(u_start, u_end))
            print(f"v={v:3d}: {row_str}")
        print("----------------------------------------\n")
    
    # Run YOLO
    yolo_results = yolo_model([rgb_cpu], verbose=False)
    for box_det in yolo_results[0].boxes:
        conf = float(box_det.conf[0])
        x1, y1, x2, y2 = map(int, box_det.xyxy[0])
        print(f"Detected Box: ({x1}, {y1}) to ({x2}, {y2}), Conf: {conf:.4f}")
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red rectangle
        cv2.putText(img_draw, f"cube {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
    # Save image
    out_dir = "/home/lorenzobarbieri/pal_mjlab_manipulation/pal_mjlab/scratch"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "debug_step_0.png")
    cv2.imwrite(out_path, cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))
    print(f"Saved debug image to: {out_path}")
    
    env.close()

if __name__ == "__main__":
    main()
