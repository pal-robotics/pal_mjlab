import os
import sys
import time
import csv
import argparse
import numpy as np
import pandas as pd
import torch
import onnxruntime as ort
from ultralytics import YOLO

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

import mjlab.tasks  # noqa: F401 – populates task registry
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.sensor import CameraSensorCfg

def check_in_frustum(box_pos, cam_pos, cam_xmat, cam_fovy, aspect_ratio=1.333):
    # box_pos shape: (num_envs, 3)
    # cam_pos shape: (num_envs, 3)
    # cam_xmat shape: (num_envs, 3, 3) or (num_envs, 9)
    if cam_xmat.ndim == 2:
        cam_xmat = cam_xmat.view(-1, 3, 3)
    diff = box_pos - cam_pos  # (num_envs, 3)
    # Project to camera coordinate frame
    points_c = torch.bmm(diff.unsqueeze(1), cam_xmat).squeeze(1)  # (num_envs, 3)
    
    x_c = points_c[:, 0]
    y_c = points_c[:, 1]
    z_c = points_c[:, 2]
    
    z_depth = -z_c
    
    fovy_rad = cam_fovy * (np.pi / 180.0)
    fovx_rad = 2 * np.arctan(np.tan(fovy_rad / 2.0) * aspect_ratio)
    
    in_front = z_depth > 0.1
    in_fov_y = torch.abs(y_c) < z_depth * np.tan(fovy_rad / 2.0)
    in_fov_x = torch.abs(x_c) < z_depth * np.tan(fovx_rad / 2.0)
    
    return in_front & in_fov_y & in_fov_x

def main():
    parser = argparse.ArgumentParser(description="Collect joint configuration data under occlusion.")
    parser.add_argument("--episodes", type=int, default=50000, help="Number of episodes to run.")
    parser.add_argument("--envs", type=int, default=512, help="Number of parallel environments.")
    args = parser.parse_args()

    onnx_path = "/home/lorenzobarbieri/pal_mjlab_manipulation/pal_mjlab/2026-06-19_14-20-32-checkpoints/2026-06-19_14-20-32.onnx"
    yolo_path = "/home/lorenzobarbieri/exchange/tiago_pro_sim_ws/runs/detect/tiago_single_class_yolo26/weights/best.pt"
    out_csv = "/home/lorenzobarbieri/pal_mjlab_manipulation/pal_mjlab/scratch/occlusion_joint_data.csv"
    
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    
    print("=" * 80)
    print("  CAMERA OCCLUSION DATA COLLECTION SCRIPT")
    print("=" * 80)
    print(f"Loading YOLO from: {yolo_path}", flush=True)
    yolo = YOLO(yolo_path)
    
    print(f"Loading ONNX policy from: {onnx_path}")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
    ort_session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = ort_session.get_inputs()[0].name
    
    print("Loading Env Config...")
    task_id = "Mjlab-Manipulation-Lift-Cube-Pal-Tiago-Pro-v0"
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = args.envs
    
    # Add a camera sensor for the head camera
    from pal_mjlab.robots.pal_tiago_pro.tiago_pro import TiagoProRobot
    robot_cfg = TiagoProRobot()
    head_cam_name = f"robot/{robot_cfg.head_camera_name}"
    
    env_cfg.scene.sensors = (env_cfg.scene.sensors or ()) + (
        CameraSensorCfg(
            name="head_realsense_camera",
            height=256,
            width=320,
            data_types=("rgb",),
            camera_name=head_cam_name,
        ),
    )
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Creating {args.envs} environments on {device}...", flush=True)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    
    robot = env.scene["robot"]
    box = env.scene["box"]
    joint_names = list(robot.joint_names)
    
    # Open CSV writer
    csv_file = open(out_csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(joint_names + ["occluded"])
    
    obs_dict, _ = env.reset()
    
    total_episodes = args.episodes
    episodes_completed = 0
    step_count = 0
    
    episodes_per_env = np.zeros(args.envs, dtype=int)
    buffer = []
    
    start_time = time.time()
    
    print(f"Starting simulation run for {total_episodes} total episodes...", flush=True)
    try:
        while episodes_completed < total_episodes:
            # Extract actor observations
            actor_obs = obs_dict["actor"]
            obs_np = actor_obs.cpu().numpy().astype(np.float32)
            
            # Policy inference (loop because batch dimension is fixed to 1 in ONNX)
            actions_list = []
            for i in range(args.envs):
                obs_np_i = obs_np[i:i+1]
                act_i = ort_session.run(None, {input_name: obs_np_i})[0]
                actions_list.append(act_i)
            actions_np = np.concatenate(actions_list, axis=0)
            actions = torch.from_numpy(actions_np).to(device)
            
            # Step environment
            obs_dict, _, terminated, truncated, _ = env.step(actions)
            
            if step_count % 10 == 0:
                # Render camera and run YOLO
                camera_sensor = env.scene.sensors["head_realsense_camera"]
                rgb_tensor = camera_sensor.data.rgb
                
                # Frustum check
                box_pos = box.data.root_pos_w if hasattr(box.data, "root_pos_w") else box.data.geom_pos_w[:, 0]
                cam_idx = env.sim.mj_model.camera(head_cam_name).id
                cam_pos = env.sim.data.cam_xpos[:, cam_idx]
                cam_xmat = env.sim.data.cam_xmat[:, cam_idx]
                cam_fovy = env.sim.mj_model.cam_fovy[cam_idx]
                
                in_frustum = check_in_frustum(box_pos, cam_pos, cam_xmat, cam_fovy)
                
                # YOLO Batch Inference
                rgb_chw = rgb_tensor.permute(0, 3, 1, 2).float() / 255.0
                yolo_results = yolo(rgb_chw, verbose=False, imgsz=320, device=device)
                
                # Robot joint positions
                joint_pos = robot.data.joint_pos.cpu().numpy()
                
                # Check detection for each environment
                for i in range(args.envs):
                    # We only care about steps where the object is geometrically in the field of view
                    if in_frustum[i].item():
                        res = yolo_results[i]
                        detected = False
                        for box_det in res.boxes:
                            if int(box_det.cls[0].item()) == 0 and float(box_det.conf[0].item()) > 0.4:
                                detected = True
                                break
                        
                        occluded_val = 1 if not detected else 0
                        row = list(joint_pos[i]) + [occluded_val]
                        buffer.append(row)
                
                # Write to file periodically
                if len(buffer) >= 1:
                    csv_writer.writerows(buffer)
                    csv_file.flush()
                    buffer.clear()
            
            # Check terminated/truncated episodes
            dones = terminated | truncated
            for i in range(args.envs):
                if dones[i].item():
                    episodes_per_env[i] += 1
                    episodes_completed += 1
                    if episodes_completed % 1 == 0:
                        elapsed = time.time() - start_time
                        fps = (step_count * args.envs) / elapsed if elapsed > 0 else 0
                        print(f"Progress: {episodes_completed}/{total_episodes} episodes completed... Speed: {fps:.1f} FPS", flush=True)
                    if episodes_completed >= total_episodes:
                        break
            
            step_count += 1
            
    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving gathered data and running analysis...")
    
    # Save remaining data
    if buffer:
        csv_writer.writerows(buffer)
    csv_file.close()
    
    print("\nData collection finished.")
    analyze_occlusions(out_csv, joint_names)
    env.close()

def analyze_occlusions(csv_path, joint_names):
    print("\n" + "=" * 80)
    print("  OCCLUSION ANALYSIS REPORT")
    print("=" * 80)
    
    if not os.path.exists(csv_path) or os.path.getsize(csv_path) < 100:
        print("Error: No data was recorded.")
        return
        
    df = pd.read_csv(csv_path)
    total_steps = len(df)
    occluded_steps = df["occluded"].sum()
    
    if total_steps == 0:
        print("No steps inside the frustum were recorded.")
        return
        
    print(f"Total steps inside camera frustum: {total_steps}")
    print(f"Occluded steps (not detected by YOLO): {occluded_steps} ({occluded_steps/total_steps*100:.2f}%)")
    
    if occluded_steps == 0:
        print("No occlusions were detected! The object was visible in all frustum-aligned steps.")
        return

    # Focus on arm_right joints and torso joints (these are the active ones that block the view)
    active_joints = [j for j in joint_names if "right" in j or "torso" in j]
    
    # Calculate statistics
    visible_df = df[df["occluded"] == 0]
    occluded_df = df[df["occluded"] == 1]
    
    rows = []
    for col in active_joints:
        v_mean = visible_df[col].mean()
        v_std = visible_df[col].std()
        o_mean = occluded_df[col].mean()
        o_std = occluded_df[col].std()
        
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((visible_df[col].var() + occluded_df[col].var()) / 2.0)
        d = (o_mean - v_mean) / pooled_std if pooled_std > 0 else 0.0
        
        # Compute correlation coefficient
        corr = df[col].corr(df["occluded"])
        
        rows.append({
            "Joint": col,
            "Visible Mean": v_mean,
            "Occluded Mean": o_mean,
            "Mean Diff": o_mean - v_mean,
            "Correlation": corr,
            "Cohen's d": d
        })
        
    stats_df = pd.DataFrame(rows)
    stats_df["Abs Corr"] = stats_df["Correlation"].abs()
    stats_df = stats_df.sort_values(by="Abs Corr", ascending=False)
    
    print("\nJoint Correlation with Occlusion:")
    print(stats_df[["Joint", "Visible Mean", "Occluded Mean", "Mean Diff", "Correlation", "Cohen's d"]].to_string(index=False))
    
    # Save statistics table
    stats_csv = os.path.join(os.path.dirname(csv_path), "occlusion_analysis_stats.csv")
    stats_df.to_csv(stats_csv, index=False)
    print(f"\nDetailed analysis saved to: {stats_csv}")
    
    # Generate recommendations for joint limit penalties
    print("\n" + "-" * 80)
    print("  JOINT PENALIZATION RECOMMENDATIONS")
    print("-" * 80)
    print("Based on the correlations, here are the joints most responsible for occlusion:")
    
    top_joints = stats_df[stats_df["Abs Corr"] > 0.1]
    if len(top_joints) == 0:
        print("No joints showed a strong correlation (> 0.1) with camera occlusion.")
    else:
        for idx, row in top_joints.iterrows():
            joint = row["Joint"]
            corr = row["Correlation"]
            v_mean = row["Visible Mean"]
            o_mean = row["Occluded Mean"]
            
            direction = "higher" if corr > 0 else "lower"
            print(f"- {joint}: Correlation is {corr:.3f}.")
            print(f"  When occluded, the joint tends to have {direction} values (Mean: {o_mean:.3f} rad) compared to when visible (Mean: {v_mean:.3f} rad).")
            print(f"  Recommendation: Add a penalty if {joint} {' > ' if corr > 0 else ' < '} {o_mean:.3f} rad.")
            
    print("=" * 80)

if __name__ == "__main__":
    main()
