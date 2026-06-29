import torch
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.manipulation.mdp import camera_rgb, camera_depth

TASK_ID = "Mjlab-Manipulation-Lift-Cube-Pal-Tiago-Pro-v0"

def main():
    device = "cpu"
    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = 1
    
    # Manually add the camera sensor to the scene config!
    env_cfg.scene.sensors = (env_cfg.scene.sensors or ()) + (
        CameraSensorCfg(
            name="head_realsense_camera",
            height=480,
            width=640,
            data_types=("rgb", "depth"),
            camera_name="robot/head_realsense_camera",
        ),
    )
    
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env.reset()
    
    # Try to extract camera images
    try:
        rgb = camera_rgb(env, "head_realsense_camera")
        print("RGB shape:", rgb.shape)
        print("RGB dtype:", rgb.dtype)
        print("RGB min, max:", rgb.min().item(), rgb.max().item())
    except Exception as e:
        print("Failed to get RGB:", e)
        
    try:
        depth = camera_depth(env, "head_realsense_camera", cutoff_distance=1.5)
        print("Depth shape:", depth.shape)
        print("Depth dtype:", depth.dtype)
        print("Depth min, max:", depth.min().item(), depth.max().item())
    except Exception as e:
        print("Failed to get Depth:", e)

    env.close()

if __name__ == "__main__":
    main()
