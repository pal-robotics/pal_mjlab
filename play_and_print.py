import sys
import os
import math
import torch
import argparse
import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

TASK_ID = "Mjlab-Manipulation-Lift-Cube-Pal-Tiago-Pro-v0"

class PrintingPolicy:
    def __init__(self, action_shape, env_wrapped):
        self.action_shape = action_shape
        self.env_wrapped = env_wrapped
        self.inner_env = env_wrapped.unwrapped
        self.obs_manager = self.inner_env.observation_manager
        self.names = self.obs_manager.active_terms.get("actor", [])
        self.shapes = self.obs_manager.group_obs_term_dim.get("actor", [])

    def __call__(self, obs) -> torch.Tensor:
        # Get current step and time
        step = self.inner_env.episode_length_buf[0].item()
        dt = self.inner_env.cfg.decimation * self.inner_env.cfg.sim.mujoco.timestep
        t = step * dt
        
        # Get actual box sizes (length, width, height) and orientation (yaw)
        box_entity = self.inner_env.scene["box"]
        geom_id = box_entity.indexing.geom_ids[0]
        box_half_sizes = self.inner_env.sim.model.geom_size[0, geom_id]
        box_full_sizes = box_half_sizes * 2.0
        
        box_quat = box_entity.data.root_link_quat_w
        from mjlab.utils.lab_api.math import euler_xyz_from_quat
        _, _, box_yaw = euler_xyz_from_quat(box_quat)
        box_yaw_val = box_yaw[0].item()
        box_yaw_deg = math.degrees(box_yaw_val)
        
        print("\n" + "=" * 80)
        print(f"Step: {step:3d} | Time: {t:.2f}s")
        print(f"Object Length (X): {box_full_sizes[0].item():.4f} m | Width (Y): {box_full_sizes[1].item():.4f} m | Height (Z): {box_full_sizes[2].item():.4f} m")
        print(f"Object World Yaw:  {box_yaw_val:.4f} rad ({box_yaw_deg:.2f}°)")
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
        for name, shape in zip(self.names, self.shapes):
            dim = math.prod(shape)
            vals = actor_obs[cursor : cursor + dim].tolist()
            formatted_vals = ", ".join([f"{v:.4f}" for v in vals])
            if name == "object_yaw" and len(vals) == 2:
                obs_yaw_rad = math.atan2(vals[1], vals[0])
                obs_yaw_deg = math.degrees(obs_yaw_rad)
                print(f"  {name:25s} shape={str(shape):8s} value=[{formatted_vals}] (yaw: {obs_yaw_rad:.4f} rad, {obs_yaw_deg:.2f}°)")
            else:
                print(f"  {name:25s} shape={str(shape):8s} value=[{formatted_vals}]")
            cursor += dim
            
        return torch.zeros(self.action_shape, device=self.inner_env.device)

def main():
    parser = argparse.ArgumentParser(description="Play environment with zero agent and print observations.")
    parser.add_argument("--viewer", type=str, choices=["auto", "native", "viser"], default="auto",
                        help="Viewer backend to use.")
    args = parser.parse_args()

    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load configurations
    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = 1
    agent_cfg = load_rl_cfg(TASK_ID)

    # Initialize environment
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    action_shape = env_wrapped.unwrapped.action_space.shape
    policy = PrintingPolicy(action_shape, env_wrapped)

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
