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

TASK_ID = "Mjlab-Manipulation-Lift-Cube-Pal-Tiago-Pro-v0"


def scan_box_mass(env_wrapped, num_episodes: int, device: str) -> None:
    """Reset the environment for `num_episodes` episodes, taking a single step in
    each, and record the min/max box mass encountered (e.g. from domain
    randomization applied on reset)."""
    inner_env = env_wrapped.unwrapped
    action_shape = env_wrapped.unwrapped.action_space.shape
    zero_action = torch.zeros(action_shape, device=device)

    min_mass = float("inf")
    max_mass = float("-inf")
    min_episode = -1
    max_episode = -1

    print(f"Scanning box mass over {num_episodes} episodes (1 step per episode)...")
    for episode in range(num_episodes):
        env_wrapped.reset()

        box_entity = inner_env.scene["box"]
        body_id = box_entity.indexing.body_ids[0]

        # Take a single step so the episode actually advances.
        env_wrapped.step(zero_action)

        mass = inner_env.sim.model.body_mass[0, body_id].item()

        if mass < min_mass:
            min_mass = mass
            min_episode = episode
        if mass > max_mass:
            max_mass = mass
            max_episode = episode

        if (episode + 1) % 50 == 0 or episode == num_episodes - 1:
            print(
                f"  Episode {episode + 1:4d}/{num_episodes} | mass = {mass:.5f} kg | "
                f"running min = {min_mass:.5f} kg | running max = {max_mass:.5f} kg"
            )

    print("\n" + "=" * 80)
    print(f"Mass scan complete over {num_episodes} episodes")
    print(f"  Min mass: {min_mass:.5f} kg (episode {min_episode})")
    print(f"  Max mass: {max_mass:.5f} kg (episode {max_episode})")
    print("=" * 80)


class PrintingPolicy:
    def __init__(self, action_shape, env_wrapped, model=None):
        self.action_shape = action_shape
        self.env_wrapped = env_wrapped
        self.inner_env = env_wrapped.unwrapped
        self.obs_manager = self.inner_env.observation_manager
        self.names = self.obs_manager.active_terms.get("actor", [])
        self.shapes = self.obs_manager.group_obs_term_dim.get("actor", [])
        self.model = model

    def __call__(self, obs) -> torch.Tensor:
        # Get current step and time
        step = self.inner_env.episode_length_buf[0].item()
        dt = self.inner_env.cfg.decimation * self.inner_env.cfg.sim.mujoco.timestep
        t = step * dt
        
        # Get actual box sizes (length, width, height), orientation (yaw), and mass
        box_entity = self.inner_env.scene["box"]
        geom_id = box_entity.indexing.geom_ids[0]
        box_half_sizes = self.inner_env.sim.model.geom_size[0, geom_id]
        box_full_sizes = box_half_sizes * 2.0

        # Get body id and mass (mass is stored per-body in MuJoCo, not per-geom)
        body_id = box_entity.indexing.body_ids[0]
        box_mass = self.inner_env.sim.model.body_mass[0, body_id]

        box_quat = box_entity.data.root_link_quat_w
        from mjlab.utils.lab_api.math import euler_xyz_from_quat
        _, _, box_yaw = euler_xyz_from_quat(box_quat)
        box_yaw_val = box_yaw[0].item()
        box_yaw_deg = math.degrees(box_yaw_val)
        
        print("\n" + "=" * 80)
        print(f"Step: {step:3d} | Time: {t:.2f}s")
        print(f"Object Length (X): {box_full_sizes[0].item():.4f} m | Width (Y): {box_full_sizes[1].item():.4f} m | Height (Z): {box_full_sizes[2].item():.4f} m")
        print(f"Object World Yaw:  {box_yaw_val:.4f} rad ({box_yaw_deg:.2f}°)")
        print(f"Object Mass: {box_mass.item():.4f} kg")
        
        # Read fingertip contact sensors (one per gripper finger) and compute contact metrics
        try:
            contact_sensor = self.inner_env.scene["box_fingertip_contact"]
        except (KeyError, AttributeError, TypeError):
            contact_sensor = None

        dist_both = False
        combined_contact = False
        try:
            robot_entity = self.inner_env.scene["robot"]

            from pal_mjlab.robots.pal_tiago_pro.tiago_pro import TiagoProRobot
            from pal_mjlab.tasks.manipulation.mdp.contact_sensor import site_contact_both_fingers
            
            robot_cfg = TiagoProRobot()
            site_ids, _ = robot_entity.find_sites([robot_cfg.fingertip_site_pattern], preserve_order=True)
            site_pos_w = robot_entity.data.site_pos_w[:, site_ids]
            obj_pos_w = box_entity.data.geom_pos_w[:, 0].unsqueeze(1)
            dist_to_obj = torch.norm(site_pos_w - obj_pos_w, dim=-1)
            dist_both = (dist_to_obj < 0.05).all(dim=-1)[0].item()

            from pal_mjlab.tasks.manipulation.mdp.observations import object_both__contact_fingers
            combined_contact_tensor = object_both__contact_fingers(
                env=self.inner_env,
                sensor_name="box_fingertip_contact",
                site_names=[robot_cfg.fingertip_site_pattern],
            )
            combined_contact = combined_contact_tensor[0, 0].item() > 0
        except Exception:
            pass

        if contact_sensor is not None and contact_sensor.data.found is not None:
            found = contact_sensor.data.found[0]  # shape: [N] (N=2 for two fingertips)
            finger_contacts = [f.item() > 0 for f in found]
            both_contacts_phys = all(finger_contacts)
            print(f"Finger Contacts (Physical): {finger_contacts} | Both Physical: {both_contacts_phys}")
            print(f"Distances to object: {dist_to_obj[0].tolist()}")
            print(f"Both Distance-based: {dist_both} | Both Combined (New Measure): {combined_contact}")
        else:
            print("Finger Contacts: Sensor not available")
        # Read reward terms
        reward_manager = self.inner_env.reward_manager
        if step > 0 and hasattr(reward_manager, "_step_reward") and reward_manager._step_reward is not None:
            step_rewards = reward_manager._step_reward[0]  # shape: [num_terms]
            total_step_reward = step_rewards.sum().item()
            total_scaled_reward = reward_manager._reward_buf[0].item()
            print("Reward values (weighted contribution of each term to the transition):")
            for term_name, val in zip(reward_manager.active_terms, step_rewards.tolist()):
                print(f"  {term_name:35s}: {val:10.4f}")
            print(f"  {'Total Step Reward (unscaled)':35s}: {total_step_reward:10.4f}")
            print(f"  {'Total Step Reward (scaled by dt)':35s}: {total_scaled_reward:10.4f}")
        else:
            print("Reward values: No transition yet (initial state)")

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
            
        if self.model is not None:
            from tensordict import TensorDict
            if not isinstance(obs, TensorDict):
                obs = TensorDict(obs, batch_size=[1])
            with torch.no_grad():
                action = self.model(obs)
            return action
        else:
            return torch.zeros(self.action_shape, device=self.inner_env.device)

def main():
    parser = argparse.ArgumentParser(description="Play environment with checkpoint model and print observations.")
    parser.add_argument("--viewer", type=str, choices=["auto", "native", "viser"], default="auto",
                        help="Viewer backend to use.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to the checkpoint model weights (e.g. .pt file). If None, runs the zero policy.")
    parser.add_argument("--scan-mass", action="store_true",
                        help="Instead of launching the viewer, reset the env once per episode, take a "
                             "single (zero-action) step, and record the min/max box mass seen over "
                             "--num-episodes episodes. Exits after printing the summary.")
    parser.add_argument("--num-episodes", type=int, default=500,
                        help="Number of episodes to run when --scan-mass is set (default: 500).")
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

    # If requested, run a headless mass scan instead of the interactive viewer.
    if args.scan_mass:
        scan_box_mass(env_wrapped, args.num_episodes, device)
        env_wrapped.close()
        return

    # Load checkpoint model if provided
    model = None
    if args.checkpoint is not None:
        from tensordict import TensorDict
        
        print("Setting up policy model...")
        actor_cfg = agent_cfg.actor
        model_cls = load_class(actor_cfg.class_name)
        
        # Initialize with dummy observations to build model
        obs_dict, _ = env.reset()
        dummy_obs = TensorDict(obs_dict, batch_size=[1])
        
        model = model_cls(
            obs=dummy_obs,
            obs_groups=getattr(agent_cfg, "obs_groups", None),
            obs_set="actor",
            output_dim=env.action_manager.total_action_dim,
            hidden_dims=actor_cfg.hidden_dims,
            activation=actor_cfg.activation,
            obs_normalization=actor_cfg.obs_normalization,
            distribution_cfg=actor_cfg.distribution_cfg,
        ).to(device)
        
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint path '{checkpoint_path}' does not exist!")
            
        print(f"Loading model weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        model.eval()
        print("Model loaded successfully!")

    action_shape = env_wrapped.unwrapped.action_space.shape
    policy = PrintingPolicy(action_shape, env_wrapped, model=model)

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