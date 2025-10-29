"""Script to play RL agent with RSL-RL."""

from dataclasses import asdict
from pathlib import Path
from typing import Literal, cast

import gymnasium as gym
import torch
import tyro
from rsl_rl.runners import OnPolicyRunner
from typing_extensions import assert_never

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
    load_cfg_from_registry,
)
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserViewer

from pal_mjlab.tasks.kangaroo_full_locomotion.flat_env_cfg import (
    KangFullFlatEnvCfg,
)

import pal_mjlab.tasks
from mjlab.third_party.isaaclab.isaaclab.utils.math import unscale_transform
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from mjlab.third_party.isaaclab.isaaclab.utils.string import (
    resolve_matching_names_values,
)


def run_play(
    task: str,
    wandb_run_path: str | None = None,
    checkpoint_file: str | None = None,
    motion_file: str | None = None,
    num_envs: int | None = None,
    total_steps: int = 2000,
    device: str | None = None,
    video: bool = False,
    video_length: int = 200,
    video_height: int | None = None,
    video_width: int | None = None,
    camera: int | str | None = None,
    render_all_envs: bool = False,
    viewer: Literal["native", "viser"] = "native",
):
    configure_torch_backends()

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO]: Using device: {device}")

    if checkpoint_file is not None and motion_file is None:
        raise ValueError(
            "Must provide `motion_file` if using `checkpoint_file`."
        )

    env_cfg = cast(
        ManagerBasedRlEnvCfg,
        load_cfg_from_registry(task, "env_cfg_entry_point"),
    )
    agent_cfg = cast(
        RslRlOnPolicyRunnerCfg,
        load_cfg_from_registry(task, "rl_cfg_entry_point"),
    )

    if num_envs is not None:
        env_cfg.scene.num_envs = num_envs
    if camera is not None:
        env_cfg.sim.render.camera = camera
    if video_height is not None:
        env_cfg.sim.render.height = video_height
    if video_width is not None:
        env_cfg.sim.render.width = video_width

    log_root_path = (
        Path("logs") / "rsl_rl" / agent_cfg.experiment_name
    ).resolve()
    print(f"[INFO]: Loading experiment from: {log_root_path}")

    if checkpoint_file is not None:
        resume_path = Path(checkpoint_file)
        if not resume_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file not found: {resume_path}"
            )
    else:
        assert wandb_run_path is not None
        resume_path = get_wandb_checkpoint_path(
            log_root_path, Path(wandb_run_path)
        )
    print(f"[INFO]: Loading checkpoint: {resume_path}")
    log_dir = resume_path.parent

    # Modify commands settings
    env_cfg.commands.twist.ranges.lin_vel_x = (1.0, 1.0)
    env_cfg.commands.twist.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.twist.ranges.ang_vel_z = (0.0, 0.0)

    env = gym.make(
        task,
        cfg=env_cfg,
        device=device,
        render_mode="rgb_array" if video else None,
    )
    if video:
        print("[INFO] Recording videos during play")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(log_dir / "videos" / "play"),
            step_trigger=lambda step: step == 0,
            video_length=video_length,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(
        env, asdict(agent_cfg), log_dir=str(log_dir), device=device
    )
    runner.load(str(resume_path), map_location=device)

    policy = runner.get_inference_policy(device=device)

    KANG_FULL_LINEAR_ACTUATORS = [
        ".*_hip_z_slider",
        ".*_hip_xy_slider_l",
        ".*_hip_xy_slider_r",
        ".*_ankle_xy_slider_l",
        ".*_ankle_xy_slider_r",
        ".*_leg_length_slider$",
    ]

    obs_dict = env.get_observations()

    robot = env.unwrapped.scene.entities["robot"]

    actuator_ids, actuator_names = robot.find_actuators(
        KANG_FULL_LINEAR_ACTUATORS
    )

    if isinstance(env_cfg, KangFullFlatEnvCfg):
        rescale_to_limits = env_cfg.actions.joint_pos.rescale_to_limits
        use_tanh = env_cfg.actions.joint_pos.rescale_to_limits
        offset = env_cfg.actions.joint_pos.offset
        scale = env_cfg.actions.joint_pos.scale

        if isinstance(env_cfg.actions.joint_pos.offset, (float, int)):
            offset = float(env_cfg.actions.joint_pos.offset)
        elif isinstance(env_cfg.actions.joint_pos.offset, dict):
            offset = torch.zeros_like(env.unwrapped.action_manager.action)
            index_list, _, value_list = resolve_matching_names_values(
                env_cfg.actions.joint_pos.offset, actuator_names
            )
            offset[:, index_list] = torch.tensor(value_list, device=env.device)
        else:
            raise ValueError(
                "Unsupported offset type."
                " Supported types are float and dict."
            )

    # Initialize lists to store quantities
    motors_force_list = []
    motors_pos_list = []
    raw_actions_list = []
    proc_actions_list = []
    left_foot_contact_force_list = []
    right_foot_contact_force_list = []

    step = 0
    with tqdm(total=total_steps, desc="Simulating", unit="step") as pbar:
        while step < total_steps:
            actions = policy(obs_dict)
            obs_dict, _, _, _ = env.step(actions)

            motors_force = robot.data.actuator_force[:, actuator_ids]
            motors_pos = robot.data.joint_pos[:, actuator_ids]

            proc_actions = actions * scale + offset
            if rescale_to_limits:
                if not use_tanh:
                    proc_actions = proc_actions.clamp(-1.0, 1.0)
                else:
                    proc_actions = torch.tanh(proc_actions)
                # rescale within the joint limits
                proc_actions = unscale_transform(
                    proc_actions,
                    robot.data.soft_joint_pos_limits[:, actuator_ids, 0],
                    robot.data.soft_joint_pos_limits[:, actuator_ids, 1],
                )

            actions_np = actions.detach().cpu().numpy()
            proc_actions_np = proc_actions.detach().cpu().numpy()

            left_foot_contact_force = robot.data.sensor_data[
                "left_foot_ground_contact"
            ][:, 1:4]
            right_foot_contact_force = robot.data.sensor_data[
                "right_foot_ground_contact"
            ][:, 1:4]

            # Store quantities
            motors_force_list.append(motors_force.detach().cpu().numpy())
            motors_pos_list.append(motors_pos.detach().cpu().numpy())
            raw_actions_list.append(actions_np)
            proc_actions_list.append(proc_actions_np)
            left_foot_contact_force_list.append(
                left_foot_contact_force.detach().cpu().numpy()
            )
            right_foot_contact_force_list.append(
                right_foot_contact_force.detach().cpu().numpy()
            )

            step += 1
            pbar.update(1)

    env.close()

    # Convert lists to numpy arrays for plotting
    motors_force_arr = np.concatenate(motors_force_list, axis=0)
    motors_pos_arr = np.concatenate(motors_pos_list, axis=0)
    raw_actions_arr = np.concatenate(raw_actions_list, axis=0)
    proc_actions_arr = np.concatenate(proc_actions_list, axis=0)
    left_foot_contact_force_arr = np.concatenate(
        left_foot_contact_force_list, axis=0
    )
    right_foot_contact_force_arr = np.concatenate(
        right_foot_contact_force_list, axis=0
    )
    # Split actuators into left and right based on their names
    left_indices = [
        i for i, name in enumerate(actuator_names) if "left" in name
    ]
    right_indices = [
        i for i, name in enumerate(actuator_names) if "right" in name
    ]
    # Plot positions
    fig_left, axs_left = plt.subplots(
        len(left_indices),
        1,
        figsize=(12, 3 * len(left_indices)),
        sharex=True,
    )
    if len(left_indices) == 1:
        axs_left = [axs_left]
    for idx, i in enumerate(left_indices):
        axs_left[idx].plot(proc_actions_arr[:, i], label="Processed Action")
        axs_left[idx].plot(motors_pos_arr[:, i], label="Motor Position")
        # Plot joint limits as dotted lines
        joint_min = (
            robot.data.soft_joint_pos_limits[0, actuator_ids[i], 0]
            .detach()
            .cpu()
            .numpy()
        )
        joint_max = (
            robot.data.soft_joint_pos_limits[0, actuator_ids[i], 1]
            .detach()
            .cpu()
            .numpy()
        )
        axs_left[idx].axhline(
            joint_min, color="red", linestyle=":", label="Joint Min"
        )
        axs_left[idx].axhline(
            joint_max, color="green", linestyle=":", label="Joint Max"
        )
        axs_left[idx].set_title(f"{actuator_names[i]}")
        axs_left[idx].set_ylabel("Position [m]")
        axs_left[idx].legend()
        axs_left[idx].grid()
    plt.xlabel("Step")
    plt.tight_layout()
    plt.show()
    fig_right, axs_right = plt.subplots(
        len(right_indices),
        1,
        figsize=(12, 3 * len(right_indices)),
        sharex=True,
    )
    if len(right_indices) == 1:
        axs_right = [axs_right]
    for idx, i in enumerate(right_indices):
        axs_right[idx].plot(proc_actions_arr[:, i], label="Processed Action")
        axs_right[idx].plot(motors_pos_arr[:, i], label="Motor Position")
        # Plot joint limits as dotted lines
        joint_min = (
            robot.data.soft_joint_pos_limits[0, actuator_ids[i], 0]
            .detach()
            .cpu()
            .numpy()
        )
        joint_max = (
            robot.data.soft_joint_pos_limits[0, actuator_ids[i], 1]
            .detach()
            .cpu()
            .numpy()
        )
        axs_right[idx].axhline(
            joint_min, color="red", linestyle=":", label="Joint Min"
        )
        axs_right[idx].axhline(
            joint_max, color="green", linestyle=":", label="Joint Max"
        )
        axs_right[idx].set_title(f"{actuator_names[i]}")
        axs_right[idx].set_ylabel("Position [m]")
        axs_right[idx].legend()
        axs_right[idx].grid()
    plt.xlabel("Step")
    plt.tight_layout()
    plt.show()

    # Plot actuator forces and velocities
    fig_left_fv, axs_left_fv = plt.subplots(
        len(left_indices),
        1,
        figsize=(12, 3 * len(left_indices)),
        sharex=True,
    )
    if len(left_indices) == 1:
        axs_left_fv = [axs_left_fv]
    for idx, i in enumerate(left_indices):
        axs_left_fv[idx].plot(motors_force_arr[:, i], label="Motor Force")
        axs_left_fv[idx].set_title(f"{actuator_names[i]}")
        axs_left_fv[idx].set_ylabel("Force (N)")
        axs_left_fv[idx].legend()
        axs_left_fv[idx].grid()
    plt.xlabel("Step")
    plt.tight_layout()
    plt.show()
    fig_right_fv, axs_right_fv = plt.subplots(
        len(right_indices),
        1,
        figsize=(12, 3 * len(right_indices)),
        sharex=True,
    )
    if len(right_indices) == 1:
        axs_right_fv = [axs_right_fv]
    for idx, i in enumerate(right_indices):
        axs_right_fv[idx].plot(motors_force_arr[:, i], label="Motor Force")
        axs_right_fv[idx].set_title(f"{actuator_names[i]}")
        axs_right_fv[idx].set_ylabel("Force (N)")
        axs_right_fv[idx].legend()
        axs_right_fv[idx].grid()
    plt.xlabel("Step")
    plt.tight_layout()
    plt.show()

    # Plot left and right foot contact forces (x, y, z components)
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    force_labels = ["X", "Y", "Z"]

    for i in range(3):
        axs[0].plot(
            left_foot_contact_force_arr[:, i],
            label=f"Left Foot {force_labels[i]}",
        )
        axs[1].plot(
            right_foot_contact_force_arr[:, i],
            label=f"Right Foot {force_labels[i]}",
        )

    axs[0].set_title("Left Foot Contact Forces")
    axs[1].set_title("Right Foot Contact Forces")
    for ax in axs:
        ax.set_ylabel("Force (N)")
        ax.legend()
        ax.grid()
    plt.xlabel("Step")
    plt.tight_layout()
    plt.show()


def main():
    """Entry point for the CLI."""
    tyro.cli(run_play)


if __name__ == "__main__":
    main()
