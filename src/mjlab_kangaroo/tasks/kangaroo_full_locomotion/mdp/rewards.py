from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def is_terminated(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.termination_manager.terminated.float()


def is_terminated_term(env: ManagerBasedRlEnv, term_key: str) -> torch.Tensor:
    """Penalize termination for specific term that don't correspond to episodic timeouts."""
    return env.termination_manager.get_term(term_key).float()


def lin_vel_z_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def ang_vel_xy_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(
        torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1
    )


def track_lin_vel_exp(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_lin_vel_b
    desired = torch.zeros_like(actual)
    desired[:, :2] = command[:, :2]
    lin_vel_error = torch.sum(torch.square(desired - actual), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_exp(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_ang_vel_b
    desired = torch.zeros_like(actual)
    desired[:, 2] = command[:, 2]
    ang_vel_error = torch.sum(torch.square(desired - actual), dim=1)
    return torch.exp(-ang_vel_error / std**2)


class feet_air_time:
    """Reward long steps taken by the feet.

    This rewards the agent for lifting feet off the ground for longer than a threshold.
    Provides continuous reward signal during flight phase and smooth command scaling.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.threshold_min = cfg.params["threshold_min"]
        self.threshold_max = cfg.params.get(
            "threshold_max", self.threshold_min + 0.3
        )
        self.asset_name = cfg.params["asset_name"]
        self.sensor_names = cfg.params["sensor_names"]
        self.num_feet = len(self.sensor_names)
        self.command_name = cfg.params["command_name"]
        self.command_threshold = cfg.params["command_threshold"]
        self.reward_mode = cfg.params.get("reward_mode", "continuous")
        self.command_scale_type = cfg.params.get(
            "command_scale_type", "smooth"
        )
        self.command_scale_width = cfg.params.get("command_scale_width", 0.2)

        asset: Entity = env.scene[self.asset_name]
        for sensor_name in self.sensor_names:
            if sensor_name not in asset.sensor_names:
                raise ValueError(
                    f"Sensor '{sensor_name}' not found in asset '{self.asset_name}'"
                )

        self.current_air_time = torch.zeros(
            env.num_envs, self.num_feet, device=env.device
        )
        self.current_contact_time = torch.zeros(
            env.num_envs, self.num_feet, device=env.device
        )
        self.last_air_time = torch.zeros(
            env.num_envs, self.num_feet, device=env.device
        )

    def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
        asset: Entity = env.scene[self.asset_name]

        contact_list = []
        for sensor_name in self.sensor_names:
            sensor_data = asset.data.sensor_data[sensor_name]
            foot_contact = sensor_data[:, 0] > 0
            contact_list.append(foot_contact)

        in_contact = torch.stack(contact_list, dim=1)
        in_air = ~in_contact

        # Detect first contact (landing).
        first_contact = (self.current_air_time > 0) & in_contact

        # Save air time when landing.
        self.last_air_time = torch.where(
            first_contact, self.current_air_time, self.last_air_time
        )

        # Update air time and contact time.
        self.current_air_time = torch.where(
            in_contact,
            torch.zeros_like(self.current_air_time),  # Reset when in contact.
            self.current_air_time + env.step_dt,  # Increment when in air.
        )

        self.current_contact_time = torch.where(
            in_contact,
            self.current_contact_time
            + env.step_dt,  # Increment when in contact.
            torch.zeros_like(self.current_contact_time),  # Reset when in air.
        )

        if self.reward_mode == "continuous":
            # Give constant reward of 1.0 for each foot that's in air and above threshold.
            exceeds_min = self.current_air_time > self.threshold_min
            below_max = self.current_air_time <= self.threshold_max
            reward_per_foot = torch.where(
                in_air & exceeds_min & below_max,
                torch.ones_like(self.current_air_time),
                torch.zeros_like(self.current_air_time),
            )
            reward = torch.sum(reward_per_foot, dim=1)
        else:
            # This mode gives (air_time - threshold) as reward on landing.
            air_time_over_min = (
                self.last_air_time - self.threshold_min
            ).clamp(min=0.0)
            air_time_clamped = air_time_over_min.clamp(
                max=self.threshold_max - self.threshold_min
            )
            reward = (
                torch.sum(air_time_clamped * first_contact, dim=1)
                / env.step_dt
            )

        command = env.command_manager.get_command(self.command_name)
        assert command is not None
        command_norm = torch.norm(command[:, :2], dim=1)
        if self.command_scale_type == "smooth":
            scale = 0.5 * (
                1.0
                + torch.tanh(
                    (command_norm - self.command_threshold)
                    / self.command_scale_width
                )
            )
            reward *= scale
        else:
            reward *= command_norm > self.command_threshold
        return reward

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.current_air_time[env_ids] = 0.0
        self.current_contact_time[env_ids] = 0.0
        self.last_air_time[env_ids] = 0.0


def foot_clearance_reward(
    env: ManagerBasedRlEnv,
    target_height: float,
    std: float,
    tanh_mult: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(
        asset.data.geom_pos_w[:, asset_cfg.geom_ids, 2] - target_height
    )
    foot_velocity_tanh = torch.tanh(
        tanh_mult
        * torch.norm(
            asset.data.geom_lin_vel_w[:, asset_cfg.geom_ids, :2], dim=2
        )
    )
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def foot_clearance_reward_2(
    env: ManagerBasedRlEnv,
    target_height: float,
    std: float,
    tanh_mult: float,
    min_clearance: float = 0.045,
    min_speed: float = 0.05,
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    # TODO: adjust if RayCaster present

    # height of each foot
    foot_height = asset.data.geom_pos_w[:, asset_cfg.geom_ids, 2]
    foot_z_target_error = (foot_height - target_height).pow(2)

    # horizontal speed of each foot
    foot_speed = torch.norm(
        asset.data.geom_lin_vel_w[:, asset_cfg.geom_ids, :2], dim=2
    )
    foot_velocity_tanh = torch.tanh(tanh_mult * foot_speed)

    # robot command velocity and body velocity
    cmd = torch.linalg.norm(
        env.command_manager.get_command(command_name), dim=1
    )
    body_vel = torch.linalg.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
    moving_env = (cmd > min_speed) | (body_vel > min_speed)

    # foot-level clearance condition, then reduce to env-level (any foot high enough)
    foot_high = foot_height > min_clearance
    any_foot_high = torch.any(foot_high, dim=1)

    # final env-level gate
    gate = moving_env & any_foot_high

    reward = torch.exp(
        -torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1) / std
    )
    reward = torch.where(gate, reward, torch.zeros_like(reward))
    return reward


def foot_clearance_stop_aware(
    env: ManagerBasedRlEnv,
    target_height: float,
    std: float,
    tanh_mult: float,
    min_clearance: float = 0.045,
    v_enter: float = 0.06,  # start “walk” around here
    v_exit: float = 0.10,  # definitely “walk” above here
    stop_height_penalty: float = 5.0,  # weight for height when stopping
    stop_speed_penalty: float = 0.5,  # weight for foot horiz speed when stopping
    command_name: str = "twist",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]

    # --- foot kinematics ---
    foot_height = asset.data.geom_pos_w[:, asset_cfg.geom_ids, 2]  # [B, nfeet]
    foot_speed_xy = torch.norm(
        asset.data.geom_lin_vel_w[:, asset_cfg.geom_ids, :2], dim=2
    )  # [B, nfeet]

    # --- positive clearance shaping for walking ---
    z_err_sq = (foot_height - target_height).pow(2)  # [B, nfeet]
    speed_tanh = torch.tanh(tanh_mult * foot_speed_xy)  # [B, nfeet]
    walk_clearance = torch.exp(
        -torch.sum(z_err_sq * speed_tanh, dim=1) / std
    )  # [B]

    # --- commanded speed magnitude (use only command to avoid feedback loops) ---
    cmd = env.command_manager.get_command(
        command_name
    )  # [B, 3] e.g. vx, vy, wz
    cmd_mag = torch.linalg.norm(cmd, dim=1)  # [B]

    # --- smooth gate: 0 at low speed, 1 at high speed (Schmitt-like with smoothing) ---
    # map cmd_mag in [v_enter, v_exit] to s in [0,1], clamp, then smoothstep
    denom = v_exit - v_enter + 1e-6
    s = ((cmd_mag - v_enter) / denom).clamp(0.0, 1.0)
    # smoothstep(s) = 3s^2 - 2s^3 (C1 continuous, nicer gradients than hard gates)
    w_move = 3 * s * s - 2 * s * s * s  # [B]
    w_stop = 1.0 - w_move

    # --- small penalty when "stopping": discourage stepping/fidgeting ---
    # penalize feet being above a tiny clearance and penalize horizontal foot motion

    foot_above = torch.clamp(foot_height - min_clearance, min=0.0)
    stop_penalty = stop_height_penalty * torch.sum(
        foot_above * foot_above, dim=1
    ) + stop_speed_penalty * torch.sum(
        foot_speed_xy * foot_speed_xy, dim=1
    )  # [B]

    # --- combine: reward is positive during motion, negative when stopped ---
    reward = w_move * walk_clearance - w_stop * stop_penalty
    return reward


def feet_slide(
    env: ManagerBasedRlEnv,
    sensor_names: list[str],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_list = []
    for sensor_name in sensor_names:
        sensor_data = asset.data.sensor_data[sensor_name]
        foot_contact = sensor_data[:, 0] > 0
        contact_list.append(foot_contact)
    contacts = torch.stack(contact_list, dim=1)
    geom_vel = asset.data.geom_lin_vel_w[:, asset_cfg.geom_ids, :2]
    return torch.sum(geom_vel.norm(dim=-1) * contacts, dim=1)


def contact_forces(
    env: ManagerBasedRlEnv,
    threshold: float,
    sensor_names: list[str],
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    forces_over_thresh = torch.zeros(env.num_envs, device=env.device)
    for sensor_name in sensor_names:
        sensor_data = asset.data.sensor_data[sensor_name]
        # found = sensor_data[:, 0]
        force = sensor_data[:, 1:4]
        # pos = sensor_data[:, -3:]
        forces_over_thresh += (force.norm(dim=-1) - threshold).clamp(min=0.0)
    return forces_over_thresh


class feet_air_contact_time:
    """Reward long feet air and contact times (up to a threshold)."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.asset_name = cfg.params["asset_name"]
        self.sensor_names = cfg.params["sensor_names"]
        self.num_feet = len(self.sensor_names)
        self.mode_time = cfg.params["mode_time"]
        self.command_name = cfg.params["command_name"]
        self.command_threshold = cfg.params["command_threshold"]

        asset: Entity = env.scene[self.asset_name]
        for sensor_name in self.sensor_names:
            if sensor_name not in asset.sensor_names:
                raise ValueError(
                    f"Sensor '{sensor_name}' not found in asset '{self.asset_name}'"
                )

        self.current_air_time = torch.zeros(
            env.num_envs, self.num_feet, device=env.device
        )
        self.current_contact_time = torch.zeros(
            env.num_envs, self.num_feet, device=env.device
        )
        self.last_air_time = torch.zeros(
            env.num_envs, self.num_feet, device=env.device
        )

    def __call__(self, env: ManagerBasedRlEnv, **kwargs) -> torch.Tensor:
        asset: Entity = env.scene[self.asset_name]

        contact_list = []
        for sensor_name in self.sensor_names:
            sensor_data = asset.data.sensor_data[sensor_name]
            foot_contact = sensor_data[:, 0] > 0
            contact_list.append(foot_contact)

        in_contact = torch.stack(contact_list, dim=1)

        # Detect first contact (landing).
        first_contact = (self.current_air_time > 0) & in_contact

        # Save air time when landing.
        self.last_air_time = torch.where(
            first_contact, self.current_air_time, self.last_air_time
        )

        # Update air time and contact time.
        self.current_air_time = torch.where(
            in_contact,
            torch.zeros_like(self.current_air_time),  # Reset when in contact.
            self.current_air_time + env.step_dt,  # Increment when in air.
        )

        self.current_contact_time = torch.where(
            in_contact,
            self.current_contact_time
            + env.step_dt,  # Increment when in contact.
            torch.zeros_like(self.current_contact_time),  # Reset when in air.
        )

        t_max = torch.max(self.current_air_time, self.current_contact_time)
        t_min = torch.clip(t_max, max=self.mode_time)
        stance_cmd_reward = torch.clip(
            self.current_contact_time - self.current_air_time,
            -self.mode_time,
            self.mode_time,
        )

        cmd = torch.norm(
            env.command_manager.get_command(self.command_name)[:, :2],
            dim=1,
            keepdim=True,
        )
        body_vel = torch.linalg.norm(
            asset.data.root_link_lin_vel_b[:, :2], dim=1, keepdim=True
        )

        reward = torch.where(
            torch.logical_or(cmd > 0.0, body_vel > self.command_threshold),
            torch.where(
                t_max < self.mode_time, t_min, torch.zeros_like(t_min)
            ),
            stance_cmd_reward,
        )

        return torch.sum(reward, dim=1)

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.current_air_time[env_ids] = 0.0
        self.current_contact_time[env_ids] = 0.0
        self.last_air_time[env_ids] = 0.0


def base_height_l2(
    env: ManagerBasedRlEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain ONLY, target height is in the world frame.
        TODO: adapt for rough terrain.
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Compute the L2 squared penalty
    return torch.square(asset.data.root_link_pos_w[:, 2] - target_height)
