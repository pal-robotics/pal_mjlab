from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor.contact_sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.utils.lab_api.string import resolve_matching_names_values


from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_DEFAULT_BOX_CFG = SceneEntityCfg("box")


# ============================================================================
# Activation / gating helpers
# ============================================================================


def _active_by_box_distance(
    env: ManagerBasedRlEnv,
    box_cfg: SceneEntityCfg,
    dist: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    invert: bool = False,
) -> torch.Tensor:
    """Shared gating helper for the *_box reward variants below.
 
    Returns 1.0 where horizontal distance between `asset_cfg` and `box_cfg`
    exceeds `dist` (still walking toward the box), or under it if
    `invert=True` (already at the box, reaching/lifting). This replaces the
    command-speed gate (`command_name`/`command_threshold`) used by the
    velocity-task originals, since box-lifting has no locomotion command —
    the walk/lift phase is determined by proximity to the box instead.
    """
    asset: Entity = env.scene[asset_cfg.name]
    box: Entity = env.scene[box_cfg.name]
    dist_norm = torch.norm(
        asset.data.root_link_pos_w[:, :2] - box.data.root_link_pos_w[:, :2], dim=-1
    )
    active = dist_norm > dist
    if invert:
        active = ~active
    return active.float()


# ============================================================================
# Reworked rewards for box lifting
# ============================================================================


def feet_air_time_box(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold_min: float = 0.05,
    threshold_max: float = 0.5,
    dist: float | None = None,
    box_cfg: SceneEntityCfg = _DEFAULT_BOX_CFG,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward feet air time within [threshold_min, threshold_max].
 
    Gated by distance to the box: with `dist` set, only rewards air time
    while still walking toward the box (dist_norm > dist), matching the
    genesis original's `active_by_distance_to_box(dist=0.60)` gate.
    """
    sensor: ContactSensor = env.scene[sensor_name]
    current_air_time = sensor.data.current_air_time
    assert current_air_time is not None
 
    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)
 
    in_air = current_air_time > 0
    num_in_air = torch.sum(in_air.float())
    mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
        num_in_air, min=1
    )
    env.extras["log"]["Metrics/air_time_mean_box"] = mean_air_time
 
    if dist is not None:
        reward = reward * _active_by_box_distance(env, box_cfg, dist, asset_cfg)
    return reward
 
 
def feet_clearance_box(
    env: ManagerBasedRlEnv,
    target_height: float,
    height_sensor_name: str,
    dist: float | None = None,
    box_cfg: SceneEntityCfg = _DEFAULT_BOX_CFG,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize deviation from target foot clearance (terrain-relative,
    via TerrainHeightSensor), weighted by foot velocity. Gated by distance
    to the box instead of commanded speed.
    """
    asset: Entity = env.scene[asset_cfg.name]
    height_sensor = env.scene[height_sensor_name]
    assert isinstance(height_sensor, TerrainHeightSensor), (
        f"feet_clearance_box requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
    )
    foot_height = height_sensor.data.heights  # [B, F]
    foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, F, 2]
    vel_norm = torch.norm(foot_vel_xy, dim=-1)
    delta = torch.abs(foot_height - target_height)
    cost = torch.sum(delta * vel_norm, dim=1)
 
    if dist is not None:
        cost = cost * _active_by_box_distance(env, box_cfg, dist, asset_cfg)
    return cost
 
 
class feet_swing_height_box:
    """Penalize deviation from target swing height, evaluated at landing.
 
    Gated by distance to the box rather than command speed.
    """
 
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        height_sensor = env.scene[cfg.params["height_sensor_name"]]
        assert isinstance(height_sensor, TerrainHeightSensor), (
            f"feet_swing_height_box requires a TerrainHeightSensor, got {type(height_sensor).__name__}"
        )
        num_feet = height_sensor.num_frames
        self.peak_heights = torch.zeros(
            (env.num_envs, num_feet), device=env.device, dtype=torch.float32
        )
        self.step_dt = env.step_dt
 
    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        height_sensor_name: str,
        target_height: float,
        dist: float,
        box_cfg: SceneEntityCfg = _DEFAULT_BOX_CFG,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        contact_sensor: ContactSensor = env.scene[sensor_name]
        height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
        foot_heights = height_sensor.data.heights
 
        in_air = contact_sensor.data.found == 0
        self.peak_heights = torch.where(
            in_air,
            torch.maximum(self.peak_heights, foot_heights),
            self.peak_heights,
        )
        first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
 
        active = _active_by_box_distance(env, box_cfg, dist, asset_cfg)
 
        error = self.peak_heights / target_height - 1.0
        cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
 
        num_landings = torch.sum(first_contact.float())
        peak_heights_at_landing = self.peak_heights * first_contact.float()
        mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
            num_landings, min=1
        )
        env.extras["log"]["Metrics/peak_height_mean_box"] = mean_peak_height
 
        self.peak_heights = torch.where(
            first_contact, torch.zeros_like(self.peak_heights), self.peak_heights
        )
        return cost
 
 
def feet_slip_box(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    dist: float,
    box_cfg: SceneEntityCfg = _DEFAULT_BOX_CFG,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize foot sliding (xy velocity while in contact). Gated by
    distance to the box rather than command speed.
    """
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
 
    active = _active_by_box_distance(env, box_cfg, dist, asset_cfg)
 
    assert contact_sensor.data.found is not None
    in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
    foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
    vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)
    vel_xy_norm_sq = torch.square(vel_xy_norm)
    cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
 
    num_in_contact = torch.sum(in_contact)
    mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
        num_in_contact, min=1
    )
    env.extras["log"]["Metrics/slip_velocity_mean_box"] = mean_slip_vel
    return cost
 
 
def soft_landing_box(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    dist: float | None = None,
    box_cfg: SceneEntityCfg = _DEFAULT_BOX_CFG,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize high impact forces at landing to encourage soft footfalls.
    Gated by distance to the box rather than command speed.
    """
    contact_sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = contact_sensor.data
    assert sensor_data.force is not None
    forces = sensor_data.force  # [B, N, 3]
    force_magnitude = torch.norm(forces, dim=-1)
    first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)
    landing_impact = force_magnitude * first_contact.float()
    cost = torch.sum(landing_impact, dim=1)
 
    num_landings = torch.sum(first_contact.float())
    mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
    env.extras["log"]["Metrics/landing_force_mean_box"] = mean_landing_force
 
    if dist is not None:
        cost = cost * _active_by_box_distance(env, box_cfg, dist, asset_cfg)
    return cost

def box_proximity(
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    box_cfg: SceneEntityCfg = _DEFAULT_BOX_CFG,
    dist: float = 0.50,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    box: Entity = env.scene[box_cfg.name]

    dist_norm = torch.norm(
        asset.data.root_link_pos_w[:, :2] - box.data.root_link_pos_w[:, :2], dim=-1
    )
    active = dist_norm > dist
    reward = torch.exp(-(dist_norm**2) / (2 * std**2))
    return torch.where(active, reward, torch.zeros_like(reward))


def hands_to_box(
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg,
    box_cfg: SceneEntityCfg = _DEFAULT_BOX_CFG,
    dist: float = 0.60,
) -> torch.Tensor:
    """`asset_cfg.body_ids` should select the hand/wrist bodies to draw
    toward the box (replaces genesis's per-link_name loop with a single
    batched body-index lookup).
    """
    asset: Entity = env.scene[asset_cfg.name]
    box: Entity = env.scene[box_cfg.name]

    box_pos = box.data.root_link_pos_w  # (N, 3)
    active = (
        torch.norm(asset.data.root_link_pos_w[:, :2] - box_pos[:, :2], dim=-1) < dist
    )

    hand_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids]  # (N, n_hands, 3)
    err = torch.norm(hand_pos - box_pos.unsqueeze(1), dim=-1)
    reward = torch.sum(torch.exp(-torch.square(err) / std**2), dim=1)

    return active.float() * reward

def hand_contact_reward (
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    sensor_data = contact_sensor.data
    assert sensor_data.force is not None
    forces = sensor_data.force  # [B, N, 3]
    force_magnitude = torch.norm(forces, dim=-1)
    
    reward = torch.tanh(force_magnitude/15.0 - 2.0) + 1.0

    return torch.sum(reward, dim=-1)

def box_height(
    env: ManagerBasedRlEnv,
    std: float,
    target_height: float = 1.0,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    box_cfg: SceneEntityCfg = _DEFAULT_BOX_CFG,
    dist: float = 0.60,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    box: Entity = env.scene[box_cfg.name]

    box_h = box.data.root_link_pos_w[:, 2]
    dist_norm = torch.norm(
        asset.data.root_link_pos_w[:, :2] - box.data.root_link_pos_w[:, :2], dim=-1
    )
    active = (dist_norm < dist).float()

    err = torch.square(box_h - target_height)
    return active * torch.exp(-err / std**2)


def look_at_box(
    env: ManagerBasedRlEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    box_cfg: SceneEntityCfg = _DEFAULT_BOX_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    box: Entity = env.scene[box_cfg.name]

    asset_pos = asset.data.root_link_pos_w[:, :2]
    box_pos = box.data.root_link_pos_w[:, :2]

    to_box = box_pos - asset_pos
    to_box_dir = to_box / torch.norm(to_box, dim=-1, keepdim=True).clamp(min=1e-6)

    quat = asset.data.root_link_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    heading = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y + w * z)], dim=-1)

    cos_err = (heading * to_box_dir).sum(dim=-1)
    err = torch.square(1.0 - cos_err)
    return torch.exp(-err / std**2)


def foot_on_ground(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    box_cfg: SceneEntityCfg = _DEFAULT_BOX_CFG,
    dist: float = 0.60,
) -> torch.Tensor:
    """Reward exactly one foot grounded once near the box (single-leg stance
    for reaching).
    """
    asset: Entity = env.scene[asset_cfg.name]
    box: Entity = env.scene[box_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]

    dist_norm = torch.norm(
        asset.data.root_link_pos_w[:, :2] - box.data.root_link_pos_w[:, :2], dim=-1
    )
    active = dist_norm > dist

    in_contact = contact_sensor.data.current_contact_time > 0.0
    num_contacts = in_contact.sum(dim=-1)
    one_grounded = (num_contacts == 1) | ~active

    return one_grounded.float()


def horizontal_vel_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    max_speed: float = 1.0,
) -> torch.Tensor:
    """Penalize horizontal speed above `max_speed` (world-frame, unlike the
    body-frame-tracking rewards above — this is a raw speed cap, not a
    heading-relative tracking term).
    """
    asset: Entity = env.scene[asset_cfg.name]
    speed = torch.norm(asset.data.root_link_lin_vel_w[:, :2], dim=-1)
    excess = torch.clamp(speed - max_speed, min=0.0)
    return torch.square(10.0 * excess)


def ang_vel_penalty(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    max_yaw_rate: float = 1.2,
) -> torch.Tensor:
    """Penalize yaw rate above `max_yaw_rate`."""
    asset: Entity = env.scene[asset_cfg.name]
    yaw_rate = torch.abs(asset.data.root_link_ang_vel_w[:, 2])
    excess = torch.clamp(yaw_rate - max_yaw_rate, min=0.0)
    return torch.square(excess)


class VariablePostureBoxLifting:
    """Like `VariablePosture`, but the two regimes (walking / lifting) are
    gated by distance to the box rather than commanded speed.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset: Entity = env.scene[asset_cfg.name]
        _, joint_names = asset.find_joints(asset_cfg.joint_names)

        _, _, std_walking = resolve_matching_names_values(
            data=cfg.params.get("std_walking"), list_of_strings=joint_names
        )
        _, _, std_lifting = resolve_matching_names_values(
            data=cfg.params.get("std_lifting"), list_of_strings=joint_names
        )

        self.std_walking = torch.tensor(std_walking, device=env.device)
        self.std_lifting = torch.tensor(std_lifting, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std_walking,
        std_lifting,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        box_cfg: SceneEntityCfg = _DEFAULT_BOX_CFG,
        dist: float = 0.60,
    ) -> torch.Tensor:
        del std_walking, std_lifting

        asset: Entity = env.scene[asset_cfg.name]
        box: Entity = env.scene[box_cfg.name]

        dist_norm = torch.norm(
            asset.data.root_link_pos_w[:, :2] - box.data.root_link_pos_w[:, :2], dim=-1
        )
        walking_mask = (dist_norm > dist).float()
        lifting_mask = (dist_norm <= dist).float()

        std = (
            self.std_walking * walking_mask.unsqueeze(1)
            + self.std_lifting * lifting_mask.unsqueeze(1)
        )

        current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
        desired_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        error_squared = torch.square(current_joint_pos - desired_joint_pos)

        return torch.exp(-torch.mean(error_squared / (std**2), dim=1))