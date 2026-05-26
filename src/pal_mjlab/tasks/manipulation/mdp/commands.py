from __future__ import annotations

import math
import torch
import mujoco
from dataclasses import dataclass, field
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_from_euler_xyz, sample_uniform

TABLE_HEIGHT = 0.5
TABLE_HALF_X = 0.35
TABLE_HALF_Y = 0.35
BOX_HALF_SIZE = 0.025

def get_table_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(
        name="table",
        pos=(TABLE_HALF_X + 0.15, 0.0, TABLE_HEIGHT / 2),
    )
    body.add_geom(
        name="table_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(TABLE_HALF_X, TABLE_HALF_Y, TABLE_HEIGHT / 2),
        rgba=(0.1, 0.1, 0.1, 1.0),
        solref=(0.001, 1),
        solimp=(0.95, 0.99, 0.001, 0.5, 2), 
    )
    return spec

def get_box_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name="box_object")
    body.add_freejoint()
    body.add_geom(
        name="box_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=(BOX_HALF_SIZE, BOX_HALF_SIZE, 1.5 * BOX_HALF_SIZE),
        rgba=(0.8, 0.2, 0.2, 1.0),
        mass=0.1,
        solref=(0.001, 1),
        solimp=(0.95, 0.99, 0.001, 0.5, 2), 
    )
    return spec

class LiftingCommand(CommandTerm):
    cfg: LiftingCommandCfg

    def __init__(self, cfg: LiftingCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.object: Entity = env.scene[cfg.entity_name]
        self.contact_sensor: ContactSensor = env.scene[cfg.contact_sensor_name]
        self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.episode_success = torch.zeros(self.num_envs, device=self.device)
        self.metrics["object_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.target_pos

    @property
    def object_pos_w(self) -> torch.Tensor:
        return self.object.data.root_link_pos_w

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self.object.data.root_link_quat_w

    @property
    def object_bottom_z(self) -> torch.Tensor:
        return self.object_pos_w[:, 2] - self.cfg.object_half_height

    @property
    def object_on_table(self) -> torch.Tensor:
        return self.contact_sensor.data.found.any(dim=-1) > 0

    def _update_metrics(self) -> None:
        position_error = torch.norm(self.target_pos - self.object_pos_w, dim=-1)
        at_goal = (position_error < self.cfg.success_threshold).float()
        self.episode_success = torch.maximum(self.episode_success, at_goal)
        self.metrics["object_height"] = self.object_bottom_z
        self.metrics["position_error"] = position_error
        self.metrics["at_goal"] = at_goal
        self.metrics["episode_success"] = self.episode_success

    def compute_success(self) -> torch.Tensor:
        return self.metrics["position_error"] < self.cfg.success_threshold

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        self.episode_success[env_ids] = 0.0

        r = self.cfg.target_position_range
        lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
        upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
        target_pos = sample_uniform(lower, upper, (n, 3), device=self.device)
        self.target_pos[env_ids] = target_pos + self._env.scene.env_origins[env_ids]

        r = self.cfg.object_pose_range
        lower = torch.tensor([r.x[0], r.y[0], 0.0], device=self.device)
        upper = torch.tensor([r.x[1], r.y[1], 0.0], device=self.device)
        pos = sample_uniform(lower, upper, (n, 3), device=self.device)
        pos[:, 2] = self.cfg.table_height + self.cfg.object_half_height
        pos = pos + self._env.scene.env_origins[env_ids]

        yaw = sample_uniform(r.yaw[0], r.yaw[1], (n,), device=self.device)
        zeros = torch.zeros(n, device=self.device)
        quat = quat_from_euler_xyz(zeros, zeros, yaw)
        pose = torch.cat([pos, quat], dim=-1)

        self.object.write_root_link_pose_to_sim(pose, env_ids=env_ids)
        self.object.write_root_link_velocity_to_sim(
            torch.zeros(n, 6, device=self.device), env_ids=env_ids
        )

    def _update_command(self) -> None:
        pass

    def _debug_vis_impl(self, visualizer) -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        for batch in env_indices:
            visualizer.add_sphere(
                center=self.target_pos[batch].cpu().numpy(),
                radius=0.03,
                color=self.cfg.viz.target_color,
                label=f"target_position_{batch}",
            )

@dataclass(kw_only=True)
class LiftingCommandCfg(CommandTermCfg):
    entity_name: str
    object_half_height: float
    table_height: float
    contact_sensor_name: str
    success_threshold: float = 0.05

    @dataclass
    class TargetPositionRangeCfg:
        x: tuple[float, float] = (0.3, 0.5)
        y: tuple[float, float] = (-0.2, 0.2)
        z: tuple[float, float] = (0.2, 0.4)

    target_position_range: TargetPositionRangeCfg = field(
        default_factory=TargetPositionRangeCfg
    )

    @dataclass
    class ObjectPoseRangeCfg:
        x: tuple[float, float] = (0.4, 0.6)
        y: tuple[float, float] = (-0.1, 0.1)
        yaw: tuple[float, float] = (math.pi, math.pi)

    object_pose_range: ObjectPoseRangeCfg = field(default_factory=ObjectPoseRangeCfg)

    @dataclass
    class VizCfg:
        target_color: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.5)

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> LiftingCommand:
        return LiftingCommand(self, env)
