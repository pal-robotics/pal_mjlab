from dataclasses import dataclass, replace

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.velocity_env_cfg import (
  LocomotionVelocityEnvCfg,
)

from pal_mjlab.robots import (
    KANG_FULL_ROBOT_CFG,
    KANG_FULL_LINEAR_ACTUATORS,
)


@dataclass
class KangFullRoughEnvCfg(LocomotionVelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.entities = {"robot": replace(KANG_FULL_ROBOT_CFG)}

        # constants
        geom_names = ["left_foot_collision", "right_foot_collision"]
        site_names = ["left_foot", "right_foot"]
        target_foot_height = 0.15

        # sensors
        feet_ground_cfg = ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(
                mode="body",
                pattern=r"^(left_ankle_xy_foot|right_ankle_xy_foot)$",
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
        )
        self_collision_cfg = ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(mode="subtree", pattern="baselink", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="baselink", entity="robot"),
            fields=("found",),
            reduce="none",
            num_slots=1,
        )
        # scene
        self.scene.sensors = (self_collision_cfg, feet_ground_cfg,)

        # actions
        self.actions.joint_pos.actuator_names = KANG_FULL_LINEAR_ACTUATORS
        offs = {
            "left_hip_z_slider": 0.0,
            "left_hip_xy_slider_l": 0.0,
            "left_hip_xy_slider_r": 0.0,
            "left_leg_length_slider": -0.54,
            "left_ankle_xy_slider_l": 0.0,
            "left_ankle_xy_slider_r": 0.0,
            "right_hip_z_slider": 0.0,
            "right_hip_xy_slider_l": 0.0,
            "right_hip_xy_slider_r": 0.0,
            "right_leg_length_slider": -0.54,
            "right_ankle_xy_slider_l": 0.0,
            "right_ankle_xy_slider_r": 0.0,
        }

        self.actions.joint_pos.offset = offs
        self.actions.joint_pos.scale = 0.1

        # observations
        self.observations.policy.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=KANG_FULL_LINEAR_ACTUATORS,
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=KANG_FULL_LINEAR_ACTUATORS,
        )
        self.observations.critic.joint_pos.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=KANG_FULL_LINEAR_ACTUATORS,
        )
        self.observations.critic.joint_vel.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=KANG_FULL_LINEAR_ACTUATORS,
        )

        # events
        self.events.foot_friction.params["asset_cfg"].geom_names = geom_names

        # rewards
        self.rewards.upright.params["asset_cfg"].body_names = ["pelvis_2_link"]
        self.rewards.pose.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=KANG_FULL_LINEAR_ACTUATORS,
        )
        # Tight control when stationary: maintain stable default pose.
        self.rewards.pose.params["std_standing"] = {
            r".*_hip_z_slider": 0.05, # yaw
            r".*_hip_xy_slider_l": 0.05,
            r".*_hip_xy_slider_r": 0.05,
            r".*_leg_length_slider": 0.05,
            r".*_ankle_xy_slider_l": 0.05,
            r".*_ankle_xy_slider_r": 0.05,
        }
        # Moderate leg freedom for stepping, loose arms for natural pendulum swing.
        self.rewards.pose.params["std_walking"] = {
            r".*_hip_z_slider": 0.15, # yaw
            r".*_hip_xy_slider_l": 0.3,
            r".*_hip_xy_slider_r": 0.3,
            r".*_leg_length_slider": 0.35,
            r".*_ankle_xy_slider_l": 0.2,
            r".*_ankle_xy_slider_r": 0.2,
        }
        self.rewards.pose.params["std_running"] = {
            r".*_hip_z_slider": 0.2,
            r".*_hip_xy_slider_l": 0.5,
            r".*_hip_xy_slider_r": 0.5,
            r".*_leg_length_slider": 0.6,
            r".*_ankle_xy_slider_l": 0.25,
            r".*_ankle_xy_slider_r": 0.25,
        }
        self.rewards.foot_clearance.params["asset_cfg"].site_names = site_names
        self.rewards.foot_swing_height.params["asset_cfg"].site_names = site_names
        self.rewards.foot_slip.params["asset_cfg"].site_names = site_names
        self.rewards.foot_swing_height.params["target_height"] = target_foot_height
        self.rewards.foot_clearance.params["target_height"] = target_foot_height
        self.rewards.body_ang_vel.params["asset_cfg"].body_names = ["pelvis_2_link"]

        # observations
        self.observations.critic.foot_height.params["asset_cfg"].site_names = site_names

        # terminations
        self.terminations.illegal_contact = None

        self.viewer.body_name = "baselink"
        self.commands.twist.viz.z_offset = 1.5


@dataclass
class KangFullRoughEnvCfg_PLAY(KangFullRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        if self.scene.terrain is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
                self.scene.terrain.terrain_generator.num_cols = 5
                self.scene.terrain.terrain_generator.num_rows = 5
                self.scene.terrain.terrain_generator.border_width = 10.0
