from dataclasses import dataclass, replace

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.spec_config import ContactSensorCfg

from pal_mjlab.robots import (
    KANG_FULL_ROBOT_CFG,
    KANG_FULL_LINEAR_ACTUATORS,
)
from pal_mjlab.tasks.kangaroo_full_locomotion.velocity_env_cfg import (
    LocomotionVelocityEnvCfg,
)


@dataclass
class KangFullRoughEnvCfg(LocomotionVelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        foot_contact_sensors = [
            ContactSensorCfg(
                name=f"{side}_foot_ground_contact",
                body1=f"{side}_ankle_xy_foot",
                body2="terrain",
                num=1,
                data=("found", "force", "pos"),
                reduce="netforce",
            )
            for side in ["left", "right"]
        ]
        robot_cfg = replace(KANG_FULL_ROBOT_CFG, sensors=tuple(foot_contact_sensors))
        self.scene.entities = {"robot": robot_cfg}

        sensor_names = [
            "left_foot_ground_contact",
            "right_foot_ground_contact",
        ]

        # ActionCfg
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

        # ObservationCfg
        self.observations.policy.joint_pos.params = {
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=KANG_FULL_LINEAR_ACTUATORS,
            )
        }
        self.observations.policy.joint_vel.params = {
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=KANG_FULL_LINEAR_ACTUATORS,
            )
        }
        self.observations.critic.joint_pos.params = {
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=KANG_FULL_LINEAR_ACTUATORS,
            )
        }
        self.observations.critic.joint_vel.params = {
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=KANG_FULL_LINEAR_ACTUATORS,
            )
        }

        # EventCfg
        self.events.foot_friction.params["asset_cfg"].geom_names = [
            r"^(left|right)_foot_collision$"
        ]

        # RewardCfg
        self.rewards.foot_clearance.params["asset_cfg"].geom_names = [
            r"^(left|right)_foot_collision$"
        ]
        self.rewards.air_contact_time.params["sensor_names"] = sensor_names
        self.rewards.full_feet_contacts.params["sensor_names"] = sensor_names
        self.rewards.feet_slide.params["sensor_names"] = sensor_names
        self.rewards.feet_slide.params["asset_cfg"].geom_names = [
            r"^(left|right)_foot_collision$"
        ]
        self.rewards.feet_too_near.params["asset_cfg"].geom_names = [
            r"^(left|right)_foot_collision$"
        ]
        self.rewards.contact_forces.params["sensor_names"] = sensor_names
        self.rewards.contact_forces.params["asset_cfg"].geom_names = [
            r"^(left|right)_foot_collision$"
        ]
        self.rewards.base_height.params["target_height"] = robot_cfg.init_state.pos[2]

        # Visualization settings
        self.viewer.body_name = "baselink"
        self.commands.twist.viz.z_offset = 1.2


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
