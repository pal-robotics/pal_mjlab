from dataclasses import dataclass, replace

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.spec_config import ContactSensorCfg

from mjlab_kangaroo.robots import (
    KANG_FULL_ROBOT_CFG,
    KANG_FULL_LINEAR_ACTUATORS,
    # KANG_FULL_REVOLUTE_ACTUATORS,
)
from mjlab_kangaroo.tasks.kangaroo_full_locomotion.velocity_env_cfg import (
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
        robot_cfg = replace(
            KANG_FULL_ROBOT_CFG, sensors=tuple(foot_contact_sensors)
        )
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

        # CommandsCfg
        self.commands.twist.rel_standing_envs = 0.2
        self.commands.twist.viz.z_offset = 0.75

        # EventCfg
        self.events.foot_friction.params["asset_cfg"].geom_names = [
            r"^(left|right)_foot_collision$"
        ]
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (-1.0, 1.0),
            "y": (-1.0, 1.0),
            "z": (-0.2, 0.2),
            "roll": (-0.52, 0.52),
            "pitch": (-0.52, 0.52),
            "yaw": (-0.79, 0.79),
        }

        # RewardCfg
        # ---- Rewards ----
        self.rewards.track_lin_vel_exp.weight = 2.0
        self.rewards.track_ang_vel_exp.weight = 2.0

        self.rewards.air_contact_time.weight = 1.0
        self.rewards.air_contact_time.params["sensor_names"] = sensor_names
        self.rewards.air_contact_time.params["command_threshold"] = 0.1
        self.rewards.air_contact_time.params["mode_time"] = 0.3

        self.rewards.foot_clearance.weight = 0.5
        self.rewards.foot_clearance.params["target_height"] = 0.20
        self.rewards.foot_clearance.params["min_clearance"] = 0.045
        self.rewards.foot_clearance.params["asset_cfg"].geom_names = [
            r"^(left|right)_foot_collision$"
        ]

        self.rewards.full_feet_contacts.weight = 0.1
        self.rewards.full_feet_contacts.params["sensor_names"] = sensor_names
        self.rewards.full_feet_contacts.params["n_contacts"] = 4

        # ---- Penalties ----
        self.rewards.dof_pos_limits = None
        self.rewards.posture = None
        self.rewards.termination_penalty.weight = -20.0
        self.rewards.lin_vel_z_l2.weight = -1.0
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.flat_orientation_l2.weight = -1.0

        self.rewards.feet_slide.weight = -0.1
        self.rewards.feet_slide.params["sensor_names"] = sensor_names
        self.rewards.feet_slide.params["asset_cfg"].geom_names = [
            r"^(left|right)_foot_collision$"
        ]

        self.rewards.feet_too_near.weight = -0.1
        self.rewards.feet_too_near.params["threshold"] = 0.2
        self.rewards.feet_too_near.params["asset_cfg"].geom_names = [
            r"^(left|right)_foot_collision$"
        ]

        self.rewards.contact_forces.weight = -0.0002
        self.rewards.contact_forces.params["threshold"] = 500
        self.rewards.contact_forces.params["sensor_names"] = sensor_names
        self.rewards.contact_forces.params["asset_cfg"].geom_names = [
            r"^(left|right)_foot_collision$"
        ]

        self.rewards.electrical_power.weight = -0.0001
        self.rewards.electrical_power.params["asset_cfg"].joint_names = {
            ".*_hip_z_slider",
            ".*_hip_xy_slider_l",
            ".*_hip_xy_slider_r",
            ".*_ankle_xy_slider_l",
            ".*_ankle_xy_slider_r",
            ".*_leg_length_slider$",
        }

        self.rewards.base_height.weight = -1.0
        self.rewards.base_height.params["target_height"] = (
            robot_cfg.init_state.pos[2]
        )

        self.curriculum.command_vel = None

        # ViewerConfig
        self.viewer.body_name = "baselink"


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
