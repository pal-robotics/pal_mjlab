from dataclasses import dataclass, replace

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.spec_config import ContactSensorCfg

from mjlab_kangaroo.robots import (
    KANG_FULL_ROBOT_CFG,
    KANG_FULL_LINEAR_ACTUATORS,
    KANG_FULL_REVOLUTE_ACTUATORS,
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
                data=("found",),
                reduce="netforce",
            )
            for side in ["left", "right"]
        ]
        g1_cfg = replace(
            KANG_FULL_ROBOT_CFG, sensors=tuple(foot_contact_sensors)
        )
        self.scene.entities = {"robot": g1_cfg}

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
        self.actions.joint_pos.scale = 0.2

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
        self.commands.twist.rel_standing_envs = 1.0
        self.commands.twist.viz.z_offset = 0.75

        # EventCfg
        self.events.foot_friction.params["asset_cfg"].geom_names = [
            r"^(left|right)_foot_collision$"
        ]
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.push_robot = None

        # RewardCfg
        self.rewards.track_lin_vel_exp.weight = 2.0
        self.rewards.track_ang_vel_exp.weight = 1.0
        self.rewards.air_time = None
        # self.rewards.air_time.weight = 2.0
        # self.rewards.air_time.params["sensor_names"] = sensor_names
        self.rewards.foot_clearance = None
        # self.rewards.foot_clearance.params["target_height"] = 0.20
        # self.rewards.foot_clearance.params["std"] = 0.02
        # self.rewards.foot_clearance.params["tanh_mult"] = 1.0
        # self.rewards.foot_clearance.params["asset_cfg"].geom_names = [
        #     r"^(left|right)_foot_collision$"
        # ]

        self.rewards.dof_pos_limits = None
        self.rewards.posture = None
        # self.rewards.posture.params["asset_cfg"].joint_names = {
        #     r".*(pelvis_(1|2)_joint|arm_(left|right)_(1|4)_joint).*",
        #     r".*arm_(left|right)_(2|3)_joint.*",
        # }
        # self.rewards.posture.params["std"] = {
        #     r".*(pelvis_(1|2)_joint|arm_(left|right)_(1|4)_joint).*": 1.0,
        #     r".*arm_(left|right)_(2|3)_joint.*": 0.3,
        # }
        # self.rewards.posture.weight = -0.1
        self.rewards.termination_penalty.weight = -200.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.feet_slide = None
        # self.rewards.feet_slide.weight = -0.1
        # self.rewards.feet_slide.params["sensor_names"] = sensor_names
        # self.rewards.feet_slide.params["asset_cfg"].geom_names = [
        #     r"^(left|right)_foot_collision$"
        # ]

        # CurriculumCfg
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
