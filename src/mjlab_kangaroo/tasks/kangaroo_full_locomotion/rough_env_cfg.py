from dataclasses import dataclass, replace

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab_kangaroo.robots import (
    KANG_FULL_ROBOT_CFG,
    KANG_FULL_LINEAR_ACTUATORS,
    KANG_FULL_REVOLUTE_ACTUATORS,
    KANG_FULL_ACTION_SCALE,
)
from mjlab_kangaroo.tasks.kangaroo_full_locomotion.velocity_env_cfg import (
    LocomotionVelocityEnvCfg,
)


@dataclass
class KangFullRoughEnvCfg(LocomotionVelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.entities = {"robot": replace(KANG_FULL_ROBOT_CFG)}

        # ActionCfg
        self.actions.joint_pos.actuator_names = (
            KANG_FULL_LINEAR_ACTUATORS + KANG_FULL_REVOLUTE_ACTUATORS
        )
        self.actions.joint_pos.scale = KANG_FULL_ACTION_SCALE

        # ObservationCfg
        self.observations.policy.joint_pos.params = {
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=KANG_FULL_LINEAR_ACTUATORS
                + KANG_FULL_REVOLUTE_ACTUATORS,
            )
        }
        self.observations.policy.joint_vel.params = {
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=KANG_FULL_LINEAR_ACTUATORS
                + KANG_FULL_REVOLUTE_ACTUATORS,
            )
        }
        self.observations.critic.joint_pos.params = {
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=KANG_FULL_LINEAR_ACTUATORS
                + KANG_FULL_REVOLUTE_ACTUATORS,
            )
        }

        self.observations.critic.joint_vel.params = {
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=KANG_FULL_LINEAR_ACTUATORS
                + KANG_FULL_REVOLUTE_ACTUATORS,
            )
        }

        # CommandsCfg
        self.commands.twist.rel_standing_envs = 0.1
        self.commands.twist.viz.z_offset = 0.8

        # EventCfg
        self.events.foot_friction.params["asset_cfg"].geom_names = [
            r"^(left|right)_foot_collision$"
        ]

        # RewardCfg
        self.rewards.air_time = None
        self.rewards.pose.params["asset_cfg"].joint_names = {
            r".*(pelvis_(1|2)_joint|arm_(left|right)_(1|4)_joint).*",
            r".*arm_(left|right)_(2|3)_joint.*",
        }
        self.rewards.pose.params["std"] = {
            r".*(pelvis_(1|2)_joint|arm_(left|right)_(1|4)_joint).*": 1.0,
            r".*arm_(left|right)_(2|3)_joint.*": 0.3,
        }

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
