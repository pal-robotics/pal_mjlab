from dataclasses import dataclass, replace

from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
    KANG_ACTION_SCALE,
    KANG_ROBOT_CFG,
)
from mjlab.tasks.velocity.velocity_env_cfg import (
    LocomotionVelocityEnvCfg,
)


@dataclass
class KangRoughEnvCfg(LocomotionVelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.entities = {"robot": replace(KANG_ROBOT_CFG)}

        self.actions.joint_pos.scale = KANG_ACTION_SCALE

        self.events.foot_friction.params["asset_cfg"].geom_names = [
            r"^(left|right)_foot_collision$"
        ]

        self.rewards.pose.params["asset_cfg"].joint_names = {
            # # r"^leg_(left|right)_(?:knee|femur|length)_joint$",
            # r".*leg_(left|right)_(2|length)_joint.*",
            # r".*leg_(left|right)_(1|3|4|5)_joint.*",
            # r".*(pelvis_(1|2)_joint|arm_(left|right)_(1|4)_joint).*",
            # r".*leg_(left|right)_(femur|knee)_joint.*",
            # r".*arm_(left|right)_(2|3)_joint.*",

            r"leg_.*_1_.*",
            r"leg_.*_2_.*",
            r"leg_.*_3_.*",
            r"leg_.*_length_.*",
            r"leg_.*_4_.*",
            r"leg_.*_5_.*",
            # Waist.
            r".*pelvis_2.*",
            r".*pelvis_1.*",
            # Arms.
            r"arm_.*_1_.*",
            r"arm_.*_2_.*",
            r"arm_.*_3_.*",
            r"arm_.*_4_.*",
        }
        self.rewards.pose.params["std"] = {
            # # r"^leg_(left|right)_(?:knee|femur|length)_joint$": 6.0,
            # r".*leg_(left|right)_(2|length)_joint.*": 6.0,
            # r".*leg_(left|right)_(1|3|4|5)_joint.*": 3.0,
            # r".*(pelvis_(1|2)_joint|arm_(left|right)_(1|4)_joint).*": 1.0,
            # r".*leg_(left|right)_(femur|knee)_joint.*": 4.0,
            # r".*arm_(left|right)_(2|3)_joint.*": 0.3,

            # Lower body.
            r"leg_.*_1_.*": 0.15,
            r"leg_.*_2_.*": 0.3,
            r"leg_.*_3_.*": 0.15,
            r"leg_.*_length_.*": 3.5,
            r"leg_.*_4_.*": 0.25,
            r"leg_.*_5_.*": 0.1,
            # Waist.
            r".*pelvis_2.*": 0.15,
            r".*pelvis_1.*": 0.1,
            # Arms.
            r"arm_.*_1_.*": 0.35,
            r"arm_.*_2_.*": 0.15,
            r"arm_.*_3_.*": 0.1,
            r"arm_.*_4_.*": 0.25,
        }

        self.rewards.air_time = None
        # self.rewards.pose = None
        self.rewards.dof_pos_limits = None
        self.rewards.action_rate_l2 = None
        self.rewards.air_time = None

        self.viewer.body_name = "base_link"
        self.commands.twist.viz.z_offset = 0.75

        self.curriculum.command_vel = None

# REWARDS IMPLEMENTATION IN VELOCITY_ENV_CFG FILE

#   pose: RewardTerm = term(
#     RewardTerm,
#     func=mdp.posture,
#     weight=1.0,
#     params={
#       "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
#       "std": [],
#     },
#   )
#   dof_pos_limits: RewardTerm = term(RewardTerm, func=mdp.joint_pos_limits, weight=-1.0)
#   action_rate_l2: RewardTerm = term(RewardTerm, func=mdp.action_rate_l2, weight=-0.1)

#   # Unused, only here as an example.
#   air_time: RewardTerm = term(
#     RewardTerm,
#     func=mdp.feet_air_time,
#     weight=0.0,
#     params={
#       "asset_name": "robot",
#       "threshold_min": 0.05,
#       "threshold_max": 0.15,
#       "command_name": "twist",
#       "command_threshold": 0.05,
#       "sensor_names": [],
#       "reward_mode": "on_landing",
#     },
#   )

@dataclass
class KangRoughEnvCfg_PLAY(KangRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Effectively infinite episode length.
        self.episode_length_s = int(1e9)

        if self.scene.terrain is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
                self.scene.terrain.terrain_generator.num_cols = 5
                self.scene.terrain.terrain_generator.num_rows = 5
                self.scene.terrain.terrain_generator.border_width = 10.0
