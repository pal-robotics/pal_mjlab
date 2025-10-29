from dataclasses import dataclass, replace

from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
    KANG_ACTION_SCALE,
    KANG_ROBOT_CFG,
)
from mjlab.tasks.velocity.velocity_env_cfg import (
    LocomotionVelocityEnvCfg,
    RewardCfg
)
from mjlab.utils.spec_config import ContactSensorCfg

@dataclass
class KangRoughEnvCfg(LocomotionVelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        foot_contact_sensors = [
            ContactSensorCfg(
                name=f"{side}_foot_ground_contact",
                body1=f"leg_{side}_5_link",
                body2="terrain",
                num=1,
                data=("found",),
                reduce="netforce",
            )
            for side in ["left", "right"]
        ]
        kangaroo_cfg = replace(KANG_ROBOT_CFG, sensors=tuple(foot_contact_sensors))
        self.scene.entities = {"robot": kangaroo_cfg}

        self.actions.joint_pos.scale = KANG_ACTION_SCALE

        sensor_names = ["left_foot_ground_contact", "right_foot_ground_contact"]
        self.events.foot_friction.params["asset_cfg"].geom_names = [
            "left_foot_collision", "right_foot_collision"
        ]


        self.rewards.pose.params["asset_cfg"].joint_names = {
            # Lower body.
            r"leg_.*_1_.*",
            r"leg_.*_2_.*",
            r"leg_.*_3_.*",
            r"leg_.*_length_.*",
            r"leg_.*_4_.*",
            r"leg_.*_5_.*",
            # r"leg_.*_femur_.*",
            # r"leg_.*_knee_.*",
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
            # Lower body.
            r"leg_.*_1_.*": 0.15,
            r"leg_.*_2_.*": 0.3,
            r"leg_.*_3_.*": 0.15,
            r"leg_.*_length_.*": 0.1,
            r"leg_.*_4_.*": 0.25,
            r"leg_.*_5_.*": 0.1,
            # r"leg_.*_femur_.*": 0.2,
            # r"leg_.*_knee_.*": 0.2,
            # Waist.
            r".*pelvis_1.*": 0.08,
            r".*pelvis_2.*": 0.15,
            # Arms.
            r"arm_.*_1_.*": 0.35,
            r"arm_.*_2_.*": 0.15,
            r"arm_.*_3_.*": 0.1,
            r"arm_.*_4_.*": 0.25,
        }

        self.rewards.action_rate_l2.weight = -0.01
        # self.rewards.air_time.weight = 1.0

        self.rewards.air_time = None

        self.viewer.body_name = "base_link"
        self.commands.twist.viz.z_offset = 0.75

        self.curriculum.command_vel = None

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
