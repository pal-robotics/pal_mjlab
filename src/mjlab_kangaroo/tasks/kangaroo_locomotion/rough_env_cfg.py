from dataclasses import dataclass, replace

from mjlab_kangaroo.robots.pal_kangaroo.kangaroo_constants import (
    KANG_ACTION_SCALE,
    KANG_ROBOT_CFG,
)
from mjlab_kangaroo.tasks.kangaroo_locomotion.velocity_env_cfg import (
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
            # r"^leg_(left|right)_(?:knee|femur|length)_joint$",
            r".*leg_(left|right)_(2|length)_joint.*",
            r".*leg_(left|right)_(1|3|4|5)_joint.*",
            r".*(pelvis_(1|2)_joint|arm_(left|right)_(1|4)_joint).*",
            r".*leg_(left|right)_(femur|knee)_joint.*",
            r".*arm_(left|right)_(2|3)_joint.*",
        }
        self.rewards.pose.params["std"] = {
            # r"^leg_(left|right)_(?:knee|femur|length)_joint$": 6.0,
            r".*leg_(left|right)_(2|length)_joint.*": 6.0,
            r".*leg_(left|right)_(1|3|4|5)_joint.*": 3.0,
            r".*(pelvis_(1|2)_joint|arm_(left|right)_(1|4)_joint).*": 1.0,
            r".*leg_(left|right)_(femur|knee)_joint.*": 4.0,
            r".*arm_(left|right)_(2|3)_joint.*": 0.3,
        }

        self.rewards.air_time = None

        self.viewer.body_name = "pelvis_2_link"


@dataclass
class KangRoughEnvCfg_PLAY(KangRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        if self.scene.terrain is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
                self.scene.terrain.terrain_generator.num_cols = 5
                self.scene.terrain.terrain_generator.num_rows = 5
                self.scene.terrain.terrain_generator.border_width = 10.0
