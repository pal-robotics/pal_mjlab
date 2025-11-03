from dataclasses import dataclass, replace

from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
    KANG_ACTION_SCALE,
    KANG_ROBOT_CFG,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.velocity_env_cfg import (
  LocomotionVelocityEnvCfg,
)

@dataclass
class KangRoughEnvCfg(LocomotionVelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.entities = {"robot": replace(KANG_ROBOT_CFG)}

        # constants
        geom_names = ["left_foot_collision", "right_foot_collision"]
        site_names = ["left_foot", "right_foot"]
        target_foot_height = 0.15

        # sensors
        feet_ground_cfg = ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(
                mode="body",
                pattern=r"^(leg_left_5_link|leg_right_5_link)$",
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
            primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            fields=("found",),
            reduce="none",
            num_slots=1,
        )
        # scene
        self.scene.sensors = (feet_ground_cfg, self_collision_cfg,)

        # actions
        self.actions.joint_pos.scale = KANG_ACTION_SCALE

        self.events.foot_friction.params["asset_cfg"].geom_names = geom_names

        # rewards
        self.rewards.upright.params["asset_cfg"].body_names = ["pelvis_2_link"]
        self.rewards.pose.params["asset_cfg"].joint_names = {
            # Lower body.
            r"leg_.*_1_.*",
            r"leg_.*_2_.*",
            r"leg_.*_3_.*",
            r"leg_.*_knee_.*",
            r"leg_.*_4_.*",
            r"leg_.*_5_.*",
            # Waist.
            r"pelvis_.*",
            # Arms.
            r"arm_.*",
        }
        self.rewards.pose.params["std_standing"] = {
            # Lower body.
            r"leg_.*_1_.*": 0.05,
            r"leg_.*_2_.*": 0.05,
            r"leg_.*_3_.*": 0.05,
            r"leg_.*_knee_.*": 0.05,
            r"leg_.*_4_.*": 0.05,
            r"leg_.*_5_.*": 0.05,
            # Waist.
            r"pelvis_.*": 0.05,
            # Arms.
            r"arm_.*": 0.05,
        }
        self.rewards.pose.params["std_walking"] = {
            # Lower body.
            r"leg_.*_1_.*": 0.15,
            r"leg_.*_2_.*": 0.3, # pitch
            r"leg_.*_3_.*": 0.15,
            r"leg_.*_knee_.*": 0.35, # knee
            r"leg_.*_4_.*": 0.25,
            r"leg_.*_5_.*": 0.1,
            # Waist.
            r"pelvis_1.*": 0.1,
            r"pelvis_2.*": 0.2,
            # Arms.
            r"arm_.*_1_.*": 0.15, # pitch
            r"arm_.*_2_.*": 0.15, # roll
            r"arm_.*_3_.*": 0.1,
            r"arm_.*_4_.*": 0.15,
        }
        self.rewards.pose.params["std_running"] = {
            # Lower body.
            r"leg_.*_1_.*": 0.2,
            r"leg_.*_2_.*": 0.5,
            r"leg_.*_3_.*": 0.2,
            r"leg_.*_knee_.*": 0.6,
            r"leg_.*_4_.*": 0.35,
            r"leg_.*_5_.*": 0.15,
            # Waist.
            r"pelvis_1.*": 0.2,
            r"pelvis_2.*": 0.3,
            # Arms.
            r"arm_.*_1_.*": 0.2,
            r"arm_.*_2_.*": 0.2,
            r"arm_.*_3_.*": 0.1,
            r"arm_.*_4_.*": 0.35,
        }
        self.rewards.foot_clearance.params["asset_cfg"].site_names = site_names
        self.rewards.foot_swing_height.params["asset_cfg"].site_names = site_names
        self.rewards.foot_slip.params["asset_cfg"].site_names = site_names
        self.rewards.foot_swing_height.params["target_height"] = target_foot_height
        self.rewards.foot_clearance.params["target_height"] = target_foot_height
        self.rewards.body_ang_vel.params["asset_cfg"].body_names = ["pelvis_2_link"]
        self.rewards.track_angular_velocity.weight = 1.0

        # observations
        self.observations.critic.foot_height.params["asset_cfg"].site_names = site_names

        # terminations
        self.terminations.illegal_contact = None
        # self.terminations.illegal_contact.params["sensor_name"] = "body_ground_contact"

        self.viewer.body_name = "base_link"
        self.commands.twist.viz.z_offset = 1.5


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
