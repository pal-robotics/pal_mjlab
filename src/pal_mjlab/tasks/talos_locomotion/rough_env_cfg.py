from dataclasses import dataclass, replace

from pal_mjlab.robots import (
    TALOS_ACTION_SCALE,
    TALOS_ROBOT_CFG,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.velocity_env_cfg import (
  LocomotionVelocityEnvCfg,
)


@dataclass
class PalTalosRoughEnvCfg(LocomotionVelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.entities = {"robot": replace(TALOS_ROBOT_CFG)}

        # constants
        geom_names = ["left_foot_collision", "right_foot_collision"]
        site_names = ["left_foot", "right_foot"]
        target_foot_height = 0.15

        # sensors
        feet_ground_cfg = ContactSensorCfg(
            name="feet_ground_contact",
            primary=ContactMatch(
                mode="subtree",
                pattern=r"^(leg_left_6_link|leg_right_6_link)$",
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
            primary=ContactMatch(mode="subtree", pattern="torso_2_link", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="torso_2_link", entity="robot"),
            fields=("found",),
            reduce="none",
            num_slots=1,
        )

        # scene
        self.scene.sensors = (feet_ground_cfg, self_collision_cfg)

        # actions
        self.actions.joint_pos.scale = TALOS_ACTION_SCALE

        # events
        self.events.foot_friction.params["asset_cfg"].geom_names = geom_names

        # rewards
        self.rewards.upright.params["asset_cfg"].body_names = ["torso_2_link"]
        # Tight control when stationary: maintain stable default pose.
        # TODO LOUIS: Find what joint has been added ????
        self.rewards.pose.params["asset_cfg"].joint_names = {
            # Lower body.
            r"leg_.*_3_.*", # pitch
            r"leg_.*_2_.*", # roll
            r"leg_.*_1_.*",
            r"leg_.*_4_.*",
            r"leg_.*_5_.*",
            r"leg_.*_6_.*",
            # Waist.
            r".*torso_2.*", # pitch
            r".*torso_1.*", # yaw
            r".*head.*",
            # Arms.
            r"arm_.*_1_.*", # yaw
            r"arm_.*_2_.*", # roll
            r"arm_.*_3_.*", # yaw
            r"arm_.*_4_.*", # elbow
            r"arm_.*_5_.*", # elbow
            r"arm_.*_6_.*", # wrist
            r"arm_.*_7_.*", # wrist
        }
        self.rewards.pose.params["std_standing"] = {
            # Lower body.
            r"leg_.*_3_.*": 0.05, # pitch
            r"leg_.*_2_.*": 0.05, # roll
            r"leg_.*_1_.*": 0.05,
            r"leg_.*_4_.*": 0.05,
            r"leg_.*_5_.*": 0.05,
            r"leg_.*_6_.*": 0.05,
            # Waist.
            r".*torso_2.*": 0.05, # pitch
            r".*torso_1.*": 0.05, # yaw
            r".*head.*": 0.05,
            # Arms.
            r"arm_.*_1_.*": 0.05, # yaw
            r"arm_.*_2_.*": 0.05, # roll
            r"arm_.*_3_.*": 0.05, # yaw
            r"arm_.*_4_.*": 0.05, # elbow
            r"arm_.*_5_.*": 0.05, # elbow
            r"arm_.*_6_.*": 0.05, # wrist
            r"arm_.*_7_.*": 0.05, # wrist
        }
        # Moderate leg freedom for stepping, loose arms for natural pendulum swing.
        self.rewards.pose.params["std_walking"] = {
            # Lower body.
            r"leg_.*_3_.*": 0.3, # pitch
            r"leg_.*_2_.*": 0.15, # roll
            r"leg_.*_1_.*": 0.15,
            r"leg_.*_4_.*": 0.35,
            r"leg_.*_5_.*": 0.25,
            r"leg_.*_6_.*": 0.1,
            # Waist.
            r".*torso_2.*": 0.1, # pitch
            r".*torso_1.*": 0.2, # yaw
            r".*head.*": 0.1,
            # Arms.
            r"arm_.*_1_.*": 0.15, # yaw
            r"arm_.*_2_.*": 0.15, # roll
            r"arm_.*_3_.*": 0.1, # yaw
            r"arm_.*_4_.*": 0.15, # elbow
            r"arm_.*_5_.*": 0.15, # elbow
            r"arm_.*_6_.*": 0.3, # wrist
            r"arm_.*_7_.*": 0.3, # wrist
        }
        self.rewards.pose.params["std_running"] = {
            # Lower body.
            r"leg_.*_3_.*": 0.5, # pitch
            r"leg_.*_2_.*": 0.2, # roll
            r"leg_.*_1_.*": 0.2,
            r"leg_.*_4_.*": 0.6,
            r"leg_.*_5_.*": 0.35,
            r"leg_.*_6_.*": 0.15,
            # Waist.
            r".*torso_2.*": 0.2, # pitch
            r".*torso_1.*": 0.3, # yaw
            r".*head.*": 0.1,
            # Arms.
            r"arm_.*_1_.*": 0.5, # yaw
            r"arm_.*_2_.*": 0.2, # roll
            r"arm_.*_3_.*": 0.15, # yaw
            r"arm_.*_4_.*": 0.35, # elbow
            r"arm_.*_5_.*": 0.35, # elbow
            r"arm_.*_6_.*": 0.3, # wrist
            r"arm_.*_7_.*": 0.3, # wrist
        }
        self.rewards.foot_clearance.params["asset_cfg"].site_names = site_names
        self.rewards.foot_swing_height.params["asset_cfg"].site_names = site_names
        self.rewards.foot_slip.params["asset_cfg"].site_names = site_names
        self.rewards.foot_swing_height.params["target_height"] = target_foot_height
        self.rewards.foot_clearance.params["target_height"] = target_foot_height
        self.rewards.body_ang_vel.params["asset_cfg"].body_names = ["torso_2_link"]

        # observations
        self.observations.critic.foot_height.params["asset_cfg"].site_names = site_names

        # terminations
        self.terminations.illegal_contact = None

        self.viewer.body_name = "base_link"
        self.commands.twist.viz.z_offset = 1.15


@dataclass
class PalTalosRoughEnvCfg_PLAY(PalTalosRoughEnvCfg):
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
