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
                mode="body",
                pattern=r"^(leg_left_6_link|leg_right_6_link)$",
                entity="robot",
            ),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"),
            reduce="netforce",
            num_slots=1,
            track_air_time=True,
        )
        # body_ground_cfg = ContactSensorCfg(
        #     name="body_ground_contact",
        #     primary=ContactMatch(
        #         mode="body",
        #         pattern=r"^(leg_left_4_link|leg_right_4_link|torso_2_link|arm_left_7_link|arm_right_7_link|arm_left_5_link|arm_right_5_link|)$",
        #         entity="robot",
        #     ),
        #     secondary=ContactMatch(mode="body", pattern="terrain"),
        #     fields=("found",),
        #     reduce="none",
        #     num_slots=1,
        # )
        self_collision_cfg = ContactSensorCfg(
            name="self_collision",
            primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
            fields=("found",),
            reduce="none",
            num_slots=1,
        )
        # scene
        self.scene.sensors = (feet_ground_cfg, self_collision_cfg) #, body_ground_cfg)

        # actions
        self.actions.joint_pos.scale = TALOS_ACTION_SCALE

        # events
        self.events.foot_friction.params["asset_cfg"].geom_names = geom_names

        # rewards
        self.rewards.upright.params["asset_cfg"].body_names = ["torso_2_link"]
        # Tight control when stationary: maintain stable default pose.
        self.rewards.pose.params["std_standing"] = {".*": 0.05,}
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
            r"arm_.*_5_.*": 0.1, # elbow
            r"arm_.*_6_.*": 0.1, # wrist
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
            r"arm_.*_5_.*": 0.1, # elbow
            r"arm_.*_6_.*": 0.1, # wrist
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
        # self.terminations.illegal_contact.params["sensor_name"] = "body_ground_contact"

        self.viewer.body_name = "base_link"
        self.commands.twist.viz.z_offset = 1.5


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
