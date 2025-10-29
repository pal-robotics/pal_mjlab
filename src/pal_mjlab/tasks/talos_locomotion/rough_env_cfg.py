from dataclasses import dataclass, replace

from pal_mjlab.robots import (
    TALOS_ACTION_SCALE,
    TALOS_ROBOT_CFG,
)
from mjlab.tasks.velocity.velocity_env_cfg import (
    LocomotionVelocityEnvCfg,
)
from mjlab.utils.spec_config import ContactSensorCfg


@dataclass
class PalTalosRoughEnvCfg(LocomotionVelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        foot_contact_sensors = [
            ContactSensorCfg(
                name=f"{side}_foot_ground_contact",
                body1=f"leg_{side}_6_link",
                body2="terrain",
                num=1,
                data=("found",),
                reduce="netforce",
            )
            for side in ["left", "right"]
        ]
        talos_cfg = replace(TALOS_ROBOT_CFG, sensors=tuple(foot_contact_sensors))
        self.scene.entities = {"robot": talos_cfg}

        sensor_names = ["left_foot_ground_contact", "right_foot_ground_contact"]
        geom_names = ["left_foot_collision", "right_foot_collision"]

        self.events.foot_friction.params["asset_cfg"].geom_names = geom_names

        self.actions.joint_pos.scale = TALOS_ACTION_SCALE

        self.rewards.air_time.params["sensor_names"] = sensor_names
        # self.rewards.pose.params["std"] = {
        #   r"^(left|right)_knee_joint$": 0.6,
        #   r"^(left|right)_hip_pitch_joint$": 0.6,
        #   r"^(left|right)_elbow_joint$": 0.6,
        #   r"^(left|right)_shoulder_pitch_joint$": 0.6,
        #   r"^(?!.*(knee_joint|hip_pitch|elbow_joint|shoulder_pitch)).*$": 0.3,
        # }
        self.rewards.pose.params["std"] = {
            # Lower body.
            r"leg_.*_3_.*": 0.3,  # pitch
            r"leg_.*_2_.*": 0.15,  # roll
            r"leg_.*_1_.*": 0.15,
            r"leg_.*_4_.*": 0.35,
            r"leg_.*_5_.*": 0.25,
            r"leg_.*_6_.*": 0.1,
            # Waist.
            r".*torso_2.*": 0.15,
            r".*torso_1.*": 0.1,
            r".*head.*": 0.1,
            # Arms.
            r"arm_.*_1_.*": 0.35,  # yaw
            r"arm_.*_2_.*": 0.15,  # roll
            r"arm_.*_3_.*": 0.1,  # yaw
            r"arm_.*_4_.*": 0.25,  # elbow
            r"arm_.*_5_.*": 0.25,  # elbow
            r"arm_.*_6_.*": 0.3,  # wrist
            r"arm_.*_7_.*": 0.3,  # wrist
        }

        self.viewer.body_name = "base_link"
        self.commands.twist.viz.z_offset = 0.75

        self.curriculum.command_vel = None


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
