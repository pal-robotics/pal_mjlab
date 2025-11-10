"""PAL Robotics Talos velocity tracking environment configurations."""

from copy import deepcopy

from pal_mjlab.robots import (
    TALOS_ACTION_SCALE,
    get_talos_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import create_velocity_env_cfg
from mjlab.utils.retval import retval

import mjlab.tasks.velocity.mdp as mdp
from mjlab.managers.manager_term_config import TerminationTermCfg

@retval
def PAL_TALOS_ROUGH_ENV_CFG() -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics Talos rough terrain velocity tracking configuration."""
    site_names = ("left_foot", "right_foot")
    geom_names = ("left_foot_collision", "right_foot_collision")
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
    # TODO Louis: Adds illegal contact termination, otherwise Talos is walking on his knees
    body_ground_cfg = ContactSensorCfg(
        name="body_ground_contact",
        primary=ContactMatch(
            mode="body",
            pattern=r"^(leg_left_4_link|leg_right_4_link|torso_2_link|arm_left_7_link|arm_right_7_link|arm_left_5_link|arm_right_5_link|)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg = create_velocity_env_cfg(
        robot_cfg=get_talos_robot_cfg(),
        action_scale=TALOS_ACTION_SCALE,
        viewer_body_name="torso_2_link",
        site_names=site_names,
        feet_sensor_cfg=feet_ground_cfg,
        self_collision_sensor_cfg=self_collision_cfg,
        foot_friction_geom_names=geom_names,
        posture_std_standing={".*": 0.05},
        posture_std_walking={
            # Lower body.
            r"leg_.*_3_.*": 0.3,  # pitch
            r"leg_.*_2_.*": 0.15,  # roll
            r"leg_.*_1_.*": 0.15,
            r"leg_.*_4_.*": 0.35,  # knee
            r"leg_.*_5_.*": 0.25,
            r"leg_.*_6_.*": 0.1,
            # Waist.
            r".*torso_2.*": 0.1,  # pitch
            r".*torso_1.*": 0.2,  # yaw
            r".*head.*": 0.1,
            # Arms.
            r"arm_.*_1_.*": 0.15,  # yaw
            r"arm_.*_2_.*": 0.15,  # roll
            r"arm_.*_3_.*": 0.1,  # yaw
            r"arm_.*_4_.*": 0.15,  # elbow
            r"arm_.*_5_.*": 0.1,  # elbow
            r"arm_.*_6_.*": 0.1,  # wrist
            r"arm_.*_7_.*": 0.2,  # wrist
        },
        posture_std_running={
            # Lower body.
            r"leg_.*_3_.*": 0.5,  # pitch
            r"leg_.*_2_.*": 0.2,  # roll
            r"leg_.*_1_.*": 0.2,
            r"leg_.*_4_.*": 0.6,
            r"leg_.*_5_.*": 0.35,
            r"leg_.*_6_.*": 0.15,
            # Waist.
            r".*torso_2.*": 0.2,  # pitch
            r".*torso_1.*": 0.3,  # yaw
            r".*head.*": 0.1,
            # Arms.
            r"arm_.*_1_.*": 0.2,  # yaw
            r"arm_.*_2_.*": 0.2,  # roll
            r"arm_.*_3_.*": 0.1,  # yaw
            r"arm_.*_4_.*": 0.35,  # elbow
            r"arm_.*_5_.*": 0.1,  # elbow
            r"arm_.*_6_.*": 0.1,  # wrist
            r"arm_.*_7_.*": 0.2,  # wrist
        },
        body_ang_vel_weight=-0.05,
        angular_momentum_weight=-0.02,
        self_collision_weight=-1.0,
        air_time_weight=0.0,
    )
    assert cfg.commands is not None
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 1.25
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, body_ground_cfg)

    cfg.terminations["illegal_contacts"] = TerminationTermCfg(
      func=mdp.illegal_contact,
      params={"sensor_name": "body_ground_contact"},
    )
    return cfg


@retval
def PAL_TALOS_FLAT_ENV_CFG() -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics Talos flat terrain velocity tracking configuration."""
    # Start with rough terrain config.
    cfg = deepcopy(PAL_TALOS_ROUGH_ENV_CFG)

    # Change to flat terrain.
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Disable terrain curriculum.
    assert cfg.curriculum is not None
    assert "terrain_levels" in cfg.curriculum
    del cfg.curriculum["terrain_levels"]

    return cfg
