"""PAL Robotics KANGAROO velocity tracking environment configurations."""

from copy import deepcopy

from pal_mjlab.robots import (
    KANGAROO_ACTION_SCALE,
    get_kangaroo_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import create_velocity_env_cfg
from mjlab.utils.retval import retval

import mjlab.tasks.velocity.mdp as mdp
from mjlab.managers.manager_term_config import TerminationTermCfg

@retval
def PAL_KANGAROO_ROUGH_ENV_CFG() -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics KANGAROO rough terrain velocity tracking configuration."""
    site_names = ("left_foot", "right_foot")
    geom_names = ("left_foot_collision", "right_foot_collision")
    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(leg_left_5_link|leg_right_5_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    body_ground_cfg = ContactSensorCfg(
        name="body_ground_contact",
        primary=ContactMatch(
            mode="body",
            pattern=r"^(leg_left_femur_link|leg_right_femur_link|leg_left_knee_link|leg_right_knee_link)$",
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
        robot_cfg=get_kangaroo_robot_cfg(),
        action_scale=KANGAROO_ACTION_SCALE,
        viewer_body_name="pelvis_2_link",
        site_names=site_names,
        feet_sensor_cfg=feet_ground_cfg,
        self_collision_sensor_cfg=self_collision_cfg,
        foot_friction_geom_names=geom_names,
        posture_std_standing={
            # Lower body.
            r"leg_.*_1_.*": 0.05,
            r"leg_.*_2_.*": 0.05,
            r"leg_.*_3_.*": 0.05,
            r"leg_.*_length_.*": 0.05,
            r"leg_.*_4_.*": 0.05,
            r"leg_.*_5_.*": 0.05,
            # Waist.
            r"pelvis_.*": 0.05,
            # Arms.
            r"arm_.*": 0.05,
        },
        posture_std_walking={
            # Lower body.
            r"leg_.*_1_.*": 0.15,
            r"leg_.*_2_.*": 0.3,  # pitch
            r"leg_.*_3_.*": 0.15,
            r"leg_.*_length_.*": 0.07,  # length
            r"leg_.*_4_.*": 0.25,
            r"leg_.*_5_.*": 0.1,
            # Waist.
            r"pelvis_1.*": 0.1,
            r"pelvis_2.*": 0.2,
            # Arms.
            r"arm_.*_1_.*": 0.15,  # pitch
            r"arm_.*_2_.*": 0.15,  # roll
            r"arm_.*_3_.*": 0.1,
            r"arm_.*_4_.*": 0.15,
        },
        posture_std_running={
            # Lower body.
            r"leg_.*_1_.*": 0.2,
            r"leg_.*_2_.*": 0.5,
            r"leg_.*_3_.*": 0.2,
            r"leg_.*_length_.*": 0.1,
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
        },
        body_ang_vel_weight=-0.05,
        angular_momentum_weight=-0.02,
        self_collision_weight=-1.0,
        air_time_weight=0.0,
    )
    assert cfg.commands is not None
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 1.15
    assert cfg.rewards["pose"].params["asset_cfg"].joint_names
    cfg.rewards["pose"].params["asset_cfg"].joint_names = {
        # Lower body.
        r"leg_.*_1_.*",
        r"leg_.*_2_.*",
        r"leg_.*_3_.*",
        r"leg_.*_length_.*",
        r"leg_.*_4_.*",
        r"leg_.*_5_.*",
        # Waist.
        r"pelvis_.*",
        # Arms.
        r"arm_.*",
    }
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, body_ground_cfg)

    cfg.terminations["illegal_contacts"] = TerminationTermCfg(
      func=mdp.illegal_contact,
      params={"sensor_name": "body_ground_contact"},
    )
    return cfg


@retval
def PAL_KANGAROO_FLAT_ENV_CFG() -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics KANGAROO flat terrain velocity tracking configuration."""
    # Start with rough terrain config.
    cfg = deepcopy(PAL_KANGAROO_ROUGH_ENV_CFG)

    # Change to flat terrain.
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Disable terrain curriculum.
    assert cfg.curriculum is not None
    assert "terrain_levels" in cfg.curriculum
    del cfg.curriculum["terrain_levels"]

    return cfg
