"""PAL Robotics KANGAROO velocity tracking environment configurations."""

from pal_mjlab.robots import (
    KANGAROO_ACTION_SCALE,
    get_kangaroo_robot_cfg,
    KANGAROO_ACTUATOR_NAMES,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from pal_mjlab.tasks.kangaroo_locomotion import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.managers.manager_term_config import TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.manager_term_config import (
    CurriculumTermCfg,
    ObservationTermCfg,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise


def pal_kangaroo_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics KANGAROO rough terrain velocity configuration."""
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": get_kangaroo_robot_cfg()}

    cfg.sim.nconmax = 45

    site_names = ("left_foot", "right_foot")
    geom_names = tuple(
        f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(0, 10)
    )
    actuated_joints = (
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
    )

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
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, body_ground_cfg)

    if (
        cfg.scene.terrain is not None
        and cfg.scene.terrain.terrain_generator is not None
    ):
        cfg.scene.terrain.terrain_generator.curriculum = True

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = KANGAROO_ACTION_SCALE
    joint_pos_action.actuator_names = KANGAROO_ACTUATOR_NAMES

    cfg.viewer.body_name = "pelvis_2_link"

    assert cfg.commands is not None
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 1.15

    cfg.observations["policy"].terms["base_lin_vel"] = None
    cfg.observations["policy"].terms["projected_gravity"] = None
    cfg.observations["policy"].terms["base_lin_acc"] = ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_lin_acc"},
        noise=Unoise(n_min=-0.5, n_max=0.5),
    )
    cfg.observations["critic"].terms["base_lin_acc"] = ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_lin_acc"},
    )
    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = site_names

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names

    cfg.rewards["pose"].params["asset_cfg"].joint_names = actuated_joints
    cfg.rewards["pose"].params["std_standing"] = {
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
    }
    cfg.rewards["pose"].params["std_walking"] = {
        # Lower body.
        r"leg_.*_1_.*": 0.15,
        r"leg_.*_2_.*": 0.3,  # pitch
        r"leg_.*_3_.*": 0.15,
        r"leg_.*_length_.*": 0.15,  # length
        r"leg_.*_4_.*": 0.25,
        r"leg_.*_5_.*": 0.1,
        # Waist.
        r"pelvis_1.*": 0.08,
        r"pelvis_2.*": 0.2,
        # Arms.
        r"arm_.*_1_.*": 0.2,  # pitch
        r"arm_.*_2_.*": 0.1,  # roll
        r"arm_.*_3_.*": 0.1,
        r"arm_.*_4_.*": 0.2,
    }
    cfg.rewards["pose"].params["std_running"] = {
        # Lower body.
        r"leg_.*_1_.*": 0.2,
        r"leg_.*_2_.*": 0.5,
        r"leg_.*_3_.*": 0.2,
        r"leg_.*_length_.*": 0.25,
        r"leg_.*_4_.*": 0.35,
        r"leg_.*_5_.*": 0.15,
        # Waist.
        r"pelvis_1.*": 0.08,
        r"pelvis_2.*": 0.3,
        # Arms.
        r"arm_.*_1_.*": 0.4,
        r"arm_.*_2_.*": 0.15,
        r"arm_.*_3_.*": 0.15,
        r"arm_.*_4_.*": 0.35,
    }

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("pelvis_2_link",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("pelvis_2_link",)

    for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["air_time"].weight = 0.25

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )
    cfg.rewards["power"] = RewardTermCfg(
        func=mdp.electrical_power_cost,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=actuated_joints)},
    )

    # cfg.rewards["self_collisions"] = None
    # cfg.rewards["air_time"] = None
    # cfg.rewards["angular_momentum"] = None
    # cfg.curriculum["air_time"] = CurriculumTermCfg(
    #   func=mdp.reward_weight,
    #   params={
    #     "reward_name": "air_time",
    #     "weight_stages": [
    #       {"step": 0, "weight": 0.25},
    #       {"step": 5000 * 24, "weight": 1.0},
    #     #   {"step": 10_000 * 24, "weight": 2.0},
    #     ],
    #   },
    # )
    cfg.curriculum["power"] = CurriculumTermCfg(
        func=mdp.reward_weight,
        params={
            "reward_name": "power",
            "weight_stages": [
                {"step": 0, "weight": 0.0},
                {"step": 5000 * 24, "weight": -0.01},
                {"step": 10000 * 24, "weight": -0.1},
            ],
        },
    )
    cfg.terminations["illegal_contacts"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": "body_ground_contact"},
    )

    # Apply play mode overrides.
    if play:
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        cfg.observations["policy"].enable_corruption = False
        cfg.events.pop("push_robot", None)

        if cfg.scene.terrain is not None:
            if cfg.scene.terrain.terrain_generator is not None:
                cfg.scene.terrain.terrain_generator.curriculum = False
                cfg.scene.terrain.terrain_generator.num_cols = 5
                cfg.scene.terrain.terrain_generator.num_rows = 5
                cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg


def pal_kangaroo_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics KANGAROO flat terrain velocity configuration."""
    cfg = pal_kangaroo_rough_env_cfg(play=play)

    # Switch to flat terrain.
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Disable terrain curriculum.
    assert cfg.curriculum is not None
    assert "terrain_levels" in cfg.curriculum
    del cfg.curriculum["terrain_levels"]

    if play:
        commands = cfg.commands
        assert commands is not None
        twist_cmd = commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
        twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

    return cfg
