"""PAL Robotics REEM-C velocity tracking environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from pal_mjlab.robots import REEMC_ACTION_SCALE, get_reemc_robot_cfg


def pal_reemc_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics REEM-C rough terrain velocity tracking configuration."""
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": get_reemc_robot_cfg()}

    site_names = ("left_foot_sole", "right_foot_sole")
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
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, body_ground_cfg)

    if (
        cfg.scene.terrain is not None
        and cfg.scene.terrain.terrain_generator is not None
    ):
        cfg.scene.terrain.terrain_generator.curriculum = True

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    # Increase action scales significantly to allow much larger movements
    # This enables larger steps and less frequent stepping
    increased_scales = {}
    for joint_name, scale in REEMC_ACTION_SCALE.items():
        if "leg_" in joint_name:
            # Leg joints: increase by 4x to allow much larger steps
            # This enables longer strides and higher step clearance
            increased_scales[joint_name] = scale * 4.0
        elif "arm_" in joint_name or "torso_" in joint_name:
            # Arms and torso: increase by 2x
            increased_scales[joint_name] = scale * 2.0
        else:
            # Head and others: increase by 1.5x
            increased_scales[joint_name] = scale * 1.5
    joint_pos_action.scale = increased_scales

    cfg.viewer.body_name = "torso_2_link"

    assert cfg.commands is not None
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.viz.z_offset = 1.15

    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = site_names

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names

    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = {
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
    }
    cfg.rewards["pose"].params["std_running"] = {
        # Lower body - significantly increased std to allow much larger leg movements
        # This enables longer strides and higher steps
        r"leg_.*_3_.*": 0.8,  # pitch (increased for larger hip movements)
        r"leg_.*_2_.*": 0.35,  # roll (increased for wider stance)
        r"leg_.*_1_.*": 0.35,  # (increased for hip yaw)
        r"leg_.*_4_.*": 0.9,  # knee (increased for larger knee flexion)
        r"leg_.*_5_.*": 0.5,  # (increased for ankle)
        r"leg_.*_6_.*": 0.3,  # (increased for ankle roll)
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
    }

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("torso_2_link",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("torso_2_link",)

    for reward_name in ["foot_clearance", "foot_swing_height", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    
    # Encourage longer steps with higher air time reward
    # Higher weight encourages longer strides (more time in air = longer step)
    cfg.rewards["air_time"].weight = 0.5

    # Increase tracking rewards to prioritize velocity tracking
    cfg.rewards["track_linear_velocity"].weight = 10.0
    cfg.rewards["track_angular_velocity"].weight = 10.0
    
    # Make linear velocity tracking less strict to improve forward tracking
    # Increased std helps with forward movement where robot struggles to step
    cfg.rewards["track_linear_velocity"].params["std"] = 1.0  # Increased from 0.5 to help forward stepping
    
    # Further reduce action rate penalty to allow slower, larger movements
    # This encourages less frequent but larger steps
    cfg.rewards["action_rate_l2"].weight = -0.02  # Reduced to allow slower stepping
    
    # Encourage much higher foot swing for larger, more visible steps
    cfg.rewards["foot_swing_height"].weight = -0.05  # Further reduced penalty
    cfg.rewards["foot_swing_height"].params["target_height"] = 0.18  # Significantly higher target for larger steps

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
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


def pal_reemc_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create PAL REEM-C flat terrain velocity configuration."""
    cfg = pal_reemc_rough_env_cfg(play=play)

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
