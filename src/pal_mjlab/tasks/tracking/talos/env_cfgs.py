"""PAL Robotics Talos flat terrain tracking configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from pal_mjlab.robots import TALOS_ACTION_SCALE, get_talos_robot_cfg


def pal_talos_flat_tracking_env_cfg(
    has_state_estimation: bool = True,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics Talos flat terrain tracking configuration."""
    cfg = make_tracking_env_cfg()

    cfg.scene.entities = {"robot": get_talos_robot_cfg()}

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = (self_collision_cfg,)

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = TALOS_ACTION_SCALE

    assert cfg.commands is not None
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.anchor_body_name = "base_link"
    motion_cmd.body_names = (
        "base_link",
        "torso_2_link",
        "leg_left_3_link",
        "leg_left_4_link",
        "leg_left_6_link",
        "leg_right_3_link",
        "leg_right_4_link",
        "leg_right_6_link",
        "arm_left_3_link",
        "arm_left_4_link",
        "arm_left_7_link",
        "arm_right_3_link",
        "arm_right_4_link",
        "arm_right_7_link",
    )

    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = r"^(left|right)_foot_collision$"
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "leg_left_6_link",
        "leg_right_6_link",
        "arm_left_7_link",
        "arm_right_7_link",
    )

    cfg.viewer.body_name = "base_link"

    # Modify observations if we don't have state estimation.
    if not has_state_estimation:
        new_policy_terms = {
            k: v
            for k, v in cfg.observations["policy"].terms.items()
            if k not in ["motion_anchor_pos_b", "base_lin_vel"]
        }
        cfg.observations["policy"] = ObservationGroupCfg(
            terms=new_policy_terms,
            concatenate_terms=True,
            enable_corruption=True,
        )

    # Apply play mode overrides.
    if play:
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        cfg.observations["policy"].enable_corruption = False
        cfg.events.pop("push_robot", None)

        # Disable RSI randomization.
        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}

        motion_cmd.sampling_mode = "start"

    return cfg
