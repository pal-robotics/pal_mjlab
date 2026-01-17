"""PAL Robotics KANGAROO recovery tracking environment configurations."""

from pal_mjlab.robots import (
    get_kangaroo_robot_cfg,
    get_kangaroo_hands_robot_cfg,
    get_kangaroo_grippers_robot_cfg,
    KANGAROO_ACTION_SCALE,
    KANGAROO_HANDS_ACTION_SCALE,
    KANGAROO_GRIPPERS_ACTION_SCALE,
    KANGAROO_ACTUATOR_NAMES,
    KANGAROO_HANDS_ACTUATOR_NAMES,
    KANGAROO_GRIPPERS_ACTUATOR_NAMES,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from pal_mjlab.tasks.recovery import mdp
from pal_mjlab.tasks.recovery.recovery_env_cfg import make_recovery_env_cfg
from mjlab.managers.manager_term_config import TerminationTermCfg, EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.manager_term_config import (
    ObservationTermCfg,
)
from mjlab.utils.noise import UniformNoiseCfg as Unoise


def pal_kangaroo_flat_recovery_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics KANGAROO rough terrain recovery configuration."""
    cfg = make_recovery_env_cfg()
    cfg.scene.entities = {"robot": get_kangaroo_robot_cfg()}
    cfg.sim.nconmax = 45
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
    geom_names = tuple(
        f"{side}_foot{i}_collision"
        for side in ("left", "right")
        for i in [0, 2, 4, 6, 8, 10]
    )

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(leg_left_5_link|leg_right_5_link)$",  # subtree so foot link is included
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
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg)

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = KANGAROO_ACTION_SCALE
    joint_pos_action.actuator_names = KANGAROO_ACTUATOR_NAMES

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )
    # cfg.rewards["power"].params["asset_cfg"].joint_names = ("leg_.*(1|2|3|4|5|length)_joint", "arm.*", "pelvis_.*")
    cfg.rewards["joint_vel_hinge"].params["asset_cfg"].joint_names = ("leg_.*(1|2|3|4|5|knee|femur)_joint", "arm.*", "pelvis_.*")



    # cfg.rewards["posture"].params["asset_cfg"].joint_names = actuated_joints
    # cfg.rewards["posture"].params["std_standing"] = {
    #     # Lower body.
    #     r"leg_.*_1_.*": 0.05,
    #     r"leg_.*_2_.*": 0.05,
    #     r"leg_.*_3_.*": 0.05,
    #     r"leg_.*_length_.*": 0.05,
    #     r"leg_.*_4_.*": 0.05,
    #     r"leg_.*_5_.*": 0.05,
    #     # Waist.
    #     r"pelvis_.*": 0.05,
    #     # Arms.
    #     r"arm_.*": 0.05,
    # }
    # cfg.rewards["posture"].params["std_rising"] = {
    #     # Lower body.
    #     r"leg_.*_1_.*": 0.5,
    #     r"leg_.*_2_.*": 0.6,  # pitch
    #     r"leg_.*_3_.*": 0.5,
    #     r"leg_.*_length_.*": 0.7,  # length
    #     r"leg_.*_4_.*": 0.5,
    #     r"leg_.*_5_.*": 0.5,
    #     # Waist.
    #     r"pelvis_1.*": 0.08,
    #     r"pelvis_2.*": 0.2,
    #     # Arms.
    #     r"arm_.*_1_.*": 0.5,  # pitch
    #     r"arm_.*_4_.*": 0.5,
    #     r"arm_.*_(?![14]_joint)\d+_joint": 0.5,
    # }
    # cfg.rewards["posture"].params["std_fallen"] = {
    #     # Lower body.
    #     r"leg_.*_1_.*": 1.0,
    #     r"leg_.*_2_.*": 1.2,
    #     r"leg_.*_3_.*": 1.0,
    #     r"leg_.*_length_.*": 1.5,
    #     r"leg_.*_4_.*": 1.0,
    #     r"leg_.*_5_.*": 1.0,
    #     # Waist.
    #     r"pelvis_1.*": 1.0,
    #     r"pelvis_2.*": 1.0,
    #     # Arms.
    #     r"arm_.*_1_.*": 1.0,
    #     r"arm_.*_4_.*": 1.0,
    #     r"arm_.*_(?![14]_joint)\d+_joint": 1.0,
    # }







    cfg.viewer.body_name = "pelvis_2_link"

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names

    # Apply play mode overrides.
    if play:
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        cfg.observations["policy"].enable_corruption = False

    return cfg
