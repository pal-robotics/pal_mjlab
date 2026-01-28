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
from mjlab.sensor import ContactMatch, ContactSensorCfg
from pal_mjlab.tasks.recovery import mdp
from pal_mjlab.tasks.recovery.recovery_env_cfg import make_recovery_env_cfg
# from mjlab.managers.event_manager import EventTermCfg
# from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
# from mjlab.managers.scene_entity_config import SceneEntityCfg
# from mjlab.managers.termination_manager import TerminationTermCfg
# from mjlab.utils.noise import UniformNoiseCfg as Unoise


def pal_kangaroo_flat_recovery_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics KANGAROO rough terrain recovery configuration."""
    cfg = make_recovery_env_cfg()
    cfg.scene.entities = {"robot": get_kangaroo_robot_cfg()}
    cfg.sim.nconmax = None  # Use default max contacts.
    
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
    body_ground_cfg = ContactSensorCfg(
        name="body_ground_contact",
        primary=ContactMatch(
            mode="body",
            pattern=r"^(leg_left_femur_link|leg_right_femur_link|leg_left_knee_link|leg_right_knee_link|pelvis_2_link|leg_left_1_link|leg_right_1_link|arm_right_4_link|arm_left_4_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = (feet_ground_cfg, self_collision_cfg, body_ground_cfg)

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = KANGAROO_ACTION_SCALE
    joint_pos_action.actuator_names = KANGAROO_ACTUATOR_NAMES

    cfg.rewards["self_collisions"].params["sensor_name"] = "self_collision"
    cfg.rewards["terrain_collisions"].params["sensor_name"] = "body_ground_contact"
    # cfg.rewards["power"].params["asset_cfg"].joint_names = ("leg_.*(1|2|3|4|5|length)_joint", "arm.*", "pelvis_.*")
    cfg.rewards["joint_vel_hinge"].params["asset_cfg"].joint_names = ("leg_.*(1|2|3|4|5|knee|femur)_joint", "arm.*", "pelvis_.*")



    cfg.rewards["posture"].params["asset_cfg"].joint_names = actuated_joints
    cfg.rewards["posture"].params["std_standing"] = {
        r"leg_.*_1_.*": 0.05,       # Hip roll: tight (~11°)
        r"leg_.*_2_.*": 0.05,      # Hip pitch: slightly looser (~14°)
        r"leg_.*_3_.*": 0.05,       # Hip roll: tight
        r"leg_.*_length_.*": 0.05, # Leg length: prefer default (~8cm variation)
        r"leg_.*_4_.*": 0.05,       # Ankle pitch: tight
        r"leg_.*_5_.*": 0.05,       # Ankle roll: tight
        r"pelvis_1.*": 0.05,       # Waist roll: very tight (~9°)
        r"pelvis_2.*": 0.05,        # Waist pitch: tight (~11°)
        r"arm_.*_1_.*": 0.05,       # Shoulder pitch: moderate (balance)
        r"arm_.*_4_.*": 0.05,       # Elbow: moderate (balance)
        r"arm_.*_(?![14]_joint)\d+_joint": 0.25,  # Other arm joints
    }
    cfg.rewards["posture"].params["std_rising"] = {
        r"leg_.*_1_.*": 0.6,       # Hip roll: moderate
        r"leg_.*_2_.*": 0.8,       # Hip pitch: more freedom (important for rising)
        r"leg_.*_3_.*": 0.6,       # Hip roll: moderate
        r"leg_.*_length_.*": 0.15, # Leg length: prefer extension
        r"leg_.*_4_.*": 0.6,       # Ankle pitch: moderate
        r"leg_.*_5_.*": 0.6,       # Ankle roll: moderate
        r"pelvis_1.*": 0.3,        # Waist roll: tighter
        r"pelvis_2.*": 0.5,        # Waist pitch: some freedom for rising motion
        r"arm_.*": 0.8,            # Arms: still flexible
    }
    cfg.rewards["posture"].params["std_fallen"] = {
        r"leg_.*_1_.*": 1.5,       # Hip roll: very free
        r"leg_.*_2_.*": 1.5,       # Hip pitch: very free
        r"leg_.*_3_.*": 1.5,       # Hip roll: very free
        r"leg_.*_length_.*": 0.3,  # Leg length: allow compression
        r"leg_.*_4_.*": 1.5,       # Ankle pitch: very free
        r"leg_.*_5_.*": 1.5,       # Ankle roll: very free
        r"pelvis_1.*": 1.2,        # Waist roll: free
        r"pelvis_2.*": 1.2,        # Waist pitch: free
        r"arm_.*": 1.5,            # Arms: very free
    }







    cfg.viewer.body_name = "pelvis_2_link"

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names

    # Apply play mode overrides.
    if play:
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        cfg.observations["policy"].enable_corruption = False

    return cfg
