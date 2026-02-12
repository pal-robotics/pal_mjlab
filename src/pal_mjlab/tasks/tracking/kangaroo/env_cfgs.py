"""PAL Robotics Talos flat terrain tracking configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.managers.observation_manager import ObservationTermCfg
from pal_mjlab.tasks.velocity.kangaroo import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from pal_mjlab.robots import KANGAROO_ACTION_SCALE, get_kangaroo_robot_cfg, KANGAROO_ACTUATOR_NAMES

def pal_kangaroo_flat_tracking_env_cfg(
    has_state_estimation: bool = True,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics Talos flat terrain tracking configuration."""
    cfg = make_tracking_env_cfg()

    cfg.scene.entities = {"robot": get_kangaroo_robot_cfg()}
    cfg.sim.mujoco.timestep = 0.002
    cfg.decimation = 10

    site_names = ("left_foot", "right_foot")
    geom_names = tuple(
        f"{side}_foot{i}_collision"
        for side in ("left", "right")
        for i in [0, 2, 4, 6, 8, 10]
    )

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
    joint_pos_action.scale = KANGAROO_ACTION_SCALE
    joint_pos_action.actuator_names = KANGAROO_ACTUATOR_NAMES

    assert cfg.commands is not None
    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.anchor_body_name = "base_link"
    motion_cmd.body_names = (
        "base_link",
        "pelvis_2_link",
        "leg_left_3_link",
        "leg_left_4_link",
        "leg_left_5_link",
        "leg_right_3_link",
        "leg_right_4_link",
        "leg_right_5_link",
        "arm_left_2_link",
        "arm_left_3_link",
        "arm_left_5_link",
        "arm_right_2_link",
        "arm_right_3_link",
        "arm_right_5_link",
    )

    # cfg.observations["actor"].terms["base_lin_vel"] = None
    # cfg.observations["actor"].terms["projected_gravity"] = None
    # cfg.observations["actor"].terms["imu_projected_gravity"] = ObservationTermCfg(
    #     func=mdp.imu_projected_gravity,
    #     params={"sensor_name": "robot/imu_quat"},
    #     noise=Unoise(n_min=-0.5, n_max=0.5),
    # )
    # cfg.observations["actor"].terms["base_lin_acc"] = ObservationTermCfg(
    #     func=mdp.builtin_sensor,
    #     params={"sensor_name": "robot/imu_lin_acc"},
    #     noise=Unoise(n_min=-0.5, n_max=0.5),
    # )
    # cfg.observations["critic"].terms["imu_projected_gravity"] = ObservationTermCfg(
    #     func=mdp.imu_projected_gravity,
    #     params={"sensor_name": "robot/imu_quat"},
    # )
    # cfg.observations["critic"].terms["base_lin_acc"] = ObservationTermCfg(
    #     func=mdp.builtin_sensor,
    #     params={"sensor_name": "robot/imu_lin_acc"},
    # )

    cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
    cfg.events["base_com"].params["asset_cfg"].body_names = ("pelvis_2_link",)

    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "leg_left_5_link",
        "leg_right_5_link",
        "arm_left_5_link",
        "arm_right_5_link",
    )

    cfg.viewer.body_name = "base_link"

    # Modify observations if we don't have state estimation.
    if not has_state_estimation:
        new_actor_terms = {
            k: v
            for k, v in cfg.observations["actor"].terms.items()
            # I added motion_anchor_ori_b but might not be necessary, 
            # and i wonder if i should add lin acc when state 
            # estimation is false
            if k not in ["motion_anchor_pos_b", "motion_anchor_ori_b", "base_lin_vel"]
        }
        cfg.observations["actor"] = ObservationGroupCfg(
            terms=new_actor_terms,
            concatenate_terms=True,
            enable_corruption=True,
        )

    # Apply play mode overrides.
    if play:
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)

        # Disable RSI randomization.
        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}

        motion_cmd.sampling_mode = "start"

    return cfg
