from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from pal_mjlab.robots import (
    get_tiago_pro_robot_cfg,
)
from pal_mjlab.tasks.reaching import mdp
from pal_mjlab.tasks.reaching.reaching_env_cfg import make_reaching_env_cfg


def pal_tiago_pro_reaching_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics TIAGo Pro reaching configuration."""
    cfg = make_reaching_env_cfg()
    cfg.viewer.body_name = "base_footprint"

    cfg.scene.entities = {"robot": get_tiago_pro_robot_cfg()}

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 0.5  # TIAGO_PRO_ACTION_SCALE

    cfg.commands["pose_command_left"].ranges.pos_x = (0.1, 0.8)
    cfg.commands["pose_command_left"].ranges.pos_y = (-0.2, 0.5)
    cfg.commands["pose_command_left"].ranges.pos_z = (0.2, 1.0)
    cfg.commands["pose_command_left"].ranges.roll = (-3.14, 3.14)
    cfg.commands["pose_command_left"].ranges.pitch = (-3.14 / 2, 3.14 / 2)
    cfg.commands["pose_command_left"].ranges.yaw = (-3.14, 3.14)

    cfg.commands["pose_command_right"].ranges.pos_x = (0.1, 0.8)
    cfg.commands["pose_command_right"].ranges.pos_y = (-0.5, 0.2)
    cfg.commands["pose_command_right"].ranges.pos_z = (0.2, 1.0)
    cfg.commands["pose_command_right"].ranges.roll = (-3.14, 3.14)
    cfg.commands["pose_command_right"].ranges.pitch = (-3.14 / 2, 3.14 / 2)
    cfg.commands["pose_command_right"].ranges.yaw = (-3.14, 3.14)

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="base_footprint", entity="robot"),
        secondary=ContactMatch(
            mode="subtree", pattern="base_footprint", entity="robot"
        ),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )
    cfg.scene.sensors = (self_collision_cfg,)

    cfg.rewards["stand_still_joint_deviation_l1"] = (
        RewardTermCfg(
            func=mdp.stand_still_joint_deviation_l1,
            weight=-5.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=("torso_lift_joint",),
                )
            },
        )
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )
    return cfg
