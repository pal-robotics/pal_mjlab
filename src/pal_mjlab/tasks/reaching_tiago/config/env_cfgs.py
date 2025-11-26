from pal_mjlab.robots import (
    get_tiago_robot_cfg,
    KANGAROO_ACTION_SCALE,
)

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from pal_mjlab.tasks.reaching_tiago.reaching_env_cfg import make_reaching_env_cfg

def pal_tiago_reaching_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics TIAGo reaching configuration."""
    cfg = make_reaching_env_cfg()

    cfg.scene.entities = {"robot": get_tiago_robot_cfg()}

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 0.5  # TIAGO_PRO_ACTION_SCALE

    cfg.commands["pose_command_left"].ranges.pos_x = (0.4, 0.8)
    cfg.commands["pose_command_left"].ranges.pos_y = (-0.5, 0.5)
    cfg.commands["pose_command_left"].ranges.pos_z = (0.1, 0.9)
    cfg.commands["pose_command_left"].ranges.roll = (-3.14, 3.14)
    cfg.commands["pose_command_left"].ranges.pitch = (-3.14/2, 3.14/2)
    cfg.commands["pose_command_left"].ranges.yaw = (-3.14, 3.14)\
    
    cfg.commands["pose_command_right"].ranges.pos_x = (0.4, 0.8)
    cfg.commands["pose_command_right"].ranges.pos_y = (-0.5, 0.5)
    cfg.commands["pose_command_right"].ranges.pos_z = (0.1, 0.9)
    cfg.commands["pose_command_right"].ranges.roll = (-3.14, 3.14)
    cfg.commands["pose_command_right"].ranges.pitch = (-3.14/2, 3.14/2)
    cfg.commands["pose_command_right"].ranges.yaw = (-3.14, 3.14)

    return cfg
