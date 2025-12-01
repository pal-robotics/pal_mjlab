from pal_mjlab.robots import (
    get_tiago_robot_cfg,
    KANGAROO_ACTION_SCALE,
)

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from pal_mjlab.tasks.reaching_tiago import mdp
from pal_mjlab.tasks.reaching_tiago.reaching_env_cfg import make_reaching_env_cfg
import mujoco

# ADD cube 
def get_cube_spec(cube_size: float = 0.02, mass: float = 0.05) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(cube_size,) * 3,
    mass=mass,
    rgba=(0.8, 0.2, 0.2, 1.0),
  )
  return spec

def pal_tiago_reaching_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics TIAGo reaching configuration."""
    cfg = make_reaching_env_cfg()

    cfg.scene.entities = {"robot": get_tiago_robot_cfg()}

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 0.5  # TIAGO_PRO_ACTION_SCALE

    cfg.commands["pose_command_left"].ranges.pos_x = (0.1, 0.8)
    cfg.commands["pose_command_left"].ranges.pos_y = (-0.2, 0.5)
    cfg.commands["pose_command_left"].ranges.pos_z = (0.2, 1.0)
    cfg.commands["pose_command_left"].ranges.roll = (-3.14, 3.14)
    cfg.commands["pose_command_left"].ranges.pitch = (-3.14/2, 3.14/2)
    cfg.commands["pose_command_left"].ranges.yaw = (-3.14, 3.14)
    
    cfg.commands["pose_command_right"].ranges.pos_x = (0.1, 0.8)
    cfg.commands["pose_command_right"].ranges.pos_y = (-0.5, 0.2)
    cfg.commands["pose_command_right"].ranges.pos_z = (0.2, 1.0)
    cfg.commands["pose_command_right"].ranges.roll = (-3.14, 3.14)
    cfg.commands["pose_command_right"].ranges.pitch = (-3.14/2, 3.14/2)
    cfg.commands["pose_command_right"].ranges.yaw = (-3.14, 3.14)

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="base_footprint", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="base_footprint", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    ee_ground_collision_cfg = ContactSensorCfg(  
        name="ee_ground_collision",
        primary=ContactMatch(
            mode="subtree",
            pattern="ee_.*",      
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="body",
            pattern="terrain",
        ),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    cfg.scene.sensors = (self_collision_cfg,ee_ground_collision_cfg,)

    cfg.rewards["stand_still_joint_deviation_l1"].params["asset_cfg"].joint_names = (
        r"torso_lift_joint",
    )

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )
    return cfg
