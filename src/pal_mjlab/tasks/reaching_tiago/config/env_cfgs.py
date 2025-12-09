from pal_mjlab.robots import (
    get_tiago_robot_cfg,
)

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.entity import EntityCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from pal_mjlab.tasks.reaching_tiago import mdp
from pal_mjlab.tasks.reaching_tiago.reaching_env_cfg import make_reaching_env_cfg
from pal_mjlab.tasks.reaching_tiago.mdp import LiftingCommandCfg
import mujoco
import random

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
    friction=(random.uniform(0.3, 1.2) , 0.005, 0.0001),
  )
  return spec

def pal_tiago_reaching_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create PAL Robotics TIAGo reaching configuration."""
    cfg = make_reaching_env_cfg()

    cfg.scene.entities = {
       "robot": get_tiago_robot_cfg(),
        "cube": EntityCfg(spec_fn=get_cube_spec),}

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 0.5  # TIAGO_PRO_ACTION_SCALE

    assert cfg.commands is not None
    lift_command = cfg.commands["lift_height"]
    assert isinstance(lift_command, LiftingCommandCfg)

    cfg.observations["policy"].terms["ee_to_cube"].params["asset_cfg"].site_names = (
    "ee_right"
    )
    cfg.rewards["ee_object_distance"].params["asset_cfg"].site_names = ("ee_right")

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
            pattern="arm_.*_7_link",      
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

    left_fingertip_collision_cfg = ContactSensorCfg(  
        name="left_fingertip_cube_collision",
        primary=ContactMatch(
            mode="subtree",
            pattern="gripper_right_fingertip_left_link",      
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="body",
            pattern="cube",
            entity="cube",
        ),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    right_fingertip_collision_cfg = ContactSensorCfg(  
        name="right_fingertip_block_collision",
        primary=ContactMatch(
            mode="subtree",
            pattern="gripper_right_fingertip_right_link",      
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="body",
            pattern="cube",
            entity="cube",
        ),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )


    cfg.scene.sensors = (self_collision_cfg,ee_ground_collision_cfg,right_fingertip_collision_cfg,left_fingertip_collision_cfg)

    cfg.rewards["stand_still_joint_deviation_l1"].params["asset_cfg"].joint_names = (
        # r"torso_lift_joint",
        r"arm_left_.*_joint",
    )

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name},
    )

    cfg.rewards["fingertips_grasp_cube"] = RewardTermCfg(
    func=mdp.fingertips_grasp_binary,
    weight=2.0,
    params={
        "left_sensor_name": left_fingertip_collision_cfg.name,
        "right_sensor_name": right_fingertip_collision_cfg.name,
    },
)

    return cfg
