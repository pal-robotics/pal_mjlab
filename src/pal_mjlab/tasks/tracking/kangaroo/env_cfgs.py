"""PAL Robotics Talos flat terrain tracking configuration."""

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from pal_mjlab.robots import (
  KANGAROO_ACTION_SCALE,
  KANGAROO_ACTUATOR_NAMES,
  get_kangaroo_robot_cfg,
)
from pal_mjlab.tasks.velocity.kangaroo import mdp


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

  body_geoms = (
    # # Femur
    "leg_left_femur_collision",
    "leg_right_femur_collision",
    # Knee
    "leg_left_knee_collision",
    "leg_left_knee_bar_collision",
    "leg_right_knee_collision",
    "leg_right_knee_bar_collision",
    # Arms (4 only — arm 3 has no collision geom)
    "arm_left_4_collision",
    "arm_right_4_collision",
    # Pelvis
    "pelvis_2_collision",
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

  # The hull points should correspond to the respective joints defined in the joint_names_group order
  # leg_*_2_joint corresponds to Hip Pitch and leg_*_3_joint corresponds to Hip roll
  cfg.rewards["convex_hull_joint_limits_hip"] = RewardTermCfg(
    func=mdp.joint_limits_convex_hull,
    weight=-10.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
      "metrics_suffix": "hipXY",
      "joint_names_group": [
        [r"leg_left_2_joint", r"leg_left_3_joint"],
        [r"leg_right_2_joint", r"leg_right_3_joint"],
      ],
      "margin": 0.0,
      "hull_points": torch.tensor(
        [
          [-0.59341, -0.26180],
          [-0.59341, -0.17453],
          [-0.59341, -0.08727],
          [-0.74176, 0.00000],
          [-0.59341, 0.08727],
          [-0.59341, 0.17453],
          [-0.59341, 0.26180],
          [-0.44506, 0.34907],
          [-0.44506, 0.43633],
          [-0.29671, 0.43633],
          [-0.14835, 0.43633],
          [0.00000, 0.43633],
          [0.14835, 0.43633],
          [0.29671, 0.43633],
          [0.44506, 0.43633],
          [0.44506, 0.34907],
          [0.59341, 0.26180],
          [0.59341, 0.17453],
          [0.59341, 0.08727],
          [0.74176, 0.00000],
          [0.59341, -0.08727],
          [0.59341, -0.17453],
          [0.59341, -0.26180],
          [0.44506, -0.34907],
          [0.44506, -0.43633],
          [0.29671, -0.43633],
          [0.14835, -0.43633],
          [0.00000, -0.43633],
          [-0.14835, -0.43633],
          [-0.29671, -0.43633],
          [-0.44506, -0.43633],
          [-0.44506, -0.43633],
        ]
      ),
    },
  )

  cfg.rewards["convex_hull_joint_limits_ankle"] = RewardTermCfg(
    func=mdp.joint_limits_convex_hull,
    weight=-10.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
      "margin": 0.0,
      "metrics_suffix": "ankleXY",
      "joint_names_group": [
        [r"leg_left_4_joint", r"leg_left_5_joint"],
        [r"leg_right_4_joint", r"leg_right_5_joint"],
      ],
      "hull_points": torch.tensor(
        [
          [0.707, 0.005],
          [0.648, 0.112],
          [0.576, 0.23],
          [0.484, 0.38],
          [0.443, 0.439],
          [0.266, 0.443],
          [0.008, 0.441],
          [-0.293, 0.45],
          [-0.46, 0.448],
          [-0.505, 0.379],
          [-0.594, 0.244],
          [-0.686, 0.098],
          [-0.744, 0.001],
          [-0.688, -0.099],
          [-0.604, -0.231],
          [-0.499, -0.394],
          [-0.445, -0.472],
          [-0.254, -0.469],
          [0.005, -0.462],
          [0.232, -0.456],
          [0.429, -0.46],
          [0.475, -0.382],
          [0.583, -0.207],
          [0.665, -0.071],
        ]
      ),
    },
  )

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["body_friction"] = EventTermCfg(
    mode="startup",
    func=mdp.randomize_field,
    domain_randomization=True,
    params={
      "asset_cfg": SceneEntityCfg("robot", geom_names=body_geoms),  # Set per-robot.
      "operation": "abs",
      "field": "geom_friction",
      "ranges": (0.3, 2.0),
      "shared_random": False,  # All foot geoms share the same friction.
    },
  )

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
