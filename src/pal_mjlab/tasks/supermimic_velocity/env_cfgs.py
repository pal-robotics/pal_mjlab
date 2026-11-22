"""PAL Kangaroo flat-ground velocity tracking, trained with SuperMimic.

Built from the PPO task's ``pal_kangaroo_flat_env_cfg``: the robot, simulation, plane,
actions, velocity command and its curriculum, resets, domain randomization and
terminations are its own. What SuperMimic needs changes (``rl/supermimic/ROADMAP.md``,
M5 v1):

- the env config is a ``SuperMimicEnvCfg`` (``imagination``) and does not auto-reset;
- heading control is off (the yaw rate is commanded directly) and pushes are removed (the
  world model cannot foresee them yet);
- the observation groups are the policy's goal (the command), its proprioception (the
  PPO actor's terms and noise) and the critic's input (heading state, command, ξ);
- every reward term is a single definition (``mdp/rewards.py``, the PPO terms' formulas
  and weights); the contact-based ones the batch cannot express are left out:
  ``self_collisions``, ``angular_momentum`` (the passive linkage is not in the state);
- ξ also holds the randomized joint friction, damping and armature;
- the failures imagination can compute are single definitions (tilt, and a collapsed
  root, a new termination), so imagined episodes end where the simulator's would.
"""

from __future__ import annotations

import dataclasses

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from pal_mjlab.rl.supermimic import terms as sm_terms
from pal_mjlab.rl.supermimic.names import (
  ImaginationCfg,
  SuperMimicEnvCfg,
  as_supermimic_env_cfg,
)
from pal_mjlab.robots import REGEX_ALL_ACTUATED_JOINTS, kangaroo_passive_leg_joints
from pal_mjlab.tasks.supermimic_velocity import mdp
from pal_mjlab.tasks.velocity.kangaroo.env_cfgs import pal_kangaroo_flat_env_cfg

FEET = ("left_foot", "right_foot")
TORSO = "pelvis_2_link"
KNEE_LINKAGE = tuple(
  f"leg_{side}_{link}_link" for side in ("left", "right") for link in ("femur", "knee")
)
"""The passive linkage whose contact with the ground ends an episode."""


# Velocity tracking, retuned from the PPO task's (weight 2, std 0.5 / 0.71) for a gradient
# method: with std 0.5 the kernel is flat at a standing robot's errors (0.8 m/s: 0.08, about
# no slope), and standing already earned ~5 of the ~7 reward per step (roadmap M5). A wider
# kernel keeps the gradient toward the command alive; a heavier linear term makes moving pay.
TRACK_LIN_WEIGHT = 4.0
TRACK_LIN_STD = 1.0
TRACK_YAW_STD = 1.0
# A narrow kernel next to the wide one: the wide one alone barely rewards precision (a
# 0.5 m/s error still earns 78% of it); a policy that walks drifted off the command.
TRACK_LIN_FINE_WEIGHT = 2.0
TRACK_LIN_FINE_STD = 0.25
# The gait terms, retuned for a gradient method: air time with soft edges (mjlab's hard
# window gives no gradient) and doubled (PPO: 0.25). Foot clearance is PPO's (-2): lowered
# to -0.5 to make the first steps cheap, it let a walking policy drag a foot.
AIR_TIME_WEIGHT = 0.5
FOOT_CLEARANCE_WEIGHT = -2.0
# Posture, rebalanced against the heavier tracking (8 per step; PPO: tracking 4 vs pose 1):
# the policy tracked in odd postures. The pose kernel loses its gradient far from the
# default pose, so a quadratic penalty (its exponent, whose gradient grows with the
# deviation) pulls back from there.
POSE_WEIGHT = 2.0
POSTURE_L2_WEIGHT = -0.1


def pal_kangaroo_flat_supermimic_env_cfg(play: bool = False) -> SuperMimicEnvCfg:
  """The PPO flat velocity task of the base Kangaroo, for SuperMimic."""
  cfg = pal_kangaroo_flat_env_cfg(play=play)
  actuated = SceneEntityCfg("robot", joint_names=(REGEX_ALL_ACTUATED_JOINTS,))
  feet = SceneEntityCfg("robot", site_names=FEET)

  # -- Command: no heading control.
  twist = cfg.commands["twist"]
  fields = {f.name: getattr(twist, f.name) for f in dataclasses.fields(twist) if f.init}
  fields.update(heading_command=False, rel_heading_envs=0.0)
  fields["ranges"] = dataclasses.replace(twist.ranges, heading=None)
  cfg.commands["twist"] = mdp.SuperMimicVelocityCommandCfg(**fields)

  # -- Events: no pushes; SuperMimic resets finished episodes itself.
  cfg.events.pop("push_robot", None)
  cfg.auto_reset = play

  # -- Metrics: the PPO config's torque metric asks for every joint's actuator, and the
  # passive knee linkage has none (the PPO task fails to build on it too).
  cfg.metrics["joint_torque_mag"].params["asset_cfg"] = actuated

  # -- Observations.
  actor = cfg.observations["actor"].terms
  proprio = {
    "base_ang_vel": ObservationTermCfg(
      func=sm_terms.site_ang_vel, params={"site": "imu"}, noise=actor["base_ang_vel"].noise
    ),
    "imu_projected_gravity": ObservationTermCfg(
      func=sm_terms.site_projected_gravity,
      params={"site": "imu"},
      noise=actor["imu_projected_gravity"].noise,
    ),
    "joint_pos": ObservationTermCfg(
      func=sm_terms.joint_pos_rel,
      params={"biased": True, "asset_cfg": actuated},
      noise=actor["joint_pos"].noise,
    ),
    "joint_vel": ObservationTermCfg(
      func=sm_terms.joint_vel_rel,
      params={"asset_cfg": actuated},
      noise=actor["joint_vel"].noise,
    ),
    "actions": ObservationTermCfg(func=sm_terms.last_action),
  }
  cfg.observations = {
    "goal": ObservationGroupCfg(
      terms={"command": ObservationTermCfg(func=mdp.velocity_command)},
      enable_corruption=False,
    ),
    "proprio": ObservationGroupCfg(
      terms=proprio,
      enable_corruption=cfg.observations["actor"].enable_corruption,
      history_length=1,
    ),
    "critic": ObservationGroupCfg(
      terms={
        "heading_state": ObservationTermCfg(func=sm_terms.heading_state),
        "command": ObservationTermCfg(func=mdp.velocity_command),
        "context": ObservationTermCfg(func=sm_terms.context),
      },
      enable_corruption=False,
    ),
  }

  # -- Rewards: the PPO terms' weights and params, as single definitions.
  old = cfg.rewards

  def term(name: str, func, **params) -> RewardTermCfg:
    return RewardTermCfg(func=func, weight=old[name].weight, params=params)

  p = {n: old[n].params for n in old}
  contact = dict(mdp.FOOT_CONTACT)
  cfg.rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=mdp.track_linear_velocity, weight=TRACK_LIN_WEIGHT, params={"std": TRACK_LIN_STD}
    ),
    "track_linear_velocity_fine": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=TRACK_LIN_FINE_WEIGHT,
      params={"std": TRACK_LIN_FINE_STD},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=old["track_angular_velocity"].weight,
      params={"std": TRACK_YAW_STD},
    ),
    "upright": term(
      "upright", mdp.upright, std=p["upright"]["std"], asset_cfg=p["upright"]["asset_cfg"]
    ),
    "pose": term(
      "pose",
      mdp.variable_posture,
      **{k: v for k, v in p["pose"].items() if k != "command_name"},
    ),
    "body_ang_vel": term(
      "body_ang_vel", mdp.body_angular_velocity_penalty, asset_cfg=p["body_ang_vel"]["asset_cfg"]
    ),
    "dof_pos_limits": term("dof_pos_limits", mdp.joint_pos_limits, asset_cfg=actuated),
    "action_rate_l2": term("action_rate_l2", sm_terms.action_rate_l2),
    **{
      n: term(n, mdp.joint_limits_convex_hull, **p[n])
      for n in ("convex_hull_joint_limits_hip", "convex_hull_joint_limits_ankle")
    },
    "joint_vel_limits": term("joint_vel_limits", mdp.joint_vel_limits, **p["joint_vel_limits"]),
    "air_time": term(
      "air_time",
      mdp.feet_air_time,
      asset_cfg=feet,
      threshold_min=p["air_time"]["threshold_min"],
      threshold_max=p["air_time"]["threshold_max"],
      command_threshold=p["air_time"]["command_threshold"],
      **contact,
    ),
    "foot_clearance": term(
      "foot_clearance",
      mdp.feet_clearance,
      target_height=p["foot_clearance"]["target_height"],
      asset_cfg=feet,
      command_threshold=p["foot_clearance"]["command_threshold"],
      **contact,
    ),
    "foot_swing_height": term(
      "foot_swing_height",
      mdp.feet_swing_height,
      target_height=p["foot_swing_height"]["target_height"],
      asset_cfg=feet,
      command_threshold=p["foot_swing_height"]["command_threshold"],
      **contact,
    ),
    "foot_slip": term(
      "foot_slip",
      mdp.feet_slip,
      asset_cfg=feet,
      command_threshold=p["foot_slip"]["command_threshold"],
      **contact,
    ),
    # The PPO weight scales a landing force (N); here a vertical speed (m/s): ~ a
    # 1000 N landing at 0.5 m/s keeps the same order of magnitude.
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=old["soft_landing"].weight * 2000.0,
      params={
        "asset_cfg": feet,
        "command_threshold": p["soft_landing"]["command_threshold"],
        **contact,
      },
    ),
  }

  cfg.rewards["pose"].weight = POSE_WEIGHT
  cfg.rewards["posture_l2"] = RewardTermCfg(
    func=mdp.posture_l2,
    weight=POSTURE_L2_WEIGHT,
    params=dict(cfg.rewards["pose"].params),
  )
  cfg.rewards["air_time"].weight = AIR_TIME_WEIGHT
  cfg.rewards["foot_clearance"].weight = FOOT_CLEARANCE_WEIGHT

  # -- Terminations: every failure is a single definition, so imagination ends imagined
  # episodes exactly where the simulator does (a learned head can be exploited: a
  # policy learned to kneel when only it covered the knees).
  fell = cfg.terminations["fell_over"]
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation, params={"limit_angle": fell.params["limit_angle"]}
  )
  # A collapsed root (far below any live state: their 1st percentile is ~0.47 m on
  # random-action rollouts), which a predicted state can reach without tipping over.
  cfg.terminations["base_too_low"] = TerminationTermCfg(
    func=mdp.root_height_below, params={"minimum_height": 0.4}
  )
  # The PPO task's contact-sensor test, made geometric: the knee linkage's capsules near
  # the ground (the linkage's passive joints from the closed chain).
  cfg.terminations["illegal_contacts"] = TerminationTermCfg(
    func=mdp.links_touch_ground, params={"bodies": KNEE_LINKAGE}
  )

  # -- SuperMimic: what it reads from the env.
  events = cfg.events
  imagination = ImaginationCfg(
    command="twist",
    failure_terms=("fell_over", "base_too_low", "illegal_contacts"),
    goal_group="goal",
    proprio_group="proprio",
    action_term="joint_pos",
    bodies=(TORSO, *KNEE_LINKAGE, *mdp.FOOT_SOLE_BODIES),
    sites=("imu", *FEET),
    passive_joints=kangaroo_passive_leg_joints,
    context=ObservationTermCfg(
      func=mdp.velocity_context,
      params={
        "friction_asset_cfg": SceneEntityCfg(
          "robot", geom_names=events["foot_friction"].params["asset_cfg"].geom_names
        ),
        "bias_asset_cfg": actuated,
        "com_asset_cfg": SceneEntityCfg(
          "robot", body_names=events["base_com"].params["asset_cfg"].body_names
        ),
        "dof_asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  )
  return as_supermimic_env_cfg(cfg, imagination)
