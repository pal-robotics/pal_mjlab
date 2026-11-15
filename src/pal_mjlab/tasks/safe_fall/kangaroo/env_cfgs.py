"""PAL Robotics KANGAROO safe-fall environment configuration."""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import MetricsTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from pal_mjlab.robots import (
  FALL_COLLISION_MESH_SUFFIX,
  FALL_TERRAIN_COLLISION,
  NATURAL_FREQ,
  REGEX_LEG_LENGTH_JOINTS_ONLY,
  get_kangaroo_fall_robot_cfg,
)
from pal_mjlab.tasks.safe_fall import mdp
from pal_mjlab.tasks.velocity.kangaroo.env_cfgs import pal_kangaroo_flat_env_cfg

# Bodies whose ground contact is measured. The feet are left out: standing on
# them is the goal, and hard landings on them are caught by the actuator
# overload and joint limit terms instead.
REGEX_IMPACT_BODIES = (
  r"^(base_link|pelvis_[12]_link|arm_(left|right)_[1-4]_link"
  r"|leg_(left|right)_([1-4]|femur|knee)_link)$"
)

# Fragility weight per body (first match wins). The torso and the pelvis with
# the hip housings carry the electronics and the leg actuators. The pelvis is
# the most expensive of all: sitting or lying on the leg actuators is the
# worst way to end up, worse than lying on the chest. Everything
# below the leg-length joint is passive linkage that can be replaced, so
# landing on the knees and the lower legs is far cheaper, but not free: the
# safe way down is a fetal position, legs folded between the pelvis and the
# ground and forearms between the chest and the ground. The forearms are plastic
# shells and the cheapest of all: they are meant to reach the ground before
# the torso and shield its front, which is its fragile side. The upper arm
# links hold the arm actuators, down to the one in the elbow, and cost as much
# as the torso: an arm should land on its forearm, not on its elbow.
REGEX_TORSO = r"pelvis_[12]_link"
REGEX_PELVIS_HIPS = r"(base_link|leg_(left|right)_[123]_link)"
REGEX_FOREARMS = r"arm_(left|right)_4_link"
REGEX_ARMS = r"arm_(left|right)_[1-4]_link"
REGEX_FEMUR = r"leg_(left|right)_femur_link"
REGEX_KNEE_SHANK = r"leg_(left|right)_(knee|4)_link"
# Every body but the arms, for the self-collision penalty.
REGEX_NOT_ARM_BODIES = r"^(?!arm_).*$"
KANGAROO_FRAGILITY = {
  REGEX_TORSO: 1.0,
  REGEX_PELVIS_HIPS: 2.0,
  REGEX_FOREARMS: 0.1,
  REGEX_ARMS: 1.0,
  REGEX_FEMUR: 0.2,
  REGEX_KNEE_SHANK: 0.1,
}

# Foot geoms (sole and ankle capsules) that a tipped start is set down on.
REGEX_FOOT_GEOMS = (
  r"^((left|right)_foot\d+_collision|leg_(left|right)_ankle(_bar)?_collision)$"
)

# Joints that can be offset at reset without breaking the knee linkage.
REGEX_RESET_NOISE_JOINTS = r"^(?!leg_.*_(length|femur|knee)_joint$).*$"

# Leg length [m] of the target squat. The joint range is 0.132-0.714 and its
# soft lower limit is 0.161.
SQUAT_LEG_LENGTH = 0.2

# Kick curriculum as (training iteration, max kick speed [m/s]). A policy
# trained on kicks of up to 5 m/s learned to step every one of them off and
# never practiced falling; it fell in half of the episodes at 6-7 m/s and
# almost always above 8 m/s. The range reaches 10 m/s so that falls at speed
# are common in training.
KICK_SPEED_CURRICULUM = ((0, 1.0), (150, 2.5), (400, 5.0), (800, 7.5), (1200, 10.0))
# Upper end of the kick range when playing a policy.
PLAY_MAX_KICK_SPEED = 8.0

# Starts that are already past recovery, at any kick speed: this share of the
# episodes starts tipped over by an angle in the range [rad], rotating at a
# rate in the range [rad/s].
TIP_FRACTION = 0.3
TIP_ANGLE_RANGE = (math.radians(15.0), math.radians(60.0))
TIP_RATE_RANGE = (0.0, 2.0)

# Most handovers are mild, and after a mild one the robot must stay on its
# knees rather than go down onto its arms: in this share of the episodes the
# kick is below the gentle maximum [m/s]. The others draw from the full range
# (KICK_SPEED_CURRICULUM), which is there for robustness.
GENTLE_KICK_FRACTION = 0.5
GENTLE_MAX_KICK_SPEED = 2.5

# A robot that falls while walking or running falls forwards: in this share of
# the episodes the kick, and the tipping if there is one, point within the
# half angle [rad] of where the robot faces. The other episodes are pushed in
# any direction, which stands for external hits and keeps the policy robust.
FORWARD_FALL_FRACTION = 0.5
FORWARD_FALL_HALF_ANGLE = math.radians(30.0)

# Starts from what the walking policy does when something goes wrong, recorded
# a detection time after the event (scripts/record_walker_states.py):
# - of the moving starts with an actuator down from the first step, this share
#   starts from a state recorded with that same actuator down;
# - of the other moving starts, this share starts from a state recorded after
#   a swinging foot was caught.
FAULT_STATE_FRACTION = 0.7
TRIP_STATE_FRACTION = 0.4
# Half of those starts are left as recorded, kicked by at most this speed
# [m/s]: the fall the walker was already in. The other half gets the full
# kick range like every other start.
AS_RECORDED_FRACTION = 0.5
AS_RECORDED_MAX_KICK_SPEED = 0.5

# Foot touchdowns per episode that are free. The fallback policy may be
# running on degraded actuators, so it must not rely on stepping a kick off;
# a step or two to orient the fall is fine.
FREE_STEPS = 2

# A second disturbance during the episode: the time to the next shove is drawn
# from the interval [s], so with 5 s episodes about half of them get one, at
# any time after the first second, including after the robot has come to rest.
# A robot that only looks stable is tested, and has to be protected when it
# goes down after all. Speed range in m/s.
SHOVE_INTERVAL_RANGE_S = (1.0, 9.0)
SHOVE_SPEED_RANGE = (0.5, 4.0)

# Share of the at-rest bonus that is paid with the forearms away from the
# chest; the rest is paid in proportion to the guard coverage.
UNGUARDED_REST_FRACTION = 0.3
# Width [m] of the kernel that turns forearm-to-zone distance into coverage.
GUARD_COVERAGE_STD = 0.2

# Share of the at-rest bonus that is paid while the pelvis or a hip housing
# is on the ground, while the torso is, and while an arm is. Kneeling is the
# place to rest. Lying on the forearms and the knees with the chest off the
# ground is the next best: when both paid the same, the policy stopped
# kneeling altogether, because going down onto the forearms always works.
PELVIS_REST_FRACTION = 0.2
TORSO_REST_FRACTION = 0.5
ARM_REST_FRACTION = 0.8

# Action scale of the arm joints, as the natural frequency [Hz] it is computed
# for. The scale is 0.25 * effort / stiffness and the stiffness follows from
# the actuators' natural frequency (3.5 Hz). That gives the arms 0.04-0.05 rad
# per unit of policy output: enough for the arm swing of walking, far too
# little to bring a forearm in front of the chest, which takes about a radian
# on several joints, or to explore doing so. Only the scale is computed for
# this lower frequency; the actuators keep their gains.
ARM_ACTION_SCALE_NATURAL_FREQ_HZ = 1.0

# Actuator faults. A failed actuator goes limp: it cannot be commanded, still
# reports its state, and its joint moves freely against its own friction (the
# actuators have no brake). The drives report their errors, so the policy is
# told which actuators are down. The first failure is there from the first
# step, as the reason the fallback took over; further ones follow within the
# first second. The leg actuators carry the robot while it walks and go into
# error far more often than the others, so each of them is this many times
# as likely to fail as an arm or waist actuator: about 5 in 6 failures hit
# a leg.
FAULT_COUNT_PROBABILITIES = (0.40, 0.40, 0.15, 0.05)
FAULT_LEG_WEIGHT = 4.0
KANGAROO_ACTUATOR_FAULTS = mdp.ActuatorFaultCfg(
  count_probabilities=FAULT_COUNT_PROBABILITIES,
  joint_weights={r"leg_.*": FAULT_LEG_WEIGHT, r".*": 1.0},
  later_onset_range_s=(0.0, 1.0),
)

# Fraction of its force limit above which an actuator counts as overloaded.
OVERLOAD_THRESHOLD = 0.7
# Rollout length the curriculum's iteration counts are expressed in.
_STEPS_PER_ITERATION = 24


# Where scripts/record_walker_states.py writes by default, relative to the
# directory the training is launched from.
DEFAULT_WALKER_STATE_BUFFER = "artifacts/kangaroo_walker_states.npz"

# How far the arms reach out is measured from the body axis, which runs in
# the torso frame from below the hips to the shoulders. The walking policy
# keeps its forearm tips within 0.44 m of it; fully extended they are at 1.06 m.
KANGAROO_ARM_SPREAD = mdp.ArmSpreadCfg(
  torso_body="pelvis_2_link",
  axis_start=(0.0, 0.0, -0.4),
  axis_end=(0.0, 0.0, 0.4),
  elbow_bodies=("arm_left_4_link", "arm_right_4_link"),
  tip_sites=("ee_left", "ee_right"),
)

# Guard zones for the forearms, in the torso frame (x forward, y left, z up),
# for the left arm; the right arm's are mirrored. The chest shell's front
# face is at x = 0.13 m, it is 0.32 m wide and its top is at z = 0.47 m; the
# shoulders are at y = +-0.11 m, z = 0.39 m. Each forearm belongs in front of
# the chest on its own side, pointing forwards or across, elbow bent: a
# bumper that reaches the ground before the chest does. The zones are large:
# about 1.3% of an arm's joint space puts its forearm inside, within 0.45 rad
# of the default pose. A single thin slab against the chest for both forearms
# was tried first; only 0.05% of the joint space reached it, with the elbow
# folded shut and sticking out, and one arm never found it.
KANGAROO_FOREARM_GUARD = mdp.ForearmGuardCfg(
  torso_body="pelvis_2_link",
  tip_box_min=(0.15, -0.15, -0.10),
  tip_box_max=(0.55, 0.30, 0.45),
  mid_box_min=(-0.05, 0.00, -0.10),
  mid_box_max=(0.40, 0.40, 0.45),
  elbow_bodies=("arm_left_4_link", "arm_right_4_link"),
  tip_sites=("ee_left", "ee_right"),
  arm_sides=(1.0, -1.0),
)

# Self-contact force [N] above which a touch counts as a hit. Below it the
# links may rest against each other.
SELF_COLLISION_FORCE_THRESHOLD = 50.0

# Force [N] above which a forearm pressed against the torso counts as
# shielding it (reported as a metric, never penalized).
SHIELD_FORCE_THRESHOLD = 50.0


def pal_kangaroo_safe_fall_env_cfg(
  play: bool = False,
  state_buffer_path: str | None = DEFAULT_WALKER_STATE_BUFFER,
  moving_fraction: float = 0.6,
) -> ManagerBasedRlEnvCfg:
  """Create the PAL Robotics KANGAROO safe-fall configuration.

  There is one fallback policy, so there is one task: a ``moving_fraction`` of
  the episodes starts from a walking state recorded with
  scripts/record_walker_states.py and the rest starts standing still. If the
  state buffer file does not exist, every episode starts standing still and a
  warning says so.

  Built on the flat velocity task so that the actor observations, the action
  scaling and the domain randomization are identical to the walking policy.
  That lets a walking actor be loaded directly, and lets a supervisor switch
  policies at runtime without changing what is sensed.
  """
  cfg = pal_kangaroo_flat_env_cfg(play=play)

  # -- Robot and terrain: mesh-accurate ground contact on separate layers.

  cfg.scene.entities = {"robot": get_kangaroo_fall_robot_cfg()}
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.collisions = (FALL_TERRAIN_COLLISION,)

  # Contact and constraint buffers, sized from peaks measured with
  # scripts/measure_fall_buffers.py (kicks up to 10 m/s, random actions):
  # - njmax is a hard per-world limit. Measured peak: 161 rows.
  # - nconmax is a shared pool sized as a per-world average. Measured worst
  #   step: 22 (every robot standing on all sole capsules right after a
  #   synchronized reset); the largest single world had 34. It also sizes the
  #   mesh-mesh collision buffers, which dominate GPU memory.
  # With these values 4096 envs use about 3.9 GiB of GPU memory and run at
  # about 33k env-steps/s on an RTX 5060 Ti 16 GB. More envs do not run faster.
  cfg.sim.nconmax = 64
  cfg.sim.njmax = 300
  cfg.sim.contact_sensor_maxmatch = 64

  # Long enough to slide to a stop after the strongest kick.
  cfg.episode_length_s = 5.0

  # -- Actions: the walker's, with a larger scale on the arm joints.

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  assert isinstance(joint_pos_action.scale, dict)
  arm_scale_factor = (
    NATURAL_FREQ / (2.0 * math.pi * ARM_ACTION_SCALE_NATURAL_FREQ_HZ)
  ) ** 2
  joint_pos_action.scale = {
    joint: scale * (arm_scale_factor if joint.startswith("arm_") else 1.0)
    for joint, scale in joint_pos_action.scale.items()
  }

  # -- Sensors

  impact_sensor = ContactSensorCfg(
    name="body_ground_impact",
    primary=ContactMatch(mode="body", pattern=REGEX_IMPACT_BODIES, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=cfg.decimation,
  )
  # Self-collision: contacts between a leg and the other leg, the pelvis or
  # the torso, with per-substep forces so that only hits are penalized. The
  # arms are deliberately not watched, against the torso, against each other
  # or against the legs: a forearm squeezed between the chest and the ground
  # is the arm doing its job, and arms resting on the folded legs are what a
  # fetal position looks like.
  self_collision_sensor = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(
      mode="subtree",
      pattern=("leg_left_1_link", "leg_right_1_link"),
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern=REGEX_NOT_ARM_BODIES, entity="robot"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=cfg.decimation,
  )
  # Forearm pressed against the torso, for the shielding metrics.
  forearm_torso_sensor = ContactSensorCfg(
    name="forearm_torso_contact",
    primary=ContactMatch(mode="body", pattern=REGEX_FOREARMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="pelvis_2_link", entity="robot"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
    history_length=cfg.decimation,
  )
  cfg.scene.sensors = tuple(
    s
    for s in (cfg.scene.sensors or ())
    if s.name not in ("body_ground_contact", "self_collision")
  ) + (impact_sensor, self_collision_sensor, forearm_torso_sensor)

  impact_cfg = mdp.ImpactCfg(
    sensor_name=impact_sensor.name, fragility=KANGAROO_FRAGILITY
  )
  step_cfg = mdp.StepCountCfg(
    feet_sensor_name="feet_ground_contact",
    impact_cfg=impact_cfg,
    free_steps=FREE_STEPS,
  )
  leg_length_cfg = SceneEntityCfg("robot", joint_names=(REGEX_LEG_LENGTH_JOINTS_ONLY,))
  # Strict definition of the safe states, used by the metrics and by
  # scripts/eval_fall.py. The reward uses a smooth version of the same idea.
  safe_state_params = {
    "impact_cfg": impact_cfg,
    "feet_sensor_name": "feet_ground_contact",
    "leg_length_max": SQUAT_LEG_LENGTH + 0.02,
    "lin_speed_max": 0.15,
    "ang_speed_max": 0.3,
    "low_height": 0.6,
    "asset_cfg": leg_length_cfg,
  }

  # -- Commands: none. The slot stays so the actor layout matches the walker.

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (0.0, 0.0)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.ranges.heading = None
  twist_cmd.heading_command = False
  twist_cmd.rel_standing_envs = 1.0
  twist_cmd.rel_heading_envs = 0.0
  twist_cmd.rel_forward_envs = 0.0
  twist_cmd.debug_vis = False

  # -- Observations: the walker's, plus which actuators are down (last, so
  # that an actor trained without them can be widened); the critic also gets
  # the impact state.

  critic_terms = cfg.observations["critic"].terms
  for terms in (cfg.observations["actor"].terms, critic_terms):
    terms["actuator_faults"] = ObservationTermCfg(
      func=mdp.actuator_fault_flags, params={"fault_cfg": KANGAROO_ACTUATOR_FAULTS}
    )
  critic_terms["body_ground_load"] = ObservationTermCfg(
    func=mdp.body_ground_load, params={"impact_cfg": impact_cfg}
  )
  critic_terms["impact_running_max"] = ObservationTermCfg(
    func=mdp.impact_running_max, params={"impact_cfg": impact_cfg}
  )
  critic_terms["base_height"] = ObservationTermCfg(func=mdp.base_height)
  critic_terms["episode_time_left"] = ObservationTermCfg(func=mdp.episode_time_left)

  # -- Events: every episode starts at the switch instant with a kick.

  for name in ("reset_base", "reset_robot_joints", "push_robot"):
    cfg.events.pop(name, None)
  # The failures are drawn first: a moving start is taken from the states that
  # were recorded with the episode's failed actuator down.
  cfg.events["actuator_faults"] = EventTermCfg(
    func=mdp.sample_actuator_faults,
    mode="reset",
    params={"fault_cfg": KANGAROO_ACTUATOR_FAULTS},
  )
  cfg.events["reset_fall_start"] = EventTermCfg(
    func=mdp.reset_fall_start,
    mode="reset",
    params={
      "max_kick_speed": PLAY_MAX_KICK_SPEED if play else KICK_SPEED_CURRICULUM[0][1],
      "min_kick_speed": 0.0,
      "max_kick_ang_speed": 0.5,
      "position_range": (-0.5, 0.5),
      "height_range": (0.006, 0.015),
      "joint_pos_noise": 0.05,
      "moving_fraction": moving_fraction if state_buffer_path else 0.0,
      "state_buffer_path": state_buffer_path,
      "tip_fraction": TIP_FRACTION,
      "tip_angle_range": TIP_ANGLE_RANGE,
      "tip_rate_range": TIP_RATE_RANGE,
      "forward_fraction": FORWARD_FALL_FRACTION,
      "forward_half_angle": FORWARD_FALL_HALF_ANGLE,
      "gentle_fraction": GENTLE_KICK_FRACTION,
      "gentle_max_kick_speed": GENTLE_MAX_KICK_SPEED,
      "trip_state_fraction": TRIP_STATE_FRACTION,
      "fault_state_fraction": FAULT_STATE_FRACTION,
      "fault_cfg": KANGAROO_ACTUATOR_FAULTS,
      "as_recorded_fraction": AS_RECORDED_FRACTION,
      "as_recorded_max_kick_speed": AS_RECORDED_MAX_KICK_SPEED,
      "ground_geom_cfg": SceneEntityCfg("robot", geom_names=(REGEX_FOOT_GEOMS,)),
      "asset_cfg": SceneEntityCfg("robot", joint_names=(REGEX_RESET_NOISE_JOINTS,)),
    },
  )
  cfg.events["actuator_faults_onset"] = EventTermCfg(
    func=mdp.apply_actuator_faults,
    mode="step",
    params={"fault_cfg": KANGAROO_ACTUATOR_FAULTS},
  )
  cfg.events["shove"] = EventTermCfg(
    func=mdp.shove,
    mode="interval",
    interval_range_s=SHOVE_INTERVAL_RANGE_S,
    params={"speed_range": SHOVE_SPEED_RANGE},
  )
  cfg.events["body_friction"] = EventTermCfg(
    mode="startup",
    func=dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", geom_names=(rf".*{FALL_COLLISION_MESH_SUFFIX}$",)
      ),
      "operation": "abs",
      "ranges": (0.3, 1.0),
      "shared_random": True,
    },
  )

  # -- Curriculum: from recoverable kicks to inevitable falls.

  cfg.curriculum = {}
  if not play:
    cfg.curriculum["kick_speed"] = CurriculumTermCfg(
      func=mdp.event_param_curriculum,
      params={
        "event_name": "reset_fall_start",
        "stages": [
          {"step": it * _STEPS_PER_ITERATION, "params": {"max_kick_speed": v}}
          for it, v in KICK_SPEED_CURRICULUM
        ],
      },
    )

  # -- Rewards

  for name in (
    "track_linear_velocity",
    "track_angular_velocity",
    "pose",
    "air_time",
    "foot_clearance",
    "foot_swing_height",
    "foot_slip",
    "soft_landing",
  ):
    cfg.rewards.pop(name, None)

  # Kept from the walking task: upright, body_ang_vel, angular_momentum,
  # dof_pos_limits, action_rate_l2, self_collisions, both convex-hull joint
  # limits and joint_vel_limits. Staying upright is only a mild preference:
  # the robot is allowed to go down, as long as it goes down softly.
  cfg.rewards["upright"].weight = 0.2
  cfg.rewards["body_ang_vel"].weight = -0.1
  # Counts the physics substeps (0 to decimation) in which a contact that
  # involves a leg exceeds the force threshold: resting a link against a leg
  # is free, hitting it is not. Contacts of the arms are not counted at all
  # (see the sensor), otherwise the penalty pushes the arms away from the body.
  cfg.rewards["self_collisions"].weight = -0.5
  cfg.rewards["self_collisions"].params["force_threshold"] = (
    SELF_COLLISION_FORCE_THRESHOLD
  )

  # Damage.
  cfg.rewards["max_impulse"] = RewardTermCfg(
    func=mdp.max_impulse_increment,
    weight=-30.0,
    params={"impact_cfg": impact_cfg},
  )
  cfg.rewards["contact_load"] = RewardTermCfg(
    func=mdp.contact_load,
    weight=-2.0,
    params={"impact_cfg": impact_cfg},
  )
  cfg.rewards["pre_impact_velocity"] = RewardTermCfg(
    func=mdp.pre_impact_velocity,
    weight=-5.0,
    params={
      "impact_cfg": impact_cfg,
      "height_scale": 0.2,
      "exclude_body_regex": REGEX_ARMS,
    },
  )
  cfg.rewards["actuator_overload"] = RewardTermCfg(
    func=mdp.actuator_overload,
    weight=-0.5,
    params={"threshold": OVERLOAD_THRESHOLD},
  )
  cfg.rewards["joint_limit_strike"] = RewardTermCfg(
    func=mdp.joint_limit_strike,
    weight=-1.0,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  )

  # Arms. The fallback only runs once something has gone wrong, and a robot
  # that looks stable in simulation may still go down on the real one. So the
  # arms are not kept for balancing: from the first step each forearm goes in
  # front of the chest on its own side and stays there, also at rest.
  cfg.rewards["forearm_guard"] = RewardTermCfg(
    func=mdp.forearm_guard_distance,
    weight=-5.0,
    params={"guard_cfg": KANGAROO_FOREARM_GUARD},
  )
  # The forearm and its tip touch down before the elbow does.
  cfg.rewards["elbow_first"] = RewardTermCfg(
    func=mdp.elbow_first,
    weight=-3.0,
    params={"guard_cfg": KANGAROO_FOREARM_GUARD},
  )
  # Once the robot is down the arms hold still.
  cfg.rewards["arm_motion_at_rest"] = RewardTermCfg(
    func=mdp.motion_at_rest,
    weight=-0.1,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(r"arm_.*_joint",))},
  )

  # Go down forwards, onto the knees and the forearms, never backwards onto
  # the pelvis.
  cfg.rewards["backward_lean"] = RewardTermCfg(
    func=mdp.backward_lean,
    weight=-3.0,
    params={"torso_body": KANGAROO_FOREARM_GUARD.torso_body},
  )

  # No running a kick off: steps beyond the free ones cost one unit each, and
  # shuffling the feet along the ground is not a way around it.
  cfg.rewards["excess_steps"] = RewardTermCfg(
    func=mdp.excess_steps,
    weight=-1.0,
    params={"step_cfg": step_cfg},
  )
  cfg.rewards["foot_slide"] = RewardTermCfg(
    func=mdp.foot_slide,
    weight=-0.5,
    params={
      "sensor_name": "feet_ground_contact",
      "impact_cfg": impact_cfg,
      "max_speed": 1.0,
      "asset_cfg": SceneEntityCfg("robot", site_names=("left_foot", "right_foot")),
    },
  )

  # Getting low and coming to rest. Lying still is worth as much as squatting
  # on the feet: the damage terms decide which one a given fall should end in.
  # Being at rest only counts in full with the chest guarded and the pelvis
  # off the ground.
  cfg.rewards["crouch"] = RewardTermCfg(
    func=mdp.crouch,
    weight=2.0,
    params={
      "target_length": SQUAT_LEG_LENGTH,
      "std": 0.15,
      "asset_cfg": leg_length_cfg,
    },
  )
  cfg.rewards["com_speed"] = RewardTermCfg(func=mdp.com_speed_l2, weight=-0.5)
  cfg.rewards["safe_state"] = RewardTermCfg(
    func=mdp.safe_state,
    weight=5.0,
    params={
      "impact_cfg": impact_cfg,
      "lin_speed_std": 0.3,
      "ang_speed_std": 0.6,
      "low_height": 0.5,
      "height_std": 0.15,
      "rest_fraction": 1.0,
      "guard_cfg": KANGAROO_FOREARM_GUARD,
      "guard_std": GUARD_COVERAGE_STD,
      "unguarded_fraction": UNGUARDED_REST_FRACTION,
      "contact_rest_fractions": {
        REGEX_PELVIS_HIPS: PELVIS_REST_FRACTION,
        REGEX_TORSO: TORSO_REST_FRACTION,
        REGEX_ARMS: ARM_REST_FRACTION,
      },
    },
  )

  # -- Terminations: falling is part of the task, so only time runs out.

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
  }

  # -- Metrics

  group_regexes = {
    "torso": REGEX_TORSO,
    "pelvis_hips": REGEX_PELVIS_HIPS,
    "forearms": REGEX_FOREARMS,
    "arms": REGEX_ARMS,
    "femur": REGEX_FEMUR,
    "knee_shank": REGEX_KNEE_SHANK,
  }
  cfg.metrics = {
    "max_weighted_impulse": MetricsTermCfg(
      func=mdp.impact_running_max_metric,
      params={"impact_cfg": impact_cfg},
      reduce="max",
    ),
    "imu_acc_peak": MetricsTermCfg(
      func=mdp.sensor_norm,
      params={"sensor_name": "robot/imu_lin_acc"},
      reduce="max",
    ),
    "squat_success": MetricsTermCfg(
      func=mdp.safe_state_metric,
      params={**safe_state_params, "kind": "squat"},
      reduce="last",
    ),
    "rest_success": MetricsTermCfg(
      func=mdp.safe_state_metric,
      params={**safe_state_params, "kind": "rest"},
      reduce="last",
    ),
    "unsafe_time_fraction": MetricsTermCfg(
      func=mdp.safe_state_metric,
      params={**safe_state_params, "kind": "unsafe"},
      reduce="mean",
    ),
    "actuator_overload": MetricsTermCfg(
      func=mdp.actuator_overload, params={"threshold": OVERLOAD_THRESHOLD}
    ),
    "steps_taken": MetricsTermCfg(
      func=mdp.steps_taken, params={"step_cfg": step_cfg}, reduce="max"
    ),
    "joint_limit_strike": MetricsTermCfg(
      func=mdp.joint_limit_strike,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
      reduce="max",
    ),
    "joint_torque_mag": MetricsTermCfg(
      func=mdp.joint_torques_magnitude,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "action_rate_l2": MetricsTermCfg(func=mdp.action_rate_l2, params={}),
    "forearm_shield": MetricsTermCfg(
      func=mdp.contact_force_above,
      params={
        "sensor_name": forearm_torso_sensor.name,
        "force_threshold": SHIELD_FORCE_THRESHOLD,
      },
      reduce="max",
    ),
    "guard_coverage": MetricsTermCfg(
      func=mdp.guard_coverage,
      params={"guard_cfg": KANGAROO_FOREARM_GUARD, "std": GUARD_COVERAGE_STD},
      reduce="last",
    ),
    "backward_lean": MetricsTermCfg(
      func=mdp.backward_lean,
      params={"torso_body": KANGAROO_FOREARM_GUARD.torso_body},
      reduce="last",
    ),
    "failed_actuators": MetricsTermCfg(
      func=mdp.num_failed_actuators,
      params={"fault_cfg": KANGAROO_ACTUATOR_FAULTS},
      reduce="last",
    ),
    "arm_tip_distance": MetricsTermCfg(
      func=mdp.arm_tip_distance,
      params={"arm_cfg": KANGAROO_ARM_SPREAD},
      reduce="max",
    ),
  }
  for group, regex in group_regexes.items():
    cfg.metrics[f"max_impulse_{group}"] = MetricsTermCfg(
      func=mdp.group_impulse,
      params={"impact_cfg": impact_cfg, "body_regex": regex},
      reduce="max",
    )
    cfg.metrics[f"contact_{group}"] = MetricsTermCfg(
      func=mdp.group_contact,
      params={"impact_cfg": impact_cfg, "body_regex": regex},
      reduce="max",
    )

  return cfg
