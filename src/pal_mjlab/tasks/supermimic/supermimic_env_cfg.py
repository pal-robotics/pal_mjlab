"""SuperMimic motion-tracking environment (arXiv 2610.28480, Appendix A).

Robot-agnostic base configuration; robot configs (e.g. ``kangaroo/env_cfgs.py``) fill in
the robot, tracked bodies, joint selections, sensors and DR targets.

Paper settings reproduced here:
  - observations (Table A.1): q - q0, q̇, IMU angular velocity, projected gravity and the
    previous action, with sensor noise and a 10-frame history; no base linear velocity;
  - goal: K=10 future reference frames, each BeyondMimic's command without state
    estimation: reference joint positions and velocities and root orientation (D6);
  - reward (Table A.2): BeyondMimic's tracking terms without self-collision, action rate
    -0.1, joint limits -2;
  - termination: root height error > 0.25 m or tilt error > 0.8, plus the clip end (D8);
  - DR (Table A.3) as the sim context ξ: foot friction, joint calibration offsets, torso
    CoM offset (sampled once per env, see the events below); no pushes;
  - sim: 5 ms physics, 4 substeps per 50 Hz control step.
The critic sees the heading-invariant state, the goal, ξ and two privileged terms (D7).
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from pal_mjlab.tasks.supermimic import mdp
from pal_mjlab.tasks.supermimic.mdp import MotionLibraryCommandCfg

# BeyondMimic's reset noise (reference state initialization).
RESET_POSE_RANGE = {
  "x": (-0.05, 0.05),
  "y": (-0.05, 0.05),
  "z": (-0.01, 0.01),
  "roll": (-0.1, 0.1),
  "pitch": (-0.1, 0.1),
  "yaw": (-0.2, 0.2),
}
RESET_VELOCITY_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.2, 0.2),
  "roll": (-0.52, 0.52),
  "pitch": (-0.52, 0.52),
  "yaw": (-0.78, 0.78),
}


def make_supermimic_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the base SuperMimic tracking configuration."""

  ##
  # Observations
  ##

  goal_terms = {
    "motion_goal": ObservationTermCfg(
      func=mdp.motion_goal, params={"command_name": "motion"}
    ),
  }

  # Robot configs set the joint selection and sensor names.
  proprio_terms = {
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"biased": True},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)
    ),
    "imu_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": ""},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "imu_projected_gravity": ObservationTermCfg(
      func=mdp.imu_projected_gravity,
      params={"sensor_name": ""},
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    "heading_state": ObservationTermCfg(
      func=mdp.robot_heading_state, params={"command_name": "motion"}
    ),
    "motion_goal": ObservationTermCfg(
      func=mdp.motion_goal, params={"command_name": "motion"}
    ),
    "sim_context": ObservationTermCfg(
      func=mdp.sim_context,
      params={
        "friction_asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per robot.
        "bias_asset_cfg": SceneEntityCfg("robot"),  # Measured joints, set per robot.
        "com_asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per robot.
      },
    ),
    "root_pose_error": ObservationTermCfg(
      func=mdp.motion_root_pose_error, params={"command_name": "motion"}
    ),
    "time_left": ObservationTermCfg(
      func=mdp.motion_time_left, params={"command_name": "motion"}
    ),
  }

  observations = {
    "goal": ObservationGroupCfg(
      terms=goal_terms, concatenate_terms=True, enable_corruption=False
    ),
    "proprio": ObservationGroupCfg(
      terms=proprio_terms,
      concatenate_terms=True,
      enable_corruption=True,
      history_length=10,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms, concatenate_terms=True, enable_corruption=False
    ),
  }

  ##
  # Actions: q_sp = scale * a + q0 (scale set per robot).
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,
      use_default_offset=True,
    )
  }

  ##
  # Commands
  ##

  commands: dict[str, CommandTermCfg] = {
    "motion": MotionLibraryCommandCfg(
      entity_name="robot",
      dataset="",  # Set per robot (or library_path).
      body_names=(),  # Set per robot; first = anchor.
      pose_range=RESET_POSE_RANGE,
      velocity_range=RESET_VELOCITY_RANGE,
      joint_position_range=(-0.1, 0.1),
      debug_vis=True,
    )
  }

  ##
  # Events: the sim context ξ (Table A.3). Friction and calibration are resampled every
  # episode. The CoM offset is sampled once per env: re-sampling it at reset triggers
  # set_const over all worlds, which tripled the step time (235 vs 72 ms at 8192 envs).
  ##

  events: dict[str, EventTermCfg] = {
    "foot_friction": EventTermCfg(
      mode="reset",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per robot.
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,
      },
    ),
    "encoder_bias": EventTermCfg(
      mode="reset",
      func=dr.encoder_bias,
      params={"asset_cfg": SceneEntityCfg("robot"), "bias_range": (-0.01, 0.01)},
    ),
    "torso_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set per robot.
        "operation": "add",
        "ranges": {0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)},
      },
    ),
  }

  ##
  # Rewards (Table A.2)
  ##

  rewards: dict[str, RewardTermCfg] = {
    "motion_root_pos": RewardTermCfg(
      func=mdp.motion_root_pos_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_root_ori": RewardTermCfg(
      func=mdp.motion_root_ori_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_pos": RewardTermCfg(
      func=mdp.motion_body_pos_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_body_ori": RewardTermCfg(
      func=mdp.motion_body_ori_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_lin_vel": RewardTermCfg(
      func=mdp.motion_body_lin_vel_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 1.0},
    ),
    "motion_body_ang_vel": RewardTermCfg(
      func=mdp.motion_body_ang_vel_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 3.14},
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
    "joint_limit": RewardTermCfg(
      func=mdp.joint_pos_limits_hard,
      weight=-2.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
  }

  ##
  # Terminations
  ##

  terminations: dict[str, TerminationTermCfg] = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "motion_end": TerminationTermCfg(
      func=mdp.motion_end, params={"command_name": "motion"}
    ),
    "anchor_failure": TerminationTermCfg(
      func=mdp.anchor_failure,
      params={"command_name": "motion", "z_threshold": 0.25, "tilt_threshold": 0.8},
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # Set per robot.
      distance=2.8,
      fovy=55.0,
      elevation=-5.0,
      azimuth=120.0,
    ),
    # nconmax is a contact pool shared by all worlds (~15 contacts per world in use);
    # njmax is a hard per-world cap on constraint rows. Rows beyond it are dropped (wrong
    # physics, "nefc overflow"): on the full library up to 636 rows were requested, and the
    # 300 used before ended in a NaN collapse at iteration ~44,600 (D42).
    sim=SimulationCfg(
      nconmax=50,
      njmax=1024,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    decimation=4,
    episode_length_s=10.0,
  )
