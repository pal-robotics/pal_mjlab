.. _Training_env:

Building a training environment
================================

An mjlab training environment is defined entirely in a single
``ManagerBasedRlEnvCfg`` dataclass. Rather than overriding methods in a class
hierarchy, you compose modular building blocks — observations, rewards,
terminations, actions, and events — and snap them together into one flat config
object.

This document walks through each piece and shows how they fit together.

Overview
--------

Every environment in mjlab follows the same structure:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Component
     - Purpose
   * - ``EntityCfg``
     - Wraps a MuJoCo XML and exposes simulation data as batched PyTorch tensors.
   * - ``SceneCfg``
     - Holds entities, terrain, and sensors; also sets ``num_envs``.
   * - ``observations``
     - Dict of ``ObservationGroupCfg``, each a named collection of observation terms.
   * - ``actions``
     - Dict of ``ActionTermCfg``, routing policy output to actuators.
   * - ``rewards``
     - Dict of ``RewardTermCfg``, summed and scaled each step.
   * - ``terminations``
     - Dict of ``TerminationTermCfg``, ending episodes on failure or time-out.
   * - ``events``
     - Dict of ``EventTermCfg``, fired on startup, reset, or at fixed intervals.
   * - ``commands``
     - Dict of ``CommandTermCfg``, generating goal signals (e.g. velocity targets).

All manager dicts follow the same pattern: each key is a human-readable name
that appears in training logs, and each value is a term config with at minimum a
``func`` field pointing to the callable that implements the term.

The XML Model
-------------

Every environment starts with a MuJoCo XML that defines the physical system.
The ``EntityCfg`` wraps this XML; at runtime, it exposes joint positions,
velocities, and other simulation data as batched tensors with shape
``[num_envs, ...]``.

.. code-block:: python

   import mujoco
   from pathlib import Path
   from mjlab.entity import EntityCfg, EntityArticulationInfoCfg, XmlActuatorCfg

   _ROBOT_XML = Path(__file__).parent / "robot.xml"

   def _get_spec() -> mujoco.MjSpec:
       return mujoco.MjSpec.from_file(str(_ROBOT_XML))

   _ARTICULATION = EntityArticulationInfoCfg(
       actuators=(XmlActuatorCfg(target_names_expr=(".*",)),),
   )

   _INIT_STATE = EntityCfg.InitialStateCfg(
       joint_pos={".*": 0.0},
       joint_vel={".*": 0.0},
   )

   def get_robot_cfg() -> EntityCfg:
       return EntityCfg(
           spec_fn=_get_spec,
           articulation=_ARTICULATION,
           init_state=_INIT_STATE,
       )

Joint name patterns in ``InitialStateCfg`` are regular expressions matched
against MuJoCo joint names. ``".*"`` matches all joints.

Observations
------------

Each observation term is a function that reads from the environment and returns
a tensor of shape ``[num_envs, dim]``. The observation manager concatenates
terms within a group into a single vector for the policy.

mjlab ships built-in terms in ``mjlab.envs.mdp``; you can also write your own.
All data is batched — every function receives and returns tensors with a leading
``num_envs`` dimension.

.. code-block:: python

   from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
   from mjlab.managers.scene_entity_config import SceneEntityCfg
   from mjlab.envs import mdp

   robot_cfg = SceneEntityCfg("robot")

   actor_terms = {
       "joint_pos": ObservationTermCfg(
           func=mdp.joint_pos_rel,
           params={"asset_cfg": robot_cfg},
       ),
       "joint_vel": ObservationTermCfg(
           func=mdp.joint_vel_rel,
           params={"asset_cfg": robot_cfg},
       ),
   }

   observations = {
       "actor": ObservationGroupCfg(
           terms=actor_terms,
           concatenate_terms=True,
           enable_corruption=True,   # applies per-term noise if configured
       ),
       "critic": ObservationGroupCfg(
           terms={**actor_terms},
           concatenate_terms=True,
           enable_corruption=False,  # critic sees clean observations
       ),
   }

The RL runner expects at minimum an ``"actor"`` and ``"critic"`` group.
Giving the critic clean observations while the actor receives noise is the
standard asymmetric actor-critic setup.

To add noise to a term, attach a noise config:

.. code-block:: python

   from mjlab.utils.noise import UniformNoiseCfg

   ObservationTermCfg(
       func=mdp.joint_pos_rel,
       params={"asset_cfg": robot_cfg},
       noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
   )

Actions
-------

Action terms route slices of the policy output to actuators. The most common
type for locomotion is ``JointPositionActionCfg``, which converts policy outputs
to joint position targets:

.. code-block:: python

   from mjlab.envs.mdp.actions import JointPositionActionCfg

   actions = {
       "joint_pos": JointPositionActionCfg(
           entity_name="robot",
           actuator_names=(".*",),   # regex matching actuator names in the XML
           scale=0.5,
           use_default_offset=True,  # outputs are relative to the default pose
       ),
   }

For tasks requiring direct force control, use ``JointEffortActionCfg`` instead.

Rewards
-------

Each reward term is a function returning a scalar per environment. The reward
manager computes a weighted sum every step. Negative weights produce penalties.
When ``scale_rewards_by_dt=True`` (the default), each term is additionally
multiplied by ``step_dt`` so cumulative episode sums are invariant to simulation
frequency.

.. code-block:: python

   from mjlab.managers.reward_manager import RewardTermCfg
   from mjlab.envs import mdp

   rewards = {
       "track_linear_velocity": RewardTermCfg(
           func=mdp.track_linear_velocity,
           weight=2.0,
           params={"command_name": "twist", "std": 0.5},
       ),
       "joint_pos_limits": RewardTermCfg(
           func=mdp.joint_pos_limits,
           weight=-1.0,
       ),
       "action_rate": RewardTermCfg(
           func=mdp.action_rate_l2,
           weight=-0.1,
       ),
   }

All individual term values are logged as episode averages during training, which
makes it straightforward to diagnose which terms dominate learning.

Terminations
------------

Termination terms end an episode early. The ``time_out`` flag distinguishes a
truncation (artificial time limit, value should be bootstrapped) from a failure
(true terminal state):

.. code-block:: python

   import math
   from mjlab.managers.termination_manager import TerminationTermCfg
   from mjlab.envs import mdp

   terminations = {
       "time_out": TerminationTermCfg(
           func=mdp.time_out,
           time_out=True,   # truncation: bootstraps value past the boundary
       ),
       "fell_over": TerminationTermCfg(
           func=mdp.bad_orientation,
           params={"limit_angle": math.radians(70.0)},
       ),
   }

Events
------

Events are hooks that fire at specific points in the environment lifecycle.
The ``mode`` field controls when:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Mode
     - When it fires
   * - ``"startup"``
     - Once when the environment is first created (e.g. randomising fixed physics properties).
   * - ``"reset"``
     - Each time an environment resets to a new episode (e.g. randomising initial state).
   * - ``"interval"``
     - At random intervals during an episode (e.g. applying push disturbances).

.. code-block:: python

   from mjlab.managers.event_manager import EventTermCfg
   from mjlab.managers.scene_entity_config import SceneEntityCfg
   from mjlab.envs import mdp
   from mjlab.envs.mdp import dr

   events = {
       "reset_base": EventTermCfg(
           func=mdp.reset_root_state_uniform,
           mode="reset",
           params={
               "pose_range": {"yaw": (-3.14, 3.14)},
               "velocity_range": {},
           },
       ),
       "foot_friction": EventTermCfg(
           func=dr.geom_friction,
           mode="startup",
           params={
               "asset_cfg": SceneEntityCfg("robot", geom_names=[]),
               "operation": "abs",
               "ranges": (0.3, 1.2),
           },
       ),
       "push_robot": EventTermCfg(
           func=mdp.push_by_setting_velocity,
           mode="interval",
           interval_range_s=(1.0, 3.0),
           params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
       ),
   }

The default ``events`` dict on ``ManagerBasedRlEnvCfg`` already includes
``reset_scene_to_default``, which resets all entities to their initial pose.
Override the dict to extend or replace this behaviour.

Assembling the Config
---------------------

Once all pieces are defined, pass them to ``ManagerBasedRlEnvCfg``:

.. code-block:: python

   from mjlab.envs import ManagerBasedRlEnvCfg
   from mjlab.scene import SceneCfg
   from mjlab.sim import MujocoCfg, SimulationCfg
   from mjlab.terrains import TerrainEntityCfg

   def make_env_cfg() -> ManagerBasedRlEnvCfg:
       return ManagerBasedRlEnvCfg(
           decimation=4,
           episode_length_s=20.0,
           sim=SimulationCfg(
               mujoco=MujocoCfg(timestep=0.005),
           ),
           scene=SceneCfg(
               terrain=TerrainEntityCfg(terrain_type="plane"),
               entities={"robot": get_robot_cfg()},
               num_envs=1,
           ),
           observations=observations,
           actions=actions,
           rewards=rewards,
           terminations=terminations,
           events=events,
       )

Timing
------

Three parameters jointly control the temporal structure of the environment:

``sim.mujoco.timestep``
   The physics integration step in seconds. The default is 0.002 s (500 Hz).
   Smaller values produce more stable physics at the cost of simulation speed.

``decimation``
   The number of physics steps executed per policy step. The policy runs at
   ``1 / (timestep × decimation)`` Hz.

``episode_length_s``
   Episode duration in seconds. The number of policy steps per episode is
   ``ceil(episode_length_s / (timestep × decimation))``.

**Example.** ``timestep=0.005`` and ``decimation=4`` gives 50 Hz policy
frequency. With ``episode_length_s=20.0`` that is 1000 policy steps per episode.

At runtime, use the environment properties:

.. code-block:: python

   env.physics_dt           # = cfg.sim.mujoco.timestep
   env.step_dt              # = cfg.sim.mujoco.timestep * cfg.decimation
   env.max_episode_length   # steps (int)
   env.max_episode_length_s # seconds (float)

Registration and Training
-------------------------

Register the environment so it can be launched by name. Each registration
pairs an environment config with an RL config specifying the network
architecture and PPO hyperparameters:

.. code-block:: python

   # __init__.py
   from mjlab.tasks import register_mjlab_task
   from .my_env_cfg import make_env_cfg, my_ppo_runner_cfg

   register_mjlab_task(
       task_id="Mjlab-MyTask-v0",
       env_cfg=make_env_cfg(),
       play_env_cfg=make_env_cfg(play=True),
       rl_cfg=my_ppo_runner_cfg(),
   )