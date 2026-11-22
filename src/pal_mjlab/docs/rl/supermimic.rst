.. _SuperMimic overview:

SuperMimic: learning through a differentiable world model
=========================================================

``pal_mjlab.rl.supermimic`` is a reusable implementation of SuperMimic
(`arXiv 2610.28480 <https://arxiv.org/abs/2610.28480>`_), a learning method that
replaces PPO's sampled policy gradients with **analytic gradients through a learned
world model**. It was first written for motion tracking
(:ref:`Kangaroo task_supermimic`); here it is a method any mjlab task can use, provided the
task describes its observations, rewards, failures and command in a form the method can
differentiate.

This page explains how the method works and how it differs from PPO in practice, how
to write a task for it, and how to tune one. The Kangaroo flat velocity task
(``Mjlab-SuperMimic-Velocity-Flat-Pal-Kangaroo``) is the running example.


How it works
------------

Every iteration does three things:

1. **Collect.** The policy, with Gaussian action noise, steps the simulator for a few
   steps (2.5 per environment on average). Each transition goes into a replay buffer:
   the robot state, the action, the command state, the domain-randomization parameters
   :math:`\xi` and the memory of the stateful reward terms.
2. **Update the world model** on short windows of replay (8 steps), autoregressively: it
   predicts the next heading-invariant robot state :math:`\hat s_{t+1}` from
   :math:`(s_t, a_t, \xi)` and a learned failure probability.
3. **Update the policy through imagination.** From states sampled from replay, the policy
   is rolled out for :math:`H` steps (32) inside the frozen world model. At every
   imagined step the task's own reward, observation and failure functions are evaluated
   on the predicted state, and the policy is moved along the exact gradient of

   .. math::

      J(\theta) = \sum_{t=0}^{H-1} \gamma^t \Big(\prod_{k \le t} c_k\Big) \hat r_{t+1}
                  + \gamma^H \Big(\prod_{k < H} c_k\Big) V(\hat s_H)

   where :math:`c_k` is the probability of not failing at step :math:`k` and :math:`V`
   is a critic fitted to the imagined :math:`\lambda`-returns, which covers what lies beyond
   the horizon.

Because the reward is differentiated rather than sampled, **every reward, observation
and failure term must be computable, differentiably, from a predicted state**. That is the
contract a task signs (see `Building a task`_).

Compared with PPO
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * -
     - PPO (rsl_rl)
     - SuperMimic
   * - Gradient
     - Sampled (likelihood ratio, advantages)
     - Analytic, through the world model and the reward functions
   * - Data use
     - On-policy, discarded after each update
     - Replay buffer (1000 steps per env), reused by the world model
   * - Policy
     - Stochastic, entropy-regularized
     - Deterministic; exploration comes from collection noise
   * - Rewards
     - Any function of the simulator, even non-smooth
     - Smooth functions of a predicted state (single definitions)
   * - Failures
     - Terminations reset the episode
     - The same terminations are also smooth probabilities in imagination
   * - Env steps per iteration
     - ``num_steps_per_env`` = 24 per env
     - ``env_steps_per_iteration`` = 2.5 per env

**Compare runs in env steps, not iterations.** With 4096 environments, a PPO iteration
collects 4096 x 24 ≈ 98k transitions; a SuperMimic iteration about 10k. 1000 PPO
iterations (98M steps) are about 10,000 SuperMimic iterations. A SuperMimic iteration is
also much more expensive (world-model training and backpropagation through 32 imagined
steps): on an A100 it runs at about 36 iterations per minute, about 20M env steps per hour.

The command curriculum (``commands_vel``) is keyed on ``env.common_step_counter``, the env
steps taken, so it advances with experience at the same rate for both methods. In
iterations, the velocity task's second stage (120k steps) arrives at PPO iteration 5000
and at SuperMimic iteration 48,000.


Training and playing
--------------------

.. code-block:: bash

   uv run train Mjlab-SuperMimic-Velocity-Flat-Pal-Kangaroo \
     --env.scene.num-envs 4096 \
     --agent.algorithm.policy.num-rollouts 4096 \
     --agent.max-iterations 4000 --agent.run-name <name>

   uv run play Mjlab-SuperMimic-Velocity-Flat-Pal-Kangaroo --checkpoint-file <run>/model_N.pt

Runs log to ``logs/rsl_rl/<experiment_name>/``. 4096 imagined rollouts need about 12 GB of
GPU memory; on smaller GPUs lower ``--agent.algorithm.policy.num-rollouts`` (512 fits in 6
GB). ``rl/supermimic/COLAB.md`` has copy-paste cells for training on Google Colab.

``torch.compile`` (``--agent.compile``, on by default) needs Triton, which Windows does not
have: there the runner falls back to eager mode, with a warning. To play a checkpoint on
Windows, set ``TORCHDYNAMO_DISABLE=1``.

Every checkpoint is exported to ONNX next to it (``model_N.onnx``): the actor, taking the
goal and proprioception groups concatenated, with the robot's joint names, defaults,
action scales and gains, plus the command's own metadata.


Key parameters
--------------

The runner config is ``SuperMimicRunnerCfg``; its ``algorithm`` is ``SuperMimicCfg``, with
one section per update. On the command line they are ``--agent.<field>`` and
``--agent.algorithm.<section>.<field>``.

.. list-table::
   :header-rows: 1
   :widths: 38 12 50

   * - Parameter
     - Default
     - Description
   * - ``env_steps_per_iteration``
     - 2.5
     - Env steps collected per env and iteration: the update-to-data ratio is its
       inverse. Lower collects less per update (more sample-efficient, slower in time).
   * - ``action_noise``
     - 0.1
     - Std of the Gaussian noise on the collected actions: the only exploration.
   * - ``replay_steps``
     - 1000
     - Replay buffer length per env.
   * - ``random_episode_start``
     - False
     - Start each env at a random point of its episode, so time-outs spread out instead
       of arriving in synchronized waves. The velocity task turns it on.
   * - ``policy.horizon``
     - 32
     - Imagined steps :math:`H`. Longer sees further (a gait cycle is ~0.6 s = 30 steps)
       but compounds model error and costs memory and time linearly.
   * - ``policy.num_rollouts``
     - 4096
     - Imagined rollouts per policy update. Memory scales with it (BPTT through
       :math:`H` steps).
   * - ``policy.learning_rate``
     - 2e-4
     - Actor learning rate. 1e-4 worked better on the velocity task: a fast actor finds
       the world model's errors before the model can correct them.
   * - ``policy.max_grad_norm`` / ``max_grad_value``
     - 1.0 / 0.1
     - Global, then element-wise gradient clipping of the actor.
   * - ``policy.gamma``
     - 0.99
     - Discount.
   * - ``world_model.window``
     - 8
     - Autoregressive window length of the world model's loss.
   * - ``world_model.batches``
     - 10
     - World-model optimizer steps per iteration.
   * - ``critic.lam``
     - 0.95
     - :math:`\lambda` of the critic's targets.
   * - ``critic.target_update``
     - 0.0
     - Polyak rate of a target critic the targets bootstrap from (0: none, the paper's
       setting). The velocity task uses 0.05.
   * - ``critic.max_return``
     - None
     - Bound on bootstrapped values: the largest return the rewards allow. The velocity
       task computes it from its reward weights (``rl_cfg.MAX_RETURN``).
   * - ``lr_decay_iterations``
     - 250,000
     - Cosine decay of every learning rate to ``lr_final_ratio`` (0.03).
   * - ``compile``
     - True
     - ``torch.compile`` the kinematics, the imagined step and the world-model loss.


Building a task
---------------

A SuperMimic task is an ordinary mjlab task: a ``ManagerBasedRlEnvCfg`` with commands,
observations, rewards and terminations, registered with ``SuperMimicRunner``. What changes
is that the method has to recompute, on predicted states, everything the env computes.
Five pieces make that possible.

1. The env config is a ``SuperMimicEnvCfg``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is an mjlab env config with one extra field, ``imagination`` (an ``ImaginationCfg``):
what the method reads from the env by name. To start from an existing PPO task, convert
its config:

.. code-block:: python

   from pal_mjlab.rl.supermimic.names import ImaginationCfg, as_supermimic_env_cfg

   cfg = pal_kangaroo_flat_env_cfg(play=play)     # the PPO task
   ...                                            # changes below
   cfg.auto_reset = play                          # SuperMimic resets envs itself in training
   return as_supermimic_env_cfg(cfg, ImaginationCfg(...))

.. list-table:: ``ImaginationCfg``
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``command``
     - Name of the command term (it must provide a ``CommandModel``, below).
   * - ``failure_terms``
     - Termination terms that count as failures (not time-outs).
   * - ``goal_group``, ``proprio_group``
     - The actor's two observation groups: the goal (encoded by the actor's goal encoder)
       and the proprioception.
   * - ``action_term``
     - The joint-position action term. Its joints are the state the world model predicts.
   * - ``bodies``, ``sites``
     - Bodies and sites whose forward kinematics the terms need. Only these are
       computed: FK over every body costs about 50% more per imagined step.
   * - ``context``
     - Observation term giving :math:`\xi`, the randomized physics the world model and the
       critic see (friction, encoder offsets, CoM offset, joint damping, ...).
   * - ``passive_joints``
     - For a closed kinematic chain, a function giving the passive joints from the
       actuated ones, so the FK covers the bodies they move (the Kangaroo's knee
       linkage: ``kangaroo_passive_leg_joints``).

The robot must have a floating base and joint-position actions. The state is
heading-invariant: root height, tilt, root velocities in the root frame, and the action
joints' positions and velocities. Root height is measured from the env origin, which is
right on flat ground only.

2. Terms are single definitions on a ``Batch``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each observation, reward and failure term is written **once**, as a pure function of a
``Batch``: one step's tensors. The ``@imaginable`` decorator turns it into an mjlab term
that evaluates it on the simulator's state, and keeps the function for the method to call
on imagined states. The simulator and imagination thus compute the same thing by
construction.

.. code-block:: python

   from pal_mjlab.rl.supermimic.imagination import Batch, imaginable

   @imaginable
   def track_linear_velocity(b: Batch, std: float) -> torch.Tensor:
     actual = b.state[:, 7:10]                    # root linear velocity, root frame
     cmd = b.c["command"]                         # the command state
     error = torch.sum(torch.square(cmd[:, :2] - actual[:, :2]), dim=1)
     error = error + torch.square(actual[:, 2])
     return torch.exp(-error / std**2)

   cfg.rewards["track_linear_velocity"] = RewardTermCfg(
     func=track_linear_velocity, weight=4.0, params={"std": 1.0}
   )

A ``Batch`` holds all a term may read:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Content
   * - ``state``
     - The heading-invariant state (``state.py``): root height, tilt (6D), root linear
       and angular velocity in the root frame, joint positions and velocities.
   * - ``pose``
     - The root's world (x, y, yaw) in the env-local frame.
   * - ``qpos``, ``qvel``
     - The same state in MuJoCo layout (floating base, then the action joints).
   * - ``kin``
     - FK of the configured ``bodies`` and ``sites``: positions, orientations, velocities.
       Index them with ``b.spec.body(name)`` and ``b.spec.site(name)``.
   * - ``c``, ``command_model``
     - The command state and its model.
   * - ``action``, ``prev_action``
     - The last two actions.
   * - ``context``
     - :math:`\xi`.
   * - ``spec``
     - Constants: joint defaults and limits, ``step_dt``, the MuJoCo model, cached lookups
       (``b.spec.cached``).
   * - ``memo``
     - A per-batch cache for what several terms share (e.g. foot heights).

What is not in the batch cannot be used: contact forces, sensor readings, height scans,
other entities. Those terms are rewritten from geometry. The velocity task detects foot
contact from the lowest point of each sole's collision capsules (``capsule_bottoms``),
and illegal knee contacts from the knee linkage's capsules.
``pal_mjlab.rl.supermimic.terms`` has single-definition versions of mjlab's common terms
(``joint_pos_rel``, ``joint_vel_rel``, ``last_action``, ``action_rate_l2``, an IMU's
``site_ang_vel`` and ``site_projected_gravity``, ``heading_state``, ``context``).

**Stateful terms.** A term that remembers across steps (a foot's time in the air, its
peak height in a swing) declares its memory. The memory is zeroed at resets, stored in
replay with each transition, and carried through the imagined rollout:

.. code-block:: python

   def _air_memory(b, asset_cfg, **_):
     return {"air_time": b.state.new_zeros(len(b.state), len(asset_cfg.site_names))}

   @imaginable(memory=_air_memory)
   def feet_air_time(b, memory, asset_cfg, threshold_min, threshold_max, ...):
     contact = ...                                       # smooth, in [0, 1]
     air = (memory["air_time"] + b.spec.step_dt) * (1.0 - contact)
     reward = ...
     return reward, {"air_time": air}

**Timing** is mjlab's: rewards see the command state the step targeted, observations the
advanced one.

3. Failures are smooth probabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A failure term returns a probability that is above 0.5 exactly where the condition holds,
usually a sigmoid of how far a limit is exceeded. ``@imaginable_termination`` makes the
env term (it fires above 0.5); imagination uses the probability itself, so its gradient
steers the policy away from failing:

.. code-block:: python

   from pal_mjlab.rl.supermimic.imagination import imaginable_termination

   @imaginable_termination
   def root_height_below(b: Batch, minimum_height: float, width: float = 0.01):
     return torch.sigmoid((minimum_height - b.state[:, 0]) / width)

List them in ``ImaginationCfg.failure_terms``. Give **every** failure a single
definition when you can. The world model also learns a failure head, but a policy finds
what that head misses: on the velocity task it learned to kneel when only the learned head
covered the knees.

4. The command provides a ``CommandModel``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Imagination has no env, so the command term provides a tensor-only twin through a
``command_model()`` method. The method treats the command state ``c`` as an opaque dict of
fixed-shape tensors.

.. code-block:: python

   class VelocityCommandModel:
     def state(self):                  # c of every env, read from the command term
       return {"command": self.cmd.command.clone()}
     def advance(self, c):             # c one step later: constant over the horizon
       return c
     def done(self, c):                # the command ends the episode: never
       return torch.zeros(len(c["command"]), dtype=torch.bool, device=c["command"].device)
     def termination_ref(self, c):     # (B, 2) target root height, projected gravity z
       ...

   class SuperMimicVelocityCommand(UniformVelocityCommand):
     def command_model(self):
       return VelocityCommandModel(self)

The command must be a function of its own state. That is why the velocity task turns
heading control off (the heading command depends on the robot's yaw) and commands the yaw
rate directly. Random resampling every few seconds is fine: imagination simply does not
foresee it.

The observation groups follow: the goal group (here the command,
``velocity_command``), the proprioception group (the PPO actor's terms, as single
definitions, with their noise), and a critic group that usually holds
``heading_state``, the goal and ``context``.

5. Register the task and check the contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   register_mjlab_task(
     task_id="Mjlab-SuperMimic-Velocity-Flat-Pal-Kangaroo",
     env_cfg=pal_kangaroo_flat_supermimic_env_cfg(),
     play_env_cfg=pal_kangaroo_flat_supermimic_env_cfg(play=True),
     rl_cfg=pal_kangaroo_velocity_supermimic_runner_cfg(),
     runner_cls=SuperMimicRunner,
   )

The runner's ``obs_groups`` must be ``{"actor": (goal_group, proprio_group), "critic":
(<one group>,)}``.

``check_contract`` drives the env with random actions and, at every step, compares what
imagination computes from the resulting state with what the env computed: each weighted
reward term, each observation, the command model's ``advance`` and ``done``, and each
failure. A term without a single definition is listed in ``report.missing``. Every task
should test it:

.. code-block:: python

   from pal_mjlab.rl.supermimic.contracts import check_contract

   report = check_contract(env, steps=120, action_std=0.5, resampled_commands=True)
   assert report.missing == []
   name, err = report.worst("reward/")
   assert err <= 1e-4, name

Errors are float32 noise (around 1e-6) when the inputs match. A large error means a term
reads something the batch does not reconstruct.


Tuning
------

A task tuned for PPO usually does not train with SuperMimic as it is. Both methods
maximize the same return, but they see it differently. PPO only needs a reward to
**rank** actions it tried. SuperMimic follows the reward's **slope** at the state it is
in: where the slope is zero or points to a poor local optimum, nothing moves the policy,
however good the reward is elsewhere. The rules below come from the velocity task.

Rewards
~~~~~~~

**Keep kernels wide enough to have a slope where the policy is.** A tracking kernel
:math:`\exp(-e^2/\sigma^2)` has gradient :math:`-2e/\sigma^2 \cdot \exp(-e^2/\sigma^2)`,
which vanishes far from the target. With PPO's :math:`\sigma = 0.5`, a standing robot
commanded at 0.8 m/s earns 0.08 of the reward and almost no slope toward moving. With
:math:`\sigma = 1` it earns 0.53, with a clear slope. For precision, add a narrow kernel
next to the wide one rather than narrowing it. The velocity task uses a wide kernel
(weight 4, std 1.0) plus a fine one (weight 2, std 0.25).

**Replace hard thresholds with soft ones.** Indicators, windows and ``>`` comparisons have
zero gradient almost everywhere. mjlab's ``feet_air_time`` rewards a time in the air
inside ``[min, max]``: as an indicator, imagination cannot see it at all. Here each edge is
a sigmoid (``edge_width`` 0.01 s), and contact itself is a sigmoid of the sole's height.
Make the soft version agree with the hard one away from the edges, so the two methods
still optimize the same thing.

**Pair saturating rewards with a penalty.** A kernel reward also loses its gradient far
from its target, exactly where the policy needs pulling back. The pose reward is
:math:`\exp(-e)`; the velocity task adds ``posture_l2`` (:math:`-0.1 \cdot e`, the same
error as a penalty), whose gradient grows with the deviation.

**Make the goal pay more than staying safe.** If standing still already earns most of the
reward (upright, posture, the yaw kernel at small yaw commands), "stand safely" is a local
optimum that gradients do not leave. PPO escapes it by sampling; a deterministic
gradient method needs the tracking terms to dominate. The velocity task raised linear
tracking from 2 to 4 + 2, then raised the pose weight from 1 to 2 when the policy started
tracking in odd postures.

**Look for proxies a policy can satisfy without the intended behavior.** Gradients find
them fast. Foot contact first read the foot site's height: a foot touching down on an edge
of its sole kept its site 3 cm up, so the policy dragged a tilted foot that counted as
airborne (full air-time reward, no slip penalty). Calibrate every proxy against the sensor
it replaces, on a **trained** policy, not only on random actions.

**Bound the penalties.** Unbounded penalties (squares of velocities, limit violations)
on states the world model has never seen can explode and dominate the gradient. Prefer
terms the robot can actually reach zero on, and failures that end the episode before
penalties grow.

Critic and policy
~~~~~~~~~~~~~~~~~

- **Critic inflation.** The critic bootstraps from itself, and the policy follows the
  critic's gradient at the horizon. On the velocity task its values grew without bound
  (121 with negative rewards). Two settings fixed it: a target critic
  (``critic.target_update`` 0.05) and a return bound (``critic.max_return``, the
  positive weights x dt / (1 - gamma)). **Update the bound when you change a positive
  reward weight**, or pass ``--agent.algorithm.critic.max-return``.
- **World-model exploitation.** When ``policy/imagined_reward`` rises above
  ``Step/reward``, the policy is exploiting model errors. Lower
  ``policy.learning_rate`` (1e-4) so the model keeps up.
- **Horizon.** :math:`H` should cover the behavior's time scale: 16 steps (0.32 s) is
  shorter than a step; 32 is the paper's. Longer horizons cost linearly in time and
  memory and compound model error.
- **Exploration** is the collection noise only (``action_noise``). Raising it (0.3) did not
  help the velocity task leave standing; reshaping the rewards did.
- **Data budget.** Judge progress in env steps. SuperMimic should need fewer env steps than
  PPO, but each one costs more wall-clock time.

Reading the logs
~~~~~~~~~~~~~~~~

mjlab's ``Episode_Reward/*``, ``Episode_Termination/*`` and ``Metrics/*`` are sums over an
episode, averaged over the episodes that **ended** in the logging window. With few
falls, those are mostly synchronized time-outs, so the curves follow episode lengths and
can contradict each other. SuperMimic also logs per-step means over all envs, which read
like a PPO run's:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Tag
     - Meaning
   * - ``Step_Reward/<term>``, ``Step_Reward/total``
     - Each term's weighted reward per second, averaged over all envs.
   * - ``Step/mean_time_to_failure_s``
     - Env time per failure (capped at 3600 s).
   * - ``Step_Termination/<term>``
     - Terminations per env and minute, by cause.
   * - ``Step_Metrics/<command>/<metric>``
     - The command's ``step_metrics()``. For velocity: ``error_vel_xy`` against
       ``standing_error_xy`` (what standing still would score), and ``speed_xy``.
   * - ``Step/reward`` vs ``policy/imagined_reward``
     - Real vs imagined reward per step: the gap shows exploitation of the world model.
   * - ``policy/value`` vs ``policy/lambda_return``
     - The critic against its targets.
   * - ``world_model/*``, ``time/*``
     - World-model losses; time per phase of an iteration.

A command term can add its own per-step metrics by defining
``step_metrics() -> dict[str, Tensor]`` (one value per env).


The Kangaroo velocity task
--------------------------

``Mjlab-SuperMimic-Velocity-Flat-Pal-Kangaroo`` (``tasks/supermimic_velocity/``) is the
PPO flat velocity task of the base Kangaroo (:ref:`Kangaroo task_velocity`), converted as
described above:

- heading control off, pushes removed (the world model cannot foresee them);
- goal = the velocity command; proprioception = the PPO actor's terms with their noise
  (the IMU from the IMU site's FK); critic = heading state, command and :math:`\xi`
  (foot friction, encoder offsets, torso CoM, joint friction, damping and armature);
- every reward a single definition of the PPO term, foot terms on the sole capsules'
  geometry. Dropped: ``self_collisions`` and ``angular_momentum`` (the passive linkage is
  not in the state);
- retuned for gradients (``env_cfgs.py`` constants): wide (4, std 1.0) and fine (2, std
  0.25) linear tracking, yaw std 1.0, soft-edged air time (0.5), pose 2 plus
  ``posture_l2``. Clearance, slip and the other PPO terms keep their weights. Soft
  landing uses the vertical speed at touchdown instead of the force (weight x2000);
- failures: tilt, root below 0.4 m, and the knee linkage's capsules touching the ground
  (the geometric version of the PPO task's contact sensor);
- a target critic and a return bound computed from the reward weights.
