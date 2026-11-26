.. _Kangaroo task_supermimic:

PAL Kangaroo - SuperMimic Motion Tracking Task
===============================================

SuperMimic (`arXiv 2610.28480 <https://arxiv.org/abs/2610.28480>`_) trains one
policy to track any motion of a large motion library. It learns a world model
of the simulator online and improves the policy by backpropagating the tracking
reward through 32 steps imagined with that model, plus a critic's value beyond
them. Episodes start at frames picked by a return-based sampler that favours
the parts of the library the policy tracks worst.

The task is ``Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers``: the Kangaroo with
7-DoF arms and grippers on flat ground. It follows the paper's observations,
reward, termination, domain randomization and hyperparameters.


Motion library
--------------

The task reads a **motion library**, a folder that packs the clips of a
retargeted dataset in three files:

+----------------+---------------------------------------------------------------+
| File           | Content                                                       |
+================+===============================================================+
| ``clips.json`` | ``fps``, ``joint_names`` and, per clip, its ``name``,         |
|                | ``category``, ``start`` frame and ``length`` in frames        |
+----------------+---------------------------------------------------------------+
| ``qpos.npy``   | (frames, 35) float32: root position [m], root quaternion      |
|                | (w, x, y, z), 28 joint positions [rad; m for the leg lengths] |
+----------------+---------------------------------------------------------------+
| ``qvel.npy``   | (frames, 34) float32: root linear velocity (world frame),     |
|                | root angular velocity (body frame), 28 joint velocities       |
+----------------+---------------------------------------------------------------+

The clips are concatenated in time at 50 Hz, the control rate. The 28 joints are
the robot's actuated joints, in the robot's order:

.. code-block:: text

   pelvis_1, pelvis_2,
   arm_left_1 ... arm_left_7, arm_right_1 ... arm_right_7,
   leg_left_1, leg_left_2, leg_left_3, leg_left_length, leg_left_4, leg_left_5,
   leg_right_1, leg_right_2, leg_right_3, leg_right_length, leg_right_4, leg_right_5

A Kangaroo leg is a slide between hip and ankle (``leg_*_length_joint``) with a
passive knee linkage closed on it (``leg_*_femur_joint``, ``leg_*_knee_joint``).
Motions are retargeted to the leg length only, so the library does not hold the
linkage: the femur and the tibia are equally long, and their angles follow from
the leg length (``kangaroo_passive_leg_joints`` in ``kangaroo_constants.py``).

Libraries are written by the SOMA retargeter (``soma-retargeter``, branch
``feat/kangaroo_grippers``) with the ``kangaroo_grippers`` target and
``"export_format": "supermimic"``:

.. code-block:: bash

   uv run python app/bvh_to_csv_converter.py --config assets/kangaroo_grippers_supermimic_config.json

Without ``--env.commands.motion.library-path``, the task reads
``artifacts/kangaroo_grippers_library`` under the working directory.


Robot mapping
-------------

The paper tracks the bodies of BeyondMimic's G1 set. On the Kangaroo they are 12
bodies, the first one being the root:

+--------------------------+----------------------------------------------+
| Body                     | Role                                         |
+==========================+==============================================+
| ``base_link``            | root                                         |
+--------------------------+----------------------------------------------+
| ``pelvis_2_link``        | torso, after the two waist joints            |
+--------------------------+----------------------------------------------+
| ``leg_*_3_link``         | hip                                          |
+--------------------------+----------------------------------------------+
| ``leg_*_5_link``         | foot                                         |
+--------------------------+----------------------------------------------+
| ``arm_*_2_link``         | shoulder                                     |
+--------------------------+----------------------------------------------+
| ``arm_*_4_link``         | elbow                                        |
+--------------------------+----------------------------------------------+
| ``arm_*_7_link``         | wrist                                        |
+--------------------------+----------------------------------------------+

The knee is not tracked: it follows from the hip, the foot and the leg length.

The actions are joint position targets, ``q0 + scale * action``, for the 28
actuated joints. The gains are the robot's. The scale is BeyondMimic's,
``0.25 * effort / Kp``; for the arms and the waist it is computed with the
stiffness of 1 Hz gains instead of the robot's 3.5 Hz, which makes it 12 times
larger (``ARM_WAIST_ACTION_SCALE_FREQ_HZ`` in
``tasks/supermimic/kangaroo/env_cfgs.py``).


Observations
------------

**Actor :**

+------------------------------------+-----------+--------------------------------------------+
| Name                               |   Shape   | Content                                    |
+====================================+===========+============================================+
| goal                               |  (620,)   | next 10 reference frames: joint positions, |
|                                    |           | joint velocities, root orientation in the  |
|                                    |           | robot root frame (6D)                      |
+------------------------------------+-----------+--------------------------------------------+
| joint_pos                          |  (280,)   | relative to the default pose               |
+------------------------------------+-----------+--------------------------------------------+
| joint_vel                          |  (280,)   |                                            |
+------------------------------------+-----------+--------------------------------------------+
| imu_ang_vel                        |   (30,)   | IMU on the torso                           |
+------------------------------------+-----------+--------------------------------------------+
| imu_projected_gravity              |   (30,)   |                                            |
+------------------------------------+-----------+--------------------------------------------+
| actions                            |  (280,)   | previous actions                           |
+------------------------------------+-----------+--------------------------------------------+

The proprioceptive terms hold the last 10 control steps, with sensor noise. There
is no base linear velocity: the policy needs no state estimator.

**Critic :**

+------------------------------------+-----------+--------------------------------------------+
| Name                               |   Shape   | Content                                    |
+====================================+===========+============================================+
| heading_state                      |   (69,)   | root height, tilt and velocities in the    |
|                                    |           | root frame; joint positions and velocities |
+------------------------------------+-----------+--------------------------------------------+
| motion_goal                        |  (620,)   | the actor's goal                           |
+------------------------------------+-----------+--------------------------------------------+
| sim_context                        |   (32,)   | foot friction, joint calibration offsets,  |
|                                    |           | torso centre of mass offset                |
+------------------------------------+-----------+--------------------------------------------+
| root_pose_error                    |    (4,)   | reference root position and yaw against    |
|                                    |           | the robot's                                |
+------------------------------------+-----------+--------------------------------------------+
| time_left                          |    (1,)   | time left in the clip                      |
+------------------------------------+-----------+--------------------------------------------+

The heading state is also the state the world model predicts.


Rewards and terminations
------------------------

+------------------------------+--------+----------------------------------------------+
| Term                         | Weight | Content                                      |
+==============================+========+==============================================+
| motion_root_pos              |   0.5  | root position, ``exp(-e / 0.3²)``            |
+------------------------------+--------+----------------------------------------------+
| motion_root_ori              |   0.5  | root orientation, ``exp(-e / 0.4²)``         |
+------------------------------+--------+----------------------------------------------+
| motion_body_pos              |   1.0  | body positions relative to the root,         |
|                              |        | ``exp(-e / 0.3²)``                           |
+------------------------------+--------+----------------------------------------------+
| motion_body_ori              |   1.0  | body orientations, ``exp(-e / 0.4²)``        |
+------------------------------+--------+----------------------------------------------+
| motion_body_lin_vel          |   1.0  | body linear velocities, ``exp(-e / 1.0²)``   |
+------------------------------+--------+----------------------------------------------+
| motion_body_ang_vel          |   1.0  | body angular velocities, ``exp(-e / 3.14²)`` |
+------------------------------+--------+----------------------------------------------+
| action_rate_l2               |  -0.1  | change of the action                         |
+------------------------------+--------+----------------------------------------------+
| joint_limit                  |  -2.0  | distance outside the joint limits            |
+------------------------------+--------+----------------------------------------------+

An episode ends at the last frame of its clip, after 10 s, or when the robot
fails: its root is more than 0.25 m off the reference height, or its tilt error
exceeds 0.8.

Per episode, the foot friction is sampled in [0.3, 1.2] and every joint gets a
calibration offset within ±0.01 rad. The centre of mass of the torso is offset
once per environment. There are no pushes.


Training
--------

.. code-block:: bash

   uv run train Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers \
     --env.scene.num-envs 4096 \
     --env.commands.motion.library-path <library>

The paper trains with 4096 environments per GPU. Runs log to
``logs/rsl_rl/kangaroo_supermimic/`` in TensorBoard format; ``--agent.resume
True`` continues the last one. Training runs on one GPU.

Every checkpoint ``model_N.pt`` comes with ``model_N.onnx``, the policy for the
robot. Its input is the actor's observation, the goal first; its output is the
28 actions. Its metadata holds what is needed to build the observation and apply
the actions: the joint names, default positions, action scales and gains, the
observation layout and the IMU site.


Playing and evaluating
----------------------

.. code-block:: bash

   uv run play Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers --checkpoint-file <run>/model_N.pt

plays the motions of the default library, ``artifacts/kangaroo_grippers_library``,
with the paper's evaluation settings: every motion from its first frame, no
randomization and no noise. mjlab's ``play`` cannot choose another library.

Three scripts come with the task. Each takes the task and the library; without
``--library-path`` they read the default one:

.. code-block:: bash

   # Replay chosen motions of a library. A fall plays out for 3 s instead of resetting.
   uv run -m pal_mjlab.tasks.supermimic.scripts.play Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers \
     --checkpoint-file <run>/model_N.pt --library-path <library> --clips <name> ...

   # Benchmark checkpoints on every clip of a library: failures, tracking errors per
   # body group, torque use, foot sliding. One JSON per checkpoint.
   uv run -m pal_mjlab.tasks.supermimic.scripts.evaluate Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers \
     --checkpoint-files <run>/model_N.pt ... --library-path <library> --output-dir <folder>

   # View the motions as retargeted, without physics or a policy.
   uv run -m pal_mjlab.tasks.supermimic.scripts.view_motion Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers \
     --library-path <library>
