.. _Kangaroo task_reaching:

PAL Kangaroo - Reaching Task
=============================

The reaching task trains Kangaroo to move each arm's end effector to a
sampled target pose while also following a base velocity command, so that
reaching and walking are learned together.

Registered tasks:

- ``Mjlab-Reaching-Flat-Pal-Kangaroo``
- ``Mjlab-Reaching-Flat-Pal-Kangaroo-Hands``
- ``Mjlab-Reaching-Flat-Pal-Kangaroo-Grippers``


Commands
--------

- ``pose_command_left`` / ``pose_command_right``: a target pose for each
  end-effector site, sampled uniformly (``UniformPoseCommand``). For
  Kangaroo, the targets are sampled in ``x ∈ [-0.6, 0.6]`` m,
  ``|y| ∈ [0.2, 0.6]`` m (on the side of each arm) and ``z ∈ [-0.6, 0.6]`` m.
- ``twist``: a base velocity command, as in the
  :ref:`velocity task <Kangaroo task_velocity>`.


Observations
------------

**Actor:** ``joint_pos``, ``joint_vel``, ``actions``, ``pose_command_left``,
``pose_command_right``, ``base_ang_vel``, ``base_lin_acc``, ``twist_command``.

**Critic:** the actor terms plus ``foot_air_time``, ``foot_contact`` and
``foot_contact_forces``.


Rewards
-------

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Name
     - Weight
   * - pos_left
     - -2.0
   * - pos_left_fine_grained
     - 2.0
   * - ee_left_orientation
     - -0.2
   * - pos_right
     - -2.0
   * - pos_right_fine_grained
     - 3.0
   * - ee_right_orientation
     - -0.2
   * - dof_pos_limits
     - -1.0
   * - action_rate_l2
     - -0.1
   * - joint_vel_hinge
     - -0.05
   * - track_lin_vel
     - 2.0
   * - track_ang_vel
     - 2.0
   * - upright
     - 1.0
   * - body_ang_vel
     - -0.05
   * - angular_momentum
     - -0.02
   * - pose_loco
     - 1.0
   * - air_time
     - 0.25
   * - foot_clearance
     - -2.0
   * - foot_swing_height
     - -0.25
   * - foot_slip
     - -0.1
   * - soft_landing
     - -1e-05
   * - self_collisions
     - -1.0
   * - action_rate_arms_l2
     - -0.0001



Terminations
------------

- **time out**
- **fell over**
- **illegal contacts** — a femur or knee link touches the terrain
