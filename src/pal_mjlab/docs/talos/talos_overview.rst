.. _Talos overview:

PAL TALOS
=========

TALOS is PAL Robotics' full-size humanoid robot.

.. image:: /_static/talos.png
   :alt: TALOS model
   :width: 300px
   :align: center

The model has 30 actuated revolute joints and no passive joints:

- torso: ``torso_1_joint``, ``torso_2_joint``
- head: ``head_1_joint``, ``head_2_joint``
- arms: ``arm_{left,right}_{1..7}_joint`` (7 per arm)
- legs: ``leg_{left,right}_{1..6}_joint`` (6 per leg)

Robot parameters (actuators, initial pose, collisions) are defined in
``talos_constants.py``.

Tasks
-----

.. list-table::
   :header-rows: 1
   :widths: 60 20 20

   * - Task ID
     - Objective
     - Terrain
   * - ``Mjlab-Velocity-Flat-Pal-Talos``
     - Velocity
     - Flat
   * - ``Mjlab-Velocity-Rough-Pal-Talos``
     - Velocity
     - Rough
   * - ``Mjlab-Tracking-Flat-Pal-Talos``
     - Tracking
     - Flat
   * - ``Mjlab-Tracking-Flat-Pal-Talos-No-State-Estimation``
     - Tracking
     - Flat

The velocity tasks are built on mjlab's ``make_velocity_env_cfg`` and the
tracking tasks on mjlab's ``make_tracking_env_cfg``, with TALOS-specific
overrides.

Velocity task
^^^^^^^^^^^^^

**Actor:** ``base_lin_vel``, ``base_ang_vel``, ``projected_gravity``,
``joint_pos``, ``joint_vel``, ``actions``, ``command``.

**Critic:** the actor terms plus ``foot_height``, ``foot_air_time``,
``foot_contact`` and ``foot_contact_forces``.

Rewards of ``Mjlab-Velocity-Flat-Pal-Talos``:

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Name
     - Weight
   * - track_linear_velocity
     - 2.0
   * - track_angular_velocity
     - 2.0
   * - upright
     - 1.0
   * - pose
     - 1.0
   * - body_ang_vel
     - -0.05
   * - angular_momentum
     - -0.02
   * - dof_pos_limits
     - -1.0
   * - action_rate_l2
     - -0.1
   * - air_time
     - 0.0
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


Terminations: time out, fell over, out of terrain bounds, illegal contacts.

Tracking task
^^^^^^^^^^^^^

**Actor:** ``command``, ``motion_anchor_pos_b``, ``motion_anchor_ori_b``,
``base_lin_vel``, ``base_ang_vel``, ``joint_pos``, ``joint_vel``,
``actions``. The ``-No-State-Estimation`` variant drops
``motion_anchor_pos_b``, ``motion_anchor_ori_b`` and ``base_lin_vel``.

**Critic:** the actor terms plus ``body_pos`` and ``body_ori``.

Rewards:

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Name
     - Weight
   * - motion_global_root_pos
     - 0.5
   * - motion_global_root_ori
     - 0.5
   * - motion_body_pos
     - 1.0
   * - motion_body_ori
     - 1.0
   * - motion_body_lin_vel
     - 1.0
   * - motion_body_ang_vel
     - 1.0
   * - action_rate_l2
     - -0.1
   * - joint_limit
     - -10.0
   * - self_collisions
     - -10.0


Terminations: time out, anchor position / orientation deviation,
end-effector position deviation.

A retargeted TALOS walking motion is provided in ``motions/talos_walk.csv``
(see the README for converting it to ``.npz``).
