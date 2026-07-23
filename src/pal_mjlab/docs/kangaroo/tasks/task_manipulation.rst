.. _Kangaroo task_manipulation:

PAL Kangaroo - Grippers manipulation task
==========================================

The grippers manipulation task trains Kangaroo to reach out with both arms,
make contact with a box resting on a table, and bring it to a target position
while remaining balanced. The environment configuration extends mjlab's
built-in manipulation environment, overriding its observations, rewards,
events and terminations with Kangaroo-specific terms.


Commands
--------

A gripper manipulation command describes the desired position of the box,
expressed relative to the robot's base frame.

The command therefore has 3 components:

- X-aligned target position (m)

- Y-aligned target position (m)

- Z-aligned target position (m)

|

Unlike the velocity task, the command here is resampled infrequently
(every 30 s by default), since the goal is a fixed reach-and-manipulate
target rather than a continuously varying motion. During training, the box
itself is also randomized in position and yaw at every reset, so the policy
learns to reach and grasp from a variety of starting configurations rather
than memorizing a single trajectory.

|

Observations
------------

Here is the list of observations used for the grippers manipulation task.
Shapes correspond to the Kangaroo grippers model (44 joints, 40 actuated).

**Actor (shape: 143):**

+------------------------------------+-----------+
| Name                               |   Shape   |
+====================================+===========+
| base_ang_vel                       |    (3,)   |
+------------------------------------+-----------+
| joint_pos                          |   (44,)   |
+------------------------------------+-----------+
| joint_vel                          |   (44,)   |
+------------------------------------+-----------+
| actions                            |   (40,)   |
+------------------------------------+-----------+
| command                            |    (3,)   |
+------------------------------------+-----------+
| box_position                       |    (3,)   |
+------------------------------------+-----------+
| imu_projected_gravity              |    (3,)   |
+------------------------------------+-----------+
| base_lin_acc                       |    (3,)   |
+------------------------------------+-----------+

**Critic (shape: 169):**

+----------------------------------------+------------+
| Name                                   |   Shape    |
+========================================+============+
| base_lin_vel                           |    (3,)    |
+----------------------------------------+------------+
| base_ang_vel                           |    (3,)    |
+----------------------------------------+------------+
| projected_gravity                      |    (3,)    |
+----------------------------------------+------------+
| joint_pos                              |   (44,)    |
+----------------------------------------+------------+
| joint_vel                              |   (44,)    |
+----------------------------------------+------------+
| actions                                |   (40,)    |
+----------------------------------------+------------+
| command                                |    (3,)    |
+----------------------------------------+------------+
| box_position                           |    (3,)    |
+----------------------------------------+------------+
| foot_height                            |    (2,)    |
+----------------------------------------+------------+
| foot_air_time                          |    (2,)    |
+----------------------------------------+------------+
| foot_contact                           |    (2,)    |
+----------------------------------------+------------+
| foot_contact_forces                    |    (6,)    |
+----------------------------------------+------------+
| hand_to_box_contact                    |    (2,)    |
+----------------------------------------+------------+
| hand_to_box_contact_forces             |    (6,)    |
+----------------------------------------+------------+
| imu_projected_gravity                  |    (3,)    |
+----------------------------------------+------------+
| base_lin_acc                           |    (3,)    |
+----------------------------------------+------------+

|

Actor observations are limited to signals available
on the real robot (IMU, joint encoders, previous actions, and an estimate of
box position). The critic additionally receives privileged information such
as the true base linear velocity, foot contact states, and hand-to-box
contact forces, which sharpen the value estimate during training without
affecting deployment.

|

Note that this pipeline is meant to be coupled with vision tools to detect the box. 
You may need to adjust or change the (``box_position``) in order to match the return
of your vision pipeline and have similar precision.

.. important::

   Simulation observations must match deployment observations in order,
   distribution and units. Any mismatch is likely to make the robot behave
   erratically on hardware.


Rewards
-------

Here is a table with the rewards used in the baseline of the grippers
manipulation task:

+--------------------------------+--------+----------------------+
| Name                           | Weight |         Type         |
+================================+========+======================+
| upright                        |   1.25 | objective            |
+--------------------------------+--------+----------------------+
| pose                           |    1.0 | regularization       |
+--------------------------------+--------+----------------------+
| hands_to_box                   |    3.0 | objective            |
+--------------------------------+--------+----------------------+
| hands_contact                  |    1.0 | objective            |
+--------------------------------+--------+----------------------+
| box_target_tracking            |    2.0 | objective            |
+--------------------------------+--------+----------------------+
| table_contact                  |   -2.0 | limits               |
+--------------------------------+--------+----------------------+
| dof_pos_limits                 |   -1.0 | limits               |
+--------------------------------+--------+----------------------+
| action_rate_l2                 |   -0.1 | regularization       |
+--------------------------------+--------+----------------------+
| self_collisions                |   -1.0 | limits               |
+--------------------------------+--------+----------------------+
| convex_hull_joint_limits_hip   |  -10.0 | limits               |
+--------------------------------+--------+----------------------+
| convex_hull_joint_limits_ankle |  -10.0 | limits               |
+--------------------------------+--------+----------------------+
| joint_vel_limits               |  -10.0 | limits               |
+--------------------------------+--------+----------------------+

|

*Objective* terms drive the manipulation goal: staying upright, closing the
distance between the hands and the box (``hands_to_box``), making and
holding contact with it (``hands_contact``), and moving the box to the
commanded target (``box_target_tracking``). *Limits* terms penalize
violations of physical or task constraints — joint ranges, self-collisions,
the femur/ankle convex-hull joint limits, and unwanted contact between the
robot's body and the table. *Regularization* terms (``pose``,
``action_rate_l2``) smooth the resulting motion. As with the other tasks, these
baseline weights are a starting point, not a guaranteed optimum.

Terminations
------------

An episode ends when one of the following conditions is met:

- **time out** — the maximum episode length is reached (treated as a
  truncation, not a failure)

- **fell over** — the base exceeds an unrecoverable roll/pitch tilt
  (limit angle of 70°)

- **out of terrain bounds** — the robot leaves the bounds of the terrain
  (treated as a truncation, not a failure)

- **box out of reach** — the box moves outside a reachable range from the
  robot

- **illegal contacts** — a femur or knee link touches the terrain
