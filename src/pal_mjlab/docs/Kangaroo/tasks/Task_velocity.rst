.. _Kangaroo task_velocity:

PAL Kangaroo - Velocity tracking task
======================================

The velocity tracking task trains Kangaroo to follow user-specified velocity
commands while walking. The environment configuration extends mjlab's built-in
velocity environment, overriding its observations, rewards, events and
terminations with Kangaroo-specific terms.


Commands
--------

A velocity command describes a desired horizontal linear velocity expressed in
the robot's base frame, together with a desired angular velocity around the
vertical axis (yaw).

The command therefore has 3 components :

- X-aligned velocity (m/s)

- Y-aligned velocity (m/s)

- Angular velocity around Z axis (rad/s)

|

The command is independent of the robot's roll and pitch orientation and of its
world position. During training, commands are resampled periodically from a
uniform distribution, so the policy learns to transition between velocities.

|

Observations
------------

Here is the list of observations used for basic locomotion training. Shapes
correspond to the full-body simple model (26 joints, 22 actuated); for the
lower-body variant, the joint and action dimensions shrink accordingly.

**Actor :**

+------------------------------------+-----------+
| Name                               |   Shape   |
+====================================+===========+
| base_ang_vel                       |    (3,)   |
+------------------------------------+-----------+
| joint_pos                          |   (26,)   |
+------------------------------------+-----------+
| joint_vel                          |   (26,)   |
+------------------------------------+-----------+
| actions                            |   (22,)   |
+------------------------------------+-----------+
| command                            |    (3,)   |
+------------------------------------+-----------+
| imu_projected_gravity              |    (3,)   |
+------------------------------------+-----------+
| base_lin_acc                       |    (3,)   |
+------------------------------------+-----------+

**Critic :**

+-------------------------------------+------------+
| Name                                |   Shape    |
+=====================================+============+
| base_lin_vel                        |    (3,)    |
+-------------------------------------+------------+
| base_ang_vel                        |    (3,)    |
+-------------------------------------+------------+
| projected_gravity                   |    (3,)    |
+-------------------------------------+------------+
| joint_pos                           |   (26,)    |
+-------------------------------------+------------+
| joint_vel                           |   (26,)    |
+-------------------------------------+------------+
| actions                             |   (22,)    |
+-------------------------------------+------------+
| command                             |    (3,)    |
+-------------------------------------+------------+
| foot_height                         |    (2,)    |
+-------------------------------------+------------+
| foot_air_time                       |    (2,)    |
+-------------------------------------+------------+
| foot_contact                        |    (2,)    |
+-------------------------------------+------------+
| foot_contact_forces                 |    (6,)    |
+-------------------------------------+------------+
| imu_projected_gravity               |    (3,)    |
+-------------------------------------+------------+
| base_lin_acc                        |    (3,)    |
+-------------------------------------+------------+

|

Actor observations are limited to signals available on the real robot (IMU,
joint encoders, previous actions). The critic receives privileged observations
--- richer in information, such as the true base linear velocity and foot
contact states --- which improve value function accuracy. Since the critic is
only used during training and not at deployment, this asymmetry has no impact
on real-world performance.

.. important::

   Simulation observations must match deployment observations in order,
   distribution and units. Any mismatch is likely to make the robot behave
   erratically on hardware.


Rewards
-------

Here is a table with the rewards used in the baseline of the velocity tracking task :

+--------------------------------+--------+----------------------+
| Name                           | Weight |         Type         |
+================================+========+======================+
| track_linear_velocity          |    2.0 | objective            |
+--------------------------------+--------+----------------------+
| track_angular_velocity         |    2.0 | objective            |
+--------------------------------+--------+----------------------+
| upright                        |    1.0 | objective            |
+--------------------------------+--------+----------------------+
| pose                           |    1.0 | regularization       |
+--------------------------------+--------+----------------------+
| body_ang_vel                   |  -0.05 | regularization       |
+--------------------------------+--------+----------------------+
| angular_momentum               |  -0.02 | regularization       |
+--------------------------------+--------+----------------------+
| dof_pos_limits                 |   -1.0 | limits               |
+--------------------------------+--------+----------------------+
| action_rate_l2                 |   -0.1 | regularization       |
+--------------------------------+--------+----------------------+
| air_time                       |   0.25 | tuning               |
+--------------------------------+--------+----------------------+
| foot_clearance                 |   -2.0 | tuning               |
+--------------------------------+--------+----------------------+
| foot_swing_height              |  -0.25 | tuning               |
+--------------------------------+--------+----------------------+
| foot_slip                      |   -0.1 | tuning               |
+--------------------------------+--------+----------------------+
| soft_landing                   | -1e-05 | tuning               |
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

Each reward term falls into one of four roles: *objective* terms drive the task
(track the commanded velocity, stay upright), *limits* terms penalize violations
of physical constraints (joint ranges, velocity limits, self-collisions),
*regularization* terms smooth the resulting motion, and *tuning* terms shape the
gait style (swing height, air time, landing softness). The baseline weights work
consistently, but they are not guaranteed to be optimal — tweaking them is the
main lever for obtaining different behaviors.

Terminations
------------

An episode ends when one of the following conditions is met :

- **time out** — the maximum episode length is reached (treated as a truncation,
  not a failure)

- **fell over** — the base exceeds an unrecoverable roll/pitch tilt

- **illegal contacts** — a femur or knee link touches the terrain