.. _Kangaroo task_velocity:

PAL Kangaroo - Velocity tracking task
============

The velocity tracking environment configuration for Kangaroo is overwritten over mjlab's built-in environment.


Commands
--------

Here, a velocity command describe a desired linear horizontal velocity in the robot's frame, cartesian space, and a desired angular velocity around the Z-axis (yaw).

Therefore, velocity command has 3 components :

- X-aligned velocity (m/s)

- Y-aligned velocity (m/s)

- Angular velocity around Z axis (rad/s)

|

This command is independent of the robot's roll and pitch orientation, and independent of its world position.

|

Observations
------------

Here is the list of observations used for basic locomotion training :

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

Actor observations are sufficient for the policy to infer proper behavior 
for the robot. The critic receives privileged observations --- richer in 
information --- which improve value function accuracy. Since the critic is 
only used during training and not at deployment, this asymmetry has no 
impact on real-world performance.

| it is essential that simulation observations match deployment observations in order, distribution and units. Otherwise, the robot is likely to behave very erratically.


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

Every reward tries to shape behavior to track objectives, respect limits, minimize costs or tune behavior in a certain direction. 
Although baseline rewards with their given default weights work consistenly, it is not guaranteed to be the best solution, reward 
weights can be tweaked to achieve different kinds of behaviors.

Terminations
------------

An episode is terminated when certain conditions are met. In this case, those conditions are the following :

- episode boundary (max episode length)

- fell over (unrecoverable roll/pitch tilt)

- illegal contacts