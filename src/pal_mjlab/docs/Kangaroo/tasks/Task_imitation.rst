.. _Kangaroo task_motion_imitation:

PAL Kangaroo - Motion Imitation Task
=====================================

The motion imitation environment trains the Kangaroo robot to reproduce reference motions
provided as input commands.


Commands
--------

The command for motion imitation is a 52-dimensional vector encoding the reference motion
state at each timestep. It includes reference root position and orientation in the global
frame, per-body target positions and orientations in the local frame, and per-body target
linear and angular velocities.

|

Unlike velocity tracking, the command is not user-specified at runtime — it is sampled
from a motion dataset and streamed to the policy frame by frame during rollouts.

|

Observations
------------

Here is the list of observations used for motion imitation training :

**Actor :**

+------------------------------------+-----------+
| Name                               |   Shape   |
+====================================+===========+
| command                            |   (52,)   |
+------------------------------------+-----------+
| base_ang_vel                       |    (3,)   |
+------------------------------------+-----------+
| joint_pos                          |   (26,)   |
+------------------------------------+-----------+
| joint_vel                          |   (26,)   |
+------------------------------------+-----------+
| actions                            |   (22,)   |
+------------------------------------+-----------+

**Critic :**

+-------------------------------------+------------+
| Name                                |   Shape    |
+=====================================+============+
| command                             |   (52,)    |
+-------------------------------------+------------+
| motion_anchor_pos_b                 |    (3,)    |
+-------------------------------------+------------+
| motion_anchor_ori_b                 |    (6,)    |
+-------------------------------------+------------+
| body_pos                            |   (42,)    |
+-------------------------------------+------------+
| body_ori                            |   (84,)    |
+-------------------------------------+------------+
| base_lin_vel                        |    (3,)    |
+-------------------------------------+------------+
| base_ang_vel                        |    (3,)    |
+-------------------------------------+------------+
| joint_pos                           |   (26,)    |
+-------------------------------------+------------+
| joint_vel                           |   (26,)    |
+-------------------------------------+------------+
| actions                             |   (22,)    |
+-------------------------------------+------------+

|

Actor observations are sufficient for the policy to infer proper behavior
for the robot. The critic receives privileged observations --- richer in
information --- which improve value function accuracy. Since the critic is
only used during training and not at deployment, this asymmetry has no
impact on real-world performance.

| It is essential that simulation observations match deployment observations in order, distribution and units. Otherwise, the robot is likely to behave very erratically.


Rewards
-------

Here is a table with the rewards used in the baseline of the motion imitation task :

+--------------------------------+--------+--------------------+
| Name                           | Weight |         Type       |
+================================+========+====================+
| motion_global_root_pos         |    0.5 | objective          |
+--------------------------------+--------+--------------------+
| motion_global_root_ori         |    0.5 | objective          |
+--------------------------------+--------+--------------------+
| motion_body_pos                |    1.0 | objective          |
+--------------------------------+--------+--------------------+
| motion_body_ori                |    1.0 | objective          |
+--------------------------------+--------+--------------------+
| motion_body_lin_vel            |    1.0 | objective          |
+--------------------------------+--------+--------------------+
| motion_body_ang_vel            |    1.0 | objective          |
+--------------------------------+--------+--------------------+
| action_rate_l2                 |   -0.1 | regularization     |
+--------------------------------+--------+--------------------+
| joint_limit                    |  -10.0 | limits             |
+--------------------------------+--------+--------------------+
| self_collisions                |  -10.0 | limits             |
+--------------------------------+--------+--------------------+
| convex_hull_joint_limits_hip   |  -10.0 | limits             |
+--------------------------------+--------+--------------------+
| convex_hull_joint_limits_ankle |  -10.0 | limits             |
+--------------------------------+--------+--------------------+

|

Motion imitation rewards measure the discrepancy between the robot's current state and the
reference motion at each timestep. Objective rewards encourage the policy to match root and
body kinematics precisely. Limit rewards penalize joint-range violations and self-collisions
to ensure physically safe behavior.


Terminations
------------

An episode is terminated when certain conditions are met. In this case, those conditions are the following :

- illegal contacts

- excessive deviation from reference motion (early termination on tracking failure)