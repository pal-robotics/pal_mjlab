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

Unlike velocity tracking, the command is not user-specified at runtime: it is sampled
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

Actor observations are limited to signals available on the real robot. The
critic receives privileged observations --- richer in information, such as
global body poses and the true base linear velocity --- which improve value
function accuracy. Since the critic is only used during training and not at
deployment, this asymmetry has no impact on real-world performance.

.. important::

   Simulation observations must match deployment observations in order,
   distribution and units. Any mismatch is likely to make the robot behave
   erratically on hardware.


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

An episode ends when one of the following conditions is met :

- **time out**: the end of the reference motion / maximum episode length is
  reached (treated as a truncation, not a failure)

- **anchor deviation**: the anchor body (base link) drifts too far from the
  reference position or orientation

- **end-effector deviation**: a tracked end-effector body (feet, hands) drifts
  too far from its reference position

Early termination on tracking failure prevents the policy from wasting samples
on unrecoverable states and is a key ingredient for stable imitation training.


Motion file
-----------

The reference motion must be provided as a ``.npz`` file. A script is provided
to convert motions from ``.csv`` to ``.npz``, and `GMR (General Motion
Retargeting) <https://github.com/YanjieZe/GMR>`_ supports retargeting motions
to PAL Robotics' Kangaroo platform.

.. warning::

   The motion file must be sampled at the same framerate as the control
   frequency used during training (if control runs at 50 Hz, the motion must be
   50 Hz). A framerate mismatch between control and motion can result in
   unexpected and most likely unstable behaviors, or training getting stuck
   early on.
