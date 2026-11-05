.. _Kangaroo task_box_lifting:

PAL Kangaroo - Box lifting task
======================================

The box lifting task trains Kangaroo to grasp a box placed in near of it and
raise it to a commanded height while maintaining balance. The environment
configuration extends mjlab's built-in velocity environment, overriding its
observations, rewards, events and terminations with Kangaroo-specific terms
for bimanual box manipulation.

In order to have proper contact signal, the arm collision capsules are 2, one
for the forearm and one smaller for the tip of the arm. This allows to keep
contacts along the arm while being able to target contact necessary for the
task.

Commands
--------

A box height command describes the desired height, in meters, that the box
should be raised to relative to the ground.

The command therefore has 1 component :

- Target box height (m)

|

During training, the target height is resampled periodically from a uniform
distribution, so the policy learns to lift the box to varying heights rather
than overfitting to a single target.

|

Observations
------------

Here is the list of observations used for box lifting training. Shapes
correspond to the full-body simple model (26 joints, 22 actuated); for the
lower-body variant, the joint and action dimensions shrink accordingly.

**Actor (shape: 87,):**

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
| command                            |    (1,)   |
+------------------------------------+-----------+
| box_position                       |    (3,)   |
+------------------------------------+-----------+
| imu_projected_gravity              |    (3,)   |
+------------------------------------+-----------+
| base_lin_acc                       |    (3,)   |
+------------------------------------+-----------+

**Critic (shape: 113,):**

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
| command                             |    (1,)    |
+-------------------------------------+------------+
| box_position                        |    (3,)    |
+-------------------------------------+------------+
| foot_height                         |    (2,)    |
+-------------------------------------+------------+
| foot_air_time                       |    (2,)    |
+-------------------------------------+------------+
| foot_contact                        |    (2,)    |
+-------------------------------------+------------+
| foot_contact_forces                 |    (6,)    |
+-------------------------------------+------------+
| hand_to_box_contact                 |    (2,)    |
+-------------------------------------+------------+
| hand_to_box_contact_forces          |    (6,)    |
+-------------------------------------+------------+
| imu_projected_gravity               |    (3,)    |
+-------------------------------------+------------+
| base_lin_acc                        |    (3,)    |
+-------------------------------------+------------+

|

Actor observations are limited to signals available on the real robot (IMU,
joint encoders, previous actions, estimated box position). The critic receives
privileged observations --- richer in information, such as the true base
linear velocity, foot contact states, and hand-to-box contact forces --- which
improve value function accuracy. Since the critic is only used during training
and not at deployment, this asymmetry has no impact on real-world performance.

A vision pipeline will be the one to determine the box estimate position. The
observation noise during training should almost match the accuracy of said 
pipeline.

.. important::

   Simulation observations must match deployment observations in order,
   distribution and units. Any mismatch is likely to make the robot behave
   erratically on hardware.


Rewards
-------

Here is a table with the rewards used in the baseline of the box lifting task :

+--------------------------------+--------+----------------------+
| Name                           | Weight |         Type         |
+================================+========+======================+
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
| box_proximity                  |    2.0 | objective            |
+--------------------------------+--------+----------------------+
| hands_to_box                   |    1.5 | objective            |
+--------------------------------+--------+----------------------+
| hands_contact                  |    1.0 | objective            |
+--------------------------------+--------+----------------------+
| box_height                     |    2.0 | objective            |
+--------------------------------+--------+----------------------+
| box_flat                       |    0.5 | tuning               |
+--------------------------------+--------+----------------------+
| look_at_box                    |    1.0 | tuning               |
+--------------------------------+--------+----------------------+
| horizontal_vel_penalty         |   -0.5 | regularization       |
+--------------------------------+--------+----------------------+
| ang_vel_penalty                |   -0.5 | regularization       |
+--------------------------------+--------+----------------------+
| grounded                       |    1.0 | objective            |
+--------------------------------+--------+----------------------+
| close_hands_penalty            |   -0.5 | regularization       |
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
(approach the box, make hand contact, raise it to the commanded height while
keeping it level and staying grounded), *limits* terms penalize violations of
physical constraints (joint ranges, velocity limits, self-collisions), *regularization*
terms smooth the resulting motion and discourage unwanted base or hand
movement, and *tuning* terms shape the supporting gait and posture (swing
height, air time, landing softness, gaze direction toward the box). The
baseline weights work consistently, but they are not guaranteed to be optimal
--- tweaking them is the main lever for obtaining different lifting behaviors.

|

You will notice that some reward terms rely on the distance to the box to be active
or not. You will notice that the 'box_proximity' reward has a distance lower than
the other reward. This is done deliberately to avoid that the robot stops close to said
distance and that the other rewards are being constantly actibvated/deactivated.

Terminations
------------

An episode ends when one of the following conditions is met :

- **time out** — the maximum episode length is reached (treated as a truncation,
  not a failure)

- **fell over** — the base exceeds an unrecoverable roll/pitch tilt

- **out of terrain bounds** — the robot moves outside the bounds of the
  terrain (treated as a truncation, not a failure)

- **illegal contacts** — a link other than the feet or hands (e.g. a femur or
  knee link) touches the terrain or the box