.. _Kangaroo task_velocity:

PAL Kangaroo - Velocity tracking task
============

The velocity tracking environment configuration for Kangaroo is overwritten over mjlab's built-in environment.

Here is the list of observations used for basic locomotion training :

    Actor :

+------------------------------------+-----------+
| Name                               |   Shape   |
+------------------------------------+-----------+
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

    Critic :

+-------------------------------------+------------+
| Name                                |   Shape    |
+-------------------------------------+------------+
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


Actor observations are sufficient for the policy to infer proper behavior 
for the robot. The critic receives privileged observations --- richer in 
information --- which improve value function accuracy. Since the critic is 
only used during training and not at deployment, this asymmetry has no 
impact on real-world performance.