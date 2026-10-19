.. _Kangaroo model_grippers:

Kangaroo - Grippers Model
==========================

The grippers model is the full-body Kangaroo equipped with a 7-DoF arm on
each side, ending in a parallel-jaw gripper. It is used for tasks that
require the robot to reach, grasp, and manipulate objects.

.. image:: /_static/kangaroo_7dof.png
   :alt: Kangaroo grippers model with 7 DoF arms and parallel-jaw grippers
   :width: 300px
   :align: center

It incorporates a simplified representation of Kangaroo's closed-chain leg
mechanism: the real linkage is reduced to a serial chain plus four passive
joints (femur and knee on each leg), while masses and inertias are computed
for all links so the dynamics remain faithful to the real robot. Each
gripper is also a closed-chain mechanism: only one joint per gripper is
directly actuated, while the remaining finger joints are coupled to it
through equality constraints (a joint-coupling constraint tying the two
inner fingers together, and connect constraints closing the loop between the
fingertips and outer fingers), reproducing the underactuated behavior of the
real hardware. Despite this mechanical coupling, all finger joints remain in
the actuator list for kinematic accuracy in simulation.

44 joints :

    40 actuators :

    - pelvis_1_joint *(revolute)*

    - pelvis_2_joint *(revolute)*

    - arm_left_1_joint *(revolute)*

    - arm_left_2_joint *(revolute)*

    - arm_left_3_joint *(revolute)*

    - arm_left_4_joint *(revolute)*

    - arm_left_5_joint *(revolute)*

    - arm_left_6_joint *(revolute)*

    - arm_left_7_joint *(revolute)*

    - arm_right_1_joint *(revolute)*

    - arm_right_2_joint *(revolute)*

    - arm_right_3_joint *(revolute)*

    - arm_right_4_joint *(revolute)*

    - arm_right_5_joint *(revolute)*

    - arm_right_6_joint *(revolute)*

    - arm_right_7_joint *(revolute)*

    - leg_left_1_joint *(revolute)*

    - leg_left_2_joint *(revolute)*

    - leg_left_3_joint *(revolute)*

    - leg_left_length_joint *(sliding)*

    - leg_left_4_joint *(revolute)*

    - leg_left_5_joint *(revolute)*

    - leg_right_1_joint *(revolute)*

    - leg_right_2_joint *(revolute)*

    - leg_right_3_joint *(revolute)*

    - leg_right_length_joint *(sliding)*

    - leg_right_4_joint *(revolute)*

    - leg_right_5_joint *(revolute)*

    - gripper_left_inner_finger_left_joint *(revolute)*

    - gripper_left_fingertip_left_joint *(revolute)*

    - gripper_left_finger_joint *(revolute)*

    - gripper_left_inner_finger_right_joint *(revolute)*

    - gripper_left_fingertip_right_joint *(revolute)*

    - gripper_left_outer_finger_right_joint *(revolute)*

    - gripper_right_inner_finger_left_joint *(revolute)*

    - gripper_right_fingertip_left_joint *(revolute)*

    - gripper_right_finger_joint *(revolute)*

    - gripper_right_inner_finger_right_joint *(revolute)*

    - gripper_right_fingertip_right_joint *(revolute)*

    - gripper_right_outer_finger_right_joint *(revolute)*


    4 passive joints :

    - leg_left_knee_joint *(revolute)*

    - leg_left_femur_joint *(revolute)*

    - leg_right_knee_joint *(revolute)*

    - leg_right_femur_joint *(revolute)*

The passive joints are not directly actuated by the policy — they follow the
motion of the closed-chain leg mechanism and are excluded from the action
space. Each arm carries a 3-joint wrist (a roll-pitch-roll sequence) beyond
the shoulder and elbow, giving the end effector enough freedom to orient the
gripper for grasping, and each end effector carries 6 finger joints
representing the parallel-jaw gripper's closed kinematic chain — for a total
of 44 joints, 40 of which are actuated by the policy.