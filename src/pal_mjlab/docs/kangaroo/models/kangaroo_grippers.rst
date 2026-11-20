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
for all links so the dynamics remain faithful to the real robot.

.. note::

   The gripper finger joints and the equality constraints that close the
   gripper linkage are currently commented out in ``kangaroo_grippers.xml``.
   The grippers are therefore rigid, and the policy only controls the arm,
   pelvis and leg joints.

32 joints :

    28 actuators :

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

    4 passive joints :

    - leg_left_knee_joint *(revolute)*

    - leg_left_femur_joint *(revolute)*

    - leg_right_knee_joint *(revolute)*

    - leg_right_femur_joint *(revolute)*

    12 inactive gripper joints (commented out in the model, not counted above):

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

The passive joints are not directly actuated by the policy — they follow the
motion of the closed-chain leg mechanism and are excluded from the action
space. Each arm carries a 3-joint wrist (a roll-pitch-roll sequence) beyond
the shoulder and elbow, giving the end effector enough freedom to orient the
gripper. This leaves 28 actuated joints controlled by the policy.
