.. _Kangaroo model_simple:

Kangaroo - Simple Model
========================

The simple model is the full-body Kangaroo with 4 DoF per arm and a fixed
forearm. It is the most used model across the different tasks in pal_mjlab.

.. image:: /_static/kangaroo_4dof.png
   :alt: Kangaroo simple model with 4 DoF arms
   :width: 300px
   :align: center

It incorporates a simplified representation of Kangaroo's closed-chain leg
mechanism: the real linkage is reduced to a serial chain plus four passive
joints (femur and knee on each leg), while masses and inertias are computed
for all links so the dynamics remain faithful to the real robot.

26 joints :

    22 actuators :

    - pelvis_1_joint *(revolute)*

    - pelvis_2_joint *(revolute)*

    - arm_left_1_joint *(revolute)*

    - arm_left_2_joint *(revolute)*

    - arm_left_3_joint *(revolute)*

    - arm_left_4_joint *(revolute)*

    - arm_right_1_joint *(revolute)*

    - arm_right_2_joint *(revolute)*

    - arm_right_3_joint *(revolute)*

    - arm_right_4_joint *(revolute)*

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

The passive joints are not directly actuated by the policy — they follow the
motion of the closed-chain leg mechanism and are excluded from the action
space, leaving 22 actuated joints controlled by the policy.