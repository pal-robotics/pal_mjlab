.. _Kangaroo model_hands:

Kangaroo - Hands Model
=======================

The hands model is the full-body Kangaroo with 5 DoF per arm, ending in a
Seed Robotics hand.

.. image:: /_static/kangaroo_hands.png
   :alt: Kangaroo hands model with 5 DoF arms and hands
   :width: 300px
   :align: center

It uses the same simplified closed-chain leg mechanism as the
:ref:`simple model <Kangaroo model_simple>`: a serial chain plus four passive
joints (femur and knee on each leg).

.. note::

   The hand joints (palm and fingers) are currently commented out in
   ``kangaroo_hands.xml``, so the hands are rigid. The policy only controls
   the arm, pelvis and leg joints.

28 joints :

    24 actuators :

    - pelvis_1_joint *(revolute)*

    - pelvis_2_joint *(revolute)*

    - arm_left_1_joint *(revolute)*

    - arm_left_2_joint *(revolute)*

    - arm_left_3_joint *(revolute)*

    - arm_left_4_joint *(revolute)*

    - arm_left_5_joint *(revolute)*

    - arm_right_1_joint *(revolute)*

    - arm_right_2_joint *(revolute)*

    - arm_right_3_joint *(revolute)*

    - arm_right_4_joint *(revolute)*

    - arm_right_5_joint *(revolute)*

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

This model is used by the ``Mjlab-Velocity-Flat-Pal-Kangaroo-Hands``,
``Mjlab-Tracking-Flat-Pal-Kangaroo-Hands``,
``Mjlab-Tracking-Flat-Pal-Kangaroo-Hands-No-State-Estimation`` and
``Mjlab-Reaching-Flat-Pal-Kangaroo-Hands`` tasks.
