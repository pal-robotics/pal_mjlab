.. _Kangaroo model_lower_body:

Kangaroo - Lower Body Model
=============================

The lower-body configuration is a reduced model of the simple model. Besides the excluded arm joints, the rest is identical.

18 joints :

    14 actuators :

    - pelvis_1_joint *(revolute)*

    - pelvis_2_joint *(revolute)*

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

This model is used by the ``Mjlab-Velocity-Flat-Pal-Kangaroo-Lower-Body`` task, where locomotion is trained without the arms, relying only on the legs and pelvis for balance and velocity tracking.
    