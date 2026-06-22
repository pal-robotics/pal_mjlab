Simple model of the Kangaroo :


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

The simple model is the most used through the different tasks in pal_mjlab.

It incorporates a simplified model of the Kangaroo's leg closed chain, and has calculated masses and inertia through all links.