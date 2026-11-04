.. _TIAGo Pro overview:

PAL TIAGo Pro
=============

TIAGo Pro is PAL Robotics' mobile manipulator. In ``pal_mjlab``, only its two
arms are controlled: the model has 14 actuated revolute joints,
``arm_{left,right}_{1..7}_joint`` (7 per arm).

.. image:: /_static/tiago_pro.png
   :alt: TIAGo Pro model
   :width: 300px
   :align: center

Robot parameters are defined in ``tiago_pro_constants.py``.

Reaching task
-------------

Task ID: ``Mjlab-Reaching-Flat-Pal-Tiago-Pro``

The policy moves each arm's end effector to a target pose sampled by
``UniformPoseCommand`` (``pose_command_left`` / ``pose_command_right``).
Unlike the :ref:`Kangaroo reaching task <Kangaroo task_reaching>`, there is
no locomotion command.

**Actor and critic:** ``joint_pos``, ``joint_vel``, ``actions``,
``pose_command_left``, ``pose_command_right``.

Rewards:

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Name
     - Weight
   * - pos_left
     - -2.0
   * - pos_left_fine_grained
     - 2.0
   * - ee_left_orientation
     - -0.2
   * - pos_right
     - -2.0
   * - pos_right_fine_grained
     - 3.0
   * - ee_right_orientation
     - -0.2
   * - dof_pos_limits
     - -1.0
   * - action_rate_l2
     - -0.003
   * - joint_vel_hinge
     - -0.05
   * - self_collisions
     - -1.0


Terminations: time out only.
