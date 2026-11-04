.. _Kangaroo overview:

PAL Kangaroo
============

Kangaroo is PAL Robotics' humanoid research platform designed for whole-body
locomotion and manipulation. The simulator supports multiple hardware
configurations, ranging from a lower-body-only model to full-body models with
different arm configurations.

The environments are implemented on top of mjlab and provide standardized
training tasks for reinforcement learning.

With 14 actuated DoF in the lower body (12 leg + 2 pelvis) and several arm
configurations, Kangaroo can perform a wide range of human-inspired tasks.
The available model variants are:

- **Lower body**: legs and pelvis only, no arms
- **Simple**: 4 DoF per arm with a fixed forearm
- **Hands**: 5 DoF per arm with a Seed Robotics hand
- **Grippers**: 7 DoF per arm with a gripper

The tasks explored here focus mainly on:

- Locomotion (velocity tracking)

- Motion imitation (reference motion tracking)

- Reaching (end-effector pose tracking while walking)


Each registered task pairs an objective (Velocity, Tracking, Reaching), a terrain
type (Flat, Rough) and a model variant:

.. list-table::
   :header-rows: 1
   :widths: 55 15 10 20

   * - Task ID
     - Objective
     - Terrain
     - Model
   * - ``Mjlab-Velocity-Flat-Pal-Kangaroo``
     - Velocity
     - Flat
     - Simple
   * - ``Mjlab-Velocity-Flat-Pal-Kangaroo-Lower-Body``
     - Velocity
     - Flat
     - Lower body
   * - ``Mjlab-Velocity-Flat-Pal-Kangaroo-Hands``
     - Velocity
     - Flat
     - Hands
   * - ``Mjlab-Velocity-Flat-Pal-Kangaroo-Grippers``
     - Velocity
     - Flat
     - Grippers
   * - ``Mjlab-Velocity-Rough-Pal-Kangaroo``
     - Velocity
     - Rough
     - Simple
   * - ``Mjlab-Velocity-Rough-Pal-Kangaroo-Lower-Body``
     - Velocity
     - Rough
     - Lower body
   * - ``Mjlab-Tracking-Flat-Pal-Kangaroo``
     - Tracking
     - Flat
     - Simple
   * - ``Mjlab-Tracking-Flat-Pal-Kangaroo-No-State-Estimation``
     - Tracking
     - Flat
     - Simple
   * - ``Mjlab-Tracking-Flat-Pal-Kangaroo-Hands``
     - Tracking
     - Flat
     - Hands
   * - ``Mjlab-Tracking-Flat-Pal-Kangaroo-Hands-No-State-Estimation``
     - Tracking
     - Flat
     - Hands
   * - ``Mjlab-Tracking-Flat-Pal-Kangaroo_Lower_Body``
     - Tracking
     - Flat
     - Lower body
   * - ``Mjlab-Tracking-Flat-Pal-Kangaroo-Lower-Body-No-State-Estimation``
     - Tracking
     - Flat
     - Lower body
   * - ``Mjlab-Reaching-Flat-Pal-Kangaroo``
     - Reaching
     - Flat
     - Simple
   * - ``Mjlab-Reaching-Flat-Pal-Kangaroo-Hands``
     - Reaching
     - Flat
     - Hands
   * - ``Mjlab-Reaching-Flat-Pal-Kangaroo-Grippers``
     - Reaching
     - Flat
     - Grippers

The ``-No-State-Estimation`` tracking variants remove the observations that need
a base state estimator on the real robot (``motion_anchor_pos_b``,
``motion_anchor_ori_b`` and ``base_lin_vel``) from the actor.

.. note::

   The lower-body tracking task ID uses underscores
   (``Kangaroo_Lower_Body``), unlike the other task IDs. Use it exactly as
   written.

The full list of tasks can be checked anytime with:

.. code-block:: bash

   uv run list-envs --keyword Kangaroo

To train a task on Kangaroo, use:

.. code-block:: bash

   uv run train {task_id} {other parameters}

The full list of parameters can be seen using:

.. code-block:: bash

   uv run train {task_id} --help

To run inference to evaluate results:

.. code-block:: bash

   uv run play {task_id} {other parameters}

|

Robot-specific parameters: actuator stiffness, damping, effort limits, initial
pose and collision setup: are defined in ``kangaroo_constants.py`` and can be
adjusted there for all model variants.

See also
--------

Models:

- :ref:`Kangaroo model_simple`
- :ref:`Kangaroo model_lower_body`
- :ref:`Kangaroo model_hands`
- :ref:`Kangaroo model_grippers`

Tasks:

- :ref:`Kangaroo task_velocity`
- :ref:`Kangaroo task_motion_imitation`
- :ref:`Kangaroo task_reaching`
