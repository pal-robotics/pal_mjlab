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


Each registered task pairs an objective (Velocity, Tracking, Reaching), a terrain
type (Flat, Rough) and a model variant:

+----+---------------------------------------------------------------------------+
| 1  | Mjlab-Reaching-Flat-Pal-Kangaroo                                          |
+----+---------------------------------------------------------------------------+
| 2  | Mjlab-Reaching-Flat-Pal-Kangaroo-Grippers                                 |
+----+---------------------------------------------------------------------------+
| 3  | Mjlab-Reaching-Flat-Pal-Kangaroo-Hands                                    |
+----+---------------------------------------------------------------------------+
| 4  | Mjlab-Tracking-Flat-Pal-Kangaroo                                          |
+----+---------------------------------------------------------------------------+
| 5  | Mjlab-Tracking-Flat-Pal-Kangaroo-No-State-Estimation                      |
+----+---------------------------------------------------------------------------+
| 6  | Mjlab-Velocity-Flat-Pal-Kangaroo                                          |
+----+---------------------------------------------------------------------------+
| 7  | Mjlab-Velocity-Flat-Pal-Kangaroo-Grippers                                 |
+----+---------------------------------------------------------------------------+
| 8  | Mjlab-Velocity-Flat-Pal-Kangaroo-Hands                                    |
+----+---------------------------------------------------------------------------+
| 9  | Mjlab-Velocity-Flat-Pal-Kangaroo-Lower-Body                               |
+----+---------------------------------------------------------------------------+
| 10 | Mjlab-Velocity-Rough-Pal-Kangaroo                                         |
+----+---------------------------------------------------------------------------+
| 11 | Mjlab-Velocity-Rough-Pal-Kangaroo-Grippers                                |
+----+---------------------------------------------------------------------------+
| 12 | Mjlab-Velocity-Rough-Pal-Kangaroo-Hands                                   |
+----+---------------------------------------------------------------------------+


The full list of tasks can be checked anytime with:

.. code-block:: bash

   uv run list-envs

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

Tasks:

- :ref:`Kangaroo task_velocity`
- :ref:`Kangaroo task_motion_imitation`
