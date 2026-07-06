.. _Kangaroo overview:

PAL Kangaroo
============

Kangaroo is PAL Robotics' humanoid research platform designed for whole-body
locomotion and manipulation. The simulator supports multiple hardware
configurations, ranging from a lower-body-only model to full-body models with
different arm configurations.

The environments are implemented on top of mjlab and provide standardized
training tasks for reinforcement learning.

With 14 DOFS in the lower body and various arm configurations (0 DOFS, 4 DOFS, 7 DOFS, ...) Kangaroo has the capability to perform many human inspired tasks.

Here, we explore mainly :  

- Locomotion  

- Motion imitation  


Here are the different tasks implemented for Kangaroo, which are relative to objective and model :  

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

In the kangaroo_constants.py file, user may define stifness and damping for all joints.

See also
--------

Models:

- :ref:`Kangaroo model_simple`
- :ref:`Kangaroo model_lower_body`

Tasks:

- :ref:`Kangaroo task_velocity`
- :ref:`Kangaroo task_motion_imitation`