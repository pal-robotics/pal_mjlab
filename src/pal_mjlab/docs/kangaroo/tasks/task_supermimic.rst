.. _Kangaroo task_supermimic:

PAL Kangaroo - SuperMimic motion tracking
==========================================

SuperMimic (arXiv 2610.28480) trains one policy to track any motion of a large
motion library. It learns a world model of the simulator online and improves the
policy by backpropagating the tracking reward through 32 steps imagined with it,
plus a critic's value beyond them. Episodes start at frames picked by a
return-based sampler that favours the parts of the library the policy tracks
worst.

The Kangaroo tasks follow the paper's settings and reward. They train on the
BONES-SEED motions retargeted to Kangaroo (about 119,000 motions, 233 h).

Tasks
-----

+------------------------------------------------------+------------------------------------------+
| Task                                                 | Robot                                    |
+======================================================+==========================================+
| Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers          | 7-DoF arms with grippers                 |
+------------------------------------------------------+------------------------------------------+
| Mjlab-SuperMimic-Flat-Pal-Kangaroo-Hands             | 5-DoF arms with Seed Robotics hands      |
+------------------------------------------------------+------------------------------------------+
| Mjlab-SuperMimic-Flat-Pal-Kangaroo-Full-Grippers     | 7-DoF arms on the closed-chain legs      |
+------------------------------------------------------+------------------------------------------+
| Mjlab-SuperMimic-PPO-Flat-Pal-Kangaroo-Grippers      | the same env trained with PPO (baseline) |
+------------------------------------------------------+------------------------------------------+
| Mjlab-SuperMimic-PPO-Flat-Pal-Kangaroo-Hands         | the same env trained with PPO (baseline) |
+------------------------------------------------------+------------------------------------------+
| Mjlab-SuperMimic-PPO-Flat-Pal-Kangaroo-Full-Grippers | the same env trained with PPO (baseline) |
+------------------------------------------------------+------------------------------------------+

Motion library
--------------

The tasks read a **motion library**, a folder with three files:

- ``clips.json``: the frame rate, the joint names, and per clip its name,
  category, start frame and length;
- ``qpos.npy``: (frames, 39) root position, root quaternion and 32 joint positions;
- ``qvel.npy``: (frames, 38) root linear and angular velocity and 32 joint velocities.

``uv run -m pal_mjlab.scripts.bones_to_library --data-dir <dataset>`` packs one
from a retargeted dataset folder: ``smoke`` (15 min), ``dev`` (2.5 h) or ``full``.
Without ``--env.commands.motion.library-path``, the tasks read
``artifacts/kangaroo_bones_seed/dataset[_5dof]/libraries/dev`` under the working
directory.

Training and evaluation
-----------------------

.. code-block:: bash

   uv sync --extra cu128 --group supermimic
   uv run train Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers \
     --env.commands.motion.library-path <library> --agent.run-name grippers-<name>
   uv run play Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers --checkpoint-file <run>/model_N.pt

Runs log to ``logs/rsl_rl/supermimic/`` (the PPO baseline to ``supermimic_ppo/``).
Training writes ``model_N.onnx`` next to every checkpoint: the policy for the
robot. Its input is the next 10 reference frames and 10 steps of
proprioception; its output is the joint position targets.

Tools:

- ``uv run -m pal_mjlab.scripts.supermimic.eval``: benchmark checkpoints on every
  clip of a library (failures, tracking errors per body group, foot sliding).
- ``uv run -m pal_mjlab.scripts.supermimic.play``: replay chosen clips of a library
  with a policy. A fall plays out for 3 s instead of resetting.
- ``uv run -m pal_mjlab.scripts.supermimic.view_motion``: the motions as
  retargeted, without physics.

The arm and waist actuators of both robots are set in one place,
``ARM_WAIST_GAIN_FREQ_HZ`` and ``ARM_WAIST_ACTION_SCALE_FREQ_HZ`` in
``tasks/supermimic/kangaroo/env_cfgs.py``. The design decisions, measurements and
reports are in ``docs/supermimic/`` at the repository root.
