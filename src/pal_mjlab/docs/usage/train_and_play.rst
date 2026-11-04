.. _train_and_play:

Training and Playing Tasks
==========================

This page shows how to launch training (``train``) and evaluation
(``play``) for every task family in ``pal_mjlab``. Both commands come from
mjlab and take the task ID as their first argument.

.. contents:: Table of Contents
   :local:
   :depth: 2

Finding a task ID
-----------------

.. code-block:: bash

   uv run list-envs --keyword Pal       # all PAL tasks
   uv run list-envs --keyword Kangaroo  # one robot (case-insensitive)

Task IDs follow ``Mjlab-<Objective>-<Terrain>-Pal-<Robot>[-<Variant>]``.
The full catalogue is on each robot page:
:ref:`Kangaroo <Kangaroo overview>`, :ref:`TALOS <Talos overview>` and
:ref:`TIAGo Pro <TIAGo Pro overview>`.

Every command accepts ``--help``, which lists all the options for that task,
including every environment (``--env.*``) and PPO runner (``--agent.*``)
field:

.. code-block:: bash

   uv run train Mjlab-Velocity-Flat-Pal-Kangaroo --help
   uv run play  Mjlab-Velocity-Flat-Pal-Kangaroo --help

.. note::

   Boolean options take an explicit value, e.g. ``--video True``.


Common options
--------------

.. list-table:: ``train``
   :header-rows: 1
   :widths: 40 60

   * - Option
     - Meaning
   * - ``--env.scene.num-envs INT``
     - Number of parallel environments (4096 is a good default on one GPU).
   * - ``--agent.max-iterations INT``
     - PPO iterations (default ``30000`` for PAL tasks).
   * - ``--agent.logger {wandb,tensorboard}``
     - Logger. The default is ``wandb``; use ``tensorboard`` to train without
       a wandb account.
   * - ``--agent.run-name STR``
     - Suffix appended to the run's log directory name.
   * - ``--agent.seed INT``
     - Random seed.
   * - ``--agent.save-interval INT``
     - Save a checkpoint every N iterations.
   * - ``--gpu-ids 0 1`` / ``--gpu-ids all``
     - Train on several GPUs.
   * - ``--video True``
     - Record videos during training (``--video-length``,
       ``--video-interval``).
   * - ``--registry-name STR``
     - Tracking tasks only: wandb artifact holding the reference motion.

.. list-table:: ``play``
   :header-rows: 1
   :widths: 40 60

   * - Option
     - Meaning
   * - ``--agent {zero,random,trained}``
     - Policy to run. ``zero`` and ``random`` need no checkpoint and are
       useful to inspect a task.
   * - ``--wandb-run-path ORG/PROJECT/RUN_ID``
     - Load the latest checkpoint of a wandb run
       (``--wandb-checkpoint-name`` picks a specific one).
   * - ``--checkpoint-file PATH``
     - Load a local checkpoint (``model_<iteration>.pt``).
   * - ``--num-envs INT``
     - Number of environments to display. This is a top-level flag, unlike
       ``--env.scene.num-envs`` in ``train``.
   * - ``--motion-file PATH`` / ``--registry-name STR``
     - Tracking tasks only: reference motion to replay.
   * - ``--viewer {auto,native,viser}``
     - Viewer backend.
   * - ``--video True``
     - Record a video instead of only displaying it.
   * - ``--no-terminations True``
     - Disable episode terminations, to watch a policy for longer.
   * - ``--device STR``
     - Device, e.g. ``cpu`` or ``cuda:0``.


Where checkpoints are saved
---------------------------

Training writes to
``logs/rsl_rl/<experiment_name>/<YYYY-MM-DD_HH-MM-SS>[_<run_name>]/``. A
``model_<iteration>.pt`` file is saved there every ``save_interval``
iterations. ``experiment_name`` is set per task family:

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Tasks
     - ``experiment_name``
   * - ``Mjlab-Velocity-*-Pal-Kangaroo*``
     - ``kangaroo_velocity``
   * - ``Mjlab-Tracking-Flat-Pal-Kangaroo``, ``-Hands`` (and
       ``-No-State-Estimation``)
     - ``kangaroo_tracking``
   * - ``Mjlab-Tracking-Flat-Pal-Kangaroo_Lower_Body``,
       ``-Lower-Body-No-State-Estimation``
     - ``kangaroo_lower_body_tracking``
   * - ``Mjlab-Reaching-Flat-Pal-Kangaroo*``
     - ``kangaroo_reaching``
   * - ``Mjlab-Velocity-*-Pal-Talos``
     - ``talos_velocity``
   * - ``Mjlab-Tracking-Flat-Pal-Talos*``
     - ``talos_tracking``
   * - ``Mjlab-Reaching-Flat-Pal-Tiago-Pro``
     - ``tiago_pro_reaching``

To resume an interrupted run from its latest checkpoint:

.. code-block:: bash

   uv run train Mjlab-Velocity-Flat-Pal-Kangaroo \
       --agent.resume True \
       --agent.load-run 2026-11-04_10-00-00   # regex on the run directory name
       # --agent.load-checkpoint model_5000.pt  # optional, default: latest


Velocity tracking
-----------------

Tasks: ``Mjlab-Velocity-{Flat,Rough}-Pal-Kangaroo[-Lower-Body]``,
``Mjlab-Velocity-Flat-Pal-Kangaroo-{Hands,Grippers}`` and
``Mjlab-Velocity-{Flat,Rough}-Pal-Talos``.

.. code-block:: bash

   # Inspect the task without a policy
   uv run play Mjlab-Velocity-Flat-Pal-Kangaroo --agent zero

   # Train
   uv run train Mjlab-Velocity-Flat-Pal-Kangaroo --env.scene.num-envs 4096

   # Rough terrain, lower body, logged to TensorBoard
   uv run train Mjlab-Velocity-Rough-Pal-Kangaroo-Lower-Body \
       --env.scene.num-envs 4096 --agent.logger tensorboard

   # Play a wandb run
   uv run play Mjlab-Velocity-Flat-Pal-Kangaroo --wandb-run-path your-org/mjlab/run-id

   # Play a local checkpoint
   uv run play Mjlab-Velocity-Flat-Pal-Kangaroo \
       --checkpoint-file logs/rsl_rl/kangaroo_velocity/<run_dir>/model_29999.pt

For the Kangaroo rough tasks, the terrain curriculum is disabled in play
mode, so the robot is shown on every terrain type.


Motion tracking (imitation)
---------------------------

Tasks: ``Mjlab-Tracking-Flat-Pal-Kangaroo[-Hands][-No-State-Estimation]``,
``Mjlab-Tracking-Flat-Pal-Kangaroo_Lower_Body``,
``Mjlab-Tracking-Flat-Pal-Kangaroo-Lower-Body-No-State-Estimation`` and
``Mjlab-Tracking-Flat-Pal-Talos[-No-State-Estimation]``.

Tracking tasks need a reference motion in ``.npz`` format, **for both train
and play**, including ``--agent zero``.

**1. Convert a retargeted CSV motion** (examples are in ``motions/``):

.. code-block:: bash

   uv run -m pal_mjlab.scripts.csv_to_npz \
       --input-file motions/kang_walk.csv \
       --output-name kang_walk \
       --input-fps 30 --output-fps 50 \
       --robot-name kangaroo \
       --upload-to-wandb True   # optional, see below

   cp /tmp/motion.npz motions/kang_walk.npz   # keep a local copy

The motion is always written to ``/tmp/motion.npz``. Copy it somewhere
permanent if you want to use it as a local file. With
``--upload-to-wandb True``, it is also logged to the ``csv_to_npz`` wandb
project and linked to the ``motions`` registry under ``--output-name``. Use
``--robot-name talos`` for TALOS.

.. warning::

   ``--output-fps`` must match the control frequency of the task (50 Hz for
   the PAL tasks), see :ref:`Kangaroo task_motion_imitation`.

**2. Train**, either from wandb or from a local file:

.. code-block:: bash

   # Motion from wandb
   uv run train Mjlab-Tracking-Flat-Pal-Kangaroo \
       --registry-name your-org/csv_to_npz/kang_walk \
       --env.scene.num-envs 4096

   # Local motion file
   uv run train Mjlab-Tracking-Flat-Pal-Kangaroo \
       --env.commands.motion.motion-file motions/kang_walk.npz \
       --env.scene.num-envs 4096

**3. Play**:

.. code-block:: bash

   # Inspect the motion without a policy
   uv run play Mjlab-Tracking-Flat-Pal-Kangaroo --agent zero --motion-file motions/kang_walk.npz

   # wandb run: the motion is resolved from the run automatically
   uv run play Mjlab-Tracking-Flat-Pal-Kangaroo --wandb-run-path your-org/mjlab/run-id

   # Local checkpoint: the motion file is required
   uv run play Mjlab-Tracking-Flat-Pal-Kangaroo \
       --checkpoint-file logs/rsl_rl/kangaroo_tracking/<run_dir>/model_29999.pt \
       --motion-file motions/kang_walk.npz

A policy trained on a ``-No-State-Estimation`` task must also be played on
that same task, since the actor observations differ.


Reaching
--------

Tasks: ``Mjlab-Reaching-Flat-Pal-Kangaroo[-Hands,-Grippers]`` (reaching
while walking) and ``Mjlab-Reaching-Flat-Pal-Tiago-Pro`` (arms only).

.. code-block:: bash

   uv run train Mjlab-Reaching-Flat-Pal-Tiago-Pro --env.scene.num-envs 4096
   uv run play  Mjlab-Reaching-Flat-Pal-Tiago-Pro --wandb-run-path your-org/mjlab/run-id

   uv run train Mjlab-Reaching-Flat-Pal-Kangaroo-Grippers --env.scene.num-envs 4096
   uv run play  Mjlab-Reaching-Flat-Pal-Kangaroo-Grippers \
       --checkpoint-file logs/rsl_rl/kangaroo_reaching/<run_dir>/model_29999.pt


Tips
----

- Always play a checkpoint on the **same task ID** it was trained on. The
  observation and action sizes differ between model variants, e.g.
  ``-Hands`` vs. the simple model.
- Training requires an NVIDIA GPU. ``play`` also works on CPU with
  ``--device cpu``.
- Start with a short run (e.g. ``--agent.max-iterations 100``) to check
  that the setup works before a full training run.
