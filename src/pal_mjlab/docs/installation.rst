.. _installation:

Installation & Setup
====================

.. contents:: Table of Contents
   :local:
   :depth: 2

Prerequisites
-------------

Before installing ``pal_mjlab``, ensure your system meets the following requirements.

**Operating System**

- Linux (recommended for training)
- macOS (evaluation only — training is not supported)

**Hardware**

- NVIDIA GPU (required for training via MuJoCo Warp)
- Sufficient VRAM to run parallel environments (4096+ environments recommended for training)

**Software**

- `uv <https://github.com/astral-sh/uv>`_ — the Python package manager used by this project
- Git
- CUDA-compatible drivers matching your MuJoCo Warp version

Installation Steps
------------------

**Step 1 — Install uv**

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

**Step 2 — Clone the repository**

.. code-block:: bash

   git clone https://github.com/pal-robotics/pal_mjlab.git
   cd pal_mjlab

**Step 3 — Sync dependencies**

.. code-block:: bash

   uv sync

This will automatically resolve and install all required Python dependencies, including ``mjlab`` and ``MuJoCo Warp``.

**Step 4 — Verify the installation**

List available PAL environments to confirm everything is working:

.. code-block:: bash

   uv run list-envs --keyword pal

You should see a list of environments such as ``Mjlab-Velocity-Flat-Pal-Kangaroo`` and ``Mjlab-Tracking-Flat-Pal-Kangaroo``.

**Step 5 — Run a quick sanity check**

Test with dummy agents before committing to a full training run:

.. code-block:: bash

   uv run play Mjlab-Velocity-Flat-Pal-Kangaroo --agent zero    # sends zero actions
   uv run play Mjlab-Velocity-Flat-Pal-Kangaroo --agent random  # sends random actions

Hardware Requirements
---------------------

.. list-table::
   :widths: 25 50 25
   :header-rows: 1

   * - Use Case
     - Requirement
     - Notes
   * - Training
     - NVIDIA GPU (CUDA-enabled)
     - Required; not supported on CPU or macOS
   * - Evaluation / Playback
     - CPU or GPU
     - macOS supported for this mode only
   * - Large-scale training
     - High-VRAM GPU (e.g. A100, H100)
     - Recommended for 4096+ parallel envs
   * - Multi-GPU training
     - Multiple NVIDIA GPUs
     - Supported via ``--gpu-ids`` flag

.. note::

   ``pal_mjlab`` is in active development. Breaking changes may occur between releases.
   It is recommended to pin to a specific release tag for production use.