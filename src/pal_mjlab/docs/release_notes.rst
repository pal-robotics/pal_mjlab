.. _release_notes:

Release Notes
=============

This page summarizes what has changed in ``pal_mjlab`` between releases, in
plain language, with a link back to the pull request (PR) that introduced
each change. If you are new to the project, this is the fastest way to
catch up on what has moved since the last tagged release.

.. contents:: Table of Contents
   :local:
   :depth: 2

Unreleased — since ``v1.0.0``
------------------------------

The changes below have landed on ``main`` since the ``v1.0.0`` tag
(2026-03-06) and are not yet part of a tagged release.

New Features & Tasks
^^^^^^^^^^^^^^^^^^^^^

- **Actuator parameters are now easy to tune.** Motor gains, armature,
  friction and effort limits for each Kangaroo actuator were previously
  hard-coded inline; they are now grouped into named parameter blocks in
  ``kangaroo_constants.py`` so you can change one actuator's behaviour
  without hunting through the file.
  (`#108 <https://github.com/pal-robotics/pal_mjlab/pull/108>`_)

- **Foot collision avoidance capsules added to the 4-DoF Kangaroo model.**
  Extra collision capsules around the feet stop the legs/feet from
  interpenetrating during testing, adding this reduces the foot stepping
  over each other.
  (`#104 <https://github.com/pal-robotics/pal_mjlab/pull/104>`_)

- **Dual-band velocity command sampling.** The velocity command generator
  can now avoid sampling commands whose magnitude falls in a "dead" low
  range, so training does not waste time near-zero velocity commands —
  particularly useful for rough-terrain/stairs training.
  Note: Command sampler is added but not yet used in the tasks
  (`#85 <https://github.com/pal-robotics/pal_mjlab/pull/85>`_)

- **New decoupled angular-velocity rewards.** Two new reward terms,
  ``track_body_ang_vel_z_exp`` (yaw tracking) and
  ``body_ang_vel_xy_l2_penalty`` (roll/pitch penalty), are available as a
  replacement for mjlab's combined ``track_angular_velocity`` /
  ``body_angular_velocity_penalty`` pair. Splitting them fixes a
  frame-mismatch bug (yaw used the body frame, the penalty used the world
  frame) and stops the roll/pitch restoring gradient from vanishing
  whenever yaw error is large — both terms can now be tuned independently.
  Note: The rewards are added, but are not used within the configurations yet.
  (`#81 <https://github.com/pal-robotics/pal_mjlab/pull/81>`_)

- **New tracking task for the Kangaroo lower body (+ 5-DoF variant).**
  Adds a motion-tracking task on top of the existing velocity-tracking
  task.
  (`#91 <https://github.com/pal-robotics/pal_mjlab/pull/91>`_)

- **New velocity-tracking task for the Kangaroo lower body.** The basic
  velocity-tracking environment for the lower-body robot.
  (`#76 <https://github.com/pal-robotics/pal_mjlab/pull/76>`_)

- **Reward set adapted for lower-body flat-ground locomotion.** Minimal
  reward changes needed to get the lower-body robot walking on flat
  ground.
  (`#87 <https://github.com/pal-robotics/pal_mjlab/pull/87>`_)

Model Updates
^^^^^^^^^^^^^

- Updated the Kangaroo 7-DoF and hands MuJoCo models with proper masses
  and inertias.
  (`#107 <https://github.com/pal-robotics/pal_mjlab/pull/107>`_)
- Updated the Kangaroo 4-DoF MuJoCo model with proper masses and inertias.
  (`#106 <https://github.com/pal-robotics/pal_mjlab/pull/106>`_)
- Replaced the TALOS collision meshes with capsule proxies, which are
  cheaper to simulate and collide.
  (`#102 <https://github.com/pal-robotics/pal_mjlab/pull/102>`_)
- Updated the Kangaroo lower-body model.
  (`#84 <https://github.com/pal-robotics/pal_mjlab/pull/84>`_)
- Set the density of the foot capsules to zero, so they no longer affect
  the robot's mass/inertia and act as pure collision geometry.
  (`#79 <https://github.com/pal-robotics/pal_mjlab/pull/79>`_)

Bug Fixes & Reliability
^^^^^^^^^^^^^^^^^^^^^^^^

- **Fixed a CUDA illegal-memory-access crash on reset with mjlab 1.5.1.**
  Training would crash with a CUDA illegal memory access error when
  environments reset. Worked around by pinning a specific mjlab commit;
  see also the memory note on this issue for how it was diagnosed.
  (`#94 <https://github.com/pal-robotics/pal_mjlab/pull/94>`_)

- **Improved policy repeatability across training runs.** Reduces the
  joint-velocity and IMU-projected-gravity observation noise so that
  repeated training runs converge to more similar behaviour, making it
  easier to compare experiments.
  (`#75 <https://github.com/pal-robotics/pal_mjlab/pull/75>`_)

Dependency Updates
^^^^^^^^^^^^^^^^^^

``pal_mjlab`` tracks ``mjlab`` closely. If you hit an environment or
dependency issue, check which of these bumps might be relevant first.

- Updated ``mjlab`` to ``1.6.0``.
  (`#103 <https://github.com/pal-robotics/pal_mjlab/pull/103>`_)
- Updated ``mjlab`` to ``1.5.3``.
  (`#99 <https://github.com/pal-robotics/pal_mjlab/pull/99>`_)
- Updated ``mjlab`` to ``1.5.2``.
  (`#95 <https://github.com/pal-robotics/pal_mjlab/pull/95>`_)
- Updated ``mjlab`` to ``1.5.1`` (superseded by the CUDA fix in
  `#94 <https://github.com/pal-robotics/pal_mjlab/pull/94>`_).
  (`#92 <https://github.com/pal-robotics/pal_mjlab/pull/92>`_)
- Updated ``mjlab`` to ``1.5.0``.
  (`#80 <https://github.com/pal-robotics/pal_mjlab/pull/80>`_)
- Updated ``mjlab`` to ``1.4.0``.
  (`#77 <https://github.com/pal-robotics/pal_mjlab/pull/77>`_)
- Updated ``mjlab`` to ``1.3.0``.
  (`#73 <https://github.com/pal-robotics/pal_mjlab/pull/73>`_)
- Updated ``mjlab`` to ``1.2.0``.
  (`#70 <https://github.com/pal-robotics/pal_mjlab/pull/70>`_)

Breaking Changes
^^^^^^^^^^^^^^^^

- **Renamed** ``pal_kangaroo_rough_env_cfg`` **to**
  ``pal_kangaroo_baseline_env_cfg``. If you reference this config directly
  (rather than through a registered Gym task id), update the import.
  (`#86 <https://github.com/pal-robotics/pal_mjlab/pull/86>`_)

Documentation & CI
^^^^^^^^^^^^^^^^^^

- Added the first version of the ``pal_mjlab`` documentation (the site
  you are reading now).
  (`#82 <https://github.com/pal-robotics/pal_mjlab/pull/82>`_)
- Added a reward-testing CI workflow to catch regressions in reward
  functions.
  (`#78 <https://github.com/pal-robotics/pal_mjlab/pull/78>`_)
- Updated the CI versions.
  (`#83 <https://github.com/pal-robotics/pal_mjlab/pull/83>`_)

Maintenance
^^^^^^^^^^^

- Changed the default checkpoint ``save_interval`` to 500 iterations.
  (`#72 <https://github.com/pal-robotics/pal_mjlab/pull/72>`_)
- Removed trailing whitespace across the codebase.
  (`#93 <https://github.com/pal-robotics/pal_mjlab/pull/93>`_)

``v1.0.0``
----------

Initial tagged release of ``pal_mjlab``.
