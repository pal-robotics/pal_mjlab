pal_mjlab Documentation
========================

``pal_mjlab`` provides PAL Robotics' robots and training tasks on top of
`mjlab <https://github.com/mujocolab/mjlab>`_.

Watch Kangaroo walking outdoor with a locomotion policy trained in ``pal_mjlab``:

.. raw:: html

   <div style="position:relative;width:100%;aspect-ratio:16/9;margin-bottom:1em;">
     <iframe src="https://www.youtube-nocookie.com/embed/ky1aEw12mwM?autoplay=1&mute=1"
             title="Kangaroo walking outdoor" style="position:absolute;inset:0;width:100%;height:100%;border:0;"
             referrerpolicy="strict-origin-when-cross-origin"
             allow="autoplay; encrypted-media; picture-in-picture; fullscreen" allowfullscreen></iframe>
   </div>

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   release_notes

.. toctree::
   :maxdepth: 1
   :caption: Reinforcement Learning

   rl/ppo
   rl/training_env_building

.. toctree::
   :maxdepth: 1
   :caption: PAL Kangaroo

   kangaroo/kangaroo_overview
   kangaroo/models/kangaroo_simple
   kangaroo/models/kangaroo_lower_body
   kangaroo/models/kangaroo_hands
   kangaroo/models/kangaroo_grippers
   kangaroo/tasks/task_velocity
   kangaroo/tasks/task_imitation
   kangaroo/tasks/task_reaching
