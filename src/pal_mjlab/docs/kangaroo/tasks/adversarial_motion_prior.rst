.. _Kangaroo adversarial_motion_prior:

Adversarial Motion Prior (AMP)
==============================

Rather than a task, Adversarial Motion Prior is a tool to train motion to have more "natural" 
behavior (human-like walking, for example). It introduces a new reward-like term, that rewards
the robot for moving like a reference motion. In order to determine if robot behavior ressembles
that of the motion, a discriminator is trained in parallel of the policy. The discriminator is trained
to return -1 when receiving a transition that comes from the robot, and to treturn 1 when it comes
from the reference. This way, during training, the policy needs to generate behavior similar
to the motion in order to trick the discriminator and maximize the reward.

|

Use case
--------

Adversarial Motion Prior can be applied to any task, as soon as the user would want stilized
movement from the policy. The idea is to not converge to optimal mechanical solutions for
the objectives, but to find a nice equilibrium between tracking objectives and imitating the
reference motion. It is also use to guide robot movement when doing complex tasks. Here are some examples :

- Stylized walking

- Box lifting

- Jumping

- ...

AMP allows to perform complex task without being limited by the lack of parameters of imitation learning,
and bypassing the complex pproblem that is learning non Markovian (not state dependant) rewards in classical RL.

Setting up an AMP environment
-----------------------------

To set up an environment that makes use of AMP, one just nees to define the task as using the the runner made
for it, 'AmpOnPolicyRunner'. Then remains to pass trough the CLI a reference motion file and to set
the weight of the discriminator reward if needed. Nothing else in the environment is related to AMP
functionning, but one would most likely have different rewards for a task with AMP that those for the
same task without it.

On paper, you don't need "style" rewards when training with AMP, since imitating the reference motion
should take care of it.

Reference motion
----------------

As of theactual implementation, the motion shall be a csv file containing observation frames (default is 
joint position and joint velocity, but it can be changed). One must set an observation group that will be used
training to train the discriminator and compute its reward signal. Indeed, the discriminator can compare robot 
motion and reference motion with observations other than actor or critic observations.

.. important::

   Simulation observation and reference moton frames must match, otherwise the robot would
   learn very erratic behavior.

The reference motion may use any frame rate; the runner resamples it by
interpolating values to match the policy's control frequency.

.. warning::

   This works well for scalar continuous values such as joint positions or
   velocities, but may behave unexpectedly for values that require
   specialized interpolation (e.g., quaternions, flags, ...).
