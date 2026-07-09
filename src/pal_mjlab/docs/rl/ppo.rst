.. _PPO overview:

Proximal Policy Optimization (PPO)
==================================

mjlab uses `rsl_rl <https://github.com/leggedrobotics/rsl_rl>`_'s implementation of PPO for reinforcement learning.

This document gives a quick overview of the algorithm and of the parameters a user may change to tune learning.

Overview
--------

PPO is an on-policy actor-critic algorithm that alternates between collecting rollouts from the environment and performing several epochs of stochastic gradient descent on the collected data. It constrains the policy update at each step to stay close to the previous policy, which prevents destructively large updates and stabilises training.

The key idea is the clipped surrogate objective:

.. math::

   L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta),\ 1 - \epsilon,\ 1 + \epsilon) \hat{A}_t \right) \right]

where ``r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)`` is the probability ratio and ``Â_t`` is the estimated advantage at timestep ``t``. The clip parameter ``ε`` limits how far the new policy can deviate from the old one.

The full objective also includes a value function loss and an entropy bonus:

.. math::

   L(\theta) = L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta](s_t)

Algorithm Flow
--------------

1. **Rollout collection** — the current policy :math:`\pi_{\theta_{old}}` interacts with the environment for a fixed number of steps, storing observations, actions, rewards, and values.
2. **Advantage estimation** — Generalised Advantage Estimation (GAE) computes :math:`\hat{A}_t` from the collected rewards and value predictions.
3. **Mini-batch updates** — the collected data is shuffled into mini-batches and the policy and value networks are updated for multiple epochs.
4. **Policy replacement** — :math:`\pi_{\theta_{old}}` is replaced with the new :math:`\pi_\theta` and the cycle repeats.

Key Parameters
--------------

The table below lists the main parameters exposed by rsl_rl's PPO runner.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Parameter
     - Default
     - Description
   * - ``num_steps_per_env``
     - 24
     - Number of environment steps collected per rollout before each update. Larger values give better gradient estimates but increase memory usage.
   * - ``num_learning_epochs``
     - 5
     - Number of passes over the collected rollout data during each update phase. Higher values extract more signal per rollout but risk overfitting to stale data.
   * - ``num_mini_batches``
     - 4
     - Number of mini-batches the rollout buffer is split into per epoch. Increasing this reduces the variance of each gradient step.
   * - ``learning_rate``
     - 1e-3
     - Step size for the Adam optimiser. Use a learning rate schedule (``schedule = "adaptive"``) to decay it automatically.
   * - ``schedule``
     - ``"adaptive"``
     - Learning rate schedule. ``"adaptive"`` scales the rate based on the KL divergence between old and new policies; ``"fixed"`` keeps it constant.
   * - ``gamma``
     - 0.99
     - Discount factor for future rewards. Lower values make the agent more myopic; higher values give a longer planning horizon.
   * - ``lam``
     - 0.95
     - GAE lambda. Controls the bias–variance trade-off in advantage estimation. ``lam=1`` gives Monte Carlo returns; ``lam=0`` gives one-step TD.
   * - ``desired_kl``
     - 0.01
     - Target KL divergence used by the adaptive learning rate schedule. Ignored when ``schedule = "fixed"``.
   * - ``max_grad_norm``
     - 1.0
     - Maximum norm for gradient clipping. Prevents excessively large updates.
   * - ``clip_param``
     - 0.2
     - The :math:`\epsilon` in the clipped surrogate objective. Larger values allow bigger policy changes per update.
   * - ``entropy_coef``
     - 0.01
     - Coefficient :math:`c_2` on the entropy bonus. Encourages exploration by penalising overly deterministic policies.
   * - ``value_loss_coef``
     - 1.0
     - Coefficient :math:`c_1` on the value function loss relative to the policy loss.
   * - ``use_clipped_value_loss``
     - ``True``
     - Whether to apply the same clipping trick to the value function update, which can stabilise critic training.

Tuning Guidelines
-----------------

**Unstable or diverging training**
   Reduce ``learning_rate`` or ``clip_param``. Enable ``schedule = "adaptive"`` with a small ``desired_kl`` (e.g. 0.005) to automatically throttle updates that are too large. Lower ``num_learning_epochs`` to reduce the number of stale-data gradient steps.

**Slow or stagnating learning**
   Increase ``learning_rate`` or ``num_steps_per_env`` to collect richer rollouts. Raise ``entropy_coef`` if the policy appears to converge prematurely. Increase ``num_learning_epochs`` to extract more signal from each rollout.

**High variance returns**
   Lower ``lam`` towards 0.9 to reduce variance in advantage estimation at the cost of some bias. Increase ``num_steps_per_env`` to improve value function targets.

**Exploration issues**
   Increase ``entropy_coef``. If the policy collapses to a single action early in training, consider annealing ``entropy_coef`` from a higher starting value.

References
----------

* Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
* Schulman, J. et al. (2015). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. arXiv:1506.02438.
* rsl_rl source code: https://github.com/leggedrobotics/rsl_rl