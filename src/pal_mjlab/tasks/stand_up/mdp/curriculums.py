"""Curriculum terms of the stand-up task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .contacts import StandCfg, is_standing

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg


class stand_up_curriculum:
  """Remove the lift, then raise the care rewards as far as the robot copes.

  Follows the share of episodes that end with the robot standing (a running
  average over about ``window`` episodes). Every ``update_interval`` steps:

  - While the lift of ``assist_event_name`` is on, it is lowered by
    ``assist_step`` if that share is above ``raise_above`` and raised by it,
    up to ``assist_level``, below ``lower_below``: the share follows the lift
    with a delay, so a step can go too far before it shows.
  - Afterwards, the weights of ``care_rewards`` are scaled by a common factor
    that is multiplied by ``care_step_factor`` above ``raise_above`` and
    divided by it below ``lower_below``. Each reward follows the factor up to
    its own maximum, the value ``care_rewards`` gives for its name.

  ``state_dict`` and ``load_state_dict`` let a runner keep the state in its
  checkpoints.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    care_rewards: dict[str, float] = params["care_rewards"]
    self._terms = [env.reward_manager.get_term_cfg(name) for name in care_rewards]
    self._base_weights = [term.weight for term in self._terms]
    self._max_scales = list(care_rewards.values())
    self._assist = env.event_manager.get_term_cfg(params["assist_event_name"])
    self.assist_level = float(params["assist_level"])
    self._max_assist_level = self.assist_level
    self.care_scale = 1.0
    self.success = 0.0
    self._last_update_step = 0
    self._apply()

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    del env_ids

  def state_dict(self) -> dict[str, float]:
    return {
      "assist_level": self.assist_level,
      "care_scale": self.care_scale,
      "success": self.success,
      "last_update_step": self._last_update_step,
    }

  def load_state_dict(self, state: dict[str, float]) -> None:
    self.assist_level = float(state["assist_level"])
    self.care_scale = float(state["care_scale"])
    self.success = float(state["success"])
    self._last_update_step = int(state["last_update_step"])
    self._apply()

  def _apply(self) -> None:
    for term, weight, max_scale in zip(
      self._terms, self._base_weights, self._max_scales, strict=True
    ):
      term.weight = weight * min(self.care_scale, max_scale)
    self._assist.params["level"] = self.assist_level

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice,
    care_rewards: dict[str, float],
    stand_cfg: StandCfg,
    assist_event_name: str,
    assist_level: float,
    assist_step: float = 0.1,
    care_step_factor: float = 1.2,
    raise_above: float = 0.95,
    lower_below: float = 0.9,
    update_interval: int = 600,
    window: int = 2000,
  ) -> dict[str, torch.Tensor]:
    del care_rewards, assist_event_name, assist_level  # Resolved in __init__.
    # Called before the reset: the state is the one the episodes ended in.
    # Environments that have not stepped yet have no episode to count.
    if not isinstance(env_ids, torch.Tensor):
      env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    ended = env_ids[env.episode_length_buf[env_ids] > 0]
    if len(ended) > 0:
      stood = float(is_standing(env, stand_cfg)[ended].float().mean())
      self.success += (stood - self.success) * min(1.0, len(ended) / window)

    if env.common_step_counter - self._last_update_step >= update_interval:
      self._last_update_step = env.common_step_counter
      if self.assist_level > 0.0:
        if self.success > raise_above:
          self.assist_level = max(0.0, round(self.assist_level - assist_step, 6))
        elif self.success < lower_below:
          self.assist_level = min(
            self._max_assist_level, round(self.assist_level + assist_step, 6)
          )
      elif self.success > raise_above:
        self.care_scale = min(max(self._max_scales), self.care_scale * care_step_factor)
      elif self.success < lower_below:
        self.care_scale = max(1.0, self.care_scale / care_step_factor)
      self._apply()
    return {
      "assist_level": torch.tensor(self.assist_level),
      "care_scale": torch.tensor(self.care_scale),
      "stood_up": torch.tensor(self.success),
    }
