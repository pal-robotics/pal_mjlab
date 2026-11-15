"""Curriculum terms for the safe-fall task."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypedDict

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg


class EventParamStage(TypedDict):
  step: int
  params: dict[str, Any]


class event_param_curriculum:
  """Update the params of an event term based on training steps.

  When ``env.common_step_counter`` reaches a stage's ``step``, its ``params``
  are written into the event term. Later stages take precedence. Used to widen
  the range of fall-inducing kicks as training progresses.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    event_name: str = cfg.params["event_name"]
    stages: list[EventParamStage] = cfg.params["stages"]
    self._term_cfg = env.event_manager.get_term_cfg(event_name)
    for i in range(1, len(stages)):
      if stages[i]["step"] < stages[i - 1]["step"]:
        raise ValueError("Curriculum stages must be in nondecreasing step order.")
    for stage in stages:
      unknown = stage["params"].keys() - self._term_cfg.params.keys()
      if unknown:
        raise KeyError(f"Unknown param(s) {unknown} on event term '{event_name}'.")
    self._stages = stages

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    event_name: str,
    stages: list[EventParamStage],
  ) -> dict[str, torch.Tensor]:
    del env_ids, event_name, stages
    logged: set[str] = set()
    for stage in self._stages:
      logged.update(stage["params"].keys())
      if env.common_step_counter >= stage["step"]:
        self._term_cfg.params.update(stage["params"])
    return {
      key: torch.tensor(float(self._term_cfg.params[key]))
      for key in logged
      if isinstance(self._term_cfg.params[key], (int, float))
    }
