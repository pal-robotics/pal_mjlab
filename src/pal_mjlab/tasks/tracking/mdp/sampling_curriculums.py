from __future__ import annotations
from typing import TYPE_CHECKING, Any
import torch
from mjlab.envs.mdp.curriculums import _apply_stages, _validate_stages

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.curriculum_manager import CurriculumTermCfg

class command_curriculum:
  """Update a command term's params based on training steps.

  Each stage specifies a ``step`` threshold and a dict of fields to update.
  When ``env.common_step_counter`` reaches a stage's ``step``, the corresponding
  values are applied to the command term config.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    command_name: str = cfg.params["command_name"]
    stages: list[Any] = cfg.params["stages"]
    self._term_cfg = env.command_manager.get_term_cfg(command_name)
    self._stages = stages
    _validate_stages(self._term_cfg, command_name, self._stages)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    stages: list[Any],
  ) -> dict[str, torch.Tensor]:
    del env_ids, command_name, stages
    return _apply_stages(self._term_cfg, env.common_step_counter, self._stages)
