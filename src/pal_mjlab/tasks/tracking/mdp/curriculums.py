from __future__ import annotations
from typing import TYPE_CHECKING, Any
import torch
from mjlab.envs.mdp.curriculums import _apply_stages, _validate_stages

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.curriculum_manager import CurriculumTermCfg

class command_curriculum:
  """Update a command term's params based on training steps or iterations.

  Each stage specifies a ``step`` threshold and a dict of fields to update.
  If ``num_steps_per_iteration`` is provided in params, the counter is divided
  by this value to treat the thresholds as iterations.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    command_name: str = cfg.params["command_name"]
    stages: list[Any] = cfg.params["stages"]
    self._term_cfg = env.command_manager.get_term_cfg(command_name)
    self._stages = stages
    self._steps_per_iteration = cfg.params.get("num_steps_per_iteration", 1)
    self._validate_stages_safe(self._term_cfg, command_name, self._stages)

  def _validate_stages_safe(self, term_cfg: Any, term_name: str, stages: list[Any]):
    """Safe version of _validate_stages that handles configs without a 'params' dict."""
    for i in range(1, len(stages)):
      if stages[i]["step"] < stages[i - 1]["step"]:
        raise ValueError(f"Stages must be in nondecreasing step order.")
    for stage in stages:
      for key in stage:
        if key not in {"step", "params"} and not hasattr(term_cfg, key):
          raise AttributeError(f"Field '{key}' does not exist on '{term_name}' config.")
      if "params" in stage:
        if not hasattr(term_cfg, "params"):
          raise AttributeError(f"'{term_name}' config does not have a 'params' dict.")
        unknown = stage["params"].keys() - term_cfg.params.keys()
        if unknown:
          raise KeyError(f"Unknown params: {unknown}")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    stages: list[Any],
  ) -> dict[str, torch.Tensor]:
    del env_ids, command_name, stages
    counter = env.common_step_counter
    if self._steps_per_iteration > 1:
      counter = counter // self._steps_per_iteration
    return _apply_stages(self._term_cfg, counter, self._stages)

