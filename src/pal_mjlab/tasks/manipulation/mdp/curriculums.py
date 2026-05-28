from __future__ import annotations

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.envs.mdp.curriculums import _apply_stages


class observation_curriculum:
  """Update an observation term's params based on training steps.

  Each stage specifies a ``step`` threshold and a ``params`` dict. When
  ``env.common_step_counter`` reaches a stage's ``step``, the params are
  applied. Later stages take precedence.

  Example::

    CurriculumTermCfg(
      func="pal_mjlab.tasks.manipulation.mdp.curriculums:observation_curriculum",
      params={
        "group_name": "camera",
        "term_name": "head_camera_keypoints",
        "stages": [
          {"step": 0, "params": {"noise_std": 0.0}},
          {"step": 5000, "params": {"noise_std": 0.02}},
        ],
      },
    )
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv):
    group_name: str = cfg.params["group_name"]
    term_name: str = cfg.params["term_name"]
    self._term_cfg = env.observation_manager.get_term_cfg(group_name, term_name)
    self._stages = cfg.params["stages"]

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    group_name: str,
    term_name: str,
    stages: list[dict],
  ) -> dict[str, torch.Tensor]:
    del env_ids, group_name, term_name, stages
    return _apply_stages(self._term_cfg, env.common_step_counter, self._stages)
