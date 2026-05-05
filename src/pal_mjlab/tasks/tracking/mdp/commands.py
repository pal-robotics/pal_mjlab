from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class PalMotionCommand(MotionCommand):
  cfg: PalMotionCommandCfg

  def _uniform_sampling(self, env_ids: torch.Tensor) -> None:
    effective_total = max(1, int(self.motion.time_step_total * self.cfg.max_time_fraction))
    self.time_steps[env_ids] = torch.randint(
      0, effective_total, (len(env_ids),), device=self.device
    )
    self.metrics["sampling_entropy"][:] = 1.0
    self.metrics["sampling_top1_prob"][:] = 1.0 / self.bin_count
    self.metrics["sampling_top1_bin"][:] = 0.5

  def _adaptive_sampling(self, env_ids: torch.Tensor) -> None:
    episode_failed = self._env.termination_manager.terminated[env_ids]
    if torch.any(episode_failed):
      current_bin_index = torch.clamp(
        (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1),
        0,
        self.bin_count - 1,
      )
      fail_bins = current_bin_index[env_ids][episode_failed]
      self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

    sampling_probabilities = (
      self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
    )
    sampling_probabilities = torch.nn.functional.pad(
      sampling_probabilities.unsqueeze(0).unsqueeze(0),
      (0, self.cfg.adaptive_kernel_size - 1),
      mode="replicate",
    )
    sampling_probabilities = torch.nn.functional.conv1d(
      sampling_probabilities, self.kernel.view(1, 1, -1)
    ).view(-1)

    effective_bin_count = max(1, int(self.bin_count * self.cfg.max_time_fraction))
    sampling_probabilities[effective_bin_count:] = 0.0
    sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

    sampled_bins = torch.multinomial(
      sampling_probabilities, len(env_ids), replacement=True
    )
    self.time_steps[env_ids] = (
      (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
      / self.bin_count
      * (self.motion.time_step_total - 1)
    ).long()

    H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
    H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else 1.0
    pmax, imax = sampling_probabilities.max(dim=0)
    self.metrics["sampling_entropy"][:] = H_norm
    self.metrics["sampling_top1_prob"][:] = pmax
    self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count


@dataclass(kw_only=True)
class PalMotionCommandCfg(MotionCommandCfg):
  max_time_fraction: float = 1.0

  @dataclass
  class VizCfg(MotionCommandCfg.VizCfg):
    pass

  viz: MotionCommandCfg.VizCfg = field(default_factory=MotionCommandCfg.VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> PalMotionCommand:
    return PalMotionCommand(self, env)
