from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  wrap_to_pi,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg

if TYPE_CHECKING:
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class PieceWiseVelocityCommand(CommandTerm):
  cfg: PieceWiseVelocityCommandCfg

  def __init__(self, cfg: PieceWiseVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    # Normalize the proportions of the sub-commands.
    self._proportions = np.array(
      [piece_cfg.proportion for piece_cfg in self.cfg.pieces.values()]
    )
    self._proportions /= np.sum(self._proportions)

    if self.cfg.seed is not None:
      seed = self.cfg.seed
    else:
      seed = np.random.randint(0, 10000)
    self.np_rng = np.random.default_rng(seed)

    # The command will get chosen at random every reset
    self._cmd_idxs = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.long
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    extra = super().reset(env_ids)
    assert isinstance(env_ids, torch.Tensor)

    chosen = self.np_rng.choice(
      len(self._proportions),
      size=env_ids.shape[0],
      p=self._proportions,
    )

    self._cmd_idxs[env_ids] = torch.from_numpy(chosen).to(
      device=self.device, dtype=torch.long
    )
    return extra

  @property
  def command(self) -> torch.Tensor:
    cmds = list(self.cfg.pieces.values())
    cmds[0].cmd.command
    return torch.zeros(self.num_envs, 3, device=self.device)

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    pass

  def _update_command(self) -> None:
    pass

  # GUI.

  def compute(self, dt: float) -> None:
    super().compute(dt)


@dataclass
class PieceCommandCfg():
  cmd: CommandTermCfg
  proportion: float = 1.0
  

@dataclass(kw_only=True)
class PieceWiseVelocityCommandCfg(CommandTermCfg):
  seed: int | None = None
  pieces: dict[str, PieceCommandCfg] = field(default_factory=dict)

  def build(self, env: ManagerBasedRlEnv) -> PieceWiseVelocityCommand:
    return PieceWiseVelocityCommand(self, env)
