from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.mdp.curriculums import VelocityStage

from .commands import PieceWiseVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def piecewise_commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  piece_stages: dict[str, list[VelocityStage]],
) -> dict[str, torch.Tensor]:
  """Per-piece velocity-range curriculum for a ``PieceWiseVelocityCommand``.

  Mirrors ``mjlab.tasks.velocity.mdp.commands_vel`` but applies each piece's
  stage list to its own ``UniformVelocityCommandCfg.ranges`` independently.
  Stages are evaluated in order; later stages whose ``step`` has been passed
  take precedence. Pieces absent from ``piece_stages`` are left untouched.

  Example::

    CurriculumTermCfg(
      func=piecewise_commands_vel,
      params={
        "command_name": "twist",
        "piece_stages": {
          "forward_range": [
            {"step": 0,         "lin_vel_x": (-0.3, -0.2)},
            {"step": 5000*24,   "lin_vel_x": (-0.4, -0.2)},
          ],
          "backward_range": [
            {"step": 0, "lin_vel_x": (-0.2, -0.2)},
          ],
        },
      },
    )
  """
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  piecewise_cfg = cast(PieceWiseVelocityCommandCfg, command_term.cfg)

  result: dict[str, torch.Tensor] = {}
  for piece_name, stages in piece_stages.items():
    piece_cfg = cast(
      UniformVelocityCommandCfg, piecewise_cfg.pieces[piece_name].cmd
    )
    for stage in stages:
      if env.common_step_counter >= stage["step"]:
        if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
          piece_cfg.ranges.lin_vel_x = stage["lin_vel_x"]
        if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
          piece_cfg.ranges.lin_vel_y = stage["lin_vel_y"]
        if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
          piece_cfg.ranges.ang_vel_z = stage["ang_vel_z"]
    result[f"{piece_name}/lin_vel_x_min"] = torch.tensor(piece_cfg.ranges.lin_vel_x[0])
    result[f"{piece_name}/lin_vel_x_max"] = torch.tensor(piece_cfg.ranges.lin_vel_x[1])
    result[f"{piece_name}/lin_vel_y_min"] = torch.tensor(piece_cfg.ranges.lin_vel_y[0])
    result[f"{piece_name}/lin_vel_y_max"] = torch.tensor(piece_cfg.ranges.lin_vel_y[1])
    result[f"{piece_name}/ang_vel_z_min"] = torch.tensor(piece_cfg.ranges.ang_vel_z[0])
    result[f"{piece_name}/ang_vel_z_max"] = torch.tensor(piece_cfg.ranges.ang_vel_z[1])
  return result