from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.viewer.debug_visualizer import DebugVisualizer

if TYPE_CHECKING:
  import mujoco
  import viser

  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class PieceWiseVelocityCommand(CommandTerm):
  """Mux command term that assigns each env to one sub-command.

  Owns and drives a set of sub-`CommandTerm`s. On every env reset, each env is
  reassigned to a piece sampled according to the configured proportions, and
  the `command` returned for env `i` is the command of the piece currently
  assigned to env `i`.
  """

  cfg: PieceWiseVelocityCommandCfg

  def __init__(self, cfg: PieceWiseVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    assert len(cfg.pieces) > 0, "PieceWiseVelocityCommand requires at least one piece."

    # Build (and own) sub-commands. The global CommandManager only sees this
    # mux, so it never calls compute/reset on the sub-commands directly.
    self._pieces: dict[str, CommandTerm] = {
      name: piece_cfg.cmd.build(env) for name, piece_cfg in cfg.pieces.items()
    }

    # Normalize proportions across pieces (in the same order as `_pieces`).
    proportions = np.array(
      [piece_cfg.proportion for piece_cfg in cfg.pieces.values()], dtype=np.float64
    )
    self._proportions = proportions / proportions.sum()

    seed = cfg.seed if cfg.seed is not None else np.random.randint(0, 10000)
    self.np_rng = np.random.default_rng(seed)

    # Piece index assigned to each env. Updated on every reset.
    self._cmd_idxs = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
    self._env_arange = torch.arange(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    # All sub-commands must produce the same shape [N, D].
    sub_cmds = [sub.command for sub in self._pieces.values()]
    expected_shape = sub_cmds[0].shape
    for c in sub_cmds[1:]:
      assert c.shape == expected_shape, (
        f"PieceWiseVelocityCommand requires all pieces to produce the same "
        f"command shape; got {expected_shape} and {c.shape}."
      )
    stacked = torch.stack(sub_cmds, dim=0)  # [P, N, D]
    return stacked[self._cmd_idxs, self._env_arange]

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

    # Reset every sub-command for these envs. Entries not assigned to a given
    # piece are harmless — their command output is never read for those envs.
    for piece_name, sub in self._pieces.items():
      sub_extra = sub.reset(env_ids)
      for k, v in sub_extra.items():
        extra[f"{piece_name}/{k}"] = v

    return extra

  def compute(self, dt: float) -> None:
    # Bypass the base class's time_left machinery — each sub-command runs its
    # own resampling schedule through its own compute().
    for sub in self._pieces.values():
      sub.compute(dt)

  # Viewer / GUI plumbing — forward each hook to all sub-commands. Per-piece
  # gating (e.g. `cfg.debug_vis`) lives on the sub-command's own cfg.

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    # Filter each piece's visualizer view to only the envs assigned to it,
    # so the viewer shows the piece active for the env being inspected.
    cmd_idxs_cpu = self._cmd_idxs.cpu().numpy()
    for piece_idx, sub in enumerate(self._pieces.values()):
      mask = cmd_idxs_cpu == piece_idx
      sub.debug_vis(_PieceVisualizerProxy(visualizer, mask))

  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    # Namespace each piece's controls under "<command_name>.<piece_name>" so
    # the viewer surfaces one slider group per piece.
    for piece_name, sub in self._pieces.items():
      sub.create_gui(
        f"{name}.{piece_name}", server, get_env_idx, on_change, request_action
      )

  def on_viewer_pause(self, paused: bool) -> None:
    for sub in self._pieces.values():
      sub.on_viewer_pause(paused)

  def apply_gui_reset(self, env_ids: torch.Tensor) -> bool:
    applied = False
    for sub in self._pieces.values():
      applied |= sub.apply_gui_reset(env_ids)
    return applied

  # Abstract methods required by CommandTerm — unused because `compute` is
  # overridden and metrics are surfaced via sub-command resets.

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    pass

  def _update_command(self) -> None:
    pass


class _PieceVisualizerProxy(DebugVisualizer):
  """DebugVisualizer that filters `get_env_indices` to envs assigned to a
  specific piece. All other operations forward to the wrapped visualizer."""

  def __init__(self, base: DebugVisualizer, assigned_mask: np.ndarray):
    self._base = base
    self._mask = assigned_mask

  @property
  def env_idx(self) -> int:  # type: ignore[override]
    return self._base.env_idx

  @property
  def show_all_envs(self) -> bool:  # type: ignore[override]
    return self._base.show_all_envs

  @property
  def meansize(self) -> float:
    return self._base.meansize

  def get_env_indices(self, num_envs: int):
    return [i for i in self._base.get_env_indices(num_envs) if self._mask[i]]

  def add_arrow(
    self,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[float, float, float, float],
    width: float = 0.015,
    label: str | None = None,
  ) -> None:
    self._base.add_arrow(start, end, color, width, label)

  def add_ghost_mesh(
    self,
    qpos: np.ndarray,
    model: mujoco.MjModel, # type: ignore
    mocap_pos: np.ndarray | None = None,
    mocap_quat: np.ndarray | None = None,
    alpha: float = 0.5,
    label: str | None = None,
  ) -> None:
    self._base.add_ghost_mesh(qpos, model, mocap_pos, mocap_quat, alpha, label)

  def add_frame(
    self,
    position: np.ndarray,
    rotation_matrix: np.ndarray,
    scale: float = 0.3,
    label: str | None = None,
    axis_radius: float = 0.01,
    alpha: float = 1.0,
    axis_colors: tuple[tuple[float, float, float], ...] | None = None,
  ) -> None:
    self._base.add_frame(
      position, rotation_matrix, scale, label, axis_radius, alpha, axis_colors
    )

  def add_sphere(
    self,
    center: np.ndarray,
    radius: float,
    color: tuple[float, float, float, float],
    label: str | None = None,
  ) -> None:
    self._base.add_sphere(center, radius, color, label)

  def add_cylinder(
    self,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    color: tuple[float, float, float, float],
    label: str | None = None,
  ) -> None:
    self._base.add_cylinder(start, end, radius, color, label)

  def add_ellipsoid(
    self,
    center: np.ndarray,
    size: np.ndarray,
    mat: np.ndarray,
    color: tuple[float, float, float, float],
    label: str | None = None,
  ) -> None:
    self._base.add_ellipsoid(center, size, mat, color, label)

  def clear(self) -> None:
    self._base.clear()


@dataclass(kw_only=True)
class PieceCommandCfg:
  cmd: CommandTermCfg
  proportion: float = 1.0


@dataclass(kw_only=True)
class PieceWiseVelocityCommandCfg(CommandTermCfg):
  pieces: dict[str, PieceCommandCfg] = field(default_factory=dict)
  seed: int | None = None
  # We don't use our own time_left (sub-commands resample independently), so
  # keep the base class's resampling timer effectively dormant.
  resampling_time_range: tuple[float, float] = (1e9, 1e9)

  def build(self, env: ManagerBasedRlEnv) -> PieceWiseVelocityCommand:
    return PieceWiseVelocityCommand(self, env)
