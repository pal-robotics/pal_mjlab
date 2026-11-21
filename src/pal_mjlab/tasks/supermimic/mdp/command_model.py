"""``MotionLibraryCommand``'s twin for SuperMimic's imagination (``rl/supermimic/command_model.py``).

The command state is the env's (clip, target frame) pair, as global library frame indices.
Each method mirrors the command term's simulator code in ``commands.py`` on that pair:
the frame advances to the clip's last frame and stays there, the episode ends on the step
that targets it, the goal is the next K frames, and the termination reference is the target
frame's root height and tilt. States are in the env's local frame, like the library.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from pal_mjlab.rl.supermimic.command_model import CommandState
from pal_mjlab.rl.supermimic.state import state_to_qpos_qvel

from .tracking import BodyState, projected_gravity_z, reference_goal, root_pose_error

if TYPE_CHECKING:
  from .commands import MotionLibraryCommand

__all__ = ["MotionLibraryCommandModel"]


class MotionLibraryCommandModel:
  """``c = {"clip": (B,) long, "frame": (B,) long}``."""

  def __init__(self, cmd: MotionLibraryCommand) -> None:
    """Built by the command (``MotionLibraryCommand.command_model``)."""
    self.cmd = cmd
    self.library = cmd.library
    self.num_future_frames = cmd.cfg.num_future_frames
    self.time_left_cap = cmd.cfg.time_left_cap_s / cmd._env.step_dt

  # Simulator side ---------------------------------------------------------------

  def state(self) -> CommandState:
    return {"clip": self.cmd.clip.clone(), "frame": self.cmd.frame.clone()}

  # Imagination side -------------------------------------------------------------

  def _last(self, c: CommandState) -> torch.Tensor:
    return self.library.clip_end[c["clip"]] - 1

  def advance(self, c: CommandState) -> CommandState:
    return {"clip": c["clip"], "frame": torch.minimum(c["frame"] + 1, self._last(c))}

  def done(self, c: CommandState) -> torch.Tensor:
    """The step that targets the clip's last frame ends the episode (``at_clip_end``)."""
    return c["frame"] >= self._last(c)

  def goal(self, c: CommandState, state: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """K future reference frames from the target frame on; past the clip end, the last."""
    root_quat = state_to_qpos_qvel(state, pose)[0][:, 3:7]
    k = torch.arange(self.num_future_frames, device=root_quat.device)
    idx = torch.minimum(c["frame"][:, None] + k, self._last(c)[:, None])
    return reference_goal(self.library.qpos[idx], self.library.qvel[idx], root_quat)

  def termination_ref(self, c: CommandState) -> torch.Tensor:
    qp = self.library.qpos[c["frame"]]
    return torch.stack([qp[:, 2], projected_gravity_z(qp[:, 3:7])], dim=-1)

  def critic_extras(
    self, c: CommandState, state: torch.Tensor, pose: torch.Tensor
  ) -> torch.Tensor:
    """(B, 5): root-pose error to the target frame (4) and time left in the clip (1)."""
    qpos, _ = state_to_qpos_qvel(state, pose)
    ref = self.library.qpos[c["frame"]]
    err = root_pose_error(qpos[:, :3], qpos[:, 3:7], ref[:, :3], ref[:, 3:7])
    left = (self._last(c) - c["frame"]).float()
    time_left = (left.clamp(max=self.time_left_cap) / self.time_left_cap)[:, None]
    return torch.cat([err, time_left], dim=-1)

  # Motion-specific: the tracking rewards' reference --------------------------------

  @torch.no_grad()
  def reference(self, c: CommandState) -> BodyState:
    """Tracked reference bodies at the target frames (local frame, like the states)."""
    f = c["frame"]
    return self.cmd.bodies(self.library.qpos[f], self.library.qvel[f])
