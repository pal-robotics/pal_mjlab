"""The command's differentiable twin: what a task provides so SuperMimic can imagine it.

In the simulator the env's command term owns the command: how it evolves, the goal the
policy sees, when the episode ends. Imagination has no env, so the task provides a
``CommandModel`` that does the same on tensors. The method only ever sees an opaque command
state ``c``: a dict of fixed-shape tensors, (B, ...) each, that it stores in replay and hands
back to the model. What the tensors mean (a clip and frame, a velocity command, ...) is the
task's business. What the command learns across episodes (e.g. a start sampler) stays
on the env's command term, which checkpoints it (``pal_mjlab/rl/command_state.py``).

The env's command term provides its model (``command_model()``), as it owns everything else
about the command (e.g. its start sampler); ``command_model_of`` fetches and checks it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

CommandState = dict[str, torch.Tensor]
"""Opaque command state c: fixed-shape tensors, (B, ...) each."""


@runtime_checkable
class CommandModel(Protocol):
  """Tensor-only twin of an env command term. ``state``/``pose`` are heading states and
  root poses (``state.py``), in the env's local frame."""

  # Simulator side (collector) -------------------------------------------------------

  def state(self) -> CommandState:
    """c_t of every env, read from the env's command term (stored in replay)."""
    ...

  # Imagination side (pure tensor functions, compilable) ------------------------------

  def advance(self, c: CommandState) -> CommandState:
    """c_{t+1} from c_t: how the command moves over one env step."""
    ...

  def done(self, c: CommandState) -> torch.Tensor:
    """(B,) bool: the step taken in c ends the episode (a time limit, not a failure)."""
    ...

  def goal(self, c: CommandState, state: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
    """(B, G) the policy's goal input, as the env's goal observation group computes it."""
    ...

  def termination_ref(self, c: CommandState) -> torch.Tensor:
    """(B, 2) target root height and projected-gravity z: the termination head's reference."""
    ...

  def critic_extras(
    self, c: CommandState, state: torch.Tensor, pose: torch.Tensor
  ) -> torch.Tensor:
    """(B, E) privileged critic features after (state, goal, context); E may be 0."""
    ...


def command_model_of(env: ManagerBasedRlEnv, command_name: str) -> CommandModel:
  """The ``CommandModel`` of the env's command term ``command_name``."""
  term = env.command_manager.get_term(command_name)
  provide = getattr(term, "command_model", None)
  if not callable(provide):
    raise TypeError(
      f"Command {command_name!r} ({type(term).__name__}) has no command_model(): "
      "SuperMimic needs the command's CommandModel to imagine it."
    )
  model = provide()
  if not isinstance(model, CommandModel):
    missing = [
      m
      for m in ("state", "advance", "done", "goal", "termination_ref", "critic_extras")
      if not callable(getattr(model, m, None))
    ]
    raise TypeError(
      f"{type(term).__name__}.command_model() gave a {type(model).__name__}, not a "
      f"CommandModel: missing {missing}."
    )
  return model
