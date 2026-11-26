"""Checkpointing the env's command terms: whatever a command learns across episodes.

A command term can keep training state outside the policy, e.g. a start sampler that learns
which parts of a motion are hard. It belongs to the command, not to the learning method,
so a runner saves it per command term: every term with ``state_dict()`` and
``load_state_dict(state)`` is saved under its name, and restored on resume.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _stateful_terms(env: ManagerBasedRlEnv) -> dict[str, Any]:
  cm = env.command_manager
  terms = {name: cm.get_term(name) for name in cm.active_terms}
  return {
    name: term
    for name, term in terms.items()
    if callable(getattr(term, "state_dict", None))
    and callable(getattr(term, "load_state_dict", None))
  }


def command_state_dicts(env: ManagerBasedRlEnv) -> dict[str, dict]:
  """``{term name: term.state_dict()}`` for every stateful command term."""
  return {name: term.state_dict() for name, term in _stateful_terms(env).items()}


def load_command_state_dicts(
  env: ManagerBasedRlEnv, states: dict[str, dict], strict: bool = True
) -> None:
  """Restore ``command_state_dicts``' output. ``strict``: every stateful term must be in
  ``states`` and every saved state must have its term (a resumed run would otherwise
  silently restart a command's curriculum)."""
  terms = _stateful_terms(env)
  if strict:
    missing, unexpected = terms.keys() - states.keys(), states.keys() - terms.keys()
    if missing or unexpected:
      raise KeyError(
        "Checkpoint and env disagree on the stateful command terms: "
        f"missing from the checkpoint {sorted(missing)}, "
        f"not in the env {sorted(unexpected)}."
      )
  for name, state in states.items():
    if name in terms:
      terms[name].load_state_dict(state)
