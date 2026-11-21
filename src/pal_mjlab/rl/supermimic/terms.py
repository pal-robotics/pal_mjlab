"""Single-definition versions of the mjlab terms SuperMimic tasks use (``imagination.py``).

Each is the mjlab term of the same name, written on a ``Batch`` so that imagination
computes it exactly as the env does. SuperMimic tasks use these in their env configs;
other tasks keep mjlab's.
"""

from __future__ import annotations

import torch

from .imagination import Batch, imaginable

__all__ = ["action_rate_l2"]


@imaginable
def action_rate_l2(b: Batch) -> torch.Tensor:
  """mjlab's ``action_rate_l2``: squared change of the raw policy action."""
  return (b.action - b.prev_action).square().sum(-1)
