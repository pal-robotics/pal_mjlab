import functools

import torch


def nan_safe(fn):
  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    return torch.nan_to_num(fn(*args, **kwargs), nan=0.0, posinf=0.0, neginf=0.0)

  return wrapper
