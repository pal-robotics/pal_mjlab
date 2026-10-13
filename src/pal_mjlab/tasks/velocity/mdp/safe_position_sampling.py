import math
import torch

def push_xy_outside_radius(xy: torch.Tensor, min_radius: float) -> torch.Tensor:
  norm = xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
  return xy * (min_radius / norm).clamp_min(1.0)


def validate_base_position(base_position, min_radius: float) -> None:
  if math.hypot(base_position[0], base_position[1]) <= min_radius:
    raise ValueError(f"base_position={base_position} is inside the robot (xy-norm <= {min_radius}).")