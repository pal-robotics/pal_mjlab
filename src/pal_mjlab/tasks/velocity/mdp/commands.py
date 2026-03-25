from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.entity import Entity
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_conjugate,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_mul,
  quat_unique,
  sample_uniform,
  wrap_to_pi,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

if TYPE_CHECKING:
  import viser

  from mjlab.envs import ManagerBasedRlEnv

class PiecewiseVelocityCommand(CommandTerm):
  """Velocity command with piecewise uniform distribution for lin/ang velocity ranges."""

  cfg: PiecewiseVelocityCommandCfg

  def __init__(self, cfg: PiecewiseVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    if self.cfg.heading_command and self.cfg.ranges.heading is None:
      raise ValueError("heading_command=True but ranges.heading is set to None.")
    if self.cfg.ranges.heading and not self.cfg.heading_command:
      raise ValueError("ranges.heading is set but heading_command=False.")

    self.robot: Entity = env.scene[cfg.entity_name]

    self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
    self.vel_command_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.heading_target = torch.zeros(self.num_envs, device=self.device)
    self.heading_error = torch.zeros(self.num_envs, device=self.device)
    self.is_heading_env = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self.is_standing_env = torch.zeros_like(self.is_heading_env)
    self.is_world_env = torch.zeros_like(self.is_heading_env)

    self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    # Set by create_gui() when the viewer is active.
    self._joystick_enabled: viser.GuiCheckboxHandle | None = None
    self._joystick_sliders: list[viser.GuiSliderHandle] = []
    self._joystick_get_env_idx: Callable[[], int] | None = None

  @property
  def command(self) -> torch.Tensor:
    return self.vel_command_b

  def _update_metrics(self) -> None:
    max_command_time = self.cfg.resampling_time_range[1]
    max_command_step = max_command_time / self._env.step_dt
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1
      )
      / max_command_step
    )
    self.metrics["error_vel_yaw"] += (
      torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2])
      / max_command_step
    )

  def _sample_piecewise_velocity(
    self, num_samples: int, ranges: list[tuple[float, float]], weights: list[float]
  ) -> torch.Tensor:
    """Sample velocities from piecewise uniform distribution.

    Args:
        num_samples: Number of samples to generate.
        ranges: List of (min, max) tuples for each range.
        weights: List of probabilities for each range (should sum to 1.0).

    Returns:
        Tensor of sampled velocities with shape (num_samples,).
    """
    # Create random values to decide which range to sample from
    rand_vals = torch.rand(num_samples, device=self.device)

    # Initialize result tensor
    result = torch.zeros(num_samples, device=self.device)

    # Cumulative probability
    cumulative_prob = 0.0

    for (lower, upper), weight in zip(ranges, weights):
      # Determine which samples fall in this range
      mask = (rand_vals >= cumulative_prob) & (rand_vals < cumulative_prob + weight)
      num_in_range = mask.sum().item()

      if num_in_range > 0:
        # Sample uniformly within this range
        sampled_values = sample_uniform(
          lower, upper, (num_in_range,), device=self.device
        )
        result[mask] = sampled_values

      cumulative_prob += weight

    return result

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    num_envs = len(env_ids)

    # Sample linear velocity x with piecewise distribution
    self.vel_command_b[env_ids, 0] = self._sample_piecewise_velocity(
      num_envs, self.cfg.ranges.lin_vel_x_ranges, self.cfg.ranges.lin_vel_x_weights
    )

    # Sample linear velocity y with piecewise distribution
    self.vel_command_b[env_ids, 1] = self._sample_piecewise_velocity(
      num_envs, self.cfg.ranges.lin_vel_y_ranges, self.cfg.ranges.lin_vel_y_weights
    )

    # Sample angular velocity z with piecewise distribution
    self.vel_command_b[env_ids, 2] = self._sample_piecewise_velocity(
      num_envs, self.cfg.ranges.ang_vel_z_ranges, self.cfg.ranges.ang_vel_z_weights
    )

    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      r = torch.empty(num_envs, device=self.device)
      self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
      self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs

    r = torch.empty(num_envs, device=self.device)
    self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    # Randomly assign world-frame envs.
    self.is_world_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_world_envs
    # Copy sampled velocities as world-frame reference for world envs.
    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]

  def _update_command(self) -> None:
    if self.cfg.heading_command:
      self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
      env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      # Get max angular velocity from the last range (highest values)
      max_ang_vel = self.cfg.ranges.ang_vel_z_ranges[-1][1]
      min_ang_vel = self.cfg.ranges.ang_vel_z_ranges[0][0]
      self.vel_command_b[env_ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * self.heading_error[env_ids],
        min=min_ang_vel,
        max=max_ang_vel,
      )

    # World-frame envs: rotate world-frame linear vel into body frame.
    if self.is_world_env.any():
      w_ids = self.is_world_env.nonzero(as_tuple=False).flatten()
      heading = self.robot.data.heading_w[w_ids]
      cos_h = torch.cos(heading)
      sin_h = torch.sin(heading)
      vx_w = self.vel_command_w[w_ids, 0]
      vy_w = self.vel_command_w[w_ids, 1]
      self.vel_command_b[w_ids, 0] = cos_h * vx_w + sin_h * vy_w
      self.vel_command_b[w_ids, 1] = -sin_h * vx_w + cos_h * vy_w

    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_env_ids, :] = 0.0
    self.vel_command_w[standing_env_ids, :] = 0.0

  # GUI and visualization methods from UniformVelocityCommand
  def create_gui(
    self,
    name: str,
    server: "viser.ViserServer",
    get_env_idx: Callable[[], int],
  ) -> None:
    """Create velocity joystick sliders in the Viser viewer."""
    from viser import Icon

    # Use the max values from the last range for GUI limits
    max_lin_x = max(abs(r[0]) for r in self.cfg.ranges.lin_vel_x_ranges + [
      (0, max(r[1] for r in self.cfg.ranges.lin_vel_x_ranges))
    ])
    max_lin_y = max(abs(r[0]) for r in self.cfg.ranges.lin_vel_y_ranges + [
      (0, max(r[1] for r in self.cfg.ranges.lin_vel_y_ranges))
    ])
    max_ang_z = max(abs(r[0]) for r in self.cfg.ranges.ang_vel_z_ranges + [
      (0, max(r[1] for r in self.cfg.ranges.ang_vel_z_ranges))
    ])

    axes = [
      ("lin_vel_x", max_lin_x),
      ("lin_vel_y", max_lin_y),
      ("ang_vel_z", max_ang_z),
    ]
    sliders: list = []

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=False)

      for label, max_val in axes:
        max_input = server.gui.add_slider(
          f"Max {label}",
          initial_value=max_val,
          step=0.1,
          min=0.1,
          max=10.0,
        )
        slider = server.gui.add_slider(
          label,
          min=-max_val,
          max=max_val,
          step=0.05,
          initial_value=0.0,
        )

        @max_input.on_update
        def _(_ev, _s=slider, _m=max_input) -> None:
          _s.min = -_m.value
          _s.max = _m.value

        sliders.append(slider)

      zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

      @zero_btn.on_click
      def _(_) -> None:
        for s in sliders:
          s.value = 0.0

    # Store GUI state for compute() override.
    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx

  def compute(self, dt: float) -> None:
    super().compute(dt)
    if self._joystick_enabled is not None and self._joystick_enabled.value:
      assert self._joystick_get_env_idx is not None
      idx = self._joystick_get_env_idx()
      for i, s in enumerate(self._joystick_sliders):
        self.vel_command_b[idx, i] = s.value

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw velocity command and actual velocity arrows."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    cmds = self.command.cpu().numpy()
    base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
    base_quat_w = self.robot.data.root_link_quat_w
    base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
    lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()
    ang_vel_bs = self.robot.data.root_link_ang_vel_b.cpu().numpy()

    scale = self.cfg.viz.scale
    z_offset = self.cfg.viz.z_offset

    for batch in env_indices:
      base_pos_w = base_pos_ws[batch]
      base_mat_w = base_mat_ws[batch]
      cmd = cmds[batch]
      lin_vel_b = lin_vel_bs[batch]
      ang_vel_b = ang_vel_bs[batch]

      # Skip if robot appears uninitialized (at origin).
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      def local_to_world(
        vec: np.ndarray, pos: np.ndarray = base_pos_w, mat: np.ndarray = base_mat_w
      ) -> np.ndarray:
        return pos + mat @ vec

      # Command linear velocity arrow (blue).
      cmd_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      cmd_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([cmd[0], cmd[1], 0])) * scale
      )
      visualizer.add_arrow(
        cmd_lin_from, cmd_lin_to, color=(0.2, 0.2, 0.6, 0.6), width=0.015
      )

      # Command angular velocity arrow (green).
      cmd_ang_from = cmd_lin_from
      cmd_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, cmd[2]])) * scale
      )
      visualizer.add_arrow(
        cmd_ang_from, cmd_ang_to, color=(0.2, 0.6, 0.2, 0.6), width=0.015
      )

      # Actual linear velocity arrow (cyan).
      act_lin_from = local_to_world(np.array([0, 0, z_offset]) * scale)
      act_lin_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([lin_vel_b[0], lin_vel_b[1], 0])) * scale
      )
      visualizer.add_arrow(
        act_lin_from, act_lin_to, color=(0.0, 0.6, 1.0, 0.7), width=0.015
      )

      # Actual angular velocity arrow (light green).
      act_ang_from = act_lin_from
      act_ang_to = local_to_world(
        (np.array([0, 0, z_offset]) + np.array([0, 0, ang_vel_b[2]])) * scale
      )
      visualizer.add_arrow(
        act_ang_from, act_ang_to, color=(0.0, 1.0, 0.4, 0.7), width=0.015
      )


@dataclass(kw_only=True)
class PiecewiseVelocityCommandCfg(CommandTermCfg):
  """Configuration for piecewise velocity command with weighted ranges."""

  class_type: type[CommandTerm] = PiecewiseVelocityCommand

  entity_name: str
  heading_command: bool = False
  heading_control_stiffness: float = 1.0
  rel_standing_envs: float = 0.0
  rel_heading_envs: float = 1.0
  rel_world_envs: float = 0.0

  @dataclass
  class Ranges:
    lin_vel_x_ranges: list[tuple[float, float]]
    lin_vel_x_weights: list[float]
    lin_vel_y_ranges: list[tuple[float, float]]
    lin_vel_y_weights: list[float]
    ang_vel_z_ranges: list[tuple[float, float]]
    ang_vel_z_weights: list[float]
    heading: tuple[float, float] | None = None

  ranges: Ranges

  @dataclass
  class VizCfg:
    z_offset: float = 0.2
    scale: float = 0.5

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> PiecewiseVelocityCommand:
    return PiecewiseVelocityCommand(self, env)

  def __post_init__(self):
    if self.heading_command and self.ranges.heading is None:
      raise ValueError(
        "The velocity command has heading commands active (heading_command=True) but "
        "the `ranges.heading` parameter is set to None."
      )
