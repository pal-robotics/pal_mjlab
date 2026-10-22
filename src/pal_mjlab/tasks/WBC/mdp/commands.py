from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
import csv
import mujoco
import mujoco_warp as mjw
from scipy.spatial.transform import Rotation as R, Slerp

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
  quat_apply_inverse,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

from pal_mjlab.scripts.csv_to_npz import ROBOT_CONFIGS
from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import (
  KANGAROO_XML,
  KANGAROO_LOWER_BODY_XML,
  KANGAROO_HANDS_XML, 
  KANGAROO_GRIPPERS_XML
)

if TYPE_CHECKING:
  from collections.abc import Callable
  from typing import Any

  import viser

  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))

ROBOT_XML_DICT = {
  "Kangaroo" : KANGAROO_XML,
  "Kangaroo_lower_body" : KANGAROO_LOWER_BODY_XML,
  "Kangaroo_hands" : KANGAROO_HANDS_XML,
  "Kangaroo_grippers" : KANGAROO_GRIPPERS_XML,
}

class MotionLoader:
    """
    Load multiple motion NPZ files as one contiguous motion library.

    Each NPZ is treated as one trajectory/clip. The individual clips are
    concatenated along the time dimension, while segment_start_idx and
    segment_length keep track of where each trajectory lives.
    """

    def __init__(
        self,
        motion_dir: str,
        body_indexes: torch.Tensor,
        anchor_body_index: int,
        step_dt,
        device: str = "cpu",
    ) -> None:

        source = Path(motion_dir).expanduser().resolve()

        if not source.is_dir():
            raise FileNotFoundError(f"Motion directory not found: {source}")

        motion_paths = sorted(source.glob("*.npz"))

        if not motion_paths:
            raise FileNotFoundError(
                f"No .npz motion files found in {source}"
            )

        # Keep body indices on CPU while loading numpy arrays.
        body_idx = body_indexes.detach().cpu().numpy()

        # Store each clip temporarily as numpy arrays.
        joint_pos_list = []
        joint_vel_list = []
        body_pos_list = []
        body_quat_list = []
        body_lin_vel_list = []
        body_ang_vel_list = []

        segment_start = []
        segment_length = []
        segment_names = []

        current_start = 0

        time_step = []

        for path in motion_paths:

            with np.load(path) as data:

                joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
                joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)

                body_pos_w = np.asarray(
                    data["body_pos_w"][:, body_idx],
                    dtype=np.float32,
                )

                body_quat_w = np.asarray(
                    data["body_quat_w"][:, body_idx],
                    dtype=np.float32,
                )

                body_lin_vel_w = np.asarray(
                    data["body_lin_vel_w"][:, body_idx],
                    dtype=np.float32,
                )

                body_ang_vel_w = np.asarray(
                    data["body_ang_vel_w"][:, body_idx],
                    dtype=np.float32,
                )

            num_frames = joint_pos.shape[0]

            if num_frames == 0:
                continue

            # Save trajectory metadata.
            segment_start.append(current_start)
            segment_length.append(num_frames)
            segment_names.append(path.stem)

            # Add this clip to the library.
            joint_pos_list.append(joint_pos)
            joint_vel_list.append(joint_vel)
            body_pos_list.append(body_pos_w)
            body_quat_list.append(body_quat_w)
            body_lin_vel_list.append(body_lin_vel_w)
            body_ang_vel_list.append(body_ang_vel_w)

            current_start += num_frames

            time_step.append(int(joint_pos.shape[0]))

        if not joint_pos_list:
            raise ValueError(f"No valid motion frames found in {source}")

        # ------------------------------------------------------------
        # Concatenate all trajectories into one motion library.
        # ------------------------------------------------------------

        self.joint_pos = torch.as_tensor(
            np.concatenate(joint_pos_list, axis=0),
            dtype=torch.float32,
            device=device,
        )

        self.joint_vel = torch.as_tensor(
            np.concatenate(joint_vel_list, axis=0),
            dtype=torch.float32,
            device=device,
        )

        self.body_pos_w = torch.as_tensor(
            np.concatenate(body_pos_list, axis=0),
            dtype=torch.float32,
            device=device,
        )

        self.body_quat_w = torch.as_tensor(
            np.concatenate(body_quat_list, axis=0),
            dtype=torch.float32,
            device=device,
        )

        self.body_lin_vel_w = torch.as_tensor(
            np.concatenate(body_lin_vel_list, axis=0),
            dtype=torch.float32,
            device=device,
        )

        self.body_ang_vel_w = torch.as_tensor(
            np.concatenate(body_ang_vel_list, axis=0),
            dtype=torch.float32,
            device=device,
        )

        # ------------------------------------------------------------
        # Trajectory / segment information.
        # ------------------------------------------------------------

        self.segment_start_idx = torch.tensor(
            segment_start,
            dtype=torch.long,
            device=device,
        )

        self.segment_length = torch.tensor(
            segment_length,
            dtype=torch.long,
            device=device,
        )

        self.segment_end_idx = (
            self.segment_start_idx + self.segment_length
        )

        self.num_trajectories = len(segment_start)

        self.segment_names = tuple(segment_names)

        self.time_step_total = max(tsl for tsl in time_step)

        self.trajectory_probs = (
          self.segment_length.float() / self.segment_length.sum().float()
        ).to(device=device)

        anchor_lin_vel = self.body_lin_vel_w[:, anchor_body_index]
        anchor_ang_vel = self.body_ang_vel_w[:, anchor_body_index]
        lin_acc = torch.zeros_like(anchor_lin_vel)
        ang_acc = torch.zeros_like(anchor_ang_vel)
        for s, e in zip(self.segment_start_idx.tolist(), self.segment_end_idx.tolist()):
          if e - s >= 2:
            lin_acc[s:e] = torch.gradient(anchor_lin_vel[s:e], spacing=step_dt, dim=0)[0]
            ang_acc[s:e] = torch.gradient(anchor_ang_vel[s:e], spacing=step_dt, dim=0)[0]
        self.anchor_lin_acc_w, self.anchor_ang_acc_w = lin_acc, ang_acc

class MotionLoaderCsv:
  """
  Load multiple motion CSV files as one contiguous motion library.

  Each CSV is treated as one trajectory/clip. The individual clips are
  concatenated along the time dimension, while segment_start_idx and
  segment_length keep track of where each trajectory lives.
  """

  def __init__(
    self,
    motion_dir: str,
    input_fps: float = 30.0,
    output_fps: float = 50.0, 
    device: str = "cpu",
) -> None:

    in_dt, out_dt = 1.0 / input_fps, 1.0 / output_fps

    source = Path(motion_dir).expanduser().resolve()

    if not source.is_dir():
      raise FileNotFoundError(f"Motion directory not found: {source}")

    motion_paths = sorted(source.glob("*.csv"))

    if not motion_paths:
      raise FileNotFoundError(
          f"No .csv motion files found in {source}"
      )

    # Store each clip temporarily as numpy arrays.
    root_pos_list = []
    root_lin_vel_list = []
    root_lin_acc_list = []
    root_quat_list = []
    root_ang_vel_list = []
    root_ang_acc_list = []
    joint_pos_list = []
    joint_vel_list = []

    segment_start = []
    segment_length = []
    segment_names = []

    current_start = 0

    time_step = []

    for path in motion_paths:
      with open(path, newline="", encoding="utf-8") as f:
        columns = list(zip(*csv.reader(f)))

      root_pos   = np.asarray(columns[0:3],  dtype=np.float32)   # (3, N)
      root_quat  = np.asarray(columns[3:7],  dtype=np.float32)   # (4, N), xyzw
      joint_pos  = np.asarray(columns[7:],   dtype=np.float32)   # (J, N)

      n_in = root_pos.shape[1]
      t_in = np.arange(n_in) * in_dt
      duration = t_in[-1]
      t_out = np.arange(0, duration, out_dt)                      # same grid as the script

      # Linear interpolation for positions and joints
      def lerp(x):
        return np.stack([np.interp(t_out, t_in, row) for row in x])

      root_pos_o  = lerp(root_pos)                                # (3, M)
      joint_pos_o = lerp(joint_pos)                               # (J, M)

      # Slerp for the quaternion (scipy uses xyzw, matching the CSV)
      slerp = Slerp(t_in, R.from_quat(root_quat.T))
      r = slerp(t_out)                                            # Rotation of length M
      root_quat_o = r.as_quat().T                                 # (4, M), xyzw

      # Velocities at the output rate
      root_lin_vel = np.gradient(root_pos_o,  out_dt, axis=1)
      joint_vel    = np.gradient(joint_pos_o, out_dt, axis=1)

      # World-frame angular velocity (central difference, padded at the ends)
      w = (r[2:] * r[:-2].inv()).as_rotvec() / (2 * out_dt)       # (M-2, 3)
      root_ang_vel = np.concatenate([w[:1], w, w[-1:]]).T         # (3, M)

      # Linear acceleration
      root_lin_acc = np.gradient(root_lin_vel, out_dt, axis=1)

      # Angular acceleration (world frame)
      root_ang_acc = np.gradient(root_ang_vel, out_dt, axis=1)


      # Sanity check to exclude explosive values in first/last frames
      _data = (root_pos_o, root_lin_vel, root_lin_acc, root_quat_o, root_ang_vel, root_ang_acc, joint_pos_o, joint_vel)

      _data_checked = self.sanity_check(_data)

      root_pos_o = _data_checked[0]
      root_lin_vel = _data_checked[0]
      root_lin_acc = _data_checked[0]

      root_quat_o = _data_checked[0]
      root_ang_vel = _data_checked[0]
      root_ang_acc = _data_checked[0]

      joint_pos_o = _data_checked[0]
      joint_vel = _data_checked[0]

      num_frames = joint_pos_o.shape[0]

      if num_frames == 0:
        continue

      # Save trajectory metadata.
      segment_start.append(current_start)
      segment_length.append(num_frames)
      segment_names.append(path.stem)

      # Add this clip to the library.

      root_pos_list.append(root_pos_o)
      root_lin_vel_list.append(root_lin_vel)
      root_lin_acc_list.append(root_lin_acc)
      root_quat_list.append(root_quat_o)
      root_ang_vel_list.append(root_ang_vel)
      root_ang_acc_list.append(root_ang_acc)
      joint_pos_list.append(joint_pos_o)
      joint_vel_list.append(joint_vel)

      current_start += num_frames

      time_step.append(int(joint_pos.shape[0]))

    if not joint_pos_list:
      raise ValueError(f"No valid motion frames found in {source}")

    # ------------------------------------------------------------
    # Concatenate all trajectories into one motion library.
    # ------------------------------------------------------------

    self.root_pos = torch.as_tensor(
      np.concatenate(root_pos_list, axis=0),
      dtype=torch.float32,
      device=device,
    )
    self.root_lin_vel = torch.as_tensor(
      np.concatenate(root_lin_vel_list, axis=0),
      dtype=torch.float32,
      device=device,
    )
    self.root_lin_acc = torch.as_tensor(
      np.concatenate(root_lin_acc_list, axis=0),
      dtype=torch.float32,
      device=device,
    )

    self.root_quat = torch.as_tensor(
      np.concatenate(root_quat_list, axis=0),
      dtype=torch.float32,
      device=device,
    )
    self.root_ang_vel = torch.as_tensor(
      np.concatenate(root_ang_vel_list, axis=0),
      dtype=torch.float32,
      device=device,
    )
    self.root_ang_acc = torch.as_tensor(
      np.concatenate(root_ang_acc_list, axis=0),
      dtype=torch.float32,
      device=device,
    )

    self.joint_pos = torch.as_tensor(
      np.concatenate(joint_pos_list, axis=0),
      dtype=torch.float32,
      device=device,
    )
    self.joint_vel = torch.as_tensor(
      np.concatenate(joint_vel_list, axis=0),
      dtype=torch.float32,
      device=device,
    )

    # ------------------------------------------------------------
    # Trajectory / segment information.
    # ------------------------------------------------------------

    self.segment_start_idx = torch.tensor(
        segment_start,
        dtype=torch.long,
        device=device,
    )

    self.segment_length = torch.tensor(
        segment_length,
        dtype=torch.long,
        device=device,
    )

    self.segment_end_idx = (
        self.segment_start_idx + self.segment_length
    )

    self.num_trajectories = len(segment_start)

    self.segment_names = tuple(segment_names)

    self.time_step_total = max(tsl for tsl in time_step)

    self.trajectory_probs = (
      self.segment_length.float() / self.segment_length.sum().float()
    ).to(device=device)

  def sanity_check(data):
    pass

class OnlineCsvToNpz :
  def __init__(
    self, robot_type: str,
    motion_dir: str,
    input_fps: float = 30.0,
    output_fps: float = 50.0, 
    device: str = "cpu",
  ):

    xml_path = ROBOT_XML_DICT[robot_type]

    self.mjm = mujoco.MjModel.from_xml_path(xml_path)
    self.mjd = mujoco.MjData(self.mjm)
    self.m = mjw.put_model(self.mjm)
    self.d = mjw.put_data(self.mjm, self.mjd, nworld=1)

    self.motion_csv = MotionLoaderCsv(
      motion_dir=motion_dir,
      input_fps=input_fps,
      output_fps=output_fps,
      device=device
    )

  def retarget(self, t : torch.Tensor, ):    
    qpos = np.concatenate([self.motion_csv.root_pos[:, t], self.motion_csv.root_quat[:, t], self.motion_csv.joint_pos[:, t]])

    rot = R.from_quat(self.motion_csv.root_quat[:, t])
    
    w_local = rot.inv().apply(self.motion_csv.root_ang_vel[:, t])
    qvel = np.concatenate([self.motion_csv.root_lin_vel[:, t], w_local, self.motion_csv.joint_vel[:, t]])

    self.d.qpos.assign(qpos[None])
    self.d.qvel.assign(qvel[None])
    mjw.forward(self.m, self.d)

    return self.d

class MotionCommand(CommandTerm):
  cfg: MotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    # if no body name passed, track all
    if self.cfg.tracked_body_names is None :
      self.cfg.tracked_body_names = self.cfg.body_names

    self.motion = MotionLoader(
      self.cfg.motion_dir, self.body_indexes, self.motion_anchor_body_index, env.step_dt, device=self.device
    )
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.fixed_time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    self._build_adaptive_bins()
    self.bin_failed_count = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self._current_bin_failed = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self.kernel = torch.tensor(
        [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
        device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    self._ghost_model = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)
    self._pending_forward = False

    self.rand_motion = torch.multinomial(
      self.motion.trajectory_probs, self.num_envs, replacement=True
    )

    self.moving_envs = torch.ones(self.num_envs, dtype=torch.bool ,device=self.device)

  def _build_adaptive_bins(self) -> None:
    """One bin per (clip, ~adaptive_bin_seconds chunk), indexed globally."""
    dev = self.device
    m = self.motion
    num_clips = m.num_trajectories
    fpb = max(int(round(self.cfg.adaptive_bin_seconds / self._env.step_dt)), 1)

    bins_per_clip = (m.segment_length + fpb - 1) // fpb             # (C,)
    clip_bin_offset = torch.cumsum(bins_per_clip, 0) - bins_per_clip  # (C,)
    total = int(bins_per_clip.sum())

    bin_clip = torch.repeat_interleave(
      torch.arange(num_clips, device=dev), bins_per_clip
    )                                                                 # (total,)
    local_bin = torch.arange(total, device=dev) - clip_bin_offset[bin_clip]
    bin_start = m.segment_start_idx[bin_clip] + local_bin * fpb
    bin_end = torch.minimum(bin_start + fpb, m.segment_end_idx[bin_clip])

    # Neighbour table for smoothing: replicate-padding applied per clip, so the
    # kernel never reaches into a different clip.
    k = self.cfg.adaptive_kernel_size
    pad = (k - 1) // 2
    first = clip_bin_offset[bin_clip]
    last = first + bins_per_clip[bin_clip] - 1
    idx = torch.arange(total, device=dev)[:, None] + (
      torch.arange(k, device=dev) - pad
    )[None, :]
    self._bin_neighbors = torch.clamp(idx, min=first[:, None], max=last[:, None])

    self._frames_per_bin = fpb
    self._bins_per_clip = bins_per_clip
    self._clip_bin_offset = clip_bin_offset
    self._bin_clip = bin_clip
    self._bin_start = bin_start
    self._bin_end = bin_end
    self.bin_count = total  # `_uniform_sampling` metrics still use this name

    # Length-proportional distribution over bins (matches the old clip lottery).
    uniform_p = (bin_end - bin_start).float()
    self._bin_uniform_p = uniform_p / uniform_p.sum()

  @property
  def ref_base_height(self) -> torch.Tensor:
    """Anchor height relative to env origin (z_I r̂_IB)."""
    return self.anchor_pos_w[:, 2:3] - self._env.scene.env_origins[:, 2:3]

  @property
  def ref_base_lin_vel_b(self) -> torch.Tensor:
    """Reference anchor linear velocity in anchor frame (B v̂_IB)."""
    return quat_apply_inverse(self.anchor_quat_w, self.anchor_lin_vel_w)*self.moving_envs.unsqueeze(-1)

  @property
  def ref_base_ang_vel_b(self) -> torch.Tensor:
    """Reference anchor angular velocity in anchor frame (B ω̂_IB)."""
    return quat_apply_inverse(self.anchor_quat_w, self.anchor_ang_vel_w)*self.moving_envs.unsqueeze(-1)

  @property
  def ref_gravity_b(self) -> torch.Tensor:
    """Reference gravity in anchor frame (B ĝ_I)."""
    return quat_apply_inverse(self.anchor_quat_w, self.robot.data.gravity_vec_w)

  @property
  def ref_base_lin_acc_b(self) -> torch.Tensor:
    return quat_apply_inverse(self.anchor_quat_w, self.anchor_lin_acc_w)*self.moving_envs.unsqueeze(-1)

  @property
  def ref_base_ang_acc_b(self) -> torch.Tensor:
    return quat_apply_inverse(self.anchor_quat_w, self.anchor_ang_acc_w)*self.moving_envs.unsqueeze(-1)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.time_steps]*self.moving_envs.unsqueeze(-1)

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps]*self.moving_envs[:, None, None]


  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps]*self.moving_envs[:, None, None]


  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]*self.moving_envs.unsqueeze(-1)

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]*self.moving_envs.unsqueeze(-1)

  @property
  def anchor_lin_acc_w(self) -> torch.Tensor:
    return self.motion.anchor_lin_acc_w[self.time_steps]*self.moving_envs.unsqueeze(-1)

  @property
  def anchor_ang_acc_w(self) -> torch.Tensor:
    return self.motion.anchor_ang_acc_w[self.time_steps]*self.moving_envs.unsqueeze(-1)

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  def _update_metrics(self):
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    num_envs = len(env_ids)

    episode_failed = (
      self._env.termination_manager.terminated[env_ids] & self.moving_envs[env_ids]
    )
    if torch.any(episode_failed):
      failed_env_ids = env_ids[episode_failed]
      motion_ids = self.rand_motion[failed_env_ids]

      local_frame = (
        self.time_steps[failed_env_ids] - self.motion.segment_start_idx[motion_ids]
      ).clamp_min(0)
      local_bin = torch.minimum(
        local_frame // self._frames_per_bin, self._bins_per_clip[motion_ids] - 1
      )
      fail_bins = self._clip_bin_offset[motion_ids] + local_bin

      self._current_bin_failed += torch.bincount(
        fail_bins, minlength=self.bin_count
      ).to(dtype=self._current_bin_failed.dtype)

    fail_p = self.bin_failed_count / self.bin_failed_count.sum().clamp_min(1e-12)
    if self.cfg.adaptive_kernel_size > 1:
      fail_p = (fail_p[self._bin_neighbors] * self.kernel).sum(dim=-1)
      fail_p = fail_p / fail_p.sum().clamp_min(1e-12)

    r = self.cfg.adaptive_uniform_ratio
    probs = (1 - r) * fail_p + r * self._bin_uniform_p

    # Optional: cap any bin at N times its uniform share, then renormalize.
    cap = self.cfg.adaptive_max_prob_ratio * self._bin_uniform_p
    probs = torch.minimum(probs, cap)
    probs = probs / probs.sum().clamp_min(1e-12)

    # Sample a global (clip, chunk) bin; the clip follows from the bin.
    sampled_bins = torch.multinomial(probs, num_envs, replacement=True)
    self.rand_motion[env_ids] = self._bin_clip[sampled_bins]

    start = self._bin_start[sampled_bins]
    length = (self._bin_end[sampled_bins] - start).clamp_min(1)
    offset = torch.minimum(
      (torch.rand(num_envs, device=self.device) * length).long(), length - 1
    )
    self.time_steps[env_ids] = start + offset

    H = -(probs * (probs + 1e-12).log()).sum()
    H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else 1.0
    pmax, imax = probs.max(dim=0)
    self.metrics["sampling_entropy"][env_ids] = H_norm
    self.metrics["sampling_top1_prob"][env_ids] = pmax
    self.metrics["sampling_top1_bin"][env_ids] = imax.float() / self.bin_count

  def _random_start_sampling(self, env_ids: torch.Tensor):
    self.rand_motion[env_ids] = torch.randint(0, self.motion.num_trajectories, (len(env_ids),), device=self.device)
    self.time_steps[env_ids] = self.motion.segment_start_idx[self.rand_motion[env_ids]]

  def _uniform_sampling(self, env_ids: torch.Tensor):
    num_envs = len(env_ids)
    
    rand_motion = torch.multinomial(
        self.motion.trajectory_probs, 
        num_envs, 
        replacement=True
    )
    self.rand_motion[env_ids] = rand_motion
    
    start_idxs = self.motion.segment_start_idx[rand_motion]
    end_idxs = self.motion.segment_end_idx[rand_motion]
    
    rand_float = torch.rand(num_envs, device=self.device)
    self.time_steps[env_ids] = (
        start_idxs + (rand_float * (end_idxs - start_idxs)).to(torch.long)
    )
    
    self.metrics["sampling_entropy"][env_ids] = 1.0  # Maximum entropy for uniform.
    self.metrics["sampling_top1_prob"][env_ids] = 1.0 / self.bin_count
    self.metrics["sampling_top1_bin"][env_ids] = 0.5  # No specific bin preference.

  def _sample_moving(self, env_ids: torch.Tensor) -> None:
    fixed = torch.rand(len(env_ids), device=self.device) < self.cfg.rel_fixed_command
    self.moving_envs[env_ids] = ~fixed
    self.fixed_time_steps[env_ids] = 0
    # Fixed envs hold the first frame of their clip.
    start = self.motion.segment_start_idx[self.rand_motion[env_ids]]
    self.time_steps[env_ids] = torch.where(fixed, start, self.time_steps[env_ids])

  def _write_reference_state_to_sim(
    self,
    env_ids: torch.Tensor,
    root_pos: torch.Tensor,
    root_ori: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
  ) -> None:
    """Clip joint positions and write root + joint state to sim."""
    soft_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos = torch.clip(joint_pos, soft_limits[:, :, 0], soft_limits[:, :, 1])
    self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    root_state = torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1)
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.reset(env_ids=env_ids)

  def _resample_command(self, env_ids: torch.Tensor):
    if self.cfg.sampling_mode == "start":
      self._random_start_sampling(env_ids)
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)
    self._sample_moving(env_ids)

    root_pos = self.body_pos_w[env_ids, 0].clone()
    root_ori = self.body_quat_w[env_ids, 0].clone()
    root_lin_vel = self.body_lin_vel_w[env_ids, 0].clone()
    root_ang_vel = self.body_ang_vel_w[env_ids, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori = quat_mul(orientations_delta, root_ori)
    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel += rand_samples[:, :3]
    root_ang_vel += rand_samples[:, 3:]

    joint_pos = self.joint_pos[env_ids].clone()
    joint_vel = self.joint_vel[env_ids]

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    )

    self._write_reference_state_to_sim(
      env_ids,
      root_pos,
      root_ori,
      root_lin_vel,
      root_ang_vel,
      joint_pos,
      joint_vel,
    )
    self._pending_forward = True

  def update_relative_body_poses(self) -> None:
    """Recompute ``body_pos_relative_w`` and ``body_quat_relative_w``.

    Called after ``reset_to_frame`` so that termination checks that
    compare relative body positions see the correct state.
    """
    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    extras = super().reset(env_ids)
    # Reset-path resamples are followed by the env's own forward(); only
    # compute-path resamples (wraparound or timer expiry) need our refresh.
    self._pending_forward = False
    return extras

  def _update_command(self, env_ids: torch.Tensor | None = None):
    if env_ids is None:
      self.time_steps += self.moving_envs
      self.fixed_time_steps += 1
    else:
      self.time_steps[env_ids] += self.moving_envs[env_ids]
      self.fixed_time_steps[env_ids] += 1
    motion_done = self.time_steps >= self.motion.segment_end_idx[self.rand_motion]
    fixed_done = (
      (self.fixed_time_steps >= self.cfg.max_timesteps_fixed)
      & ~self.moving_envs
    )
    wrap_ids = torch.nonzero(motion_done | fixed_done, as_tuple=False).squeeze(-1)

    if wrap_ids.numel() > 0:
      self._resample_command(wrap_ids)

    # _resample_command writes qpos/qvel but does not refresh derived
    # quantities; forward() so update_relative_body_poses reads the
    # post-teleport robot anchor instead of the stale pre-resample pose.
    # Covers both the wraparound above and a timer-expiry resample that ran
    # in compute before this call.
    if self._pending_forward:
      self._pending_forward = False
      self._env.sim.forward()

    self.update_relative_body_poses()

    # Fold failure counts into the EMA only on the per-step update so
    # manual resets do not decay it faster.
    if env_ids is None and self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw ghost robot or frames based on visualization mode."""
    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        # Build a ghost model with only visual geoms visible. Collision geoms (nonzero
        # contype/conaffinity) get alpha=0 so the viewer's alpha filter excludes them.
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        for gi in range(self._ghost_model.ngeom):
          if (
            self._ghost_model.geom_contype[gi] != 0
            or self._ghost_model.geom_conaffinity[gi] != 0
          ):
            self._ghost_model.geom_rgba[gi, 3] = 0
          else:
            self._ghost_model.geom_rgba[gi] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.entity_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      for batch in env_indices:
        qpos = np.zeros(self._env.sim.mj_model.nq)
        qpos[free_joint_q_adr[0:3]] = self.body_pos_w[batch, 0].cpu().numpy()
        qpos[free_joint_q_adr[3:7]] = self.body_quat_w[batch, 0].cpu().numpy()
        qpos[joint_q_adr] = self.joint_pos[batch].cpu().numpy()

        visualizer.add_ghost_mesh(
          qpos,
          model=self._ghost_model,
          label=f"ghost_{batch}",
        )

    elif self.cfg.viz.mode == "frames":
      for batch in env_indices:
        desired_body_pos = self.body_pos_w[batch].cpu().numpy()
        desired_body_quat = self.body_quat_w[batch]
        desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

        current_body_pos = self.robot_body_pos_w[batch].cpu().numpy()
        current_body_quat = self.robot_body_quat_w[batch]
        current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

        for i, body_name in enumerate(self.cfg.body_names):
          visualizer.add_frame(
            position=desired_body_pos[i],
            rotation_matrix=desired_body_rotm[i],
            scale=0.08,
            label=f"desired_{body_name}_{batch}",
            axis_colors=_DESIRED_FRAME_COLORS,
          )
          visualizer.add_frame(
            position=current_body_pos[i],
            rotation_matrix=current_body_rotm[i],
            scale=0.12,
            label=f"current_{body_name}_{batch}",
          )

        desired_anchor_pos = self.anchor_pos_w[batch].cpu().numpy()
        desired_anchor_quat = self.anchor_quat_w[batch]
        desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=desired_anchor_pos,
          rotation_matrix=desired_rotation_matrix,
          scale=0.1,
          label=f"desired_anchor_{batch}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )

        current_anchor_pos = self.robot_anchor_pos_w[batch].cpu().numpy()
        current_anchor_quat = self.robot_anchor_quat_w[batch]
        current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
        visualizer.add_frame(
          position=current_anchor_pos,
          rotation_matrix=current_rotation_matrix,
          scale=0.15,
          label=f"current_anchor_{batch}",
        )

  def create_gui(
    self,
    name: str,
    server: viser.ViserServer,
    get_env_idx: Callable[[], int],
    on_change: Callable[[], None] | None = None,
    request_action: Callable[[str, Any], None] | None = None,
  ) -> None:
    """Create motion scrubber controls in the Viser viewer."""
    max_frame = int(self.motion.time_step_total) - 1

    with server.gui.add_folder(name.capitalize()):
      scrubber = server.gui.add_slider(
        "Frame",
        min=0,
        max=max_frame,
        step=1,
        initial_value=0,
      )

      @scrubber.on_update
      def _(_) -> None:
        idx = get_env_idx()
        motion_id = self.rand_motion[idx]
        start = self.motion.segment_start_idx[motion_id]
        end = self.motion.segment_end_idx[motion_id]
        self.time_steps[idx] = torch.minimum(start + int(scrubber.value), end - 1)
        if on_change is not None:
          on_change()

      all_envs_cb = server.gui.add_checkbox("All envs", initial_value=True)
      start_btn = server.gui.add_button("Start Here")

      @start_btn.on_click
      def _(_) -> None:
        if request_action is not None:
          request_action(
            "CUSTOM",
            {"type": "gui_reset", "all_envs": all_envs_cb.value},
          )

    self._scrubber_handles = (scrubber, all_envs_cb, start_btn)
    self._set_scrubber_disabled(True)

  def _set_scrubber_disabled(self, disabled: bool) -> None:
    """Enable or disable the motion scrubber GUI controls."""
    for handle in self._scrubber_handles:
      handle.disabled = disabled

  def on_viewer_pause(self, paused: bool) -> None:
    if hasattr(self, "_scrubber_handles"):
      self._set_scrubber_disabled(not paused)

  def apply_gui_reset(self, env_ids: torch.Tensor) -> bool:
    if not hasattr(self, "_scrubber_handles"):
      return False
    frame = int(self._scrubber_handles[0].value)
    self.reset_to_frame(env_ids, frame)
    # reset_to_frame writes qpos/qvel; forward so update_relative_body_poses
    # reads the scrubbed pose instead of the stale pre-scrub kinematics.
    self._env.sim.forward()
    self.update_relative_body_poses()
    return True

  def reset_to_frame(self, env_ids: torch.Tensor, frame: int) -> None:
    self.moving_envs[env_ids] = True
    self.fixed_time_steps[env_ids] = 0
    start = self.motion.segment_start_idx[self.rand_motion[env_ids]]
    end = self.motion.segment_end_idx[self.rand_motion[env_ids]]
    self.time_steps[env_ids] = torch.clamp(start + frame, min=start, max=end - 1)
    self._write_reference_state_to_sim(
      env_ids,
      self.body_pos_w[env_ids, 0],
      self.body_quat_w[env_ids, 0],
      self.body_lin_vel_w[env_ids, 0],
      self.body_ang_vel_w[env_ids, 0],
      self.joint_pos[env_ids],
      self.joint_vel[env_ids],
    )


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
  motion_dir: str
  """Directory with one .npz per clip; all clips are loaded as one library."""

  anchor_body_name: str
  """Body used as the reference frame (e.g. base_link)."""

  body_names: tuple[str, ...]
  """Bodies loaded from the motion files; must include the anchor."""

  tracked_body_names: tuple[str, ...] | None = None
  """Bodies used for tracking terms. None = all body_names."""

  entity_name: str
  """Name of the robot entity in the scene."""

  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  """Root pose noise added at reset (keys: x, y, z, roll, pitch, yaw)."""

  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  """Root velocity noise added at reset (keys: x, y, z, roll, pitch, yaw)."""

  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  """Uniform noise (rad) added to the reference joint positions at reset."""

  # --- Adaptive sampling (only used when sampling_mode == "adaptive") ---

  adaptive_kernel_size: int = 3
  """Smoothing window (in bins, within a clip) applied to the failure mass.
  1 = no smoothing."""

  adaptive_lambda: float = 0.8
  """Kernel weights are [1, lambda, lambda^2, ...], normalized. With the current
  neighbor-table indexing, the bin right after a failed bin gets the largest share."""

  adaptive_alpha: float = 0.001
  """EMA rate for the failure statistics. Smaller = slower, smoother adaptation."""

  adaptive_uniform_ratio: float = 0.6
  """Fraction of resets drawn uniformly (length-proportional over all bins),
  ignoring failures. Failure-driven sampling gets the remaining (1 - ratio).
  0 = pure failure-driven, 1 = uniform."""

  adaptive_max_prob_ratio: float = 20.0
  """Cap on any bin's probability, as a multiple of its uniform share. Prevents a
  single (clip, chunk) from taking over. Set very large (e.g. 1e9) to disable."""

  adaptive_bin_seconds: float = 1.0
  """Bin (chunk) duration in seconds. Smaller = finer targeting but noisier
  failure statistics."""

  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"
  """'adaptive': failure-weighted bins mixed with uniform. 'uniform': random clip
  and frame. 'start': first frame of a random clip (used for play/eval)."""

  # --- Fixed (stand-still) commands ---

  rel_fixed_command: float = 0.0
  """Fraction of envs that hold the first frame of their clip with zeroed velocity
  references. 0 = disabled (the moving_envs path becomes a no-op)."""

  max_timesteps_fixed: int = 250
  """Steps a fixed env holds its pose before being resampled
  (250 steps = 5 s at dt 0.02)."""

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    """'ghost': translucent reference robot. 'frames': coordinate frames on bodies."""

    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)
    """RGBA color of the ghost robot."""

  viz: VizCfg = field(default_factory=VizCfg)
  """Debug visualization settings."""

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return MotionCommand(self, env)


class OnlineMotionCommand(MotionCommand) :
  
  def __init__(self, cfg: OnlineMotionCommandCfg, env: ManagerBasedRlEnv):
    super(MotionCommand, self).__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    # if no body name passed, track all
    if self.cfg.tracked_body_names is None :
      self.cfg.tracked_body_names = self.cfg.body_names

    self.motion = MotionLoader(
      self.cfg.motion_dir, self.body_indexes, self.motion_anchor_body_index, env.step_dt, device=self.device
    )
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.fixed_time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    self._build_adaptive_bins()
    self.bin_failed_count = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self._current_bin_failed = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self.kernel = torch.tensor(
        [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
        device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    self._ghost_model = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)
    self._pending_forward = False

    self.rand_motion = torch.multinomial(
      self.motion.trajectory_probs, self.num_envs, replacement=True
    )

    self.moving_envs = torch.ones(self.num_envs, dtype=torch.bool ,device=self.device)


@dataclass
class OnlineMotionCommandCfg(MotionCommandCfg) :
  a : 1