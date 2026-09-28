"""Useful methods for recorders."""

from __future__ import annotations

from typing import TYPE_CHECKING
import csv

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.managers.recorder_manager import RecorderTerm

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class CsvRecorder(RecorderTerm):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self._file = open(cfg.params["path"], "w", newline="")
    self._writer = csv.writer(self._file)
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    assert asset_cfg is not None, "Pass asset_cfg param to CsvRecorder term"
    self.asset: Entity = env.scene[asset_cfg.name]

    joint_names = cfg.params["joint_names"]
    self._joint_ids = torch.as_tensor(
      self.asset.find_joints(joint_names, preserve_order=True)[0],
      device=env.device,
      dtype=torch.long,
    )

  def record_post_step(self):
    # Skip envs that just reset: their terminal pair was written in record_pre_reset
    # and their action is now zeroed.
    mask = ~self._env.reset_buf

    root_pos = self.asset.data.root_link_pos_w[mask].clone()
    root_pos[:, :2] -= self._env.scene.env_origins[mask][:, :2]

    # wxyz -> xyzw
    root_ori = self.asset.data.root_link_quat_w[mask][:, [1, 2, 3, 0]]

    # Reorder joints to match the CSV convention.
    joint_pos = self.asset.data.joint_pos[mask][:, self._joint_ids]

    root_pos = root_pos.cpu().numpy()
    root_ori = root_ori.cpu().numpy()
    joint_pos = joint_pos.cpu().numpy()

    for rp, ro, jp in zip(root_pos, root_ori, joint_pos):
      self._writer.writerow(rp.tolist() + ro.tolist() + jp.tolist())

  def close(self):
    self._file.close()