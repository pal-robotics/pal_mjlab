from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions.joint_actions import JointAction

if TYPE_CHECKING:
    from mjlab.envs.manager_based_env import ManagerBasedEnv
    from pal_mjlab.tasks.kangaroo_full_locomotion.mdp.actions import (
        actions_config as custom_actions_cfg,
    )
from mjlab.third_party.isaaclab.isaaclab.utils.math import unscale_transform


class JointPositionToLimitsAction(JointAction):
    """Joint position action term that scales the input actions to the joint limits and applies them to the
    articulation's joints.

    This class is similar to the :class:`JointPositionAction` class. However, it performs additional
    re-scaling of input actions to the actuator joint position limits.

    While processing the actions, it performs the following operations:

    1. Apply scaling to the raw actions based on :attr:`actions_cfg.JointPositionToLimitsActionCfg.scale`.
    2. Clip (or squash via tanh) the scaled actions to the range [-1, 1] and re-scale them to the joint
       limits if :attr:`actions_cfg.JointPositionToLimitsActionCfg.rescale_to_limits` is set to True.
       :attr:`actions_cfg.JointPositionToLimitsActionCfg.pre_rescale_offset` can be added to the actions before
       clipping, through which it maps raw_actions' zeros to values different than the joint limits's mean.

    The processed actions are then sent as position commands to the articulation's joints.
    """

    def __init__(
        self,
        cfg: custom_actions_cfg.JointPositionToLimitsCfg,
        env: ManagerBasedEnv,
    ):
        super().__init__(cfg=cfg, env=env)

        if cfg.rescale_to_limits:
            self._rescale_to_limits = True
        else:
            self._rescale_to_limits = False

        if cfg.use_tanh:
            self._use_tanh = True
        else:
            self._use_tanh = False

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions * self._scale + self._offset
        if self.cfg.rescale_to_limits:
            if not self._use_tanh:  # clip to [-1, 1]
                actions = (self._processed_actions).clamp(-1.0, 1.0)
            else:  # squash with tanh
                actions = torch.tanh(self._processed_actions)
            # rescale within the joint limits
            actions = unscale_transform(
                actions,
                self._asset.data.soft_joint_pos_limits[:, self._actuator_ids, 0],
                self._asset.data.soft_joint_pos_limits[:, self._actuator_ids, 1],
            )
            self._processed_actions[:] = actions[:]

    def apply_actions(self):
        self._asset.write_joint_position_target_to_sim(
            self._processed_actions, self._actuator_ids
        )
