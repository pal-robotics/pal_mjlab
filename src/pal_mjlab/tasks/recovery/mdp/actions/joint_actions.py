from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction

if TYPE_CHECKING:
    from mjlab.envs.manager_based_env import ManagerBasedEnv
    from mjlab.envs.mdp.actions import actions_config


class JointPositionSettledAction(JointPositionAction):
    def __init__(
        self, cfg: actions_config.JointPositionSettledActionCfg, env: ManagerBasedEnv
    ):
        super().__init__(cfg=cfg, env=env)

        # settle_time is in seconds, convert to steps
        settle_steps = int(cfg.settle_time / env.step_dt)
        settle_steps = max(settle_steps, 0)

        self._settle_steps = torch.full(
            (self.num_envs,),
            settle_steps,
            device=self.device,
            dtype=torch.long,
        )
        self._steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        processed_actions = self._raw_actions * self._scale + self._offset
        current_joint_pos = self._asset.data.joint_pos[:, self._joint_ids]

        # mask: True if env is already settled
        settled_mask = self._steps >= self._settle_steps
        settled_mask = settled_mask.view(-1, 1)

        self._processed_actions[:] = torch.where(
            settled_mask, processed_actions, current_joint_pos
        )

        self._steps += 1

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._steps[env_ids] = 0