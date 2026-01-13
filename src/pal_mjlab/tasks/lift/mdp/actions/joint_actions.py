# from __future__ import annotations

# from typing import TYPE_CHECKING

# import torch

# from mjlab.entity import Entity
# from mjlab.managers.action_manager import ActionTerm
# from mjlab.utils.lab_api.string import resolve_matching_names_values

# if TYPE_CHECKING:
#     from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
#     from mjlab.envs.mdp.actions import actions_config


# class BinaryJointAction(ActionTerm):
#     """Base class for binary joint actions.

#     One scalar action per environment:
#       - bool: False/0 -> open, True/1 -> close
#       - float: < 0 -> open, >= 0 -> close

#     Internally this selects between predefined open/close joint commands.
#     """

#     _asset: Entity

#     def __init__(self, cfg: actions_config.BinaryJointActionCfg, env: ManagerBasedRlEnv):
#         super().__init__(cfg=cfg, env=env)

#         joint_ids, self._joint_names = self._asset.find_joints(cfg.actuator_names)
#         self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)

#         self._num_joints = len(self._joint_ids)

#         self._action_dim = 1
#         self._raw_actions = torch.zeros(self.num_envs, self._action_dim, device=self.device)
#         self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

#         self._open_command = torch.zeros(self.num_envs, self._num_joints, device=self.device)
#         self._close_command = torch.zeros(self.num_envs, self._num_joints, device=self.device)

#         idx_list, _, value_list = resolve_matching_names_values(
#             cfg.open_command_expr, self._joint_names
#         )
#         self._open_command[:, idx_list] = torch.as_tensor(value_list, device=self.device)

#         idx_list, _, value_list = resolve_matching_names_values(
#             cfg.close_command_expr, self._joint_names
#         )
#         self._close_command[:, idx_list] = torch.as_tensor(value_list, device=self.device)


#     @property
#     def open_command(self) -> torch.Tensor:
#         """Per-env, per-joint open command."""
#         return self._open_command

#     @property
#     def close_command(self) -> torch.Tensor:
#         """Per-env, per-joint close command."""
#         return self._close_command

#     @property
#     def raw_action(self) -> torch.Tensor:
#         """Last raw binary actions from the policy (num_envs, 1)."""
#         return self._raw_actions

#     @property
#     def action_dim(self) -> int:
#         """One scalar per env for this binary action."""
#         return self._action_dim


#     def process_actions(self, actions: torch.Tensor):
#         """Map scalar actions to per-joint open/close commands.

#         actions: (num_envs,) or (num_envs, 1)
#         """
  
#         actions = actions.view(self.num_envs, self._action_dim)
#         self._raw_actions[:] = actions

#         if actions.dtype == torch.bool:
#             binary_mask = (actions == 0)
#         else:
#             binary_mask = (actions < 0.0)

#         self._processed_actions = torch.where(
#             binary_mask,
#             self._open_command,
#             self._close_command,
#         )

#     def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
#         if env_ids is None:
#             env_ids = slice(None)
#         self._raw_actions[env_ids] = 0.0


# class BinaryJointPositionAction(BinaryJointAction):
#     """Binary joint position control: writes directly into joint positions."""

#     def __init__(
#         self,
#         cfg: actions_config.BinaryJointPositionActionCfg,
#         env: ManagerBasedRlEnv,
#     ):
#         super().__init__(cfg=cfg, env=env)

#         if cfg.use_default_offset:
#             self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

#     def apply_actions(self):
#         """Apply processed open/close commands as joint positions."""
#         self._asset.write_joint_position_to_sim(
#             self._processed_actions,
#             joint_ids=self._joint_ids,
#         )
