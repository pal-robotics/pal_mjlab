from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions.joint_actions import JointPositionAction
from mjlab.envs.mdp.actions import actions_config

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass
class MirroredJointPositionActionCfg(actions_config.JointPositionActionCfg):
    """
    Mirror processed targets from source actuators to destination actuators.

    mirror_pairs: {dst_actuator_name: src_actuator_name}
      - dst receives the same target as src
    """
    mirror_pairs: dict[str, str] = field(default_factory=lambda: {
        "gripper_left_outer_finger_right_joint":  "gripper_left_outer_finger_left_joint",
        "gripper_right_outer_finger_right_joint": "gripper_right_outer_finger_left_joint",
    })


class MirroredJointPositionAction(JointPositionAction):
    def __init__(self, cfg: MirroredJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)

        # Map actuator/joint name -> index into _processed_actions columns
        src_index = {name: i for i, name in enumerate(self._joint_names)}

        dst_joint_ids: list[int] = []
        src_indices: list[int] = []

        for dst_name, src_name in cfg.mirror_pairs.items():
            try:
                src_indices.append(src_index[src_name])
            except KeyError as e:
                raise ValueError(
                    f"mirror_pairs src '{src_name}' must be in actuator_names. "
                    f"Resolved joints: {self._joint_names}"
                ) from e

            ids, names = self._asset.find_joints_by_actuator_names((dst_name,))
            if len(ids) != 1:
                raise ValueError(f"Could not uniquely resolve dst actuator/joint '{dst_name}'. Got: {names}")
            dst_joint_ids.append(ids[0])

        self._mirror_dst_joint_ids = torch.as_tensor(dst_joint_ids, device=self.device, dtype=torch.long)
        self._mirror_src_indices = torch.as_tensor(src_indices, device=self.device, dtype=torch.long)

    def apply_actions(self) -> None:
        # Apply policy-controlled joints
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

        # Mirror processed targets: src cols -> dst joints
        self._asset.set_joint_position_target(
            self._processed_actions[:, self._mirror_src_indices],
            joint_ids=self._mirror_dst_joint_ids,
        )


MirroredJointPositionActionCfg.class_type = MirroredJointPositionAction
