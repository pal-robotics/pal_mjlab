from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch

from mjlab.envs.mdp.actions.joint_actions import JointPositionAction
from mjlab.envs.mdp.actions import actions_config  # same module your peer snippet uses

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass
class MirroredJointPositionActionCfg(actions_config.JointPositionActionCfg):
    """
    Same as JointPositionActionCfg, but also mirrors processed targets
    from source joints to destination joints at apply_actions() time.

    mirror_pairs: (dst_actuator_name, src_actuator_name)
      - dst receives the same target as src
    """
    mirror_pairs: Sequence[tuple[str, str]] = (
        ("gripper_left_outer_finger_right_joint",  "gripper_left_outer_finger_left_joint"),
        ("gripper_right_outer_finger_right_joint", "gripper_right_outer_finger_left_joint"),
    )


class MirroredJointPositionAction(JointPositionAction):
    def __init__(self, cfg: MirroredJointPositionActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)

        # Build dst joint ids and src indices into self._joint_names
        name_to_src_index = {n: i for i, n in enumerate(self._joint_names)}

        dst_joint_ids: list[int] = []
        src_indices: list[int] = []

        for dst_act_name, src_act_name in cfg.mirror_pairs:
            if src_act_name not in name_to_src_index:
                raise ValueError(
                    f"mirror_pairs src '{src_act_name}' must be in actuator_names. "
                    f"Got actuator_names resolved to joints: {self._joint_names}"
                )

            dst_ids, dst_names = self._asset.find_joints_by_actuator_names((dst_act_name,))
            if len(dst_ids) != 1:
                raise ValueError(f"Could not uniquely resolve dst actuator/joint '{dst_act_name}'. Got: {dst_names}")

            dst_joint_ids.append(dst_ids[0])
            src_indices.append(name_to_src_index[src_act_name])

        self._mirror_dst_joint_ids = torch.tensor(dst_joint_ids, device=self.device, dtype=torch.long)
        self._mirror_src_indices = torch.tensor(src_indices, device=self.device, dtype=torch.long)

    def apply_actions(self) -> None:
        # 1) Apply the policy-controlled joints (SOURCE)
        self._asset.set_joint_position_target(
            self._processed_actions, joint_ids=self._joint_ids
        )

        # 2) Mirror: copy processed target from src joints -> dst joints
        mirrored_targets = self._processed_actions[:, self._mirror_src_indices]
        self._asset.set_joint_position_target(
            mirrored_targets, joint_ids=self._mirror_dst_joint_ids
        )

MirroredJointPositionActionCfg.class_type = MirroredJointPositionAction
