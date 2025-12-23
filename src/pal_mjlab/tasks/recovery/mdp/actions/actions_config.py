from dataclasses import dataclass

from pal_mjlab.tasks.recovery.mdp.actions import joint_actions
from mjlab.managers.action_manager import ActionTerm
from mjlab.envs.mdp.actions import JointPositionActionCfg


@dataclass(kw_only=True)
class JointPositionSettledActionCfg(JointPositionActionCfg):
    class_type: type[ActionTerm] = joint_actions.JointPositionSettledAction
    use_default_offset: bool = True
    settle_time: float = 20