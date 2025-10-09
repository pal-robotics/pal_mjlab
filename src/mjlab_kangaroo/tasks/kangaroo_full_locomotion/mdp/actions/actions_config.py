from dataclasses import dataclass

from mjlab.envs.mdp.actions.actions_config import JointActionCfg
from mjlab.managers.action_manager import ActionTerm

from mjlab_kangaroo.tasks.kangaroo_full_locomotion.mdp.actions import (
    joint_actions as custom_joint_actions,
)


@dataclass(kw_only=True)
class JointPositionToLimitsCfg(JointActionCfg):
    class_type: type[ActionTerm] = (
        custom_joint_actions.JointPositionToLimitsAction
    )
    rescale_to_limits: bool = True
    """Whether to rescale the action to the joint limits. Defaults to True.

    If True, the input actions are rescaled to the joint limits, i.e., the action value in
    the range [-1, 1] corresponds to the joint lower and upper limits respectively.

    Note:
        This operation is performed after applying the scale factor.
    """
    use_tanh: bool = False
    """Whether to use tanh to scale the actions in the range [-1, 1]. Defaults to False."""
