from dataclasses import dataclass

from pal_mjlab.tasks.reaching_tiago.mdp.actions import joint_actions
from mjlab.managers.action_manager import ActionTerm
from mjlab.managers.manager_term_config import ActionTermCfg


@dataclass(kw_only=True)
class BinaryJointActionCfg(ActionTermCfg):
  actuator_names: list[str]
  """List of actuator names or regex expressions that the action will be mapped to."""
  open_command_expr: dict[str, float]
  """Dictionary of open command expressions for the binary joint action."""
  close_command_expr: dict[str, float]
  """Dictionary of close command expressions for the binary joint action."""


@dataclass(kw_only=True)
class BinaryJointPositionActionCfg(BinaryJointActionCfg):
  class_type: type[ActionTerm] = joint_actions.BinaryJointPositionAction
  use_default_offset: bool = True