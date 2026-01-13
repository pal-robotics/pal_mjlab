# from dataclasses import dataclass

# from pal_mjlab.tasks.lift.mdp.actions import joint_actions
# from pal_mjlab.tasks.lift.mdp.actions import mirrored_joint_position_action
# from mjlab.managers.action_manager import ActionTerm
# from mjlab.managers.manager_term_config import ActionTermCfg


# @dataclass(kw_only=True)
# class BinaryJointActionCfg(ActionTermCfg):
#   actuator_names: list[str]
#   """List of actuator names or regex expressions that the action will be mapped to."""
#   open_command_expr: dict[str, float]
#   """Dictionary of open command expressions for the binary joint action."""
#   close_command_expr: dict[str, float]
#   """Dictionary of close command expressions for the binary joint action."""


# @dataclass(kw_only=True)
# class BinaryJointPositionActionCfg(BinaryJointActionCfg):
#   class_type: type[ActionTerm] = joint_actions.BinaryJointPositionAction
#   use_default_offset: bool = True


# @dataclass(kw_only=True)
# class MirroredJointPositionActionCfg(ActionTermCfg):
#     asset_name: str
#     """Name of the asset in the scene (e.g. 'robot')."""

#     # SOURCE (policy sees only these)
#     actuator_names: list[str]
#     """Actuators/joints controlled by the policy (action_dim = len(source joints))."""

#     # # DESTINATION(mirrored targets, not in policy action space)
#     mirror_actuator_names: list[str]
#     """Actuators/joints that receive mirrored targets."""

#     mirror_pairs: dict[str, str]
#     """
#     Mapping dst_actuator_name -> src_actuator_name.
#     Order must correspond to mirror_actuator_names.
#     """

#     mirror_sign: list[float] | None = None

#     use_default_offset: bool = True
#     scale: float = 1.0
#     offset: float = 0.0

#     class_type: type[ActionTerm] = mirrored_joint_position_action.MirroredJointPositionAction
