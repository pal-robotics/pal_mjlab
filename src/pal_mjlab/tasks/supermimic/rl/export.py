"""ONNX export of the SuperMimic policy for the robot: one file per policy, any motion.

The ONNX is the actor (observation normalizer, goal encoder, decoder): input ``obs``
(1, obs_dim), output ``actions`` (1, num_actions). The joint targets are
``action_offset + action_scale * actions`` on ``joint_names`` (metadata). The robot builds
``obs`` as the env does, with no noise:

  obs = [goal, proprio]
  goal     ``state.reference_goal`` of the next K reference frames: per frame, the reference
           positions and velocities of ``joint_names`` and the reference root orientation in
           the robot root frame (rot6d: the first two columns of its rotation matrix)
  proprio  per term in ``observation_names`` order, the last ``history_length`` control steps,
           oldest first; at the start every slot holds the first observation:
           joint_pos  q - default_joint_pos of ``joint_names``
           joint_vel  q̇ of the same joints
           imu_ang_vel, imu_projected_gravity  in the frame of ``imu_site``
           actions    the previous ``actions`` output (zeros at the start)

Training episodes start at the reference pose, so the reference's root yaw at its first frame
must match the robot's: rotate the motion about z before building goals (a yaw offset reads as
a tracking error).
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.rl.exporter_utils import attach_metadata_to_onnx

from ..mdp.commands import MotionLibraryCommand
from .models import EncoderDecoderModel

OBS_GROUPS = ("goal", "proprio")  # the actor's input, in this order


def policy_metadata(env: ManagerBasedRlEnv, checkpoint: str = "") -> dict:
  """Everything the robot needs to build ``obs`` and apply ``actions``."""
  robot = env.scene["robot"]
  action = cast(JointPositionAction, env.action_manager.get_term("joint_pos"))
  cmd = cast(MotionLibraryCommand, env.command_manager.get_term("motion"))
  om = env.observation_manager
  names, dims, history = [], [], []
  for group in OBS_GROUPS:
    for name, dim in zip(
      om.active_terms[group], om.group_obs_term_dim[group], strict=True
    ):
      cfg = om.get_term_cfg(group, name)
      assert cfg.scale is None and cfg.clip is None and cfg.delay_max_lag == 0, name
      steps = max(cfg.history_length, 1)
      names.append(name)
      dims.append(dim[-1] // steps)
      history.append(steps)
  imu_sites = {  # scene names are "<entity>/<name>"
    robot.spec.sensor(
      om.get_term_cfg("proprio", t).params["sensor_name"].split("/")[-1]
    ).objname.split("/")[-1]
    for t in ("imu_ang_vel", "imu_projected_gravity")
  }
  assert len(imu_sites) == 1, imu_sites
  joint_names = list(cmd.library.joint_names)
  # The policy's joints are the library's: measured, commanded and in the goal.
  assert list(action.target_names) == joint_names, (
    "actions must drive the library's joints"
  )
  ctrl = {a.target.split("/")[-1]: a.id for a in robot.spec.actuators}
  ids = [ctrl[j] for j in joint_names]
  model = env.sim.mj_model
  return {
    "checkpoint": checkpoint,
    "control_dt": env.step_dt,
    "joint_names": joint_names,
    "default_joint_pos": robot.data.default_joint_pos[0, cmd.joint_ids].tolist(),
    "action_scale": _per_action(action.scale, action.action_dim),
    "action_offset": _per_action(action.offset, action.action_dim),
    "joint_stiffness": model.actuator_gainprm[ids, 0].tolist(),
    "joint_damping": (-model.actuator_biasprm[ids, 2]).tolist(),
    "observation_names": names,
    "observation_terms_dim": dims,
    "observation_terms_history_length": history,
    "imu_site": imu_sites.pop(),
    "goal_num_future_frames": cmd.cfg.num_future_frames,
    "goal_root_body": robot.body_names[0],
  }


def _per_action(value: torch.Tensor | float, dim: int) -> list[float]:
  """An action term's scale or offset (a float, or one row per env) as a list per action."""
  if isinstance(value, torch.Tensor):
    return value[0].tolist()
  return [float(value)] * dim


def export_policy(
  env: ManagerBasedRlEnv,
  actor: EncoderDecoderModel,
  path: str | Path,
  checkpoint: str = "",
) -> None:
  """Write the actor to ``path`` as ONNX with ``policy_metadata``."""
  model = actor.as_onnx(verbose=False).to("cpu").eval()
  torch.onnx.export(
    model,
    model.get_dummy_inputs(),
    str(path),
    export_params=True,
    opset_version=18,
    input_names=model.input_names,
    output_names=model.output_names,
    dynamic_axes={},
    dynamo=False,
  )
  attach_metadata_to_onnx(str(path), policy_metadata(env, checkpoint))
