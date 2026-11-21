"""ONNX export of the SuperMimic policy for the robot: one file per policy.

The ONNX is the actor (observation normalizer, goal encoder, decoder): input ``obs``
(1, obs_dim), output ``actions`` (1, num_actions). The joint targets are
``action_offset + action_scale * actions`` on ``joint_names`` (metadata), tracked by PD
gains ``joint_stiffness`` / ``joint_damping``. The robot builds ``obs`` as the env does,
with no noise: the actor groups' terms in ``observation_names`` order, each the last
``observation_terms_history_length`` control steps, oldest first (at the start every slot
holds the first observation). Lists of the same length describe each term:
``observation_terms_dim`` and ``observation_sites`` (the frame of a site term, such as
the IMU's, else empty; ``imu_site`` too when there is exactly one).

The generic proprioception terms (``terms.py``): ``joint_pos_rel`` is q - default_joint_pos
of ``joint_names`` (plus the encoder offsets the robot's encoders already include),
``joint_vel_rel`` q̇, ``site_ang_vel`` / ``site_projected_gravity`` the angular velocity
and gravity in the site's frame, ``last_action`` the previous ``actions`` (zeros at the
start). What the goal holds is the command's to say: its command term may add metadata
(``export_metadata()``, e.g. ``MotionLibraryCommand``'s).
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.rl.exporter_utils import attach_metadata_to_onnx

from .imagination import batch_source
from .models import EncoderDecoderModel
from .names import ImaginationCfg, imagination_cfg

_AFFINE = 1  # mjtBias.mjBIAS_AFFINE: a position actuator's bias


def policy_metadata(
  env: ManagerBasedRlEnv, checkpoint: str = "", names: ImaginationCfg | None = None
) -> dict:
  """Everything the robot needs to build ``obs`` and apply ``actions``: the generic
  metadata, then the command term's ``export_metadata()`` if it has one."""
  names = names or imagination_cfg(env)
  spec = batch_source(env).spec
  robot = env.scene[spec.entity_name]
  action = cast(JointPositionAction, env.action_manager.get_term(names.action_term))
  om = env.observation_manager
  obs_names, dims, history, sites = [], [], [], []
  for group in names.actor_groups:
    for name, dim in zip(
      om.active_terms[group], om.group_obs_term_dim[group], strict=True
    ):
      cfg = om.get_term_cfg(group, name)
      assert cfg.scale is None and cfg.clip is None and cfg.delay_max_lag == 0, name
      steps = max(cfg.history_length, 1)
      obs_names.append(name)
      dims.append(dim[-1] // steps)
      history.append(steps)
      sites.append(str(cfg.params.get("site", "")))
  # The policy's joints: the action joints, which are the state's (and the measured ones).
  joint_names = list(spec.joint_names)
  ctrl = {a.target.split("/")[-1]: a.id for a in robot.spec.actuators}
  ids = [ctrl[j] for j in joint_names]
  model = env.sim.mj_model
  kp, kd = model.actuator_gainprm[ids, 0], -model.actuator_biasprm[ids, 2]
  position = (model.actuator_biastype[ids] == _AFFINE) & np.isclose(
    model.actuator_biasprm[ids, 1], -kp
  )
  if not position.all():
    bad = [j for j, ok in zip(joint_names, position, strict=True) if not ok]
    raise ValueError(
      f"Joints {bad} are not driven by MuJoCo position actuators: their PD gains cannot "
      "be read from the model for the export."
    )
  meta = {
    "checkpoint": checkpoint,
    "control_dt": env.step_dt,
    "joint_names": joint_names,
    "default_joint_pos": spec.q0.tolist(),
    "action_scale": _per_action(action.scale, action.action_dim),
    "action_offset": _per_action(action.offset, action.action_dim),
    "joint_stiffness": kp.tolist(),
    "joint_damping": kd.tolist(),
    "observation_names": obs_names,
    "observation_terms_dim": dims,
    "observation_terms_history_length": history,
    "observation_sites": sites,
  }
  if len({s for s in sites if s}) == 1:  # one IMU: kept under its earlier key
    meta["imu_site"] = next(s for s in sites if s)
  command = env.command_manager.get_term(names.command)
  extra = getattr(command, "export_metadata", None)
  if callable(extra):
    added = extra()
    clash = sorted(set(added) & set(meta))
    if clash:
      raise ValueError(f"The command's export metadata redefines {clash}.")
    meta.update(added)
  return meta


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
  names: ImaginationCfg | None = None,
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
  attach_metadata_to_onnx(str(path), policy_metadata(env, checkpoint, names))