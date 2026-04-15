"""Custom domain randomization events for PAL tracking tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from mjlab.actuator import BuiltinPositionActuator, IdealPdActuator
from mjlab.actuator.xml_actuator import XmlActuator
from mjlab.entity import Entity
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.envs.mdp.dr._core import _DEFAULT_ASSET_CFG
from mjlab.envs.mdp.dr._types import resolve_distribution

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@requires_model_fields("actuator_delayprm")
def control_delay(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  delay_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform"] = "uniform",
) -> None:
  """Randomize actuator control delay (requires MuJoCo 3.0+).

  Args:
    env: The environment.
    env_ids: Environment IDs to randomize. If None, randomizes all.
    delay_range: (min, max) for delay randomization in seconds.
    asset_cfg: Asset configuration specifying which entity and actuators.
    distribution: Distribution type ("uniform" or "log_uniform").
  """
  asset: Entity = env.scene[asset_cfg.name]

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  if isinstance(asset_cfg.actuator_ids, list):
    actuators = [asset.actuators[i] for i in asset_cfg.actuator_ids]
  elif isinstance(asset_cfg.actuator_ids, slice):
    actuators = asset.actuators[asset_cfg.actuator_ids]
  else:
    actuators = [asset.actuators[asset_cfg.actuator_ids]]

  for actuator in actuators:
    ctrl_ids = actuator.global_ctrl_ids
    num_actuators = len(ctrl_ids)

    dist = resolve_distribution(distribution)
    delay_samples = dist.sample(
      torch.tensor(delay_range[0], device=env.device),
      torch.tensor(delay_range[1], device=env.device),
      (len(env_ids), num_actuators),
      env.device,
    )

    # In MuJoCo, delayprm[0] represents the delay time constant.
    env.sim.model.actuator_delayprm[env_ids[:, None], ctrl_ids, 0] = delay_samples


@requires_model_fields("actuator_gainprm", "actuator_biasprm")
def p_gain(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  kp_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: Literal["uniform", "log_uniform"] = "uniform",
) -> None:
  """Randomize actuator proportional (P) gain by scaling the nominal value.

  Samples a scale factor from ``kp_range`` and applies it to both
  ``actuator_gainprm[..., 0]`` (kp) and ``actuator_biasprm[..., 1]`` (-kp)
  so that the position-feedback terms remain consistent.

  Args:
    env: The environment.
    env_ids: Environment IDs to randomize. If None, randomizes all.
    kp_range: (min, max) scale factor applied to the nominal P gain,
      e.g. ``(0.925, 1.05)`` for ±~7.5 % variation.
    asset_cfg: Asset configuration specifying which entity and actuators.
    distribution: Distribution type ("uniform" or "log_uniform").
  """
  asset: Entity = env.scene[asset_cfg.name]

  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  else:
    env_ids = env_ids.to(env.device, dtype=torch.int)

  if isinstance(asset_cfg.actuator_ids, list):
    actuators = [asset.actuators[i] for i in asset_cfg.actuator_ids]
  elif isinstance(asset_cfg.actuator_ids, slice):
    actuators = asset.actuators[asset_cfg.actuator_ids]
  else:
    actuators = [asset.actuators[asset_cfg.actuator_ids]]

  for actuator in actuators:
    ctrl_ids = actuator.global_ctrl_ids

    dist = resolve_distribution(distribution)
    kp_scale = dist.sample(
      torch.tensor(kp_range[0], device=env.device),
      torch.tensor(kp_range[1], device=env.device),
      (len(env_ids), len(ctrl_ids)),
      env.device,
    )

    if isinstance(actuator, BuiltinPositionActuator) or (
      isinstance(actuator, XmlActuator) and actuator.command_field == "position"
    ):
      default_gainprm = env.sim.get_default_field("actuator_gainprm")
      default_biasprm = env.sim.get_default_field("actuator_biasprm")
      env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 0] = (
        default_gainprm[ctrl_ids, 0] * kp_scale
      )
      env.sim.model.actuator_biasprm[env_ids[:, None], ctrl_ids, 1] = (
        default_biasprm[ctrl_ids, 1] * kp_scale
      )

    elif isinstance(actuator, IdealPdActuator):
      assert actuator.stiffness is not None
      assert actuator.default_stiffness is not None
      actuator.set_gains(
        env_ids,
        kp=actuator.default_stiffness[env_ids] * kp_scale,
        kd=None,
      )

    else:
      raise TypeError(
        f"p_gain only supports BuiltinPositionActuator, "
        f"XmlActuator (position), and IdealPdActuator, "
        f"got {type(actuator).__name__}"
      )
