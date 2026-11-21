"""SuperMimic's contract (``rl/supermimic/contracts.py``) on the Kangaroo motion task.

Every reward and observation term must give, on a simulator state rebuilt as imagination
sees a prediction, what the env computed, with the right command state; and the command
model must move and end the command as the env does. A new SuperMimic task gets the same
check by calling ``check_contract`` on its env.
"""

from pathlib import Path

import pytest
import torch
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp import observations as mj_obs
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from pal_mjlab.rl.supermimic.command_model import command_model_of
from pal_mjlab.rl.supermimic.contracts import check_contract
from pal_mjlab.tasks.supermimic.kangaroo.env_cfgs import (
  pal_kangaroo_grippers_supermimic_env_cfg,
)

LIBRARY = Path(__file__).parent / "fixtures" / "library_tiny"


def _env(extra_group: bool = False) -> ManagerBasedRlEnv:
  cfg = pal_kangaroo_grippers_supermimic_env_cfg()
  cfg.scene.num_envs = 4
  cfg.commands["motion"].library_path = str(LIBRARY)
  cfg.commands["motion"].compile_kinematics = False
  if extra_group:  # mjlab's own term: no single definition
    cfg.observations["extra"] = ObservationGroupCfg(
      terms={"mjlab_joint_pos": ObservationTermCfg(func=mj_obs.joint_pos_rel)},
      enable_corruption=False,
    )
  return ManagerBasedRlEnv(cfg=cfg, device=get_test_device())


TOLERANCE = {"reward/": 1e-4, "obs/": 1e-4, "command/": 0.0}


def test_the_motion_task_meets_the_contract():
  env = _env()
  report = check_contract(env, steps=80, end_terms=("motion_end",))
  env.close()
  assert report.missing == []
  assert report.resets > 0, "the rollout must cover episode ends and resets"
  expected = (
    {f"reward/{n}" for n in env.cfg.rewards if env.cfg.rewards[n].weight}
    | {f"obs/{g}/{t}" for g, grp in env.cfg.observations.items() for t in grp.terms}
  )
  assert expected <= set(report.errors)
  for prefix, tol in TOLERANCE.items():
    name, err = report.worst(prefix)
    assert err <= tol, f"{name}: {err:.2e} > {tol:.0e}"


def test_the_checker_catches_violations(monkeypatch):
  env = _env(extra_group=True)
  model = command_model_of(env, "motion")
  advance = model.advance
  monkeypatch.setattr(  # a command model that moves the command two frames per step
    model, "advance", lambda c: advance(advance(c))
  )
  report = check_contract(env, steps=10)
  env.close()
  assert report.missing == ["obs/extra/mjlab_joint_pos"]
  assert report.errors["command/advance/frame"] >= 1
  # The goal and time-left observations read the (wrongly) advanced command state.
  assert report.errors["obs/goal/motion_goal"] > 1e-3
