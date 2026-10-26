"""Replay and training entry points (D45): play.py's config (library, clips, held failure), the held
failure itself, and train.py's --library."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import tyro
from conftest import get_test_device
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks import registry
from pal_mjlab.scripts.supermimic import play, train
from pal_mjlab.tasks.supermimic import mdp
from pal_mjlab.tasks.supermimic.kangaroo.env_cfgs import (
  pal_kangaroo_grippers_supermimic_env_cfg,
  pal_kangaroo_hands_supermimic_env_cfg,
)
from pal_mjlab.tasks.supermimic.mdp import terminations

FIXTURES = Path(__file__).parent / "fixtures"
TASK = "Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers-ArmWaist1Hz"


def test_held_failure_waits_for_its_duration(monkeypatch):
  """Ends an episode only after the failure lasted `seconds` without a break; reset per env."""
  env = SimpleNamespace(num_envs=2, device="cpu", step_dt=0.5)
  failed = iter(
    torch.tensor(v)
    for v in ([1, 0], [1, 1], [0, 1], [1, 1], [1, 1])
  )
  monkeypatch.setattr(terminations, "anchor_failure", lambda *a, **k: next(failed).bool())
  term = terminations.anchor_failure_held(cfg=None, env=env)
  out = [term(env, "motion", 0.25, 0.8, 1.0).tolist() for _ in range(4)]
  assert out == [[False, False], [True, False], [False, True], [False, True]]
  term.reset(env_ids=torch.tensor([1]))
  assert term(env, "motion", 0.25, 0.8, 1.0).tolist() == [True, False]


def test_play_config_sets_library_clips_and_held_failure():
  cfg = registry._REGISTRY[TASK].play_env_cfg
  motion = cfg.commands["motion"]
  saved = (motion.library_path, motion.clips, dict(cfg.terminations))
  try:
    args = play.PlayArgs(checkpoint_file="m.pt", library="lib", clips=("230324/flip_360_001__A304",))
    run_cfg = play.configure(TASK, args)
    run_cfg = play.configure(TASK, args)  # twice: still one held failure
    assert (run_cfg.checkpoint_file, run_cfg.num_envs) == ("m.pt", 1)
    assert (motion.library_path, motion.clips) == ("lib", ("230324/flip_360_001__A304",))
    failure = cfg.terminations["anchor_failure"]
    assert failure.func is mdp.anchor_failure_held
    assert failure.params == {**saved[2]["anchor_failure"].params, "seconds": play.FALL_HOLD_S}
  finally:
    motion.library_path, motion.clips = saved[:2]
    cfg.terminations.clear()
    cfg.terminations.update(saved[2])


@pytest.mark.parametrize(
  "make_cfg, library",
  [(pal_kangaroo_grippers_supermimic_env_cfg, "library_tiny"),
   (pal_kangaroo_hands_supermimic_env_cfg, "library_tiny_hands")],
)
def test_held_failure_runs_in_the_play_env(make_cfg, library):
  """The class-based term builds and fires in a real play env (large random actions make it fall)."""
  cfg = make_cfg(play=True)
  cfg.scene.num_envs = 2
  cfg.commands["motion"].library_path = str(FIXTURES / library)
  cfg.commands["motion"].compile_kinematics = False
  params = {**cfg.terminations["anchor_failure"].params, "seconds": 0.2}
  cfg.terminations["anchor_failure"].func = mdp.anchor_failure_held
  cfg.terminations["anchor_failure"].params = params
  env = ManagerBasedRlEnv(cfg=cfg, device=get_test_device())
  env.reset()
  gen = torch.Generator(device=env.device).manual_seed(1)
  fired = 0
  for _ in range(100):
    action = 3.0 * torch.randn(
      2, env.action_manager.total_action_dim, device=env.device, generator=gen
    )
    env.step(action)
    fired += int(env.termination_manager.get_term("anchor_failure").sum())
  env.close()
  assert fired > 0


def test_train_takes_a_library():
  lib, rest = tyro.cli(
    train.LibraryArg,
    args=["--library", "some/lib", "--agent.run-name", "x"],
    add_help=False,
    return_unknown_args=True,
  )
  assert lib.library == "some/lib"
  assert rest == ["--agent.run-name", "x"]
