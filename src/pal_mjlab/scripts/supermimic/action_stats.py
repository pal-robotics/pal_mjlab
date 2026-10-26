"""Per-joint actions of a tracking policy on the benchmark (play config, paper F.1).

Plays a stratified sample of library clips once from their first frame with a deterministic
policy and records, for every step of every env still in its first episode:

  action        raw policy output a, per actuated joint (action order)
  q, q_ref      joint position after the step and the reference at the frame it is scored
                against, both relative to the default pose q0
  alpha         rad of setpoint per action unit (the action term's scale)

From these: the hold action (q - q0)/alpha, the action rate a_t - a_{t-1}, its penalty
0.1 * (a_t - a_{t-1})^2, and how much each joint follows its reference.

  python -m pal_mjlab.scripts.supermimic.action_stats --task <task> --checkpoint model.pt \\
    --out logs/action_stats/<tag>.npz [--num-clips 128] [--max-steps 500] [--device cpu]
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import mjlab.tasks  # noqa: F401  (populates the task registry)
import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from pal_mjlab.tasks.supermimic.library import MotionLibrary, resolve_library
from pal_mjlab.tasks.supermimic.mdp import MotionLibraryCommand


def stratified_clips(library: MotionLibrary, n: int, seed: int) -> np.ndarray:
  """n clip ids, proportional to each category's clip count (at least one each), sorted."""
  rng = np.random.default_rng(seed)
  cats = library.clip_category.cpu().numpy()
  picked = []
  for c in np.unique(cats):
    ids = np.flatnonzero(cats == c)
    k = max(1, round(n * len(ids) / len(cats)))
    picked += list(rng.choice(ids, size=min(k, len(ids)), replace=False))
  return np.sort(np.array(picked[:n] if len(picked) > n else picked))


def main(
  task: str,
  checkpoint: Path,
  out: Path,
  num_clips: int = 128,
  max_steps: int = 500,
  seed: int = 0,
  device: str = "cpu",
) -> None:
  cfg = load_env_cfg(task, play=True)
  motion = cfg.commands["motion"]
  motion.compile_kinematics = device != "cpu"
  library = MotionLibrary(
    resolve_library(motion.library_path, motion.dataset, motion.subset)
  )
  clips = stratified_clips(library, num_clips, seed)
  cfg.scene.num_envs = len(clips)
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  vec = RslRlVecEnvWrapper(env)

  runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
  runner = runner_cls(vec, asdict(load_rl_cfg(task)), None, device)
  runner.load(str(checkpoint), load_cfg={"actor": True}, map_location=device)
  policy = runner.get_inference_policy(device)

  cmd: MotionLibraryCommand = env.command_manager.get_term("motion")
  term = env.action_manager.get_term("joint_pos")
  ids = term._target_ids
  names = list(term.target_names)
  alpha = term._scale.expand(env.num_envs, -1)[0].cpu().numpy()
  robot = env.scene["robot"]
  q0 = robot.data.default_joint_pos[0, ids]

  cmd.set_eval_clips(torch.as_tensor(clips, device=device))
  obs, _ = vec.reset()
  n = env.num_envs
  finished = torch.zeros(n, dtype=torch.bool, device=device)
  rows = {"env": [], "action": [], "action_prev": [], "q": [], "q_ref": []}
  prev = torch.zeros(n, len(ids), device=device)
  with torch.no_grad():
    longest = int(library.clip_length[torch.as_tensor(clips)].max()) + 2
    for _ in range(min(longest, max_steps)):
      target = cmd.frame.clone()
      action = policy(obs)
      obs, _, dones, _ = vec.step(action)
      live = (~finished) & (~dones.bool())  # state after the step is valid
      raw = term.raw_action if hasattr(term, "raw_action") else term._raw_actions
      q_ref = cmd.library.qpos[target][:, 7:][:, ids] - q0
      q = robot.data.joint_pos[:, ids] - q0
      sel = live.nonzero().squeeze(-1)
      rows["env"].append(sel.cpu().numpy())
      rows["action"].append(raw[sel].cpu().numpy())
      rows["action_prev"].append(prev[sel].cpu().numpy())
      rows["q"].append(q[sel].cpu().numpy())
      rows["q_ref"].append(q_ref[sel].cpu().numpy())
      prev = raw.clone()
      finished |= dones.bool()
      if finished.all():
        break

  out.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    out,
    joint_names=np.array(names),
    alpha=alpha,
    clips=clips,
    clip_names=np.array([library.clip_names[i] for i in clips]),
    task=task,
    checkpoint=str(checkpoint),
    **{k: np.concatenate(v) for k, v in rows.items()},
  )
  print(
    f"{task}: {len(clips)} clips, {sum(len(e) for e in rows['env'])} steps -> {out}"
  )


if __name__ == "__main__":
  tyro.cli(main)
