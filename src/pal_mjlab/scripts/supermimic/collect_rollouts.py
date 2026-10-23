"""Collect world-model training data: policy rollouts in the training env (M2).

Each checkpoint's deterministic policy acts with the paper's collection noise
a + N(0, σ²) (σ = 0.1) in the training env (domain randomization on, return-based resets),
like the transitions SuperMimic's replay buffer holds. The env runs with
``auto_reset=False`` so terminal states are kept before the manual reset.

Per checkpoint, arrays of shape (T, E, ·) are written to ``<out_dir>/<checkpoint name>/``:
  qpos, qvel             state before the step (root position in the env's local frame)
  next_qpos, next_qvel   state after the step (the terminal state if the episode ended)
  action                 executed (noisy) action
  context                sim context ξ: foot friction, calibration offsets, torso CoM
  clip, frame            reference clip and the target frame the step is scored against
  reward                 the env's tracking reward for the step (term values, no dt)
  failure, done          root height/tilt failure; any episode end (failure, clip end, cap)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import mjlab.tasks  # noqa: F401  (populates the task registry)
import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from pal_mjlab.tasks.supermimic.mdp import MotionLibraryCommand


def collect(
  env: ManagerBasedRlEnv,
  vec: RslRlVecEnvWrapper,
  policy,
  steps: int,
  action_noise: float,
  generator: torch.Generator,
) -> dict[str, np.ndarray]:
  cmd: MotionLibraryCommand = env.command_manager.get_term("motion")
  context_cfg = env.observation_manager.get_term_cfg("critic", "sim_context")
  rm = env.reward_manager
  tracking = [i for i, n in enumerate(rm.active_terms) if n.startswith("motion_")]
  origins = env.scene.env_origins

  def local(qpos: torch.Tensor) -> torch.Tensor:
    qpos = qpos.clone()
    qpos[:, :3] -= origins
    return qpos

  buf: dict[str, list[np.ndarray]] = {}

  def put(name: str, x: torch.Tensor) -> None:
    buf.setdefault(name, []).append(x.detach().cpu().numpy())

  vec.reset()
  obs = vec.get_observations()
  for _ in range(steps):
    put("qpos", local(cmd.robot_qpos))
    put("qvel", cmd.robot_qvel)
    put("clip", cmd.clip.int())
    put("frame", cmd.frame.int())
    put("context", context_cfg.func(env, **context_cfg.params))
    with torch.no_grad():
      action = policy(obs)
    action = action + action_noise * torch.randn(
      action.shape, device=action.device, generator=generator
    )
    put("action", action)
    obs, _, dones, _ = vec.step(action)
    # auto_reset=False: the sim still holds the post-step (terminal) state.
    put("next_qpos", local(cmd.robot_qpos))
    put("next_qvel", cmd.robot_qvel)
    put("reward", rm._step_reward[:, tracking].sum(-1))
    put("failure", env.termination_manager.get_term("anchor_failure"))
    done = dones.bool()
    put("done", done)
    if done.any():
      env.reset(env_ids=done.nonzero().squeeze(-1))
      obs = vec.get_observations()
  return {k: np.stack(v) for k, v in buf.items()}


def main(
  task: str,
  checkpoints: list[Path],
  out_dir: Path,
  num_envs: int = 4096,
  steps: int = 500,
  action_noise: float = 0.1,
  seed: int = 0,
  device: str = "cuda",
) -> None:
  configure_torch_backends()
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = num_envs
  cfg.auto_reset = False
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  vec = RslRlVecEnvWrapper(env)
  runner = MjlabOnPolicyRunner(vec, asdict(load_rl_cfg(task)), None, device)
  generator = torch.Generator(device=device).manual_seed(seed)
  cmd: MotionLibraryCommand = env.command_manager.get_term("motion")
  for ckpt in checkpoints:
    runner.load(str(ckpt), map_location=device)
    policy = runner.get_inference_policy(device)
    data = collect(env, vec, policy, steps, action_noise, generator)
    out = out_dir / ckpt.stem
    out.mkdir(parents=True, exist_ok=True)
    for name, arr in data.items():
      np.save(out / f"{name}.npy", arr)
    meta = {
      "task": task,
      "checkpoint": str(ckpt),
      "library": str(cmd.library.path),
      "num_envs": num_envs,
      "steps": steps,
      "action_noise": action_noise,
      "step_dt": env.step_dt,
      "failures": int(data["failure"].sum()),
      "episodes_ended": int(data["done"].sum()),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(
      f"{ckpt.stem}: {steps} x {num_envs} transitions, {meta['failures']} failures, "
      f"{meta['episodes_ended']} episode ends -> {out}",
      flush=True,
    )


if __name__ == "__main__":
  tyro.cli(main)
