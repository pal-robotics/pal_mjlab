"""M0 benchmark: cost of SuperMimic's learning updates on the local GPU.

Times one iteration's learning work with random data and the planned shapes:
  - policy update: BPTT through a frozen world model over H imagined steps, analytic
    tracking reward through torch FK (14 bodies + IMU) at every step, critic bootstrap;
  - world-model fit: 10 optimizer steps on 2048 windows of k=8 autoregressive steps;
  - critic fit: 5 epochs x 4 minibatches over the imagined states.
Sizes: "paper" (Table A.4) and a smaller "dev" variant; precisions fp32, tf32, bf16 autocast.

Numbers are throughput/memory only (random weights, no learning).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import mjlab  # noqa: F401
import torch
import tyro
from pal_mjlab.robots.pal_kangaroo.kangaroo_constants import get_kangaroo_grippers_spec
from pal_mjlab.tasks.supermimic.kangaroo.env_cfgs import TRACKED_BODIES
from pal_mjlab.tasks.supermimic.kinematics import (
  TorchKinematics,
  compile_kinematic_model,
  quat_mul,
)
from torch import nn

TRACKED = TRACKED_BODIES
NJ, NA = 32, 28  # joints in the state (incl. passive femur/knee), actuated joints
STATE = 1 + 6 + 3 + 3 + NJ + NJ  # height, tilt 6D, root lin/ang vel (root frame), q, qd
POSE = 3  # integrated planar position + yaw
XI = 1 + NJ + 3  # friction, calibration offsets, torso CoM offset
OBS = NA + NA + 3 + 3 + NA  # q - q0, qd, IMU ang vel, projected gravity, a_{t-1}
GOAL = (
  NJ + NJ + 6 + 3 + 3 + 1
)  # ref q, qd, root ori (6D, rel.), root lin/ang vel, height
HIST, K = 10, 10


@dataclass(frozen=True)
class Sizes:
  name: str
  wm: tuple[int, ...]
  enc: tuple[int, ...]
  dec: tuple[int, ...]
  critic: tuple[int, ...] = (256, 128)


SIZES = {
  "paper": Sizes(
    "paper", (2048, 2048, 2048), (2048, 1024, 512, 256), (2048, 1024, 512, 256)
  ),
  "dev": Sizes("dev", (1024, 1024, 1024), (1024, 512, 256), (1024, 512, 256)),
}


def mlp(inp: int, hidden: tuple[int, ...], out: int) -> nn.Sequential:
  layers: list[nn.Module] = []
  for h in hidden:
    layers += [nn.Linear(inp, h), nn.ELU()]
    inp = h
  layers.append(nn.Linear(inp, out))
  return nn.Sequential(*layers)


def rot6d_to_quat(r6: torch.Tensor) -> torch.Tensor:
  a, b = r6[..., :3], r6[..., 3:]
  x = nn.functional.normalize(a, dim=-1)
  y = nn.functional.normalize(b - (x * b).sum(-1, keepdim=True) * x, dim=-1)
  z = torch.linalg.cross(x, y, dim=-1)
  m = torch.stack([x, y, z], dim=-1)
  w = 0.5 * torch.sqrt(
    (1.0 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]).clamp_min(1e-6)
  )
  return torch.stack(
    [
      w,
      (m[..., 2, 1] - m[..., 1, 2]) / (4 * w),
      (m[..., 0, 2] - m[..., 2, 0]) / (4 * w),
      (m[..., 1, 0] - m[..., 0, 1]) / (4 * w),
    ],
    dim=-1,
  )


def state_to_q(
  state: torch.Tensor, pose: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Benchmark stand-in for the real state -> (qpos, qvel) map (same op count)."""
  tilt = rot6d_to_quat(state[:, 1:7])
  half = 0.5 * pose[:, 2:3]
  zero = torch.zeros_like(half)
  yaw = torch.cat([torch.cos(half), zero, zero, torch.sin(half)], -1)
  quat = quat_mul(yaw, tilt)
  qpos = torch.cat([pose[:, :2], state[:, :1], quat, state[:, 13 : 13 + NJ]], dim=-1)
  qvel = torch.cat([state[:, 7:13], state[:, 13 + NJ :]], dim=-1)
  return qpos, qvel


def tracking_reward(fk, qpos, qvel, ref_qpos, ref_qvel) -> torch.Tensor:
  if fk is None:  # no FK: a learned-reward-head stand-in of similar arithmetic
    err = (qpos[:, 7:] - ref_qpos[:, 7:]).square().sum(-1) + (
      qvel - ref_qvel
    ).square().sum(-1)
    return torch.exp(-err) * 0.02
  rob = fk(qpos, qvel)
  with torch.no_grad():
    ref = fk(ref_qpos, ref_qvel)
  pos = torch.exp(-((rob.body_pos - ref.body_pos) ** 2).sum(-1).mean(-1) / 0.3**2)
  ori = torch.exp(-((rob.body_quat - ref.body_quat) ** 2).sum(-1).mean(-1) / 0.4**2)
  lin = torch.exp(
    -((rob.body_lin_vel - ref.body_lin_vel) ** 2).sum(-1).mean(-1) / 1.0**2
  )
  ang = torch.exp(
    -((rob.body_ang_vel - ref.body_ang_vel) ** 2).sum(-1).mean(-1) / 3.14**2
  )
  imu = rob.site_ang_vel.square().sum((-1, -2)) * 0.0  # obs head cost, no reward
  return (pos + ori + lin + ang + imu) * 0.02


def bench(
  sizes: Sizes, batch: int, horizon: int, precision: str, fk, fk_mode: str
) -> dict:
  dev = "cuda"
  torch.backends.cuda.matmul.allow_tf32 = precision == "tf32"
  torch.backends.cudnn.allow_tf32 = precision == "tf32"
  autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16")
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()
  base = torch.cuda.memory_allocated()

  wm = mlp(STATE + NA + XI, sizes.wm, STATE + POSE + 1).to(dev)
  enc = mlp(GOAL * K, sizes.enc, 32).to(dev)
  dec = mlp(32 + OBS * HIST, sizes.dec, NA).to(dev)
  critic = mlp(STATE + GOAL * K + XI + POSE, sizes.critic, 1).to(dev)
  pol_opt = torch.optim.Adam(
    [*enc.parameters(), *dec.parameters()], 2e-4, betas=(0.9, 0.95)
  )
  wm_opt = torch.optim.Adam(wm.parameters(), 2e-4)
  cr_opt = torch.optim.Adam(critic.parameters(), 2e-4)

  def policy_update() -> tuple[torch.Tensor, torch.Tensor]:
    wm.requires_grad_(False)
    s = torch.randn(batch, STATE, device=dev) * 0.1
    s[:, 0] += 0.9
    pose = torch.zeros(batch, POSE, device=dev)
    xi = torch.randn(batch, XI, device=dev)
    hist = torch.randn(batch, HIST, OBS, device=dev)
    goals = torch.randn(batch, horizon + K, GOAL, device=dev)
    ref_qpos = torch.randn(batch, horizon, 7 + NJ, device=dev)
    ref_qpos[..., 3:7] = nn.functional.normalize(ref_qpos[..., 3:7], dim=-1)
    ref_qvel = torch.randn(batch, horizon, 6 + NJ, device=dev)
    c = torch.ones(batch, device=dev)
    ret = torch.zeros(batch, device=dev)
    states = []
    with autocast:
      for t in range(horizon):
        g = goals[:, t : t + K].flatten(1)
        a = dec(torch.cat([torch.tanh(enc(g)), hist.flatten(1)], -1))
        out = wm(torch.cat([s, a, xi], -1)).float()
        s = s + out[:, :STATE]
        pose = pose + out[:, STATE : STATE + POSE]
        done = torch.sigmoid(out[:, -1])
        qpos, qvel = state_to_q(s, pose)
        r = tracking_reward(fk, qpos, qvel, ref_qpos[:, t], ref_qvel[:, t])
        c = c * (1.0 - done)
        ret = ret + 0.99**t * c * r
        obs = torch.cat(
          [
            s[:, 13 : 13 + NA],
            s[:, 13 + NJ : 13 + NJ + NA],
            s[:, 10:13],
            s[:, 1:4],
            a.float(),
          ],
          -1,
        )
        hist = torch.cat([hist[:, 1:], obs[:, None]], 1)
        states.append(s.detach())
      g = goals[:, horizon : horizon + K].flatten(1)
      v = critic(torch.cat([s, g, xi, pose], -1)).squeeze(-1).float()
      loss = -(ret + 0.99**horizon * c * v).mean()
    pol_opt.zero_grad(set_to_none=True)
    loss.backward()
    params = [*enc.parameters(), *dec.parameters()]
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    torch.nn.utils.clip_grad_value_(params, 0.1)
    pol_opt.step()
    wm.requires_grad_(True)
    return torch.stack(states, 1), goals

  def wm_fit() -> None:
    for _ in range(10):
      s_true = torch.randn(2048, 9, STATE, device=dev)
      a = torch.randn(2048, 8, NA, device=dev)
      xi = torch.randn(2048, XI, device=dev)
      d = (torch.rand(2048, 8, device=dev) < 0.01).float()
      with autocast:
        s, loss = s_true[:, 0], 0.0
        for j in range(8):
          out = wm(torch.cat([s, a[:, j], xi], -1)).float()
          s = s + out[:, :STATE]
          loss = loss + ((s - s_true[:, j + 1]) ** 2).mean()
          loss = loss + nn.functional.binary_cross_entropy_with_logits(
            out[:, -1], d[:, j]
          )
      wm_opt.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0)
      wm_opt.step()

  def critic_fit(states: torch.Tensor, goals: torch.Tensor) -> None:
    n = states.shape[0] * states.shape[1]
    x = torch.cat(
      [
        states.flatten(0, 1),
        torch.randn(n, GOAL * K + XI + POSE, device=dev),
      ],
      -1,
    )
    target = torch.randn(n, device=dev)
    for _ in range(5):
      for idx in torch.randperm(n, device=dev).chunk(4):
        with autocast:
          loss = ((critic(x[idx]).squeeze(-1).float() - target[idx]) ** 2).mean()
        cr_opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        cr_opt.step()

  def timed(fn, reps: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
      fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps

  result = {
    "sizes": sizes.name, "batch": batch, "horizon": horizon, "precision": precision, "fk": fk_mode
  }  # fmt: skip
  try:
    states, goals = policy_update()  # warm-up
    wm_fit()
    critic_fit(states, goals)
    result["policy_update_s"] = timed(policy_update, 3)
    result["wm_fit_s"] = timed(wm_fit, 2)
    states, goals = policy_update()
    result["critic_fit_s"] = timed(lambda: critic_fit(states, goals), 2)
    result["peak_mem_gb"] = (torch.cuda.max_memory_allocated() - base) / 1e9
    result["learn_s_per_iter"] = (
      result["policy_update_s"] + result["wm_fit_s"] + result["critic_fit_s"]
    )
  except torch.OutOfMemoryError:
    result["oom"] = True
  return result  # networks are freed on return; the next bench empties the cache


def main(
  sizes: tuple[str, ...] = ("dev", "paper"),
  batches: tuple[int, ...] = (1024, 2048, 4096),
  precisions: tuple[str, ...] = ("bf16", "tf32", "fp32"),
  fk_modes: tuple[str, ...] = ("eager",),
  horizon: int = 32,
  out_json: Path | None = None,
) -> None:
  model = compile_kinematic_model(get_kangaroo_grippers_spec())
  fk_eager = TorchKinematics(model, TRACKED, ("imu",), device="cuda")
  fks = {"eager": fk_eager, "none": None}
  if "compiled" in fk_modes:
    fks["compiled"] = torch.compile(fk_eager.forward)
  free, total = torch.cuda.mem_get_info()
  print(
    f"GPU: {torch.cuda.get_device_name()}, free {free / 1e9:.1f} / {total / 1e9:.1f} GB"
  )
  results = []
  for fk_mode in fk_modes:
    for size in sizes:
      for precision in precisions:
        for batch in batches:
          r = bench(SIZES[size], batch, horizon, precision, fks[fk_mode], fk_mode)
          results.append(r)
          print(
            json.dumps(
              {k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}
            )
          )
  if out_json is not None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
  tyro.cli(main)
