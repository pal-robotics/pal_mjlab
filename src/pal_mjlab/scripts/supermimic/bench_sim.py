"""M0 benchmark: raw MuJoCo Warp throughput and closed-chain behaviour of Kangaroo (grippers).

For each sim setting and env count, every env replays a random clip of a motion library
with the PD actuators: the root is pinned to the reference (written every control step)
and the actuated joints get the next reference frame as position target. This isolates
the joint/closed-chain dynamics from balance, so it runs without a policy.

Reports: control steps/s (num_envs x 50 Hz steps), GPU memory, four-bar loop residual
(world distance between the two anchors of each leg's <connect>), joint tracking error
and NaN count.

Run via the GPU queue, e.g.
  gpu-run --label supermimic-bench --max 30m -- .venv/bin/python -m \
    pal_mjlab.scripts.supermimic.bench_sim --library ~/Projects/supermimic/kangaroo_bones_seed/dataset/libraries/smoke
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import mjlab  # noqa: F401  (loads the task registry before pal_mjlab.robots)
import numpy as np
import torch
import tyro
from mjlab.scene import Scene, SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.sim.sim import Simulation
from mjlab.terrains import TerrainEntityCfg
from pal_mjlab.robots import get_kangaroo_grippers_robot_cfg
from pal_mjlab.tasks.supermimic.kinematics import quat_rotate


@dataclass(frozen=True)
class SimSetting:
  name: str
  timestep: float
  decimation: int
  iterations: int = 10
  ls_iterations: int = 20


SETTINGS = (
  SimSetting("paper_5ms_x4", 0.005, 4),
  SimSetting("wbc_2ms_x10", 0.002, 10),
)


def _gpu_used_gb() -> float:
  free, total = torch.cuda.mem_get_info()
  return (total - free) / 1e9


def run(
  setting: SimSetting,
  num_envs: int,
  qpos_lib: torch.Tensor,
  qvel_lib: torch.Tensor,
  starts: np.ndarray,
  lengths: np.ndarray,
  steps: int,
  nconmax: int,
  njmax: int,
  seed: int,
) -> dict:
  device = "cuda"
  base_mem = _gpu_used_gb()
  scene_cfg = SceneCfg(
    terrain=TerrainEntityCfg(terrain_type="plane"),
    num_envs=num_envs,
    entities={"robot": get_kangaroo_grippers_robot_cfg()},
  )
  scene = Scene(scene_cfg, device=device)
  model = scene.compile()
  sim_cfg = SimulationCfg(
    nconmax=nconmax,
    njmax=njmax,
    mujoco=MujocoCfg(
      timestep=setting.timestep,
      iterations=setting.iterations,
      ls_iterations=setting.ls_iterations,
    ),
  )
  sim = Simulation(num_envs=num_envs, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  robot = scene["robot"]
  mj = sim.mj_model

  joint_names = list(robot.joint_names)
  actuated = [
    i for i, n in enumerate(joint_names) if "femur" not in n and "knee" not in n
  ]
  act_ids = torch.tensor(actuated, device=device)

  # Each env replays a random clip from a random start frame (wrapping inside the clip).
  rng = np.random.default_rng(seed)
  clip = rng.integers(0, len(starts), num_envs)
  offset = (rng.random(num_envs) * lengths[clip]).astype(np.int64)
  clip_start = torch.as_tensor(starts[clip], device=device)
  clip_len = torch.as_tensor(lengths[clip], device=device)
  offset_t = torch.as_tensor(offset, device=device)

  def frame(t: int) -> torch.Tensor:
    return clip_start + (offset_t + t) % clip_len

  env_origins = scene.env_origins

  def write_reference(idx: torch.Tensor, joints_too: bool) -> None:
    qp, qv = qpos_lib[idx], qvel_lib[idx]
    root = torch.cat(
      [
        qp[:, :3] + env_origins,
        qp[:, 3:7],
        qv[:, :3],
        quat_rotate(qp[:, 3:7], qv[:, 3:6]),
      ],
      dim=-1,
    )
    robot.write_root_state_to_sim(root)
    if joints_too:
      robot.write_joint_state_to_sim(qp[:, 7:], qv[:, 6:])

  # Four-bar anchors: <connect> anchor in body1 frame, and its body2-frame counterpart.
  eq_ids = [e for e in range(mj.neq) if mj.eq_type[e] == 0]  # mjEQ_CONNECT
  b1 = torch.as_tensor(mj.eq_obj1id[eq_ids], device=device)
  b2 = torch.as_tensor(mj.eq_obj2id[eq_ids], device=device)
  a1 = torch.as_tensor(mj.eq_data[eq_ids, 0:3], dtype=torch.float32, device=device)
  a2 = torch.as_tensor(mj.eq_data[eq_ids, 3:6], dtype=torch.float32, device=device)

  def loop_residual() -> torch.Tensor:
    xpos, xquat = sim.data.xpos, sim.data.xquat
    p1 = xpos[:, b1] + quat_rotate(xquat[:, b1], a1.expand(num_envs, -1, -1))
    p2 = xpos[:, b2] + quat_rotate(xquat[:, b2], a2.expand(num_envs, -1, -1))
    return (p1 - p2).norm(dim=-1)

  write_reference(frame(0), joints_too=True)
  sim.forward()
  scene.update(setting.timestep)

  def control_step(t: int) -> None:
    write_reference(frame(t), joints_too=False)
    robot.set_joint_position_target(
      qpos_lib[frame(t + 1)][:, 7:][:, act_ids], joint_ids=act_ids
    )
    for _ in range(setting.decimation):
      scene.write_data_to_sim()
      sim.step()
      scene.update(setting.timestep)

  warmup = 20
  for t in range(warmup):
    control_step(t)
  torch.cuda.synchronize()
  res_max, res_sum, err_sum, n = 0.0, 0.0, 0.0, 0
  t0 = time.perf_counter()
  for t in range(warmup, warmup + steps):
    control_step(t)
  torch.cuda.synchronize()
  elapsed = time.perf_counter() - t0

  # Quality pass (not timed): residual and tracking error along a fresh replay.
  nan_envs = torch.zeros(num_envs, dtype=torch.bool, device=device)
  for t in range(warmup + steps, warmup + 2 * steps):
    control_step(t)
    r = loop_residual()
    res_max = max(res_max, float(r.max()))
    res_sum += float(r.mean())
    q = sim.data.qpos[:, 7:][:, act_ids]
    err_sum += float((q - qpos_lib[frame(t + 1)][:, 7:][:, act_ids]).abs().mean())
    nan_envs |= ~torch.isfinite(sim.data.qpos).all(dim=-1)
    n += 1

  out = {
    "setting": setting.name,
    "num_envs": num_envs,
    "control_steps_per_s": num_envs * steps / elapsed,
    "ms_per_vector_step": 1e3 * elapsed / steps,
    "gpu_mem_gb": _gpu_used_gb() - base_mem,
    "loop_residual_mean_mm": 1e3 * res_sum / n,
    "loop_residual_max_mm": 1e3 * res_max,
    "joint_track_err_mean_rad": err_sum / n,
    "nan_envs": int(nan_envs.sum()),
    "cuda_graph": bool(sim.use_cuda_graph),
  }
  return out


def main(
  library: Path,
  num_envs: tuple[int, ...] = (1024, 2048, 4096),
  steps: int = 200,
  nconmax: int = 100,
  njmax: int = 300,
  seed: int = 0,
  out_json: Path | None = None,
) -> None:
  meta = json.loads((library / "clips.json").read_text())
  starts = np.array([c["start"] for c in meta["clips"]])
  lengths = np.array([c["length"] for c in meta["clips"]])
  qpos_lib = torch.from_numpy(np.load(library / "qpos.npy")).cuda()
  qvel_lib = torch.from_numpy(np.load(library / "qvel.npy")).cuda()
  print(
    f"GPU: {torch.cuda.get_device_name()}, library {library.name}: {len(starts)} clips"
  )
  results = []
  for setting in SETTINGS:
    for n in num_envs:
      r = run(
        setting, n, qpos_lib, qvel_lib, starts, lengths, steps, nconmax, njmax, seed
      )
      results.append(r)
      print(
        json.dumps(
          {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}
        )
      )
  if out_json is not None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
  tyro.cli(main)
