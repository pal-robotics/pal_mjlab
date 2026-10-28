# SuperMimic on Kangaroo

SuperMimic (arXiv 2610.28480) on PAL Kangaroo with 7-DoF arms, trained on the BONES-SEED retarget in
`~/Projects/supermimic/kangaroo_bones_seed`. A second variant, 5-DoF arms with Seed Robotics hands
(their 2-DoF wrist in the place of arm joints 6–7, fingers fixed), has its own tasks (`...-Kangaroo-Hands`,
D42) and retarget (`dataset_5dof`).

- [ROADMAP.md](ROADMAP.md): milestones and their gates
- [DECISIONS.md](DECISIONS.md): what we chose, why, and when to revisit (D0–D45)
- [LEARNINGS.md](LEARNINGS.md): facts and measurements
- `m0/`: raw benchmark outputs
- `m1/`: gain A/B (`gain_ab.md`), PPO-Sim baseline (`ppo_sim_baseline.md`), action-scale note
- `m2/`: offline world model: `world_model.md` (short), `report.pdf` (full write-up), `make_figures.py`
- `m3/`: imagination and policy-update checks: `imagination.md` (short), `report.pdf`, `make_figures.py`
- `m4/`: the online loop on `dev` vs PPO-Sim: `online_loop.md` (short), `report.pdf`, `make_figures.py`;
  the 24 h cluster runs with the paper's settings and the M5 options: `cluster_runs.md`,
  `cluster_report.pdf`, `make_cluster_figures.py`; the stability study: `stability.md`,
  `stability_report.pdf`, `make_stability_figures.py`
- `m5/`: the 24 h `full` and `dev` runs with the D39 defaults and their benchmark: `runs.md`, `make_figures.py`

## Environment

```bash
uv sync --extra cu128 --group supermimic  # plain `uv sync` drops torch-cu128 and pyarrow
```

## Motion libraries: download, train, replay

Everything works on a **motion library**, a folder with three files: what training reads, what the motion
browser downloads (smoke, dev, full, the list's motions, or one motion; each with a README) and what play takes.

| file | content |
|---|---|
| `clips.json` | fps (50), joint names, and per clip its name (`<date>/<clip>`), category, start frame and length |
| `qpos.npy` | (frames, 39) float32: root position (m), root quaternion (w, x, y, z), 32 joint positions (rad) |
| `qvel.npy` | (frames, 38) float32: root linear velocity (world), root angular velocity (body), 32 joint velocities |

Clips are stacked in time: clip *i* is rows `start` to `start + length` of both arrays.

```bash
# 0. Code and environment
git clone -b omm/supermimic git@github.com:pal-robotics/pal_mjlab.git && cd pal_mjlab
uv sync --extra cu128 --group supermimic  # plain `uv sync` drops torch-cu128 and pyarrow
TASK=Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers-ArmWaist1Hz  # hands: ...-Kangaroo-Hands-ArmWaist1Hz

# 1. A library: download it from the motion browser (helium-kangaroo-retarget, or kangaroo-motions with
#    the token), or pack a subset of the dataset here: python -m pal_mjlab.scripts.bones_to_library --subset 10h

# 2. Train on it. Logs and checkpoints in logs/rsl_rl/supermimic/<time>_<robot>-<run name>/; the same
#    command again resumes from the newest checkpoint of that run name (preemptions).
uv run python -m pal_mjlab.scripts.supermimic.train $TASK --library <library> --agent.run-name <name>
#    local 16 GB GPU (D16): add --agent.algorithm.precision bf16 --agent.replay-steps 500
#    full (233.5 h): a cluster GPU; the library takes 13 GB of GPU memory

# 3. Replay it with a trained policy, in MuJoCo physics: the library's motions in random order, one after
#    another, or the ones given with --clips (one repeats). No reset when the robot fails: the fall plays
#    out, and the next motion starts 3 s later (D45). MuJoCo's window with a display, else viser.
uv run python -m pal_mjlab.scripts.supermimic.play $TASK --checkpoint-file <run>/model_N.pt --library <library>
uv run python -m pal_mjlab.scripts.supermimic.play $TASK --checkpoint-file <run>/model_N.pt --library <library> \
  --clips 230324/flip_360_001__A304
#    the motions as retargeted, without physics or a policy
uv run python -m pal_mjlab.scripts.supermimic.view_motion <library> [--clips <name> ...]

# 4. The policy for the robot: training writes <run>/model_N.onnx next to every checkpoint (D48); for
#    checkpoints saved before that (any library of the robot sets up the env):
uv run python -m pal_mjlab.scripts.supermimic.export_onnx $TASK --library <library> <run>/model_N.pt
```

**ONNX policy** (`tasks/supermimic/rl/export.py`): input `obs` (1, obs_dim), output `actions`; joint targets
`action_offset + action_scale * actions`. `obs` is the goal, then 10 steps of proprioception (oldest first),
as listed in the file's metadata. The goal is `state.reference_goal` of the next 10 reference frames: joint
positions and velocities, and the reference root orientation in the robot root frame. One file runs any
motion. The motion's root yaw at its first frame must match the robot's.

**How it trains** (Algorithm A.1, D0–D44): every iteration the 4096 simulated robots collect about 2.5 steps
each with exploration noise into a replay buffer, the world model fits 10 batches of 8-step windows from it,
and the policy improves by backpropagating the tracking reward through 32 steps imagined with the world model,
plus a critic's value beyond them. Episodes start at frames picked by the return-based sampler: half uniformly
over 1 s bins of the library, the rest favouring bins the policy does badly on. A checkpoint is saved every
`--agent.save-interval` iterations.

```bash
# Cluster with a driver older than CUDA 12.4 (Leonardo: 535 = 12.2): mjlab disables CUDA graphs and the
# simulator runs about 8x slower. Load the forward-compatibility libcuda first, and check the log has no
# "CUDA Graphs disabled" line:
module load cuda/12.6 && export LD_LIBRARY_PATH=$CUDA_HOME/compat:$LD_LIBRARY_PATH

# Ablations (paper Fig. 4): --agent.algorithm.policy.use-critic False
#   --agent.algorithm.policy.horizon 16    --agent.algorithm.policy.termination rule (D31)
# mjlab CLI syntax: tuples comma-separated (--agent.algorithm.world-model.hidden 1024,1024,1024,
#   one element: 256,), booleans as --flag False.

# Watch and benchmark
tensorboard --logdir logs/rsl_rl/supermimic
uv run python -m pal_mjlab.scripts.supermimic.eval --task $TASK --library <library> \
  --checkpoints <run>/model_N.pt --out-dir results/   # every clip of the library once (paper F.1)
```

Key TensorBoard scalars: `episode/return`, `eval/return_gap_head` (imagined vs sim, D31),
`termination/head_auc` vs `termination/rule_auc`, `world_model/loss`, `policy/objective`,
`time/*`, `memory/max_allocated_gb`.

## Commands

```bash
# Motion library: smoke (0.25 h), dev (2.5 h), full (233.5 h) or hours (10h). Writes to
# <data-dir>/libraries/<subset> (default data dir: kangaroo_bones_seed/dataset; the training
# CLI also packs a missing subset on first use).
.venv/bin/python -m pal_mjlab.scripts.bones_to_library --subset dev

# Tests (CPU, ~35 s)
.venv/bin/python -m pytest tests/supermimic -q

# PPO-Sim baseline (D17): resumes from the latest checkpoint if the job is preempted and rerun
gpu-run --detach --label sm-ppo --max 12h -- .venv/bin/python -m pal_mjlab.scripts.supermimic.train \
  Mjlab-SuperMimic-PPO-Flat-Pal-Kangaroo-Grippers
# other library: --library <folder>; logs: logs/rsl_rl/supermimic_ppo

# M2 world model (GPU, through the queue). Write paths out in full: the queue's shell does not
# word-split a $VAR into several arguments.
D=logs/supermimic_m2/data; P=logs/rsl_rl/supermimic_ppo_sim_1hz/2026-10-22_21-11-56
gpu-run --label sm-collect --max 40m -- .venv/bin/python -m pal_mjlab.scripts.supermimic.collect_rollouts \
  --task Mjlab-SuperMimic-PPO-Flat-Pal-Kangaroo-Grippers-ArmWaist1Hz \
  --checkpoints $P/model_1000.pt $P/model_4000.pt $P/model_8800.pt --out-dir $D
gpu-run --label sm-wm --max 2h -- .venv/bin/python -m pal_mjlab.scripts.supermimic.train_world_model \
  --data $D/model_1000 $D/model_4000 $D/model_8800 --out-dir logs/supermimic_m2/wm/main
gpu-run --label sm-wm-eval --max 30m -- .venv/bin/python -m pal_mjlab.scripts.supermimic.eval_world_model \
  --models main=logs/supermimic_m2/wm/main/wm_best.pt --data $D/model_1000 $D/model_4000 $D/model_8800 \
  --out-json logs/supermimic_m2/eval/all.json
gpu-run --label sm-grad --max 30m -- .venv/bin/python -m pal_mjlab.scripts.supermimic.grad_check \
  --model logs/supermimic_m2/wm/main/wm_best.pt --data $D/model_1000 $D/model_4000 $D/model_8800 \
  --out-json logs/supermimic_m2/eval/grad_main.json
.venv/bin/python docs/supermimic/m2/make_figures.py && (cd docs/supermimic/m2 && latexmk -pdf report.tex)

# M3 checks: the PPO-Sim actor on the frozen M2 world model (closed loop, then 300 updates);
# variants: --no-use-critic --critic-iterations 0 | --actor-lr 2e-5 | --termination rule
gpu-run --label sm-m3 --max 45m -- .venv/bin/python -m pal_mjlab.scripts.supermimic.m3_check \
  --out-json logs/supermimic_m3/m3.json
.venv/bin/python docs/supermimic/m3/make_figures.py && (cd docs/supermimic/m3 && latexmk -pdf report.tex)

# M4 stability study (m4/stability.md): extremes of the imagined reward and the critic's error
# before/after the predicted fall, from training checkpoints (--precision bf16 for local runs)
gpu-run --label sm-diag --max 30m -- .venv/bin/python -m pal_mjlab.scripts.supermimic.reward_extremes \
  --task Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers-ArmWaist1Hz --checkpoints <run>/model_N.pt \
  --out-dir logs/supermimic_diag/reward_extremes

# Benchmarks (GPU, through the queue)
gpu-run --label sm-sim --max 30m -- .venv/bin/python -m pal_mjlab.scripts.supermimic.bench_sim \
  --library ~/Projects/supermimic/kangaroo_bones_seed/dataset/libraries/smoke --out-json docs/supermimic/m0/bench_sim.json
gpu-run --label sm-bptt --max 45m -- .venv/bin/python -m pal_mjlab.scripts.supermimic.bench_bptt \
  --out-json docs/supermimic/m0/bench_bptt_eager.json
```
