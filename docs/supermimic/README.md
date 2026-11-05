# SuperMimic on Kangaroo

SuperMimic (arXiv 2610.28480) on PAL Kangaroo, trained on BONES-SEED retargeted to the robot
([kangaroo-bones-seed](https://github.com/OscarMrZ/kangaroo-bones-seed): the retarget pipeline and dataset). Two
robots, each with its own tasks and retarget: 7-DoF arms with grippers, and 5-DoF arms with Seed Robotics hands
(their 2-DoF wrist in the place of arm joints 6–7, fingers fixed; D42).

- [DECISIONS.md](DECISIONS.md): what we chose, why, and when to revisit (D0–D57)
- [LEARNINGS.md](LEARNINGS.md): facts and measurements
- `audit/`: the robot models against PAL's URDF
- `tight/report.md`: the record of the tight tracking runs (D53, rolled back by D57) and of the robot's gains with
  larger action scales (B2, D55)

DECISIONS and LEARNINGS are records: the milestone reports `m0/`–`m5/`, `ROADMAP.md` and the research scripts they cite
are not part of this repository (D52).

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
git clone -b feat/supermimic git@github.com:pal-robotics/pal_mjlab.git && cd pal_mjlab
uv sync --extra cu128 --group supermimic  # plain `uv sync` drops torch-cu128 and pyarrow
TASK=Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers  # hands: Mjlab-SuperMimic-Flat-Pal-Kangaroo-Hands

# 1. A library: download it from the motion browser, or pack a subset of a retargeted dataset folder:
#    uv run -m pal_mjlab.scripts.bones_to_library --subset dev --data-dir <dataset>
#    Without the flag below, the tasks read artifacts/kangaroo_bones_seed/dataset[_5dof]/libraries/dev:
#    link the retarget there (ln -s <kangaroo_bones_seed> artifacts/kangaroo_bones_seed).

# 2. Train on it. Logs and checkpoints in logs/rsl_rl/supermimic/<time>_<run name>/ (default run name: the
#    robot; start your own with it). Resume a run with --agent.resume True --agent.load-run '.*_<run name>'.
uv run train $TASK --env.commands.motion.library-path <library> --agent.run-name grippers-<name>
#    local 16 GB GPU (D16): add --agent.algorithm.precision bf16 --agent.replay-steps 500
#    full (233.5 h): a cluster GPU; the library takes 13 GB of GPU memory

# 3. Replay it with a trained policy, in MuJoCo physics: the library's motions in random order, one after
#    another, or the ones given with --clips (one repeats). No reset when the robot fails: the fall plays
#    out, and the next motion starts 3 s later (D45). MuJoCo's window with a display, else viser.
uv run -m pal_mjlab.scripts.supermimic.play $TASK --checkpoint-file <run>/model_N.pt --library <library>
uv run -m pal_mjlab.scripts.supermimic.play $TASK --checkpoint-file <run>/model_N.pt --library <library> \
  --clips 230710/walk_ff_loop_315_R_001__A419
#    mjlab's own player, on the task's default library (dev) with the evaluation settings:
uv run play $TASK --checkpoint-file <run>/model_N.pt
#    the motions as retargeted, without physics or a policy
uv run -m pal_mjlab.scripts.supermimic.view_motion <library> [--clips <name> ...]

# 4. The policy for the robot: training writes <run>/model_N.onnx next to every checkpoint (D48).
```

**ONNX policy** (`tasks/supermimic/rl/export.py`): input `obs` (1, obs_dim), output `actions`; joint targets
`action_offset + action_scale * actions`. `obs` is the goal, then 10 steps of proprioception (oldest first),
as listed in the file's metadata. The goal is `state.reference_goal` of the next 10 reference frames: joint
positions and velocities, and the reference root orientation in the robot root frame. One file runs any
motion. The motion's root yaw at its first frame must match the robot's.

**How it trains** (Algorithm A.1, D0–D57): every iteration the 4096 simulated robots collect about 2.5 steps
each with exploration noise into a replay buffer, the world model fits 10 batches of 8-step windows from it,
and the policy improves by backpropagating the tracking reward through 32 steps imagined with the world model,
plus a critic's value beyond them. Episodes start at frames picked by the return-based sampler: half uniformly
over 1 s bins of the library, the rest favouring bins the policy does badly on. A checkpoint is saved every
`--agent.save-interval` iterations.

```bash
# A GPU driver older than CUDA 12.4 (e.g. 535 = 12.2): mjlab disables CUDA graphs and the
# simulator runs about 8x slower. Load the forward-compatibility libcuda first, and check the log has no
# "CUDA Graphs disabled" line:
module load cuda/12.6 && export LD_LIBRARY_PATH=$CUDA_HOME/compat:$LD_LIBRARY_PATH

# Ablations (paper Fig. 4): --agent.algorithm.policy.use-critic False   --agent.algorithm.policy.horizon 16
# mjlab CLI syntax: tuples comma-separated (--agent.algorithm.world-model.hidden 1024,1024,1024,
#   one element: 256,), booleans as --flag False.

# Watch and benchmark
tensorboard --logdir logs/rsl_rl/supermimic
uv run -m pal_mjlab.scripts.supermimic.eval --task $TASK --library <library> \
  --checkpoints <run>/model_N.pt --out-dir results/   # every clip of the library once (paper F.1)
```

Key TensorBoard scalars: `episode/return`, `eval/return_gap_head` (imagined vs sim, D31),
`termination/head_auc`, `world_model/loss`, `policy/objective`,
`time/*`, `memory/max_allocated_gb`.

## Configuration

- **Tasks** (D52): `Mjlab-SuperMimic-Flat-Pal-Kangaroo-{Grippers,Hands}` and the PPO-Sim baseline
  `Mjlab-SuperMimic-PPO-Flat-Pal-Kangaroo-{Grippers,Hands}` (logs: `logs/rsl_rl/supermimic_ppo`).
- **Paper settings:** `tasks/supermimic/supermimic_env_cfg.py` (env, reward: Table A.2, D57) and
  `rl/supermimic.py` (algorithm, Table A.5).
- **Kangaroo:** `tasks/supermimic/kangaroo/env_cfgs.py` (tracked bodies, joints, sensors). The arm and waist
  actuators of both robots are set in one place: `ARM_WAIST_GAIN_FREQ_HZ` (their gains' natural frequency; None:
  the robot's 3.5 Hz) and `ARM_WAIST_ACTION_SCALE_FREQ_HZ` (the frequency their action scales are computed at, 1 Hz;
  D55).
- **A policy runs with the gains it was trained with** (D55). Checkpoints from before D55 (1 Hz gains) need
  `ARM_WAIST_GAIN_FREQ_HZ = 1.0` and `ARM_WAIST_ACTION_SCALE_FREQ_HZ = None` to play or benchmark. Each ONNX file
  carries its joint stiffness and damping for the robot.

## Velocity head (D56)

`Mjlab-SuperMimic-Velocity-Flat-Pal-Kangaroo-Hands` drives the Hands robot with a base twist command: (vx, vy)
[m/s] in the root's heading frame and the yaw rate [rad/s]. A velocity encoder (paper App. B, G) maps the command
and the proprio history into the latent of a frozen Hands policy, whose decoder acts. The encoder, its critic and
the world model (refit online) train; the motion policy stays as it is.

```bash
# Pack the walk/jog/run library (34 h) of the 5-DoF dataset once.
uv run -m pal_mjlab.scripts.bones_to_library --subset locomotion --data-dir <dataset_5dof>
# Post-train on a Hands checkpoint (run B: hands-tight-full-24h/model_140000.pt). Logs in
# logs/rsl_rl/supermimic_velocity/<time>_<run name>.
uv run train Mjlab-SuperMimic-Velocity-Flat-Pal-Kangaroo-Hands --agent.motion-checkpoint <run>/model_N.pt \
  --env.commands.motion.library-path <dataset_5dof>/libraries/locomotion --agent.run-name hands-<name>
#    local 16 GB GPU: add --agent.algorithm.precision bf16 --agent.replay-steps 500 and the smoke library's path
#    other command ranges: --env.commands.velocity.ranges.lin-vel-x -1.0,1.5 (and lin-vel-y, ang-vel-z)
uv run -m pal_mjlab.scripts.supermimic.play Mjlab-SuperMimic-Velocity-Flat-Pal-Kangaroo-Hands \
  --checkpoint-file <run>/model_N.pt --library <dataset_5dof>/libraries/locomotion
#    the same without --library: uv run play <task> --checkpoint-file ..., which reads the task's default artifacts/ path
```

Each env draws a command source at every reset and every 5 s: half the time the reference clip's own velocity
(a centered 25-frame mean), otherwise a held command uniform over vx [−1.5, 2], vy ±1.32 m/s and ωz ±3.35 rad/s,
zero 10 % of the time. `model_N.onnx` takes `obs = [velocity_command (3), proprio]`. The task pins the frozen
policy's actuators (run B's): a policy trained with other gains needs them in
`pal_kangaroo_hands_velocity_env_cfg`. On top of the paper's reward, a `rest_pose` term holds the arms and hands near
the default pose (D58). TensorBoard: `env/Episode_Reward/track_lin_vel`, `track_yaw_rate`, `rest_pose`, `policy/align`.

## Commands

```bash
# Motion library: smoke (0.25 h), dev (2.5 h), full (233.5 h) or locomotion (walk/jog/run, the velocity task's).
# Writes to <dataset>/libraries/<subset>.
uv run -m pal_mjlab.scripts.bones_to_library --subset dev --data-dir <dataset>

# Tests (CPU)
uv run pytest tests/supermimic -q

# PPO-Sim baseline (D17)
uv run train Mjlab-SuperMimic-PPO-Flat-Pal-Kangaroo-Grippers --env.commands.motion.library-path <library>
```
