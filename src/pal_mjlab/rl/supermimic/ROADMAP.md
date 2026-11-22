# SuperMimic as a reusable learning method: state and roadmap

Working notes for continuing this work in a fresh session. Read this first, then the code.
Branch: `MAC/wbc-supermimic`. Last updated 2026-11-22 (M5 phase 1 implemented; first
locomotion trainings).

## 1. The idea

A colleague implemented SuperMimic (arXiv 2610.28480: motion tracking trained by
backpropagating through a learned, differentiable world model) on branch `omm/wbc`, as one
self-contained motion-tracking task.

Goal of this branch: **SuperMimic as a reusable learning method**. The method lives in
`pal_mjlab/rl/supermimic`; a task provides its env and a small, explicit interface. End
objective: SuperMimic trains tasks that are not motion tracking, ultimately **locomotion**.

Design rules:
- What the env knows by name is configured (`ImaginationCfg`, in the env config).
- **Every observation, reward and failure term is defined once**, as a pure function of one
  step's tensors (`Batch`); the env's term is generated from it (`@imaginable`), so the
  simulator and imagination compute the same thing. A term imagination cannot compute is
  refused, listed, rather than silently left out.
- The command (and everything it owns: its start sampler, its export metadata) belongs to
  the env's command term, which provides its tensor twin (`command_model()`).

## 2. Where things stand

| Milestone | Content |
|---|---|
| M0 (`1e8cccb`) | The colleague's task ported to `tasks/supermimic/` on `rl/supermimic`. |
| M1 (`8d11905`) | No hardcoded env names. |
| M2a (`5c6aa6e`..`b8df475`) | `CommandModel`: the command's twin (`state`, `advance`, `done`, `termination_ref`); opaque command state in replay; robot state from the robot (state joints = action joints); command terms checkpoint their own state (`rl/command_state.py`). |
| M2b (`06a8415`..`8df3a78`) | Single-definition terms (`imagination.py`, `terms.py`); configuration in the env (`SuperMimicEnvCfg.imagination`); the heads name no term. |
| M2c (`c1b84a8`) | Task-agnostic ONNX export; the command adds its own metadata (`export_metadata()`). |
| M3 (`575f1e1`) | `contracts.check_contract`: a task's terms vs imagination on a simulator rollout. |
| M5 phase 1 | The flat velocity task for the base Kangaroo (section 4). |

`rl/supermimic` imports nothing from `tasks/` and reads no task attribute.
`pytest tests/supermimic`: **91 passed**.

Layout:
```
rl/command_state.py     checkpointing of command terms' state (e.g. a start sampler)
rl/supermimic/          the method
  supermimic.py           world model / policy / critic updates, imagination (Eqs. 1-3)
  world_model.py          learned dynamics + learned termination head
  heads.py                the env's terms on imagined batches (rewards, obs, failures)
  imagination.py          Batch, BatchSpec, BatchSource, @imaginable(_termination), term
                          memory, terrain query, capsule bottoms
  terms.py                single-definition versions of mjlab's terms
  command_model.py        the CommandModel protocol
  contracts.py            check_contract
  replay.py, runner.py    replay buffer; mjlab runner (collection, loop, checkpoints)
  state.py, names.py,     heading state; ImaginationCfg / SuperMimicEnvCfg;
  config.py, export.py    SuperMimicRunnerCfg; ONNX export
  COLAB.md                training on a Colab GPU (copy-paste cells)
utils/math/kinematics.py  TorchKinematics: differentiable FK (bodies, sites)
tasks/supermimic/         the Kangaroo motion-tracking task
tasks/supermimic_velocity/  the Kangaroo flat velocity task (M5)
```

The robot contract (explicit, not task coupling): a floating base; joint-position actions
whose joints are the state's; FK bodies and sites listed in `ImaginationCfg` (FK over every
body costs ~50% more per imagined step); a closed chain's passive joints given as a function
(`ImaginationCfg.passive_joints`); MuJoCo position actuators (export); `auto_reset=False`
in training. Root height is relative to the env origin, not the terrain: to change for
rough terrain.

## 3. Key decisions and results (M2-M3)

- **Single definitions over twins** (M2b): twin dicts would mean two definitions per term
  and consistency tests; an "imagined env" running mjlab's own terms on predicted state was
  also rejected. Consequences: the IMU observations are the IMU site's FK (a predicted state
  has no sensors; FK = the sensors to 4.8e-6, mjlab runs `forward()` before observations);
  rewards see the command state the step targeted, observations the advanced one (as mjlab
  updates the command after the rewards).
- **Performance**: a step's terms share one simulator batch (cached on the env step, qpos,
  qvel and `c`; its FK on qpos/qvel alone) and `Batch.memo` (e.g. a motion's reference
  bodies, once per batch). FK is launch-bound without `torch.compile` (~31 ms per call
  whatever the rows): one extra FK per reset costs ~1% of a training iteration here.
- **Parity of the motion task** (seeded 100 iterations, `logs/rsl_rl/kangaroo_supermimic/`):
  M2a and M2b match M1 within 0-2% up to iteration 20 (the protocol); after iteration 50
  runs spread widely run to run (an extra sample of the M2a code went through the same rough
  patch as the M2b rerun). Reference runs: `2026-11-21_18-25-30_m2a_mine`,
  `2026-11-20_01-33-23_m1_mine`.

## 4. M5: locomotion

### Phase 1 (implemented): flat-ground velocity tracking, base Kangaroo

Task `Mjlab-SuperMimic-Velocity-Flat-Pal-Kangaroo` (`tasks/supermimic_velocity/`), built
from the PPO task's `pal_kangaroo_flat_env_cfg()` (robot, sim, plane, actions, command and
its curriculum, resets, domain randomization). Changes, and why:

| Topic | What |
|---|---|
| Env | `SuperMimicEnvCfg`; `auto_reset` only in play; no pushes (to give to the world model later); random episode starts in training (`random_episode_start`: time-outs spread, otherwise they come in waves every 400 iterations). |
| Command | heading control off (the yaw rate is commanded); `VelocityCommandModel`: `c = {"command"}`, constant over the horizon (resampling every 3-8 s is not imagined), never `done`, termination reference = nominal height, upright. |
| Observations | goal = the command; proprio = the PPO actor's terms and noise (IMU gyro / gravity from the IMU site's FK); critic = heading state, command, ξ. |
| ξ | foot friction, encoder offsets, torso CoM, and the randomized joint friction loss, damping and armature (104 values, named parts). |
| Rewards | the PPO terms' formulas and weights as single definitions: 10 kinematic terms, and the foot terms on the foot sites' FK: clearance, slip, and the stateful air time, swing height, soft landing (term memory; touchdown from a smooth contact proxy, `FOOT_CONTACT` 0.02 m / 5 mm, which agrees with the contact sensor 94% of the time; soft landing uses the foot's vertical speed at touchdown in place of the force, weight x2000: unchecked). Dropped: `self_collisions`, `angular_momentum` (the passive linkage). |
| Failures | all single definitions (imagination ends episodes exactly where the sim does): `fell_over` (tilt, tested on its cosine: acos' gradient is infinite upright), `base_too_low` (new: root < 0.4 m), `illegal_contacts` made geometric: the knee linkage's capsules from FK with the passive joints (closed chain), which matches the contact sensor exactly (360/360 contacts, no false alarm). |
| Critic | a target critic (Polyak 0.05) and a return bound (15, from the reward weights): without them the critic's values inflated themselves through bootstrapping (value 121 with negative rewards). |

The task meets `check_contract` (command exact, observations <= 4e-6, rewards <= 6e-6,
failures exact). Also found: the PPO flat task does not build here (its convex-hull reward
plots to `/tmp`, its torque metric asks for the passive joints' actuators); the SuperMimic
task repoints that metric.

### First trainings (laptop, 1024 envs, 512 rollouts; `logs/rsl_rl/kangaroo_supermimic_velocity/`)

| Run | Setup | Outcome |
|---|---|---|
| v1 | H 32, learned failure head | the imagined robot collapses unchecked; unbounded penalties explode |
| v3 | H 32, tilt + low root analytic | the critic diverges from iteration ~45 |
| v4 | H 16 | stable, but the policy kneels: the one failure only the learned head covered |
| v5 | H 16, all failures analytic | the critic inflates itself (before the critic fix) |
| v6 | H 16, + critic fix | stable; episodes up to 520-620 steps |
| v7 | v6 + actor lr 1e-4 | stable; episodes up to 945-970 of 1000 steps (iterations 575-650) |

What v7 learned: **to stand, not to walk.** Its xy velocity error equals standing still's
(0.65 m/s; mean speed 0.04-0.09 m/s); the tracking rewards are what standing scores. Late
in v7 the training curves dropped, but not the policy: deterministically, the iteration-750
policy falls 4 times in 300 steps x 512 envs and its world model predicts its states better
than at 600 (16-step error 3 mm / 0.3 deg). Those curves only average the episodes that
end (falls, and synchronized time-outs: fixed since, `random_episode_start`).

The budget explains it: v7 is **2M env steps**; PPO walks clearly after 1000 iterations x
4096 envs x 24 steps = **~98M**. SuperMimic should need fewer env steps than PPO (it learns
a model from every transition and follows exact gradients), not fewer hours on a laptop; and
on velocity tracking, discovering a gait is an exploration problem where gradient methods
can stall in "stand still safely". Whether it walks sooner than PPO is what to measure next.

### Colab runs (A100/L4, 4096 envs and rollouts, H 32, actor lr 1e-4)

Evaluated deterministically on fresh envs (xy error vs the standing-still error on the same
commands): `colab_h32_lr1e-4` stands at 250 and 500 (error = standing, 0.04 m/s), moves at
750 (0.18 m/s) but not toward the command (0.670 vs 0.624), falling more. `action_noise
0.3` looked the same. The episode curves mislead here (sums / fixed constants, averaged over
ended episodes; time-outs come in waves every ~420 iterations once falls are rare).

Diagnosis: the PPO rewards suit PPO, not gradients. Standing earns ~5 of ~7 per step (yaw
commands are small; upright and posture are full), and the linear tracking kernel (std 0.5)
is flat at standing errors: no gradient toward the command. Retuned
(`env_cfgs.TRACK_LIN_WEIGHT`, `TRACK_*_STD`): linear weight 2 -> 4, stds 0.5 / 0.71 -> 1.0;
the return bound follows (19). Result (`track4_std1`, 2000 it, ~20M env steps): **it
walks**, but does not track the command yet.

Second retune (gradient-visible gait terms): `feet_air_time` with soft window edges
(mjlab's hard window has zero gradient: invisible to imagination), weight 0.25 -> 0.5; a
narrow linear tracking kernel next to the wide one (weight 2, std 0.25) for precision;
`foot_clearance` -2 -> -0.5 (it charged the first, low steps); return bound 24.
Third retune (`gait_v2` at 3000 it tracks, in odd postures): pose weight 1 -> 2, plus
`posture_l2` (-0.1, the pose kernel's exponent as a penalty: gradient where the kernel's
has faded); `variable_posture` and `upright` match mjlab's to 1e-6 (checked); upright
unchanged. Return bound 26.
Logging reworked to read like PPO's: `step/*` are means over all envs at every step
(each reward term per second, terminations per minute, mean time to failure, the velocity
command's tracking error vs standing still's via `step_metrics()`); mjlab's episode logs
stay, but are biased while episodes end mostly in falls.
Fallback if it still stands: walking data from a PPO
checkpoint in replay (mixed collection), or distillation first.

### Next

1. **A fair run on a good GPU** (`COLAB.md`): horizon 32 (the paper's; a gait cycle is
   longer than H 16's 0.32 s), 4096 envs and rollouts, `torch.compile`, actor lr 1e-4 (a
   flag for now), to ~20M env steps. Compare tracking reward vs **env steps** with PPO's.
2. Only if it still stands at a comparable budget: rebalance tracking vs stability, check the
   command curriculum, or add exploration.
3. Then: log the world model's multi-step error on fresh data; compare seeds.

### Phase 2 (rough terrain, later)

Heightfield terrain query (`imagination.terrain_height`, used by the foot terms and capsule
bottoms); terrain features into the world model; terrain-relative root height; a learned
contact head where geometry is not enough; pushes into the world model.

## 5. Other milestones and open decisions

- **M4, WBC with SuperMimic**: a `CommandModel` for `tasks/WBC`'s `OnlineMotionCommand`
  (one motion pipeline) vs porting the colleague's command; give its adaptive sampler a
  `state_dict`; its rewards as single definitions (check angular momentum, CoP / CoM).
- **Multiple goal encoders** (the paper distills several goal observations in one run): not
  implemented. Missing: `goal_groups`, a multi-encoder actor, its training (each encoder
  through imagination vs latent distillation onto a teacher), one ONNX per encoder.
- Start samplers: owned by the command term (decided); possibly one binned sampler with a
  pluggable score shared by `MotionLibraryCommand` and `OnlineMotionCommand` (with M4).

## 6. Dev environment (Windows laptop, RTX 3060 6 GB): read before running anything

- **CUDA torch is installed by hand into `.venv`** (`torch 2.10.0+cu128`,
  `torchvision 0.25.0`). pyproject has no PyTorch index, so on Windows `uv sync` /
  `uv run` silently reinstall the **CPU** build. Use `.venv/Scripts/python.exe` or
  `.venv/Scripts/train.exe` directly, or `uv run --no-sync`. To restore:
  `uv pip install --python .venv/Scripts/python.exe torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128 --reinstall-package torch --reinstall-package torchvision`
- **No Triton on Windows** -> no `torch.compile`: train with `--agent.compile False` (and
  `--env.commands.motion.compile-kinematics False` for the motion task); play with
  `$env:TORCHDYNAMO_DISABLE=1`.
- **Set `PYTHONUTF8=1`**, otherwise mjlab's manager tables crash the cp1252 console.
- **VRAM**: `num_rollouts=4096` needs ~12 GB (BPTT through 32 steps); use
  `--agent.algorithm.policy.num-rollouts 512`. `--env.scene.num-envs 1024` is fine.
- **Never run two trainings at once on this GPU** (VRAM spills to shared memory).
- **Process priority**: the venv's `train.exe` / `python.exe` are launchers; the training
  runs in a separate base `python.exe`: lower that one.
- Play a velocity checkpoint:
  `$env:PYTHONUTF8=1; $env:MJLAB_WARP_QUIET=1; $env:TORCHDYNAMO_DISABLE=1; .\.venv\Scripts\play.exe Mjlab-SuperMimic-Velocity-Flat-Pal-Kangaroo --checkpoint-file <model_N.pt> --viewer native --device cuda --num-envs 4`
- Tensorboard 2.20 needs `pkg_resources` (gone from setuptools >= 81, the venv has 82):
  `uvx --with "setuptools<81" tensorboard==2.20.0 --logdir logs/rsl_rl/<experiment>`.
- `test_play_launches` needs symlinks; it falls back to copying on Windows.
- Import-order trap is fixed (`pal_mjlab/__init__` imports mjlab first); keep it that way.

**Motion data (never commit data)**: the real library (`artifacts/kangaroo_grippers_library`,
from the SOMA retargeter) is not available. A stand-in was built from two DCMR LAFAN clips
(jumps1_subject1, sprint1_subject2; Kangaroo-grippers CSVs in
`../DCMR/data/check/batch/`, 30 fps) resampled to 50 Hz:
```
.venv/Scripts/python.exe -I artifacts/tools/dcmr_to_library.py artifacts/kangaroo_grippers_library 30 50 \
  jumps1_subject1=jumps=../DCMR/data/check/batch/jumps1_subject1.csv \
  sprint1_subject2=sprint=../DCMR/data/check/batch/sprint1_subject2.csv
```
`artifacts/` is gitignored; the helper scripts live in `artifacts/tools/` (not in git).

## 7. Parity protocol for the motion task (run after changes to the method)

1. `PYTHONUTF8=1 .venv/Scripts/python.exe -m pytest tests/supermimic -q`
2. Short seeded training (identical flags for every run you compare):
   ```
   PYTHONUTF8=1 MJLAB_WARP_QUIET=1 .venv/Scripts/train.exe Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers \
     --env.scene.num-envs 1024 --env.commands.motion.library-path <abs path to library> \
     --env.commands.motion.compile-kinematics False --agent.compile False \
     --agent.algorithm.policy.num-rollouts 512 --agent.seed 42 --agent.max-iterations 100 \
     --agent.save-interval 100000 --agent.log-interval 5 --agent.run-name <name>
   ```
3. Compare with `logs/rsl_rl/kangaroo_supermimic/2026-11-21_18-25-30_m2a_mine`:
   `.venv/Scripts/python.exe -I artifacts/tools/compare_runs.py <run A> <run B>`.
   Expect agreement within a few % up to ~it 20 (no bit-exactness, wide spread after ~50).
