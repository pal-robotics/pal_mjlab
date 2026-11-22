# SuperMimic as a reusable learning method: state and roadmap

Our working notes, for picking the work up in a fresh session. Read this first, then the
code; the user-facing documentation is `docs/rl/supermimic.rst`.
Branch: `MAC/wbc-supermimic`. Last updated 2026-11-22 (M5 phase 1: the robot walks;
tracking and gait quality in progress).

## 1. The idea

omm implemented SuperMimic (arXiv 2610.28480: motion tracking trained by backpropagating
through a learned, differentiable world model) on branch `omm/wbc`, as one self-contained
motion-tracking task.

On this branch we make **SuperMimic a reusable learning method**: the method lives in
`pal_mjlab/rl/supermimic`, and a task provides its env and a small, explicit interface. The
end goal is to train tasks other than motion tracking with it, first of all **locomotion**.

Design rules we keep to:
- What the method reads from the env by name is configured (`ImaginationCfg`, in the env
  config).
- **Every observation, reward and failure term is defined once**, as a pure function of one
  step's tensors (`Batch`); the env's term is generated from it (`@imaginable`), so the
  simulator and imagination compute the same thing. A term imagination cannot compute is
  refused and listed, never silently left out.
- The command, and everything it owns (start sampler, export metadata), belongs to the env's
  command term, which provides its tensor twin (`command_model()`).

## 2. Where we stand

| Milestone | What we did |
|---|---|
| M0 (`1e8cccb`) | Ported omm's task to `tasks/supermimic/` on top of `rl/supermimic`. |
| M1 (`8d11905`) | Removed the hardcoded env names. |
| M2a (`5c6aa6e`..`b8df475`) | `CommandModel`, the command's twin (`state`, `advance`, `done`, `termination_ref`); opaque command state in replay; robot state from the robot (state joints = action joints); command terms checkpoint their own state (`rl/command_state.py`). |
| M2b (`06a8415`..`8df3a78`) | Single-definition terms (`imagination.py`, `terms.py`); configuration in the env (`SuperMimicEnvCfg.imagination`); the heads name no term. |
| M2c (`c1b84a8`) | Task-agnostic ONNX export; the command adds its own metadata (`export_metadata()`). |
| M3 (`575f1e1`) | `contracts.check_contract`: a task's terms vs imagination on a simulator rollout. |
| M5 phase 1 (`3566186`..`9f0114b`) | The flat velocity task for the base Kangaroo, its reward retunes, PPO-like step logs, the docs page (section 4). |

`rl/supermimic` imports nothing from `tasks/` and reads no task attribute.
`pytest tests/supermimic`: 92 tests. One is flaky: `test_the_velocity_task_meets_the_contract`
fails about 2 runs in 3 on `reward/air_time` (1.3-1.5e-4 against a 1e-4 tolerance), most
likely the sole-contact sigmoid's 0.2 mm width amplifying float32 noise near touchdown. To fix.

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
  replay.py, runner.py    replay buffer; mjlab runner (collection, loop, step logs, checkpoints)
  state.py, names.py,     heading state; ImaginationCfg / SuperMimicEnvCfg;
  config.py, export.py    SuperMimicRunnerCfg; ONNX export
  COLAB.md                training on a Colab GPU (copy-paste cells)
utils/math/kinematics.py  TorchKinematics: differentiable FK (bodies, sites)
tasks/supermimic/         the Kangaroo motion-tracking task
tasks/supermimic_velocity/  the Kangaroo flat velocity task (M5)
docs/rl/supermimic.rst    the method, building a task, tuning vs PPO, logs
```

The robot contract (explicit, not task coupling): a floating base; joint-position actions
whose joints are the state's; FK bodies and sites listed in `ImaginationCfg` (FK over every
body costs ~50% more per imagined step); a closed chain's passive joints given as a function
(`ImaginationCfg.passive_joints`); MuJoCo position actuators (export); `auto_reset=False`
in training. Root height is relative to the env origin, not the terrain: to change for
rough terrain.

## 3. Key decisions (M2-M3)

- **Single definitions over twins** (M2b). We rejected twin dicts (two definitions per term,
  plus consistency tests) and an "imagined env" running mjlab's own terms on predicted
  state. Consequences: the IMU observations are the IMU site's FK (a predicted state has no
  sensors; FK matches the sensors to 4.8e-6, since mjlab runs `forward()` before
  observations); rewards see the command state the step targeted, observations the
  advanced one (mjlab updates the command after the rewards).
- **Performance.** A step's terms share one simulator batch (cached on the env step, qpos,
  qvel and `c`; its FK on qpos/qvel alone) and `Batch.memo` (e.g. a motion's reference
  bodies, once per batch). FK is launch-bound without `torch.compile` (~31 ms per call
  whatever the rows): one extra FK per reset costs ~1% of a training iteration.
- **Parity of the motion task** (seeded 100 iterations, `logs/rsl_rl/kangaroo_supermimic/`):
  M2a and M2b match M1 within 0-2% up to iteration 20 (the protocol, section 7); after
  iteration 50 runs spread widely from run to run. Reference runs:
  `2026-11-21_18-25-30_m2a_mine`, `2026-11-20_01-33-23_m1_mine`.

## 4. M5: locomotion

### Phase 1: flat-ground velocity tracking, base Kangaroo

Task `Mjlab-SuperMimic-Velocity-Flat-Pal-Kangaroo` (`tasks/supermimic_velocity/`). We built
it from the PPO task's `pal_kangaroo_flat_env_cfg()` (robot, sim, plane, actions, command
and its curriculum, resets, domain randomization) and changed what SuperMimic needs:

| Topic | What we did |
|---|---|
| Env | `SuperMimicEnvCfg`; `auto_reset` only in play; no pushes (to give to the world model later); random episode starts in training (`random_episode_start`), otherwise time-outs come in waves every ~400 iterations. |
| Command | Heading control off (we command the yaw rate); `VelocityCommandModel`: `c = {"command"}`, constant over the horizon (resampling every 3-8 s is not imagined), never `done`, termination reference = nominal height, upright. `step_metrics()` for the step logs. |
| Observations | Goal = the command; proprio = the PPO actor's terms and noise (IMU gyro and gravity from the IMU site's FK); critic = heading state, command, ξ. |
| ξ | Foot friction, encoder offsets, torso CoM, and the randomized joint friction loss, damping and armature (104 values, named parts). |
| Rewards | The PPO terms' formulas as single definitions: 10 kinematic terms, and the foot terms on FK: clearance, slip, and the stateful air time, swing height and soft landing (term memory). Foot height and velocity come from the foot sites; contact from the sole capsules' lowest point (`FOOT_CONTACT`, `FOOT_SOLES`). Soft landing uses the foot's vertical speed at touchdown in place of the force (weight x2000, not checked). We dropped `self_collisions` and `angular_momentum` (the passive linkage). Retuned weights: below. |
| Failures | All single definitions, so imagination ends episodes exactly where the sim does: `fell_over` (tilt, tested on its cosine, since acos' gradient is infinite upright), `base_too_low` (new: root < 0.4 m), and `illegal_contacts` made geometric: the knee linkage's capsules from FK with the passive joints, which matches the contact sensor exactly (360/360 contacts, no false alarm). |
| Critic | A target critic (Polyak 0.05) and a return bound (`rl_cfg.MAX_RETURN`, from the positive reward weights; 26 now). Without them the critic's values inflated themselves through bootstrapping (value 121 with negative rewards). |

The task meets `check_contract` (command exact, observations <= 4e-6, rewards <= 6e-6 before
the sole contact, failures exact). We also found that the PPO flat task does not build here
(its convex-hull reward plots to `/tmp`, its torque metric asks for the passive joints'
actuators); the SuperMimic task repoints that metric.

### What we tried, in order

**Laptop runs** (1024 envs, 512 rollouts; `logs/rsl_rl/kangaroo_supermimic_velocity/`):

| Run | Setup | Outcome |
|---|---|---|
| v1 | H 32, learned failure head | The imagined robot collapses unchecked; unbounded penalties explode. |
| v3 | H 32, tilt + low root analytic | The critic diverges from iteration ~45. |
| v4 | H 16 | Stable, but the policy kneels: the one failure only the learned head covered. |
| v5 | H 16, all failures analytic | The critic inflates itself (before the critic fix). |
| v6 | H 16, + critic fix | Stable; episodes up to 520-620 steps. |
| v7 | v6 + actor lr 1e-4 | Stable; episodes up to 945-970 of 1000 steps. Learned **to stand, not to walk**. |

v7's xy velocity error equalled standing still's (0.65 m/s). It was only 2M env steps;
PPO walks clearly after 1000 iterations x 4096 envs x 24 steps = ~98M. So we moved to a GPU.

**Colab runs** (A100, 4096 envs and rollouts, H 32, actor lr 1e-4, ~20M env steps per hour).
We evaluate checkpoints deterministically on fresh envs, comparing the xy error with the
standing-still error on the same commands.

| Run | What we changed | Outcome |
|---|---|---|
| `colab_h32_lr1e-4` | the PPO rewards | Stands at 250 and 500; moves at 750 (0.18 m/s) but not toward the command (0.670 vs 0.624). `action_noise 0.3` looked the same. |
| `track4_std1` | linear tracking weight 2 -> 4; tracking stds 0.5 / 0.71 -> 1.0 | **Walks** (2000 it), but does not track the command. |
| `gait_v2` | air time with soft window edges, 0.25 -> 0.5; a fine linear kernel (2, std 0.25) next to the wide one; foot clearance -2 -> -0.5 | Tracks at 3000 it, in odd postures. |
| `posture_v3` | pose 1 -> 2, plus `posture_l2` (-0.1) | Walks fine, but drags its right foot, tilted, not landing flat. |
| next | sole-capsule contact; foot clearance back to -2 | To run. |

What we learned (written up for users in `docs/rl/supermimic.rst`, Tuning):
- **The PPO rewards suit PPO, not gradients.** Standing earned ~5 of ~7 per step (yaw
  commands are small; upright and posture are full), and the std-0.5 tracking kernel is flat
  at standing errors, so there was no gradient toward the command. Wider kernels and heavier
  tracking fixed it.
- **Hard thresholds are invisible to imagination.** mjlab's air-time window has zero gradient;
  we gave it sigmoid edges (`edge_width` 0.01 s).
- **Kernels fade where the policy needs pulling back.** `posture_l2` is the pose kernel's
  exponent as a penalty. We checked `variable_posture` and `upright` against mjlab's (to
  1e-6) and left upright unchanged.
- **Proxies get exploited.** The first contact proxy read the foot site's height. A foot
  touching down on an edge of its sole has its site ~3 cm up, so the policy dragged a tilted
  foot that passed as airborne (air time 0.72 vs the sensor's 0.29, slip unpenalized). We
  now read the sole capsules' lowest point (threshold 0.2 mm): it agrees 100% with the
  sensor on the walking policy, and air time, slip and swing height match the sensor-based
  terms on both feet. Cost: +21 ms per eager FK + contact forward/backward on 4096 rows
  (laptop); less under compile.
- **The episode logs mislead here.** mjlab's episode logs are sums averaged over the
  episodes that ended (mostly falls, then synchronized time-outs). We added `Step_*` logs:
  means over all envs at every step (each reward term per second, terminations per minute,
  mean time to failure, the command's error vs standing still's).
- **The curriculum is keyed on env steps** (`common_step_counter`), so it matches PPO per
  sample: stage 2 (120k steps) is at PPO iteration 5000, SuperMimic iteration 48,000. All
  our runs stayed in stage 1 (vx ±1, yaw ±0.5).

### Next

1. Run the sole-contact version on Colab (`COLAB.md`) and check the gait (flat feet, no
   dragging) and the tracking error vs standing.
2. If feet still tilt: a flat-foot orientation penalty. If tracking stalls: H 48 as a
   separate experiment, or the per-joint posture stds.
3. Fix the flaky contract test (air-time tolerance or states away from the contact edge).
4. Log the world model's multi-step error on fresh data; compare seeds; compare tracking vs
   **env steps** with PPO's.
5. Fallback if it stops progressing: walking data from a PPO checkpoint in replay (mixed
   collection), or distillation first.
6. Later: scale the curriculum stages if we want faster commands within a run.

### Phase 2 (rough terrain, later)

Heightfield terrain query (`imagination.terrain_height`, used by the foot terms and capsule
bottoms); terrain features into the world model; terrain-relative root height; a learned
contact head where geometry is not enough; pushes into the world model.

## 5. Other milestones and open decisions

- **M4, WBC with SuperMimic**: a `CommandModel` for `tasks/WBC`'s `OnlineMotionCommand`
  (one motion pipeline) vs porting omm's command; give its adaptive sampler a `state_dict`;
  its rewards as single definitions (check angular momentum, CoP / CoM).
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
- **No Triton on Windows**, so no `torch.compile`: train with `--agent.compile False` (and
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
- Build the docs without touching `.venv`:
  `uvx --from "sphinx>=8.1.3" --with "sphinx-book-theme>=1.1.4" sphinx-build -b html src/pal_mjlab/docs <out dir>`.
- `test_play_launches` needs symlinks; it falls back to copying on Windows.
- We fixed an import-order trap (`pal_mjlab/__init__` imports mjlab first); keep it that way.

**Motion data (never commit data)**: the real library (`artifacts/kangaroo_grippers_library`,
from the SOMA retargeter) is not available. We built a stand-in from two DCMR LAFAN clips
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
2. Short seeded training (identical flags for every run we compare):
   ```
   PYTHONUTF8=1 MJLAB_WARP_QUIET=1 .venv/Scripts/train.exe Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers \
     --env.scene.num-envs 1024 --env.commands.motion.library-path <abs path to library> \
     --env.commands.motion.compile-kinematics False --agent.compile False \
     --agent.algorithm.policy.num-rollouts 512 --agent.seed 42 --agent.max-iterations 100 \
     --agent.save-interval 100000 --agent.log-interval 5 --agent.run-name <name>
   ```
3. Compare with `logs/rsl_rl/kangaroo_supermimic/2026-11-21_18-25-30_m2a_mine`:
   `.venv/Scripts/python.exe -I artifacts/tools/compare_runs.py <run A> <run B>`.
   Expect agreement within a few % up to ~iteration 20 (no bit-exactness; wide spread after
   ~50).
