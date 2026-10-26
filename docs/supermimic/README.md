# SuperMimic on Kangaroo

SuperMimic (arXiv 2610.28480) on PAL Kangaroo with 7-DoF arms, trained on the BONES-SEED retarget in
`~/Projects/supermimic/kangaroo_bones_seed`. A second variant, 5-DoF arms with Seed Robotics hands
(unactuated), has its own tasks (`...-Kangaroo-Hands`, D42) and retarget (`dataset_5dof`).

- [ROADMAP.md](ROADMAP.md): milestones and their gates
- [DECISIONS.md](DECISIONS.md): what we chose, why, and when to revisit (D0–D43)
- [LEARNINGS.md](LEARNINGS.md): facts and measurements
- `m0/`: raw benchmark outputs
- `m1/`: gain A/B (`gain_ab.md`), PPO-Sim baseline (`ppo_sim_baseline.md`), action-scale note
- `m2/`: offline world model: `world_model.md` (short), `report.pdf` (full write-up), `make_figures.py`
- `m3/`: imagination and policy-update checks: `imagination.md` (short), `report.pdf`, `make_figures.py`
- `m4/`: the online loop on `dev` vs PPO-Sim: `online_loop.md` (short), `report.pdf`, `make_figures.py`;
  the 24 h cluster runs with the paper's settings and the M5 options: `cluster_runs.md`,
  `cluster_report.pdf`, `make_cluster_figures.py`; the stability study: `stability.md`,
  `stability_report.pdf`, `make_stability_figures.py`
- `m5/`: the 24 h `full` and `dev` runs with the D39 defaults: `runs.md`

## Environment

```bash
uv sync --extra cu128 --group supermimic  # plain `uv sync` drops torch-cu128 and pyarrow
```

## Train SuperMimic with the mjlab CLI (M4, any mjlab machine)

```bash
# 1. Code and environment
git clone -b omm/supermimic git@github.com:pal-robotics/pal_mjlab.git && cd pal_mjlab
uv sync --extra cu128 --group supermimic

# 2. Data: the dataset folder of the BONES-SEED retarget (kangaroo_bones_seed/dataset, README
#    inside): retargeted motions (motions/, .pkl) and packed training libraries (libraries/<subset>/).
rsync -a --info=progress2 <host>:Projects/supermimic/kangaroo_bones_seed/dataset/ ../dataset/
#    (or download from the motion browser: the motions zip, or one library's three files)

# 3. Train. TASK=Mjlab-SuperMimic-Flat-Pal-Kangaroo-Grippers-ArmWaist1Hz (the gains of PPO-Sim).
#    --env.commands.motion.dataset may be relative. Subset: smoke | dev (2.5 h) | full | hours
#    (1, 10h, 40h: stratified by category, nested); a subset not yet in libraries/ is packed on
#    first use. Logs and checkpoints: logs/rsl_rl/kangaroo_supermimic/<run>/.
#    a) dev, local 16 GB GPU (D16: bf16 learning updates, replay of 500 steps per env)
uv run python -m pal_mjlab.scripts.supermimic.train $TASK --env.commands.motion.dataset ../dataset \
  --env.commands.motion.subset dev --agent.algorithm.precision bf16 --agent.replay-steps 500 \
  --agent.run-name dev-local
#    b) dev, cluster: the paper's settings (TF32, replay of 1000 steps per env)
uv run python -m pal_mjlab.scripts.supermimic.train $TASK --env.commands.motion.dataset ../dataset \
  --env.commands.motion.subset dev --agent.run-name dev-paper
#    c) full (250.9 h), cluster: the paper's settings; the library takes 13.9 GB of GPU memory
uv run python -m pal_mjlab.scripts.supermimic.train $TASK --env.commands.motion.dataset ../dataset \
  --env.commands.motion.subset full --agent.run-name full-paper
#    Cluster driver older than CUDA 12.4 (Leonardo: 535 = 12.2): mjlab disables CUDA graphs and
#    the simulator runs about 8x slower. Load the forward-compatibility libcuda first, and check
#    the log has no "CUDA Graphs disabled" line:
#      module load cuda/12.6 && export LD_LIBRARY_PATH=$CUDA_HOME/compat:$LD_LIBRARY_PATH
#    scripts.supermimic.train is mjlab's train that resumes from the newest model_*.pt of the
#    runs with the same --agent.run-name (preemptions: rerun the same command); plain
#    `uv run train $TASK ...` always starts fresh.

# Hands (5-DoF arms, Seed Robotics hands; D42): the same commands with
#   TASK=Mjlab-SuperMimic-Flat-Pal-Kangaroo-Hands-ArmWaist1Hz and the 5-DoF dataset folder
#   (kangaroo_bones_seed/dataset_5dof); logs in logs/rsl_rl/kangaroo_hands_supermimic/.

# Ablations (paper Fig. 4): --agent.algorithm.policy.use-critic False
#   --agent.algorithm.policy.horizon 16    --agent.algorithm.policy.termination rule (D31)
# mjlab CLI syntax: tuples comma-separated (--agent.algorithm.world-model.hidden 1024,1024,1024,
#   one element: 256,), booleans as --flag False.

# Watch, play, benchmark
tensorboard --logdir logs/rsl_rl/kangaroo_supermimic
#    a policy on a library (a downloaded one, a subset of the dataset, or one motion of either;
#    D43): mjlab's play plus --library and --clips (mjlab's play takes no env overrides)
uv run python -m pal_mjlab.scripts.supermimic.play $TASK --checkpoint-file <run>/model_N.pt \
  --library ../dataset/libraries/full --clips 230324/flip_360_001__A304
#    the motions themselves, as retargeted (kinematic replay, no policy; MuJoCo viewer)
uv run python -m pal_mjlab.scripts.supermimic.view_motion ../dataset/libraries/dev \
  --clips 230324/flip_360_001__A304
uv run python -m pal_mjlab.scripts.supermimic.eval --task $TASK \
  --checkpoints <run>/model_N.pt --out-dir results/   # every clip of the subset once (paper F.1)
```

Key TensorBoard scalars: `episode/return`, `eval/return_gap_head` (imagined vs sim, D31),
`termination/head_auc` vs `termination/rule_auc`, `world_model/loss`, `policy/objective`,
`time/*`, `memory/max_allocated_gb`.

## Commands

```bash
# Motion library: smoke (0.25 h), dev (2.5 h), full (250.9 h) or hours (10h). Writes to
# <data-dir>/libraries/<subset> (default data dir: kangaroo_bones_seed/dataset; the training
# CLI also packs a missing subset on first use).
.venv/bin/python -m pal_mjlab.scripts.bones_to_library --subset dev

# Tests (CPU, ~35 s)
.venv/bin/python -m pytest tests/supermimic -q

# PPO-Sim baseline (D17): resumes from the latest checkpoint if the job is preempted and rerun
gpu-run --detach --label sm-ppo --max 12h -- .venv/bin/python -m pal_mjlab.scripts.supermimic.train \
  Mjlab-SuperMimic-PPO-Flat-Pal-Kangaroo-Grippers
# other library: --env.commands.motion.library-path <folder>; logs: logs/rsl_rl/kangaroo_supermimic_ppo

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
