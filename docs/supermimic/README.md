# SuperMimic on Kangaroo

SuperMimic (arXiv 2610.28480) on PAL Kangaroo with 7-DoF arms, trained on the BONES-SEED retarget in
`~/Projects/supermimic/kangaroo_bones_seed`.

- [ROADMAP.md](ROADMAP.md): milestones and their gates
- [DECISIONS.md](DECISIONS.md): what we chose, why, and when to revisit (D0–D31)
- [LEARNINGS.md](LEARNINGS.md): facts and measurements
- `m0/`: raw benchmark outputs
- `m1/`: gain A/B (`gain_ab.md`), PPO-Sim baseline (`ppo_sim_baseline.md`), action-scale note
- `m2/`: offline world model: `world_model.md` (short), `report.pdf` (full write-up), `make_figures.py`

## Environment

```bash
uv sync --extra cu128 --group supermimic  # plain `uv sync` drops torch-cu128 and pyarrow
```

## Commands

```bash
# Motion library: smoke (0.25 h), dev (2.5 h) or full (250.9 h).
# Writes to kangaroo_bones_seed/data/library/<subset>.
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

# Benchmarks (GPU, through the queue)
gpu-run --label sm-sim --max 30m -- .venv/bin/python -m pal_mjlab.scripts.supermimic.bench_sim \
  --library ~/Projects/supermimic/kangaroo_bones_seed/data/library/smoke --out-json docs/supermimic/m0/bench_sim.json
gpu-run --label sm-bptt --max 45m -- .venv/bin/python -m pal_mjlab.scripts.supermimic.bench_bptt \
  --out-json docs/supermimic/m0/bench_bptt_eager.json
```
