# SuperMimic on Kangaroo

SuperMimic (arXiv 2610.28480) on PAL Kangaroo with 7-DoF arms, trained on the BONES-SEED retarget in
`~/Projects/supermimic/kangaroo_bones_seed`.

- [ROADMAP.md](ROADMAP.md): milestones and their gates
- [DECISIONS.md](DECISIONS.md): what we chose, why, and when to revisit (D0–D20)
- [LEARNINGS.md](LEARNINGS.md): facts and measurements
- `m0/`: raw benchmark outputs

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

# Benchmarks (GPU, through the queue)
gpu-run --label sm-sim --max 30m -- .venv/bin/python -m pal_mjlab.scripts.supermimic.bench_sim \
  --library ~/Projects/supermimic/kangaroo_bones_seed/data/library/smoke --out-json docs/supermimic/m0/bench_sim.json
gpu-run --label sm-bptt --max 45m -- .venv/bin/python -m pal_mjlab.scripts.supermimic.bench_bptt \
  --out-json docs/supermimic/m0/bench_bptt_eager.json
```
