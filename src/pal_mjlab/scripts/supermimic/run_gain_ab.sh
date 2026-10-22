#!/usr/bin/env bash
# Gain A/B (D27): one arm of the experiment = PPO-Sim for a fixed number of iterations,
# then the paper's benchmark on its checkpoints. Safe to rerun after preemption: training
# resumes from the latest checkpoint up to the same total, finished stages are skipped.
#
#   src/pal_mjlab/scripts/supermimic/run_gain_ab.sh <task> <tag> [total_iterations]
set -euo pipefail
cd "$(dirname "$0")/../../../.."

TASK=$1
TAG=$2
TOTAL=${3:-1000}
EXP=supermimic_gain_ab_$TAG
RUNS=logs/rsl_rl/$EXP
OUT=logs/supermimic_eval/gain_ab_$TAG
PY=.venv/bin/python

last_iteration() {
  ls "$RUNS"/*/model_*.pt 2>/dev/null | sed -E 's/.*model_([0-9]+)\.pt/\1/' | sort -n | tail -1
}

done_it=$(last_iteration || true)
done_it=${done_it:-0}
if [ "$done_it" -lt $((TOTAL - 1)) ]; then
  $PY -m pal_mjlab.scripts.supermimic.train "$TASK" \
    --agent.experiment-name "$EXP" \
    --agent.max-iterations $((TOTAL - done_it)) \
    --agent.save-interval 200
fi

checkpoints=()
for it in 200 400 600 800 $((TOTAL - 1)); do
  f=$(ls "$RUNS"/*/model_"$it".pt 2>/dev/null | tail -1 || true)
  [ -n "$f" ] && checkpoints+=("$f")
done
$PY -m pal_mjlab.scripts.supermimic.eval --task "$TASK" --checkpoints "${checkpoints[@]}" --out-dir "$OUT"
