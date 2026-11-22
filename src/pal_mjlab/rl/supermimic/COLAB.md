# Training the SuperMimic velocity task on Google Colab

Copy each block into its own Colab cell, in order. Runtime: **Runtime > Change runtime
type > GPU** (A100 or L4; a T4's 16 GB is tight for 4096 imagined rollouts).

## 1. Clone the branch

```python
!git clone --depth 1 -b MAC/wbc-supermimic https://github.com/pal-robotics/pal_mjlab.git /content/pal_mjlab
%cd /content/pal_mjlab
!git log --oneline -1
```

## 2. Install (uv; Linux gets CUDA torch from PyPI)

```python
!pip install -q uv
!uv sync --extra cu128
!uv run python -c "import torch, mujoco_warp; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. Optional: save the runs to Google Drive (Colab sessions end)

```python
from google.colab import drive
drive.mount("/content/drive")
!mkdir -p /content/drive/MyDrive/supermimic_logs
!mkdir -p logs/rsl_rl && ln -sfn /content/drive/MyDrive/supermimic_logs logs/rsl_rl/kangaroo_supermimic_velocity
```

## 4. Train in the background

The paper's setting (horizon 32, 4096 imagined rollouts, `torch.compile`, which works on
Linux) with the actor learning rate halved (the laptop runs showed it delays the world
model's exploitation). One iteration collects 4096 envs x 2.5 steps = ~10k env steps: 2000
iterations are ~20M env steps (PPO walks clearly at ~98M).

```python
!nohup uv run train Mjlab-SuperMimic-Velocity-Flat-Pal-Kangaroo \
  --env.scene.num-envs 4096 \
  --agent.algorithm.policy.num-rollouts 4096 \
  --agent.algorithm.policy.learning-rate 1e-4 \
  --agent.seed 42 --agent.max-iterations 2000 \
  --agent.save-interval 250 --agent.log-interval 10 \
  --agent.run-name colab_h32_lr1e-4 > /content/train.log 2>&1 &
```

Follow the console (re-run the cell to refresh):

```python
!grep "SuperMimic\] it" /content/train.log | tail -5; tail -3 /content/train.log; nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv
```

The first iterations are slow while `torch.compile` compiles. If memory runs out, lower
`--agent.algorithm.policy.num-rollouts` (2048).

## 5. Tensorboard

```python
%load_ext tensorboard
%tensorboard --logdir /content/pal_mjlab/logs/rsl_rl/kangaroo_supermimic_velocity
```

What to read (the details are in `ROADMAP.md`, M5):

- `env/Episode_Reward/track_linear_velocity` (per second of a 20 s episode, weight 2):
  standing still scores ~0.3-0.5, real tracking approaches 2. The laptop policies stood
  still; this is the curve that tells walking.
- `sim/reward` vs `policy/imagined_reward`: when the imagined reward rises above the real
  one, the policy is exploiting the world model.
- `policy/value` vs `policy/lambda_return`: they should track each other, under the
  return bound (15).
- `episode/*` only averages the episodes that ended in each window; episode starts are
  randomized, so time-outs (20 s) spread over training.

## 6. Get a checkpoint back

```python
!ls logs/rsl_rl/kangaroo_supermimic_velocity/*/
from google.colab import files
# files.download("logs/rsl_rl/kangaroo_supermimic_velocity/<run>/model_2000.pt")
```

To watch it on the laptop (Windows, no Triton):

```powershell
$env:PYTHONUTF8=1; $env:MJLAB_WARP_QUIET=1; $env:TORCHDYNAMO_DISABLE=1; .\.venv\Scripts\play.exe Mjlab-SuperMimic-Velocity-Flat-Pal-Kangaroo --checkpoint-file <path to model_N.pt> --viewer native --device cuda --num-envs 4
```
