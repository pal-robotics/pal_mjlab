"""Script to train RL agent with RSL-RL."""

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast
import torch

import tyro

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlBaseRunnerCfg
  registry_name: str | None = None
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  log_root: str = "logs/rsl_rl"
  """Root directory under which experiment logs are written."""
  torchrunx_log_dir: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  resume_actor_only: bool = True
  """When resuming, load only the actor from the checkpoint. The critic, optimizer
  state, and iteration counter are left fresh."""
  resume_read_onnx_metadata: bool = True
  """When resuming, also look for the ONNX export next to the checkpoint (or in the
  W&B run) and read its metadata."""
  resume_keep_onnx_metadata: list[str] = field(default_factory=list)
  """ONNX metadata keys to carry over from the resumed checkpoint's ONNX into every
  ONNX exported by this run (e.g. ['original_run_path']). Requires a runner that
  supports `inherited_onnx_metadata`."""
  freeze_actor_iterations: int = 0
  """Critic warm-up: keep the actor frozen for this many iterations after resuming.
  Counts toward agent.max_iterations. 0 disables."""
  freeze_actor_critic_lr: float = 1e-4
  """Fixed LR used during the warm-up phase (the adaptive schedule is disabled)."""
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def find_onnx_for_checkpoint(
  resume_path: Path, wandb_run_path: str | None = None
) -> Path | None:
  """Locate the ONNX file that belongs to a checkpoint, if any."""
  # 1) Same directory: model_4000.onnx, policy.onnx, or any single *.onnx.
  candidates = [resume_path.with_suffix(".onnx"), resume_path.parent / "policy.onnx"]
  for c in candidates:
    if c.exists():
      return c
  others = sorted(resume_path.parent.glob("*.onnx"))
  if others:
    if len(others) > 1:
      print(f"[WARN]: Multiple ONNX files found, using {others[0].name}.")
    return others[0]

  # 2) W&B run: download the matching .onnx next to the checkpoint.
  if wandb_run_path is not None:
    try:
      import wandb

      run = wandb.Api().run(str(wandb_run_path))
      onnx_files = [f for f in run.files() if f.name.endswith(".onnx")]
      if not onnx_files:
        return None
      preferred = [f for f in onnx_files if Path(f.name).stem == resume_path.stem]
      chosen = (preferred or onnx_files)[0]
      chosen.download(root=str(resume_path.parent), replace=True)
      return resume_path.parent / chosen.name
    except Exception as e:  # noqa: BLE001
      print(f"[WARN]: Could not fetch ONNX from W&B: {e}")
  return None


def read_onnx_metadata(onnx_path: Path) -> dict[str, str]:
  """Read the key/value metadata stored in an ONNX model."""
  import onnx

  model = onnx.load(str(onnx_path), load_external_data=False)
  return {p.key: p.value for p in model.metadata_props}

def _count_trainable(module) -> int:
  return sum(p.numel() for p in module.parameters() if p.requires_grad)


def set_actor_frozen(runner, frozen: bool) -> None:
  """Freeze/unfreeze every actor parameter (MLP weights and action-noise std)."""
  alg = runner.alg
  actor = alg.actor if hasattr(alg, "actor") else alg.policy.actor
  for p in actor.parameters():
    p.requires_grad_(not frozen)
  # Older rsl_rl keeps the action std on the policy object instead of the actor.
  if hasattr(alg, "policy"):
    for name in ("std", "log_std"):
      p = getattr(alg.policy, name, None)
      if isinstance(p, torch.nn.Parameter):
        p.requires_grad_(not frozen)
  print(
    f"[INFO]: actor {'FROZEN' if frozen else 'UNFROZEN'} "
    f"(trainable actor params: {_count_trainable(actor)})"
  )
  if frozen:
    assert _count_trainable(actor) == 0, "Actor still has trainable params."
    critic = alg.critic if hasattr(alg, "critic") else alg.policy.critic
    assert _count_trainable(critic) > 0, "Critic has no trainable params."

def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  registry_name: str | None = None

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task:
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check if motion_file is already set (e.g., via CLI --env.commands.motion.motion-file).
    if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
      print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")
    elif cfg.registry_name:
      # Download from WandB registry.
      registry_name = cast(str, cfg.registry_name)
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      raise ValueError(
        "For tracking tasks, provide either:\n"
        "  --registry-name your-org/motions/motion-name (download from WandB)\n"
        "  --env.commands.motion.motion-file /path/to/motion.npz (local file)"
      )

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.agent.resume:
    if cfg.wandb_run_path is not None:
      # Load checkpoint from W&B.
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      if rank == 0:
        run_id = resume_path.parent.name
        checkpoint_name = resume_path.name
        cached_str = "cached" if was_cached else "downloaded"
        print(
          f"[INFO]: Loading checkpoint from W&B: {checkpoint_name} "
          f"(run: {run_id}, {cached_str})"
        )
    else:
      # Load checkpoint from local filesystem.
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  runner_kwargs = {}
  if is_tracking_task:
    runner_kwargs["registry_name"] = registry_name

  # Write config files before runner creation, since the runner mutates agent_cfg
  # in-place (e.g., injecting non-serializable objects).
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  add_wandb_tags(cfg.agent.wandb_tags)
  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if cfg.resume_actor_only:
      print("[INFO]: Resuming actor only (critic/optimizer/iteration not loaded).")
      runner.load(
        str(resume_path),
        load_cfg={
          "actor": True,
          "critic": False,
          "optimizer": False,
          "iteration": False,
          "rnd": False,
        },
      )
    else:
      runner.load(str(resume_path))

    # Read the ONNX metadata that belongs to the resumed checkpoint.
    onnx_metadata: dict[str, str] | None = None
    if cfg.resume_read_onnx_metadata or cfg.resume_keep_onnx_metadata:
      onnx_path = find_onnx_for_checkpoint(resume_path, cfg.wandb_run_path)
      if onnx_path is None:
        print("[WARN]: No ONNX file found for this checkpoint; skipping metadata.")
      else:
        onnx_metadata = read_onnx_metadata(onnx_path)
        print(f"[INFO]: Read ONNX metadata from {onnx_path} ({len(onnx_metadata)} keys)")
        for k, v in onnx_metadata.items():
          print(f"  {k}: {v if len(v) < 120 else v[:117] + '...'}")
        if rank == 0:
          dump_yaml(log_dir / "params" / "resumed_onnx_metadata.yaml", onnx_metadata)
        # TODO: validate onnx_metadata against cfg.env here
        # (e.g. joint order, action scale, PD gains) and warn on mismatch.

    # Hand the requested keys to the runner so every exported ONNX keeps them.
    if onnx_metadata is not None and cfg.resume_keep_onnx_metadata:
      kept = {
        k: onnx_metadata[k] for k in cfg.resume_keep_onnx_metadata if k in onnx_metadata
      }
      missing = [k for k in cfg.resume_keep_onnx_metadata if k not in onnx_metadata]
      if missing:
        print(f"[WARN]: Requested metadata keys not in previous ONNX: {missing}")
      if not hasattr(runner, "inherited_onnx_metadata"):
        print(
          f"[WARN]: {type(runner).__name__} does not support "
          "'inherited_onnx_metadata'; kept keys will NOT be written."
        )
      print(f"[INFO]: Keeping ONNX metadata from previous run: {list(kept)}")
      runner.inherited_onnx_metadata = kept  # picked up by the runner's save()

  n_freeze = cfg.freeze_actor_iterations if resume_path is not None else 0
  if n_freeze > 0:
    alg = runner.alg
    saved = (alg.schedule, alg.learning_rate)

    # With the adaptive schedule, a frozen actor gives KL ~ 0, so rsl_rl would
    # ramp the LR up every step and blow up the critic. Pin it during warm-up.
    alg.schedule = "fixed"
    alg.learning_rate = cfg.freeze_actor_critic_lr
    for g in alg.optimizer.param_groups:
      g["lr"] = cfg.freeze_actor_critic_lr

    set_actor_frozen(runner, True)
    actor_before = [p.detach().clone() for p in (alg.actor if hasattr(alg, "actor") else alg.policy.actor).parameters()]

    runner.learn(num_learning_iterations=n_freeze, init_at_random_ep_len=True)

    actor_now = list((alg.actor if hasattr(alg, "actor") else alg.policy.actor).parameters())
    assert all(torch.equal(b, a) for b, a in zip(actor_before, actor_now)), (
      "Actor changed during the frozen phase!"
    )

    set_actor_frozen(runner, False)
    alg.schedule, alg.learning_rate = saved
    for g in alg.optimizer.param_groups:
      g["lr"] = alg.learning_rate

  runner.learn(
    num_learning_iterations=max(cfg.agent.max_iterations - n_freeze, 0),
    init_at_random_ep_len=(n_freeze == 0),
  )

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)

  # Create log directory once before launching workers.
  log_root_path = (Path(args.log_root) / args.agent.experiment_name).resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir)


def main():
  maybe_print_top_level_help("train")

  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()