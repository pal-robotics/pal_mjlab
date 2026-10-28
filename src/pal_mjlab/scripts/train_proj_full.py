"""Script to train RL agent with RSL-RL (supports projector runners)."""

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import tyro

import mjlab
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
  pretrained_checkpoint: str | None = None
  """Full checkpoint of the base policy. Used by projector runners (via the
  PRETRAINED_CKPT env var) to supply the frozen backbone, critic and obs
  normalizer. Ignored by other runners."""

  # ------------------------------------------------------------------ projector
  # All of these are injected into the agent config dict right before the runner
  # is built, so VelocityProjectorOnPolicyRunner picks them up from train_cfg.
  # None means "leave unset" (the runner falls back to its own defaults).
  projector_obs_map: list[tuple[int, int]] | None = None
  """Explicit obs remap [(cur_idx, base_idx), ...]. Takes priority over the
  term-name based map below. Leave None to use the term lists."""
  projector_cur_terms: list[tuple[str, int]] | None = field(
    default_factory=lambda: [
      ("base_ang_vel", 3),
      ("joint_pos", 22),
      ("joint_vel", 22),
      ("actions", 22),
      ("command", 3),
      ("imu_projected_gravity", 3),
    ]
  )
  """Ordered (name, dim) obs terms of the CURRENT env (sums to 75)."""
  projector_base_terms: list[tuple[str, int]] | None = field(
    default_factory=lambda: [
      ("base_ang_vel", 3),
      ("joint_pos", 26),
      ("joint_vel", 26),
      ("actions", 22),
      ("command", 3),
      ("imu_projected_gravity", 3),
    ]
  )
  """Ordered (name, dim) obs terms of the BASE policy (sums to 83)."""
  # Terms with the same name but different dims (the robots have different joint
  # counts) cannot be matched automatically. For those, give (cur_local_idx,
  # base_local_idx) pairs, local to the term. Terms with equal dims (actions,
  # command, ...) are matched 1:1 in order. If "actions" is given here it is also
  # reused (reversed) as the output-head action map unless projector_act_map is set.

  projector_term_maps: dict[str, list[tuple[int, int]]] | None = None
  projector_act_map: list[tuple[int, int]] | None = None
  """Optional action remap [(backbone_idx, cur_idx), ...]. Only needed if the
  action ordering differs from the base policy."""

  resume_path: str | None = None
  """Explicit checkpoint to resume from: a full model_*.pt or a heads/model_*.pt.
  Overrides agent.resume / load_run / load_checkpoint / W&B loading."""
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
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def _is_projector_runner(runner_cls: type) -> bool:
  return hasattr(runner_cls, "_heads_path")


def _build_obs_map(cur_terms, base_terms, term_maps) -> list[tuple[int, int]]:
  """[(cur_idx, base_idx), ...] by term name. Terms present in both with equal
  dims are matched 1:1; an explicit per-term map overrides. Same-name terms with
  different dims are left unmapped and learned by the dense projection."""
  term_maps = term_maps or {}

  def offsets(terms):
    out, o = {}, 0
    for name, dim in terms:
      out[name] = (o, int(dim))
      o += int(dim)
    return out

  cur, base = offsets(cur_terms), offsets(base_terms)
  pairs: list[tuple[int, int]] = []
  for name, (co, cd) in cur.items():
    if name not in base:
      continue
    bo, bd = base[name]
    if name in term_maps:
      for c, b in term_maps[name]:
        if not (0 <= c < cd and 0 <= b < bd):
          raise ValueError(
            f"projector_term_maps['{name}']: pair ({c}, {b}) out of range "
            f"(cur dim {cd}, base dim {bd})"
          )
        pairs.append((co + c, bo + b))
    elif cd == bd:
      pairs += [(co + j, bo + j) for j in range(cd)]
    else:
      print(
        f"[INFO] Term '{name}' dim {cd} (current) vs {bd} (base): "
        f"not index-mapped, learned by the projection layer."
      )
  return pairs


def _projector_overrides(cfg: TrainConfig) -> dict:
  """Resolve the projector obs / action maps from the agent cfg into explicit
  index lists. Hidden sizes and freeze flag are read by the runner directly from
  train_cfg, so they are not injected here."""
  a = cfg.agent
  cur_terms = getattr(a, "projector_cur_terms", None)
  base_terms = getattr(a, "projector_base_terms", None)
  term_maps = getattr(a, "projector_term_maps", None)

  if (cur_terms is None) != (base_terms is None):
    raise ValueError(
      "Set both projector_cur_terms and projector_base_terms in the runner cfg "
      "(or neither)."
    )

  # Plain lists (not tuples) so the dict dumps cleanly to YAML.
  o: dict = {}

  obs_map = getattr(a, "projector_obs_map", None)  # explicit map wins
  if obs_map is None and cur_terms is not None:
    obs_map = _build_obs_map(cur_terms, base_terms, term_maps)
    cur_dim = sum(int(d) for _, d in cur_terms)
    base_dim = sum(int(d) for _, d in base_terms)  # type: ignore[union-attr]
    print(
      f"[INFO] Projector obs map: {len(obs_map)} dims remapped "
      f"(current obs {cur_dim}, base obs {base_dim}, "
      f"{base_dim - len(obs_map)} base inputs filled with normalizer mean)"
    )
  if obs_map is not None:
    o["projector_obs_map"] = [list(p) for p in obs_map]

  act_map = getattr(a, "projector_act_map", None)
  if act_map is None and term_maps and "actions" in term_maps:
    # Obs pairs are (cur, base); the output head wants (backbone_idx, cur_idx).
    act_map = [(b, c) for c, b in term_maps["actions"]]
  if act_map is not None:
    o["projector_act_map"] = [list(p) for p in act_map]

  return o


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

  # Runner comes from the task registry.
  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner
  is_projector = _is_projector_runner(runner_cls)
  print(f"[INFO] Runner: {runner_cls.__name__}")

  # Work out what we're resuming from (if anything) before touching the env.
  explicit_resume = (
    Path(cfg.resume_path).expanduser().resolve() if cfg.resume_path else None
  )
  if explicit_resume is not None and not explicit_resume.exists():
    raise FileNotFoundError(f"resume_path not found: {explicit_resume}")

  projector_overrides: dict = {}
  if is_projector:
    if cfg.pretrained_checkpoint:
      ckpt = Path(cfg.pretrained_checkpoint).expanduser().resolve()
      if not ckpt.exists():
        raise FileNotFoundError(f"pretrained_checkpoint not found: {ckpt}")
      os.environ["PRETRAINED_CKPT"] = str(ckpt)
      print(f"[INFO] Base policy: {ckpt}")
    elif not os.environ.get("PRETRAINED_CKPT"):
      raise ValueError(
        "Projector runner needs the base policy to build the frozen backbone, "
        "including when resuming.\n"
        "  --pretrained-checkpoint /path/to/model_XXXX.pt"
      )
    projector_overrides = _projector_overrides(cfg)
  # Non-projector runners simply never see the projector_* settings.
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
  if explicit_resume is not None:
    resume_path = explicit_resume
  elif cfg.agent.resume:
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

  # Inject projector settings so the runner reads them from train_cfg.
  # Done before the YAML dump so the exact remap used is recorded with the run.
  if projector_overrides:
    agent_cfg.update(projector_overrides)
    if rank == 0:
      print(f"[INFO] Projector settings: {projector_overrides}")

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
    kind = "heads" if resume_path.parent.name == "heads" else "full"
    print(f"[INFO]: Loading {kind} checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
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

    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY
      + ("MUJOCO*", "PRETRAINED_CKPT"),
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