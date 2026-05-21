from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from pal_mjlab.rl.flash_sac_runner import PalFlashSACRunner

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

torch.set_float32_matmul_precision("high")

@dataclass
class PlayFlashSACConfig:
    checkpoint_file: str
    device: str | None = None
    num_envs: int | None = None
    video: bool = False
    video_length: int = 200
    viewer: Literal["native", "viser"] = "native"
    no_terminations: bool = False


def run_play(task_id: str, cfg: PlayFlashSACConfig):
    configure_torch_backends()

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)

    if cfg.num_envs is not None:
        env_cfg.scene.num_envs = cfg.num_envs

    if cfg.no_terminations:
        env_cfg.terminations = {}

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    if cfg.video:
        env = VideoRecorder(
            env,
            video_folder=Path(cfg.checkpoint_file).parent / "videos" / "play",
            step_trigger=lambda step: step == 0,
            video_length=cfg.video_length,
            disable_logger=True,
        )

    # wrap env for RL
    from mjlab.rl import RslRlVecEnvWrapper
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # -------------------------
    # FLASH-SAC RUNNER
    # -------------------------
    runner = PalFlashSACRunner(env, agent_cfg, log_dir=None, device=device)

    ckpt_path = Path(cfg.checkpoint_file)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[INFO] Loading FlashSAC checkpoint: {ckpt_path}")

    runner.load(str(ckpt_path.parent))  # IMPORTANT: folder, not file

    policy = runner.get_inference_policy(device=device)

    # -------------------------
    # VIEWER
    # -------------------------
    if cfg.viewer == "native":
        NativeMujocoViewer(env, policy).run()
    elif cfg.viewer == "viser":
        ViserPlayViewer(env, policy).run()
    else:
        raise ValueError(cfg.viewer)

    env.close()


def main():
    import mjlab.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
    )

    cfg = tyro.cli(
        PlayFlashSACConfig,
        args=remaining,
        default=PlayFlashSACConfig(checkpoint_file=""),
    )

    run_play(chosen_task, cfg)


if __name__ == "__main__":
    main()