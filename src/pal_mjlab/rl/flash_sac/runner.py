from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict, fields
from typing import Any, Literal, Tuple

import torch

from pal_mjlab.rl.flash_sac.agent import FlashSACAgent, FlashSACConfig
from rsl_rl.utils.logger import Logger
from mjlab.utils.spaces import Box


@dataclass
class FlashSACRunnerCfg:
    # --- Experiment ---
    experiment_name: str = "kangaroo_velocity"
    save_interval: int = 500
    max_interaction_steps: int = 30_000

    # --- Eval / logging cadence ---
    evaluation_per_interaction_step: int = 10_000
    metrics_per_interaction_step: int = 1_000
    recording_per_interaction_step: int = 10_000
    logging_per_interaction_step: int = 500
    save_buffer_per_interaction_step: int = 0       # 0 = disabled

    num_eval_episodes: int = 5
    num_record_episodes: int = 1

    # --- Agent ---
    seed: int = 42
    device_type: str = "cuda"
    normalize_reward: bool = True
    normalized_G_max: float = 100.0
    asymmetric_observation: bool = True

    # --- Replay buffer ---
    buffer_max_length: int = 1_000_000
    buffer_min_length: int = 10_000
    buffer_device_type: str = "cuda"
    sample_batch_size: int = 256

    # --- Learning rate schedule ---
    learning_rate_init: float = 1e-4
    learning_rate_peak: float = 3e-4
    learning_rate_end: float = 1e-5
    learning_rate_warmup_rate: float = 0.01
    learning_rate_warmup_step: int = 10_000
    learning_rate_decay_rate: float = 0.5
    learning_rate_decay_step: int = 500_000

    # --- Actor network ---
    actor_num_blocks: int = 3
    actor_hidden_dim: int = 512
    actor_bc_alpha: float = 0.0
    actor_noise_zeta_mu: float = 1.0
    actor_noise_zeta_max: int = 10
    actor_update_period: int = 2

    # --- Critic network ---
    critic_num_blocks: int = 3
    critic_hidden_dim: int = 512
    critic_num_bins: int = 51
    critic_min_v: float = -200.0
    critic_max_v: float = 200.0
    critic_target_update_tau: float = 0.005

    # --- Temperature / entropy ---
    temp_initial_value: float = 1.0
    temp_target_sigma: float = 0.5
    temp_target_entropy: float = 0.0           # overridden at agent init

    # --- RL ---
    gamma: float = 0.99
    n_step: int = 3
    updates_per_interaction_step: float = 1.0

    # --- torch.compile ---
    use_compile: bool = True
    compile_mode: str = "auto"
    use_amp: bool = True

    # --- Checkpoint loading ---
    load_optimizer: bool = True
    load_reward_normalizer: bool = True

    # --- mjlab_bridge
    run_name: str = ""
    resume:bool = False
    clip_actions: float | None = None
    wandb_project: str = "mjlab"
    wandb_tags: Tuple[str, ...] = ()
    



class FlashSACRunner:

    def __init__(
        self,
        env,
        cfg: FlashSACRunnerCfg | dict,   # accept both
        log_dir: str | None = None,
        device: str = "cuda",
    ) -> None:
        self.env = env
        self.device = device

        # Normalize: if a dict was passed (e.g. from asdict() in the train script),
        # reconstruct the dataclass so attribute access is consistent.
        if isinstance(cfg, dict):
            # _configure_multi_gpu mutates cfg, so keep it as a dict internally
            # OR convert and handle mutation separately — easiest: keep as dataclass
            # but store multi_gpu separately.

            known_keys = {f.name for f in fields(FlashSACRunnerCfg)}
            filtered = {k: v for k, v in cfg.items() if k in known_keys}
            cfg = FlashSACRunnerCfg(**filtered)
        
        self.cfg = cfg
        self._configure_multi_gpu()

        self.log_dir = log_dir

        self.current_interaction_step = 0

        obs_dim = (
            env.unwrapped.single_observation_space.spaces["actor"].shape[0] +
            env.unwrapped.single_observation_space.spaces["critic"].shape[0]
            if cfg.asymmetric_observation
            else env.unwrapped.single_observation_space.spaces["actor"].shape[0]
        )
        observation_space = Box(
            low=float("-inf"), high=float("inf"), shape=(obs_dim,), dtype="float32"
        )
        action_space = env.action_space

        env.reset()

        if cfg.asymmetric_observation:
            actor_observation_size = env.unwrapped.single_observation_space.spaces["actor"].shape
        else:
            actor_observation_size = env.unwrapped.single_observation_space.shape

        env_info = {
            "actor_observation_size": actor_observation_size,
        }

        # Build FlashSACConfig from runner cfg fields
        agent_cfg = FlashSACConfig(**{
            f.name: getattr(cfg, f.name) for f in fields(FlashSACConfig)
        })

        self.agent = FlashSACAgent(
            observation_space=observation_space,
            action_space=action_space,
            env_info=env_info,
            cfg=agent_cfg,
        )

    def learn(self) -> None:
        observations, _ = self.env.reset()
        print(type(observations), observations.keys())
        print(observations["actor"].shape)

        transition = None
        update_counter = 0.0
        update_info: dict[str, Any] = {}

        save_path_base = os.path.join(self.log_dir or ".", self.cfg.experiment_name)
        os.makedirs(save_path_base, exist_ok=True)

        start_step = self.current_interaction_step
        total_steps = start_step + self.cfg.max_interaction_steps

        import tqdm
        for interaction_step in tqdm.tqdm(
            range(start_step + 1, total_steps + 1),
            smoothing=0.1,
            mininterval=0.5,
        ):
            env_step = interaction_step * self.env.num_envs

            with torch.no_grad():
            # --- collect action ---
                if self.agent.can_start_training() and transition is not None:
                    actions = self.agent.sample_actions(
                        interaction_step,
                        prev_transition=transition,
                        training=True,
                    )
                else:
                    action_dim = self.env.unwrapped.action_space.shape[1]
                    actions = torch.zeros(self.env.num_envs, action_dim, device=self.device)

            actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)

            # --- step env ---
            next_observations, rewards, dones, env_infos = self.env.step(actions)
            truncateds = env_infos.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))
            terminateds = dones.bool() & ~truncateds

            # For done envs the buffer needs the terminal obs, not the post-reset obs.
            # env.step() already patches next_observations with post-reset obs for done
            # envs, so we read terminal obs from extras if available, otherwise fall back.
            next_buffer_observations = next_observations.clone()
            if "terminal_obs" in env_infos:
                done_envs = (terminateds | truncateds).nonzero(as_tuple=False).flatten()
                if len(done_envs) > 0:
                    next_buffer_observations["actor"][done_envs] = env_infos["terminal_obs"]["actor"][done_envs]
                    next_buffer_observations["critic"][done_envs] = env_infos["terminal_obs"]["critic"][done_envs]
                    

            # --- build transition ---
            # asymmetric: actor sees "actor" obs, critic sees full "critic" obs
            obs_for_buffer = (
                torch.cat([next_observations["actor"], next_observations["critic"]], dim=-1)
                if self.cfg.asymmetric_observation
                else next_observations["actor"]
            )
            prev_obs_for_buffer = (
                torch.cat([observations["actor"], observations["critic"]], dim=-1)
                if self.cfg.asymmetric_observation
                else observations["actor"]
            )
            next_buffer_obs_flat = (
                torch.cat([next_buffer_observations["actor"], next_buffer_observations["critic"]], dim=-1)
                if self.cfg.asymmetric_observation
                else next_buffer_observations["actor"]
            )

            transition = {
                "observation": prev_obs_for_buffer.cpu().numpy(),
                "action": actions.cpu().numpy(),
                "reward": rewards.cpu().numpy(),
                "terminated": terminateds.cpu().numpy(),
                "truncated": truncateds.cpu().numpy(),
                "next_observation": next_buffer_obs_flat.cpu().numpy(),
            }
            self.agent.process_transition(transition)
            transition["next_observation"] = obs_for_buffer.cpu().numpy()
            observations = next_observations

            if "episode_info" in env_infos:
                pass  # hook logger here

            # --- update ---
            if self.agent.can_start_training():
                update_counter += self.cfg.updates_per_interaction_step
                while update_counter >= 1:
                    update_info = self.agent.update()
                    update_counter -= 1

                # evaluation
                if (
                    self.cfg.evaluation_per_interaction_step
                    and interaction_step % self.cfg.evaluation_per_interaction_step == 0
                ):
                    self._evaluate()

                # logging
                if (
                    self.cfg.logging_per_interaction_step
                    and interaction_step % self.cfg.logging_per_interaction_step == 0
                ):
                    self._log(env_step, update_info)

                # checkpoint
                if interaction_step % self.cfg.save_interval == 0:
                    save_path = os.path.join(save_path_base, f"step_{interaction_step}")
                    self.save(save_path)

                # buffer checkpoint
                if (
                    self.cfg.save_buffer_per_interaction_step
                    and interaction_step % self.cfg.save_buffer_per_interaction_step == 0
                ):
                    save_path = os.path.join(save_path_base, f"step_{interaction_step}")
                    self.agent.save_replay_buffer(save_path)

            self.current_interaction_step = interaction_step

        # final save
        self.save(os.path.join(save_path_base, f"step_{self.current_interaction_step}_final"))

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        self.agent.save(path)
        torch.save({"interaction_step": self.current_interaction_step}, os.path.join(path, "runner_state.pt"))

    def load(self, path: str) -> None:
        self.agent.load(path)
        runner_state_path = os.path.join(path, "runner_state.pt")
        if os.path.exists(runner_state_path):
            state = torch.load(runner_state_path, map_location=self.device)
            self.current_interaction_step = state["interaction_step"]

    def load_replay_buffer(self, path: str) -> None:
        self.agent.load_replay_buffer(path)

    def _evaluate(self) -> None:
        # Hook: plug in your evaluate() call here
        pass

    def _log(self, env_step: int, update_info: dict[str, Any]) -> None:
        # Hook: plug in your logger here
        print(f"[step {env_step}] " + " | ".join(f"{k}: {v:.4f}" for k, v in update_info.items()))

    def _configure_multi_gpu(self) -> None:
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_config = None   # ← store on self, not cfg
            return

        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        self.multi_gpu_config = {          # ← same here
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }

        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # Validate multi-GPU configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # Initialize torch distributed
        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        # Set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)