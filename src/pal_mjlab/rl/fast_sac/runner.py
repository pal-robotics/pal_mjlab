"""FastSAC runner.

Extends MjlabOnPolicyRunner to handle FastSAC-specific concerns:
- Overrides learn() to remove PPO-specific rnd_cfg references
- No actor/critic model configs (FastSAC constructs them internally)
- ONNX export alongside checkpoints
"""

from __future__ import annotations

import os
import time
from copy import deepcopy

import torch
import wandb
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper


# NOTE: We hijack the on-policy runner, even though FastSAC is off-policy!
# The runner is responsible for data collection and logging, we let the algororithm implementation handle the
# replay buffer and training updates internally.
class FastSACRunner(MjlabOnPolicyRunner):
  """Runner for FastSAC that adapts the on-policy loop for off-policy SAC."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    **kwargs,
  ) -> None:
    del kwargs
    cfg = deepcopy(train_cfg)
    cfg.setdefault("algorithm", {})
    # RSL-RL's logger expects this key to exist.
    cfg["algorithm"].setdefault("rnd_cfg", None)
    # Skip MjlabOnPolicyRunner.__init__ since we already deepcopied and
    # FastSAC doesn't use actor/critic model configs. Call OnPolicyRunner
    # directly.
    OnPolicyRunner.__init__(self, env, cfg, log_dir, device)

  def learn(
    self,
    num_learning_iterations: int,
    init_at_random_ep_len: bool = False,
  ) -> None:
    """Training loop adapted for FastSAC (off-policy)."""
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf,
        high=int(self.env.max_episode_length),
      )

    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()

    if self.is_distributed:
      print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
      self.alg.broadcast_parameters()

    self.logger.init_logging_writer()

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      start = time.time()

      # Collect one step (num_steps_per_env=1 for off-policy)
      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          actions = self.alg.act(obs)
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          obs, rewards, dones = (
            obs.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
          )
          self.alg.process_env_step(obs, rewards, dones, extras)
          self.logger.process_env_step(rewards, dones, extras, None)

        collect_time = time.time() - start
        start = time.time()

        # No-op for SAC but keeps the interface consistent
        self.alg.compute_returns(obs)

      # Perform SAC gradient updates from replay buffer
      loss_dict = self.alg.update()

      learn_time = time.time() - start
      self.current_learning_iteration = it

      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=None,
      )

      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

    if self.logger.writer is not None:
      self.save(
        os.path.join(
          self.logger.log_dir,
          f"model_{self.current_learning_iteration}.pt",
        )
      )
      self.logger.stop_logging_writer()

  def save(self, path: str, infos=None) -> None:
    """Save checkpoint with environment state and ONNX export."""
    super().save(path, infos)

    # Export ONNX alongside the checkpoint
    if self.logger.logger_type in ["wandb"] and wandb.run:
      policy_dir = os.path.dirname(path)
      filename = os.path.basename(policy_dir) + ".onnx"
      try:
        self.export_policy_to_onnx(policy_dir, filename)
        run_name: str = wandb.run.name  # type: ignore[assignment]
        metadata = get_base_metadata(self.env.unwrapped, run_name)
        onnx_path = os.path.join(policy_dir, filename)
        attach_metadata_to_onnx(onnx_path, metadata)
        wandb.save(onnx_path, base_path=policy_dir)
      except Exception as e:
        # ONNX export is best-effort; don't fail the checkpoint
        print(e)
        pass

  def get_inference_policy(self, device: str | None = None):
    """Return a deterministic FastSAC policy callable for play."""
    self.alg.eval_mode()
    device = device or self.device

    actor = self.alg.get_policy().to(device)
    obs_normalizer = self.alg.obs_normalizer.to(device)
    obs_normalizer.eval()
    obs_group_names = tuple(getattr(self.alg, "_actor_obs_group_names", ("actor",)))

    def _normalize(obs: torch.Tensor) -> torch.Tensor:
      try:
        return obs_normalizer(obs, update=False)
      except TypeError:
        return obs_normalizer(obs)

    @torch.no_grad()
    def policy(obs: TensorDict) -> torch.Tensor:
      flat_obs = torch.cat([obs[name] for name in obs_group_names], dim=-1).to(device)
      norm_obs = _normalize(flat_obs)
      return actor.explore(norm_obs, deterministic=True)

    return policy

  def export_policy_to_onnx(self, path, filename = "policy.onnx", verbose = False):
    #super().export_policy_to_onnx(path, filename, verbose)     # DO NOT CALL SUPER, SINCE IT COMES FROM ON_POLICY_RUNNER
    onnx_model = self.alg.get_policy().as_onnx(verbose, self.alg.actor_obs_groups_dict, self.alg.obs_dict)

    onnx_model.to("cpu")
    onnx_model.eval()

    if not os.path.exists(path):
      os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)

    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),  # type: ignore
      save_path,
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=onnx_model.input_names,  # type: ignore
      output_names=onnx_model.output_names,  # type: ignore
      dynamic_axes={},
      dynamo = False,
    )

    return 