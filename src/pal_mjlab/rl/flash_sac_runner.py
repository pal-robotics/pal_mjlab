from __future__ import annotations

import os
import torch
from pathlib import Path

from pal_mjlab.rl.flash_sac.runner import FlashSACRunner, FlashSACRunnerCfg
from rsl_rl.env import VecEnv


class PalFlashSACRunner(FlashSACRunner):
    """FlashSAC runner that persists environment state across checkpoints."""

    def __init__(
        self,
        env: VecEnv,
        cfg: FlashSACRunnerCfg,
        log_dir: str | None = None,
        device: str = "cuda",
    ) -> None:
        super().__init__(env, cfg, log_dir, device)

    # -------------------------
    # SAVE
    # -------------------------
    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)

        # ✅ delegate ALL model state to agent
        self.agent.save(path)

        # runner metadata only
        torch.save(
            {
                "interaction_step": self.current_interaction_step,
            },
            os.path.join(path, "runner_state.pt"),
        )

    # -------------------------
    # LOAD
    # -------------------------
    def load(self, path: str) -> None:
        self.agent.load(path)

        runner_state_path = os.path.join(path, "runner_state.pt")
        if os.path.exists(runner_state_path):
            state = torch.load(runner_state_path, map_location=self.device)
            self.current_interaction_step = state["interaction_step"]

    # -------------------------
    # ENV STATE (optional)
    # -------------------------
    def save_env_state(self, path: str) -> None:
        torch.save(
            {
                "common_step_counter": self.env.unwrapped.common_step_counter,
            },
            os.path.join(path, "env_state.pt"),
        )

    def load_env_state(self, path: str) -> None:
        state_path = os.path.join(path, "env_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            self.env.unwrapped.common_step_counter = state["common_step_counter"]

    # -------------------------
    # ONNX EXPORT (FIXED)
    # -------------------------
    def export_policy_to_onnx(
        self, path: str, filename: str = "policy.onnx", verbose: bool = False
    ) -> None:
        actor = self.agent._actor.network  # stable entry point

        actor.eval()
        actor.to("cpu")

        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)

        dummy = torch.zeros(1, self.agent._actor_observation_dim)

        torch.onnx.export(
            actor,
            dummy,
            save_path,
            export_params=True,
            opset_version=18,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes=None,
            dynamo=False,
            verbose=verbose,
        )

        print(f"[PalFlashSACRunner] Exported ONNX to {save_path}")

    def get_inference_policy(self, device: str = None):
        device = device or self.device

        def policy(obs):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)

            prev_transition = {"next_observation": obs_t}

            with torch.no_grad():
                actions = self.agent.sample_actions(
                    interaction_step=self.current_interaction_step,
                    prev_transition=prev_transition,
                    training=False,
                )
            return actions

        return policy