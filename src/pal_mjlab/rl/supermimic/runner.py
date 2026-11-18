from __future__ import annotations

import os
import time
import torch

from .supermimic_wrapper import SupermimicVecEnv
from .config import (
    SupermimicEncoderModelCfg, 
    SupermimicDecoderModelCfg, 
    SupermimicCriticModelCfg, 
    SupermimicRunnerCfg,
    SupermimicAlgorithmCfg,
)
from .models import (
    BaseModel,
    EncoderModel,
    DecoderModel,
    CriticModel,
)

class SupermimicRunner:
    """Runner for reinforcement learning algorithms using the supermimic training loop."""

    alg: None #REPLACE_WITH_SUPERMIMIC'S (BPTT)

    def __init__(self, env: SupermimicVecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Construct the runner, algorithm, and logging stack."""
        self.env = env
        self.cfg = train_cfg
        self.device = device

        # Setup multi-GPU training if enabled
        self._configure_multi_gpu()

        ###
        # TO BE IMPLEMENTED
        ###

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the learning loop for the specified number of iterations."""

        ###
        # TO BE IMPLEMENTED
        ###

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save the models and training state to a given path and upload them if external logging is used."""
        
        ###
        # TO BE IMPLEMENTED
        ###

    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict:
        """Load the models and training state from a given path."""

        ###
        # TO BE IMPLEMENTED
        ###

    def get_inference_policy(self, device: str | None = None) -> dict[str, BaseModel]:
        """Return the policy on the requested device for inference."""
        """With supermimic, will return encoders and the only decoder, separate."""

        ###
        # TO BE IMPLEMENTED
        ###

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        """Export the model to a Torch JIT file."""

        ###
        # TO BE IMPLEMENTED
        ###

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
        """Export the model into an ONNX file."""
        """With supermimic, encoders and decoder are separate models."""

        ###
        # TO BE IMPLEMENTED
        ###

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        """Register a repository path whose git status should be logged."""

        ###
        # TO BE IMPLEMENTED
        ###

    def _configure_multi_gpu(self) -> None:
        """Configure multi-gpu training."""
        # Check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # If not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.cfg["multi_gpu"] = None
            return

        # Get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # Make a configuration dictionary
        self.cfg["multi_gpu"] = {
            "global_rank": self.gpu_global_rank,  # Rank of the main process
            "local_rank": self.gpu_local_rank,  # Rank of the current process
            "world_size": self.gpu_world_size,  # Total number of processes
        }

        # Check if user has device specified for local rank
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
