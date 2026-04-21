"""Custom RL Model Architectures for Kangaroo Tracking."""

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules.mlp import MLP
from rsl_rl.modules.normalization import EmpiricalNormalization


class HistoryEncoderModel(MLPModel):
    """Custom model that extracts a history observation group and parses it through a 1D Conv."""

    def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs):
        """Initialize the HistoryEncoderModel.
        
        It assumes the environment observation dictionary contains:
        - `{obs_set}`: Main group (current observation, e.g., 'actor')
        - `{obs_set}_history`: History group (e.g., 'actor_history') with shape [Batch, History, ObsDim]
        """
        # Save a reference to the main set
        self.main_obs_set = obs_set
        self.history_obs_set = f"{obs_set}_history"
        
        # Validate that the history group is available
        if self.history_obs_set not in obs:
            raise ValueError(
                f"HistoryEncoderModel requires '{self.history_obs_set}' observation group to be "
                f"present in the environment. Found: {list(obs.keys())}"
            )
            
        current_obs = obs[self.main_obs_set]
        history_obs = obs[self.history_obs_set]

        # Extract dimensions
        self.actor_obs_dim = current_obs.shape[-1]
        self.history_obs_dim = history_obs.shape[-1] # From [Batch, History, ObsDim]
        self.latent_hist_dim = 64
        
        # Override parent initialization so it doesn't crash configuring rsl_rl defaults.
        # rsl_rl's MLPModel tries to create self.obs_normalizer for the dimension of self.obs_groups.
        # It thinks we only have 'actor', which is fine. The normalizer will handle the current frame.
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        
        # Optional: A separate normalizer for the incoming history buffer.
        # We share the same dimension as current_obs normalizer, but handle it explicitly.
        self.history_normalizer = None
        if self.obs_normalizer is not None:
            # Create a separate normalizer for the history terms to track rolling mean/std.
            self.history_normalizer = EmpiricalNormalization(
                shape=[self.history_obs_dim],
            )
            self.history_normalizer.to(current_obs.device)

        # 1. Temporal Convolutional History Encoder
        self.history_encoder = nn.Sequential(
            # Input shape expected: [Batch, Channels=history_obs_dim, Time=history_length]
            nn.Conv1d(in_channels=self.history_obs_dim, out_channels=32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.AdaptiveMaxPool1d(1), # Auto-pool across the entire time sequence! Outputs [Batch, 64, 1]
            nn.Flatten(),            # Outputs [Batch, 64]
            nn.Linear(64, self.latent_hist_dim),
            nn.ELU(),
        )
        
        # 2. Main Actor Policy
        actor_input_dim = self.actor_obs_dim + self.latent_hist_dim
        hidden_dims = kwargs.get("hidden_dims", (512, 256, 128))
        activation = kwargs.get("activation", "elu")
        
        # We completely replace the MLP instantiated by the superclass
        self.mlp = MLP(actor_input_dim, output_dim, hidden_dims, activation)
        self.mlp.to(current_obs.device)
        self.history_encoder.to(current_obs.device)
        print(f"[{self.main_obs_set.upper()}] Configured HistoryEncoderModel (TCN). "
              f"Input: {self.actor_obs_dim}, Hist latent: {self.latent_hist_dim}.")

    def get_latent(self, obs, masks=None, hidden_state=None):
        """Build the model latent by explicitly processing history and current state."""
        # 1. Process Current Observation
        current_obs = obs[self.main_obs_set]
        if self.obs_normalizer is not None:
            current_obs = self.obs_normalizer(current_obs)
            
        # 2. Process History Observation
        history_obs = obs[self.history_obs_set]
        if self.history_normalizer is not None:
            history_obs = self.history_normalizer(history_obs)
            
        # PyTorch Conv1d expects shape [Batch, Channels, SequenceLength].
        # Environment provides [Batch, SequenceLength, Channels].
        # We transpose the spatial and temporal dimensions.
        history_obs = history_obs.transpose(-1, -2)
        
        # 3. Encode history
        encoded_hist = self.history_encoder(history_obs)
        
        # 4. Concatenate
        latent = torch.cat([current_obs, encoded_hist], dim=-1)
        
        return latent

    def eval(self):
        super().eval()
        if self.history_normalizer is not None:
            self.history_normalizer.eval()
            
    def train(self, mode=True):
        super().train(mode)
        if self.history_normalizer is not None:
            if mode:
                self.history_normalizer.train()
            else:
                self.history_normalizer.eval()
