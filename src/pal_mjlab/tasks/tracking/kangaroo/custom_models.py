"""Custom RL Model Architectures for Kangaroo Tracking with A-RMA."""

import torch
import torch.nn as nn
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules.mlp import MLP
from rsl_rl.modules.normalization import EmpiricalNormalization


class PrivilegedEncoder(nn.Module):
    """MLP Encoder to compress privileged information (e_t) into a latent (z_t).
    
    Architecture: 1 hidden layer of size 256.
    """

    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ELU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, e_t: torch.Tensor):
        return self.encoder(e_t)


class ArmaActorModel(MLPModel):
    """Actor Model for A-RMA Phase 1.
    
    Inputs:
    - o_t (from 'actor' group, last frame): Current observation.
    - z_t (from 'privileged' group via encoder): 8D extrinsics.
    """

    def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs):
        self.main_obs_set = obs_set  # 'actor'
        self.priv_obs_set = "privileged"
        
        # Dimensions
        # In history-enabled envs, obs[obs_set] is [Batch, History, Dim]
        self.obs_dim = obs[self.main_obs_set].shape[-1]
        self.priv_dim = 167 
        self.z_dim = 8
        
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        
        # Separate normalizer for privileged info
        self.priv_normalizer = EmpiricalNormalization(shape=[self.priv_dim]).to(obs[self.main_obs_set].device)

        # Privileged Encoder
        self.privileged_encoder = PrivilegedEncoder(self.priv_dim, self.z_dim).to(obs[self.main_obs_set].device)
        
        # Policy MLP
        # Input: [o_t, z_t]
        actor_input_dim = self.obs_dim + self.z_dim
        hidden_dims = kwargs.get("hidden_dims", (512, 256, 128))
        activation = kwargs.get("activation", "elu")
        
        self.mlp = MLP(actor_input_dim, output_dim, hidden_dims, activation).to(obs[self.main_obs_set].device)

    def _get_obs_dim(self, obs, obs_groups, obs_set):
        """Override to allow history-based 3D observations."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim

    def get_latent(self, obs, masks=None, hidden_state=None):
        h_t = obs[self.main_obs_set] # [Batch, 50, Dim]
        e_t = obs[self.priv_obs_set] # [Batch, 167]
        
        # Use only the CURRENT frame (o_t) for the policy in Phase 1
        o_t = h_t[:, -1, :] 
        
        # Normalization
        # Note: self.obs_normalizer is calibrated to [Dim] which matches o_t
        if self.obs_normalizer is not None:
            o_t = self.obs_normalizer(o_t)
        e_t = self.priv_normalizer(e_t)
        
        # Encoding
        z_t = self.privileged_encoder(e_t)
        
        return torch.cat([o_t, z_t], dim=-1)

    def forward(self, obs, stochastic_output=False, **kwargs):
        latent = self.get_latent(obs)
        out = self.mlp(latent)
        if hasattr(self, "distribution") and self.distribution is not None:
            if stochastic_output:
                self.distribution.update(out)
                return self.distribution.sample()
            else:
                return self.distribution.deterministic_output(out)
        return out


class ArmaCriticModel(MLPModel):
    """Critic Model for A-RMA Phase 1.
    
    Inputs:
    - o_t (from 'critic' group, last frame): Current observation.
    - e_t (from 'privileged' group): 167D raw environment vector.
    """

    def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs):
        self.main_obs_set = obs_set  # 'critic'
        self.priv_obs_set = "privileged"
        
        self.obs_dim = obs[self.main_obs_set].shape[-1]
        self.priv_dim = 167
        
        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
        self.priv_normalizer = EmpiricalNormalization(shape=[self.priv_dim]).to(obs[self.main_obs_set].device)
        
        # Value MLP
        # Input: [o_t, e_t]
        critic_input_dim = self.obs_dim + self.priv_dim
        hidden_dims = kwargs.get("hidden_dims", (512, 256, 128))
        activation = kwargs.get("activation", "elu")
        
        self.mlp = MLP(critic_input_dim, output_dim, hidden_dims, activation).to(obs[self.main_obs_set].device)

    def _get_obs_dim(self, obs, obs_groups, obs_set):
        """Override to allow history-based 3D observations."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim

    def get_latent(self, obs, masks=None, hidden_state=None):
        h_t = obs[self.main_obs_set]
        e_t = obs[self.priv_obs_set]
        
        o_t = h_t[:, -1, :]
        
        if self.obs_normalizer is not None:
            o_t = self.obs_normalizer(o_t)
        e_t = self.priv_normalizer(e_t)
        
        return torch.cat([o_t, e_t], dim=-1)

    def forward(self, obs, **kwargs):
        latent = self.get_latent(obs)
        return self.mlp(latent)
