"""Custom RL Model Architectures for Kangaroo Tracking with A-RMA."""

import torch
import torch.nn as nn
import copy
from rsl_rl.models.mlp_model import MLPModel, _OnnxMLPModel, _TorchMLPModel
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


class AdaptationModule(nn.Module):
    """Temporal Convolutional Network (TCN) for A-RMA Phase 2.
    
    Predicts estimated latent z_hat from history buffer.
    Architecture:
    - 3 Convolution layers (kernels 8, 5, 5; strides 4, 1, 1).
    - 32 channels.
    - 1 Hidden layer of size 256.
    """

    def __init__(self, input_dim: int, history_length: int, latent_dim: int = 8, channels: int = 32):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self._get_flatten_size(history_length, channels), 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )

    def _get_flatten_size(self, history_length, channels):
        # L_out = floor((L_in + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1)
        l = history_length
        l = (l - 8) // 4 + 1
        l = (l - 5) // 1 + 1
        l = (l - 5) // 1 + 1
        return channels * l

    def forward(self, history: torch.Tensor):
        """
        Args:
            history: [Batch, HistoryLength, ObsDim]
        Returns:
            z_hat: [Batch, LatentDim]
        """
        # Conv1d expects [Batch, Channels, Length]. Observation dimension is our channels.
        x = history.permute(0, 2, 1)
        return self.tcn(x)


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
        self.priv_dim = obs[self.priv_obs_set].shape[-1]
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

        # Adaptation Module (TCN) - Initialized as None for Phase 1
        # Will be injected/assigned for Phase 3 training and Deployment.
        self.adaptation_module = None

    def _get_obs_dim(self, obs, obs_groups, obs_set):
        """Override to allow history-based 3D observations."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim

    def update_normalization(self, obs: dict[str, torch.Tensor]) -> None:
        """Override to extract last frame from history before updating normalizer."""
        if self.obs_normalizer is not None:
            h_t = obs[self.main_obs_set]
            o_t = h_t[:, -1, :] if h_t.ndim == 3 else h_t
            self.obs_normalizer.update(o_t)
        
        if hasattr(self, "priv_normalizer"):
            e_t = obs[self.priv_obs_set]
            self.priv_normalizer.update(e_t)

    def get_latent(self, obs, masks=None, hidden_state=None, z_hat=None):
        h_t = obs[self.main_obs_set] # [Batch, History, Dim]
        
        # STATIC START TILING (Phase 2 & Phase 3)
        # If the oldest frame in history is all zeros, we just reset.
        # We tile the current observation (which inherently has v=0 thanks to disable_rsi)
        # back across the buffer to avoid the zero-teleportation shock for the TCN.
        if h_t.ndim == 3 and h_t[:, 0, :].abs().max() < 1e-4:
            h_t = h_t[:, -1, :].unsqueeze(1).repeat(1, h_t.shape[1], 1)
            
        # Use only the CURRENT frame (o_t) for the policy in Phase 1
        o_t = h_t[:, -1, :] if h_t.ndim == 3 else h_t        
        # Normalization
        # Note: self.obs_normalizer is calibrated to [Dim] which matches o_t
        if self.obs_normalizer is not None:
            o_t = self.obs_normalizer(o_t)

        if z_hat is not None:
            # Overridden z_hat (for Phase 2 DAgger or manual override)
            z_t = z_hat
        elif self.adaptation_module is not None:
            # Phase 3 / Deployment: Use TCN to estimate latent from history
            z_t = self.adaptation_module(h_t)
        else:
            # Standard Phase 1: Use privileged encoder
            e_t = obs[self.priv_obs_set] # [Batch, 167]
            e_t = self.priv_normalizer(e_t)
            z_t = self.privileged_encoder(e_t)
        
        return torch.cat([o_t, z_t], dim=-1)

    def forward(self, obs, stochastic_output=False, **kwargs):
        latent = self.get_latent(obs, **kwargs)
        out = self.mlp(latent)
        if hasattr(self, "distribution") and self.distribution is not None:
            if stochastic_output:
                self.distribution.update(out)
                return self.distribution.sample()
            else:
                return self.distribution.deterministic_output(out)
        return out

    def as_onnx(self, verbose=False):
        """Return a version of the model compatible with ONNX export (includes TCN)."""
        return _OnnxArmaActorModel(self, verbose)

    def as_jit(self):
        """Return a version of the model compatible with Torch JIT export (includes TCN)."""
        return _TorchArmaActorModel(self)


class _OnnxArmaActorModel(nn.Module):
    """Custom ONNX export wrapper for ArmaActorModel.
    
    Input: [Batch, 50, Dim] History Tensor
    Output: Action Tensor
    """
    def __init__(self, model, verbose):
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        
        # Deployment uses the Adaptation Module (TCN)
        self.adaptation_module = copy.deepcopy(model.adaptation_module)
        
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
            
        self.obs_dim = model.obs_dim
        self.history_length = 75
        self.input_size = self.obs_dim # For compatibility with standard runners

    def forward(self, history: torch.Tensor):
        # 1. Extract and normalize current frame (o_t)
        # Handle cases where history is 2D (single frame) or 3D
        if history.ndim == 3:
            # STATIC START TILING (Hardware Deployment Parity)
            # If the oldest frame is zeros, repeat the current frame across the history
            # mask out everything else. This handles "cold starts" cleanly in C++.
            history_is_empty = history[:, 0, :].abs().max() < 1e-4
            if history_is_empty:
                history = history[:, -1, :].unsqueeze(1).repeat(1, history.shape[1], 1)
            o_t = history[:, -1, :]
        else:
            o_t = history
            
        o_t = self.obs_normalizer(o_t)
        
        # 2. Get latent estimate (z_hat)
        if self.adaptation_module is not None and history.ndim == 3:
            # Phase 3 / Deployment: Use TCN to estimate latent from history
            z_hat = self.adaptation_module(history)
        else:
            # Phase 1 / Fallback: Return zero latent
            # We use 8 as the default latent dimension
            z_hat = torch.zeros(o_t.shape[0], 8, device=o_t.device)
        
        # 3. Policy Forward
        latent = torch.cat([o_t, z_hat], dim=-1)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    def get_dummy_inputs(self):
        return (torch.zeros(1, self.history_length, self.obs_dim),)

    @property
    def input_names(self):
        return ["obs_history"]

    @property
    def output_names(self):
        return ["actions"]


class _TorchArmaActorModel(_OnnxArmaActorModel):
    """Custom JIT export wrapper for ArmaActorModel."""
    def __init__(self, model):
        super().__init__(model, verbose=False)

    @torch.jit.export
    def reset(self):
        pass


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
        self.priv_dim = obs[self.priv_obs_set].shape[-1]
        
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

    def update_normalization(self, obs: dict[str, torch.Tensor]) -> None:
        """Override to extract last frame from history before updating normalizer."""
        if self.obs_normalizer is not None:
            h_t = obs[self.main_obs_set]
            o_t = h_t[:, -1, :] if h_t.ndim == 3 else h_t
            self.obs_normalizer.update(o_t)
        
        if hasattr(self, "priv_normalizer"):
            e_t = obs[self.priv_obs_set]
            self.priv_normalizer.update(e_t)

    def get_latent(self, obs, masks=None, hidden_state=None):
        h_t = obs[self.main_obs_set]
        e_t = obs[self.priv_obs_set]
        
        o_t = h_t[:, -1, :] if h_t.ndim == 3 else h_t
        
        if self.obs_normalizer is not None:
            o_t = self.obs_normalizer(o_t)
        e_t = self.priv_normalizer(e_t)
        
        return torch.cat([o_t, e_t], dim=-1)

    def forward(self, obs, **kwargs):
        latent = self.get_latent(obs)
        return self.mlp(latent)
