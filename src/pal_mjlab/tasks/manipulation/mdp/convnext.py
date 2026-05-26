import torch
import torch.nn as nn
from rsl_rl.models.cnn_model import CNNModel
from rsl_rl.models.mlp_model import MLPModel
from tensordict import TensorDict
from typing import Any

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return input + x

class SpatialSoftmax(nn.Module):
    def __init__(self, height: int, width: int, temperature: float = 0.5):
        super().__init__()
        self.height = height
        self.width = width
        self.temperature = temperature
        
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, self.height),
            torch.linspace(-1.0, 1.0, self.width),
            indexing="ij"
        )
        self.register_buffer('pos_x', pos_x.reshape(1, 1, -1))
        self.register_buffer('pos_y', pos_y.reshape(1, 1, -1))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feature.size()
        features = feature.reshape(B, C, -1)
        weights = torch.softmax(features / self.temperature, dim=-1)
        expected_x = (weights * self.pos_x).sum(dim=-1)
        expected_y = (weights * self.pos_y).sum(dim=-1)
        return torch.stack([expected_x, expected_y], dim=-1).reshape(B, C * 2)

class ConvNeXtBackbone(nn.Module):
    def __init__(self, num_keypoints: int = 6):
        super().__init__()
        # 128x128 -> 64x64
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            LayerNorm2d(32)
        )
        self.stage1 = nn.Sequential(
            ConvNeXtBlock(32),
            ConvNeXtBlock(32)
        )
        # 64x64 -> 32x32
        self.downsample = nn.Sequential(
            LayerNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=2, stride=2)
        )
        self.stage2 = nn.Sequential(
            ConvNeXtBlock(64),
            ConvNeXtBlock(64),
            ConvNeXtBlock(64)
        )
        
        self.head = nn.Conv2d(64, num_keypoints, kernel_size=1)
        self.spatial_softmax = SpatialSoftmax(32, 32, temperature=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.downsample(x)
        x = self.stage2(x)
        x = self.head(x)
        x = self.spatial_softmax(x)
        return x

class SpatialSoftmaxConvNeXt(nn.Module):
    """Wrapper that adapts the custom ConvNeXt keypoint extractor for RSL-RL PPO model interfaces."""
    def __init__(self, num_keypoints: int = 6):
        super().__init__()
        self.convnext = ConvNeXtBackbone(num_keypoints=num_keypoints)
        self._output_dim = num_keypoints * 2

    @property
    def output_dim(self) -> int:
        """Total flattened keypoints dimension (C * 2)."""
        return self._output_dim

    @property
    def output_channels(self) -> None:
        """Always None indicating flat keypoint predictions."""
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convnext(x)

class SpatialSoftmaxConvNeXtModel(CNNModel):
    """CNN PPO model using custom ConvNeXt feature extractor with Spatial Softmax."""
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        cnn_cfg: dict[str, dict] | dict[str, Any],
        cnns: nn.ModuleDict | None = None,
        hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict[str, Any] | None = None,
    ) -> None:
        self._get_obs_dim(obs, obs_groups, obs_set)

        if cnns is not None:
            if set(cnns.keys()) != set(self.obs_groups_2d):
                raise ValueError("Shared encoders must match active 2D groups.")
            _cnns = cnns
        else:
            _cnns = {}
            for obs_group in self.obs_groups_2d:
                # The ConvNeXt model has a fixed shape and structure
                _cnns[obs_group] = SpatialSoftmaxConvNeXt(num_keypoints=6)

        self.cnn_latent_dim = 0
        for cnn in _cnns.values():
            self.cnn_latent_dim += int(cnn.output_dim)

        MLPModel.__init__(
            self,
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )

        if isinstance(_cnns, nn.ModuleDict):
            self.cnns = _cnns
        else:
            self.cnns = nn.ModuleDict(_cnns)
