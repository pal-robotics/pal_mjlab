import torch
from mjlab.rl.spatial_softmax import SpatialSoftmaxCNNModel
from tensordict import TensorDict
import numpy as np

# Mock observation
obs = TensorDict({
    "actor": torch.randn(1, 10),
    "camera": torch.randn(1, 1, 128, 128)
}, batch_size=[1])

obs_groups = {
    "actor": ["actor", "camera"]
}

cnn_cfg = {
    "output_channels": [16, 32],
    "kernel_size": [5, 3],
    "stride": [2, 2],
    "padding": "zeros",
    "activation": "elu",
    "max_pool": False,
    "global_pool": "none",
    "spatial_softmax": True,
    "spatial_softmax_temperature": 1.0,
}

model = SpatialSoftmaxCNNModel(
    obs=obs,
    obs_groups=obs_groups,
    obs_set="actor",
    output_dim=8,
    cnn_cfg=cnn_cfg,
)

# Test with a hot spot in the input
# Note: Since the CNN is untrained, a hot spot in the input might not result 
# in a hot spot in the output. But let's see.
with torch.no_grad():
    # Create a "cube" at (100, 100)
    camera_obs = torch.zeros(1, 1, 128, 128)
    camera_obs[0, 0, 90:110, 90:110] = 1.0 
    
    kp = model.cnns["camera"](camera_obs)
    kp = kp.reshape(-1, 2)
    
    print(f"Keypoints for hot spot at (100, 100):")
    for i in range(5):
        print(f"  KP {i}: {kp[i].cpu().numpy()}")

    # Try with a very low temperature
    model.cnns["camera"].spatial_softmax.temperature = 0.01
    kp_low = model.cnns["camera"](camera_obs)
    kp_low = kp_low.reshape(-1, 2)
    print(f"Keypoints with temperature 0.01:")
    for i in range(5):
        print(f"  KP {i}: {kp_low[i].cpu().numpy()}")
