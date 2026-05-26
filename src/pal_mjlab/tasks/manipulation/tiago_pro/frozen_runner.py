import os
import torch
from mjlab.tasks.manipulation.rl.runner import ManipulationOnPolicyRunner

class VisionFrozenRunner(ManipulationOnPolicyRunner):
    """
    A custom runner that loads pre-trained CNN/ConvNeXt weights and permanently
    freezes the backbone during RL training (requires_grad = False).
    """
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        
        # Determine model type and load appropriate backbone weights
        # We try 'pretrained_convnext.pth' first, falling back to 'pretrained_backbone.pth'
        backbone_path = "pretrained_convnext.pth"
        is_convnext = True
        
        if not os.path.exists(backbone_path):
            backbone_path = "pretrained_backbone.pth"
            is_convnext = False
            
        if os.path.exists(backbone_path):
            print(f"[Frozen Runner] Loading pre-trained backbone from {backbone_path}...")
            weights = torch.load(backbone_path, map_location=device)
            
            # Map keys based on model architecture
            mapped_weights = {}
            for k, v in weights.items():
                if is_convnext:
                    # ConvNeXt keys directly map into 'cnns.camera.convnext.*'
                    mapped_k = f"cnns.camera.convnext.{k}"
                else:
                    # CNN keys map from 'cnn.X.*' to 'cnns.camera.cnn.X.*'
                    if k.startswith("cnn."):
                        mapped_k = f"cnns.camera.{k}"
                    else:
                        mapped_k = f"cnns.camera.cnn.{k}"
                mapped_weights[mapped_k] = v
            
            # Load robustly into both actor and critic
            loaded = False
            for model_attr in ["actor_critic", "actor", "critic"]:
                if hasattr(self.alg, model_attr):
                    obj = getattr(self.alg, model_attr)
                    if model_attr == "actor_critic":
                        for sub_model in [obj.actor, obj.critic]:
                            missing, unexpected = sub_model.load_state_dict(mapped_weights, strict=False)
                            print(f"[Frozen Runner] Loaded into actor_critic.{sub_model.__class__.__name__}. Missing keys: {len([mk for mk in missing if mk.startswith('cnns')])}, Unexpected keys: {len(unexpected)}")
                        loaded = True
                    else:
                        missing, unexpected = obj.load_state_dict(mapped_weights, strict=False)
                        print(f"[Frozen Runner] Loaded into {model_attr}. Missing keys: {len([mk for mk in missing if mk.startswith('cnns')])}, Unexpected keys: {len(unexpected)}")
                        loaded = True
            if not loaded:
                print("[Frozen Runner] ERROR: Could not locate actor or critic models in algorithm!")
        else:
            print("[Frozen Runner] WARNING: No pre-trained backbone weights found. Starting from random weights.")

        # Permanently freeze the backbone parameters
        self._freeze_backbone()

    def _freeze_backbone(self):
        frozen_count = 0
        for model in [self.alg.actor, self.alg.critic]:
            if hasattr(model, "cnns"):
                for param in model.cnns.parameters():
                    param.requires_grad = False
                    frozen_count += 1
            else:
                print(f"[Frozen Runner] WARNING: Could not find 'cnns' in {model.__class__.__name__}. Skipping.")
        print(f"[Frozen Runner] Success: Permanently FROZE {frozen_count} backbone parameters.")

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        print("[Frozen Runner] Starting training with PERMANENTLY FROZEN visual backbone features.")
        return super().learn(num_learning_iterations=num_learning_iterations, init_at_random_ep_len=init_at_random_ep_len)
