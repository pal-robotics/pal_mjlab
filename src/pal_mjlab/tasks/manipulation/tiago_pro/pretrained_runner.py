import os
import torch
from mjlab.tasks.manipulation.rl.runner import ManipulationOnPolicyRunner


class VisionPretrainedRunner(ManipulationOnPolicyRunner):
  """
  A unified custom runner that loads pre-trained CNN or ConvNeXt backbone weights
  and handles both "frozen" (permanently locked) and "curriculum" (initially frozen,
  then unfrozen after N iterations) RL training modes.
  """

  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    super().__init__(env, train_cfg, log_dir, device)

    # 1. Read runner configuration parameters
    self.backbone_mode = getattr(train_cfg, "backbone_mode", "frozen")
    self.unfreeze_at = getattr(train_cfg, "unfreeze_at", 300)
    self.fine_tune_lr = getattr(train_cfg, "fine_tune_lr", 1e-5)

    print(f"[VisionPretrainedRunner] Initializing in '{self.backbone_mode.upper()}' mode.")

    # 2. Determine model architecture type from actor class name
    actor_class_name = self.alg.actor.__class__.__name__
    is_convnext = "ConvNeXt" in actor_class_name
    
    # Select default weight path based on backbone
    backbone_path = "pretrained_convnext.pth" if is_convnext else "pretrained_backbone.pth"
    if not os.path.exists(backbone_path):
      backbone_path = "pretrained_backbone.pth" if is_convnext else "pretrained_convnext.pth"

    # 3. Load pre-trained visual backbone weights
    if os.path.exists(backbone_path):
      print(f"[VisionPretrainedRunner] Loading pre-trained backbone weights from {backbone_path}...")
      weights = torch.load(backbone_path, map_location=device)

      # Map keys dynamically to match PPO's actor/critic SpatialSoftmax structure
      mapped_weights = {}
      for k, v in weights.items():
        if is_convnext:
          # ConvNeXt keys map to 'cnns.camera.convnext.*'
          mapped_k = f"cnns.camera.convnext.{k}"
        else:
          # CNN keys map from 'cnn.X.*' to 'cnns.camera.cnn.X.*'
          if k.startswith("cnn."):
            mapped_k = f"cnns.camera.{k}"
          else:
            mapped_k = f"cnns.camera.cnn.{k}"
        mapped_weights[mapped_k] = v

      # Load weights into both actor and critic
      loaded = False
      for model_attr in ["actor_critic", "actor", "critic"]:
        if hasattr(self.alg, model_attr):
          obj = getattr(self.alg, model_attr)
          if model_attr == "actor_critic":
            for sub_model in [obj.actor, obj.critic]:
              missing, unexpected = sub_model.load_state_dict(mapped_weights, strict=False)
              print(
                f"[VisionPretrainedRunner] Loaded into actor_critic.{sub_model.__class__.__name__}. "
                f"Missing keys: {len([mk for mk in missing if mk.startswith('cnns')])}, Unexpected: {len(unexpected)}"
              )
            loaded = True
          else:
            missing, unexpected = obj.load_state_dict(mapped_weights, strict=False)
            print(
                f"[VisionPretrainedRunner] Loaded into {model_attr}. "
                f"Missing keys: {len([mk for mk in missing if mk.startswith('cnns')])}, Unexpected: {len(unexpected)}"
            )
            loaded = True
            
      if not loaded:
        print("[VisionPretrainedRunner] ERROR: Could not locate actor or critic models in algorithm!")
    else:
      print(f"[VisionPretrainedRunner] WARNING: No pre-trained backbone found at '{backbone_path}'. Starting from random weights.")

    # 4. Set initial gradient states
    if self.backbone_mode == "frozen":
      self._set_cnn_grad(False)
    elif self.backbone_mode == "curriculum":
      # Freeze initially
      if self.current_learning_iteration < self.unfreeze_at:
        print(f"[VisionPretrainedRunner] Phase 1: Backbone is FROZEN until iteration {self.unfreeze_at}.")
        self._set_cnn_grad(False)
      else:
        print("[VisionPretrainedRunner] Phase 2: Resuming with UNFROZEN backbone.")
        self._set_cnn_grad(True)
        self._set_lr(self.fine_tune_lr)

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
    if self.backbone_mode == "frozen":
      print("[VisionPretrainedRunner] Starting PPO training with PERMANENTLY FROZEN visual backbone.")
    elif self.backbone_mode == "curriculum":
      print(f"[VisionPretrainedRunner] Starting PPO curriculum training (unfreeze at iteration {self.unfreeze_at}).")
      
      # Hook into the algorithm's parameter update loop
      original_update = self.alg.update

      def curriculum_update():
        if self.current_learning_iteration == self.unfreeze_at:
          print(
              f"\n[VisionPretrainedRunner] >>> Iteration {self.unfreeze_at} reached. "
              f"UNFREEZING backbone and setting LR to {self.fine_tune_lr} <<<\n"
          )
          self._set_cnn_grad(True)
          self._set_lr(self.fine_tune_lr)
        return original_update()

      self.alg.update = curriculum_update

    return super().learn(
      num_learning_iterations=num_learning_iterations,
      init_at_random_ep_len=init_at_random_ep_len,
    )

  def _set_cnn_grad(self, requires_grad: bool):
    frozen_count = 0
    for model in [self.alg.actor, self.alg.critic]:
      if hasattr(model, "cnns"):
        for param in model.cnns.parameters():
          param.requires_grad = requires_grad
          frozen_count += 1
      else:
        print(f"[VisionPretrainedRunner] WARNING: Could not find 'cnns' in {model.__class__.__name__}. Skipping.")
    print(f"[VisionPretrainedRunner] Set requires_grad={requires_grad} for {frozen_count} backbone parameters.")

  def _set_lr(self, lr: float):
    for param_group in self.alg.optimizer.param_groups:
      param_group["lr"] = lr
    if hasattr(self.alg, "learning_rate"):
      self.alg.learning_rate = lr
