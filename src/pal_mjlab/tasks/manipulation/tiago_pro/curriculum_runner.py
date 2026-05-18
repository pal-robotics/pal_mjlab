import torch
from mjlab.tasks.manipulation.rl.runner import ManipulationOnPolicyRunner

class VisionCurriculumRunner(ManipulationOnPolicyRunner):
    """
    A custom runner that implements a two-phase training curriculum:
    1. Freeze CNN for the first 300 iterations.
    2. Unfreeze CNN with a reduced learning rate (1e-5) from iteration 300 onwards.
    """
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        
        # Load pre-trained CNN backbone
        import os
        backbone_path = "pretrained_backbone.pth"
        if os.path.exists(backbone_path):
            print(f"[Curriculum] Loading pre-trained CNN backbone from {backbone_path}...")
            weights = torch.load(backbone_path, map_location=device)
            
            # Map keys: 'cnn.X.*' -> 'cnns.camera.cnn.X.*'
            mapped_weights = {}
            for k, v in weights.items():
                if k.startswith("cnn."):
                    mapped_k = f"cnns.camera.{k}"
                    mapped_weights[mapped_k] = v
            
            # Load into both actor and critic robustly
            loaded = False
            for model_attr in ["actor_critic", "actor", "critic"]:
                if hasattr(self.alg, model_attr):
                    obj = getattr(self.alg, model_attr)
                    if model_attr == "actor_critic":
                        for sub_model in [obj.actor, obj.critic]:
                            missing, unexpected = sub_model.load_state_dict(mapped_weights, strict=False)
                            print(f"[Curriculum] Loaded into actor_critic.{sub_model.__class__.__name__}. Missing keys: {len([mk for mk in missing if mk.startswith('cnns')])}, Unexpected keys: {len(unexpected)}")
                        loaded = True
                    else:
                        missing, unexpected = obj.load_state_dict(mapped_weights, strict=False)
                        print(f"[Curriculum] Loaded into {model_attr}. Missing keys: {len([mk for mk in missing if mk.startswith('cnns')])}, Unexpected keys: {len(unexpected)}")
                        loaded = True
            if not loaded:
                print("[Curriculum] ERROR: Could not locate actor or critic models in algorithm!")
        else:
            print("[Curriculum] WARNING: No pre-trained backbone found at 'pretrained_backbone.pth'. Starting from random weights.")

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        unfreeze_at = 30000
        fine_tune_lr = 1e-5
        
        # 1. Initial State Check
        if self.current_learning_iteration < unfreeze_at:
            print(f"[Curriculum] Initializing Phase 1: CNN backbone is FROZEN until iteration {unfreeze_at}.")
            self._set_cnn_grad(False)
        else:
            print(f"[Curriculum] Resuming Phase 2: CNN backbone is already UNFROZEN.")
            self._set_cnn_grad(True)
            self._set_lr(fine_tune_lr)

        # 2. Hook into the algorithm update loop to catch the 300-iteration mark
        original_update = self.alg.update
        
        def curriculum_update():
            # Check if we just hit the transition point
            if self.current_learning_iteration == unfreeze_at:
                print(f"\n[Curriculum] >>> Iteration {unfreeze_at} reached. UNFREEZING CNN and setting LR to {fine_tune_lr} <<<\n")
                self._set_cnn_grad(True)
                self._set_lr(fine_tune_lr)
            
            return original_update()

        # Inject our wrapped update
        self.alg.update = curriculum_update
        
        # 3. Call super().learn ONCE with the full iteration count
        # This ensures logs and progress bars show the correct total (e.g., 30,000)
        return super().learn(num_learning_iterations=num_learning_iterations, init_at_random_ep_len=init_at_random_ep_len)

    def _set_cnn_grad(self, requires_grad: bool):
        for model in [self.alg.actor, self.alg.critic]:
            if hasattr(model, "cnns"):
                for param in model.cnns.parameters():
                    param.requires_grad = requires_grad
            else:
                print(f"[WARN] Could not find 'cnns' in {model.__class__.__name__}. Skipping grad update.")

    def _set_lr(self, lr: float):
        # Update learning rate in the optimizer
        for param_group in self.alg.optimizer.param_groups:
            param_group['lr'] = lr
        # Sync the algorithm's internal LR tracker
        if hasattr(self.alg, "learning_rate"):
            self.alg.learning_rate = lr
