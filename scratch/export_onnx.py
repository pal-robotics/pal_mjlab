
import torch
import torch.nn as nn

# Import the mjlab metadata utilities we found earlier
from mjlab.rl.exporter_utils import attach_metadata_to_onnx


def get_base_metadata_fixed(env, run_path: str):
  """Custom metadata extractor bypassing the hardcoded assertion in mjlab."""
  robot = env.scene["robot"]
  joint_action = env.action_manager.get_term("joint_pos")

  joint_name_to_ctrl_id = {}
  for actuator in robot.spec.actuators:
    joint_name = actuator.target.split("/")[-1]
    joint_name_to_ctrl_id[joint_name] = actuator.id

  ctrl_ids_natural = [
    joint_name_to_ctrl_id[jname]
    for jname in robot.joint_names
    if jname in joint_name_to_ctrl_id
  ]
  joint_stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids_natural, 0]
  joint_damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids_natural, 2]

  action_scale = joint_action._scale
  if hasattr(action_scale, "cpu"):
    action_scale = action_scale[0].cpu().tolist()

  return {
    "run_path": run_path,
    "joint_names": list(robot.joint_names),
    "joint_stiffness": joint_stiffness.tolist(),
    "joint_damping": joint_damping.tolist(),
    "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
    "command_names": list(env.command_manager.active_terms),
    "observation_names": env.observation_manager.active_terms["actor"],
    "action_scale": action_scale,
  }


class ExportableActor(nn.Module):
  def __init__(self, state_dict):
    super().__init__()
    self.has_normalizer = "obs_normalizer._mean" in state_dict
    if self.has_normalizer:
      self.register_buffer("mean", state_dict["obs_normalizer._mean"])
      self.register_buffer("std", state_dict["obs_normalizer._std"])

    # Build MLP
    mlp_state = {
      k.replace("mlp.", ""): v for k, v in state_dict.items() if k.startswith("mlp.")
    }
    indices = sorted(
      list(set(int(k.split(".")[0]) for k in mlp_state.keys() if k[0].isdigit()))
    )

    layers = []
    for i, idx in enumerate(indices):
      w = mlp_state[f"{idx}.weight"]
      out_dim, in_dim = w.shape
      layers.append(nn.Linear(in_dim, out_dim))
      if i < len(indices) - 1:
        layers.append(nn.ELU())

    self.mlp = nn.Sequential(*layers)
    self.mlp.load_state_dict(mlp_state)

  def forward(self, obs):
    if self.has_normalizer:
      obs = (obs - self.mean) / self.std
      obs = torch.clamp(obs, -5.0, 5.0)
    return self.mlp(obs)


def export_rsl_rl_policy(pt_path: str, onnx_path: str, env, run_path: str):
  print(f"Loading RSL-RL checkpoint from {pt_path}...")
  loaded_dict = torch.load(pt_path, map_location="cpu")

  # The checkpoint has 'actor_state_dict' directly
  actor_state_dict = loaded_dict.get(
    "actor_state_dict", loaded_dict.get("model_state_dict", loaded_dict)
  )

  # Dynamically build the actor with normalization baked in
  actor = ExportableActor(actor_state_dict)
  actor.eval()

  num_obs = actor.mlp[0].in_features

  print(f"Exporting to ONNX: {onnx_path}...")
  dummy_input = torch.randn(1, num_obs)

  torch.onnx.export(
    actor,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=18,
    input_names=["obs"],
    output_names=["actions"],
    dynamic_axes={"obs": {0: "batch_size"}, "actions": {0: "batch_size"}},
  )
  print("ONNX file created.")

  # Attach the required metadata for pal_policy_deployer
  print("Attaching mjlab metadata...")
  metadata = get_base_metadata_fixed(env, run_path)
  attach_metadata_to_onnx(onnx_path, metadata)
  print(f"Done! Successfully exported {onnx_path}")


if __name__ == "__main__":
  # Initialize the pal_mjlab environment
  from mjlab.envs import ManagerBasedRlEnv
  from pal_mjlab.tasks.manipulation.tiago_pro.env_cfgs import lift_env_cfg

  print("Initializing environment to extract metadata...")
  env_cfg = lift_env_cfg(play=True)
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")

  pt_file = "/home/lorenzobarbieri/model_1500.pt"
  onnx_file = "/home/lorenzobarbieri/exchange/tiago_pro_sim_ws/src/pal_policy_deployer/pal_policy_deployer/models/tiagopro/omniscent_debug.onnx"

  # Run the export
  export_rsl_rl_policy(pt_file, onnx_file, env, "omniscent_debug")
