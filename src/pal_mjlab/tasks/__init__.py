# Monkey patch mjlab get_base_metadata to support RelativeJointPositionAction
import torch
import mjlab.rl.exporter_utils
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions.actions import BaseAction

def patched_get_base_metadata(
  env: ManagerBasedRlEnv, run_path: str
) -> dict[str, list | str | float]:
  robot: Entity = env.scene["robot"]
  joint_action = env.action_manager.get_term("joint_pos")
  assert isinstance(joint_action, BaseAction)
  # Build mapping from joint name to actuator ID for natural joint order.
  # Each spec actuator controls exactly one joint (via its target field).
  joint_name_to_ctrl_id = {}
  for actuator in robot.spec.actuators:
    joint_name = actuator.target.split("/")[-1]
    joint_name_to_ctrl_id[joint_name] = actuator.id
  # Get actuator IDs in natural joint order (same order as robot.joint_names).
  ctrl_ids_natural = [
    joint_name_to_ctrl_id[jname]
    for jname in robot.joint_names  # global joint order
    if jname in joint_name_to_ctrl_id  # skip non-actuated joints
  ]
  joint_stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids_natural, 0]
  joint_damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids_natural, 2]
  return {
    "run_path": run_path,
    "joint_names": list(robot.joint_names),
    "joint_stiffness": joint_stiffness.tolist(),
    "joint_damping": joint_damping.tolist(),
    "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
    "command_names": list(env.command_manager.active_terms),
    "observation_names": env.observation_manager.active_terms["actor"],
    "action_scale": joint_action._scale[0].cpu().tolist()
    if isinstance(joint_action._scale, torch.Tensor)
    else joint_action._scale,
  }

mjlab.rl.exporter_utils.get_base_metadata = patched_get_base_metadata

try:
  import mjlab.tasks.manipulation.rl.runner
  mjlab.tasks.manipulation.rl.runner.get_base_metadata = patched_get_base_metadata
except ImportError:
  pass

try:
  import mjlab.tasks.velocity.rl.runner
  mjlab.tasks.velocity.rl.runner.get_base_metadata = patched_get_base_metadata
except ImportError:
  pass

try:
  import mjlab.tasks.tracking.rl.runner
  mjlab.tasks.tracking.rl.runner.get_base_metadata = patched_get_base_metadata
except ImportError:
  pass


# Monkey patch MjlabOnPolicyRunner to support loading and running ONNX checkpoints directly in play script
import cv2
from mjlab.rl.runner import MjlabOnPolicyRunner

original_load = MjlabOnPolicyRunner.load
original_get_inference_policy = MjlabOnPolicyRunner.get_inference_policy

def patched_load(self, path: str, *args, **kwargs):
  if path.endswith(".onnx"):
    print(f"[INFO]: Loading ONNX policy model from {path} via OpenCV DNN...")
    self._onnx_net = cv2.dnn.readNetFromONNX(path)
    return {}
  return original_load(self, path, *args, **kwargs)

def patched_get_inference_policy(self, device=None):
  if hasattr(self, "_onnx_net"):
    print("[INFO]: Returning inference policy wrapping ONNX model.")
    obs_groups = self.alg.get_policy().obs_groups
    def onnx_policy(obs) -> torch.Tensor:
      if isinstance(obs, dict) or hasattr(obs, "keys"):
        obs_list = [obs[g] for g in obs_groups]
        flat_obs = torch.cat(obs_list, dim=-1)
      else:
        flat_obs = obs
      obs_np = flat_obs.cpu().numpy()
      self._onnx_net.setInput(obs_np)
      out = self._onnx_net.forward()
      return torch.from_numpy(out).to(flat_obs.device)
    return onnx_policy
  return original_get_inference_policy(self, device)

MjlabOnPolicyRunner.load = patched_load
MjlabOnPolicyRunner.get_inference_policy = patched_get_inference_policy


from mjlab.utils.lab_api.tasks.importer import import_packages

_BLACKLIST_PKGS = ["utils", ".mdp"]

import_packages(__name__, _BLACKLIST_PKGS)

