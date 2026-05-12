# Training Information Report

## 1. Environment: ManagerBasedRlEnvCfg

### Rewards
| Name | Weight | Parameters |
| :--- | :--- | :--- |
| reaching_object | 5.0 | std: 0.1000 |
| lifting_object | 5.0 |  |
| object_goal_tracking | 25.0 | std: 0.3000 |
| object_goal_tracking_fine_grained | 10.0 | std: 0.0500 |
| arm_table_contact_penalty | -0.5 | sensor_names: ['ee_ground_collision', 'gripper_table_contact'] |

### Observation Groups

#### Group: actor
| Term | Function | Parameters |
| :--- | :--- | :--- |
| joint_pos | joint_pos_rel |  |
| joint_vel | joint_vel_rel |  |
| actions | last_action |  |
| gripper_pos | joint_pos_rel |  |
| ee_position | ee_position_in_robot_base_frame |  |
| goal_position | target_position_in_robot_base_frame |  |

#### Group: critic
| Term | Function | Parameters |
| :--- | :--- | :--- |
| joint_pos | joint_pos_rel |  |
| joint_vel | joint_vel_rel |  |
| actions | last_action |  |
| object_position | object_position_in_robot_root_frame |  |
| object_orientation | object_orientation_in_robot_root_frame |  |
| target_object_position | target_position_in_robot_base_frame |  |
| gripper_pos | joint_pos_rel |  |
| ee_position | ee_position_in_robot_base_frame |  |

#### Group: camera
| Term | Function | Parameters |
| :--- | :--- | :--- |
| wrist_camera_depth | camera_depth | sensor_name: wrist_realsense_camera, cutoff_distance: 1.0000 |

### Terminations
| Name | Function | Parameters |
| :--- | :--- | :--- |
| time_out | time_out |  |
| nan_term | nan_detection |  |
| arm_contact_while_lifting | arm_contact_while_lifting_term | sensor_names: ['ee_ground_collision', 'gripper_table_contact'], command_name: lift_height, asset_cfg: SceneEntityCfg(name='robot', joint_names=None, joint_ids=slice(None, None, None), body_names=None, body_ids=slice(None, None, None), geom_names=None, geom_ids=slice(None, None, None), site_names=('ee_right',), site_ids=slice(None, None, None), actuator_names=None, actuator_ids=slice(None, None, None), tendon_names=None, tendon_ids=slice(None, None, None), camera_names=None, camera_ids=slice(None, None, None), light_names=None, light_ids=slice(None, None, None), material_names=None, material_ids=slice(None, None, None), pair_names=None, pair_ids=slice(None, None, None), preserve_order=False) |

## 2. RL Configuration (PPO)

- **Experiment Name**: lift_depth
- **Max Iterations**: 30000
- **Steps per Env**: 24
- **Learning Rate**: 0.001
- **Entropy Coef**: 0.01

### CNN Architecture
- **Output Channels**: [16, 32]
- **Kernel Sizes**: [5, 3]
- **Strides**: [2, 2]
- **Spatial Softmax**: True
- **Spatial Softmax Temperature**: 1.0

### MLP Architecture
- **Actor Hidden Dims**: (256, 256, 128)
- **Critic Hidden Dims**: (256, 256, 128)
- **Activation**: elu