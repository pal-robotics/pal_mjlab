.. _Kangaroo overview:

PAL Kangaroo
============

Kangaroo is one of PAL Robotics' humanoid platforms.  
With 14 DOFS in the lower body and various arm configurations (0 DOFS, 4 DOFS, 7 DOFS, ...) Kangaroo has the capability to perform many human inspired tasks.

Here, we explore mainly :  
- Locomotion  
- Motion imitation  

Here are the different tasks implemented for Kangaroo, which are relative to objective and model :  
| 1  | Mjlab-Reaching-Flat-Pal-Kangaroo  
| 2  | Mjlab-Reaching-Flat-Pal-Kangaroo-Grippers  
| 3  | Mjlab-Reaching-Flat-Pal-Kangaroo-Hands  
| 4  | Mjlab-Tracking-Flat-Pal-Kangaroo  
| 5  | Mjlab-Tracking-Flat-Pal-Kangaroo-Lower-Body  
| 6  | Mjlab-Tracking-Flat-Pal-Kangaroo-Lower-Body-No-State-Estimation  
| 7  | Mjlab-Tracking-Flat-Pal-Kangaroo-No-State-Estimation  
| 8  | Mjlab-Velocity-Flat-Pal-Kangaroo  
| 9  | Mjlab-Velocity-Flat-Pal-Kangaroo-Grippers  
| 10 | Mjlab-Velocity-Flat-Pal-Kangaroo-Hands  
| 11 | Mjlab-Velocity-Flat-Pal-Kangaroo-Lower-Body  
| 12 | Mjlab-Velocity-Rough-Pal-Kangaroo  
| 13 | Mjlab-Velocity-Rough-Pal-Kangaroo-Grippers  
| 14 | Mjlab-Velocity-Rough-Pal-Kangaroo-Hands  

The full list of tasks can be checked anytime with ::  

    bash  
    uv run list-envs  

To train a task on Kangaroo, use ::

    bash  
    uv run train {task_id} {other parameters}  

The full list of parameters can be seen using ::

    bash  
    uv run train {task_id} --help  

To run inference to evaluate results ::

    bash  
    uv run play {task_id} {other parameters}  