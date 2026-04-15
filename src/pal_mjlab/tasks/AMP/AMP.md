# Adversarial motion priors
  
To launch training with Adversarial Motion Prior on G1 (training environment is provided) :
  
'''bash
uv run train Mjlab-AMP-Flat-Unitree-G1 --motion-file-amp {filename}.csv --amp-weight {value} --resample {source_FPS} --{other options}
'''
  
(required) motion file amp csv file containing motion data (each row is observation for a single frame)
  
(optionnal) amp weight is 1.0 by default
  
(optionnal) resample uses linear interpolation to resample motion data from source fps to training step (automatically calculated). If not specified, no resampling is done.
  
Important : Observations in csv should match with observations specified in amp environment (joint_position, joint_velocity, root_linear_velocity ...).
  
To launch inference it is done as usual :
  
'''bash
un run play Mjlab-AMP-Flat-Unitree-G1 --checkpoint-file {checkpoint path}
'''
  
In a few words, an amp environment describes high level objectives (velocity, stability, ...) while using a discriminator to reward motion style using a reference motion.
  
___________________________________________________________________
  
Implementation based on paper :  
https://arxiv.org/abs/2104.02180  
"AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control"  
By : Xue Bin Peng, Ze Ma, Pieter Abbeel, Sergey Levine, Angjoo Kanazawa  