import os
env_id = "Swimmer-v2"
seed = 1 
reward_scale=0.5 
os.system('python -m baselines.fcn.run_humanoid'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale) + ' --env=' + env_id)
os.system('python -m baselines.scn.run_humanoid'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale) + ' --env=' + env_id)
os.system('python -m baselines.ppo1.run_humanoid'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale) + ' --env=' + env_id)
os.system('python -m baselines.linear.run_humanoid'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale) + ' --env=' + env_id)
