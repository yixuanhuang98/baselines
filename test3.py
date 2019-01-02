import os
env_id = "Swimmer-v2"
seed_list = [1,2,3]
reward_scale_list=[1]
for seed in seed_list:
    for reward_scale in reward_scale_list:
        os.system('python -m baselines.fcn.run_humanoid'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale) + ' --env=' + env_id)
        os.system('python -m baselines.linear.run_humanoid'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale) + ' --env=' + env_id)
