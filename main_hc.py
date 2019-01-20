import os
env_id = "HalfCheetah-v2"
seed_list = [1,2,3,4,5]
reward_scale_list=[0.1]
for seed in seed_list:
    for reward_scale in reward_scale_list:
        #if seed is 3 or seed is 4 or seed is 5:
        #os.system('python -m baselines.fcn.run_humanoid'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale) + ' --env=' + env_id)
        os.system('python -m baselines.scn.run_humanoid'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale) + ' --env=' + env_id)
        #os.system('python -m baselines.ppo1.run_humanoid'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale) + ' --env=' + env_id)
        #os.system('python -m baselines.linear.run_humanoid'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale) + ' --env=' + env_id)

        #os.system('python -m baselines.nlfcn.run_humanoid'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale) + ' --env=' + env_id)
        #os.system('python -m baselines.ppo1relu.run_humanoid'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale) + ' --env=' + env_id)
        #os.system('python -m baselines.scn64.run_humanoid'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale) + ' --env=' + env_id)
