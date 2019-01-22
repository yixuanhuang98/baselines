import os
env_id = "Hopper-v2"
seed_list = [1,2,3,4,5]
reward_scale_list=[0.1]
for seed in seed_list:
    for reward_scale in reward_scale_list:
        os.system('python -m baselines.stu.run_humanoid'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale) + ' --env=' + env_id +'--model-path=model/hyb_hp_'+str(seed))
