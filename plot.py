import numpy as np
import matplotlib.pyplot as plt

env_id = "HalfCheetah-v2"
seeds = [1]
reward_scales=[0.1]
for seed in seeds:
    for reward_scale in reward_scales:
        linear_rewards = np.loadtxt('./baselines/ppo1/data/'+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'_rew.txt')* (1 / reward_scale)
        linear_ts = np.loadtxt('./baselines/ppo1/data/'+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'_ts.txt')
        fcn_rewards = np.loadtxt('./baselines/fcn/data/'+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'_rew.txt')* (1 / reward_scale)
        fcn_ts = np.loadtxt('./baselines/fcn/data/'+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'_ts.txt')


        fig, ax = plt.subplots()

        ax.plot(linear_ts, linear_rewards,  label='LINEAR')
        ax.plot(fcn_ts, fcn_rewards,  label='fcn-1')

        legend = ax.legend(loc='lower right')

        plt.savefig("./results/"+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'.png')
