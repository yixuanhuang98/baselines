import numpy as np
import matplotlib.pyplot as plt

env_id = "HalfCheetah-v2"
seeds = [1]
reward_scales=[0.1]
for seed in seeds:
    for reward_scale in reward_scales:
        linear_rewards = np.loadtxt('./baselines/ppo1/data/'+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'_rew.txt')* (1 / reward_scale)
        linear_ts = np.loadtxt('./baselines/ppo1/data/'+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'_ts.txt')
        lc_rewards = np.loadtxt('./baselines/ppo1/data/'+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'_rew_lc.txt')* (1 / reward_scale)
        lc_ts = np.loadtxt('./baselines/ppo1/data/'+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'_ts_lc.txt')


        fig, ax = plt.subplots()

        ax.plot(linear_ts, linear_rewards,  label='LINEAR')
        ax.plot(lc_ts, lc_rewards,  label='LC')

        legend = ax.legend(loc='lower right')

        plt.savefig("./results/"+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'.png')