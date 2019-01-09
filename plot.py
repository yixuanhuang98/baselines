import numpy as np
import matplotlib.pyplot as plt

env_id = "HalfCheetah-v2"
seeds = [2]
reward_scales=[0.1]
for seed in seeds:
    for reward_scale in reward_scales:
        linear_rewards = np.loadtxt('./baselines/ppo1/data/'+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'_rew_t.txt')* (1 / reward_scale)
        linear_ts = np.loadtxt('./baselines/ppo1/data/'+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'_ts_t.txt')
        #fcn_rewards = np.loadtxt('./baselines/fcn/data/'+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'_rew.txt')* (1 / reward_scale)
        #fcn_ts = np.loadtxt('./baselines/fcn/data/'+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'_ts.txt')
        lc_rewards = np.loadtxt('./baselines/ppoc/data/'+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'_rew_t.txt')* (1 / reward_scale)
        lc_ts = np.loadtxt('./baselines/ppoc/data/'+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'_ts_t.txt')


        fig, ax = plt.subplots()

        ax.plot(linear_ts, linear_rewards,  label='LINEAR')
        #ax.plot(fcn_ts, fcn_rewards,  label='fcn-1')
        ax.plot(lc_ts, lc_rewards,  label='lc')

        legend = ax.legend(loc='lower right')

        plt.savefig("./results/"+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'t.png')
