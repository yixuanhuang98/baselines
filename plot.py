import numpy as np
import matplotlib.pyplot as plt

env_id = "Swimmer-v2"
seeds = [1]
reward_scales=[0.1,0.5,1.0]
for seed in seeds:
    for reward_scale in reward_scales:
        fcn_rewards = np.loadtxt('./baselines/fcn/data/'+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'_rew.txt') * (1 / reward_scale)
        fcn_ts = np.loadtxt('./baselines/fcn/data/'+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'_ts.txt')
        scn_rewards = np.loadtxt('./baselines/scn/data/'+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'_rew.txt')* (1 / reward_scale)
        scn_ts = np.loadtxt('./baselines/scn/data/'+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'_ts.txt')
        mlp_rewards = np.loadtxt('./baselines/ppo1/data/'+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'_rew.txt')* (1 / reward_scale)
        mlp_ts = np.loadtxt('./baselines/ppo1/data/'+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'_ts.txt')
        lin_rewards = np.loadtxt('./baselines/linear/data/'+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'_rew.txt')* (1 / reward_scale)
        lin_ts = np.loadtxt('./baselines/linear/data/'+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'_ts.txt')
        nlfcn_rewards = np.loadtxt('./baselines/nlfcn/data/'+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'_rew.txt')* (1 / reward_scale)
        nlfcn_ts = np.loadtxt('./baselines/nlfcn/data/'+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'_ts.txt')



        fig, ax = plt.subplots()

        ax.plot(fcn_ts, fcn_rewards,  label='l-FCN with 3 actors')
        ax.plot(scn_ts, scn_rewards,  label='SCN16')
        ax.plot(mlp_ts, mlp_rewards,  label='MLP64')
        ax.plot(lin_ts, lin_rewards,  label='LINEAR')
        ax.plot(nlfcn_ts, nlfcn_rewards,  label='nl-FCN with 6 actors')


        legend = ax.legend(loc='lower right')

        plt.savefig("./results/"+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'.png')
