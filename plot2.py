import glob
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt

matplotlib.rcParams.update({'font.size': 8})

def plot(seeds, reward_scale, alg, env_id):
    rws = []
    tss = -1
    max_len = -1
    for seed in seeds:

        rw = np.loadtxt('./baselines/'+alg+'/data/'+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'_rew.txt') * (1 / reward_scale)
        if rw.size > max_len:
            max_len = rw.size
            tss = np.loadtxt('./baselines/'+alg+'/data/'+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'_ts.txt')
        rws.append(rw)

    for i in range(len(rws)):
        if rws[i].size < max_len:
            rws[i] = np.pad(rws[i],(0,max_len-rws[i].size),'edge')
    rws = np.stack(rws)
    max = np.amax(rws, axis=0)
    min = np.amin(rws, axis=0)
    mean = np.mean(rws, axis=0)
    return tss, max, min, mean




env_id = "Hopper-v2"
#seeds = [1,2,3,4,5]
#reward_scale=0.1
algs = ['fcn']
seeds = range(11,16)
plt.title("Hopper-v2")

#plt.subplot(1,2,1)
m = []
for seed in seeds:
    for alg in algs:
        means = np.loadtxt('./baselines/' + alg + '/data/'+env_id+'_s'+str(seed)+'_means.txt')
        m.append(np.mean(means))
        tseeds = range(1,16)
        plt.plot(tseeds, means, label="{}-{}".format(alg, seed))

print('Mean: ', np.mean(m))
print('Std: ', np.std(m))

plt.legend(loc=4)
plt.xlabel('Random Seed')
plt.ylabel('Mean Reward')
axes = plt.gca()
axes.set_ylim([2000,3800])




#plt.subplot(1,2,2)
#for seed in seeds:
#    for alg in algs:
#        stds = np.loadtxt('./baselines/' + alg + '/data/'+env_id+'_s'+str(seed)+'_stds.txt')
#        tseeds = range(1,16)
#        plt.plot(tseeds, stds, label="{}-{}".format(alg, seed))

#plt.legend(loc=4)
#plt.xlabel('Random Seed')
#plt.ylabel('Standard Deviation')
#axes = plt.gca()
#axes.set_ylim([0, 1200])

plt.show()

#plt.savefig("./check/"+env_id+'.png')
