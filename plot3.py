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
algs = ['fcn', 'scn', 'linear', 'ppo1']

seeds = range(1,11)

performance = {}

for alg in algs:
    performance[alg] = []
    m = []
    std = []
    for seed in seeds:
        means = np.loadtxt('./baselines/' + alg + '/data/'+env_id+'_s'+str(seed)+'_means.txt')[0:14]
        m.append(np.mean(means))
        std.append(np.std(means))
        #tseeds = range(1,16)
        #plt.plot(tseeds, means, label="{}-{}".format(alg, seed))

    #plt.errorbar(seeds, m, std,  linestyle='None', marker='.', capsize=3, label="{}".format("mlp" if alg == 'ppo1' else alg))

    print("Algorithm: ", alg)
    print('Mean: ', np.mean(m))
    print('Std: ', np.mean(std))
    performance[alg].append(np.mean(std))

'''
seeds = range(1,16)
for alg in algs:
    m = []
    std = []
    for seed in seeds:
        means = np.loadtxt('./baselines/' + alg + '/data/'+env_id+'_s'+str(seed)+'_means.txt')[0:14]
        m.append(np.mean(means))
        std.append(np.std(means))
        #tseeds = range(1,16)
        #plt.plot(tseeds, means, label="{}-{}".format(alg, seed))

    #plt.errorbar(seeds, m, std,  linestyle='None', marker='.', capsize=3, label="{}".format("mlp" if alg == 'ppo1' else alg))

    print("Algorithm: ", alg)
    print('Mean: ', np.mean(m))
    print('Std: ', np.mean(std))
    performance[alg].append(np.mean(std))
'''

# data to plot
y_pos = np.arange(len(algs))
n_groups = 2

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.20
opacity = 0.8

a = index
for alg in algs:
    plt.bar(a, performance[alg], bar_width,
                 alpha=opacity,
                 label=alg)
    a = a + bar_width


plt.xlabel('Random Seeds')
plt.ylabel('Standard Deviation')
plt.title('Hopper-v2')
plt.xticks(index + bar_width, ('1-10', '1-15'))
plt.legend()

plt.tight_layout()
plt.show()