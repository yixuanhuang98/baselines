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


def plot2(seeds, reward_scale, alg, env_id, e):
    rws = []
    tss = -1
    max_len = -1
    for seed in seeds:

        rw = np.loadtxt('./baselines/'+alg+'/data/'+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'_rew'+e+'.txt') * (1 / reward_scale)
        if rw.size > max_len:
            max_len = rw.size
            tss = np.loadtxt('./baselines/'+alg+'/data/'+env_id+'_s'+str(seed)+'_r'+str(reward_scale)+'_ts'+e+'.txt')
        rws.append(rw)

    for i in range(len(rws)):
        if rws[i].size < max_len:
            rws[i] = np.pad(rws[i],(0,max_len-rws[i].size),'edge')
    rws = np.stack(rws)
    max = np.amax(rws, axis=0)
    min = np.amin(rws, axis=0)
    mean = np.mean(rws, axis=0)
    return tss, max, min, mean


env_ids = ["Walker2d-v2", "Hopper-v2", "Swimmer-v2", "HalfCheetah-v2", "Humanoid-v2"]
seeds = range(1,16)
reward_scale = 0.1
alg = 'scn'
for env_id in env_ids:
    ts, ma, mi, avg = plot2(seeds, reward_scale, alg, env_id, "")
    print(env_id + ": ", np.amax(avg))
