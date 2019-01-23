import os
import argparse
import matplotlib.pyplot as plt
import numpy as np


def plot_helper(seeds, reward_scale, alg, env_id):
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
    std = np.std(rws, axis=0)
    max = np.amax(rws, axis=0)
    mean = np.mean(rws, axis=0)
    return tss, std,max, mean

def plot(algs, reward_scale, env_id):

    seeds = [1,2,3,4,5]

    for alg in algs:
        ts,std,max, avg =plot_helper(seeds, reward_scale, alg, env_id)
        print(alg+" max reward: %f"%(np.max(max)))
        print(alg+" max mean reward: %f"%(np.max(avg)))
        plt.fill_between(ts,avg+std, avg-std, alpha=0.5)
        plt.plot(ts, avg, label="{}".format(alg))
        reward_scale = 0.1

    fontsize = 15
    plt.title(env_id,fontsize=fontsize+2)
    plt.legend(loc=4)
    labels = ['0.0', '0.4', '0.8', '1.2', '1.6','2.0']
    x = [0,400000,800000,1200000,1600000,2000000]
    plt.xticks(x, labels, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.xlabel('Number of Timesteps (M)', fontsize=fontsize)
    plt.ylabel('Rewards', fontsize=fontsize)

    plt.savefig("./plot/"+env_id+'.pdf',bbox_inches="tight")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e","--env_id", required=True, help="environment id")
    args = vars(parser.parse_args())

    seed_list = [1,2,3,4,5]
    reward_scale = [0.1,1.0]
    algs = ["fcn", "scn", "ppo1"]
    if args["env_id"] == "Humanoid":
        env_id = "Humanoid-v2"
        for seed in seed_list:
            os.system('python -m baselines.humanoid.run'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale[1]) + ' --env=' + env_id + " --model-path=model/fcn_"+env_id+str(seed))
            os.system('python  -m baselines.scn.run'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale[0]) + ' --env=' + env_id + " --model-path=model/fcn_"+env_id+str(seed))
            os.system('python  -m baselines.ppo1.run'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale[0]) + ' --env=' + env_id + " --model-path=model/fcn_"+env_id+str(seed))

        #plot(algs, reward_scale[1], env_id)

    elif args["env_id"] == "Walker2d" or args["env_id"] == "Swimmer":
        env_id = args["env_id"] + "-v2"
        for seed in seed_list:
            os.system('python  -m baselines.fcn.run'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale[1]) + ' --env=' + env_id + " --model-path=model/fcn_"+env_id+str(seed))
            #os.system('python  -m baselines.scn.run'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale[0]) + ' --env=' + env_id + " --model-path=model/fcn_"+env_id+str(seed))
            #os.system('python  -m baselines.ppo1.run'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale[0]) + ' --env=' + env_id + " --model-path=model/fcn_"+env_id+str(seed))
        
        #plot(algs, reward_scale[1], env_id)

    elif args["env_id"] == "InvertedPendulum" or args["env_id"] == "InvertedDoublePendulum":
        env_id = args["env_id"] + "-v2"
        for seed in seed_list:
            os.system('python  -m baselines.fcn.run'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale[0]) + ' --env=' + env_id + " --model-path=model/fcn_"+env_id+str(seed))
            os.system('python  -m baselines.scn.run'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale[0]) + ' --env=' + env_id + " --model-path=model/fcn_"+env_id+str(seed))
            os.system('python  -m baselines.ppo1.run'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale[0]) + ' --env=' + env_id + " --model-path=model/fcn_"+env_id+str(seed))
        plot(algs, reward_scale[0], env_id)

    elif args["env_id"] == "Hopper":
        env_id = "Hopper-v2"
        for seed in seed_list:
            os.system('python  -m baselines.fcn_alt.run'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale[0]) + ' --env=' + env_id + " --model-path=model/fcn_"+env_id+str(seed))
            #os.system('python  -m baselines.scn.run'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale[0]) + ' --env=' + env_id + " --model-path=model/fcn_"+env_id+str(seed))
            #os.system('python  -m baselines.ppo1.run'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale[0]) + ' --env=' + env_id + " --model-path=model/fcn_"+env_id+str(seed))
        #plot(algs, reward_scale[0], env_id)

    elif args["env_id"] == "HalfCheetah":
        env_id = "HalfCheetah-v2"
        for seed in seed_list:
            os.system('python  -m baselines.nlfcn.run'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale[1]) + ' --env=' + env_id + " --model-path=model/fcn_"+env_id+str(seed))
            #os.system('python  -m baselines.scn.run'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale[0]) + ' --env=' + env_id + " --model-path=model/fcn_"+env_id+str(seed))
            #os.system('python  -m baselines.ppo1.run'+ ' --seed='+str(seed) + ' --reward_scale='+str(reward_scale[0]) + ' --env=' + env_id + " --model-path=model/fcn_"+env_id+str(seed))
        plot(algs, reward_scale[1], env_id)

    else:
        print("Environment not included. Valid environment names: \n Humanoid\n Walker2d\n Swimmer\n InvertedPendulum\n InvertedDoublePendulum\n Hopper\n HalfCheetah\n")


if __name__ == '__main__':
    main()
