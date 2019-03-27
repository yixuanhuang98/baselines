import os
import time
import numpy as np
import os.path as osp
from baselines import logger
from collections import deque
from baselines.common import explained_variance, set_global_seeds
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common.policies import build_policy
from baselines.ppo2.mb_model import MBModel
import tensorflow as tf
import baselines.common.tf_util as U
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments



import argparse
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def load(data_dir,num_files):
    Ss = np.load(data_dir+"obs_1.npy")
    As = np.load(data_dir+"actions_1.npy")
    Ds = np.load(data_dir+"dones_1.npy")


    for i in range(1,num_files-1):
        cur_s = np.load(data_dir+"obs_"+str(i+1)+".npy")
        cur_a = np.load(data_dir+"actions_"+str(i+1)+".npy")
        cur_d = np.load(data_dir+"dones_"+str(i+1)+".npy")
        Ss = np.concatenate((Ss,cur_s),axis=0)
        As = np.concatenate((As,cur_a),axis=0)
        Ds = np.concatenate((Ds,cur_d),axis=0)

    Ss = np.reshape(Ss,(Ss.shape[0],-1))
    As = np.reshape(As,(As.shape[0],-1))
    Ds = np.reshape(Ds,(Ds.shape[0],-1))

    index = np.ravel(np.argwhere(Ds))

    print(Ss.shape)

    S = np.delete(Ss,index,0)[:-1]
    A = np.delete(As,index,0)[:-1]
    index = index + 1
    S_primes = np.delete(Ss,index,0)[1:]

    print(S_primes.shape)

    return S,A, S_primes


def generate(model,sas,env,horizon):
    #ob = env.reset()
    #sps = np.array([ob for _ in range(horizon)])
    sps = []
    print(sas)
    i = 0
    for s,a in sas:
        sp = model.predict(s,a)
        sps.append(sp)
        i += 1

    print(i)

    sps = np.asarray(sps)
    sps = np.reshape(sps,(sps.shape[0],-1))
    print(sps.shape)
    return sps


def train(num_iter,ss,acs,s_primes,env,save_dir,lr=0.01,adam_epsilon=1e-5):

    ob_space = env.observation_space
    ac_space = env.action_space

    model = MBModel("model",ob_space,ac_space,3,256)


    batch_size = 128
    num_batch =  s_primes.shape[0] // batch_size



    for i in range(int(num_iter)):
        print("-----------------iteration"+str(i)+"---------------------------")

        loss_list = []

        for j in range(num_batch-1):
            xs = ss[j*batch_size:(j+1)*batch_size]
            xa = acs[j*batch_size:(j+1)*batch_size]

            y = s_primes[j*batch_size:(j+1)*batch_size]

            #*loss, g = lossandgrad(xs,xa,y)
            #self.adam.update(g, lr)
            #grad = grad(x,y)
            #print(grad)
            loss = model.train(xs,xa,y)
            loss_list.append(*loss)
        print(np.mean(loss_list))

    return model



def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)\

    parser.add_argument('--num_timesteps', type=int, default=1e2)
    parser.add_argument('--data_dir',type=str,default="./mb_data/")
    parser.add_argument('--num_files',type=int,default=1)
    parser.add_argument('--save_dir',type=str,default=None)
    parser.add_argument('--env',type=str,default='Ant-v2')
    parser.add_argument('--lr',type=float,default=0.01)

    args = parser.parse_args()

    Ss,As, S_primes = load(args.data_dir, args.num_files)

    env = make_mujoco_env(args.env, seed=0)

    model = train(args.num_timesteps,Ss,As,S_primes,env,args.save_dir)

    if (args.save_dir != None):
        U.save_state(args.save_dir)

if __name__ == '__main__':
    main()
