#!/usr/bin/env python3
import os
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common import tf_util as U
from baselines import logger
import numpy as np
import gym

def train(env_id,num_timesteps, seed, model_path=None, ratio=0.1):
    from baselines.scn import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=16, num_hid_layers=2)
    env = make_mujoco_env(env_id, seed)

    # parameters below were the best found in a simple random search
    # these are good enough to make humanoid walk, but whether those are
    # an absolute best or not is not certain
    env = RewScale(env, ratio)
    pi = pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=2048,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10,
            optim_stepsize=3e-4,
            optim_batchsize=64,
            gamma=0.99,
            lam=0.95,
            schedule='linear',
            env_name= env_id,
            seed = seed,
            scale = ratio
        )
    env.close()
    if model_path:
        U.save_state(model_path)

    return pi

class RewScale(gym.RewardWrapper):
    def __init__(self, env, scale):
        gym.RewardWrapper.__init__(self, env)
        self.scale = scale
    def reward(self, r):
        return r * self.scale

def main():
    logger.configure()
    parser = mujoco_arg_parser()
    parser.add_argument('--model-path', default=os.path.join(logger.get_dir(), 'humanoid_policy'))
    parser.set_defaults(num_timesteps=int(2e6))

    args = parser.parse_args()

    if not args.play:
        # train the model
        train(env_id = args.env,num_timesteps=args.num_timesteps, seed=args.seed, model_path=args.model_path, ratio=args.reward_scale)
    else:
        # construct the model object, load pre-trained model and render
        pi = train(env_id = args.env,num_timesteps=1, seed=args.seed)
        U.load_state(args.model_path)

        seeds = range(1, 16)
        stds = []
        means = []
        for seed in seeds:
            env = make_mujoco_env(args.env, seed=seed)

            ob = env.reset()

            eps = 0

            eprets = []
            rews = []
            while eps < args.num_timesteps:
                action = pi.act(stochastic=False, ob=ob)[0]
                ob, rew, done, _ = env.step(action)
                rews.append(rew)

                if done:
                    eps += 1
                    ob = env.reset()
                    epret = np.sum(rews)
                    # print(epret)
                    eprets.append(epret)
                    rews = []

            std = np.std(eprets)
            stds.append(std)
            mean = np.mean(eprets)
            means.append(mean)
            print("average reward: %f" % mean)
            print("Seed: %f" % seed)

        np.savetxt('./baselines/scn/data/'+args.env+'_s'+str(args.seed)+'_stds.txt', np.asarray(stds))
        np.savetxt('./baselines/scn/data/'+args.env+'_s'+str(args.seed)+'_means.txt', np.asarray(means))

if __name__ == '__main__':
    main()
