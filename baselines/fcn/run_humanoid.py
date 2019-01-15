#!/usr/bin/env python3
import os
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common import tf_util as U
from baselines import logger
import numpy as np

import gym

def train(env_id, num_timesteps, seed, model_path=None, num_actors=1, ratio=0.1):
    from baselines.fcn import fcn_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return fcn_policy.FcnPolicy(name=name, ob_space=ob_space, ac_space=ac_space, num_actors = num_actors,
            hid_size=64, num_hid_layers=2)
    env = make_mujoco_env(env_id, seed)

    # parameters below were the best found in a simple random search
    # these are good enough to make humanoid walk, but whether those are
    # an absolute best or not is not certain
    env = RewScale(env,ratio)
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

    # specify model path will save the model
    parser.add_argument('--model-path', default=os.path.join(logger.get_dir(), args.env+'_policy'))
    parser.add_argument('--obstd', default=0)
    parser.add_argument('--acstd', default=0)
    parser.set_defaults(num_timesteps=int(2e6))
    parser.set_defaults(num_actors=int(3))

    args = parser.parse_args()

    if not args.play:
        # train the model
        train(env_id = args.env,num_timesteps=args.num_timesteps, seed=args.seed, model_path=args.model_path, num_actors = args.num_actors, ratio=args.reward_scale)
    else:
        # construct the model object, load pre-trained model and render
        pi = train(num_timesteps=1, seed=args.seed)

        # load the saved model
        U.load_state(args.model_path)
        env = make_mujoco_env(args.env, seed=args.seed)

        ob = env.reset()

        timestep = 0

        rews = []

        while timestep < args.num_timesteps:
            action = pi.act(stochastic=False, ob=ob)[0]
            timestep += 1
            #TODO: Add noise to action
            if acstd:
                action = action + np.random.normal(0,arg.acstd,action.shape)

            ob, rew, done, _ =  env.step(action)
            rews.append(rew)
            #TODO: add noise to observation
            if obstd:
                ob = ob + np.random.normal(0,arg.obstd,ob.shape)

            env.render()
            if done:
                ob = env.reset()
                if obstd:
                    ob = ob + np.random.normal(0,arg.obstd,ob.shape)

        print("average reward: %d" % np.mean(rews))

if __name__ == '__main__':
    main()
