import os

import numpy as np
import matplotlib.pyplot as plt

from baselines.fcn1.model_based_policy import ModelBasedPolicy
import baselines.fcn1.utils
from baselines.fcn1.logger import logger
from baselines.fcn1.timer import timeit
import pybullet_envs.bullet.racecarGymEnv as e
#import pybullet_envs.bullet.minitaur_gym_env as e
import gym
import tensorflow as tf
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC

EP_MAX = 1000
EP_LEN = 500
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 1,1
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization

class PPO(object):  # ppo for choice model

    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


class ModelBasedRL(object):

    def __init__(self,
                 num_init_random_rollouts=15, # pre : 10 
                 max_rollout_length=500, # pre : 500
                 num_onplicy_iters=10,
                 num_onpolicy_rollouts=1000,
                 training_epochs=300,   # pre:60
                 training_batch_size=512,
                 render=False,
                 mpc_horizon=15,  # pre 15
                 num_random_action_selection=4096, # pre 4096
                 nn_layers=3):
        #self._env1 = env
        self._env = e.RacecarGymEnv(isDiscrete=False ,renders=False) #env\
        #self._env = env = e.MinitaurBulletEnv(render=False)
        #self._env = gym.make('Ant-v3').unwrapped    # action dimention is 8
        self.cost_func = self._env.choice_cost_func
        self._max_rollout_length = max_rollout_length
        self._num_onpolicy_iters = num_onplicy_iters
        self._num_onpolicy_rollouts = num_onpolicy_rollouts
        self._training_epochs = training_epochs
        self._training_batch_size = training_batch_size
        self._render = render
        self.total_prediction = 6
        self.ppo_choice = PPO()

        #logger.info('Gathering random dataset')
        #self._random_dataset = self._gather_rollouts(baselines.fcn1.utils.RandomPolicy(self._env),
        #                                             num_init_random_rollouts)

        # train the model
        self.model = SAC(MlpPolicy, self._env, verbose=1)
        import stable_baselines
        self._random_dataset = self.model.learn(total_timesteps=1000, log_interval=10, num_init_random_rollouts=num_init_random_rollouts)
        self.model.save("racecar_3e5_free_3")

        #logger.info('Creating policy')
        self._policy = ModelBasedPolicy(self._env,
                                        self._random_dataset,
                                        horizon=mpc_horizon,
                                        num_random_action_selection=num_random_action_selection)

        timeit.reset()
        timeit.start('total')

    # def _gather_rollouts(self, policy, num_rollouts):
    #     dataset = utils.Dataset()

    #     for _ in range(num_rollouts):
    #         pre_state = self._env.reset()
    #         #pre_state = state
    #         action = policy.get_action(pre_state)
    #         state, reward, done, _ = self._env.step(action)
    #         print(state)
    #         state_gap = state
    #         print(state_gap)
    #         state_gap = state - pre_state
    #         print(state_gap)
    #         done = False
    #         t = 0
    #         while not done:
    #             if self._render:
    #                 timeit.start('render')
    #                 self._env.render()
    #                 timeit.stop('render')
    #             timeit.start('get action')
    #             action = policy.get_action(state)
    #             timeit.stop('get action')
    #             timeit.start('env step')
    #             next_state, reward, done, _ = self._env.step(action)
    #             timeit.stop('env step')
    #             done = done or (t >= self._max_rollout_length)
    #             dataset.add(state[2:], action, next_state, reward, done)
    #             pre_state = state
    #             state = next_state
    #             state_gap = state
    #             state_gap = state - pre_state
    #             t += 1

    #     return dataset

    def _gather_rollouts_choice(self, policy, num_rollouts):
        dataset = baselines.fcn1.utils.Dataset()
        model_free = SAC.load("racecar_3e5_reward")
        total_ob = []
        total_ac = []
        for ep in range(num_rollouts):
            state = self._env.reset()
            done = False
            t = 0
            print(ep)
            while not done:
                if self._render:
                    timeit.start('render')
                    self._env.render()
                    timeit.stop('render')
                this_state = state
                total_safe_cost = 0
                this_state_free = state
                total_free_cost = 0
                for i in range(self.total_prediction):
                    this_action = policy.get_action(this_state)
                    next_state = policy.predict(this_state[2:],this_state,this_action)
                    total_safe_cost += self.cost_func(this_state,this_action,next_state)
                    this_state = next_state
                    
                    this_action_free = policy.get_action(this_state_free)
                    next_state_free = policy.predict(this_state_free[2:], this_state_free, this_action_free)
                    total_free_cost += self.cost_func(this_state_free, this_action_free, next_state_free)
                    this_state_free = next_state_free
                
                timeit.start('get action')
                choice_result = np.zeros((1,1))
                if(total_safe_cost <= total_free_cost):
                    action = policy.get_action(state)
                    choice_result[0][0] = 1
                elif(total_safe_cost > total_free_cost):
                    action, _ = model_free.predict(state)
                    choice_result[0][0] = -1
                timeit.stop('get action')
                timeit.start('env step')
                next_state, reward, done, _ = self._env.step(action)
                timeit.stop('env step')
                if(ep >= 0):
                    total_ob.append(state)
                    total_ac.append(choice_result[0])
                done = done or (t >= self._max_rollout_length)
                ori_state = state[2:]
                truth_state = state
                dataset.add(ori_state, truth_state, action, next_state, reward, done)
                state = next_state
                t += 1
            if(True):
                final_output = []
                final_output = np.concatenate((total_ob,total_ac),axis = 1)
                np.savetxt('/home/gao-4144/yixuan/txt_result/switch_12',(final_output))

        return dataset

    def _gather_rollouts_q2(self, policy, num_rollouts):
        dataset = baselines.fcn1.utils.Dataset()
        random_policy = baselines.fcn1.utils.RandomPolicy(self._env)
        model_free = SAC.load("racecar_3e5_reward")
        all_ep_r = []
        total_ob = []
        total_ac = []
        for ep in range(num_rollouts):
            state = self._env.reset()
            done = False
            t = 0
            print('reset')
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0
            while not done:
                #if(num_rollouts == self._num_onpolicy_rollouts):
                #print(state[:2])
                choice_action = self.ppo_choice.choose_action(state[1])
                if(choice_action >= 0):
                    action = policy.get_action(state)
                else:
                    action, _ = model_free.predict(state)
                # if(num_rollouts == self._num_onpolicy_rollouts):
                #     print(state[:2])
                #     random_action = random_policy.get_action(state)
                if self._render:
                    timeit.start('render')
                    self._env.render()
                    timeit.stop('render')
                # timeit.start('get action')
                # action = policy.get_action(state)
                # timeit.stop('get action')
                # if(num_rollouts == self._num_onpolicy_rollouts and abs(state[1]) < 1 ):
                #     action = random_action
                # print('action')
                # print(action)
                timeit.start('env step')
                next_state, reward, done, _ = self._env.step(action)
                timeit.stop('env step')
                if(ep >= 10):
                    total_ob.append(state)
                    total_ac.append(action)
                buffer_s.append(state[1])
                buffer_a.append(choice_action)
                buffer_r.append((reward+8)/8)
                ep_r += reward
                done = done or (t >= self._max_rollout_length)
                ori_state = state[2:]
                truth_state = state
                dataset.add(ori_state, truth_state, action, next_state, reward, done)

                state = next_state
                if (t+1) % BATCH == 0 or t == EP_LEN-1 :
                    v_s_ = self.ppo_choice.get_v(next_state)
                    discounted_r = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.ppo_choice.update(bs, ba, br)
                t += 1
            if(ep>=30 and ep % 5 == 0):
                final_output = []
                final_output = np.concatenate((total_ob,total_ac),axis = 1)
                np.savetxt('/home/gao-4144/yixuan/txt_result/switch_8',(final_output))
            if ep == 0: all_ep_r.append(ep_r)
            else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
            print(
                'Ep: %i' % ep,
                "|Ep_r: %i" % ep_r,
                ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
            )

        return dataset

    def _gather_rollouts(self, policy, num_rollouts):
        dataset = baselines.fcn1.utils.Dataset()
        random_policy = baselines.fcn1.utils.RandomPolicy(self._env)
        total_ob = []
        for _ in range(num_rollouts):
            state = self._env.reset()
            done = False
            t = 0
            print('reset')
            while not done:
                if(num_rollouts == self._num_onpolicy_rollouts):
                    print(state[:2])
                    total_ob.append(state)
                    random_action = random_policy.get_action(state)
                if self._render:
                    timeit.start('render')
                    self._env.render()
                    timeit.stop('render')
                timeit.start('get action')
                action = policy.get_action(state)
                # print('action')
                # print(action)
                timeit.stop('get action')
                if(num_rollouts == self._num_onpolicy_rollouts and abs(state[1]) < 2 ):
                    action = random_action
                timeit.start('env step')
                next_state, reward, done, _ = self._env.step(action)
                timeit.stop('env step')
                done = done or (t >= self._max_rollout_length)
                ori_state = state[2:]
                truth_state = state
                dataset.add(ori_state, truth_state, action, next_state, reward, done)

                state = next_state
                t += 1
            np.savetxt('/home/dingcheng/Documents/safe_learning/txt_result/no_switch_2',total_ob)
        

        return dataset


    def _train_policy(self, dataset):
        """
        Train the model-based policy

        implementation details:
            (a) Train for self._training_epochs number of epochs
            (b) The dataset.random_iterator(...)  method will iterate through the dataset once in a random order
            (c) Use self._training_batch_size for iterating through the dataset
            (d) Keep track of the loss values by appending them to the losses array
        """
        timeit.start('train policy')

        losses = []
        ### PROBLEM 1
        ### YOUR CODE HERE
        for i in range(self._training_epochs):
            for states, truth_states, actions, next_states, rewards, dones in dataset.random_iterator(self._training_batch_size):
                losses.append(self._policy.train_step(states, truth_states, actions, next_states))

        #logger.record_tabular('TrainingLossStart', losses[0])
        #logger.record_tabular('TrainingLossFinal', losses[-1])

        timeit.stop('train policy')

    def _log(self, dataset):
        timeit.stop('total')
        dataset.log()
        logger.dump_tabular(print_func=logger.info)
        logger.debug('')
        for line in str(timeit).split('\n'):
            logger.debug(line)
        timeit.reset()
        timeit.start('total')

    def run_q1(self):
        """
        Train on a dataset, and see how good the learned dynamics model's predictions are.

        implementation details:
            (i) Train using the self._random_dataset
            (ii) For each rollout, use the initial state and all actions to predict the future states.
                 Store these predicted states in the pred_states list.
                 NOTE: you should *not* be using any of the states in states[1:]. Only use states[0]
            (iii) After predicting the future states, we have provided plotting code that plots the actual vs
                  predicted states and saves these to the experiment's folder. You do not need to modify this code.
        """
        logger.info('Training policy....')
        ### PROBLEM 1
        ### YOUR CODE HERE
        self._train_policy(self._random_dataset)

        logger.info('Evaluating predictions...')
        # for r_num, (states, actions, _, _, _) in enumerate(self._random_dataset.rollout_iterator()):
        #     pred_states = []

        #     ### PROBLEM 1
        #     ### YOUR CODE HERE
        #     pred_state = states[0]
        #     for act in actions:
        #         pred_state = self._policy.predict(pred_state, act)
        #         pred_states.append(pred_state)

        #     states = np.asarray(states)
        #     pred_states = np.asarray(pred_states)
            

        #     state_dim = states.shape[1]
        #     rows = int(np.sqrt(state_dim))
        #     cols = state_dim // rows
        #     f, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
        #     f.suptitle('Model predictions (red) versus ground truth (black) for open-loop predictions')
        #     for i, (ax, state_i, pred_state_i) in enumerate(zip(axes.ravel(), states.T, pred_states.T)):
        #         ax.set_title('state {0}'.format(i))
        #         ax.plot(state_i, color='k')
        #         ax.plot(pred_state_i, color='r')
        #     plt.tight_layout()
        #     plt.subplots_adjust(top=0.90)
        #     f.savefig(os.path.join(logger.dir, 'prediction_{0:03d}.jpg'.format(r_num)), bbox_inches='tight')

        for r_num, (states, truth_states, actions, _, _, _) in enumerate(self._random_dataset.rollout_iterator()):
            pred_states = []

            ### PROBLEM 1
            ### YOUR CODE HERE
            pred_state = truth_states[0]
            for act in actions:
                pred_state = self._policy.predict(pred_state[2:], pred_state, act)
                pred_states.append(pred_state)

            states = np.asarray(truth_states)
            pred_states = np.asarray(pred_states)
            

            state_dim = states.shape[1]
            rows = int(np.sqrt(state_dim))
            cols = state_dim // rows
            f, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
            f.suptitle('Model predictions (red) versus ground truth (black) for open-loop predictions')
            for i, (ax, state_i, pred_state_i) in enumerate(zip(axes.ravel(), states.T, pred_states.T)):
                ax.set_title('state {0}'.format(i))
                ax.plot(state_i, color='k')
                ax.plot(pred_state_i, color='r')
            plt.tight_layout()
            plt.subplots_adjust(top=0.90)
            f.savefig(os.path.join(logger.dir, 'prediction_{0:03d}.jpg'.format(r_num)), bbox_inches='tight')


        logger.info('All plots saved to folder')

    def run_q2(self):
        """
        Train the model-based policy on a random dataset, and evaluate the performance of the resulting policy
        """
        logger.info('Random policy')
        self._log(self._random_dataset)

        logger.info('Training policy....')
        ### PROBLEM 2
        ### YOUR CODE HERE
        self._train_policy(self._random_dataset)

        logger.info('Evaluating policy...')
        ### PROBLEM 2
        ### YOUR CODE HERE
        eval_dataset = self._gather_rollouts_choice(self._policy, self._num_onpolicy_rollouts)

        logger.info('Trained policy')
        self._log(eval_dataset)

    def run_q3(self):
        """
        Starting with the random dataset, train the policy on the dataset, gather rollouts with the policy,
        append the new rollouts to the existing dataset, and repeat
        """
        dataset = self._random_dataset

        itr = -1
        logger.info('Iteration {0}'.format(itr))
        logger.record_tabular('Itr', itr)
        self._log(dataset)

        for itr in range(self._num_onpolicy_iters + 1):
            logger.info('Iteration {0}'.format(itr))
            logger.record_tabular('Itr', itr)

            ### PROBLEM 3
            ### YOUR CODE HERE
            logger.info('Training policy...')
            self._train_policy(dataset)

            ### PROBLEM 3
            ### YOUR CODE HERE
            logger.info('Gathering rollouts...')
            new_dataset = self._gather_rollouts(self._policy, self._num_onpolicy_rollouts)

            ### PROBLEM 3
            ### YOUR CODE HERE
            logger.info('Appending dataset...')
            dataset.append(new_dataset)

            self._log(new_dataset)



