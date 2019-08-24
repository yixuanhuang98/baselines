from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from stable_baselines.sac.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import SAC
from baselines.common.distributions import make_pdtype, CategoricalPdType
import os
import argparse
import time
import numpy as np

#from half_cheetah_env import HalfCheetahEnv
from baselines.fcn1.logger import logger
from baselines.fcn1.model_based_rl import ModelBasedRL

class FcnPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, num_actors, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.cpdtype = cpdtype = CategoricalPdType(num_actors)
        self.pdtype = pdtype = make_pdtype(ac_space)
        #self.model_safe = SAC.load("/Users/huangyixuan/Downloads/racecar_1e5_3_2.9")
        self.model_free = SAC.load("/home/dingcheng/Downloads/racecar_free_5e5")
        #self.model_free = SAC.load("/home/gao-4144/yixuan/racecar_with_reward_1e5")
        self.mbrl = ModelBasedRL()
        self.mbrl._train_policy(self.mbrl._random_dataset)
        #logger.setup('debug')
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope('vf'):
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        with tf.variable_scope('dec'):
            last_out = obz
            last_out = tf.layers.dense(last_out, hid_size, name='dec', kernel_initializer=U.normc_initializer(1.0))

            # get the hidden_dicision
            # TODO: compare: the output layer be cpdtype.param_shape()[0]//2 or hid_size?
            hidden_decision = tf.layers.dense(last_out, num_actors, name="final", kernel_initializer=U.normc_initializer(0.01))

        # get the choice probability distribution
        self.cpd = cpdtype.pdfromflat(hidden_decision)
        #TODO: not sure of sampling or mode
        self.choice = ch =self.cpd.sample()

        with tf.variable_scope('pol'):
            last_outs = []
            actors = []
            for i in range(num_actors):
                last_outs.append(tf.layers.dense(obz, hid_size, name='sub%i'%(i+1), kernel_initializer=U.normc_initializer(1.0)))
                actors.append(tf.layers.dense(last_outs[i],pdtype.param_shape()[0]//2,name="final%i"%(i+1),kernel_initializer=U.normc_initializer(0.01)))

            self.actors = tf.stack(actors)

            ch = tf.reshape(ch,[-1])
            r = tf.range(tf.shape(ch)[0])
            ch = tf.cast(ch,tf.int32)

            ch_nd = tf.stack([ch,r],axis=1)

            mean = tf.gather_nd(self.actors, ch_nd)
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())

        self._act = U.function([stochastic, ob], [ac,ch,self.vpred])

    def act(self, stochastic, ob):
        ac1,ch1, vpred1 =  self._act(stochastic, ob[None])
        # print('choice')
        # print(ch1[0])  choice 0 and choice 1
        # if(ch1[0] == 1):
        #     ac1[:2], _ = self.model_safe.predict(ob)
        # else:
        #     ac1[:2], _ = self.model_free.predict(ob)
        if(ch1[0] == 1):
            #print("fcn_policy.py: get action")
            ac1[:2] = self.mbrl._policy.get_action(ob)

            # load the policy
            #saved_model_path = "/home/dingcheng/Documents/safe_learning/saved_models/1566617329"
            #policy = tf.contrib.saved_model.load_keras_model(saved_model_path)
            #ac1[0] = np.ndarray.tolist(policy.predict(np.array(ob).reshape(1,len(ob))))[0]
        # else:
        #     ac1[:2], _ = self.model_free.predict(ob)
        return ac1[0],ch1, vpred1[0]

    def pd_given_ch(self, choice,ac_space, gaussian_fixed_var=True):
        choice = tf.reshape(choice,[-1])
        choice = tf.cast(choice,tf.int32)

        r = tf.range(tf.shape(choice)[0])
        ch_nd = tf.stack([choice,r],axis=1)

        mean = tf.gather_nd(self.actors, ch_nd)

        with tf.variable_scope('pol'):
            logstd = tf.get_variable(name="logstd", shape=[1, self.pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            pdparams = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        return self.pdtype.pdfromflat(pdparams)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

