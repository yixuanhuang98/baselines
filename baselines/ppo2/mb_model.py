import tensorflow as tf
import functools

from baselines.common.tf_util import get_session, save_variables, load_variables
from baselines.common.tf_util import initialize
from baselines.common.mpi_running_mean_std import RunningMeanStd

import baselines.common.tf_util as U
import gym
from baselines.common.distributions import make_pdtype
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments

try:
    from baselines.common.mpi_adam_optimizer import MpiAdamOptimizer
    from mpi4py import MPI
    from baselines.common.mpi_util import sync_from_root
except ImportError:
    MPI = None

class MBModel(object):
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)


    def _init(self,ob_space,ac_space,num_hidden,hid_size,lr=0.01,adam_epsilon=1e-5):

        with tf.variable_scope('mb_model', reuse=tf.AUTO_REUSE):
            assert isinstance(ob_space, gym.spaces.Box)

            sequence_length = None

            ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
            ac = U.get_placeholder(name="ac", dtype=tf.float32, shape=[sequence_length] + list(ac_space.shape))

            sa = tf.concat([ob,ac],1)

            last_out = sa

            for i in range(num_hidden):
                last_out = tf.nn.relu(tf.layers.dense(last_out, hid_size, name='fc%i'%(i+1), kernel_initializer=U.normc_initializer(1.0)))

            self.prediction = tf.layers.dense(last_out, ob_space.shape[0], name='final', kernel_initializer=U.normc_initializer(0.01))


        self.scope = tf.get_variable_scope().name

        self._predict = U.function([ob,ac], [self.prediction])

        target = U.get_placeholder(name="tg", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        loss = tf.reduce_mean(tf.square(target - self.prediction))
        var_list = self.get_trainable_variables()

        self._lossandgrad = U.function([ob,ac,target], [loss]+[U.flatgrad(loss, var_list)])
        self.adam = MpiAdam(var_list, epsilon=adam_epsilon)
        U.initialize()
        self.adam.sync()

    def predict(self,ob,ac):
        s_prime = self._predict(ob,ac)
        return s_prime

    def train(self,ob,ac,target,lr=0.01):
        *loss, g = self._lossandgrad(ob,ac,target)
        self.adam.update(g, lr)
        return loss

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []
