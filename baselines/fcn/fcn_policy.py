from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype, CategoricalPdType

class FcnPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        self.name = name
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, num_actors, hid_size, num_hid_layers, masks, gaussian_fixed_var=True):

        self.masks = tf.stack(masks)

        assert isinstance(ob_space, gym.spaces.Box)

        self.cpdtype = cpdtype = CategoricalPdType(num_actors)
        self.pdtype = pdtype = make_pdtype(ac_space)
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
        ch = self.cpd.sample()

        with tf.variable_scope('pol'):

            self.h = last_out = tf.layers.dense(obz, hid_size, name='fc1', kernel_initializer=U.normc_initializer(1.0))

            ch = tf.reshape(ch,[-1])
            masks = tf.gather(self.masks, ch)
            masks = tf.cast(masks,tf.float32)

            last_out = tf.multiply(last_out, masks)

            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = tf.layers.dense(last_out,pdtype.param_shape()[0]//2,name="fc2",kernel_initializer=U.normc_initializer(0.01))
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='fc2', kernel_initializer=U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []
        stochastic = tf.placeholder(dtype=tf.bool, shape=())

        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())

        self._act = U.function([stochastic, ob], [ac,ch,self.vpred])

    def act(self, stochastic, ob):
        ac1,ch1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0],ch1, vpred1[0]

    def pd_given_ch(self, choice,ac_space, gaussian_fixed_var=True):
        choice = tf.reshape(choice,[-1])
        masks = tf.gather(self.masks, choice)
        masks = tf.cast(masks,tf.float32)

        last_out = tf.multiply(self.h, masks)
        with tf.variable_scope(self.name, reuse=True):
            with tf.variable_scope('pol', reuse=True):
                if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                    mean = tf.layers.dense(last_out,self.pdtype.param_shape()[0]//2,name="fc2",kernel_initializer=U.normc_initializer(0.01))
                    logstd = tf.get_variable(name="logstd", shape=[1, self.pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
                    pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
                else:
                    pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='fc2', kernel_initializer=U.normc_initializer(0.01))
        return self.pdtype.pdfromflat(pdparam)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []
