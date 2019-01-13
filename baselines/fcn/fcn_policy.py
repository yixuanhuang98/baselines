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
            print(self.scope)

    def _init(self, ob_space, ac_space, num_actors, hid_size, num_hid_layers, gaussian_fixed_var=True):
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
        #TODO: not sure of sampling or mode
        stochastic_ch = tf.placeholder(name="stu_ch", dtype=tf.bool, shape=())
        ch = U.switch(stochastic_ch, self.cpd.sample(), self.cpd.mode())

        with tf.variable_scope('pol'):
            last_outs = []
            actors = []
            params = []
            pdparams = []
            for i in range(num_actors):
                last_outs.append(tf.layers.dense(obz, hid_size, name='sub%i'%(i+1), kernel_initializer=U.normc_initializer(1.0)))

            ch = tf.reshape(ch,[-1])
            r = tf.range(tf.shape(ch)[0])
            ch = tf.cast(ch,tf.int32)
            ch_nd = tf.stack([ch,r],axis=1)

            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                for i in range(num_actors):
                    actors.append(tf.layers.dense(last_outs[i],pdtype.param_shape()[0]//2,name="final%i"%(i+1),kernel_initializer=U.normc_initializer(0.01)))
                    logstd = tf.get_variable(name="logstd%i"%(i+1), shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
                    pdparams.append(tf.concat([actors[i], actors[i] * 0.0 + logstd], axis=1))

                self.actors = tf.stack(pdparams)
                pdparam = tf.gather_nd(self.actors, ch_nd)

            else:
                for i in range(num_actors):
                    actors.append(tf.layers.dense(last_outs[i],pdtype.param_shape()[0],name="final%i"%(i+1),kernel_initializer=U.normc_initializer(0.01)))
                self.actors = tf.stack(actors)
                pdparam = tf.gather_nd(self.actors, ch_nd)

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []
        stochastic_ac = tf.placeholder(name="stu_ac", dtype=tf.bool, shape=())
        ac = U.switch(stochastic_ac, self.pd.sample(), self.pd.mode())

        self._act = U.function([stochastic_ch, stochastic_ac, ob], [ac,ch,self.vpred])

    def act(self, stochastic_ch, stochastic_ac, ob):
        ac1,ch1, vpred1 =  self._act(stochastic_ch, stochastic_ac, ob[None])
        return ac1[0],ch1, vpred1[0]

    def pd_given_ch(self, choice,ac_space, gaussian_fixed_var=True):
        choice = tf.reshape(choice,[-1])
        choice = tf.cast(choice,tf.int32)

        r = tf.range(tf.shape(choice)[0])
        ch_nd = tf.stack([choice,r],axis=1)
        pdparams = tf.gather_nd(self.actors, ch_nd)
        return self.pdtype.pdfromflat(pdparams)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []
