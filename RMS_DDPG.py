"""
Reward Memory Shaping Deep Deterministic Policy Gradient (RMS_DDPG)
-----------------------------------------
tensorflow >=2.0.0
tensorlayer >=2.0.0
------

"""
import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl


#####################  hyper parameters  ####################

RANDOMSEED = 1037
LR_A = 0.005                # learning rate for actor
LR_C = 0.005                # learning rate for critic
GAMMA = 0.95                # reward discount
TAU = 0.01                  # soft replacement
MEMORY_CAPACITY = 10000     # size of replay buffer
BATCH_SIZE = 32             # update batchsize
MAX_EPISODES = 500          # total number of episodes for training
VAR = 2                    # control exploration
HIDDEN_SIZE = 32

############################### RMS_DDPG  ####################################

class RMS_DDPG(object):
    """
    DDPG class
    """
    def __init__(self, s_dim, a_dim):
        # memory用于储存数据
        # 保存个数MEMORY_CAPACITY，s_dim * 2 + a_dim + 1：分别是两个state，一个action，和一个reward
        self.memory = np.zeros((MEMORY_CAPACITY, 1 ,s_dim * 2 + a_dim + 1), dtype=np.float32) # 增加一个维度适应lstm的输入
        self.pointer = 0
        self.a_dim, self.s_dim = a_dim, s_dim

        W_init = tf.random_normal_initializer(mean=0, stddev=0.5)
        b_init = tf.constant_initializer(0.1)

        self.actor_loss = []
        self.critic_loss = []

        # 建立actor网络，输入s，输出a (None,a_dim)
        def get_actor(input_state_shape,name = ''):
            """
            Build actor network
            :param input_state_shape: state
            :param name: name
            :return: action
            """
            inputs = tl.layers.Input(input_state_shape,name="Actor_input")
            rnn_out, _  = tl.layers.LSTMRNN(units=HIDDEN_SIZE,return_last_output=True)(inputs)
            rnn_out = tl.layers.Reshape(shape=[-1,HIDDEN_SIZE],name='A_reshape')(rnn_out)
            # print(rnn_out)
            x = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A1')(rnn_out)
            x = tl.layers.Dense(n_units=a_dim, act=tf.nn.tanh, W_init=W_init, b_init=b_init, name='A_out')(x)
            # x = tl.layers.Lambda(lambda x: np.array(NODE_NUM-1) * x)(x)  
            return tl.models.Model(inputs=inputs, outputs=x, name='Actor'+ name)

        #建立Critic网络，输入s，a。输出Q值
        def get_critic(input_state_shape, input_action_shape, name=''):
            """
            Build critic network
            :param input_state_shape: state
            :param input_action_shape: act
            :param name: name
            :return: Q value Q(s,a)
            """
            s = tl.layers.Input(input_state_shape, name='C_s_input')
            a = tl.layers.Input(input_action_shape, name='C_a_input')
            x = tl.layers.Concat(2)([s, a])
            rnn_out , _  = tl.layers.LSTMRNN(units=HIDDEN_SIZE,return_last_output=True)(x)
            rnn_out = tl.layers.Reshape(shape=[-1,HIDDEN_SIZE],name='C_reshape')(rnn_out)
            x = tl.layers.Dense(n_units=64, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l1')(rnn_out)
            # x = tl.layers.Dense(n_units=128, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l2')(x)
            x = tl.layers.Dense(n_units=1, W_init=W_init, b_init=b_init, name='C_out')(x)
            return tl.models.Model(inputs=[s, a], outputs=x, name='Critic' + name)

        self.actor = get_actor([None, None, s_dim])
        self.critic = get_critic([None, None,s_dim], [None, None, a_dim])
        self.actor.train()
        self.critic.train()

        #更新参数，只用于首次赋值，之后就没用了
        def copy_para(from_model, to_model):
            """
            Copy parameters for soft updating
            :param from_model: latest model
            :param to_model: target model
            :return: None
            """
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        #建立actor_target网络，并和actor参数一致，不能训练
        self.actor_target = get_actor([None, None,s_dim], name='_target')
        copy_para(self.actor, self.actor_target)
        self.actor_target.eval()

        #建立critic_target网络，并和actor参数一致，不能训练
        self.critic_target = get_critic([None,None, s_dim], [None, None,a_dim], name='_target')
        copy_para(self.critic, self.critic_target)
        self.critic_target.eval()

        self.R = tl.layers.Input([None, 1], tf.float32, 'r')

        #建立ema，滑动平均值
        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

        self.actor_opt = tf.optimizers.Adam(LR_A)
        self.critic_opt = tf.optimizers.Adam(LR_C)

    def ema_update(self):
        """
        滑动平均更新
        """
        # 其实和之前的硬更新类似，不过在更新赋值之前，用一个ema.average。
        paras = self.actor.trainable_weights + self.critic.trainable_weights    #获取要更新的参数包括actor和critic的
        self.ema.apply(paras)                                                   #主要是建立影子参数
        for i, j in zip(self.actor_target.trainable_weights + self.critic_target.trainable_weights, paras):
            i.assign(self.ema.average(j))                                       # 用滑动平均赋值

    # 选择动作，把s带进入，输出a
    def choose_action(self, s):
        """
        Choose action
        :param s: state
        :return: act (None,a_dim)
        """
        return self.actor(np.array(s, dtype=np.float32))

    def learn(self):
        """
        Update parameters
        :return: None
        """
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)    #随机BATCH_SIZE个随机数
        bt = self.memory[indices, :,:]                    #根据indices，选取数据bt，相当于随机
        bs = bt[:, : ,:self.s_dim]                         #从bt获得数据s 维度扩展
        ba = bt[:, :, self.s_dim:self.s_dim + self.a_dim]  #从bt获得数据a
        br = bt[:, :, -self.s_dim - 1:-self.s_dim]         #从bt获得数据r
        bs_ = bt[:, :, -self.s_dim:]                       #从bt获得数据s'

        # Critic：
        # Critic更新和DQN很像，不过target不是argmax了，是用critic_target计算出来的。
        # br + GAMMA * q_
        with tf.GradientTape() as tape:
            a_ = self.actor_target(bs_)
            a_ = np.expand_dims(a_,axis=1) # 维度扩展
            q_ = self.critic_target([bs_, a_])
            y = br + GAMMA * q_
            q = self.critic([bs, ba])
            # critic_loss = tf.losses.mean_squared_error(y, q)
            critic_loss = tf.reduce_mean(tf.square(y-q))
        c_grads = tape.gradient(critic_loss, self.critic.trainable_weights)
        self.critic_opt.apply_gradients(zip(c_grads, self.critic.trainable_weights))
        self.critic_loss.append(critic_loss)

        # Actor：
        # Actor的目标就是获取最多Q值的。
        with tf.GradientTape() as tape:
            a = self.actor(bs)
            a = np.expand_dims(a,axis=1)
            q = self.critic([bs, a])
            # a_ = self.actor_target(bs_)
            # a_ = np.expand_dims(a_,axis=1) # 维度扩展
            # q_ = self.critic_target([bs_,a_])
            # a_loss = -tf.losses.mean_squared_error(q, q_)
            a_loss = -tf.reduce_mean(q)  # 注意这里用负号，是梯度上升！也就是离目标会越来越远的，就是越来越大。
        a_grads = tape.gradient(a_loss, self.critic.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grads, self.critic.trainable_weights))
        self.actor_loss.append(a_loss)

        self.ema_update()


    # 保存s，a，r，s_
    def store_transition(self, s, a, r, s_):
        """
        Store data in data buffer
        :param s: state
        :param a: act
        :param r: reward
        :param s_: next state
        :return: None
        """
        # 整理s，s_,方便直接输入网络计算
        s = s.astype(np.float32)
        s = np.expand_dims(s,axis=1)
        s_ = s_.astype(np.float32)
        s_ = np.expand_dims(s_,axis=1)
        a = np.expand_dims(a,axis=1)
        r = np.expand_dims([r],axis=1)
        r = np.expand_dims(r,axis=1)

        #把s, a, [r], s_横向堆叠
        transition = np.concatenate((s,a,r,s_),axis=2)

        #pointer是记录了曾经有多少数据进来。
        #index是记录当前最新进来的数据位置。
        #所以是一个循环，当MEMORY_CAPACITY满了以后，index就重新在最底开始了
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        #把transition，也就是s, a, [r], s_存进去。
        self.memory[index, :, :] = transition
        self.pointer += 1

    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        if not os.path.exists('model'):
            os.makedirs('model')

        tl.files.save_weights_to_hdf5('model/rms_ddpg_actor.hdf5', self.actor)
        tl.files.save_weights_to_hdf5('model/rms_ddpg_actor_target.hdf5', self.actor_target)
        tl.files.save_weights_to_hdf5('model/rms_ddpg_critic.hdf5', self.critic)
        tl.files.save_weights_to_hdf5('model/rms_ddpg_critic_target.hdf5', self.critic_target)

    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        tl.files.load_hdf5_to_weights_in_order('model/rms_ddpg_actor.hdf5', self.actor)
        tl.files.load_hdf5_to_weights_in_order('model/rms_ddpg_actor_target.hdf5', self.actor_target)
        tl.files.load_hdf5_to_weights_in_order('model/rms_ddpg_critic.hdf5', self.critic)
        tl.files.load_hdf5_to_weights_in_order('model/rms_ddpg_critic_target.hdf5', self.critic_target)

# test 
if __name__ == '__main__':
    model = RMS_DDPG(140,10)
    # s = [[0,0,1,0]]
    memory = np.zeros((100,1,(10+2*2)*10),dtype=np.float32)
    lice = np.random.choice(100,size = BATCH_SIZE) 
    bt = memory[lice,:] # (32,1,140)
    bs = bt[:, : ,:140]
    print(bs.shape)
    action = model.choose_action(bs) # (32,a_dim)
    ac = np.expand_dims(action,axis=1)
    print(ac.shape)
    h = np.concatenate((bs,ac),axis=2)
    print(h.shape)
