import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from RMS_DDPG import *
import EDGE_ENV
from EDGE_DEFINE import *
import time,sys,os
import matplotlib.pyplot as plt
from RMS_DDPG_main import Logger

class DDPG(object):
    """
    DDPG class
    """
    def __init__(self, s_dim, a_dim):
        # memory用于储存跑的数据的数组：
        # 保存个数MEMORY_CAPACITY，s_dim * 2 + a_dim + 1：分别是两个state，一个action，和一个reward
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.a_dim, self.s_dim = a_dim, s_dim

        W_init = tf.random_normal_initializer(mean=0, stddev=0.5)
        b_init = tf.constant_initializer(0.1)

        self.actor_loss = []
        self.critic_loss = []

        # 建立actor网络，输入s，输出a
        def get_actor(input_state_shape, name=''):
            """
            Build actor network
            :param input_state_shape: state
            :param name: name
            :return: action
            """
            inputs = tl.layers.Input(input_state_shape, name='A_input')
            x = tl.layers.Dense(n_units=128, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='A1')(inputs)
            x = tl.layers.Dense(n_units=128,act=tf.nn.relu,W_init=W_init,b_init=b_init,name='A2')(x)
            x = tl.layers.Dense(n_units=a_dim, act=tf.tanh, W_init=W_init, b_init=b_init, name='A_out')(x)
            # x = tl.layers.Lambda(lambda x: np.array(a_bound) * x)(x) # [0,a_bound]    
            return tl.models.Model(inputs=inputs, outputs=x, name='Actor' + name)

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
            x = tl.layers.Concat(1)([s,a])
            x = tl.layers.Dense(n_units=128, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l1')(x)
            x = tl.layers.Dense(n_units=128, act=tf.nn.relu, W_init=W_init, b_init=b_init, name='C_l2')(x)
            x = tl.layers.Dense(n_units=1, W_init=W_init, b_init=b_init, name='C_out')(x)
            return tl.models.Model(inputs=[s, a], outputs=x, name='Critic' + name)

        self.actor = get_actor([None, s_dim])
        self.critic = get_critic([None, s_dim], [None, a_dim])
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
        self.actor_target = get_actor([None, s_dim], name='_target')
        copy_para(self.actor, self.actor_target)
        self.actor_target.eval()

        #建立critic_target网络，并和actor参数一致，不能训练
        self.critic_target = get_critic([None, s_dim], [None, a_dim], name='_target')
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
        :return: act
        """
        return self.actor(np.array([s], dtype=np.float32))[0]

    def learn(self):
        """
        Update parameters
        :return: None
        """
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)    #随机BATCH_SIZE个随机数
        bt = self.memory[indices, :]                    #根据indices，选取数据bt，相当于随机
        bs = bt[:, :self.s_dim]                         #从bt获得数据s
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]  #从bt获得数据a
        br = bt[:, -self.s_dim - 1:-self.s_dim]         #从bt获得数据r
        bs_ = bt[:, -self.s_dim:]                       #从bt获得数据s'

        # Critic：
        # Critic更新和DQN很像，不过target不是argmax了，是用critic_target计算出来的。
        # br + GAMMA * q_
        with tf.GradientTape() as tape:
            a_ = self.actor_target(bs_)
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
            q = self.critic([bs, a])
            a_loss = -tf.reduce_mean(q)  # 注意这里用负号，是梯度上升！也就是离目标会越来越远的，就是越来越大。
        a_grads = tape.gradient(a_loss, self.actor.trainable_weights)
        self.actor_opt.apply_gradients(zip(a_grads, self.actor.trainable_weights))
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
        s_ = s_.astype(np.float32)

        #把s, a, [r], s_横向堆叠
        transition = np.hstack((s, a, [[r]], s_))

        #pointer是记录了曾经有多少数据进来。
        #index是记录当前最新进来的数据位置。
        #所以是一个循环，当MEMORY_CAPACITY满了以后，index就重新在最底开始了
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        #把transition，也就是s, a, [r], s_存进去。
        self.memory[index, :] = transition
        self.pointer += 1

    def save_ckpt(self):
        """
        save trained weights
        :return: None
        """
        if not os.path.exists('model'):
            os.makedirs('model')

        tl.files.save_weights_to_hdf5('model/mb_ddpg_actor.hdf5', self.actor)
        tl.files.save_weights_to_hdf5('model/mb_ddpg_actor_target.hdf5', self.actor_target)
        tl.files.save_weights_to_hdf5('model/mb_ddpg_critic.hdf5', self.critic)
        tl.files.save_weights_to_hdf5('model/mb_ddpg_critic_target.hdf5', self.critic_target)

    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        tl.files.load_hdf5_to_weights_in_order('model/mb_ddpg_actor.hdf5', self.actor)
        tl.files.load_hdf5_to_weights_in_order('model/mb_ddpg_actor_target.hdf5', self.actor_target)
        tl.files.load_hdf5_to_weights_in_order('model/mb_ddpg_critic.hdf5', self.critic)
        tl.files.load_hdf5_to_weights_in_order('model/mb_ddpg_critic_target.hdf5', self.critic_target)

if __name__ == "__main__":
    sys.stdout = Logger(sys.stdout)  # record log
    sys.stderr = Logger(sys.stderr)  # record error 
    tf.random.set_seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)
    s_dim = (MS_NUM+2*RESOURCE_NUM)*NODE_NUM
    a_dim = NODE_NUM

    ddpg = DDPG(s_dim,a_dim)
    t = float('inf')
    t_min = float('inf')
    reward_buffer = []
    # step_sum = 0
    # ms_image = [2,3,3,4] # 4
    # ms_image = [2,2,3,3,3,4] # 6
    # ms_image = [2,2,2,3,3,3,4,4] # 8
    ms_image = [2,2,2,2,3,3,3,4,4,4] # 10
    num = sum(ms_image)
    deploy_best = []
    t_best = 0
    Depflag = 0

    for episode in range(MAX_EPISODES):
        step_rew = 0
        s = EDGE_ENV.initial_state()
        ms_list = [i for i in range(MS_NUM)]
        ms_init_image = ms_image
        flag = True # 部署成功
        count = 0
        while len(ms_list)!=0:
            ms_idx = random.choice(ms_list)
            ms_list.remove(ms_idx)
            image_num = ms_init_image[ms_idx]
            for _ in range(image_num):
                count += 1
                a = ddpg.choose_action(s)
                a = np.clip(np.random.normal(a,VAR),-1,1)
                snew, act_idx = EDGE_ENV.update_state(s,a,ms_idx)
                s_new = np.reshape(snew,(MS_NUM+2*RESOURCE_NUM,NODE_NUM))
                if flag == False:
                    break
                # 如果可用资源足够，可以布署，获得累加奖励, 若无法部署，终止
                elif s_new[MS_NUM+1][act_idx]>0 and s_new[MS_NUM+3][act_idx]>0 and count<=num-1:
                    reward = count / num
                    step_rew += reward

                elif (s_new[MS_NUM+1][act_idx]<=0 or s_new[MS_NUM+3][act_idx]<=0) and count<=num-1:
                    reward = 0
                    step_rew += reward
                    flag = False
                    # break
                ddpg.store_transition(s,a,reward,snew)
                if ddpg.pointer >  MEMORY_CAPACITY:
                    # VAR *= 0.995
                    ddpg.learn()
                s = np.reshape(s_new,(1,s_dim))

        if flag == False:
            ddpg.store_transition(s,a,step_rew/10,snew)

        elif flag == True:
            Depflag += 1 # 用于记录第一次部署成功计算时延,后续比较
            if Depflag == 1:
                t_new = EDGE_ENV.cal_access_delay(s_new)
                t = t_new
                t_min = t_new
                # continue
            # 计算微服务访问时延
            t_new = EDGE_ENV.cal_access_delay(s_new)
            if episode >0 and t_new <= t_min:
                reward = (t_min - t_new) * 0.5 +1
                t_min = t_new
                deploy_best = s_new
            elif episode >0 and t_new < t:
                reward = (t - t_new) * 0.1 + 1
            # elif t_new > t_min:
            #     reward = -1
            else:
                reward = 0
            step_rew += reward
            # step_sum += step_rew
            ddpg.store_transition(s,a,step_rew/10,snew)
            t = t_new

        if episode == 0:
            reward_buffer.append(step_rew * 0.05)
        else:
            reward_buffer.append(reward_buffer[-1] * 0.95 + step_rew * 0.05 )
            
        print('episode: {} || reward: {:.4f} || t_min: {:.4f} || reward_s: {:.4f}'.format(episode,step_rew,t_min,reward_buffer[-1]))
        # reward_buffer.append(step_sum)
    # print("best deployment:", deploy_best)
    # print("best delay:", t_min)

    ddpg.save_ckpt()

    actor_loss_list = ddpg.actor_loss

    critic_loss_list = ddpg.critic_loss
    
    # plt
    plt.figure(1)
    plt.plot(np.arange(len(reward_buffer)),reward_buffer, linewidth = 2.5)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.savefig("./figure/Reward_mbddpg.png")
    # plt.show()
    # plt.figure(2)
    # plt.plot(np.arange(len(actor_loss_list)),actor_loss_list, linewidth = 1.5)
    # plt.xlabel("Episodes")
    # plt.ylabel("Actor")
    # plt.savefig("./figure/actor.png")
    # plt.figure(3)
    # plt.plot(np.arange(len(critic_loss_list)),critic_loss_list, linewidth = 1.5)
    # plt.xlabel("Episodes")
    # plt.ylabel("Critic")
    # plt.savefig("./figure/critic.png")