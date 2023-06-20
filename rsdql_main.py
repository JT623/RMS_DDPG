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

class DQN(object):
    def __init__(self,s_dim,a_dim) :
        # memory用于储存数据
        # 保存个数MEMORY_CAPACITY，s_dim * 2 + a_dim + 1：分别是两个state，一个action，和一个reward
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.s_dim = s_dim
        self.a_dim = a_dim
        W_init = tf.random_normal_initializer(mean=0, stddev=0.1)
        b_init = tf.constant_initializer(0.1)
        self.qloss = []

        def qnet(in_shape):
            inputs = tl.layers.Input(in_shape,name="q_input")
            fc1 = tl.layers.Dense(n_units=128,act=tf.nn.relu,W_init=W_init,b_init=b_init,name="fc1")(inputs)
            fc2 = tl.layers.Dense(n_units=128,act=tf.nn.relu,W_init=W_init,b_init=b_init,name="fc2")(fc1)
            out = tl.layers.Dense(n_units=a_dim,act=tf.tanh,name="out")(fc2)
            return tl.models.Model(inputs=inputs,outputs=out)
        
        self.qnet = qnet([None,s_dim])
        self.qnet.train()

        def copy_para(from_model, to_model):
            for i, j in zip(from_model.trainable_weights, to_model.trainable_weights):
                j.assign(i)

        self.q_target = qnet([None,s_dim])
        copy_para(self.qnet,self.q_target)
        self.q_target.eval()

        self.qnet_opt = tf.optimizers.Adam(LR_A)
        self.ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)  # soft replacement

    def ema_update(self):
        # 其实和之前的硬更新类似，不过在更新赋值之前，用一个ema.average。
        paras = self.qnet.trainable_weights
        self.ema.apply(paras)                                                  
        for i, j in zip(self.q_target.trainable_weights, paras):
            i.assign(self.ema.average(j)) 

    def choose_action(self,s):
        return self.qnet(np.array(s, dtype=np.float32))
    
    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)    #随机BATCH_SIZE个随机数
        bt = self.memory[indices, :]                    #根据indices，选取数据bt，相当于随机
        bs = bt[:, :self.s_dim]                         #从bt获得数据s
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]  #从bt获得数据a
        br = bt[:, -self.s_dim - 1:-self.s_dim]         #从bt获得数据r
        bs_ = bt[:, -self.s_dim:]                       #从bt获得数据s'

        with tf.GradientTape() as tape:
            q = self.qnet(bs)
            # q = np.argmax(q,axis=1)
            # q = np.reshape(q,(-1,1))
            q_v = self.q_target(bs_)
            # maxq = np.argmax(q_v,axis=1)
            y = -br + GAMMA * q_v
            # q_loss = tf.losses.mean_squared_error(y, q)
            q_loss = tf.reduce_mean(tf.square(q-y))
        c_grads = tape.gradient(q_loss, self.qnet.trainable_weights)
        self.qnet_opt.apply_gradients(zip(c_grads, self.qnet.trainable_weights))
        self.ema_update()
        self.qloss.append(q_loss)

    def store_transition(self, s, a, r, s_):
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

        tl.files.save_weights_to_hdf5('model/dql.hdf5', self.qnet)
        tl.files.save_weights_to_hdf5('model/dql_target.hdf5', self.q_target)

    def load_ckpt(self):
        """
        load trained weights
        :return: None
        """
        tl.files.load_hdf5_to_weights_in_order('model/dql.hdf5', self.qnet)
        tl.files.load_hdf5_to_weights_in_order('model/dql_target.hdf5', self.q_target)

if __name__ == "__main__":
    sys.stdout = Logger(sys.stdout)  # record log
    sys.stderr = Logger(sys.stderr)  # record error 
    tf.random.set_seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)
    s_dim = (MS_NUM+2*RESOURCE_NUM)*NODE_NUM
    a_dim = NODE_NUM

    dql = DQN(s_dim,a_dim)
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
                a = dql.choose_action(s)
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
                dql.store_transition(s,a,reward,snew)
                if dql.pointer >  MEMORY_CAPACITY:
                    # VAR *= 0.995
                    dql.learn()
                s = np.reshape(s_new,(1,s_dim))

        if flag == False:
            dql.store_transition(s,a,step_rew/10,snew)

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
            dql.store_transition(s,a,step_rew/10,snew)
            t = t_new

        if episode == 0:
            reward_buffer.append(step_rew*0.05)
        else:
            reward_buffer.append(reward_buffer[-1] * 0.95 + step_rew * 0.05 )
            
        print('episode: {} || reward: {:.4f} || t_min: {:.4f} || reward_s: {:.4f}'.format(episode,step_rew,t_min,reward_buffer[-1]))
        # reward_buffer.append(step_sum)
    # print("best deployment:", deploy_best)
    # print("best delay:", t_min)

    dql.save_ckpt()
    q_loss = dql.qloss
    
    # plt
    plt.figure(1)
    plt.plot(np.arange(len(reward_buffer)),reward_buffer, linewidth = 2.5)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.savefig("./figure/Reward_dql.png")
    # plt.show()
    # plt.figure(2)
    # plt.plot(np.arange(len(q_loss)),q_loss, linewidth = 1.5)
    # plt.xlabel("Episodes")
    # plt.ylabel("Q loss")
    # plt.savefig("./figure/q_loss.png")