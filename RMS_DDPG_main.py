from RMS_DDPG import *
import EDGE_ENV
from EDGE_DEFINE import *
import time,sys,os
import matplotlib.pyplot as plt
# log recorder
class Logger(object):

    def __init__(self, stream=sys.stdout):
        output_dir = "./logfile/"  # folder 
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        log_name_time = time.strftime('%Y-%m-%d %H-%M-%S',time.localtime(time.time()))
        log_name = log_name_time + ".txt"
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

if __name__ == "__main__":
    sys.stdout = Logger(sys.stdout)  # record log
    sys.stderr = Logger(sys.stderr)  # record error 
    tf.random.set_seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)
    s_dim = (MS_NUM+2*RESOURCE_NUM)*NODE_NUM
    a_dim = NODE_NUM

    ddpg = RMS_DDPG(s_dim,a_dim)
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
                s_ = np.expand_dims(s,axis = 1) # 维度扩展
                a = ddpg.choose_action(s_)
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
            reward_buffer.append(step_rew*0.05)
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
    plt.savefig("./figure/Reward_rmsddpg.png")
    # # plt.show()
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