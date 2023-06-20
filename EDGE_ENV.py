from EDGE_DEFINE import *
import numpy as np
import math

v = 3*10**8
ms_image = [2,2,2,2,3,3,3,4,4,4]
# 初始化环境状态shape(ms_num+2*resources,node_num)
def initial_state():
    deploy_state = np.zeros(shape=(MS_NUM,NODE_NUM)) # 初始化部署状态，全0
    deploy_state = np.reshape(deploy_state,(1,MS_NUM*NODE_NUM))
    cpu_state = np.zeros(shape=(2,NODE_NUM)) # cpu使用和剩余
    memory_state = np.zeros(shape=(2,NODE_NUM)) # mem使用和剩余
    for i in range(NODE_NUM):
        cpu_state[1][i] = EDGE_NODE(i).cpu # 初始化剩余cpu资源
        memory_state[1][i] = EDGE_NODE(i).memory # 初始化剩余memory资源
    cpu_state = np.reshape(cpu_state,(1,2*NODE_NUM))
    memory_state = np.reshape(memory_state,(1,2*NODE_NUM))
    resource = np.append(cpu_state,memory_state) # 拼接到一起
    state = np.append(deploy_state,resource)
    state = np.reshape(state,(1,(MS_NUM+2*RESOURCE_NUM)*NODE_NUM))
    
    return state

# 更新状态空间
def update_state(state,action,ms_idx):
    state_new = np.reshape(state,(MS_NUM+2*RESOURCE_NUM,NODE_NUM))
    action = np.reshape(action,(1,NODE_NUM))
    act_idx = np.argmax(action)
    # print("choose node",act_idx)
    state_new[ms_idx][act_idx] += 1
    state_new[MS_NUM][act_idx] += MS(ms_idx).cpu
    state_new[MS_NUM+1][act_idx] -= MS(ms_idx).cpu
    state_new[MS_NUM+2][act_idx] += MS(ms_idx).memory
    state_new[MS_NUM+3][act_idx] -= MS(ms_idx).memory
    state_new = np.reshape(state_new,(1,(MS_NUM+2*RESOURCE_NUM)*NODE_NUM))

    return state_new , act_idx

def assign_pods(state):
    state = np.reshape(state,(MS_NUM,NODE_NUM))
    idx = np.argmax(state,axis=1)
    nv = np.zeros(shape=(MS_NUM,NODE_NUM))
    for ms_id,node_id in enumerate(idx):
        mu = get_mu()[ms_id][node_id]
        n_pods = math.ceil(LAMBDA / mu)
        nv[ms_id][node_id] = n_pods
    return nv

# 拉伸部署镜像数
def update_assign(state,nv):
    max_delay = 100000
    state = np.reshape(state,(MS_NUM,NODE_NUM))
    ms_list = ms_initial(MS_NUM)
    edge_list = edge_initial(NODE_NUM)
    nv_new = nv
    for node in range(NODE_NUM):
        cpu = edge_list[node].cpu
        memory = edge_list[node].memory
        for idx,ms in enumerate(nv[:,node]):
            if ms != 0:
                cpu -= ms_list[idx].cpu * ms
                memory -= ms_list[idx].memory * ms
            else:
                continue
        while cpu > 0 and memory > 0:
            if np.argmax(state[:,node])==0:
                break
            theta = []
            for idx,ms in enumerate(state[:,node]):
                if ms == 1:
                    nv[idx][node] += 1
                    theta.append(cal_access_delay(state,nv))
                    cpu -= ms_list[idx].cpu
                    memory -= ms_list[idx].memory
                else:
                    theta.append(max_delay)
            min_idx = np.argmin(np.array(theta))
            nv_new[min_idx][node] += 1
        # print("边缘节点%2d剩余cpu资源为%2d,剩余memory资源%2d,无法继续分配"%(node,cpu,memory))
    return nv_new



#   [EDGE0,EDGE1,...]
edge_node = edge_initial(NODE_NUM) 
# {
#   User0:[[MS0,MS1],[MS1,MS3]...]
# }
user_request = get_user_request(USER_NUM)

def jiechen(n):
    k =1
    if n == 0:
        return 1
    else:
        for i in range(1,n+1):
            k *= i
        return k

def cal_exetime(ms_id,image_num):
    LAMBDA = get_lamda()[ms_id]
    mu = get_mu()[ms_id]
    exe = 1 / mu
    ro = LAMBDA / (image_num * mu)
    if(ro==1):
        print("设置异常")
    p0 = 0
    for i in range(image_num):
        p0 += jiechen(i)*(LAMBDA / mu)**i
    p0 += 1 / ( (1 / (jiechen(image_num)*(1-ro))) * (LAMBDA / mu)**image_num )
    lq = ( (image_num*ro)**image_num / (jiechen(image_num) * (1-ro)**2) ) * ro * p0
    wq = lq / LAMBDA
    t_exe = exe + wq
    return t_exe

# 寻找所有路由
def choose_route(deploy_tate,ms_chain):
    path = [] # 存放当前请求链所有可路由的情况
    all_path = []
    ans = []
    for ms in ms_chain:
        ms_id = ms.id
        node = deploy_tate[ms_id]
        s = set() # 存放部署了ms的节点
        for node_idx in range(len(node)):
            if(node[node_idx]>0):
                s.add(node_idx)
        ans.append(s)
    # 回溯算法寻找所有路径
    dfs(path,ans,all_path)
    return all_path

def dfs(path,ans,res):
    if not ans:
        res.append(list(path))
        return
    
    for item in ans[0]:
        path.append(item)
        dfs(path,ans[1:],res)
        path.pop()

def cal_access_delay(deploy_state):
    access_delay = 0
    for user in user_request:
        # User object
        req = user_request[user] # [[a1][a2]]
        app_time = 0
        for app in req:
            route = choose_route(deploy_state,app)
            # MS object
            edge_idx_list = []
            ms_id_list = []
            for ms in app:
                ms_idx = ms.id
                iamge_num = ms_image[ms_idx]
                ms_id_list.append(ms_idx)
                node = deploy_state[ms_idx]
                for edge_idx in range(len(node)):
                    if node[edge_idx]>0:
                        edge_idx_list.append(edge_idx)
                # per ms exe time
                exe_time = cal_exetime(ms_idx,iamge_num)
                app_time += exe_time
                # t_up & t_down & tp
                t_up = 0
                t_p = 0
                for edge_node_idx in edge_idx_list:
                    user_edge_b = getUserEdgeBandwidth(user.id,edge_node_idx)
                    t_up += (ms.memory / user_edge_b) * 2
                    edge = edge_node[edge_node_idx]
                    distance = calDis(user,edge)
                    t_p += distance / v
                t_up_avg = t_up / len(edge_idx_list)
                t_p_avg = t_p / len(edge_idx_list)
                app_time += t_up_avg
                app_time += t_p_avg
            # route delay
            all_route = choose_route(deploy_state,app)
            comtime = 0
            for route in all_route:
                start ,end= 0,1
                while end<len(route):
                    bandwidth = getBandwidth(route[start],route[end])
                    ms_data = getMSdata(ms_id_list[start],ms_id_list[end])
                    if bandwidth != 0:
                        comtime += ms_data / bandwidth
                    start += 1
                    end += 1
            comtime_avg = comtime / len(all_route)
            app_time += comtime_avg
        access_delay += app_time
    return access_delay


if __name__ == "__main__":
    # state = initial_state()
    # state = np.reshape(state,(2+2+MS_NUM,NODE_NUM))
    # nv = assign_pods(state)
    # nv_new = update_assign(state,nv)
    li = [[1,2],[2,3],[4,5,6]]
    res = []
    path = []
    dfs(path,li,res)
    print(res)

