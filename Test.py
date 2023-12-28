import EDGE_ENV
from FFD import *
from GMDA import *


def create_state():
    """
    初始化环境状态
    :return: shape(ms_num+2*resources,node_num)
    """
    deploy_state = np.zeros(shape=(MS_NUM, NODE_NUM))  # 初始化部署状态，全0
    deploy_state = np.reshape(deploy_state, (1, MS_NUM * NODE_NUM))
    cpu_state = np.zeros(shape=(2, NODE_NUM))  # cpu使用和剩余
    memory_state = np.zeros(shape=(2, NODE_NUM))  # mem使用和剩余
    node_list = []
    for i in range(NODE_NUM):
        edge_node = EDGE_NODE(i)
        node_list.append(edge_node)
        cpu_state[1][i] = edge_node.cpu  # 初始化剩余cpu资源
        memory_state[1][i] = edge_node.memory  # 初始化剩余memory资源
    cpu_state = np.reshape(cpu_state, (1, 2 * NODE_NUM))
    memory_state = np.reshape(memory_state, (1, 2 * NODE_NUM))
    resource = np.append(cpu_state, memory_state)  # 拼接到一起
    state = np.append(deploy_state, resource)
    # benkz TODO 2023/12/12:这一句没什么意义吧
    # state = np.reshape(state, (1, (MS_NUM + 2 * RESOURCE_NUM) * NODE_NUM))
    return state, node_list


if __name__ == "__main__":
    initial_ms_list = []
    # 创建每种微服务并计算其需求资源
    for i in range(MS_NUM):
        ms = MS(i)
        instance_num = ms_image[i]
        resource = cpu_rate*instance_num * ms.cpu + memory_rate*instance_num * ms.memory
        # 创建包含(微服务,需求资源)元组加入列表
        ms_tuple = (ms, resource)
        initial_ms_list.append(ms_tuple)
        # print(f"ms{i} cpu resource: {ms.cpu} and memory resource: {ms.memory}")
        # print(f"ms{i} eta resource: {resource}")

    # 初始化边缘服务器环境
    initial_state, node_list = create_state()

    # FFD部署算法
    FFD_alg = FFD(initial_ms_list, initial_state, node_list)
    # FFD部署后state
    FFD_deploy_state = FFD_alg.FFD_deploy()
    # FFD请求路由时延
    FFD_delay = EDGE_ENV.cal_access_delay(FFD_deploy_state)
    FFD_load = EDGE_ENV.cal_load(FFD_deploy_state,eta)
    # FFD_average_delay= FFD_delay/req_num
    print(f"FFD部署算法请求路由时延: {FFD_delay}")
    print(f"FFD资源利用率方差为: {FFD_load}")
    # GMDA部署算法
    GMDA_alg = GMDA(initial_ms_list, initial_state, node_list)
    # GMDA部署后state
    GMDA_deploy_state = GMDA_alg.GMDA_deploy()
    # GMDA请求路由时延
    GMDA_delay = EDGE_ENV.cal_access_delay(GMDA_deploy_state)
    GMDA_load = EDGE_ENV.cal_load(GMDA_deploy_state,eta)
    # GMDA_average_delay = GMDA_delay/req_num
    print(f"GMDA部署算法请求路由时延: {GMDA_delay}")
    print(f"GMDA资源利用率方差为: {GMDA_load}")
    # RMS_DDPG部署算法
    # RMS_DDPG_main.main(req_num)

