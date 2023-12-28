import EDGE_DEFINE
from EDGE_DEFINE import *
import Test
import numpy as np
import copy


class GMDA:
    def __init__(self, initial_ms_list, initial_state, initial_node_list) -> None:
        """
        FFD类
        :param initial_ms_list: 初始微服务列表
        :param initial_state: 初始部署状态
        :param initial_node_list: 初始服务器节点列表
        :return
        """
        # 值拷贝一份参数
        self.initial_ms_list = copy.deepcopy(initial_ms_list)
        self.initial_state = copy.deepcopy(initial_state)
        self.initial_node_list = copy.deepcopy(initial_node_list)
        self.ms_image = copy.deepcopy(EDGE_DEFINE.ms_image)

        # 拆分限度（和论文意义不一样，这里的指最多安排几个实例，W<=服务实例数）
        # benkz TODO 2023/12/13:设置多少合适？
        self.W = 1

    def update_deploy_state(self, state, ms, deploy_num, ms_list, free_list, activate_list, deploy_node, deploy_all):
        """
        更新部署状态、节点资源、N_m
        :param state: 旧状态
        :param ms: 部署的服务种类
        :param deploy_num: 部署的服务数目
        :param ms_list: 已排序待部署的微服务列表
        :param free_list: 完全闲置的服务器列表
        :param activate_list: 已部署还有剩余资源的服务器
        :param deploy_node: 部署的服务器节点
        :param deploy_all: 是否一次性部署所有实例
        :return:
        """
        # place deploy_num microservice m instances at the node
        deploy_node.cpu -= ms.cpu * deploy_num
        deploy_node.memory -= ms.memory * deploy_num

        # update the N_m
        self.ms_image[ms.id] -= deploy_num

        # update the number of instance num of the state
        state[ms.id][deploy_node.id] += deploy_num

        # update the cpu resource of the state
        state[MS_NUM][deploy_node.id] += ms.cpu * deploy_num
        state[MS_NUM + 1][deploy_node.id] -= ms.cpu * deploy_num

        # update the memory resource of the state
        state[MS_NUM + 2][deploy_node.id] += ms.memory * deploy_num
        state[MS_NUM + 3][deploy_node.id] -= ms.memory * deploy_num

        # remove the microservice m from the ms_list
        if deploy_all:
            for tuple in ms_list:
                if tuple[0] == ms:
                    ms_list.remove(tuple)
                    break

        # move the deploy_node to the activate_list if it is selected from the free_list
        if deploy_node in free_list:
            free_list.remove(deploy_node)
            activate_list.append(deploy_node)

    def GMDA_deploy(self):
        """
        GMDA部署算法
        :return: 部署后的状态
        """
        # 对元组列表按值从大到小排序
        ms_list = sorted(self.initial_ms_list, key=lambda x: x[1], reverse=True)

        # 改变initial_state的数据结构
        state = np.reshape(self.initial_state, (MS_NUM + 2 * RESOURCE_NUM, NODE_NUM))

        # 初始化算法list
        # free_list(完全闲置的服务器)
        free_list = self.initial_node_list
        # activate_list(已部署还有剩余资源的服务器)
        activate_list = []
        # available_list(可供部署的服务器)
        # available_list = []

        # 遍历整个元组列表将所有微服务进行部署
        while len(ms_list) != 0:
            # 取出需求资源最多的微服务
            # benkz TODO 2023/12/13:pop
            ms_tuple = ms_list[0]
            ms = ms_tuple[0]
            # resource = ms_tuple[1]

            available_list = []
            N_m = self.ms_image[ms.id]

            # 在activate_list中查找可否拆分部署
            for activate_node in activate_list:
                rest_deploy_num = min(activate_node.cpu // ms.cpu, activate_node.memory // ms.memory)
                # benkz TODO 2023/12/13:
                if rest_deploy_num >= self.W:
                    available_list.append(activate_node)

            # 在available_list中查找可否一次性部署
            if len(available_list) == 0:
                for free_node in free_list:
                    rest_deploy_num = min(free_node.cpu // ms.cpu, free_node.memory // ms.memory)
                    if rest_deploy_num >= N_m:
                        available_list.append(free_node)

            # 在free_list中查找可否拆分部署
            if len(available_list) == 0:
                for free_node in free_list:
                    rest_deploy_num = min(free_node.cpu // ms.cpu, free_node.memory // ms.memory)
                    # benkz TODO 2023/12/13:
                    if rest_deploy_num >= self.W:
                        available_list.append(free_node)

            # 部署失败的情况
            # benkz TODO 2023/12/13:还剩余{X}个实例等待部署怎么计算？
            if len(available_list) == 0:
                print(f"服务{ms.id}部署失败，还剩余XXXXX个实例等待部署！")
                for tuple in ms_list:
                    if tuple[0] == ms:
                        ms_list.remove(tuple)
                        break
                continue

            # 根据available_list中服务器可用资源降序排序
            # benkz TODO 2023/12/13:
            available_list = sorted(available_list, key=lambda x: (cpu_rate*x.cpu + memory_rate*x.memory), reverse=True)

            # 部署微服务、更新资源、状态、计数器list
            for node in available_list:
                # 该节点能剩余部署的微服务实例数
                rest_deploy_num = min(node.cpu // ms.cpu, node.memory // ms.memory)
                # 该微服务剩余需部署的实例数
                N_m = self.ms_image[ms.id]

                # 一次性部署情况
                if rest_deploy_num >= N_m:
                    self.update_deploy_state(state, ms, N_m, ms_list, free_list, activate_list, node, True)
                    break
                # 拆分部署情况
                else:
                    self.update_deploy_state(state, ms, rest_deploy_num, ms_list, free_list, activate_list, node, False)

        # 转化数据结构
        # state = np.reshape(state, (1, (MS_NUM + 2 * RESOURCE_NUM) * NODE_NUM))
        return state
