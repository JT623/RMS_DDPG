import EDGE_DEFINE
from EDGE_DEFINE import *
import numpy as np
import copy


class FFD:
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

    def update_deploy_state(self, state, ms, deploy_num):
        """
        更新state和node
        :param state: 状态
        :param ms: 部署的微服务
        :param deploy_num: 部署的实例数
        :return:
        """
        for node in self.initial_node_list:
            # 当前节点能部署该微服务的最大实例数
            max_deploy_num = min(min(node.cpu // ms.cpu, node.memory // ms.memory), deploy_num)
            # 该微服务剩余部署实例数
            rest_num = deploy_num - max_deploy_num

            if max_deploy_num == 0:
                # 遍历到最后一个无法部署，则实例部署失败
                if self.initial_node_list.index(node) == len(self.initial_node_list) - 1:
                    print(f"服务{ms.id}部署失败，还剩余{rest_num}个实例等待部署！")
                    return
                else:
                    # 遍历下一个节点
                    continue

            # 更新state的实例镜像数
            state[ms.id][node.id] += max_deploy_num

            # 更新state的cup资源
            state[MS_NUM][node.id] += ms.cpu * max_deploy_num
            state[MS_NUM + 1][node.id] -= ms.cpu * max_deploy_num

            # 更新state的memory资源
            state[MS_NUM + 2][node.id] += ms.memory * max_deploy_num
            state[MS_NUM + 3][node.id] -= ms.memory * max_deploy_num

            # 更新部署微服务ms的node的资源
            node.cpu = node.cpu - ms.cpu * max_deploy_num
            node.memory = node.memory - ms.memory * max_deploy_num

            # 递归拆分部署
            if rest_num > 0:
                self.update_deploy_state(state, ms, rest_num)

            # 部署成功
            break

    def FFD_deploy(self):
        """
        FFD部署算法
        :return: 部署后的状态
        """
        # 对元组列表按值从大到小排序
        ms_list = sorted(self.initial_ms_list, key=lambda x: x[1], reverse=True)

        # 改变initial_state的数据结构
        state = np.reshape(self.initial_state, (MS_NUM + 2 * RESOURCE_NUM, NODE_NUM))

        # 遍历整个元组列表将所有微服务进行部署
        for (ms, resource) in ms_list:
            image_num = self.ms_image[ms.id]
            self.update_deploy_state(state, ms, image_num)

        # 转化数据结构
        # state = np.reshape(state, (1, (MS_NUM + 2 * RESOURCE_NUM) * NODE_NUM))
        return state