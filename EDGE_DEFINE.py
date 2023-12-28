import random
import math

random.seed(1037)  # 随机种子 0 1 1037
cpu_rate = 1
memory_rate = 0.01
eta = 0.8

# cpu_rate = 0.3
# memory_rate = 0.7

# 实验一
USER_NUM = 3 # 用户数目

# 实验二
MS_NUM = 4  # 微服务数目
# ms_image = [2,3,3,4] # 4
# ms_image = [2,2,3,3,3,4] # 6
# ms_image = [2,2,2,3,3,3,4,4] # 8
ms_image = [2, 2, 2, 2, 3, 3, 3, 4, 4, 4]  # 微服务镜像数目

# 实验三
NODE_NUM = 4  # 边缘节点数目


APP_CLASS = 10  # 应用种类数目
RESOURCE_NUM = 2  # 资源种类数目

class MS:
    def __init__(self,id) -> None:
        self.id = id
        self.cpu = random.uniform(1,2)
        self.memory =random.randint(10,15)
    
    def getCPU():
        pass

    def getMemory():
        pass


class EDGE_NODE:
    def __init__(self,id) -> None:
        self.id = id
        self.xloc = random.uniform(10,100)
        self.yloc = random.uniform(5,35)
        self.cpu = random.randint(15,30)
        self.memory = random.randint(200,300)

    def calCPU(self):
        pass

    def calMemory(self):
        pass

class USER:
    def __init__(self,id) -> None:
        self.id = id
        self.xloc = random.uniform(0,140)
        self.yloc = random.uniform(0,40)


    def get_request(self,num):
        user_request = []
        for _ in range(num):
            app_index = 'app'+ str(random.randint(0,APP_CLASS-1))
            APP_NAME_LIST = getAPP_Service()
            app = APP_NAME_LIST[app_index]
            user_request.append(app)
        return user_request

# define the app size of different app name
def AllappList():
    app_list = {}
    for i in range(APP_CLASS):
        name = "app" + str(i)
        if i<2:
            app_list[name] = 2
        elif i<8:
            app_list[name] = 3
        else:
            app_list[name] = 4
    return app_list

# get all ms_chain of different app
# {
#  'app0':[ms0,ms1,ms2],
#  'app1':[ms2,ms3,...],
#   ...,
# }
# type :dict
def getAPP_Service():
    appName = AllappList()
    for app in appName:
        APP = []
        size = appName[app]
        ms_list = ms_initial(ms_num=MS_NUM)
        for _ in range(size):
            ms = random.choice(ms_list)
            APP.append(ms)
            ms_list.remove(ms)
        appName[app] = APP  
    return appName

def calDis(node1,node2):
    disx = (node1.xloc - node2.xloc)**2
    disy = (node1.yloc - node2.yloc)**2
    dis = math.sqrt(disx+disy)
    return dis

# list
# [USER0,USER1,...]
def user_initial(user_num):
    user_list = []
    for i in range(user_num):
        user_list.append(USER(i))
    return user_list

# edge initial 
# list[EDGE0,EDGE1,...]
def edge_initial(node_num):
    edge_node_list = []
    for i in range(node_num):
        edge_node_list.append(EDGE_NODE(i))
    return edge_node_list

# ms initial
# list [MS0,MS1,...]
def ms_initial(ms_num):
    ms_list = []
    for i in range(ms_num):
        ms_list.append(MS(i))
    return ms_list

def get_mu():
    MU = []
    for _ in range(MS_NUM):
        MU.append(random.randint(3,5))
    return MU

def get_lamda():
    lamda = []
    for _ in range(MS_NUM):
        lamda.append(random.randint(3,5))
    return lamda

def getMSdata(ms1,ms2):
    MS_DATA = []
    for i in range(MS_NUM):
        data = []
        for j in range(MS_NUM):
            if i==j:
                data.append(0)
            else:
                data.append(random.uniform(0,3))
        MS_DATA.append(data)
    return MS_DATA[ms1][ms2]

def getBandwidth(node1,node2):
    EDGE_BANDWIDTH = []
    for i in range(NODE_NUM):
        band = []
        for j in range(NODE_NUM):
            if i == j:
                band.append(0)
            else:
                band.append(random.randint(1,5))
        EDGE_BANDWIDTH.append(band)
    return EDGE_BANDWIDTH[node1][node2]

def getUserEdgeBandwidth(user,edge):
    USER_EDGE_BAND = []
    for i in range(USER_NUM):
        band = []
        for j in range(NODE_NUM):
            band.append(random.randint(2,10))
        USER_EDGE_BAND.append(band)
    return USER_EDGE_BAND[user][edge]

# get request of user
# {
#   user0:[app0,app1,...],
#   user1:[app4,app6,...],
#   ...,
# }
# type: dict
def get_user_request(user_num):
    user = user_initial(user_num)
    user_list = {}
    for item in user:
        user_list[item] = item.get_request(random.randint(1,3))
    return user_list


def requests_num():
    usr_req = get_user_request(USER_NUM)
    req_num = 0
    for item in usr_req:
        req_num += len(usr_req[item])
    return req_num