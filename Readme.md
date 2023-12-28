## 环境
tensorflow >=2.0.0
tensorlayer >=2.0.0
numpy == 1.24.2

## 模型 reward shaping
step1:根据每种微服务设定的不同的到达率lambd和服务器处理能力mu，初始化得到镜像数目
step2:每一步部署一个镜像，部署成功获得正向奖励reward,部署失败获得惩罚
step3:所有镜像部署完毕，计算时延，计算累加reward赋予最后一步

## 量化实验
### 模型稳健性 2 table + 2 fig
调整随机seed = 0,1,1037（lr = 0.005）
1.改变边缘节点数 nodes = 4,6,8,10 
2.微服务种类数 num = 4,6,8,10
### 时延性能 3 fig
1.请求数
2.边缘节点
3.微服务种类数

### 资源利用率（load balancing）3 fig
1.cpu
2.mem
3.load
调整权重值作量化实验

## 消融实验 3 * 2 fig
1.lr 
2.soft update factor mu
3.exploration factor var *
4.lstm *
5.hidden layers *

## 对比算法
mb_ddpg rsdql ffd GMDA

## 备注
mb_ddpg.py and rsdql.py 均是根据论文核心思想复现运用在本模型上，为了适配本文建立的模型，有些许不同
为了验证有效性，保持超参数基本一致

