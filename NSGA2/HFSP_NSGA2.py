# coding:utf-8
import math
import random
import os
from Initial import *
import numpy as np
from CalFitness import FitDHHFSP,EnergySave_DHHFSP
from GA import NSGA2POX,NSGA2POXES2
import  copy
from Tool import *
from LocalSearch import *
#from DDQN_model import DoubleDQN
from HFSP_Instance import J_num,State,M_j,PT,JNN, D_pq, v, V_r
from HFSP_env import Scheduling as Sch
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
import torch
import pickle
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Linear, ReLU, Sequential
import parl
from parl.utils import logger, summary
from parl.utils.utils import check_model_method
from parl.utils.scheduler import LinearDecayScheduler
from parl.algorithms.paddle import DQN, DDQN
#parameter
ps=200;Pc=1.0;Pm=0.2
lr=0.005;batch_size=32
EPSILON = 0.9               # greedy policy
GAMMA = 0.95                 # reward discount
TARGET_REPLACE_ITER = 7   # target update frequency
MEMORY_CAPACITY = 512
UPDATE_TARGET_STEP = 200
N_ACTIONS = 4  # 6种候选的算子
EPOCH=1
MaxNFEs = 20000 if J_num < 40 else 500 * J_num
print(torch.cuda.is_available())

class QNetwork(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(QNetwork, self).__init__()
        hid1_size = 128
        hid2_size = 128
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, act_dim)
    def forward(self, obs):
        h1 = F.relu(self.fc1(obs))
        h2 = F.relu(self.fc2(h1))
        Q = self.fc3(h2)
        return Q
class Agent(parl.Agent):
    def __init__(self, algorithm, update_target_step, act_dim, e_greed=0.1, e_greed_decrement=0):
        super(Agent, self).__init__(algorithm)
        assert isinstance(act_dim, int)
        self.act_dim = act_dim
        self.global_step = 0
        self.update_target_steps = update_target_step
        self.e_greed = e_greed
        self.e_greed_decrement = e_greed_decrement
    def sample(self, obs):
        sample = np.random.random()
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)
        else:
            if np.random.random() < 0.01:
                act = np.random.randint(self.act_dim)
            else:
                act = self.predict(obs)
        self.e_greed = max(0.01, self.e_greed - self.e_greed_decrement)
        return act
    def predict(self, obs):
        obs = paddle.to_tensor(obs, dtype='float32')
        pred_q = self.alg.predict(obs)
        act = int(pred_q.argmax())
        return act
    def learn(self, obs, act, reward, next_obs, terminal):
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1
        act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)
        terminal = np.expand_dims(terminal, axis=-1)
        obs = paddle.to_tensor(obs, dtype='float32')
        act = paddle.to_tensor(act, dtype='int32')
        reward = paddle.to_tensor(reward, dtype='float32')
        next_obs = paddle.to_tensor(next_obs, dtype='float32')
        terminal = paddle.to_tensor(terminal, dtype='float32')
        loss = self.alg.learn(obs, act, reward, next_obs, terminal)
        return float(loss)

AP = []
AFit = []  # 精英归档
# 开始独立运行GMA start independent run for GMA
# 针对一定数量的轮次进行迭代
for rround in range(0, 2):
    # 初始化染色体和适应度值
    p_chrom= RandomRule2(ps, J_num, State)
    fitness = [[0 for _ in range(3)] for _ in range(ps)]
    NFEs = 0  # 函数评估次数
    # 计算每个解的适应度
    for i in range(ps):
        s = Sch()
        s.Decode(p_chrom[i])
        fitness[i][0]=s.fitness# 加工时间
        fitness[i][1]=s.fitness1# 机器空闲时间
        fitness[i][2]=s.fitness2# AGV运行距离
    # 构建双重深度 Q 网络 (Double Deep Q-Network, DQN)
    # obs_dim = 8
    # act_dim = 12
    # model = QNetwork(obs_dim=obs_dim, act_dim=act_dim)
    # alg = DDQN(model, gamma=GAMMA, lr=lr)
    # agent = Agent(alg, UPDATE_TARGET_STEP, act_dim=act_dim, e_greed=0.1, e_greed_decrement=1e-6)
    i = 1
    # 主要的优化循环
    while NFEs < MaxNFEs:
        print(' 第', rround + 1, '轮，迭代次数', i)
        i = i + 1
        # 应用 NSGA2POXES 算法更新染色体和适应度
        p_chrom, fitness = NSGA2POXES2(p_chrom, fitness, Pc, Pm, ps, J_num, PT, State, M_j)
        NFEs += 2 * ps
        PF = pareto(fitness)#包含帕累托前沿解在原适应度值数组中的索引的列表

        # 更新精英归档以保存非支配解
        if len(AFit) == 0:
            AP = copy.copy([p_chrom[index] for index in PF])
            AFit = copy.copy([fitness[index] for index in PF])
        else:#将新的帕累托前沿解添加到 AP AF AFit中
            AP = np.vstack((AP, [p_chrom[index] for index in PF]))
            AFit = np.vstack((AFit, [fitness[index] for index in PF]))
            #AP.append(copy.copy([p_chrom[index] for index in PF]))
            #AFit.append(copy.copy([fitness[index] for index in PF]))

        # 使用基于 DQN 的策略进行局部搜索
        # L = len(AFit)
        # current_state = list(np.zeros(J_num, dtype=int))#[0,0,...,0]
        # next_state = list(np.zeros(J_num, dtype=int))#[0,0,...,0]
        # Fit = np.zeros(3)#[0,0,0]
        #
        # for j in range(L):
        #     current_state = copy.copy(AP[j])
        #     at = agent.sample(current_state)
        #
        #     k = int(action)
        #     # 根据选择的动作应用不同的操作
        #     if k == 0:
        #         P1 = DSwap2(AP[j], AFit[j], J_num, State, PT)
        #     elif k == 1:
        #         #P1, F1 = DInsert5(AP[j, :], AF[j, :], AFit[j, :], N, F, JDD, JP)
        #         P1 = AP[j]
        #     elif k == 2:
        #         #P1, F1 = PSwap(AP[j, :], AF[j, :], AFit[j, :], N, F, JDD, JP)
        #         P1 = AP[j]
        #     elif k == 3:
        #         #P1, F1 = PInsert(AP[j, :], AF[j, :], AFit[j, :], N, F, JDD, JP)
        #         P1 = AP[j]
        #
        #     s = Sch(Job, Machine, State, PT, TT, AGV, AGV_speed, cur_site)
        #     s.Decode(P1)
        #     Fit[0] = s.fitness  # 加工时间
        #     Fit[1] = s.fitness1  # AGV运行距离
        #     Fit[2] = s.fitness2  # 机器空闲时间
        #     NFEs = NFEs + 1
        #     nr = NDS(Fit, AFit[j])#非支配排序函数，用于比较两个个体的适应度以判断支配关系。
        #
        #     # 更新精英归档并根据局部搜索结果应用奖励
        #     if nr == 1:#Fit 支配 AFit
        #         AP[j] = copy.copy(P1)
        #         AFit[j] = copy.copy(Fit)
        #         reward = 20
        #     elif nr == 0:#没有支配关系
        #         AP.append(P1)
        #         AFit.append(Fit)
        #         if AFit[j][0] < Fit[0]:
        #             reward = 15
        #         else:
        #             reward = 0
        #         if AFit[j][1] < Fit[1]:
        #             reward = 10
        #     else:#AFit 支配 Fit
        #         reward = 0
        #     next_state = copy.copy(P1)
        #     dq_net.store_transition(current_state, action, reward, next_state)
        #     if dq_net.memory_counter > 50:
        #         for epoch in range(EPOCH):
        #             dq_net.learn()
    # 将精英解写入文本文件中
plt.figure(1)
plt.plot([row[0] for row in AFit])
plt.title("完工时间趋势图")
plt.figure(2)
plt.plot([row[1] for row in AFit])
plt.title("机器平均空闲率趋势图")
plt.figure(3)
plt.plot([row[2] for row in AFit])
plt.title("AGV平均空闲率趋势图")
plt.show()
PF = pareto(AFit)
make_span3D, Idle_time3D, Agv_S3D = [], [], []
for i in PF:
    if AFit[i][1] != 0.0:
        make_span3D.append(AFit[i][0])
        Idle_time3D.append(AFit[i][1])
        Agv_S3D.append(AFit[i][2])
NSGA2_fit = []
NSGA2_fit.append(make_span3D)
NSGA2_fit.append(Idle_time3D)
NSGA2_fit.append(Agv_S3D)
# with open('data/NSGA2_fit.pkl', 'wb') as f:
#     pickle.dump(NSGA2_fit, f)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(make_span3D[:], Idle_time3D[:], Agv_S3D[:], c='r', marker='o')
ax.set_title('ThreeDimensional Trend')
ax.set_xlabel('Makespan')
ax.set_ylabel('Mac_idlerate')
ax.set_zlabel('AGV_idlerate')
plt.show()
s = Sch()
s.Decode(AP[PF[-1]])
s.Gantt()
    # AFit = [AFit[index] for index in PF]
    # PF = pareto(AFit)
    # l = len(PF)
    # obj = [row[:3] for row in AFit]
    # newobj = []
    # for i in range(l):
    #     newobj.append(obj[PF[i]])
    # newobj = list(set(tuple(row) for row in newobj))# np.unique(newobj, axis=0)  # 删除重复行
    # resPATH = 'HFSP\\res' + str(rround + 1) + '.txt'
    # f = open(resPATH, "w", encoding='utf-8')
    # l = len(newobj)
    # for i in range(l):
    #     item = '%5.2f %6.2f %7.2f\n' % (newobj[i][0], newobj[i][1], newobj[i][2])  # 格式化写入文本文件
    #     f.write(item)
    # f.close()
print('finish running')
