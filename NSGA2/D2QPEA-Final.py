# coding:utf-8
import math
import random

from DataRead import DataReadDHHJSP
import os
from Initial import *
import numpy as np
from CalFitness import FitDHHFSP,EnergySave_DHHFSP
from GA import NSGA2POX,NSGA2POXES
import copy
from Tool import *
from LocalSearch import *
from DDQN_model import DoubleDQN
import torch

FILENAME=['20J3S2F.txt','20J5S2F.txt','20J3S3F.txt','20J5S3F.txt',
          '40J3S2F.txt','40J5S2F.txt','40J3S3F.txt','40J5S3F.txt',
          '60J3S2F.txt','60J5S2F.txt','60J3S3F.txt','60J5S3F.txt',
          '80J3S2F.txt','80J5S2F.txt','80J3S3F.txt','80J5S3F.txt',
          '100J3S2F.txt','100J5S2F.txt','100J3S3F.txt','100J5S3F.txt']
#FILENAME=['6J2S2F.txt','6J3S2F.txt','7J2S2F.txt','7J3S2F.txt','8J2S2F.txt',
#          '8J3S2F.txt','9J2S2F.txt','9J3S2F.txt','10J2S2F.txt','10J3S2F.txt']
#parameter
filenum=20
runtime=10
#ps=80;Pc=1.0;Pm=0.2;
ps=100;Pc=1.0;Pm=0.2
lr=0.005;batch_size=32
EPSILON = 0.9               # greedy policy
GAMMA = 0.95                 # reward discount
TARGET_REPLACE_ITER = 7   # target update frequency
MEMORY_CAPACITY = 512
N_ACTIONS = 4  # 6种候选的算子
EPOCH=1
print(torch.cuda.is_available())

DATAPATH='./DATASET_DHHFSP/'
INSPATH=[];RESPATH=[]
for file in range(filenum):
    temp=DATAPATH+FILENAME[file]
    INSPATH.append(temp)
    if file<9:
        id='0'+str(file+1)
    else:
        id=str(file+1)

    temp2='DHHFSP'+id+'\\'
    RESPATH.append(temp2)

for file in range(0,20):
    N,TS,F,NS,time,JP,JDD=DataReadDHHJSP(INSPATH[file])#工件数 时间步长 车间数量 每个时间槽的机器数量 加工时间[车间][时间槽][工件] 每个任务的优先级 每个任务的交货期
    MaxNFEs = 400 * N
    if MaxNFEs < 20000:
        MaxNFEs = 20000
    # 创建文件路径来存储每次独立运行的pareto解决方案集 create filepath to store the pareto solutions set for each independent run
    respath = 'CPTEST2\\D2QPEA\\'
    sprit = '\\'
    respath = respath + RESPATH[file]
    isExist = os.path.exists(respath)
    # 如果没有创建结果路径 if the result path has not been created
    if not isExist:
        currentpath = os.getcwd()
        os.makedirs(currentpath + sprit + respath)
    print(RESPATH[file], 'is being Optimizing\n')
    # 开始独立运行GMA start independent run for GMA
    # 针对一定数量的轮次进行迭代
    for rround in range(0, 20):
        # 初始化染色体和适应度值
        p_chrom, f_chrom, _ = HInitial(ps, N, F, TS, time, JP, JDD)
        fitness = np.zeros(shape=(ps, 3))
        NFEs = 0  # 函数评估次数

        # 计算每个解的适应度
        for i in range(ps):
            fitness[i, 0], fitness[i, 1], fitness[i, 2] = EnergySave_DHHFSP(p_chrom[i, :], f_chrom[i, :], N, time, F, TS, NS, JP, JDD)
            # 客户满意度，任务调度的总能耗，导致最大逾期时间的作业的索引
        AP = []
        AF = []
        AFit = []  # 精英归档

        # 构建双重深度 Q 网络 (Double Deep Q-Network, DQN)
        N_STATES = 2 * N
        CountOpers = np.zeros(N_ACTIONS)
        PopCountOpers = []
        dq_net = DoubleDQN(N_STATES, N_ACTIONS, BATCH_SIZE=batch_size, LR=lr, EPSILON=EPSILON, GAMMA=GAMMA, MEMORY_CAPACITY=MEMORY_CAPACITY, TARGET_REPLACE_ITER=TARGET_REPLACE_ITER)

        i = 1
        # 主要的优化循环
        while NFEs < MaxNFEs:
            print(FILENAME[file] + ' 第', rround + 1, '轮，迭代次数', i)
            i = i + 1

            # 应用 NSGA2POXES 算法更新染色体和适应度
            p_chrom, f_chrom, fitness = NSGA2POXES(p_chrom, f_chrom, fitness, Pc, Pm, ps, N, time, F, TS, NS, JP, JDD)
            NFEs = NFEs + 2 * ps
            PF = pareto(fitness)#包含帕累托前沿解在原适应度值数组中的索引的列表

            # 更新精英归档以保存非支配解
            if len(AFit) == 0:
                AP = copy.copy(p_chrom[PF, :])
                AF = copy.copy(f_chrom[PF, :])
                AFit = copy.copy(fitness[PF, :])
            else:#将新的帕累托前沿解添加到 AP AF AFit中
                AP = np.vstack((AP, p_chrom[PF, :]))
                AF = np.vstack((AF, f_chrom[PF, :]))
                AFit = np.vstack((AFit, fitness[PF, :]))

            # 使用基于 DQN 的策略进行局部搜索
            L = len(AFit)
            current_state = np.zeros(N_STATES, dtype=int)#[0,0,...,0]
            next_state = np.zeros(N_STATES, dtype=int)#[0,0,...,0]
            Fit = np.zeros(3)#[0,0,0]

            for j in range(L):
                current_state[0:N] = copy.copy(AP[j, :])
                current_state[N:N * 2] = copy.copy(AF[j, :])#染色体+适应度
                action = dq_net.choose_action(current_state)
                k = int(action)

                # 根据选择的动作应用不同的操作
                if k == 0:
                    P1, F1 = DSwap(AP[j, :], AF[j, :], AFit[j, :], N, F, JDD, JP)
                elif k == 1:
                    P1, F1 = DInsert5(AP[j, :], AF[j, :], AFit[j, :], N, F, JDD, JP)
                elif k == 2:
                    P1, F1 = PSwap(AP[j, :], AF[j, :], AFit[j, :], N, F, JDD, JP)
                elif k == 3:
                    P1, F1 = PInsert(AP[j, :], AF[j, :], AFit[j, :], N, F, JDD, JP)

                Fit[0], Fit[1], Fit[2] = EnergySave_DHHFSP(P1, F1, N, time, F, TS, NS, JP, JDD)
                NFEs = NFEs + 1
                nr = NDS(Fit, AFit[j, :])#非支配排序函数，用于比较两个个体的适应度以判断支配关系。

                # 更新精英归档并根据局部搜索结果应用奖励
                if nr == 1:#Fit 支配 AFit
                    AP[j, :] = copy.copy(P1)
                    AF[j, :] = copy.copy(F1)
                    AFit[j, :] = copy.copy(Fit)
                    reward = 20
                elif nr == 0:#没有支配关系
                    AP = np.vstack((AP, P1))
                    AF = np.vstack((AF, F1))
                    AFit = np.vstack((AFit, Fit))
                    if AFit[j, 0] < Fit[0]:
                        reward = 15
                    else:
                        reward = 0
                    if AFit[j, 1] < Fit[1]:
                        reward = 10
                else:#AFit 支配 Fit
                    reward = 0
                next_state[0:N] = copy.copy(P1)
                next_state[N:N * 2] = copy.copy(F1)
                dq_net.store_transition(current_state, action, reward, next_state)
                if dq_net.memory_counter > 50:
                    for epoch in range(EPOCH):
                        dq_net.learn()
        # 将精英解写入文本文件中
        PF = pareto(AFit)
        AP = AP[PF, :]
        AF = AF[PF, :]
        AFit = AFit[PF, :]
        PF = pareto(AFit)
        l = len(PF)
        obj = AFit[:, 0:2]
        newobj = []
        for i in range(l):
            newobj.append(obj[PF[i], :])
        newobj = np.unique(newobj, axis=0)  # 删除重复行
        tmp = 'res'
        resPATH = respath + sprit + tmp + str(rround + 1) + '.txt'
        f = open(resPATH, "w", encoding='utf-8')
        l = len(newobj)
        for i in range(l):
            item = '%5.2f %6.2f\n' % (newobj[i][0], newobj[i][1])  # 格式化写入文本文件
            f.write(item)
        f.close()
    print('完成 ' + FILENAME[file])
print('finish running')
