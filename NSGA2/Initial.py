# coding:utf-8
import copy

import numpy as np
import random
import math
from NSGA2.Tool import mymin
def initial(popsize,N,F):
    #create operation sequence and machine selection vectors
    p_chrom=np.zeros(shape=(popsize,N),dtype=int)
    f_chrom = np.zeros(shape=(popsize, N), dtype=int)

    chrom=np.zeros(N,dtype=int)
    FC = np.zeros(N, dtype=int)
    #generate operation sequence randomly
    for i in range(N):
        chrom[i]=i
        FC[i]=i%F

    tmp=chrom;tmp2=FC
    random.shuffle(tmp)
    random.shuffle(tmp2)
    p_chrom[0,:]=tmp
    f_chrom[0,:]=tmp2

    for i in range(1,popsize):
        tmp=p_chrom[i-1,:]
        random.shuffle(tmp)
        p_chrom[i,:]=tmp
        tmp2 = f_chrom[i - 1, :]
        random.shuffle(tmp2)
        f_chrom[i, :] = tmp2
    #finish generate operation sequencing sizeing ps
    return p_chrom,f_chrom



def LoadBalanceRule(popsize,time,N,F,TS):
    p_chrom = np.zeros(shape=(popsize, N), dtype=int)
    f_chrom = np.zeros(shape=(popsize, N), dtype=int)

    chrom = np.zeros(N, dtype=int)
    FC = np.zeros(N, dtype=int)
    # generate operation sequence randomly
    for i in range(N):
        chrom[i] = i
    tmp = chrom;
    random.shuffle(tmp)
    p_chrom[0, :] = tmp

    for i in range(1, popsize):
        tmp = p_chrom[i - 1, :]
        random.shuffle(tmp)
        p_chrom[i, :] = tmp

    TotFTime=np.zeros(shape=(F, N))
    for f in range(F):
        for i in range(N):
            for k in range(TS):
                TotFTime[f][i]=TotFTime[f][i]+time[f][k][i];

    for n in range(popsize):
        f_load=np.zeros(F);
        for i in range(N):
            min_f=mymin(f_load);
            if len(min_f)>1:
                ptime = TotFTime[min_f,i]
                min_tf = np.argmin(ptime);
                f_chrom[n][i]=min_f[min_tf];
                x=min_f[min_tf];
                f_load[x]=f_load[x]+TotFTime[x][i];
            else:
                min_f=min_f[0]
                f_chrom[n][i]=min_f
                f_load[min_f]=f_load[min_f]+TotFTime[min_f][i];

    return p_chrom,f_chrom

def RankRule(popsize, JP, JDD, N, F):
    # 初始化染色体和适应度值数组
    p_chrom = np.zeros(shape=(popsize, N), dtype=int)
    f_chrom = np.zeros(shape=(popsize, N), dtype=int)

    # 对作业的加工时间和交货日期进行升序排序得到索引数组
    p_index = np.argsort(JP)
    d_index = np.argsort(JDD)

    # 初始化加工时间和交货日期的概率数组
    p_pro = np.zeros(N)
    d_pro = np.zeros(N)

    # 初始化加权和概率数组
    s_pro = np.zeros(N)

    # 根据排名计算加工时间和交货日期的概率
    for i in range(N):
        p_pro[p_index[i]] = (N - i) / N
        d_pro[d_index[i]] = (N - i) / N

    # 初始化工序数组
    FC = np.zeros(N, dtype=int)

    # 生成随机的工序序列
    for i in range(N):
        FC[i] = i % F
        s_pro[i] = p_pro[i] + d_pro[i]

    # 根据加权和概率排序得到作业的处理顺序
    P = np.argsort(s_pro)[::-1]

    # 将第一个染色体设置为按照处理顺序的作业序列，第一个工序序列随机排列
    p_chrom[0, :] = P
    tmp2 = copy.copy(FC)
    random.shuffle(tmp2)
    f_chrom[0, :] = copy.copy(tmp2)

    # 生成其余染色体，每个染色体的作业序列相同，工序序列随机排列
    for i in range(1, popsize):
        p_chrom[i, :] = copy.copy(P)
        tmp2 = f_chrom[i - 1, :]
        random.shuffle(tmp2)
        f_chrom[i, :] = tmp2

    return p_chrom, f_chrom, s_pro

def RandomRule(popsize,N,F):
    #创建操作程序和机器选择向量 create operation sequence and machine selection vectors
    p_chrom=np.zeros(shape=(popsize,N),dtype=int)
    f_chrom = np.zeros(shape=(popsize, N), dtype=int)

    chrom=np.zeros(N,dtype=int)
    FC = np.zeros(N, dtype=int)
    #随机生成操作顺序 generate operation sequence randomly
    for i in range(N):
        chrom[i]=i
        FC[i]=i%F
    tmp=chrom
    tmp2=FC
    random.shuffle(tmp)
    random.shuffle(tmp2)
    p_chrom[0,:]=tmp
    f_chrom[0,:]=tmp2

    for i in range(1,popsize):
        tmp=p_chrom[i-1,:]
        random.shuffle(tmp)
        p_chrom[i,:]=tmp
        tmp2=f_chrom[i-1,:]
        random.shuffle(tmp2)
        f_chrom[i,:]=tmp2
    #完成生成操作排序大小ps finish generate operation sequencing sizeing ps
    return p_chrom,f_chrom
def RandomRule2(popsize,N,F):#HFSP
    # 使用 随机 规则生成染色体
    p_chrom=[]
    for i in range(popsize):
        tmp = []
        for j in range(F):
            tmp.extend([i for i in range(1,N+1)])
        random.shuffle(tmp)
        p_chrom.append(tmp)
    return p_chrom

def HInitial(popsize, N, F, TS, time, JP, JDD):
    # 计算子种群规模
    sub_ps = math.ceil(popsize / 5)
    sub_ps3 = popsize - sub_ps * 2
    # 使用 随机 规则生成染色体和适应度值
    p_chrom1, f_chrom1 = RandomRule(sub_ps3, N, F)
    # 使用 排位 规则生成染色体和适应度值
    p_chrom2, f_chrom2, select_pro = RankRule(sub_ps, JP, JDD, N, F)
    # 使用 负载均衡 规则生成染色体和适应度值
    p_chrom3, f_chrom3 = LoadBalanceRule(sub_ps, time, N, F, TS)
    # 将子种群的染色体和适应度值合并
    p_chrom = np.vstack((p_chrom1, p_chrom2, p_chrom3))
    f_chrom = np.vstack((f_chrom1, f_chrom2, f_chrom3))

    return p_chrom, f_chrom, select_pro

def HInitial2(popsize, N, F,TS,time,JP,JDD):
    sub_ps = math.ceil(popsize / 10)
    sub_ps3 = popsize - sub_ps * 2
    p_chrom2, f_chrom2,select_pro = RankRule(sub_ps,JP, JDD, N, F)
    p_chrom3, f_chrom3 = LoadBalanceRule(sub_ps,time, N, F, TS)
    p_chrom1, f_chrom1 = RandomRule(sub_ps3,N,F)

    p_chrom = np.vstack((p_chrom1,p_chrom2,p_chrom3))
    f_chrom = np.vstack((f_chrom1,f_chrom2,f_chrom3))

    return p_chrom, f_chrom

def HInitial3(popsize, N, F,TS,time,JP,JDD):
    sub_ps = math.ceil(popsize / 4);
    sub_ps3 = popsize - sub_ps * 2;
    [p_chrom2, f_chrom2] = RankRule(sub_ps,JP, JDD, N, F);
    p_chrom3, f_chrom3 = LoadBalanceRule(sub_ps,time, N, F, TS);
    [p_chrom1, f_chrom1] = RandomRule(sub_ps3,N,F);

    p_chrom = np.vstack((p_chrom1,p_chrom2,p_chrom3))
    f_chrom = np.vstack((f_chrom1,f_chrom2,f_chrom3))

    return p_chrom, f_chrom

def HInitial4(popsize, N, F,TS,time,JP,JDD):
    sub_ps = math.ceil(popsize / 3);
    sub_ps3 = popsize - sub_ps * 2;
    [p_chrom2, f_chrom2] = RankRule(sub_ps,JP, JDD, N, F);
    p_chrom3, f_chrom3 = LoadBalanceRule(sub_ps,time, N, F, TS);
    [p_chrom1, f_chrom1] = RandomRule(sub_ps3,N,F);

    p_chrom = np.vstack((p_chrom1,p_chrom2,p_chrom3))
    f_chrom = np.vstack((f_chrom1,f_chrom2,f_chrom3))

    return p_chrom, f_chrom