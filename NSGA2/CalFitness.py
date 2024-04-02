# coding:utf-8
import copy

import numpy as np

def FitHFSP(p_chrom,FJ,f_index,TS, time ,NS, JP, JDD):
    #processing power and idle power
    N=len(FJ)
    finish = np.zeros((N, TS));
    start = np.zeros((N, TS));
    workpower = 0;totalidletime = 0;
    W_power = 4;Idle_power = 1;
    for k in range(TS):
        mftime = np.zeros(NS[k]);s = k;
        for i in range(N):
            t = p_chrom[i];
            if k == 0:
                if i==0:
                    start[i][k] = 0;
                    finish[i][k] = start[i][k] + time[f_index][s][t];
                    mftime[0] = finish[i][k]
                else:
                    m_index = np.argmin(mftime); #找到最小完工时间的机器
                    start[i][k] = mftime[m_index]
                    finish[i][k] = start[i][k] + time[f_index][s][t];
                    mftime[m_index] = finish[i][k]
                workpower = workpower + time[f_index][s][t] * W_power;
            else:
                if i == 0:
                    start[i][k] = finish[i][k-1]
                    finish[i][k] = start[i][k] + time[f_index][s][t];
                    mftime[0] = finish[i][k]
                else:
                    m_index = np.argmin(mftime)
                    start[i][k] = max(finish[i][k-1], mftime[m_index]);
                    finish[i][k] = start[i][k] + time[f_index][s][t];
                    totalidletime = totalidletime + start[i][k] - mftime[m_index]
                    mftime[m_index] = finish[i][k]
                workpower = workpower + time[f_index][s][t] * W_power;
                    #start decoding

    try:
        Cmax = finish[N - 1][TS - 1];
    except:
        Cmax = 0;
    TEC = workpower + totalidletime * Idle_power;
    DueData_Validate = 0;
    Custom_satisified = 0;
    MaxDue = 0;MaxJob = 1;
    for i in range(N):
        x = finish[i][TS - 1] - JDD[p_chrom[i]];
        DueData_Validate = max(0, x);
        if DueData_Validate > 0:
            temp = 0;
            p = JP[p_chrom[i]];
            if p == 1:
                temp = DueData_Validate * 2;
                # 必保的项目，死命令的项目，如果超过交货期则不可接受，那么客户满意度为无穷大，这里为10000
                Custom_satisified = Custom_satisified + temp;
            elif p == 2:
                temp = DueData_Validate * 1;
                Custom_satisified = Custom_satisified + temp;
            elif p == 3:
                temp = DueData_Validate * 0.5;
                Custom_satisified = Custom_satisified + temp;
            if MaxDue < temp:
                MaxDue = temp;
                MaxJob = p_chrom[i];

    return Cmax,TEC,Custom_satisified,MaxJob,MaxDue

def FitDHHFSP(p_chrom,f_chrom,N,time,F,TS,NS, JP, JDD):

    P0=[];P=[];FJ=[]
    for f in range(F):
        P.append([])
        FJ.append([])

    for i in range(N):
        t1=p_chrom[i]
        t3=f_chrom[t1]
        P[t3].append(p_chrom[i])
        FJ[t3].append(i)
    sub_f_fit=np.zeros(shape=(F,5))

    for f in range(F):
        sub_f_fit[f][0],sub_f_fit[f][1],sub_f_fit[f][2],sub_f_fit[f][3],sub_f_fit[f][4]=FitHFSP(P[f],FJ[f],f,TS, time ,NS, JP, JDD)

    fit1=sub_f_fit[0][0]
    fit3 = 1;fit2 = 0;fit4 = 0;
    for f in range(F):
        fit2 = sub_f_fit[f][1] + fit2;
        fit4 = sub_f_fit[f][2] + fit4;
        if fit1 < sub_f_fit[f][0]:
            fit1 = sub_f_fit[f][0]
            fit3 = f;
    index= np.argmax(sub_f_fit[:, 4]);
    fit5 = sub_f_fit[index][3];
    return fit4,fit2,fit5

def RightShift(p_chrom,FJ,f_index,TS, time ,NS, JP, JDD):
    #processing power and idle power
    # 初始化各种参数和数组
    N = len(FJ)
    finish = np.zeros((N, TS))
    start = np.zeros((N, TS))
    workpower = 0
    totalidletime = 0
    W_power = 4
    Idle_power = 1
    ms = np.zeros((N, TS))

    # 循环处理每个时间步
    for k in range(TS):
        mftime = np.zeros(NS[k])
        s = k
        # 循环处理每个作业
        for i in range(N):
            t = p_chrom[i]
            if k == 0:# 第一个时间步的处理
                if i == 0:
                    start[i][k] = 0
                    finish[i][k] = start[i][k] + time[f_index][s][t]
                    mftime[0] = finish[i][k]
                    ms[i][k] = 0
                else:
                    m_index = np.argmin(mftime) #找到最小完工时间的机器
                    start[i][k] = mftime[m_index]
                    finish[i][k] = start[i][k] + time[f_index][s][t]
                    mftime[m_index] = finish[i][k]
                    ms[i][k] = m_index
                workpower = workpower + time[f_index][s][t] * W_power
            else:# 非第一个时间步的处理
                if i == 0:
                    start[i][k] = finish[i][k-1]
                    finish[i][k] = start[i][k] + time[f_index][s][t]
                    mftime[0] = finish[i][k]
                    ms[i][k] = 0
                else:
                    m_index = np.argmin(mftime)
                    start[i][k] = max(finish[i][k-1], mftime[m_index])
                    finish[i][k] = start[i][k] + time[f_index][s][t]
                    totalidletime = totalidletime + start[i][k] - mftime[m_index]
                    mftime[m_index] = finish[i][k]
                    ms[i][k] = m_index
                workpower = workpower + time[f_index][s][t] * W_power
                    #start decoding
    try:
        Cmax = finish[N-1][TS-1]
    except:
        Cmax=0
        print(N,FJ)

    # 备份完成时间和开始时间数组，用于下一个操作
    finish2 = copy.copy(finish)
    start2 = copy.copy(start)
    Idletime2 = np.zeros((N, TS))
    totalidletime2 = 0

    # 循环处理每个时间步，从后往前
    for k in range(TS - 1, -1, -1):
        mstime = np.zeros(NS[k])
        s = k
        # 循环处理每个作业，从后往前
        for i in range(N - 1, -1, -1):
            t = p_chrom[i]
            # 最后一个时间步的处理
            if k == TS - 1:
                if i == N - 1:
                    cms = ms[i][k]
                    cms = int(cms)
                    mstime[cms] = start2[i][k]
                else:
                    cms = ms[i][k]
                    cms = int(cms)
                    if mstime[cms] == 0:
                        mstime[cms] = start2[i][k]
                    else:
                        if finish[i][k] < mstime[cms] and finish[i][k] < JDD[p_chrom[i]]:
                            finish2[i][k] = min(mstime[cms], JDD[p_chrom[i]])
                            start2[i][k] = finish2[i][k] - time[f_index][s][t]
                            totalidletime2 = totalidletime2 + mstime[cms] - finish2[i][k]
                            Idletime2[i][k] = mstime[cms] - finish2[i][k]
                            mstime[cms] = start2[i][k]
            else:
                # 非最后一个时间步的处理
                if i == N - 1:
                    cms = ms[i][k]
                    cms = int(cms)
                    mstime[cms] = start2[i][k]
                else:
                    cms = ms[i][k]
                    cms = int(cms)
                    if mstime[cms] == 0:
                        mstime[cms] = start2[i][k]
                    else:
                        finish2[i][k] = min(mstime[cms], start2[i][k + 1])
                        start2[i][k] = finish2[i][k] - time[f_index][s][t]
                        totalidletime2 = totalidletime2 + mstime[cms] - finish2[i][k]
                        Idletime2[i][k] = mstime[cms] - finish2[i][k]
                        mstime[cms] = start2[i][k]

    TEC = workpower + totalidletime * Idle_power
    TEC2= workpower + totalidletime2 * Idle_power
    DueData_Validate = 0
    Custom_satisified = 0
    MaxDue = 0;MaxJob = 1
    # 计算逾期数据验证、客户满意度和最大逾期时间等
    for i in range(N):
        x = finish[i][TS-1]-JDD[p_chrom[i]]
        DueData_Validate = max(0, x)
        if DueData_Validate > 0:
            temp=0
            p = JP[p_chrom[i]]
            if p == 1:
                temp = DueData_Validate * 2
                # 必保的项目，死命令的项目，如果超过交货期则不可接受，那么客户满意度为无穷大，这里为10000
                Custom_satisified = Custom_satisified + temp
            elif p==2:
                temp = DueData_Validate * 1
                Custom_satisified = Custom_satisified + temp
            elif p == 3:
                temp = DueData_Validate * 0.5
                Custom_satisified = Custom_satisified + temp
            if MaxDue < temp:
                MaxDue = temp
                MaxJob = p_chrom[i]

    return Cmax,TEC2,Custom_satisified,MaxJob,MaxDue#Cmax: 作业在流水线上完成加工的最大时间。TEC2: 表示任务调度的总能耗，包括加工所需的能耗和空闲时间所消耗的能耗。
                                                    #Custom_satisified: 表示客户满意度，根据作业的交货期和实际完成时间之间的差异进行计算。
                                                    #MaxJob: 表示导致最大逾期时间的作业的索引。MaxDue: 表示最大逾期时间，即任务最晚完成时间与实际完成时间之间的时间差。


def EnergySave_DHHFSP(p_chrom, f_chrom, N, time, F, TS, NS, JP, JDD):
    # 初始化P0、P和FJ数组，用于存储作业的信息
    P0 = []
    P = []
    FJ = []
    for f in range(F):
        P.append([])
        FJ.append([])

    # 将作业分配到不同的工序中
    for i in range(N):
        t1 = p_chrom[i]#工件序号
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        FJ[t3].append(i)

    # 计算每个工序的适应度值
    sub_f_fit = np.zeros(shape=(F, 5))
    for f in range(F):
        sub_f_fit[f][0], sub_f_fit[f][1], sub_f_fit[f][2], sub_f_fit[f][3], sub_f_fit[f][4] = RightShift(P[f], FJ[f], f, TS, time, NS, JP, JDD)
        # [0]: 作业在流水线上完成加工的最大时间。
        # [1]: 任务调度的总能耗，包括加工所需的能耗和空闲时间所消耗的能耗。
        # [2]: 客户满意度，根据作业的交货期和实际完成时间之间的差异进行计算。
        # [3]: 导致最大逾期时间的作业的索引。
        # [4]: 最大逾期时间，即任务最晚完成时间与实际完成时间之间的时间差。

    # 计算总适应度值
    fit1 = sub_f_fit[0][0]
    fit3 = 1
    fit2 = 0
    fit4 = 0
    for f in range(F):
        fit2 = sub_f_fit[f][1] + fit2
        fit4 = sub_f_fit[f][2] + fit4
        if fit1 < sub_f_fit[f][0]:
            fit1 = sub_f_fit[f][0]
            fit3 = f
    index = np.argmax(sub_f_fit[:, 4])
    fit5 = sub_f_fit[index][3]

    return fit4, fit2, fit5# 客户满意度，任务调度的总能耗，导致最大逾期时间的作业的索引
