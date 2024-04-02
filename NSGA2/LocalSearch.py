# coding:utf-8
import copy
import math
import random
from NSGA2.Tool import *
import numpy as np

def Swap(p_chrom,N):
    #swap for operation sequence as mutation operator
    SH=N
    p1=math.floor(random.random()*N)
    p2 = math.floor(random.random() * N)
    while p1==p2:
        p2 = math.floor(random.random() * N)
    t = copy.copy(p_chrom[p1])
    p_chrom[p1] = copy.copy(p_chrom[p2])
    p_chrom[p2] = copy.copy(t);

    return p_chrom

def Insert(p_chrom,N):
    #swap for operation sequence as mutation operator
    SH=N
    pos1=math.floor(random.random()*N)
    pos2 = math.floor(random.random() * N)
    while pos1==pos2:
        pos2 = math.floor(random.random() * N)
    low = min(pos1, pos2)
    up = max(pos1, pos2)
    tmp = p_chrom[up];
    for i in range(up, low, -1):
        p_chrom[i] = copy.copy(p_chrom[i - 1])
    p_chrom[low] = copy.copy(tmp)

    return p_chrom

def DInsert(p_chrom,f_chrom,N,F,JDD):
    f_num = np.zeros(F, dtype=int);
    for f in range(F):
        f_num[f] = len(find_all_index(f_chrom, f));
    for f in range(F):
        if f_num[f] == 0:
            FC = np.zeros(N, dtype=int)
            # generate operation sequence randomly
            for i in range(N):
                FC[i] = i % F
            tmp2 = FC
            random.shuffle(tmp2)
            f_chrom = copy.copy(tmp2)

    P = [];
    FJ = []
    for f in range(F):
        P.append([])
        FJ.append([])

    for i in range(N):
        t1 = p_chrom[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        FJ[t3].append(i)


    sf = math.floor(random.random() * F);
    SP = copy.copy(P[sf]);
    SL = len(SP);
    J1 = math.floor(random.random() * SL);
    pos1 = FJ[sf][J1];
    J1 = SP[J1];

    J2 = math.floor(random.random() * SL);
    pos2 = FJ[sf][J2];
    J2 = SP[J2];
    count = 0;
    while count < 10:
        if J1 == J2:
            J2 = math.floor(random.random() * SL);
            pos2 = FJ[sf][J2];
            J2 = SP[J2];
        else:
            if pos2 > pos1 and JDD[J2] < JDD[J1]:
                break;

            if pos2 < pos1 and JDD[J2] > JDD[J1]:
                break;

        count = count + 1;
    low = min(pos1, pos2)
    up = max(pos1, pos2)
    tmp = p_chrom[up];
    for i in range(up, low, -1):
        p_chrom[i] = copy.copy(p_chrom[i - 1])
    p_chrom[low] = copy.copy(tmp)

    return p_chrom,f_chrom

def DInsert2(p_chrom,f_chrom,fitness,N,F,JDD):
    f_num = np.zeros(F, dtype=int);
    for f in range(F):
        f_num[f] = len(find_all_index(f_chrom, f));
    for f in range(F):
        if f_num[f] == 0:
            FC = np.zeros(N, dtype=int)
            # generate operation sequence randomly
            for i in range(N):
                FC[i] = i % F
            tmp2 = FC
            random.shuffle(tmp2)
            f_chrom = copy.copy(tmp2)

    maxj = int(fitness[2]);
    cf = f_chrom[maxj];
    P = [];
    FJ = []
    for f in range(F):
        P.append([])
        FJ.append([])

    for i in range(N):
        t1 = p_chrom[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        FJ[t3].append(i)


    SP = copy.copy(P[cf]);
    SL = len(SP);
    J1 = maxj;
    pos1 = find_all_index(p_chrom, J1)[0];
    posJ= find_all_index(SP, J1)[0];

    J2 = math.floor(random.random() * (SL-posJ));
    pos2 = FJ[cf][J2];
    J2 = SP[J2];
    count = 0;
    while count<10:
        if J1 == J2:
            J2 = math.floor(random.random() * (SL - posJ));
            pos2 = FJ[cf][J2];
            J2 = SP[J2];
        else:
            if JDD[J2] > JDD[J1]:
                break;
        count = count+1;

    low = min(pos1, pos2)
    up = max(pos1, pos2)
    tmp = p_chrom[up];
    for i in range(up, low, -1):
        p_chrom[i] = copy.copy(p_chrom[i - 1])
    p_chrom[low] = copy.copy(tmp)

    return p_chrom,f_chrom

def DInsert3(p_chrom,f_chrom,fitness,N,F,JDD,JP):
    f_num = np.zeros(F, dtype=int);
    for f in range(F):
        f_num[f] = len(find_all_index(f_chrom, f));
    for f in range(F):
        if f_num[f] == 0:
            FC = np.zeros(N, dtype=int)
            # generate operation sequence randomly
            for i in range(N):
                FC[i] = i % F
            tmp2 = FC
            random.shuffle(tmp2)
            f_chrom = copy.copy(tmp2)

    maxj = int(fitness[2]);
    cf = f_chrom[maxj];
    P = [];
    FJ = []
    for f in range(F):
        P.append([])
        FJ.append([])

    for i in range(N):
        t1 = p_chrom[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        FJ[t3].append(i)


    SP = copy.copy(P[cf]);
    SL = len(SP);
    J1 = maxj;
    pos1 = find_all_index(p_chrom, J1)[0];
    posJ= find_all_index(SP, J1)[0];

    J2 = math.floor(random.random() * (SL-posJ));
    pos2 = FJ[cf][J2];
    J2 = SP[J2];

    for i in range(posJ,-1,-1):
        J2 = i;
        pos2 = FJ[cf][J2];
        J2 = SP[J2];
        if JDD[J2] > JDD[J1]:
            t = copy.copy(p_chrom[pos1])
            p_chrom[pos1] = copy.copy(p_chrom[pos2])
            p_chrom[pos2] = copy.copy(t);
        elif JP[J2] > JP[J1]:
            t = copy.copy(p_chrom[pos1])
            p_chrom[pos1] = copy.copy(p_chrom[pos2])
            p_chrom[pos2] = copy.copy(t);


    return p_chrom,f_chrom

# 关键工作前找好合适的位置，先按时间，后按轻重缓急进行调换 find the suitable place before critical job and swap them according to duedate first and priorities second
def DSwap(p_chrom,f_chrom,fitness,N,F,JDD,JP):
    '''
    f_num = np.zeros(F, dtype=int);
    for f in range(F):
        f_num[f] = len(find_all_index(f_chrom, f));
    for f in range(F):
        if f_num[f] == 0:
            FC = np.zeros(N, dtype=int)
            # generate operation sequence randomly
            for i in range(N):
                FC[i] = i % F
            tmp2 = FC
            random.shuffle(tmp2)
            f_chrom = copy.copy(tmp2)
    '''
    maxj = int(fitness[2])
    cf = f_chrom[maxj]
    P = []
    FJ = []
    for f in range(F):
        P.append([])
        FJ.append([])

    for i in range(N):
        t1 = p_chrom[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        FJ[t3].append(i)

    SP = copy.copy(P[cf])
    SL = len(SP)
    J1 = maxj
    pos1 = find_all_index(p_chrom, J1)[0]
    posJ= find_all_index(SP, J1)[0]

    J2 = math.floor(random.random() * (SL-posJ))
    pos2 = FJ[cf][J2]
    J2 = SP[J2]

    for i in range(posJ,-1,-1):
        J2 = i
        pos2 = FJ[cf][J2]
        J2 = SP[J2]
        if JDD[J2] > JDD[J1]:
            t = copy.copy(p_chrom[pos1])
            p_chrom[pos1] = copy.copy(p_chrom[pos2])
            p_chrom[pos2] = copy.copy(t)
            break;
        elif JDD[J2] == JDD[J1] and JP[J2] > JP[J1]:
            t = copy.copy(p_chrom[pos1])
            p_chrom[pos1] = copy.copy(p_chrom[pos2])
            p_chrom[pos2] = copy.copy(t)
            break
    return p_chrom,f_chrom
# 关键工作前找好合适的位置，先按时间，后按轻重缓急进行调换 find the suitable place before critical job and swap them according to duedate first and priorities second
def DSwap2(p_chrom,fitness,N,F,PT):
    # posJ=
    #
    # for i in range(posJ, -1, -1):  # 从posJ开始逆向循环
    #     J2 = i  # 设置J2的值为i
    #     pos2 = FJ[J2]  # 获取染色体cf中FJ列表对应位置的值
    #     J2 = SP[J2]  # 获取SP列表中对应位置的值
    #     if JDD[J2] > JDD[J1]:  # 如果J2的JDD值大于J1的JDD值
    #         t = copy.copy(p_chrom[pos1])  # 复制p_chrom中pos1位置的值到t
    #         p_chrom[pos1] = copy.copy(p_chrom[pos2])  # 将p_chrom中pos2位置的值复制到pos1位置
    #         p_chrom[pos2] = copy.copy(t)  # 将t的值复制到pos2位置
    #         break  # 结束循环
    #     elif JDD[J2] == JDD[J1] and JP[J2] > JP[J1]:  # 如果J2的JDD值等于J1的JDD值并且J2的JP值大于J1的JP值
    #         t = copy.copy(p_chrom[pos1])  # 复制p_chrom中pos1位置的值到t
    #         p_chrom[pos1] = copy.copy(p_chrom[pos2])  # 将p_chrom中pos2位置的值复制到pos1位置
    #         p_chrom[pos2] = copy.copy(t)  # 将t的值复制到pos2位置
    #         break  # 结束循环
    return p_chrom  # 返回修改后的p_chrom列表
# 在关键工作前找合适的位置，先按时间，后按轻重缓急，将关键工作插入关键工作 find the suitable place before critical job and insert the latter into the former according to duedate first and priorities second
def DInsert5(p_chrom,f_chrom,fitness,N,F,JDD,JP):
    '''
    f_num = np.zeros(F, dtype=int);
    for f in range(F):
        f_num[f] = len(find_all_index(f_chrom, f));
    for f in range(F):
        if f_num[f] == 0:
            FC = np.zeros(N, dtype=int)
            # generate operation sequence randomly
            for i in range(N):
                FC[i] = i % F
            tmp2 = FC
            random.shuffle(tmp2)
            f_chrom = copy.copy(tmp2)
    '''
    maxj = int(fitness[2]);
    cf = f_chrom[maxj];
    P = [];
    FJ = []
    for f in range(F):
        P.append([])
        FJ.append([])

    for i in range(N):
        t1 = p_chrom[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        FJ[t3].append(i)


    SP = copy.copy(P[cf]);
    SL = len(SP);
    J1 = maxj;
    pos1 = find_all_index(p_chrom, J1)[0];
    posJ= find_all_index(SP, J1)[0];

    J2 = math.floor(random.random() * (SL-posJ));
    pos2 = FJ[cf][J2];
    J2 = SP[J2];

    for i in range(posJ,-1,-1):
        J2 = i;
        pos2 = FJ[cf][J2];
        J2 = SP[J2];
        if JDD[J2] > JDD[J1]:
            low = min(pos1, pos2)
            up = max(pos1, pos2)
            tmp = p_chrom[up];
            for i in range(up, low, -1):
                p_chrom[i] = copy.copy(p_chrom[i - 1])
            p_chrom[low] = copy.copy(tmp)
            break;
        elif JDD[J2] == JDD[J1] and JP[J2] > JP[J1]:
            low = min(pos1, pos2)
            up = max(pos1, pos2)
            tmp = p_chrom[up];
            for i in range(up, low, -1):
                p_chrom[i] = copy.copy(p_chrom[i - 1])
            p_chrom[low] = copy.copy(tmp)
            break;
    return p_chrom,f_chrom

# find the suitable place before critical job and insert the latter into the former according to duedate first and priorities second
def PInsert4(p_chrom,f_chrom,fitness,N,F,JDD,JP):
    f_num = np.zeros(F, dtype=int);
    for f in range(F):
        f_num[f] = len(find_all_index(f_chrom, f));
    for f in range(F):
        if f_num[f] == 0:
            FC = np.zeros(N, dtype=int)
            # generate operation sequence randomly
            for i in range(N):
                FC[i] = i % F
            tmp2 = FC
            random.shuffle(tmp2)
            f_chrom = copy.copy(tmp2)

    maxj = int(fitness[2]);
    cf = f_chrom[maxj];
    P = [];
    FJ = []
    for f in range(F):
        P.append([])
        FJ.append([])

    for i in range(N):
        t1 = p_chrom[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        FJ[t3].append(i)


    SP = copy.copy(P[cf]);
    SL = len(SP);
    J1 = maxj;
    pos1 = find_all_index(p_chrom, J1)[0];
    posJ= find_all_index(SP, J1)[0];

    J2 = math.floor(random.random() * (SL-posJ));
    pos2 = FJ[cf][J2];
    J2 = SP[J2];

    for i in range(posJ,-1,-1):
        J2 = i;
        pos2 = FJ[cf][J2];
        J2 = SP[J2];
        if JP[J2] > JP[J1]:
            low = min(pos1, pos2)
            up = max(pos1, pos2)
            tmp = p_chrom[up];
            for i in range(up, low, -1):
                p_chrom[i] = copy.copy(p_chrom[i - 1])
            p_chrom[low] = copy.copy(tmp)
            break;
        elif JP[J2] == JP[J1] and JDD[J2] > JDD[J1]:
            low = min(pos1, pos2)
            up = max(pos1, pos2)
            tmp = p_chrom[up];
            for i in range(up, low, -1):
                p_chrom[i] = copy.copy(p_chrom[i - 1])
            p_chrom[low] = copy.copy(tmp)
            break;
    return p_chrom,f_chrom

# 在关键工作前找到合适的位置，按优先级进行调换，按次要顺序进行调换 find the suitable place before critical job and swap them according to priorities first and  duedate second
def PSwap(p_chrom,f_chrom,fitness,N,F,JDD,JP):
    '''
    f_num = np.zeros(F, dtype=int);
    for f in range(F):
        f_num[f] = len(find_all_index(f_chrom, f));
    for f in range(F):
        if f_num[f] == 0:
            FC = np.zeros(N, dtype=int)
            # generate operation sequence randomly
            for i in range(N):
                FC[i] = i % F
            tmp2 = FC
            random.shuffle(tmp2)
            f_chrom = copy.copy(tmp2)
    '''
    maxj = int(fitness[2]);
    cf = f_chrom[maxj];
    P = [];
    FJ = []
    for f in range(F):
        P.append([])
        FJ.append([])

    for i in range(N):
        t1 = p_chrom[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        FJ[t3].append(i)


    SP = copy.copy(P[cf]);
    SL = len(SP);
    J1 = maxj;
    pos1 = find_all_index(p_chrom, J1)[0];
    posJ= find_all_index(SP, J1)[0];

    J2 = math.floor(random.random() * (SL-posJ));
    pos2 = FJ[cf][J2];
    J2 = SP[J2];

    for i in range(posJ,-1,-1):
        J2 = i;
        pos2 = FJ[cf][J2];
        J2 = SP[J2];
        if JP[J2] > JP[J1]:
            t = copy.copy(p_chrom[pos1])
            p_chrom[pos1] = copy.copy(p_chrom[pos2])
            p_chrom[pos2] = copy.copy(t);
            break;
        elif JP[J2] == JP[J1] and JDD[J2] > JDD[J1]:
            t = copy.copy(p_chrom[pos1])
            p_chrom[pos1] = copy.copy(p_chrom[pos2])
            p_chrom[pos2] = copy.copy(t);
            break;
    return p_chrom,f_chrom

def PInsert(p_chrom,f_chrom,fitness,N,F,JDD,JP):
    '''
    f_num = np.zeros(F, dtype=int);
    for f in range(F):
        f_num[f] = len(find_all_index(f_chrom, f));
    for f in range(F):
        if f_num[f] == 0:
            FC = np.zeros(N, dtype=int)
            # generate operation sequence randomly
            for i in range(N):
                FC[i] = i % F
            tmp2 = FC
            random.shuffle(tmp2)
            f_chrom = copy.copy(tmp2)
    '''
    maxj = int(fitness[2]);
    cf = f_chrom[maxj];
    P = [];
    FJ = []
    for f in range(F):
        P.append([])
        FJ.append([])

    for i in range(N):
        t1 = p_chrom[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        FJ[t3].append(i)

    SP = copy.copy(P[cf]);
    SL = len(SP);
    J1 = maxj;
    pos1 = find_all_index(p_chrom, J1)[0];
    posJ = find_all_index(SP, J1)[0];

    J2 = math.floor(random.random() * (SL - posJ));
    pos2 = FJ[cf][J2];
    J2 = SP[J2];

    for i in range(posJ, -1, -1):
        J2 = i;
        pos2 = FJ[cf][J2];
        J2 = SP[J2];
        if JP[J2] > JP[J1]:
            break;
        elif JP[J2] == JP[J1] and JDD[J2] > JDD[J1]:
            break;

    low = min(pos1, pos2)
    up = max(pos1, pos2)
    tmp = p_chrom[up];
    for i in range(up, low, -1):
        p_chrom[i] = copy.copy(p_chrom[i - 1])
    p_chrom[low] = copy.copy(tmp)
    return p_chrom,f_chrom


def FInsert2(p_chrom,f_chrom,fitness,N,F,JDD,JP):
    f_num = np.zeros(F, dtype=int);
    for f in range(F):
        f_num[f] = len(find_all_index(f_chrom, f));
    for f in range(F):
        if f_num[f] == 0:
            FC = np.zeros(N, dtype=int)
            # generate operation sequence randomly
            for i in range(N):
                FC[i] = i % F
            tmp2 = FC
            random.shuffle(tmp2)
            f_chrom = copy.copy(tmp2)

    maxj = int(fitness[2]);
    cf = f_chrom[maxj];
    P = [];
    FJ = []
    for f in range(F):
        P.append([])
        FJ.append([])

    for i in range(N):
        t1 = p_chrom[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        FJ[t3].append(i)
    sf1=cf
    sf2 = math.floor(random.random() * F);
    while sf1==sf2:
        sf2 = math.floor(random.random() * F);
    J1 = maxj
    pos1 = find_all_index(p_chrom,J1)[0];
    SP = copy.copy(P[sf2]);
    SL = len(SP);

    J2 = math.floor(random.random() * SL);
    pos2 = FJ[sf2][J2];
    J2 = SP[J2];
    for i in range(SL-1,-1,-1):
        J2 = i;
        pos2 = FJ[sf2][J2];
        J2 = SP[J2];
        if JDD[J2] > JDD[J1]:
            break;
        elif JDD[J2] == JDD[J1] and JP[J2] > JP[J1]:
            break;

    low = min(pos1, pos2)
    up = max(pos1, pos2)
    tmp = p_chrom[up];
    for i in range(up, low, -1):
        p_chrom[i] = copy.copy(p_chrom[i - 1])
    p_chrom[low] = copy.copy(tmp)
    f_chrom[J1] = sf2;

    f_num = np.zeros(F, dtype=int);
    for f in range(F):
        f_num[f] = len(find_all_index(f_chrom, f));
    for f in range(F):
        if f_num[f] == 0:
            FC = np.zeros(N, dtype=int)
            # generate operation sequence randomly
            for i in range(N):
                FC[i] = i % F
            tmp2 = FC
            random.shuffle(tmp2)
            f_chrom = copy.copy(tmp2)
    return p_chrom,f_chrom

def FInsert(p_chrom,f_chrom,fitness,N,F,JDD):
    '''
    f_num = np.zeros(F, dtype=int);
    for f in range(F):
        f_num[f] = len(find_all_index(f_chrom, f));
    for f in range(F):
        if f_num[f] == 0:
            FC = np.zeros(N, dtype=int)
            # generate operation sequence randomly
            for i in range(N):
                FC[i] = i % F
            tmp2 = FC
            random.shuffle(tmp2)
            f_chrom = copy.copy(tmp2)
    '''
    maxj = int(fitness[2]);
    cf = f_chrom[maxj];
    P = [];
    FJ = []
    for f in range(F):
        P.append([])
        FJ.append([])

    for i in range(N):
        t1 = p_chrom[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        FJ[t3].append(i)
    sf1=cf
    sf2 = math.floor(random.random() * F);
    while sf1==sf2:
        sf2 = math.floor(random.random() * F);
    J1 = maxj
    pos1 = find_all_index(p_chrom,J1)[0];
    SP = copy.copy(P[sf2]);
    SL = len(SP);
    J2 = math.floor(random.random() * SL);
    pos2 = FJ[sf2][J2];
    J2 = SP[J2];
    count = 0;
    while count < 10:
        if J1 == J2:
            J2 = math.floor(random.random() * SL);
            pos2 = FJ[sf2][J2];
            J2 = SP[J2];
        else:
            if JDD[J2] > JDD[J1]:
                break;
        count = count + 1;
    low = min(pos1, pos2)
    up = max(pos1, pos2)
    tmp = p_chrom[up];
    for i in range(up, low, -1):
        p_chrom[i] = copy.copy(p_chrom[i - 1])
    p_chrom[low] = copy.copy(tmp)
    f_chrom[J1] = sf2;

    f_num = np.zeros(F, dtype=int);
    for f in range(F):
        f_num[f] = len(find_all_index(f_chrom, f));
    for f in range(F):
        if f_num[f] == 0:
            FC = np.zeros(N, dtype=int)
            # generate operation sequence randomly
            for i in range(N):
                FC[i] = i % F
            tmp2 = FC
            random.shuffle(tmp2)
            f_chrom = copy.copy(tmp2)
    return p_chrom,f_chrom

def FSwap2(p_chrom,f_chrom,fitness,N,F,JDD):
    f_num = np.zeros(F, dtype=int);
    for f in range(F):
        f_num[f] = len(find_all_index(f_chrom, f));
    for f in range(F):
        if f_num[f] == 0:
            FC = np.zeros(N, dtype=int)
            # generate operation sequence randomly
            for i in range(N):
                FC[i] = i % F
            tmp2 = FC
            random.shuffle(tmp2)
            f_chrom = copy.copy(tmp2)


    P = [];
    FJ = []
    for f in range(F):
        P.append([])
        FJ.append([])

    for i in range(N):
        t1 = p_chrom[i]
        t3 = f_chrom[t1]
        P[t3].append(p_chrom[i])
        FJ[t3].append(i)

    sf1= math.floor(random.random() * F);
    sf2 = math.floor(random.random() * F);
    while sf1==sf2:
        sf2 = math.floor(random.random() * F);
    SP = copy.copy(P[sf1]);
    SL = len(SP);
    J1 = math.floor(random.random() * SL);
    pos1 = FJ[sf1][J1];
    J1 = SP[J1];


    SP = copy.copy(P[sf2]);
    SL = len(SP);
    J2 = math.floor(random.random() * SL);
    pos2 = FJ[sf2][J2];
    J2 = SP[J2];
    count = 0;
    while count < 10:
        if J1 == J2:
            J2 = math.floor(random.random() * SL);
            pos2 = FJ[sf2][J2];
            J2 = SP[J2];
        else:
            if pos2 < pos1 and JDD[J2] > JDD[J1]:
                break;

            if pos2 > pos1 and JDD[J2] < JDD[J1]:
                break;
        count = count + 1;
    t = copy.copy(p_chrom[pos1])
    p_chrom[pos1] = copy.copy(p_chrom[pos2])
    p_chrom[pos2] = copy.copy(t);

    f_chrom[J2] = sf1;
    f_chrom[J1] = sf2;

    f_num = np.zeros(F, dtype=int);
    for f in range(F):
        f_num[f] = len(find_all_index(f_chrom, f));
    for f in range(F):
        if f_num[f] == 0:
            FC = np.zeros(N, dtype=int)
            # generate operation sequence randomly
            for i in range(N):
                FC[i] = i % F
            tmp2 = FC
            random.shuffle(tmp2)
            f_chrom = copy.copy(tmp2)
    return p_chrom,f_chrom
