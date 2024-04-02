# coding:utf-8
import numpy as np

import numpy as np
import math
import random
from NSGA2.Tool import *
import copy
from NSGA2.FastNDSort import *
from NSGA2.CalFitness import *
from HFSP_Instance import J_num,State,M_j,PT,JNN, D_pq, v, V_r
from NSGA2.HFSP_env import Scheduling as Sch
from HFSP_Env_MDDQN_NSGA2_240109 import Situation
from collections import defaultdict
def TSelection(p_chrom,f_chrom,fitness,ps,N):
    #mating selection pool
    pool_size=ps
    P_pool = np.zeros(shape=(ps, N), dtype=int)
    F_pool = np.zeros(shape=(ps, N), dtype=int) #fitness of pool solutions
    # compeitor number
    tour=2
    # tournament selection
    for i in range(pool_size):
        index1=int(math.floor(random.random()*ps))
        index2 = int(math.floor(random.random() * ps))
        while index1==index2:
            index2 = int(math.floor(random.random() * ps))
        f1=fitness[index1,0:2]
        f2=fitness[index2,0:2]
        if (NDS(f1, f2) == 1):
            P_pool[i,:]=p_chrom[index1,:]
            F_pool[i,:]=f_chrom[index1,:]
        elif(NDS(f1, f2) == 2):
            P_pool[i, :] = p_chrom[index2, :]
            F_pool[i, :] = f_chrom[index2, :]
        else:
            if random.random() <= 0.5:
                P_pool[i, :] = p_chrom[index1, :]
                F_pool[i, :] = f_chrom[index1, :]
            else:
                P_pool[i, :] = p_chrom[index2, :]
                F_pool[i, :] = f_chrom[index2, :]
    return P_pool,F_pool
def TSelection2(p_chrom,fitness,ps,N):
    #mating selection pool
    pool_size=ps
    P_pool = [[0 for _ in range(N)] for _ in range(ps)]#np.zeros(shape=(ps, N), dtype=int)
    # compeitor number
    tour=2
    # 锦标赛选择 tournament selection
    for i in range(pool_size):
        index1=int(math.floor(random.random()*ps))
        index2 = int(math.floor(random.random() * ps))
        while index1==index2:
            index2 = int(math.floor(random.random() * ps))
        f1=fitness[index1][0:2]
        f2=fitness[index2][0:2]
        if (NDS(f1, f2) == 1):
            P_pool[i]=p_chrom[index1]
        elif(NDS(f1, f2) == 2):
            P_pool[i] = p_chrom[index2]
        else:
            if random.random() <= 0.5:
                P_pool[i] = p_chrom[index1]
            else:
                P_pool[i] = p_chrom[index2]
    return P_pool
def TSelection3(p_chrom_job, p_chrom_mac, p_chrom_agv, fitness, ps, N):
    #mating selection pool
    pool_size=ps
    job_pool = [[0 for _ in range(N)] for _ in range(ps)]#np.zeros(shape=(ps, N), dtype=int)
    mac_pool = [[0 for _ in range(N)] for _ in range(ps)]#np.zeros(shape=(ps, N), dtype=int)
    agv_pool = [[0 for _ in range(N)] for _ in range(ps)]#np.zeros(shape=(ps, N), dtype=int)
    # compeitor number
    tour=2
    # 锦标赛选择 tournament selection
    for i in range(pool_size):
        index1=int(math.floor(random.random()*ps))
        index2 = int(math.floor(random.random() * ps))
        while index1==index2:
            index2 = int(math.floor(random.random() * ps))
        f1=fitness[index1][0:2]
        f2=fitness[index2][0:2]
        if (NDS(f1, f2) == 1):
            job_pool[i]=p_chrom_job[index1]
            mac_pool[i] = p_chrom_mac[index1]
            agv_pool[i] = p_chrom_agv[index1]
        elif(NDS(f1, f2) == 2):
            job_pool[i] = p_chrom_job[index2]
            mac_pool[i] = p_chrom_mac[index2]
            agv_pool[i] = p_chrom_agv[index2]
        else:
            if random.random() <= 0.5:
                job_pool[i] = p_chrom_job[index1]
                mac_pool[i] = p_chrom_mac[index1]
                agv_pool[i] = p_chrom_agv[index1]
            else:
                job_pool[i] = p_chrom_job[index2]
                mac_pool[i] = p_chrom_mac[index2]
                agv_pool[i] = p_chrom_agv[index2]
    return job_pool, mac_pool, agv_pool
def POX(P1,P2,N):
    #inital offerspring
    NP1=P1
    NP2=P2
    #index of each operation in P1 and P2
    ci1 = np.zeros(N, dtype=int)
    ci2 = np.zeros(N, dtype=int)
    # store some jobs in J1
    temp=[random.random() for _ in range(N) ]
    temp=mylistRound(temp)
    J1=find_all_index(temp,1)#find the index where value equal to 1
    for j in range(N):
        if Ismemeber(P1[j], J1)==1: #if is in job set J
            ci1[j] = P1[j]+1
        if Ismemeber(P2[j], J1)==0: #if is not in job set J
            ci2[j] = P2[j]+1
    index_1_1 = find_all_index(ci1,0) # find the empty positions in ci1
    index_1_2 = find_all_index_not(ci2,0) # find the positions in ci2 which is not zero

    index_2_1 = find_all_index(ci2,0)
    index_2_2 = find_all_index_not(ci1,0)
    l1=len(index_1_1);l2=len(index_2_1)
    for j in range(l1):
        ci1[index_1_1[j]] = NP2[index_1_2[j]]
    for j in range(l2):
        ci2[index_2_1[j]] = NP1[index_2_2[j]]
    l1 = len(index_2_2);l2 = len(index_1_2)
    for j in range(l1):
        ci1[index_2_2[j]] = ci1[index_2_2[j]]-1
    for j in range(l2):
        ci2[index_1_2[j]] = ci2[index_1_2[j]] - 1
    NP1=ci1
    NP2 =ci2
    return NP1,NP2
def POX2(P1,P2,N):
    #inital offerspring
    NP1=P1; NP2=P2
    #index of each operation in P1 and P2
    ci1 = [0 for _ in range(N)]
    ci2 = [0 for _ in range(N)]#np.zeros(N, dtype=int)
    # store some jobs in J1
    temp=[random.random() for _ in range(N) ]
    temp=mylistRound(temp)
    J1=find_all_index(temp,1)#find the index where value equal to 1
    for j in range(N):
        if Ismemeber(P1[j], J1)==1: #if is in job set J
            ci1[j] = P1[j]+1 # 该部分索引元素不变
        if Ismemeber(P2[j], J1)==0: #if is not in job set J
            ci2[j] = P2[j]+1 # 该部分索引元素不变
    index_ci1_0 = find_all_index(ci1,0) # find the empty positions in ci1
    index_ci2_fei0 = find_all_index_not(ci2,0) # find the positions in ci2 which is not zero

    index_ci1_fei0 = find_all_index_not(ci1,0)
    index_ci2_0 = find_all_index(ci2,0)
    l1=len(index_ci1_0);l2=len(index_ci2_0)
    for j in range(l1):
        ci1[index_ci1_0[j]] = NP2[index_ci2_fei0[j]]
    for j in range(l2):
        ci2[index_ci2_0[j]] = NP1[index_ci1_fei0[j]]
    l1 = len(index_ci1_fei0);l2 = len(index_ci2_fei0)
    for j in range(l1):
        ci1[index_ci1_fei0[j]] = ci1[index_ci1_fei0[j]]-1
    for j in range(l2):
        ci2[index_ci2_fei0[j]] = ci2[index_ci2_fei0[j]] - 1
    NP1 = ci1; NP2 = ci2
    return NP1,NP2
def rePOX2(P1,P2,N,G1_job):
    #inital offerspring
    NP1=P1; NP2=P2
    #index of each operation in P1 and P2
    ci1 = [0 for _ in range(N)]
    ci2 = [0 for _ in range(N)]#np.zeros(N, dtype=int)
    # store some jobs in J1
    temp=[random.random() for _ in range(N) ]
    temp=mylistRound(temp)
    for i in range(len(G1_job)):
        temp[i] = 1
    J1=find_all_index(temp,1)#find the index where value equal to 1
    for j in range(N):
        if Ismemeber(P1[j], J1)==1: #if is in job set J
            ci1[j] = P1[j]+1 # 该部分索引元素不变
        if Ismemeber(P2[j], J1)==0: #if is not in job set J
            ci2[j] = P2[j]+1 # 该部分索引元素不变
    index_ci1_0 = find_all_index(ci1,0) # find the empty positions in ci1
    index_ci2_fei0 = find_all_index_not(ci2,0) # find the positions in ci2 which is not zero

    index_ci1_fei0 = find_all_index_not(ci1,0)
    index_ci2_0 = find_all_index(ci2,0)
    l1=len(index_ci1_0);l2=len(index_ci2_0)
    for j in range(l1):
        ci1[index_ci1_0[j]] = NP2[index_ci2_fei0[j]]
    for j in range(l2):
        ci2[index_ci2_0[j]] = NP1[index_ci1_fei0[j]]
    l1 = len(index_ci1_fei0);l2 = len(index_ci2_fei0)
    for j in range(l1):
        ci1[index_ci1_fei0[j]] = ci1[index_ci1_fei0[j]]-1
    for j in range(l2):
        ci2[index_ci2_fei0[j]] = ci2[index_ci2_fei0[j]] - 1
    NP1 = ci1; NP2 = ci2
    return NP1,NP2
def POX_agv(F1,F2,N):
    nf1 = copy.copy(F1);nf2 = copy.copy(F2);
    #s = [random.random() for _ in range(N)]
    #s = mylistRound(s)
    for i in range(N):
        s=round(random.random()*1)
        if (s == 1):
            temp = copy.copy(nf1[i])
            nf1[i] = copy.copy(nf2[i])
            nf2[i] = copy.copy(temp)
    return nf1, nf2
def PMX(P1,P2,N):
    #partially matching crossover operator (PMX) 1985
    #inital offerspring
    pos1 = math.floor(N/2-N/4)
    pos2 = math.floor(N-pos1)
    pos2=int(pos2)
    L = min(pos1, pos2)
    U = max(pos1, pos2)
    np1 = np.zeros(N,dtype=int)
    np2 = np.zeros(N,dtype=int)
    part1 = P1[L:U]

    part2 = P2[L:U]
    np1 = copy.copy(P1)
    np2 = copy.copy(P2)
    np1[L:U]=copy.copy(part2)
    try:
        np2[L:U]=copy.copy(part1)
    except:
        print('something wrong')
    tot=sum(i for i in range(N))
    for j in range(N):
        if j<L or j>U-1:
            x=find_all_index(part2,np1[j])
            if len(x)!=0:
                np1[j]=part1[x]
                x = find_all_index(part2, np1[j])
                while len(x)!=0:
                    np1[j] = part1[x]
                    x = find_all_index(part2, np1[j])

            x2 = find_all_index(part1, np2[j])
            if len(x2) != 0:
                np2[j] = part2[x2]
                x2 = find_all_index(part1, np2[j])
                while len(x2) != 0:
                    np2[j] = part2[x2]
                    x2 = find_all_index(part1, np2[j])
    return np1,np2

def UX_F(F1,F2,N,F):
    nf1 = copy.copy(F1);nf2 = copy.copy(F2);
    #s = [random.random() for _ in range(N)]
    #s = mylistRound(s)
    for i in range(N):
        s=round(random.random()*1)
        if (s == 1):
            temp = copy.copy(nf1[i]);
            nf1[i] = copy.copy(nf2[i]);
            nf2[i] = copy.copy(temp);
    f_num = np.zeros(F,dtype=int);
    for f in range(F):
        f_num[f]= len(find_all_index(nf1, f));
    for f in range(F):
        if f_num[f]==0:
            FC = np.zeros(N, dtype=int)
            # generate operation sequence randomly
            for i in range(N):
                FC[i] = i % F
            tmp2 = copy.copy(FC)
            random.shuffle(tmp2)
            nf1 = copy.copy(tmp2)

    f_num = np.zeros(F, dtype=int);
    for f in range(F):
        f_num[f] = len(find_all_index(nf2, f));
    for f in range(F):
        if f_num[f] == 0:
            FC = np.zeros(N, dtype=int)
            # generate operation sequence randomly
            for i in range(N):
                FC[i] = i % F
            tmp2 = copy.copy(FC)
            random.shuffle(tmp2)
            nf2= copy.copy(tmp2)
    return nf1,nf2

def mutation_p(p_chrom,N):
    #swap for operation sequence as mutation operator
    SH=N
    p1 = math.floor(random.random() * N)
    p2 = math.floor(random.random() * N)
    while p1==p2:
        p2 = math.floor(random.random() * N)
    t = copy.copy(p_chrom[p1])
    p_chrom[p1] = copy.copy(p_chrom[p2])
    p_chrom[p2] = copy.copy(t)
    return p_chrom
def remutation_p(p_chrom,N,G1_job):
    #swap for operation sequence as mutation operator
    p1 = random.randint(len(G1_job), N-1) # math.floor(random.random() * N)
    p2 = random.randint(len(G1_job), N-1)
    while p1==p2:
        p2 = random.randint(len(G1_job), N-1)
    t = copy.copy(p_chrom[p1])
    p_chrom[p1] = copy.copy(p_chrom[p2])
    p_chrom[p2] = copy.copy(t)
    return p_chrom

def mutation_f(f_chrom,N,F):
    #swap for operation sequence as mutation operator
    pos1=int(np.floor(random.random()*N))
    cf=f_chrom[pos1]
    f=math.floor(random.random()*F)
    while cf==f:
        f=np.floor(random.random()*F)
        f=int(f)
        f_chrom[pos1]=f
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
            f_chrom= copy.copy(tmp2)

    return f_chrom
def mutation_m(m_chrom, J_num, State, M_j):
    #swap for operation sequence as mutation operator
    pos1=int(np.floor(random.random() * J_num * State))
    cf=m_chrom[pos1]
    f=math.floor(random.random() * M_j[pos1 // J_num + 1])
    while cf==f:
        f=np.floor(random.random() * M_j[pos1 // J_num + 1])
        f=int(f)
        m_chrom[pos1]=f
    # 不能让机器一个都不加工
    for s in range(State):
        FF=M_j[s+1]
        f_num = np.zeros(FF, dtype=int);
        for f in range(FF):
            f_num[f] = len(find_all_index(m_chrom[J_num * s:J_num * (s + 1)], f));
        for f in range(FF):
            if f_num[f] == 0:
                FC = np.zeros(J_num, dtype=int)
                # generate operation sequence randomly
                for i in range(J_num):
                    FC[i] = i % FF
                tmp2 = FC
                random.shuffle(tmp2)
                m_chrom[J_num * s:J_num * (s + 1)]= copy.copy(tmp2)
    return m_chrom
def remutation_m(m_chrom, J_num, State, M_j, G1_mac):
    #swap for operation sequence as mutation operator
    G1_mac_s1, G1_mac_s2, G1_mac_s3 = split_by_occurrence(G1_mac)
    G1 = [G1_mac_s1, G1_mac_s2, G1_mac_s3]; G1_index = []
    for i in range(len(G1)):
        for j in range(len(G1[i])):
            G1_index.append(J_num * i + j)
    while True:
        pos = int(np.floor(random.random() * J_num * State))
        if pos not in G1_index:
            pos1 = pos
            break
    cf=m_chrom[pos1]
    f=math.floor(random.random() * M_j[pos1 // J_num + 1])
    while cf==f:
        f=np.floor(random.random() * M_j[pos1 // J_num + 1])
        f=int(f)
        m_chrom[pos1]=f
    # 不能让机器一个都不加工
    for s in range(State):
        FF=M_j[s+1]
        f_num = np.zeros(FF, dtype=int);
        for f in range(FF):
            f_num[f] = len(find_all_index(m_chrom[J_num * s:J_num * (s + 1)], f));
        for f in range(FF):
            if f_num[f] == 0:
                FC = np.zeros(J_num, dtype=int)
                # generate operation sequence randomly
                for i in range(J_num):
                    FC[i] = i % FF
                tmp2 = FC
                random.shuffle(tmp2)
                m_chrom[J_num * s:J_num * (s + 1)]= copy.copy(tmp2)
    return m_chrom
def mutation_a(a_chrom, N, v):
    #swap for operation sequence as mutation operator
    pos1=int(np.floor(random.random() * N))
    cf=a_chrom[pos1]
    f=random.randint(1, v)
    while cf==f:
        f=random.randint(1, v)
        a_chrom[pos1]=f
    # 不能让AGV一个都不运输
    f_num = np.zeros(v, dtype=int);
    for f in range(v):
        f_num[f] = len(find_all_index(a_chrom, f+1));
    for f in range(v):
        if f_num[f] == 0:
            FC = np.zeros(N, dtype=int)
            # generate operation sequence randomly
            for i in range(N):
                FC[i] = i % v + 1
            tmp2 = FC
            random.shuffle(tmp2)
            a_chrom= copy.copy(tmp2)
    return a_chrom
def remutation_a(a_chrom, N, v, G1_agv):
    #swap for operation sequence as mutation operator
    pos1=random.randint(len(G1_agv), N - 1) # int(np.floor(random.random() * J_num))
    cf=a_chrom[pos1]
    f=random.randint(1, v)
    while cf==f:
        f=random.randint(1, v)
        a_chrom[pos1]=f
    # 不能让AGV一个都不运输
    f_num = np.zeros(v, dtype=int);
    for f in range(v):
        f_num[f] = len(find_all_index(a_chrom, f+1));
    for f in range(v):
        if f_num[f] == 0:
            FC = np.zeros(N, dtype=int)
            # generate operation sequence randomly
            for i in range(N):
                FC[i] = i % v + 1
            tmp2 = FC
            random.shuffle(tmp2)
            a_chrom= copy.copy(tmp2)
    return a_chrom

def NSGA2(p_chrom,f_chrom,fitness,Pc,Pm,ps,N,time,F,TS,NS, JP,JDD):
    Pool_P, Pool_F = TSelection(p_chrom, f_chrom, fitness,ps,N);
    CP = [];CF = [];CFit = []
    for i in range(ps):
        if random.random() < Pc:
            index = math.floor(random.random() * ps)
            P1 = Pool_P[index, :]
            F1 = Pool_F[index, :]
            [np1, np2] = PMX(p_chrom[i, :], P1, N)
            [nf1, nf2] = UX_F(f_chrom[i, :], F1, N,F)
            if random.random() < Pm:
                np1 = mutation_p(np1,N)
                nf1 = mutation_f(nf1,N,F)
            if random.random() < Pm:
                np2 = mutation_p(np2, N)
                nf2 = mutation_f(nf2, N, F)
        f1=np.zeros(3);f2=np.zeros(3);
        f1[0],f1[1],f1[2] = FitDHHFSP(np1, nf1,N,time,F,TS,NS, JP, JDD)
        f2[0],f2[1],f2[2] = FitDHHFSP(np2, nf2,N,time,F,TS,NS, JP, JDD)
        if len(CP) == 0:
            CP.append(np1);CFit.append(f1);CF.append(nf1)
            CP = np.vstack((CP, np2));CFit = np.vstack((CFit, f2));CF=np.vstack((CF, nf2))
        else:
            CP = np.vstack((CP, np1));CP = np.vstack((CP, np2));
            CF = np.vstack((CF, nf1));CF = np.vstack((CF, nf2));
            CFit = np.vstack((CFit, f1));CFit = np.vstack((CFit, f2))
    QP = np.vstack((p_chrom, CP))
    QF = np.vstack((f_chrom, CF))
    QFit = np.vstack((fitness, CFit))
    QP,QF, QFit = DeleteReapt(QP,QF,QFit,ps)
    TopRank = FastNDS(QFit, ps)
    p_chrom = QP[TopRank, :];
    f_chrom = QF[TopRank, :];
    fitness = QFit[TopRank, :]

    return p_chrom,f_chrom,fitness

def NSGA2POX(p_chrom,f_chrom,fitness,Pc,Pm,ps,N,time,F,TS,NS, JP,JDD):
    Pool_P, Pool_F = TSelection(p_chrom, f_chrom, fitness,ps,N);
    CP = [];CF = [];CFit = []
    for i in range(ps):
        if random.random() < Pc:
            index = math.floor(random.random() * ps)
            P1 = Pool_P[index, :]
            F1 = Pool_F[index, :]
            [np1, np2] = POX(p_chrom[i, :], P1, N)
            [nf1, nf2] = UX_F(f_chrom[i, :], F1, N,F)
            if random.random() < Pm:
                np1 = mutation_p(np1,N)
                nf1 = mutation_f(nf1,N,F)
            if random.random() < Pm:
                np2 = mutation_p(np2, N)
                nf2 = mutation_f(nf2, N, F)
        f1=np.zeros(3);f2=np.zeros(3);
        f1[0],f1[1],f1[2] = FitDHHFSP(np1, nf1,N,time,F,TS,NS, JP, JDD)
        f2[0],f2[1],f2[2] = FitDHHFSP(np2, nf2,N,time,F,TS,NS, JP, JDD)
        if len(CP) == 0:
            CP.append(np1);CFit.append(f1);CF.append(nf1)
            CP = np.vstack((CP, np2));CFit = np.vstack((CFit, f2));CF=np.vstack((CF, nf2))
        else:
            CP = np.vstack((CP, np1));CP = np.vstack((CP, np2));
            CF = np.vstack((CF, nf1));CF = np.vstack((CF, nf2));
            CFit = np.vstack((CFit, f1));CFit = np.vstack((CFit, f2))
    QP = np.vstack((p_chrom, CP))
    QF = np.vstack((f_chrom, CF))
    QFit = np.vstack((fitness, CFit))
    QP,QF, QFit = DeleteReapt(QP,QF,QFit,ps)
    TopRank = FastNDS(QFit, ps)
    p_chrom = QP[TopRank, :];
    f_chrom = QF[TopRank, :];
    fitness = QFit[TopRank, :]

    return p_chrom,f_chrom,fitness

def NSGA2POXES(p_chrom,f_chrom,fitness,Pc,Pm,ps,N,time,F,TS,NS, JP,JDD):
    Pool_P, Pool_F = TSelection(p_chrom, f_chrom, fitness,ps,N);
    CP = [];CF = [];CFit = []
    for i in range(ps):
        index = math.floor(random.random() * ps)
        P1 = Pool_P[index, :]
        F1 = Pool_F[index, :]
        np1=copy.copy(p_chrom[i, :]);np2=copy.copy(P1);
        nf1 = copy.copy(f_chrom[i, :]);nf2 = copy.copy(F1);
        if random.random() < Pc:
            [np1, np2] = POX(p_chrom[i, :], P1, N)
            [nf1, nf2] = UX_F(f_chrom[i, :], F1, N,F)
            if random.random() < Pm:
                np1 = mutation_p(np1,N)
                nf1 = mutation_f(nf1,N,F)
            if random.random() < Pm:
                np2 = mutation_p(np2, N)
                nf2 = mutation_f(nf2, N, F)
        f1=np.zeros(3);f2=np.zeros(3);
        f1[0],f1[1],f1[2] = EnergySave_DHHFSP(np1, nf1,N,time,F,TS,NS, JP, JDD)
        f2[0],f2[1],f2[2] = EnergySave_DHHFSP(np2, nf2,N,time,F,TS,NS, JP, JDD)
        if len(CP) == 0:
            CP.append(np1);CFit.append(f1);CF.append(nf1)
            CP = np.vstack((CP, np2));CFit = np.vstack((CFit, f2));CF=np.vstack((CF, nf2))
        else:
            CP = np.vstack((CP, np1));CP = np.vstack((CP, np2));
            CF = np.vstack((CF, nf1));CF = np.vstack((CF, nf2));
            CFit = np.vstack((CFit, f1));CFit = np.vstack((CFit, f2))
    QP = np.vstack((p_chrom, CP))
    QF = np.vstack((f_chrom, CF))
    QFit = np.vstack((fitness, CFit))
    QP,QF, QFit = DeleteReapt(QP,QF,QFit,ps)
    TopRank = FastNDS(QFit, ps)
    p_chrom = QP[TopRank, :]
    f_chrom = QF[TopRank, :]
    fitness = QFit[TopRank, :]

    return p_chrom,f_chrom,fitness
def NSGA2POXES2(p_chrom,fitness,Pc,Pm,ps,N,time,F,NS):
    Pool_P = TSelection2(p_chrom,fitness,ps,N)
    CP = [];CFit = []
    for i in range(ps):
        index = math.floor(random.random() * ps)
        P1 = Pool_P[index]
        np1=copy.copy(p_chrom[i])
        np2=copy.copy(P1)
        if random.random() < Pc:
            [np1, np2] = POX2(p_chrom[i], P1, N*F)
            if random.random() < Pm:
                np1 = mutation_p(np1,N*F)
            if random.random() < Pm:
                np2 = mutation_p(np2, N*F)
        f1=list(np.zeros(3))
        f2=list(np.zeros(3))
        s = Sch()
        s.Decode(np1)
        f1[0],f1[1],f1[2] = s.fitness,s.fitness1,s.fitness2
        s2 = Sch()
        s2.Decode(np2)
        f2[0], f1[1], f2[2] = s2.fitness, s2.fitness1, s2.fitness2
        if len(CP) == 0:
            CP.append(np1);CFit.append(f1)
            CP = np.vstack((CP, np2));CFit = np.vstack((CFit, f2))
        else:
            CP = np.vstack((CP, np1));CP = np.vstack((CP, np2))
            CFit = np.vstack((CFit, f1));CFit = np.vstack((CFit, f2))
    QP = list(np.vstack((p_chrom, CP)))
    QFit = list(np.vstack((fitness, CFit)))
    QP,QFit = DeleteReapt2(QP,QFit,ps)
    TopRank = FastNDS2(QFit, ps)
    p_chrom = [QP[index] for index in TopRank]
    fitness = [QFit[index] for index in TopRank]

    return p_chrom,fitness
def NSGA2POXES3(p_chrom,fitness,Pc,Pm,ps,N,time,F,NS,n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id):
    Pool_P = TSelection2(p_chrom,fitness,ps,N)
    CP = [];CFit = []
    for i in range(ps):
        index = math.floor(random.random() * ps)
        P1 = Pool_P[index]
        np1=copy.copy(p_chrom[i])
        np2=copy.copy(P1)
        if random.random() < Pc:
            [np1, np2] = POX2(p_chrom[i], P1, N*F)
            if random.random() < Pm:
                np1 = mutation_p(np1,N*F)
            if random.random() < Pm:
                np2 = mutation_p(np2, N*F)
        f1=list(np.zeros(3))
        f2=list(np.zeros(3))
        s = Situation(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id)
        s.Stage_Decode(np1)
        f1[0],f1[1],f1[2] = s.fitness1,s.fitness2,s.fitness3
        s2 = Situation(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id)
        s2.Stage_Decode(np2)
        f2[0], f1[1], f2[2] = s2.fitness1, s2.fitness2, s2.fitness3
        if len(CP) == 0:
            CP.append(np1);CFit.append(f1)
            CP = np.vstack((CP, np2));CFit = np.vstack((CFit, f2))
        else:
            CP = np.vstack((CP, np1));CP = np.vstack((CP, np2))
            CFit = np.vstack((CFit, f1));CFit = np.vstack((CFit, f2))
    QP = list(np.vstack((p_chrom, CP)))
    QFit = list(np.vstack((fitness, CFit)))
    QP,QFit = DeleteReapt2(QP,QFit,ps)
    TopRank = FastNDS2(QFit, ps)
    p_chrom = [QP[index] for index in TopRank]
    fitness = [QFit[index] for index in TopRank]

    return p_chrom,fitness
def NSGA2POXES4(p_chrom_job, p_chrom_mac, p_chrom_agv, fitness, Pc, Pm, ps, NS, n, J_num, State, M_j, PT, D_pq, JNN, v, V_r, JN_ip, ST_i, MNN, suanli, xunlian_id):
    Pool_job, Pool_mac, Pool_agv = TSelection3(p_chrom_job, p_chrom_mac, p_chrom_agv, fitness, ps, J_num * State)
    CP,CM,CA = [],[],[];CFit = []
    for i in range(ps):
        index = math.floor(random.random() * ps)
        P1, M1, A1 = Pool_job[index], Pool_mac[index], Pool_agv[index]
        np1=copy.copy(p_chrom_job[i]);np2=copy.copy(P1)
        nm1=copy.copy(p_chrom_mac[i]);nm2=copy.copy(M1)
        na1=copy.copy(p_chrom_agv[i]);na2=copy.copy(A1)
        if random.random() < Pc:
            [np1, np2] = POX2(p_chrom_job[i], P1, J_num * State)
            [nm1, nm2] = POX_agv(p_chrom_mac[i], M1, J_num * State)
            [na1, na2] = POX_agv(p_chrom_agv[i], A1, J_num * State)
            # [na1, na2] = POX2(p_chrom_agv[i], A1, J_num * State)
            if random.random() < Pm:
                np1 = mutation_p(np1, J_num * State)
                nm1 = mutation_m(nm1, J_num, State, M_j)
                na1 = mutation_a(na1, J_num * State, v)
            if random.random() < Pm:
                np2 = mutation_p(np2, J_num * State)
                nm2 = mutation_m(nm2, J_num, State, M_j)
                na2 = mutation_a(na2, J_num * State, v)
        f1=list(np.zeros(3))
        f2=list(np.zeros(3))
        s = Situation(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id)
        s.Stage_Decode3(np1,nm1,na1)
        f1[0],f1[1],f1[2] = s.fitness1,s.fitness2,s.fitness3
        s2 = Situation(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id)
        s2.Stage_Decode3(np2,nm2,na2)
        f2[0], f1[1], f2[2] = s2.fitness1, s2.fitness2, s2.fitness3
        if len(CP) == 0:
            CP.append(np1);CM.append(nm1);CA.append(na1);CFit.append(f1)
            CP = np.vstack((CP, np2));CM = np.vstack((CM, nm2));CA = np.vstack((CA, na2));CFit = np.vstack((CFit, f2))
        else:
            CP = np.vstack((CP, np1));CM = np.vstack((CM, nm1));CA = np.vstack((CA, na1))
            CP = np.vstack((CP, np2));CM = np.vstack((CM, nm2));CA = np.vstack((CA, na2))
            CFit = np.vstack((CFit, f1));CFit = np.vstack((CFit, f2))
    QP = list(np.vstack((p_chrom_job, CP)))
    QM = list(np.vstack((p_chrom_mac, CM)))
    QA = list(np.vstack((p_chrom_agv, CA)))
    QFit = list(np.vstack((fitness, CFit)))
    QP,QM,QA,QFit = DeleteReapt3(QP,QM,QA,QFit,ps)
    TopRank = FastNDS2(QFit, ps)
    p_chrom_job = [QP[index] for index in TopRank]
    p_chrom_mac = [QM[index] for index in TopRank]
    p_chrom_agv = [QA[index] for index in TopRank]
    fitness = [QFit[index] for index in TopRank]

    return p_chrom_job, p_chrom_mac, p_chrom_agv, fitness
def reNSGA2POXES4(p_chrom_job, p_chrom_mac, p_chrom_agv, fitness, Pc, Pm, ps, NS, n, J_num, State, M_j, PT, D_pq, JNN, v, V_r, JN_ip, ST_i, MNN, suanli, xunlian_id, G1_job, G1_mac, G1_agv):
    Pool_job, Pool_mac, Pool_agv = TSelection3(p_chrom_job, p_chrom_mac, p_chrom_agv, fitness, ps, J_num * State)
    CP,CM,CA = [],[],[];CFit = []
    for i in range(ps):
        index = math.floor(random.random() * ps)
        P1, M1, A1 = Pool_job[index], Pool_mac[index], Pool_agv[index]
        np1=copy.copy(p_chrom_job[i]);np2=copy.copy(P1)
        nm1=copy.copy(p_chrom_mac[i]);nm2=copy.copy(M1)
        na1=copy.copy(p_chrom_agv[i]);na2=copy.copy(A1)
        if random.random() < Pc:
            [np1, np2] = rePOX2(p_chrom_job[i], P1, J_num * State, G1_job)
            [nm1, nm2] = POX_agv(p_chrom_mac[i], M1, J_num * State)
            [na1, na2] = POX_agv(p_chrom_agv[i], A1, J_num * State)
            # [na1, na2] = POX2(p_chrom_agv[i], A1, J_num * State)
            if random.random() < Pm:
                np1 = remutation_p(np1, J_num * State, G1_job)
                nm1 = remutation_m(nm1, J_num, State, M_j, G1_mac)
                na1 = remutation_a(na1, J_num * State, v, G1_agv)
            if random.random() < Pm:
                np2 = remutation_p(np2, J_num * State, G1_job)
                nm2 = remutation_m(nm2, J_num, State, M_j, G1_mac)
                na2 = remutation_a(na2, J_num * State, v, G1_agv)
        f1=list(np.zeros(3))
        f2=list(np.zeros(3))
        s = Situation(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id)
        s.Stage_Decode3(np1,nm1,na1)
        f1[0],f1[1],f1[2] = s.fitness1,s.fitness2,s.fitness3
        s2 = Situation(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id)
        s2.Stage_Decode3(np2,nm2,na2)
        f2[0], f1[1], f2[2] = s2.fitness1, s2.fitness2, s2.fitness3
        if len(CP) == 0:
            CP.append(np1);CM.append(nm1);CA.append(na1);CFit.append(f1)
            CP = np.vstack((CP, np2));CM = np.vstack((CM, nm2));CA = np.vstack((CA, na2));CFit = np.vstack((CFit, f2))
        else:
            CP = np.vstack((CP, np1));CM = np.vstack((CM, nm1));CA = np.vstack((CA, na1))
            CP = np.vstack((CP, np2));CM = np.vstack((CM, nm2));CA = np.vstack((CA, na2))
            CFit = np.vstack((CFit, f1));CFit = np.vstack((CFit, f2))
    QP = list(np.vstack((p_chrom_job, CP)))
    QM = list(np.vstack((p_chrom_mac, CM)))
    QA = list(np.vstack((p_chrom_agv, CA)))
    QFit = list(np.vstack((fitness, CFit)))
    QP,QM,QA,QFit = DeleteReapt3(QP,QM,QA,QFit,ps)
    TopRank = FastNDS2(QFit, ps)
    p_chrom_job = [QP[index] for index in TopRank]
    p_chrom_mac = [QM[index] for index in TopRank]
    p_chrom_agv = [QA[index] for index in TopRank]
    fitness = [QFit[index] for index in TopRank]

    return p_chrom_job, p_chrom_mac, p_chrom_agv, fitness
def TiHuanQianDuan(G1, p_chrom):
    for j in range(len(G1)):
        if p_chrom[j] != G1[j]:
            for k in range(j+1, len(p_chrom)):
                if p_chrom[k] == G1[j]:
                    p_chrom[k] = p_chrom[j]
                    break
            p_chrom[j] = G1[j]
    return p_chrom
def split_by_occurrence(lst):
    occurrences = defaultdict(list)
    for elem in lst:
        occurrences[elem].append(elem)
    a, b, c = [], [], []
    for elem, occurrences_list in occurrences.items():
        if len(occurrences_list) >= 1:
            a.append(occurrences_list[0])
        if len(occurrences_list) >= 2:
            b.append(occurrences_list[1])
        if len(occurrences_list) >= 3:
            c.append(occurrences_list[2])
    return a, b, c
def NSGA2PMXES(p_chrom,f_chrom,fitness,Pc,Pm,ps,N,time,F,TS,NS, JP,JDD):
    Pool_P, Pool_F = TSelection(p_chrom, f_chrom, fitness,ps,N);
    CP = [];CF = [];CFit = []
    for i in range(ps):
        if random.random() < Pc:
            index = math.floor(random.random() * ps)
            P1 = Pool_P[index, :]
            F1 = Pool_F[index, :]
            [np1, np2] = PMX(p_chrom[i, :], P1, N)
            [nf1, nf2] = UX_F(f_chrom[i, :], F1, N,F)
            if random.random() < Pm:
                np1 = mutation_p(np1,N)
                nf1 = mutation_f(nf1,N,F)
            if random.random() < Pm:
                np2 = mutation_p(np2, N)
                nf2 = mutation_f(nf2, N, F)
        f1=np.zeros(3);f2=np.zeros(3);
        f1[0],f1[1],f1[2] = EnergySave_DHHFSP(np1, nf1,N,time,F,TS,NS, JP, JDD)
        f2[0],f2[1],f2[2] = EnergySave_DHHFSP(np2, nf2,N,time,F,TS,NS, JP, JDD)
        if len(CP) == 0:
            CP.append(np1);CFit.append(f1);CF.append(nf1)
            CP = np.vstack((CP, np2));CFit = np.vstack((CFit, f2));CF=np.vstack((CF, nf2))
        else:
            CP = np.vstack((CP, np1));CP = np.vstack((CP, np2));
            CF = np.vstack((CF, nf1));CF = np.vstack((CF, nf2));
            CFit = np.vstack((CFit, f1));CFit = np.vstack((CFit, f2))
    QP = np.vstack((p_chrom, CP))
    QF = np.vstack((f_chrom, CF))
    QFit = np.vstack((fitness, CFit))
    QP,QF, QFit = DeleteReapt(QP,QF,QFit,ps)
    TopRank = FastNDS(QFit, ps)
    p_chrom = QP[TopRank, :];
    f_chrom = QF[TopRank, :];
    fitness = QFit[TopRank, :]

    return p_chrom,f_chrom,fitness


def NSGA2MOX(p_chrom,f_chrom,fitness,Pc,Pm,ps,N,time,F,TS,NS, JP,JDD):
    Pool_P, Pool_F = TSelection(p_chrom, f_chrom, fitness,ps,N);
    CP = [];CF = [];CFit = []
    for i in range(ps):
        if random.random() < Pc:
            index = math.floor(random.random() * ps)
            P1 = Pool_P[index, :]
            F1 = Pool_F[index, :]
            if random.random()<0.5:
                [np1, np2] = POX(p_chrom[i, :], P1, N)
            else:
                [np1, np2] = PMX(p_chrom[i, :], P1, N)
            [nf1, nf2] = UX_F(f_chrom[i, :], F1, N,F)
            if random.random() < Pm:
                np1 = mutation_p(np1,N)
                nf1 = mutation_f(nf1,N,F)
            if random.random() < Pm:
                np2 = mutation_p(np2, N)
                nf2 = mutation_f(nf2, N, F)
        f1=np.zeros(3);f2=np.zeros(3);
        f1[0],f1[1],f1[2] = FitDHHFSP(np1, nf1,N,time,F,TS,NS, JP, JDD)
        f2[0],f2[1],f2[2] = FitDHHFSP(np2, nf2,N,time,F,TS,NS, JP, JDD)
        if len(CP) == 0:
            CP.append(np1);CFit.append(f1);CF.append(nf1)
            CP = np.vstack((CP, np2));CFit = np.vstack((CFit, f2));CF=np.vstack((CF, nf2))
        else:
            CP = np.vstack((CP, np1));CP = np.vstack((CP, np2));
            CF = np.vstack((CF, nf1));CF = np.vstack((CF, nf2));
            CFit = np.vstack((CFit, f1));CFit = np.vstack((CFit, f2))
    QP = np.vstack((p_chrom, CP))
    QF = np.vstack((f_chrom, CF))
    QFit = np.vstack((fitness, CFit))
    QP,QF, QFit = DeleteReapt(QP,QF,QFit,ps)
    TopRank = FastNDS(QFit, ps)
    p_chrom = QP[TopRank, :];
    f_chrom = QF[TopRank, :];
    fitness = QFit[TopRank, :]

    return p_chrom,f_chrom,fitness

def MOEADPOX(p_chrom,f_chrom,index,T,neighbour,Pc,Pm,N,time,F,TS,NS, JP,JDD):
    nei = neighbour[index, :]
    R1 = math.floor(random.random() * T)
    R1 = nei[R1]
    R2 = math.floor(random.random() * T)
    R2 = nei[R2]

    np1 = copy.copy(p_chrom[R1, :])
    np2 = copy.copy(p_chrom[R2, :])
    nf1 = copy.copy(f_chrom[R1, :])
    nf2= copy.copy(f_chrom[R2, :])
    while R1 == R2:
        R2 = math.floor(random.random() * T)
        R2 = nei[R2]
    if random.random() < Pc:
        [np1, np2] = POX(p_chrom[R1, :], p_chrom[R2, :], N)
        [nf1, nf2] = UX_F(f_chrom[R1, :], f_chrom[R2, :], N, F)
        if random.random() < Pm:
            np1 = mutation_p(np1, N)
            nf1 = mutation_f(nf1, N, F)
        if random.random() < Pm:
            np2 = mutation_p(np2, N)
            nf2 = mutation_f(nf2, N, F)

        f1=np.zeros(3);f2=np.zeros(3);
        f1[0],f1[1],f1[2] = FitDHHFSP(np1, nf1,N,time,F,TS,NS, JP, JDD)
        f2[0],f2[1],f2[2] = FitDHHFSP(np2, nf2,N,time,F,TS,NS, JP, JDD)

    return np1,nf1,f1,np2,nf2,f2