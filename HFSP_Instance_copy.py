import math
import random
import numpy as np
import torch
import pickle
# from HFSP_Env_MDDQN_NSGA2_231204 import Situation
# s = Situation()
# random.seed(32)
def lot_sizeCalculation(Job):#批处理大小计算
    lot_size = [[0]]
    JNN = [[0]]
    n = 1
    for l in range(len(Job)):
        lot_size_variety = []
        JN = []
        #batch = random.randint(3, 5)
        if Job[l] % 20 > 0:
            #batch_number = Job[l] // (batch - 1)
            for i in range(Job[l] // 20):
                lot_size_variety.append(20)
                JN.append(n)
                n += 1
            lot_size_variety.append(Job[l] % 20)
            JN.append(n)
            n += 1
        else:
            #batch_number = Job[l] // batch
            for i in range(Job[l] // 20):
                lot_size_variety.append(20)
                JN.append(n)
                n += 1
        # batch_number = Job[l] // batch
        # for i in range(batch-1):
        #     lot_size_variety.append(batch_number)
        # lot_size_variety.append(Job[l] - batch_number * (batch - 1))
        lot_size.append(lot_size_variety)
        JNN.append(JN)
    return lot_size,JNN
#State:阶段，即工件有几道工序，Job:工件数组，Machine['type':list],对应各阶段的并行机数量
def Generate(State,JobArray,Machine):
    PT=[]
    LS,JNN=lot_sizeCalculation(JobArray)
    for i in range(State+1):#第i个加工阶段
        Si=[]
        if i >0:
            for j in range(Machine[i]):#第j个机器
                S0 = []
                for k in range(1,len(LS)):#第k种工件
                    time = random.randint(1,50)
                    for jk in range(len(LS[k])):#第jk个工件
                        S0.append(time * LS[k][jk])
                Si.append(S0)
            PT.append(Si)
        else:
            PT.append(Si)
    return PT,LS,JNN
def find_element_index(mat, target):  # 返回目标元素所在的行和列索引
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            if mat[i][j] == target:
                return i, j
def create_delivery_lists(delivery_dates, num_per_kind):
    delivery_lists = []
    for num in num_per_kind:
        delivery_lists.extend([delivery_dates[0]] * num)
        delivery_dates.pop(0)
    return delivery_lists
def get_row_lengths(matrix):
    row_lengths = []
    for row in matrix:
        row_lengths.append(len(row))
    return row_lengths
'''所有工件种类、阶段数、工件批次（所有种类工件一起编号）、机器序号（所有阶段机器一起编号）、AGV序号都是从 1 开始编号'''
'''所有工件种类内的批数序号p、阶段内的机器序号k都是从 0 开始编号'''
# for i in [20]:
#     n = i # 工件种类数
#     N_i = [random.randint(100, 200) for i in range(i)] # 种类i的工件数量
#     # N_i = [20, 50, 40, 80, 60, 45, 70] # 种类i的工件数量
#     # n = len(N_i) # 工件种类数
#     State=3 # 阶段数
#     v=5 # AGV数量
#     V_r=1 #AGV的运行速度
#     for M_j in [[1, 4, 5, 3],[1, 8, 10, 6],[1, 12, 15, 9]]: # 各阶段机器数，0阶段机器数为1，也就是分拣区
#         m = sum(M_j) - 1 # 机器数量
#         MNN = [] # 按阶段给机器分组[[0], [1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12]]
#         current_num = 0
#         for num in M_j:
#             sub_list = list(range(current_num, current_num + num))
#             MNN.append(sub_list)
#             current_num += num
#         PT,JN_ip,JNN=Generate(State, N_i, M_j)#PT=[阶段][机器序号（阶段内）][批次]工件加工时间，JN_ip 作业i的第p个子批包含多少工件数[种类][批次]，JNN 按工件种类给每批工件分组[[0],[1],[2, 3, 4],[5, 6],[7, 8, 9, 10],[11, 12, 13], [14, 15, 16], [17, 18, 19, 20]]
#         J_num=len(PT[1][0])-1
#         cur_site=0
#         D_pq=[]# 两个位置p和q之间的运输距离;如果p = q,D_pq= 0
#         for k in range(sum(M_j)):
#             T1 = []
#             for j in range(sum(M_j)):
#                 if j==k or k==0:
#                     T1.append(0)
#                 elif j > k:
#                     m_sta_k, m_id_k = find_element_index(MNN, k)
#                     m_sta_j, m_id_j = find_element_index(MNN, j)
#                     T1.append((((m_sta_j-m_sta_k)*60)**2+((m_id_j-m_id_k)*30)**2)**0.5)
#                 else:
#                     T1.append(D_pq[j][k])
#             D_pq.append(T1)
#         ST_i=[random.randint(1, 50) for i in range(n + 1)] # 单个作业i成套分拣时间
#         print('工件种数：', n)
#         print('每种工件数量：', N_i)
#         print('PT：', [[ro[1:] for ro in row] for row in PT[1:]])
#         print('单个作业i成套分拣时间：', ST_i[1:])
#         print('两个位置p和q之间的运输距离：', D_pq)
#         Suanli = []
#         Suanli.append(n)
#         Suanli.append(N_i)
#         Suanli.append(PT)
#         Suanli.append(ST_i)
#         Suanli.append(D_pq)
#         Suanli.append(M_j)
#         path = 'data/Suanli2/Suanli_n{}_m{}.pkl'.format(i,m)
#         with open(path, 'wb') as f:
#             pickle.dump(Suanli, f)

#测试
with open('data/Suanli2/Suanli_n5_m12.pkl', 'rb') as f:
    Suanli = pickle.load(f)
# N_i = Suanli[1] # 种类i的工件数量
# State=3 # 阶段数
# PT = Suanli[2]
# M_j = Suanli[5]  # 各阶段机器数，0阶段机器数为1，也就是分拣区
# JN_ip, JNN = lot_sizeCalculation(N_i)
# Time = []
# for i in range(State+1):#第i个加工阶段
#     Si = []
#     if i > 0:
#         for j in range(M_j[i]):  # 第j个机器
#             S0 = []
#             for k in range(1, len(JN_ip)):  # 第k种工件
#                 S0.append(int(PT[i][j][JNN[k][0]-1] / JN_ip[k][0]))
#             Si.append(S0)
#         Time.append(Si)
# PT=[]; JN_ip = [[0]]; JNN = [[0]]; n = 1
# for l in range(len(N_i)):
#     # lot_size_variety = []
#     # for i in range(N_i[l] // 10):
#     #     lot_size_variety.append(10)
#     # if N_i[l] % 10 > 0:
#     #     lot_size_variety.append(N_i[l] % 10)
#     # n = len(lot_size_variety)
#     # half_n = n // 2  # 取整，确保遍历一半的元素
#     # # 遍历数组的一半
#     # for i in range(half_n):
#     #     # 生成一个10以内的随机数
#     #     random_num = random.randint(0, 10)
#     #     # 以50%的概率决定是加还是减
#     #     add_or_subtract = random.choice([True, False])
#     #     # 如果决定加，则原位置加随机数，对称位置减随机数
#     #     if add_or_subtract:
#     #         lot_size_variety[i] += random_num
#     #         # 对称位置是 i - 1
#     #         lot_size_variety[n-i-2] -= random_num
#     #     else:
#     #         lot_size_variety[i] -= random_num
#     #         lot_size_variety[n-i-2] += random_num
#     # lot_size_variety = [x for x in lot_size_variety if x != 0]
#     # # 打乱其他元素的位置
#     # random.shuffle(lot_size_variety)
#     # JN_ip.append(lot_size_variety)
#     JN_ip.append([N_i[l]])
# n = 1
# for i in range(1,len(JN_ip)):
#     jnn = []
#     for j in range(len(JN_ip[i])):
#         jnn.append(n)
#         n = n + 1
#     JNN.append(jnn)
# for i in range(State+1):#第i个加工阶段
#     Si=[]
#     if i >0:
#         for j in range(M_j[i]):#第j个机器
#             S0 = []
#             for k in range(1,len(JN_ip)):#第k种工件
#                 time = Time[i-1][j][k-1]
#                 for jk in range(len(JN_ip[k])):#第jk个工件
#                     S0.append(time * JN_ip[k][jk])
#             Si.append(S0)
#         PT.append(Si)
#     else:
#         PT.append(Si)
# Suanli[2] = PT
# # Suanli.append(JN_ip)
# Suanli.append(JN_ip)
# Suanli.append(JNN)
# path = 'data/Suanli2/BuSuanli_n5_m12.pkl'
# with open(path, 'wb') as f:
#     pickle.dump(Suanli, f)
