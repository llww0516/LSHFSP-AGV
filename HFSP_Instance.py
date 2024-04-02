import math
import random
import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
import pandas as pd
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
                for k in range(1, len(LS)):#第k种工件
                    time = random.randint(1,50)
                    for jk in range(len(LS[k])):#第jk个工件
                        S0.append(time * LS[k][jk])
                Si.append(S0)
            PT.append(Si)
        else:
            PT.append(Si)
    return PT,LS,JNN
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
xunlian_id = 1 # 训练第几次
suanli = '_n10_m36'
v=3 # AGV数量
path = 'data/Suanli2/Suanli{}.pkl'.format(suanli)
with open(path, 'rb') as f:
    Suanli = pickle.load(f)
# N_i = [20, 50, 40, 80, 60, 45, 70] # 种类i的工件数量
# n = len(N_i) # 工件种类数
N_i = Suanli[1] # 种类i的工件数量
n = Suanli[0] # 工件种类数
State=3 # 阶段数
V_r=1 #AGV的运行速度
# M_j=[1, 4, 5, 3] # 各阶段机器数，0阶段机器数为1，也就是分拣区
M_j = Suanli[5] # 各阶段机器数，0阶段机器数为1，也就是分拣区
m = sum(M_j) - 1 # 机器数量
MNN = [] # 按阶段给机器分组[[0], [1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12]]
current_num = 0
for num in M_j:
    sub_list = list(range(current_num, current_num + num))
    MNN.append(sub_list)
    current_num += num
PT,JN_ip,JNN=Generate(State, N_i, M_j)#PT=[阶段][机器序号（阶段内）][批次]工件加工时间，JN_ip 作业i的第p个子批包含多少工件数[种类][批次]，JNN 按工件种类给每批工件分组[[0],[1],[2, 3, 4],[5, 6],[7, 8, 9, 10],[11, 12, 13], [14, 15, 16], [17, 18, 19, 20]]
PT = Suanli[2]
J_num=len(PT[1][0])-1
ST_i = Suanli[3]
# D_pq=[]# 两个位置p和q之间的运输距离;如果p = q,D_pq= 0
# for i in range(sum(M_j) + 1):
#     T1 = []
#     for j in range(sum(M_j) + 1):
#         if j==i or i==0:
#             T1.append(0)
#         elif j > i:
#             T1.append(random.randint(1, 30))
#         else:
#             T1.append(D_pq[j][i])
#     D_pq.append(T1)
D_pq = Suanli[4]
import pandas as pd
def Spacing_P(make_span_rule3D, Idle_time_rule3D, Agv_S_rule3D):
    dis_min = []
    for i in range(len(make_span_rule3D)):
        minmin = float('inf')
        for j in range(len(make_span_rule3D)):
            if i != j:
                if minmin > (make_span_rule3D[i]**2-make_span_rule3D[j]**2)+(Idle_time_rule3D[i]**2-Idle_time_rule3D[j]**2)+(Agv_S_rule3D[i]**2-Agv_S_rule3D[j]**2):
                    minmin = (make_span_rule3D[i] ** 2 - make_span_rule3D[j] ** 2) + (Idle_time_rule3D[i] ** 2 - Idle_time_rule3D[j] ** 2) + (Agv_S_rule3D[i] ** 2 - Agv_S_rule3D[j] ** 2)
        dis_min.append(minmin)
    K = 0
    for i in dis_min:
        K += np.square(i - sum(dis_min)/len(dis_min))
    a = np.sqrt(K / (len(dis_min)-1))
    return a
def HV(make_span_rule3D, Idle_time_rule3D, Agv_S_rule3D, HV_R):
    hypervolume = 0
    for i in range(len(make_span_rule3D)):
        # 计算每个解到参考点的立方体体积
        hypervolume += (make_span_rule3D[i]-HV_R[0]-100)*(Idle_time_rule3D[i] - HV_R[1]-100)*(Agv_S_rule3D[i] - HV_R[2]-100)
    return hypervolume
def Shuchu(O1, O2, O3, suanfa, HV_R, data_pj):
    data_pj.loc['{}'.format(suanfa), :] = [suanfa, round(Spacing_P(O1, O2, O3), 2),round(HV(O1, O2, O3, HV_R), 2)]

def mainnn(suanli, v, xunlian_id, ps, Pc, Pm, LEARNING_RATE, GAMMA, tanlan, data_pj):
    file_path = 'data_out/tiankou3/{}_v{}/result{}_{}_{}_{}_{}_{}_{}_{}_{}.xlsx'.format(suanli, v, suanli, v, xunlian_id, ps, Pc, Pm, LEARNING_RATE, GAMMA, tanlan)  # 替换为你的Excel文件路径
    data = pd.read_excel(file_path)
    make_span3D, Idle_time3D, Agv_S3D = [], [], []
    # 将每一列数据存储到列表中
    for column_name in data.columns:
        if column_name == '完工时间':
            make_span3D=list(data[column_name])
        elif column_name == '空闲时间':
            Idle_time3D=list(data[column_name])
        elif column_name == '运输距离':
            Agv_S3D=list(data[column_name])
    HV_R = []
    HV_R.append(min(make_span3D))
    HV_R.append(min(Idle_time3D))
    HV_R.append(min(Agv_S3D))
    Shuchu(make_span3D, Idle_time3D, Agv_S3D,
           't{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(suanli, v, xunlian_id, ps, Pc, Pm, LEARNING_RATE, GAMMA,
                                                     tanlan), HV_R, data_pj)

def duqu(suanli, v):
    TIANKOU = [
        [100, 0.1, 0.02, 0.0001, 0.5, 0.1],
        [100, 0.2, 0.04, 0.005, 0.6, 0.2],
        [100, 0.3, 0.06, 0.001, 0.7, 0.3],
        [100, 0.4, 0.08, 0.05, 0.8, 0.4],
        [100, 0.5, 0.1, 0.01, 0.9, 0.5],
        [125, 0.1, 0.04, 0.001, 0.8, 0.5],
        [125, 0.2, 0.06, 0.05, 0.9, 0.1],
        [125, 0.3, 0.08, 0.01, 0.5, 0.2],
        [125, 0.4, 0.1, 0.0001, 0.6, 0.3],
        [125, 0.5, 0.02, 0.005, 0.7, 0.4],
        [150, 0.1, 0.06, 0.01, 0.6, 0.4],
        [150, 0.2, 0.08, 0.0001, 0.7, 0.5],
        [150, 0.3, 0.10, 0.005, 0.8, 0.1],
        [150, 0.4, 0.02, 0.001, 0.9, 0.2],
        [150, 0.5, 0.04, 0.05, 0.5, 0.3],
        [175, 0.1, 0.08, 0.005, 0.9, 0.3],
        [175, 0.2, 0.10, 0.001, 0.5, 0.4],
        [175, 0.3, 0.02, 0.05, 0.6, 0.5],
        [175, 0.4, 0.04, 0.01, 0.7, 0.1],
        [175, 0.5, 0.06, 0.0001, 0.8, 0.2],
        [200, 0.1, 0.10, 0.05, 0.7, 0.2],
        [200, 0.2, 0.02, 0.01, 0.8, 0.3],
        [200, 0.3, 0.04, 0.0001, 0.9, 0.4],
        [200, 0.4, 0.06, 0.005, 0.5, 0.5],
        [200, 0.5, 0.08, 0.001, 0.6, 0.1],
    ]
    data_pj = pd.DataFrame({'Site': [], 'spacing': [], 'HV': []})
    data_out_pj = 'data_out/tiankou3/result{}_v{}.xlsx'.format(suanli,v)
    for ii in range(len(TIANKOU)):
        ps, Pc, Pm, LEARNING_RATE, GAMMA, tanlan = TIANKOU[ii][0], TIANKOU[ii][1], TIANKOU[ii][2], TIANKOU[ii][3], TIANKOU[ii][4], TIANKOU[ii][5]
        mainnn(suanli, v, 1, ps, Pc, Pm, LEARNING_RATE, GAMMA, tanlan, data_pj)
    # for ps in [125, 150, 175, 200]:
    #     mainnn(suanli, v, 1, ps, 0.1, 0.02, 0.005, 0.8, 0.1, data_pj)
    # for Pc in [0.3, 0.5, 0.7, 0.9]:
    #     mainnn(suanli, v, 1, 100, Pc, 0.02, 0.005, 0.8, 0.1, data_pj)
    # for Pm in [0.04, 0.06, 0.08, 0.1]:
    #     mainnn(suanli, v, 1, 100, 0.1, Pm, 0.005, 0.8, 0.1, data_pj)
    # for LEARNING_RATE in [0.0001, 0.001, 0.05, 0.01]:
    #     mainnn(suanli, v, 1, 100, 0.1, 0.02, LEARNING_RATE, 0.8, 0.1, data_pj)
    # for GAMMA in [0.5, 0.6, 0.7, 0.9]:
    #     mainnn(suanli, v, 1, 100, 0.1, 0.02, 0.005, GAMMA, 0.1, data_pj)
    # for tanlan in [0.2, 0.3, 0.4, 0.5]:
    #     mainnn(suanli, v, 1, 100, 0.1, 0.02, 0.005, 0.8, tanlan, data_pj)
    # mainnn(suanli, v, 1, 100, 0.1, 0.02, 0.005, 0.8, 0.1, data_pj)
    data_pj.to_excel(data_out_pj, sheet_name='评价指标', index=False)
# duqu('_n15_m12', 5)
# print(suanli,'_v',v)
# print(JN_ip,JNN)
# print('工件种数：', n)
# print('每种工件数量：', N_i)
# print('PT：', [[ro[1:] for ro in row] for row in PT[1:]])
# print('单个作业i成套分拣时间：', ST_i[1:])
# print('各阶段机器数：', M_j)
#测试
