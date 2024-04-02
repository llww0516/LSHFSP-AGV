# coding:utf-8
import numpy as np

def mymin(arr):
    index=[]
    l=len(arr)
    if l>1:
        minvalue=min(arr)
        for i in range(l):
            if arr[i]==minvalue:
                index.append(i)
    else:
        index.append(0)
    return index

def mylistRound(arr):
    l=len(arr)
    for i in range(l):
        arr[i]=round(arr[i])
    return arr

def find_all_index(arr, item):
    return [i for i, a in enumerate(arr) if a == item]

def find_all_index_not(arr, item):
    l=len(arr)
    flag=np.zeros(l)
    index=find_all_index(arr,item)
    flag[index]=1
    not_index=find_all_index(flag,0)
    return not_index


def NDS(fit1, fit2):
    """
    非支配排序函数，用于比较两个个体的适应度以判断支配关系。
    参数:
    fit1 (list): 第一个个体的适应度值，包含两个维度。
    fit2 (list): 第二个个体的适应度值，包含两个维度。
    返回:
    v (int): 支配关系的判断结果。0 表示 fit1 和 fit2 没有支配关系，1 表示 fit1 支配 fit2，2 表示 fit2 支配 fit1。
    """
    v = 0
    dom_less = 0  # fit1 的适应度维度值小于 fit2 的数量
    dom_equal = 0  # fit1 和 fit2 的适应度维度值相等的数量
    dom_more = 0  # fit1 的适应度维度值大于 fit2 的数量
    # 遍历两个适应度维度，进行比较
    for k in range(2):
        if fit1[k] > fit2[k]:
            dom_more += 1
        elif fit1[k] == fit2[k]:
            dom_equal += 1
        else:
            dom_less += 1
    # 根据支配关系判断结果赋值给 v
    if dom_less == 0 and dom_equal != 2:
        v = 2  # fit2 支配 fit1
    if dom_more == 0 and dom_equal != 2:
        v = 1  # fit1 支配 fit2
    return v

def Ismemeber(item,list):
    l=len(list)
    flag=0
    for i in range(l):
        if list[i]==item:
            flag=1
            break
    return flag

def DeleteReapt(QP,QF,QFit,ps):
    row=np.size(QFit,0)
    i=0
    while i<row:
        if i>=row:
            #print('break 1')
            break

        F=QFit[i,:]
        j=i+1
        while j<row:
            if QFit[j][0]==F[0] and QFit[j][1]==F[1]:
                QP = np.delete(QP, j, axis=0)
                QF = np.delete(QF, j, axis=0)
                QFit = np.delete(QFit, j, axis=0)
                j=j-1
                row=row-1
                if row<2*ps+1:
                    break
            j=j+1
        i=i+1
        if row < 2 * ps + 1:
            #print('break 2')
            break
    return QP,QF,QFit
def DeleteReapt2(QP,QFit,ps):
    row=np.size(QFit,0)
    i=0
    while i<row:
        if i>=row:
            #print('break 1')
            break
        F=QFit[i]
        j=i+1
        while j<row:
            if QFit[j][0]==F[0] and QFit[j][1]==F[1]:
                QP = list(np.delete(QP, j, axis=0))
                QFit = list(np.delete(QFit, j, axis=0))
                j=j-1
                row=row-1
                if row<2*ps+1:
                    break
            j=j+1
        i=i+1
        if row < 2 * ps + 1:
            #print('break 2')
            break
    return QP,QFit
def DeleteReapt3(QP,QM,QA,QFit,ps):
    # row=np.size(QFit,0)
    row = len(QFit)
    i=0
    while i<row:
        if i>=row:
            break
        F=QFit[i]
        j=i+1
        while j<row:
            if QFit[j][0]==F[0] and QFit[j][1]==F[1]:
                # QP = list(np.delete(QP, j, axis=0))
                # QM = list(np.delete(QM, j, axis=0))
                # QA = list(np.delete(QA, j, axis=0))
                # QFit = list(np.delete(QFit, j, axis=0))
                QP = QP[:j] + QP[j + 1:]
                QM = QM[:j] + QM[j + 1:]
                QA = QA[:j] + QA[j + 1:]
                QFit = QFit[:j] + QFit[j + 1:]
                j=j-1
                row=row-1
                if row<2*ps+1:
                    break
            j=j+1
        i=i+1
        # if row < 2 * ps + 1:
        #     #print('break 2')
        #     break
    return QP,QM,QA,QFit
def DeleteReaptE(QP,QF,QFit): #for elite strategy
    row=np.size(QFit,0)
    i=0
    while i<row:
        if i>=row:
            #print('break 1')
            break

        F=QFit[i,:]
        j=i+1
        while j<row:
            if QFit[j][0]==F[0] and QFit[j][1]==F[1]:
                QP = np.delete(QP, j, axis=0)
                QF = np.delete(QF, j, axis=0)
                QFit = np.delete(QFit, j, axis=0)
                j=j-1
                row=row-1
            j=j+1
        i=i+1

    return QP,QF,QFit

def DeleteReaptACOMOEAD(QP,QF,QFit,QT): #for elite strategy
    row=np.size(QFit,0)
    i=0
    while i<row:
        if i>=row:
            #print('break 1')
            break

        F=QFit[i,:]
        j=i+1
        while j<row:
            if QFit[j][0]==F[0] and QFit[j][1]==F[1]:
                QP = np.delete(QP, j, axis=0)
                QF = np.delete(QF, j, axis=0)
                QFit = np.delete(QFit, j, axis=0)
                QT = np.delete(QT, j, axis=0)
                j=j-1
                row=row-1
            j=j+1
        i=i+1

    return QP,QF,QFit,QT

def DeleteReaptE2(QP,QF,QFit,Fnum): #for elite strategy
    row=np.size(QFit,0)
    i=0
    while i<row:
        if i>=row:
            #print('break 1')
            break

        F=QFit[i,:]
        j=i+1
        while j<row:
            if QFit[j][0]==F[0] and QFit[j][1]==F[1]:
                QP = np.delete(QP, j, axis=0)
                QF = np.delete(QF, j, axis=0)
                QFit = np.delete(QFit, j, axis=0)
                j=j-1
                row=row-1
            f_num = np.zeros(Fnum, dtype=int);
            for f in range(Fnum):
                f_num[f] = len(find_all_index(QF[j,:], f));
            for f in range(Fnum):
                if f_num[f] == 0:
                    QP = np.delete(QP, j, axis=0)
                    QF = np.delete(QF, j, axis=0)
                    QFit = np.delete(QFit, j, axis=0)
                    j = j - 1
                    row = row - 1
            j=j+1
        i=i+1

    return QP,QF,QFit

def pareto(fitness):
    """
    根据帕累托前沿排序算法，从给定的适应度值数组中选择帕累托前沿的解集。
    参数：
    fitness (二维数组)：包含每个解的适应度值的二维数组，每行代表一个解，每列代表一个适应度值。
    返回：
    PF (列表)：包含帕累托前沿解在原适应度值数组中的索引的列表。
    """
    PF = []  # 存放帕累托前沿解的索引
    L = np.size(fitness, axis=0)  # 解的总数（数组行数）
    pn = np.zeros(L, dtype=int)  # 存放每个解被支配的次数

    num_objectives = np.size(fitness, axis=1)  # 目标函数数量

    # 遍历所有解，计算被支配次数
    for i in range(L):
        for j in range(L):
            if i != j and dominates(fitness[i], fitness[j], num_objectives):
                pn[j] += 1

        # 如果解i未被其他解支配，将其索引添加到帕累托前沿中
        if pn[i] == 0:
            PF.append(i)

    return PF
def dominates(x, y, num_objectives):
    """
    判断解 x 是否支配解 y。
    """
    dominates_x = False
    dominates_y = False

    for k in range(num_objectives):
        if x[k] < y[k]:
            dominates_x = True
        elif x[k] > y[k]:
            dominates_y = True

    return dominates_x and not dominates_y
def pareto2(fitness):
    """
    根据帕累托前沿排序算法，从给定的适应度值数组中选择帕累托前沿的解集。
    参数：
    fitness (二维数组)：包含每个解的适应度值的二维数组，每行代表一个解，每列代表一个适应度值。
    返回：
    PF (列表)：包含帕累托前沿解在原适应度值数组中的索引的列表。
    """
    PF = []  # 存放帕累托前沿解的索引
    L = np.size(fitness, axis=0)  # 解的总数（数组行数）
    pn = np.zeros(L, dtype=int)  # 存放每个解被支配的次数
    # 遍历所有解，计算被支配次数
    for i in range(L):
        for j in range(L):
            dom_less = 0
            dom_equal = 0
            dom_more = 0
            # 对每个目标函数值进行比较，判断解的支配关系
            for k in range(3):  # 目标函数数量（这里是两个目标函数）
                if fitness[i][k] > fitness[j][k]:
                    dom_more = dom_more + 1
                elif fitness[i][k] == fitness[j][k]:
                    dom_equal = dom_equal + 1
                else:
                    dom_less = dom_less + 1
            # 判断解i是否被解j支配
            if dom_less == 0 and dom_equal != 2:
                pn[i] = pn[i] + 1
        # 如果解i未被其他解支配，将其索引添加到帕累托前沿中
        if pn[i] == 0:
            PF.append(i)
    return PF