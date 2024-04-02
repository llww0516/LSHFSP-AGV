import numpy as np
from numpy import *
import copy
import math
from PyQt5 import QtWidgets, QtCore
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
# from HFSP_Instance import n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,MNN,ST_i#,DL,DL_bat

class Machines:
    #初始化机器类
    def __init__(self):
        self.start=[]#开始时间
        self.end=[0]#结束时间
        self._on=[]#
        self._on_agv=[]
        self.T=[]#所用时间
        self.last_ot=0#完成时间
        self.Idle = []#空闲时间集合
    #更新机器类
    def update(self,s,e,t,on):
        self.start.append(s)
        self.end.append(e)
        self.T.append(t)
        self.last_ot=e
        self._on.append(on)  # 一共加工过的工件序号
    def update2(self,s,e,t,on,agv):
        self.start.append(s)
        self.end.append(e)
        self.T.append(t)
        self.last_ot=e
        self._on.append(on)  # 一共加工过的工件序号
        self._on_agv.append(agv)
    # 更新空闲时间
    def idle_time(self):
        try:
            if len(self.start) > 1:
                K = [[self.end[-2], self.start[-1]]]
                self.Idle.extend(K)
        except:
            pass

class job:
    def __init__(self,idx):
        self.start = []  # 开始时间
        self.end = [0]  # 结束时间
        self.on_m = [0]  # 各工序加工机器编号（工序内）
        self.T = []  # 所用时间
        self.last_ot = 0  # 完成时间
        self.idx=idx
        self.cur_site=0#当前位置

    def get_info(self):
        return self.end,self.cur_site,self.on_m
            #加工结束时间，当前位置，      各工序加工机器编号（工序内）

    def update(self,s,e,t,op_m):
        self.start.append(s)
        self.end.append(e)
        self.T.append(t)
        self.last_ot = e
        self.on_m.append(op_m)  # 所有阶段加工的机器序号
        self.cur_site = op_m # 下一工序加工机器序号

class Agv:
    def __init__(self,AGV_idx):
        self.idx=AGV_idx # AGV编号
        self.cur_site=1 # AGV当前位置
        self.cur_site2=[0,0] # AGV当前位置
        self.using_time=[]
        self.T = []  # 所用时间
        self._on=[]
        self._to=[]
        self.end=0

    def ST(self,s,t1,t2):
        if s > (self.end+t1):#工件和AGV那个最晚到当前分拣区，以此时间点为AGV负载开始时间点
            return s - t1, s + t2  # 输出AGV空载开始时间点，负载结束时间点
        else:
            return self.end, self.end + t1 + t2  # 输出AGV空载开始时间点，负载结束时间点

    def update(self,s,e,trans1,trans2,J_site,J_m, m_sta, m_id,_on=None):
        self.using_time.append([s,s+trans1])#[空载开始，空载结束]
        self.using_time.append([e-trans2, e])#[负载开始，负载结束]
        self.T.append(trans1+trans2)
        self._on.append(None)
        self._on.append(_on)#负载工件批次序号
        self._to.extend([J_site,J_m])#[负载开始机器序号，负载结束机器序号]
        self.end=e#负载结束时间点
        self.cur_site=J_m#更新当前位置
        self.cur_site2 = [(m_sta-1)*60, m_id*30]  # 更新当前位置

class Situation:
    def __init__(self,n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id):
        self.n=n
        self.J_num=J_num
        self.State=State
        self.M_j=M_j
        self.PT=PT
        self.D_pq=D_pq
        self.JNN=JNN
        self.v=v
        self.V_r=V_r
        self.JN_ip=JN_ip
        self.ST_i=ST_i
        self.MNN=MNN
        self.suanli=suanli
        self.xunlian_id = xunlian_id
        self.reset()
    def reset(self):
        self.Jobs=[]#[0，工件1批次1,工件1批次2,。。]
        for i in range(self.J_num+1):
            J = job(i)
            self.Jobs.append(J)
        self.AL_jk=[] # j阶段的机器k [[],阶段1[机器1，机器2，。。。],阶段2[],阶段3[]]
        for i in range(len(self.M_j)):    #突出机器的阶段性，即各阶段有哪些机器
            if i == 0:
                self.AL_jk.append([])
            else:
                State_i=[]
                for j in range(self.M_j[i]):
                    M=Machines()
                    State_i.append(M)
                self.AL_jk.append(State_i)
        self.AGVs=[]#[0,AGV1,AGV2,。。]
        for k in range(self.v + 1):
            agv = Agv(k)
            self.AGVs.append(agv)
        self.fitness1 = 0
        self.fitness2 = 0
        self.fitness3 = 0
        self.Done = []
        self.CRJBJL = [[0 for _ in row] for row in self.JNN]  # 工件批次完成记录表 [[0], [0], [0, 0, 0], [0, 0], [0, 0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0, 0]]
        self.CRJB = [0 for i in range(self.n+1)] # 工件批次完成率
        self.CRS = [0 for i in range(self.n+1)] # 作业完成率
        self.Makespan_pre = [0 for i in range(self.J_num+1)] # 工件预计完工时间
        for Job in range(1,self.J_num+1):
            j_arr, j_bat = self.find_element_index(self.JNN, Job)  # 工件种类序号 批次号
            self.Makespan_pre[Job] = self.shengyu_jiagong_banyun(Job, 0, self.ST_i[j_arr] * self.JN_ip[j_arr][j_bat])
        self.Makespan_pre[0] = 0
        self.UK = []  # 各机器利用率 [[], [0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0]]
        self.yunshu_AGV = [0 for _ in range(self.v+1)] # AGV运输距离 [0, 0, 0, 0]
        self.m = 1.0
        self.s = 99999
        self.u = 0.0
        self.a = 1.0
        for i in range(self.State+1):
            self.UK.append([0 for i in range(self.M_j[i])])
    def reschedule(self, faultyMachine, faultyTime_Mac, repairTime_Mac, faultyAGV, faultyTime_AGV, repairTime_AGV):
        self.faultyMachine = faultyMachine
        self.faultyTime_Mac = faultyTime_Mac
        self.repairTime_Mac = repairTime_Mac
        self.faultyAGV = faultyAGV
        self.faultyTime_AGV = faultyTime_AGV
        self.repairTime_AGV = repairTime_AGV
    def calculate_sum(self,matrix):# 计算列表总和
        total_sum = 0
        for row in matrix:
            total_sum += sum(row)
        return total_sum
    def count_element(self,matrix, target):# 返回目标元素的个数
        count = 0
        for row in matrix:
            count += row.count(target)
        return count
    def find_element_index(self,mat, target):# 返回目标元素所在的行和列索引
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                if mat[i][j] == target:
                    return i, j
    def find_all_indices(self,lst, target):# 返回目标元素所有索引
        indices = []
        for idx, value in enumerate(lst):
            if value == target:
                indices.append(idx)
        return indices
    def find_all_min_indices(self,lst):# 返回目标列表所有最小值的索引
        min_value = min(lst)
        indices = [idx for idx, value in enumerate(lst) if value == min_value]
        return indices
    def find_all_max_indices(self,lst):# 返回目标列表所有最大值的索引
        max_value = max(lst)
        indices = [idx for idx, value in enumerate(lst) if value == max_value]
        return indices
    def shengyu_jiagong_banyun(self,Job,state,time):# 计算Job工件state阶段之后（不包括state阶段）的剩余加工时间和搬运时间
        Ji = self.Jobs[Job]  # 当前批次工件类
        pts_sum = time
        for s in range(state+1, self.State):
            pt_sum = 0
            if s == state+1:
                for j in range(len(self.AL_jk[s])):
                    pt_sum += self.PT[s][j][Job] + self.D_pq[Ji.cur_site][self.MNN[s][j]] / self.V_r
            else:
                d_pq = 0
                for j in range(len(self.AL_jk[s])):
                    pt_sum += self.PT[s][j][Job]
                    for i in range(len(self.AL_jk[s + 1])):
                        d_pq += self.D_pq[self.MNN[s+1][i]][self.MNN[s][j]] / self.V_r
                pt_sum += d_pq / len(self.AL_jk[s + 1])
            pts_sum += pt_sum / len(self.AL_jk[s])
        return pts_sum

    def scheduling(self, Job, Machine, Agv):  # 根据行动（选择的工件，机器），将加工工序所在工件、加工位置、加工时间开始、结束均存储起来
        # 更新调度后的加工状态（原始状态）
        #  确定工件开始时间
        Ji = self.Jobs[Job]  # 当前批次工件类
        j_sta, j_mac = self.find_element_index(self.MNN, Ji.cur_site)  # 工件当前所在机器的阶段（0、1、2...state）和机器序号（阶段内）
        j_arr, j_bat = self.find_element_index(self.JNN, Job)  # 工件种类序号 批次号
        m_sta, m_id = self.find_element_index(self.MNN, Machine)  # 下一工序机器所在阶段和序号（阶段内）
        J_m = self.AL_jk[m_sta][m_id]  # 下一工序机器
        A_i = self.AGVs[Agv]  # 运输工件到下一工序的AGV
        last_ot = Ji.last_ot  # 工件上道工序加工结束时间
        last_mt = J_m.last_ot  # 工件下一工序机器上道工序加工结束时间
        if j_sta < 1:
            Start_time = max(last_ot + self.ST_i[j_arr] * self.JN_ip[j_arr][j_bat], last_mt)
        else:
            J_end, J_site, on_m = Ji.get_info()  # 加工结束时间，当前位置，各工序加工机器编号（工序内）
            trans1 = self.D_pq[A_i.cur_site][J_site]  # AGV到工件当前位置距离
            trans2 = self.D_pq[J_site][self.MNN[m_sta][m_id]]  # 工件到下一工序机器距离
            start, end = A_i.ST(J_end[-1], trans1 / self.V_r, trans2 / self.V_r)  # start：AGV空载开始时间点，end：负载结束时间点
            A_i.update(start, end, trans1 / self.V_r, trans2 / self.V_r, J_site, Machine, m_sta, m_id, Ji.idx)
            Start_time = max(A_i.end, last_mt)  # 以AGV送到和机器结束上一个工件加工时间大者为开始加工时间
            '''AGV状态信息'''
            # self.UK_AGV[Agv] = 1.0 if A_i.end == 0 else (A_i.end - A_i.using_time[0][0] - sum(A_i.T)) / (A_i.end - A_i.using_time[0][0])  # AGV空闲率
            self.yunshu_AGV[Agv] = sum(A_i.T)  # AGV运输距离
        pt = self.PT[m_sta][m_id][Job]  # 即将加工的工序加工时间
        end_time = Start_time + pt
        J_m.update2(Start_time, end_time, pt, Job, Agv)
        J_m.idle_time()
        Ji.update(Start_time, end_time, pt, Machine)
        '''工件状态信息'''
        j_arr, j_bat = self.find_element_index(self.JNN, Job)  # 找到Job的种类序号和批次号
        self.CRJB[j_arr] = self.CRJBJL[j_arr].count(self.State) / len(self.JNN[j_arr])  # 工件批次完成率
        CRJB_sum = 0  # 该类工件阶段完成数
        for i in range(len(self.JNN[j_arr])):
            CRJB_sum += self.JN_ip[j_arr][i] * self.CRJBJL[j_arr][i]  # 该类工件每批数量*该批完成阶段数
        self.CRS[j_arr] = CRJB_sum / (sum(self.JN_ip[j_arr]) * self.State)  # 该类工件阶段完成率
        self.Makespan_pre[Job] = Ji.last_ot + self.shengyu_jiagong_banyun(Job, m_sta, 0)  # 该批工件预计完工时间=当前时间+剩余加工时间+剩余搬运时间

        self.CRJBJL[j_arr][j_bat] += 1  # 该批工件完成阶段数+1
        '''机器状态信息'''
        self.UK[m_sta][m_id] = 0 if J_m.last_ot == 0 else sum(J_m.T) / (J_m.last_ot - J_m.start[0])  # 机器利用率

    def rescheduling(self, Job, Machine, Agv):  # 根据行动（选择的工件，机器），将加工工序所在工件、加工位置、加工时间开始、结束均存储起来
        # 更新调度后的加工状态（原始状态）
        #  确定工件开始时间
        Ji = self.Jobs[Job]  # 当前批次工件类
        j_sta, j_mac = self.find_element_index(self.MNN, Ji.cur_site)  # 工件当前所在机器的阶段（0、1、2...state）和机器序号（阶段内）
        j_arr, j_bat = self.find_element_index(self.JNN, Job)  # 工件种类序号 批次号
        j_fenjian = self.ST_i[j_arr] * self.JN_ip[j_arr][j_bat] # 工件分拣时间
        m_sta, m_id = self.find_element_index(self.MNN, Machine)  # 下一工序机器所在阶段和序号（阶段内）
        J_m = self.AL_jk[m_sta][m_id]  # 下一工序机器
        A_i_1 = self.AGVs[Agv]  # 运输工件到下一工序的AGV
        a_sta, a_mac = self.find_element_index(self.MNN, A_i_1.cur_site)  # AGV当前所在机器的阶段（0、1、2...state）和机器序号（阶段内）
        J_end, J_site, on_m = Ji.get_info()  # 加工结束时间，当前位置，各工序加工机器编号（工序内）
        last_mt = J_m.last_ot  # 工件下一工序机器上道工序加工结束时间
        if j_sta < 1:
            A_i_2 = A_i_1
            Start_time = max(j_fenjian, last_mt)
        else:
            min_tf = 99999
            trans1 = ((A_i_1.cur_site2[0] - (j_sta - 1) * 60) ** 2 + (A_i_1.cur_site2[1] - j_mac * 30) ** 2) ** 0.5 # AGV到工件当前位置距离
            # trans1 = self.D_pq[A_i_1.cur_site][J_site]
            trans2 = self.D_pq[J_site][Machine]  # 工件到下一工序机器距离
            start, end = A_i_1.ST(J_end[-1], trans1 / self.V_r, trans2 / self.V_r)  # start：AGV空载开始时间点，end：负载结束时间点
            best_s, best_e, t1, t2 = start, end, trans1 / self.V_r, trans2 / self.V_r
            if self.faultyAGV[Agv] == 1 and len(A_i_1.using_time) != 0:
                A_i_2 = A_i_1
                if self.faultyTime_AGV[Agv] < start < self.faultyTime_AGV[Agv] + self.repairTime_AGV:# 在空载前故障
                    for agv in [i for i in range(1, len(self.AGVs)) if i!= Agv]:
                        trans1 = ((self.AGVs[agv].cur_site2[0] - (j_sta - 1) * 60) ** 2 + (self.AGVs[agv].cur_site2[1] - j_mac * 30) ** 2) ** 0.5
                        start, end = self.AGVs[agv].ST(J_end[-1], trans1 / self.V_r, trans2 / self.V_r)  # start：AGV空载开始时间点，end：负载结束时间点
                        if end < min_tf and not (start <= self.faultyTime_AGV[agv] <= end):
                            best_s, best_e, t1, t2 = start, end, trans1 / self.V_r, trans2 / self.V_r
                            A_i_2 = self.AGVs[agv]
                            min_tf = best_e
                    Start_time = max(best_e, last_mt)  # 以AGV送到和机器结束上一个工件加工时间大者为开始加工时间
                elif start < self.faultyTime_AGV[Agv] < start+trans1:# 空载时故障
                    per = (self.faultyTime_AGV[Agv] - start) / trans1 # AGV空载前进进度
                    for agv in [i for i in range(1, len(self.AGVs)) if i!= Agv]:
                        j_sta, j_mac = self.find_element_index(self.MNN, J_site)  # 工件当前所在机器的阶段（0、1、2...state）和机器序号（阶段内）
                        trans1 = ((self.AGVs[agv].cur_site2[0] - (j_sta - 1) * 60) ** 2 + (self.AGVs[agv].cur_site2[1] - j_mac * 30) ** 2) ** 0.5
                        start, end = self.AGVs[agv].ST(J_end[-1], trans1 / self.V_r, trans2 / self.V_r)  # start：AGV空载开始时间点，end：负载结束时间点
                        if end < min_tf and not (start <= self.faultyTime_AGV[agv] <= end):
                            best_s, best_e, t1, t2 = start, end, trans1 / self.V_r, trans2 / self.V_r
                            A_i_2 = self.AGVs[agv]
                            min_tf = best_e
                    Start_time = max(best_e, last_mt)  # 以AGV送到和机器结束上一个工件加工时间大者为开始加工时间
                elif end-trans2 < self.faultyTime_AGV[Agv] < end:# 负载时故障
                    per = (self.faultyTime_AGV[Agv] - end + trans2) / trans2 # AGV负载前进进度
                    for agv in [i for i in range(1, len(self.AGVs)) if i!= Agv]:
                        trans1 = ((self.AGVs[agv].cur_site2[0] - (a_sta-1-(a_sta-j_sta)*per)*60) ** 2 + (self.AGVs[agv].cur_site2[1] - (a_mac-(a_mac-j_mac)*per) * 30) ** 2) ** 0.5 # AGV到故障AGV位置距离
                        trans2 = (((m_sta-a_sta-(a_sta-j_sta)*per)*60) ** 2 + ((m_id-a_mac-(a_mac-j_mac)*per) * 30) ** 2) ** 0.5 # 工件到下一工序机器距离
                        start, end = self.AGVs[agv].ST(J_end[-1], trans1 / self.V_r, trans2 / self.V_r)  # start：AGV空载开始时间点，end：负载结束时间点
                        if end < min_tf and not (start <= self.faultyTime_AGV[agv] <= end):
                            best_s, best_e, t1, t2 = start, end, trans1 / self.V_r, trans2 / self.V_r
                            A_i_2 = self.AGVs[agv]
                            min_tf = best_e
                    Start_time = max(best_e, last_mt)  # 以AGV送到和机器结束上一个工件加工时间大者为开始加工时间
                else:# 负载完成后故障
                    A_i_2 = A_i_1
                    Start_time = max(end, last_mt)  # 以AGV送到和机器结束上一个工件加工时间大者为开始加工时间
            else:
                A_i_2 = A_i_1
                Start_time = max(end, last_mt)  # 以AGV送到和机器结束上一个工件加工时间大者为开始加工时间
        pt = self.PT[m_sta][m_id][Job] # 即将加工的工序加工时间
        end_time = Start_time + pt
        if self.faultyMachine == Machine :
            if self.faultyTime_Mac + self.repairTime_Mac < Start_time:
                if J_m.end[-1] < self.faultyTime_Mac + self.repairTime_Mac:
                    J_m.update(self.faultyTime_Mac, self.faultyTime_Mac + self.repairTime_Mac, self.repairTime_Mac, 999)
                Start_time = self.Genxin_AGV(j_sta, j_fenjian, last_mt, A_i_1, A_i_2, j_mac, J_site, Machine, J_end, Agv, a_sta, a_mac, Ji, m_sta, m_id, J_m, pt, Job)
                end_time = Start_time + pt
                J_m.update(Start_time, end_time, pt, Job)
                J_m.idle_time()
            elif self.faultyTime_Mac < Start_time < self.faultyTime_Mac+self.repairTime_Mac:
                Machine = random.choice([i for i in self.MNN[m_sta] if i!= Machine])
                m_sta, m_id = self.find_element_index(self.MNN, Machine)  # 下一工序机器所在阶段和序号（阶段内）
                J_m = self.AL_jk[m_sta][m_id]  # 下一工序机器
                last_mt = J_m.last_ot  # 工件下一工序机器上道工序加工结束时间
                Start_time = self.Genxin_AGV(j_sta,j_fenjian,last_mt,A_i_1, A_i_2,j_mac,J_site,Machine,J_end,Agv,a_sta,a_mac,Ji,m_sta,m_id,J_m,pt,Job)
                end_time = Start_time + pt
                J_m.update(Start_time,end_time,pt,Job)
                J_m.idle_time()
            elif Start_time < self.faultyTime_Mac < end_time:
                Start_time = self.Genxin_AGV(j_sta,j_fenjian,last_mt,A_i_1, A_i_2,j_mac,J_site,Machine,J_end,Agv,a_sta,a_mac,Ji,m_sta,m_id,J_m,pt,Job)
                end_time = Start_time + pt + self.repairTime_Mac
                J_m.update(Start_time,self.faultyTime_Mac,self.faultyTime_Mac-Start_time,Job)
                J_m.update(self.faultyTime_Mac,self.faultyTime_Mac+self.repairTime_Mac,self.repairTime_Mac,999)
                J_m.update(self.faultyTime_Mac+self.repairTime_Mac,self.repairTime_Mac+pt+Start_time,pt+Start_time-self.faultyTime_Mac,Job)
                J_m.idle_time()
            else:
                Start_time = self.Genxin_AGV(j_sta,j_fenjian,last_mt,A_i_1, A_i_2,j_mac,J_site,Machine,J_end,Agv,a_sta,a_mac,Ji,m_sta,m_id,J_m,pt,Job)
                end_time = Start_time + pt
                J_m.update(Start_time,end_time,pt,Job)
                J_m.idle_time()
        else:
            Start_time = self.Genxin_AGV(j_sta, j_fenjian, last_mt, A_i_1, A_i_2, j_mac, J_site, Machine, J_end, Agv, a_sta, a_mac, Ji, m_sta, m_id, J_m, pt, Job)
            end_time = Start_time + pt
            J_m.update(Start_time, end_time, pt, Job)
            J_m.idle_time()
        Ji.update(Start_time,end_time,pt,Machine)
        '''工件状态信息'''
        j_arr, j_bat = self.find_element_index(self.JNN, Job)  # 找到Job的种类序号和批次号
        self.CRJB[j_arr] = self.CRJBJL[j_arr].count(self.State) / len(self.JNN[j_arr])# 工件批次完成率
        CRJB_sum=0 # 该类工件阶段完成数
        for i in range(len(self.JNN[j_arr])):
            CRJB_sum += self.JN_ip[j_arr][i] * self.CRJBJL[j_arr][i] # 该类工件每批数量*该批完成阶段数
        self.CRS[j_arr] = CRJB_sum / (sum(self.JN_ip[j_arr]) * self.State) # 该类工件阶段完成率
        self.Makespan_pre[Job] = Ji.last_ot + self.shengyu_jiagong_banyun(Job, m_sta, 0) # 该批工件预计完工时间=当前时间+剩余加工时间+剩余搬运时间

        self.CRJBJL[j_arr][j_bat] += 1  # 该批工件完成阶段数+1
        '''机器状态信息'''
        self.UK[m_sta][m_id] = 0 if J_m.last_ot == 0 else sum(J_m.T) / (J_m.last_ot - J_m.start[0]) # 机器利用率
    def Genxin_AGV(self,j_sta,j_fenjian,last_mt,A_i_1,A_i_2,j_mac,J_site,Machine,J_end,Agv,a_sta,a_mac,Ji,m_sta,m_id,J_m,pt,Job):
        if j_sta < 1:
            Start_time = max(j_fenjian, last_mt)
        else:
            min_tf = 99999
            trans1 = ((A_i_1.cur_site2[0] - (j_sta - 1) * 60) ** 2 + (A_i_1.cur_site2[1] - j_mac * 30) ** 2) ** 0.5  # AGV到工件当前位置距离
            # trans1 = self.D_pq[A_i_1.cur_site][J_site]
            trans2 = self.D_pq[J_site][Machine]  # 工件到下一工序机器距离
            start, end = A_i_1.ST(J_end[-1], trans1 / self.V_r, trans2 / self.V_r)  # start：AGV空载开始时间点，end：负载结束时间点
            best_s, best_e, t1, t2 = start, end, trans1 / self.V_r, trans2 / self.V_r
            if self.faultyAGV[Agv] == 1 :
                if self.faultyTime_AGV[Agv] + self.repairTime_AGV < start:  # 在空载前故障
                    A_i_1.using_time.append([self.faultyTime_AGV[Agv], self.faultyTime_AGV[Agv] + self.repairTime_AGV])  # [故障开始，故障结束]
                    A_i_1.T.append(0)
                    A_i_1._on.append(999)  # 负载工件批次序号
                    A_i_1.end = self.faultyTime_AGV[Agv] + self.repairTime_AGV  # 负载结束时间点
                    A_i_2 = A_i_1
                    A_i_2.update(start, end, trans1 / self.V_r, trans2 / self.V_r, J_site, Machine, m_sta, m_id, Ji.idx)
                    Start_time = max(end, last_mt)  # 以AGV送到和机器结束上一个工件加工时间大者为开始加工时间
                elif self.faultyTime_AGV[Agv] < start < self.faultyTime_AGV[Agv] + self.repairTime_AGV:  # 在空载时故障
                    for agv in [i for i in range(1, len(self.AGVs)) if i != Agv]:
                        trans1 = ((self.AGVs[agv].cur_site2[0] - (j_sta - 1) * 60) ** 2 + (self.AGVs[agv].cur_site2[1] - j_mac * 30) ** 2) ** 0.5
                        start, end = self.AGVs[agv].ST(J_end[-1], trans1 / self.V_r, trans2 / self.V_r)  # start：AGV空载开始时间点，end：负载结束时间点
                        if end < min_tf and not (start <= self.faultyTime_AGV[agv] <= end):
                            best_s, best_e, t1, t2 = start, end, trans1 / self.V_r, trans2 / self.V_r
                            A_i_2 = self.AGVs[agv]
                            min_tf = best_e
                    A_i_2.update(best_s, best_e, t1, t2, J_site, Machine, m_sta, m_id, Ji.idx)  # 选出最优的AGV
                    Start_time = max(best_e, last_mt)  # 以AGV送到和机器结束上一个工件加工时间大者为开始加工时间
                elif start < self.faultyTime_AGV[Agv] < start + trans1:  # 空载时故障
                    per = (self.faultyTime_AGV[Agv] - start) / trans1  # AGV空载前进进度
                    A_i_1.using_time.append([start, self.faultyTime_AGV[Agv]])  # [空载开始，故障开始]
                    A_i_1.using_time.append([self.faultyTime_AGV[Agv], self.faultyTime_AGV[Agv] + self.repairTime_AGV])  # [故障开始，故障结束]
                    A_i_1.T.append(self.faultyTime_AGV[Agv] - start)
                    A_i_1._on.append(None)
                    A_i_1._on.append(999)  # 负载工件批次序号
                    A_i_1.end = self.faultyTime_AGV[Agv] + self.repairTime_AGV  # 负载结束时间点
                    A_i_1.cur_site = Machine  # 更新当前位置
                    A_i_1.cur_site2 = [(a_sta - 1 - (a_sta - j_sta) * per) * 60, (a_mac - (a_mac - j_mac) * per) * 30]  # 更新当前故障位置
                    for agv in [i for i in range(1, len(self.AGVs)) if i != Agv]:
                        trans1 = ((self.AGVs[agv].cur_site2[0] - (j_sta - 1) * 60) ** 2 + (self.AGVs[agv].cur_site2[1] - j_mac * 30) ** 2) ** 0.5
                        start, end = self.AGVs[agv].ST(J_end[-1], trans1 / self.V_r, trans2 / self.V_r)  # start：AGV空载开始时间点，end：负载结束时间点
                        if end < min_tf and not (start <= self.faultyTime_AGV[agv] <= end):
                            best_s, best_e, t1, t2 = start, end, trans1 / self.V_r, trans2 / self.V_r
                            A_i_2 = self.AGVs[agv]
                            min_tf = best_e
                    A_i_2.update(best_s, best_e, t1, t2, J_site, Machine, m_sta, m_id, Ji.idx)  # 选出最优的AGV
                    Start_time = max(best_e, last_mt)  # 以AGV送到和机器结束上一个工件加工时间大者为开始加工时间
                elif end - trans2 < self.faultyTime_AGV[Agv] < end:  # 负载时故障
                    per = (self.faultyTime_AGV[Agv] - end + trans2) / trans2  # AGV负载前进进度
                    A_i_1.update(start, self.faultyTime_AGV[Agv], trans1, self.faultyTime_AGV[Agv] - end + trans2, J_site, Machine, m_sta, m_id, Ji.idx)
                    A_i_1.using_time.append([self.faultyTime_AGV[Agv], self.faultyTime_AGV[Agv] + self.repairTime_AGV])  # [故障开始，故障结束]
                    A_i_1.T.append(0)
                    A_i_1._on.append(999)  # 负载工件批次序号
                    A_i_1.end = self.faultyTime_AGV[Agv] + self.repairTime_AGV  # 负载结束时间点
                    A_i_1.cur_site2 = [(a_sta - 1 - (a_sta - j_sta) * per) * 60, (a_mac - (a_mac - j_mac) * per) * 30]  # 更新当前故障位置
                    for agv in [i for i in range(1, len(self.AGVs)) if i != Agv]:
                        trans1 = ((self.AGVs[agv].cur_site2[0] - (a_sta - 1 - (a_sta - j_sta) * per) * 60) ** 2 + (self.AGVs[agv].cur_site2[1] - (a_mac - (a_mac - j_mac) * per) * 30) ** 2) ** 0.5  # AGV到故障AGV位置距离
                        trans2 = (((m_sta - a_sta - (a_sta - j_sta) * per) * 60) ** 2 + ((m_id - a_mac - (a_mac - j_mac) * per) * 30) ** 2) ** 0.5  # 工件到下一工序机器距离
                        start, end = self.AGVs[agv].ST(J_end[-1], trans1 / self.V_r, trans2 / self.V_r)  # start：AGV空载开始时间点，end：负载结束时间点
                        if end < min_tf and not (start <= self.faultyTime_AGV[agv] <= end):
                            best_s, best_e, t1, t2 = start, end, trans1 / self.V_r, trans2 / self.V_r
                            A_i_2 = self.AGVs[agv]
                            min_tf = best_e
                    A_i_2.update(best_s, best_e, t1, t2, J_site, Machine, m_sta, m_id, Ji.idx)  # 选出最优的AGV
                    Start_time = max(best_e, last_mt)  # 以AGV送到和机器结束上一个工件加工时间大者为开始加工时间
                else:  # 负载完成后故障
                    A_i_2 = A_i_1
                    A_i_2.update(start, end, trans1 / self.V_r, trans2 / self.V_r, J_site, Machine, m_sta, m_id, Ji.idx)
                    Start_time = max(end, last_mt)  # 以AGV送到和机器结束上一个工件加工时间大者为开始加工时间
            else:
                A_i_2 = A_i_1
                A_i_2.update(start, end, trans1 / self.V_r, trans2 / self.V_r, J_site, Machine, m_sta, m_id, Ji.idx)
                Start_time = max(end, last_mt)  # 以AGV送到和机器结束上一个工件加工时间大者为开始加工时间
            '''AGV状态信息'''
            # self.UK_AGV[Agv] = 1.0 if A_i_1.end == 0 else (A_i_1.end - A_i_1.using_time[0][0] - sum(A_i_1.T)) / (A_i_1.end - A_i_1.using_time[0][0])  # AGV空闲率
            self.yunshu_AGV[A_i_2.idx] = round(sum(A_i_2.T), 2)  # AGV运输距离
        return Start_time

    def Features(self):  # 同时反应局部信息和整体信息，定义机器特征，从中选择合适的工件
        #1 平均工件批次完成率
        CRJB_ave = sum(self.CRJB) / self.n
        #2 平均作业完成率
        CRS_ave = sum(self.CRS) / self.n
        #3 作业完成率的标准差
        K = 0
        for uk in self.CRS:
            K += np.square(uk - CRS_ave)
        CRS_std = np.sqrt(K / self.n)
        #4 机器平均利用率
        U_ave = self.calculate_sum(self.UK) / (sum(self.M_j) - 1)
        #5 机器平均利用率标准差
        K = 0
        for i in range(len(self.UK)):
            for j in range(len(self.UK[i])):
                K += np.square(j - U_ave)
        U_std = np.sqrt(K / (sum(self.M_j) - 1))
        #6 工件完工时间
        C_max = self.AL_jk[1][0].last_ot
        for i in range(1, len(self.AL_jk)):
            for j in range(len(self.AL_jk[i])):
                if self.AL_jk[i][j].last_ot > C_max:
                    C_max = self.AL_jk[i][j].last_ot
        #7 机器全部空闲时间
        Idle_sum = 0
        for i in range(1, len(self.AL_jk)):
            Idle_time = self.Idle_time_jisuan(i)
            Idle_sum += sum(Idle_time)
            # Mac_end = []
            # for j in range(len(self.AL_jk[i])):
            #     if len(self.AL_jk[i][j].end)>0:
            #         Mac_end.append(self.AL_jk[i][j].end[-1])
            # for j in range(len(self.AL_jk[i])):
            #     if len(self.AL_jk[i][j].end) == 1:
            #         Idle_sum += max(Mac_end) - self.AL_jk[i][j].end[-1]
            #     else:
            #         for m in self.AL_jk[i][j].Idle:
            #             Idle_sum += m[1] - m[0]
        #8 AGV总运输距离
        # x, y = 0, 0
        # for i in range(1, self.v + 1):
        #     x += 0 if self.AGVs[i].end == 0 else self.AGVs[i].end - self.AGVs[i].using_time[0][0] - sum(self.AGVs[i].T)
        #     y += 0 if self.AGVs[i].end == 0 else self.AGVs[i].end - self.AGVs[i].using_time[0][0]
        # S_sum = 1.0 if y == 0 else x / y
        S_sum = sum(self.yunshu_AGV)

        return CRJB_ave, CRS_ave, CRS_std, U_ave, U_std, round(C_max,2), Idle_sum, S_sum
    def reward(self, NSGA2_fit1, NSGA2_fit2, NSGA2_fit3, Multi_O=True):
        if Multi_O:
            reward = []
            m = self.Makespan_pre[1]
            for i in range(1, self.J_num + 1):
                if m < self.Makespan_pre[i]:
                    m = self.Makespan_pre[i]
            if m < self.s:
                rt1 = 1
            elif m > self.s:
                rt1 = -1
            else:
                rt1 = 0
            self.s = m
            reward.append(rt1)

            u = self.calculate_sum(self.UK) / (sum(self.M_j) - 1)
            if u > self.u:
                rt2 = 1
            elif u < self.u:
                rt2 = -1
            else:
                rt2 = 0
            self.u = u
            reward.append(rt2)

            # a = self.calculate_sum(self.UK_AGV) / self.v
            # if a > self.a:
            #     rt3 = -1
            # elif a < self.a:
            #     rt3 = 1
            # else:
            #     rt3 = 0
            # self.a = a
            x,y = 0,0
            for i in range(1,self.v+1):
                x += sum(self.AGVs[i].T)
                y += 0 if self.AGVs[i].end == 0 else self.AGVs[i].end - self.AGVs[i].using_time[0][0]
            a = x / y
            if a < self.a:
                rt3 = 1
            elif a > self.a:
                rt3 = -1
            else:
                rt3 = 0
            self.a = a
            reward.append(rt3)
        else:
            rt = 0
            m = self.Makespan_pre[1]
            for i in range(1, self.J_num+1):
                if m < self.Makespan_pre[i]:
                    m = self.Makespan_pre[i]
            if m < self.s:
                rt += 1
            elif m > self.s:
                rt += -1
            else:
                rt += 0
            self.s = m

            # u = self.calculate_sum(self.UK) / (sum(self.M_j) - 1)
            # if u > self.u:
            #     rt += 1
            # elif u < self.u:
            #     rt += -1
            # else:
            #     rt += 0
            # self.u = u

            x,y = 0,0
            for i in range(1,self.v+1):
                x += 0 if self.AGVs[i].end == 0 else self.AGVs[i].end - self.AGVs[i].using_time[0][0] - sum(self.AGVs[i].T)
                y += 0 if self.AGVs[i].end == 0 else self.AGVs[i].end - self.AGVs[i].using_time[0][0]
            a = 1.0 if y == 0 else x / y
            if a > self.a:
                rt += -1
            elif a < self.a:
                rt += 1
            else:
                rt += 0
            self.a = a
            reward = rt
        return reward
    def reward2(self, step, NSGA2_fit1, NSGA2_fit2, NSGA2_fit3, MDDQN_fit1, MDDQN_fit2, MDDQN_fit3, Multi_O=True):
        if Multi_O:
            reward = []
            m = self.AL_jk[1][0].last_ot
            for i in range(1, len(self.AL_jk)):
                for j in range(len(self.AL_jk[i])):
                    if self.AL_jk[i][j].last_ot > m:
                        m = self.AL_jk[i][j].last_ot
            if m < min(NSGA2_fit1[step], MDDQN_fit1[step]):
                rt1 = 1
            elif m > min(NSGA2_fit1[step], MDDQN_fit1[step]):
                rt1 = -1
            else:
                rt1 = 0
            reward.append(rt1)

            u = 0
            for i in range(1, len(self.AL_jk)):
                Idle_time = self.Idle_time_jisuan(i)
                u += sum(Idle_time)
            if u > min(NSGA2_fit2[step], MDDQN_fit2[step]):
                rt2 = 1
            elif u < min(NSGA2_fit2[step], MDDQN_fit2[step]):
                rt2 = -1
            else:
                rt2 = 0
            reward.append(rt2)

            a = sum(self.yunshu_AGV)
            if a < min(NSGA2_fit3[step], MDDQN_fit3[step]):
                rt3 = 1
            elif a > min(NSGA2_fit3[step], MDDQN_fit3[step]):
                rt3 = -1
            else:
                rt3 = 0
            reward.append(rt3)
        else:
            rt = 0
            m = self.AL_jk[1][0].last_ot
            for i in range(1, len(self.AL_jk)):
                for j in range(len(self.AL_jk[i])):
                    if self.AL_jk[i][j].last_ot > m:
                        m = self.AL_jk[i][j].last_ot
            if m < 0.9*min(NSGA2_fit1[step], MDDQN_fit1[step]):
                rt += 3
            elif 0.9*min(NSGA2_fit1[step], MDDQN_fit1[step]) <= m < min(NSGA2_fit1[step], MDDQN_fit1[step]):
                rt += 1
            elif 1.1*min(NSGA2_fit1[step], MDDQN_fit1[step]) >= m > min(NSGA2_fit1[step], MDDQN_fit1[step]):
                rt += -1
            elif m > 1.1*min(NSGA2_fit1[step], MDDQN_fit1[step]):
                rt += -3
            else:
                rt += 0

            u = 0
            for i in range(1, len(self.AL_jk)):
                Idle_time = self.Idle_time_jisuan(i)
                u += sum(Idle_time)
            if u < 0.9*min(NSGA2_fit2[step], MDDQN_fit2[step]):
                rt += 3
            elif 0.9*min(NSGA2_fit2[step], MDDQN_fit2[step]) <= u < min(NSGA2_fit2[step], MDDQN_fit2[step]):
                rt += 1
            elif 1.1*min(NSGA2_fit2[step], MDDQN_fit2[step]) >= u > min(NSGA2_fit2[step], MDDQN_fit2[step]):
                rt += -1
            elif u > 1.1*min(NSGA2_fit2[step], MDDQN_fit2[step]):
                rt += -3
            else:
                rt += 0

            a = sum(self.yunshu_AGV)
            if a < 0.9*min(NSGA2_fit3[step], MDDQN_fit3[step]):
                rt += 3
            elif 0.9*min(NSGA2_fit3[step], MDDQN_fit3[step]) <= a < min(NSGA2_fit3[step], MDDQN_fit3[step]):
                rt += 1
            elif 1.1*min(NSGA2_fit3[step], MDDQN_fit3[step]) >= a > min(NSGA2_fit3[step], MDDQN_fit3[step]):
                rt += -1
            elif a > 1.1*min(NSGA2_fit3[step], MDDQN_fit3[step]):
                rt += -3
            else:
                rt += 0
            reward = rt
        return reward
    def NDS(self, state_ed, state_now):
        v = 0
        dom_less = 0  # state_ed 的指标小于 state_now 的数量
        dom_equal = 0  # state_ed 和 state_now 的指标值相等的数量
        dom_more = 0  # state_ed 的指标大于 state_now 的数量
        # 遍历两个适应度维度，进行比较
        for k in range(5,8):
            if state_ed[k] > state_now[k]:
                dom_more += 1
            elif state_ed[k] == state_now[k]:
                dom_equal += 1
            else:
                dom_less += 1
        # 根据支配关系判断结果赋值给 v
        if dom_less == 0 and dom_equal != 3:
            v = 2  # state_now 支配 state_ed，当前状态更好
        if dom_more == 0 and dom_equal != 3:
            v = 1  # state_ed 支配 state_now，上一状态更好
        return v
    def Makespan_max(self):# 选择预计作业完工时间最长的作业。
        dl = min(self.Makespan_pre)
        for i in range(1, self.J_num + 1):
            j_sta, j_mac = self.find_element_index(self.MNN, self.Jobs[i].cur_site)  # 工件当前所在机器的阶段（0、1、2...state）和机器序号（阶段内）
            if j_sta == self.State:
                self.Done.append(i)
                self.Makespan_pre[i] = self.Jobs[i].last_ot
                continue
            elif j_sta == 0:
                j_arr, j_bat = self.find_element_index(self.JNN, i)  # 工件种类序号 批次号
                pts_sum = self.shengyu_jiagong_banyun(i, j_sta, self.ST_i[j_arr] * self.JN_ip[j_arr][j_bat])  # 剩余加工时间+剩余搬运时间+分拣时间
                if self.Jobs[i].last_ot + pts_sum > dl:  # 预计完工时间=当前时间+剩余加工时间+剩余搬运时间+分拣时间
                    dl = self.Jobs[i].last_ot + pts_sum
                    Job_i = i
            elif j_sta > 0 and j_sta < self.State:
                pts_sum = self.shengyu_jiagong_banyun(i, j_sta, 0)  # 剩余加工时间+剩余搬运时间
                if self.Jobs[i].last_ot + pts_sum > dl:  # 预计完工时间=当前时间+剩余加工时间+剩余搬运时间
                    dl = self.Jobs[i].last_ot + pts_sum
                    Job_i = i
        Ji = self.Jobs[Job_i]  # 当前批次工件类
        return Job_i, Ji
    def SYJGSJ_Max(self):# 选择剩余加工+搬运时间最长的作业
        dl = -1
        for i in range(1, self.J_num + 1):
            j_sta, j_mac = self.find_element_index(self.MNN, self.Jobs[i].cur_site)  # 工件当前所在机器的阶段（0、1、2...state）和机器序号（阶段内）
            if j_sta == self.State:
                self.Done.append(i)
                self.Makespan_pre[i] = self.Jobs[i].last_ot
                continue
            else:#if j_sta == 0:
                j_arr, j_bat = self.find_element_index(self.JNN, i)  # 工件种类序号 批次号
                if j_sta == 0:
                    pts_sum = self.shengyu_jiagong_banyun(i, j_sta, self.ST_i[j_arr] * self.JN_ip[j_arr][j_bat])  # 剩余加工时间+剩余搬运时间+分拣时间
                else:
                    pts_sum = self.shengyu_jiagong_banyun(i, j_sta, 0)  # 剩余加工时间+剩余搬运时间
                # pts_sum = 0
                # for s in range(j_sta + 1, self.State + 1):
                #     pt_sum = 0
                #     for j in range(len(self.AL_jk[s])):
                #         pt_sum += self.PT[s][j][i]
                #     pts_sum += pt_sum / len(self.AL_jk[s])
                if pts_sum > dl:
                    dl = pts_sum
                    Job_i = i
            # elif j_sta > 0 and j_sta < self.State:
            #     pts_sum = self.shengyu_jiagong_banyun(i, j_sta, 0)  # 剩余加工时间+剩余搬运时间
            #     if pts_sum > dl:
            #         dl = pts_sum
            #         Job_i = i
            # print(i, j_sta, dl, pts_sum,Job_i)
        Ji = self.Jobs[Job_i]  # 当前批次工件类
        return Job_i, Ji
    def FIFO(self):# 根据FIFO调度规则，选择最先完成前一阶段的子批作为下一个操作
        dl = [99999]
        for i in range(1, self.J_num + 1):
            j_sta, j_mac = self.find_element_index(self.MNN, self.Jobs[i].cur_site)  # 工件当前所在机器的阶段（0、1、2...state）和机器序号（阶段内）
            if j_sta == self.State:
                self.Done.append(i)
                self.Makespan_pre[i] = self.Jobs[i].last_ot
                dl.append(99999)
            elif j_sta < self.State:
                dl.append(self.Jobs[i].last_ot)
        if dl.count(min(dl)) > 1:  # 当最先完成前一阶段的子批数为2个及以上，则从中随机选择一个
            Job_i = random.choice(self.find_all_min_indices(dl))  # 从中随机选择一个
        else:  # 只有一个最先完成前一阶段的子批
            Job_i = dl.index(min(dl))  # 就选它
        Ji = self.Jobs[Job_i]  # 当前批次工件类
        return Job_i, Ji
    def ZZKY_Mac(self, Job_i, Ji):# 将选择的操作分配到最早可用的机器上，若存在多个可选机器，则选择运输距离最短的机器。
        j_sta, j_mac = self.find_element_index(self.MNN, Ji.cur_site)  # 工件当前所在机器的阶段（0、1、2...state）和机器序号（阶段内）
        j_arr, j_bat = self.find_element_index(self.JNN, Job_i)  # 工件种类序号 批次号
        m_sta = j_sta + 1
        last_Md = [self.AL_jk[m_sta][M_i].last_ot for M_i in range(self.M_j[m_sta])]  # 下一阶段机器的完成时间 m_sta阶段[机器0完成时间,机器1完成时间,。。]
        O_et = [0 for i in range(self.M_j[m_sta])]  # m_sta阶段[工件i在机器0开始加工时间,工件i在机器1开始加工时间,。。]
        if j_sta == 0:  # 工件处于第一阶段前
            for i in range(self.M_j[m_sta]):
                O_et[i] = max(last_Md[i], self.ST_i[j_arr] * self.JN_ip[j_arr][j_bat])  # max(当前阶段机器的完成时间,工件i个数*该种分拣时间)
            if O_et.count(min(O_et)) > 1:  # 当结束加工时间最小机器数为2个及以上，则从中随机选择一个机器
                m_id = random.choice(self.find_all_min_indices(O_et))  # 从中随机选择一个机器
            else:  # 只有一台机器结束加工时间最短
                m_id = O_et.index(min(O_et))  # 就选它
        else:
            for i in range(self.M_j[m_sta]):
                O_et[i] = max(last_Md[i], Ji.last_ot + self.D_pq[Ji.cur_site][self.MNN[m_sta][i]] / self.V_r)  # max(当前阶段机器的完成时间,工件i结束当前工序并被AGV送到下一机器时间)
            if O_et.count(min(O_et)) > 1:  # 当结束加工时间最小机器数为2个及以上，则选择运输距离最短的机器
                mac_min = self.find_all_indices(O_et, min(O_et))
                TT_min = self.D_pq[Ji.cur_site][self.MNN[m_sta][mac_min[0]]]
                m_id = mac_min[0]
                for i in mac_min:
                    if self.D_pq[Ji.cur_site][self.MNN[m_sta][i]] < TT_min:
                        TT_min = self.D_pq[Ji.cur_site][self.MNN[m_sta][i]]
                        m_id = i
            else:  # 只有一台机器结束加工时间最短
                m_id = O_et.index(min(O_et))  # 就选它
        return j_sta, m_sta, m_id
    def Idle_Max(self, Job_i, Ji):# 将选择的操作分配到下一操作阶段空闲时间最长的机器上。
        j_sta, j_mac = self.find_element_index(self.MNN, Ji.cur_site)  # 工件当前所在机器的阶段（0、1、2...state）和机器序号（阶段内）
        m_sta = j_sta + 1
        Idle_time = self.Idle_time_jisuan(m_sta)
        if Idle_time.count(max(Idle_time)) > 1:  # 当空闲时间最长机器数为2个及以上，则从中随机选择一个机器
            m_id = random.choice(self.find_all_max_indices(Idle_time))  # 从中随机选择一个机器
        else:  # 只有一台机器空闲时间最长
            m_id = Idle_time.index(max(Idle_time))  # 就选它
        return j_sta, m_sta, m_id
    def PT_Min(self, Job_i, Ji):# 将选择的操作分配到下一操作加工时间最短的机器上。
        j_sta, j_mac = self.find_element_index(self.MNN, Ji.cur_site)  # 工件当前所在机器的阶段（0、1、2...state）和机器序号（阶段内）
        m_sta = j_sta + 1
        jiagong_time = 99999
        for i in range(len(self.AL_jk[m_sta])):
            if self.PT[m_sta][i][Job_i] < jiagong_time:
                jiagong_time = self.PT[m_sta][i][Job_i]
                m_id = i
        return j_sta, m_sta, m_id
    def Idle_time_jisuan(self, m_sta):
        Idle_time = []
        Mac_start, Mac_end = [], []
        for j in range(len(self.AL_jk[m_sta])):
            if len(self.AL_jk[m_sta][j].T)>0:
                Mac_start.append(self.AL_jk[m_sta][j].start[0])
                Mac_end.append(self.AL_jk[m_sta][j].end[-1])
            # else:
                # Mac_start.append(0)
                # Mac_end.append(0)
        for j in range(len(self.AL_jk[m_sta])):
            if sum(Mac_start) == 0:
                Idle_time.append(0)
            else:
                Idle_time.append(round(max(Mac_end) - min(Mac_start) - sum(self.AL_jk[m_sta][j].T),2))
        return Idle_time

    def ZZKY_AGV(self, Ji, m_sta, m_id):# 选择最早可用的AGV，当早可用AGV数为2个及以上，则从中选择一个最近的
        J_end, J_site, on_m = Ji.get_info()  # 加工结束时间，当前位置，各工序加工机器编号（工序内）
        best_agv = None
        min_tf = 99999
        best_s, best_e, t1, t2 = None, None, None, None
        AGV_end = [99999]
        for agv in range(1, len(self.AGVs)):
            AGV_end.append(self.AGVs[agv].end)  # AGV结束上一任务时间
        if AGV_end.count(min(AGV_end)) > 1:  # 当早可用AGV数为2个及以上，则从中选择一个最近的
            for agv in self.find_all_min_indices(AGV_end):
                j_sta, j_mac = self.find_element_index(self.MNN, J_site)  # 工件当前所在机器的阶段（0、1、2...state）和机器序号（阶段内）
                trans1 = ((self.AGVs[agv].cur_site2[0]-(j_sta-1)*60)**2+(self.AGVs[agv].cur_site2[1]-j_mac*30)**2)**0.5
                # trans1 = self.D_pq[self.AGVs[agv].cur_site][J_site]  # AGV到工件当前位置距离
                # trans2 = self.D_pq[J_site][self.MNN[m_sta][m_id]]  # 工件到下一工序机器距离
                # start, end = self.AGVs[agv].ST(J_end[-1], trans1 / self.V_r, trans2 / self.V_r)  # start：AGV空载开始时间点，end：负载结束时间点
                if trans1 < min_tf:
                    # best_s, best_e, t1, t2 = start, end, trans1 / self.V_r, trans2 / self.V_r
                    best_agv = self.AGVs[agv]
                    min_tf = trans1
            # best_agv.update(best_s, best_e, t1, t2, J_site, self.MNN[m_sta][m_id], Ji.idx)  # 选出最优的AGV
        else:# 只有一台AGV最早可用
            best_agv = self.AGVs[AGV_end.index(min(AGV_end))]
            # agv = AGV_end.index(min(AGV_end))  # 就选它
            # trans1 = self.D_pq[self.AGVs[agv].cur_site][J_site]  # AGV到工件当前位置距离
            # trans2 = self.D_pq[J_site][self.MNN[m_sta][m_id]]  # 工件到下一工序机器距离
            # start, end = self.AGVs[agv].ST(J_end[-1], trans1 / self.V_r, trans2 / self.V_r)  # start：AGV空载开始时间点，end：负载结束时间点
            # if trans1 < min_tf:
            #     best_s, best_e, t1, t2 = start, end, trans1 / self.V_r, trans2 / self.V_r
            #     best_agv = self.AGVs[agv]
            #     min_tf = trans1
            # best_agv.update(best_s, best_e, t1, t2, J_site, self.MNN[m_sta][m_id], Ji.idx)  # 选出最优的AGV
        return best_agv
    def FAST_AGV(self, Ji, m_sta, m_id):# 选择使工件最快开始运输的AGV
        J_end, J_site, on_m = Ji.get_info()  # 加工结束时间，当前位置，各工序加工机器编号（工序内）
        best_agv = None
        min_tf = 99999
        best_s, best_e, t1, t2 = None, None, None, None
        for agv in range(1, len(self.AGVs)):
            j_sta, j_mac = self.find_element_index(self.MNN, J_site)  # 工件当前所在机器的阶段（0、1、2...state）和机器序号（阶段内）
            trans1 = ((self.AGVs[agv].cur_site2[0] - (j_sta - 1) * 60) ** 2 + (self.AGVs[agv].cur_site2[1] - j_mac * 30) ** 2) ** 0.5
            # trans1 = self.D_pq[self.AGVs[agv].cur_site][J_site]  # AGV到工件当前位置距离
            trans2 = self.D_pq[J_site][self.MNN[m_sta][m_id]]  # 工件到下一工序机器距离
            start, end = self.AGVs[agv].ST(J_end[-1], trans1 / self.V_r, trans2 / self.V_r)  # start：AGV空载开始时间点，end：负载结束时间点
            if end < min_tf:
                best_s, best_e, t1, t2 = start, end, trans1 / self.V_r, trans2 / self.V_r
                best_agv = self.AGVs[agv]
                min_tf = best_e
        # best_agv.update(best_s, best_e, t1, t2, J_site, self.MNN[m_sta][m_id], Ji.idx)  # 选出最优的AGV
        return best_agv
    def Junheng_AGV(self, Ji, m_sta, m_id):# 选择总运输距离最短的AGV
        J_end, J_site, on_m = Ji.get_info()  # 加工结束时间，当前位置，各工序加工机器编号（工序内）
        best_agv = None
        min_tf = 99999
        best_s, best_e, t1, t2 = None, None, None, None
        for agv in range(1, len(self.AGVs)):
            # trans1 = self.D_pq[self.AGVs[agv].cur_site][J_site]  # AGV到工件当前位置距离
            # trans2 = self.D_pq[J_site][self.MNN[m_sta][m_id]]  # 工件到下一工序机器距离
            # start, end = self.AGVs[agv].ST(J_end[-1], trans1 / self.V_r, trans2 / self.V_r)  # start：AGV空载开始时间点，end：负载结束时间点
            S_sum = sum(self.AGVs[agv].T)
            if S_sum < min_tf:
                # best_s, best_e, t1, t2 = start, end, trans1 / self.V_r, trans2 / self.V_r
                best_agv = self.AGVs[agv]
                min_tf = S_sum
        # best_agv.update(best_s, best_e, t1, t2, J_site, self.MNN[m_sta][m_id], Ji.idx)  # 选出最优的AGV
        return best_agv
    def rule1(self):#规则1 选择预计作业完工时间最大的作业，然后将选择的操作分配到最早可用的机器上，若存在多个可选机器，则选择运输距离最短的机器。
        Job_i, Ji = self.Makespan_max()
        j_sta, m_sta, m_id = self.ZZKY_Mac(Job_i, Ji)
        if j_sta == 0:
            return Job_i, self.MNN[m_sta][m_id], 0  # 工件分拣时不调度AGV，返回多余的AGV序号
        else:
            best_agv = self.ZZKY_AGV(Ji, m_sta, m_id)
            return Job_i, self.MNN[m_sta][m_id], best_agv.idx

    def rule2(self):#规则2 选择剩余加工时间最短的作业，将选择的操作分配到最早可用的机器上，若存在多个可选机器，则选择运输距离最短的机器。
        Job_i, Ji = self.SYJGSJ_Max()
        j_sta, m_sta, m_id = self.ZZKY_Mac(Job_i, Ji)
        if j_sta == 0:
            return Job_i, self.MNN[m_sta][m_id], 0  # 工件分拣时不调度AGV，返回多余的AGV序号
        else:
            best_agv = self.ZZKY_AGV(Ji, m_sta, m_id)
            return Job_i, self.MNN[m_sta][m_id], best_agv.idx

    def rule3(self):#规则3 选择预计作业完工时间最大的作业，然后将选择的操作分配到下一操作阶段平均利用率最低的机器上。
        Job_i, Ji = self.Makespan_max()
        j_sta, m_sta, m_id = self.Idle_Max(Job_i, Ji)
        if j_sta == 0:
            return Job_i, self.MNN[m_sta][m_id], 0  # 工件分拣时不调度AGV，返回多余的AGV序号
        else:
            best_agv = self.ZZKY_AGV(Ji, m_sta, m_id)
            return Job_i, self.MNN[m_sta][m_id], best_agv.idx

    def rule4(self):#规则4 选择剩余加工时间最短的作业，然后将选择的操作分配到下一操作阶段平均利用率最低的机器上。
        Job_i, Ji = self.SYJGSJ_Max()
        j_sta, m_sta, m_id = self.Idle_Max(Job_i, Ji)
        if j_sta == 0:
            return Job_i, self.MNN[m_sta][m_id], 0  # 工件分拣时不调度AGV，返回多余的AGV序号
        else:
            best_agv = self.ZZKY_AGV(Ji, m_sta, m_id)
            return Job_i, self.MNN[m_sta][m_id], best_agv.idx

    def rule5(self):#规则5 根据FIFO调度规则，选择最先完成前一阶段的子批作为下一个操作，然后将选择的操作分配到下一操作阶段最早可用的机器上，若存在多个可选机器，则选择运输距离最短的机器。
        Job_i, Ji = self.FIFO()
        j_sta, m_sta, m_id = self.ZZKY_Mac(Job_i, Ji)
        if j_sta == 0:
            return Job_i, self.MNN[m_sta][m_id], 0  # 工件分拣时不调度AGV，返回多余的AGV序号
        else:
            best_agv = self.ZZKY_AGV(Ji, m_sta, m_id)
            return Job_i, self.MNN[m_sta][m_id], best_agv.idx

    def rule6(self):#规则6 根据FIFO调度规则，选择最先完成前一阶段的子批作为下一个操作，然后将选择的操作分配到下一操作阶段平均利用率最低的机器上。
        Job_i, Ji = self.FIFO()
        j_sta, m_sta, m_id = self.Idle_Max(Job_i, Ji)
        if j_sta == 0:
            return Job_i, self.MNN[m_sta][m_id], 0  # 工件分拣时不调度AGV，返回多余的AGV序号
        else:
            best_agv = self.ZZKY_AGV(Ji, m_sta, m_id)
            return Job_i, self.MNN[m_sta][m_id], best_agv.idx
    def rule7(self):#规则1 选择预计作业完工时间最大的作业，然后将选择的操作分配到最早可用的机器上，若存在多个可选机器，则选择运输距离最短的机器。
        Job_i, Ji = self.Makespan_max()
        j_sta, m_sta, m_id = self.ZZKY_Mac(Job_i, Ji)
        if j_sta == 0:
            return Job_i, self.MNN[m_sta][m_id], 0  # 工件分拣时不调度AGV，返回多余的AGV序号
        else:
            best_agv = self.FAST_AGV(Ji, m_sta, m_id)
            return Job_i, self.MNN[m_sta][m_id], best_agv.idx
    def rule8(self):#规则2 选择剩余加工时间最短的作业，将选择的操作分配到最早可用的机器上，若存在多个可选机器，则选择运输距离最短的机器。
        Job_i, Ji = self.SYJGSJ_Max()
        j_sta, m_sta, m_id = self.ZZKY_Mac(Job_i, Ji)
        if j_sta == 0:
            return Job_i, self.MNN[m_sta][m_id], 0  # 工件分拣时不调度AGV，返回多余的AGV序号
        else:
            best_agv = self.FAST_AGV(Ji, m_sta, m_id)
            return Job_i, self.MNN[m_sta][m_id], best_agv.idx
    def rule9(self):#规则3 选择预计作业完工时间最大的作业，将选择的操作分配到下一操作阶段空闲时间最大的机器上。
        Job_i, Ji = self.Makespan_max()
        j_sta, m_sta, m_id = self.Idle_Max(Job_i, Ji)
        if j_sta == 0:
            return Job_i, self.MNN[m_sta][m_id], 0  # 工件分拣时不调度AGV，返回多余的AGV序号
        else:
            best_agv = self.FAST_AGV(Ji, m_sta, m_id)
            return Job_i, self.MNN[m_sta][m_id], best_agv.idx
    def rule10(self):#规则4 选择剩余加工时间最短的作业，将选择的操作分配到下一操作阶段空闲时间最大的机器上。
        Job_i, Ji = self.SYJGSJ_Max()
        j_sta, m_sta, m_id = self.Idle_Max(Job_i, Ji)
        if j_sta == 0:
            return Job_i, self.MNN[m_sta][m_id], 0  # 工件分拣时不调度AGV，返回多余的AGV序号
        else:
            best_agv = self.FAST_AGV(Ji, m_sta, m_id)
            return Job_i, self.MNN[m_sta][m_id], best_agv.idx
    def rule11(self):#规则5 根据FIFO调度规则，选择最先完成前一阶段的子批作为下一个操作，然后将选择的操作分配到下一操作阶段最早可用的机器上，若存在多个可选机器，则选择运输距离最短的机器。
        Job_i, Ji = self.FIFO()
        j_sta, m_sta, m_id = self.ZZKY_Mac(Job_i, Ji)
        if j_sta == 0:
            return Job_i, self.MNN[m_sta][m_id], 0  # 工件分拣时不调度AGV，返回多余的AGV序号
        else:
            best_agv = self.FAST_AGV(Ji, m_sta, m_id)
            return Job_i, self.MNN[m_sta][m_id], best_agv.idx
    def rule12(self):#规则6 根据FIFO调度规则，选择最先完成前一阶段的子批作为下一个操作，将选择的操作分配到下一操作阶段空闲时间最大的机器上。
        Job_i, Ji = self.FIFO()
        j_sta, m_sta, m_id = self.Idle_Max(Job_i, Ji)
        if j_sta == 0:
            return Job_i, self.MNN[m_sta][m_id], 0  # 工件分拣时不调度AGV，返回多余的AGV序号
        else:
            best_agv = self.FAST_AGV(Ji, m_sta, m_id)
            return Job_i, self.MNN[m_sta][m_id], best_agv.idx
    def rule13(self):#规则1 选择预计作业完工时间最大的作业，然后将选择的操作分配到最早可用的机器上，若存在多个可选机器，则选择运输距离最短的机器。
        Job_i, Ji = self.Makespan_max()
        j_sta, m_sta, m_id = self.ZZKY_Mac(Job_i, Ji)
        if j_sta == 0:
            return Job_i, self.MNN[m_sta][m_id], 0  # 工件分拣时不调度AGV，返回多余的AGV序号
        else:
            best_agv = self.Junheng_AGV(Ji, m_sta, m_id)
            return Job_i, self.MNN[m_sta][m_id], best_agv.idx
    def rule14(self):#规则2 选择剩余加工时间最短的作业，将选择的操作分配到最早可用的机器上，若存在多个可选机器，则选择运输距离最短的机器。
        Job_i, Ji = self.SYJGSJ_Max()
        j_sta, m_sta, m_id = self.ZZKY_Mac(Job_i, Ji)
        if j_sta == 0:
            return Job_i, self.MNN[m_sta][m_id], 0  # 工件分拣时不调度AGV，返回多余的AGV序号
        else:
            best_agv = self.Junheng_AGV(Ji, m_sta, m_id)
            return Job_i, self.MNN[m_sta][m_id], best_agv.idx
    def rule15(self):#规则3 选择预计作业完工时间最大的作业，将选择的操作分配到下一操作阶段空闲时间最大的机器上。
        Job_i, Ji = self.Makespan_max()
        j_sta, m_sta, m_id = self.Idle_Max(Job_i, Ji)
        if j_sta == 0:
            return Job_i, self.MNN[m_sta][m_id], 0  # 工件分拣时不调度AGV，返回多余的AGV序号
        else:
            best_agv = self.Junheng_AGV(Ji, m_sta, m_id)
            return Job_i, self.MNN[m_sta][m_id], best_agv.idx
    def rule16(self):#规则4 选择剩余加工时间最短的作业，将选择的操作分配到下一操作阶段空闲时间最大的机器上。
        Job_i, Ji = self.SYJGSJ_Max()
        j_sta, m_sta, m_id = self.Idle_Max(Job_i, Ji)
        if j_sta == 0:
            return Job_i, self.MNN[m_sta][m_id], 0  # 工件分拣时不调度AGV，返回多余的AGV序号
        else:
            best_agv = self.Junheng_AGV(Ji, m_sta, m_id)
            return Job_i, self.MNN[m_sta][m_id], best_agv.idx
    def rule17(self):#规则5 根据FIFO调度规则，选择最先完成前一阶段的子批作为下一个操作，然后将选择的操作分配到下一操作阶段最早可用的机器上，若存在多个可选机器，则选择运输距离最短的机器。
        Job_i, Ji = self.FIFO()
        j_sta, m_sta, m_id = self.ZZKY_Mac(Job_i, Ji)
        if j_sta == 0:
            return Job_i, self.MNN[m_sta][m_id], 0  # 工件分拣时不调度AGV，返回多余的AGV序号
        else:
            best_agv = self.Junheng_AGV(Ji, m_sta, m_id)
            return Job_i, self.MNN[m_sta][m_id], best_agv.idx
    def rule18(self):#规则6 根据FIFO调度规则，选择最先完成前一阶段的子批作为下一个操作，将选择的操作分配到下一操作阶段空闲时间最大的机器上。
        Job_i, Ji = self.FIFO()
        j_sta, m_sta, m_id = self.Idle_Max(Job_i, Ji)
        if j_sta == 0:
            return Job_i, self.MNN[m_sta][m_id], 0  # 工件分拣时不调度AGV，返回多余的AGV序号
        else:
            best_agv = self.Junheng_AGV(Ji, m_sta, m_id)
            return Job_i, self.MNN[m_sta][m_id], best_agv.idx
    def rule19(self):#规则3 选择预计作业完工时间最大的作业，然后将选择的操作分配到下一操作阶段平均利用率最低的机器上。
        Job_i, Ji = self.Makespan_max()
        j_sta, m_sta, m_id = self.PT_Min(Job_i, Ji)
        if j_sta == 0:
            return Job_i, self.MNN[m_sta][m_id], 0  # 工件分拣时不调度AGV，返回多余的AGV序号
        else:
            best_agv = self.ZZKY_AGV(Ji, m_sta, m_id)
            return Job_i, self.MNN[m_sta][m_id], best_agv.idx

    def rule20(self):#规则4 选择剩余加工时间最短的作业，然后将选择的操作分配到下一操作阶段平均利用率最低的机器上。
        Job_i, Ji = self.SYJGSJ_Max()
        j_sta, m_sta, m_id = self.PT_Min(Job_i, Ji)
        if j_sta == 0:
            return Job_i, self.MNN[m_sta][m_id], 0  # 工件分拣时不调度AGV，返回多余的AGV序号
        else:
            best_agv = self.ZZKY_AGV(Ji, m_sta, m_id)
            return Job_i, self.MNN[m_sta][m_id], best_agv.idx

    def rule21(self):#规则5 根据FIFO调度规则，选择最先完成前一阶段的子批作为下一个操作，然后将选择的操作分配到下一操作阶段最早可用的机器上，若存在多个可选机器，则选择运输距离最短的机器。
        Job_i, Ji = self.FIFO()
        j_sta, m_sta, m_id = self.PT_Min(Job_i, Ji)
        if j_sta == 0:
            return Job_i, self.MNN[m_sta][m_id], 0  # 工件分拣时不调度AGV，返回多余的AGV序号
        else:
            best_agv = self.ZZKY_AGV(Ji, m_sta, m_id)
            return Job_i, self.MNN[m_sta][m_id], best_agv.idx
    def rule22(self):#规则3 选择预计作业完工时间最大的作业，将选择的操作分配到下一操作阶段空闲时间最大的机器上。
        Job_i, Ji = self.Makespan_max()
        j_sta, m_sta, m_id = self.PT_Min(Job_i, Ji)
        if j_sta == 0:
            return Job_i, self.MNN[m_sta][m_id], 0  # 工件分拣时不调度AGV，返回多余的AGV序号
        else:
            best_agv = self.FAST_AGV(Ji, m_sta, m_id)
            return Job_i, self.MNN[m_sta][m_id], best_agv.idx
    def rule23(self):#规则4 选择剩余加工时间最短的作业，将选择的操作分配到下一操作阶段空闲时间最大的机器上。
        Job_i, Ji = self.SYJGSJ_Max()
        j_sta, m_sta, m_id = self.PT_Min(Job_i, Ji)
        if j_sta == 0:
            return Job_i, self.MNN[m_sta][m_id], 0  # 工件分拣时不调度AGV，返回多余的AGV序号
        else:
            best_agv = self.FAST_AGV(Ji, m_sta, m_id)
            return Job_i, self.MNN[m_sta][m_id], best_agv.idx
    def rule24(self):#规则5 根据FIFO调度规则，选择最先完成前一阶段的子批作为下一个操作，然后将选择的操作分配到下一操作阶段最早可用的机器上，若存在多个可选机器，则选择运输距离最短的机器。
        Job_i, Ji = self.FIFO()
        j_sta, m_sta, m_id = self.PT_Min(Job_i, Ji)
        if j_sta == 0:
            return Job_i, self.MNN[m_sta][m_id], 0  # 工件分拣时不调度AGV，返回多余的AGV序号
        else:
            best_agv = self.FAST_AGV(Ji, m_sta, m_id)
            return Job_i, self.MNN[m_sta][m_id], best_agv.idx
    def rule25(self):#规则3 选择预计作业完工时间最大的作业，将选择的操作分配到下一操作阶段空闲时间最大的机器上。
        Job_i, Ji = self.Makespan_max()
        j_sta, m_sta, m_id = self.PT_Min(Job_i, Ji)
        if j_sta == 0:
            return Job_i, self.MNN[m_sta][m_id], 0  # 工件分拣时不调度AGV，返回多余的AGV序号
        else:
            best_agv = self.Junheng_AGV(Ji, m_sta, m_id)
            return Job_i, self.MNN[m_sta][m_id], best_agv.idx
    def rule26(self):#规则4 选择剩余加工时间最短的作业，将选择的操作分配到下一操作阶段空闲时间最大的机器上。
        Job_i, Ji = self.SYJGSJ_Max()
        j_sta, m_sta, m_id = self.PT_Min(Job_i, Ji)
        if j_sta == 0:
            return Job_i, self.MNN[m_sta][m_id], 0  # 工件分拣时不调度AGV，返回多余的AGV序号
        else:
            best_agv = self.Junheng_AGV(Ji, m_sta, m_id)
            return Job_i, self.MNN[m_sta][m_id], best_agv.idx
    def rule27(self):#规则5 根据FIFO调度规则，选择最先完成前一阶段的子批作为下一个操作，然后将选择的操作分配到下一操作阶段最早可用的机器上，若存在多个可选机器，则选择运输距离最短的机器。
        Job_i, Ji = self.FIFO()
        j_sta, m_sta, m_id = self.PT_Min(Job_i, Ji)
        if j_sta == 0:
            return Job_i, self.MNN[m_sta][m_id], 0  # 工件分拣时不调度AGV，返回多余的AGV序号
        else:
            best_agv = self.Junheng_AGV(Ji, m_sta, m_id)
            return Job_i, self.MNN[m_sta][m_id], best_agv.idx
    #解码
    def Stage_Decode(self,CHS):#CHS：初始种群
        for Job_i in CHS:#i：工件序号
            Ji = self.Jobs[Job_i]  # 当前批次工件类
            last_od=Ji.last_ot#工件i完成时间
            #机器调度
            j_sta, m_sta, m_id = self.ZZKY_Mac(Job_i, Ji)
            J_m = self.AL_jk[m_sta][m_id]  # 下一工序机器
            #AGV调度
            if j_sta == 0:
                agv = 0
            else:
                best_agv = self.ZZKY_AGV(Ji, m_sta, m_id)
                agv = best_agv.idx
            self.scheduling(Job_i, self.MNN[m_sta][m_id], agv)
        obs = self.Features()  # 更新状态特征
        # 1 工件完工时间
        self.fitness1 = float(obs[5])
        # 2 机器空闲时间
        self.fitness2 = float(obs[6])
        # 3 AGV运输距离
        self.fitness3 = float(obs[7])
    def Stage_Decode2(self,CHS):#CHS：初始种群
        fitness1,fitness2,fitness3 = [], [], []
        for Job_i in CHS:#i：工件序号
            Ji = self.Jobs[Job_i]  # 当前批次工件类
            last_od=Ji.last_ot#工件i完成时间
            #机器调度
            j_sta, m_sta, m_id = self.ZZKY_Mac(Job_i, Ji)
            J_m = self.AL_jk[m_sta][m_id]  # 下一工序机器
            #AGV调度
            if j_sta == 0:
                agv = 0
            else:
                best_agv = self.ZZKY_AGV(Ji, m_sta, m_id)
                agv = best_agv.idx
            self.scheduling(Job_i, self.MNN[m_sta][m_id], agv)
            obs = self.Features()  # 更新状态特征
            # 2 机器空闲时间
            fitness2.append(float(obs[6]))
            # 3 AGV运输距离
            fitness3.append(float(obs[7]))
        for Job_i in CHS:
            # 1 工件完工时间
            fitness1.append(float(obs[5]))
        return fitness1,fitness2,fitness3
    def Stage_Decode3(self,CHS1,CHS2,CHS3):#CHS：初始种群
        T_r = [0 for _ in range(self.J_num+1)]
        i = 0
        for Job_i in CHS1:#i：工件序号
            T_r[Job_i] = T_r[Job_i] + 1  # 为工件序号计数，算当前阶段
            Ji = self.Jobs[Job_i]  # 当前批次工件类
            #机器调度
            j_sta = T_r[Job_i] - 1
            m_sta = j_sta + 1
            m_id = CHS2[j_sta * self.J_num + self.count_element([T_r],m_sta)-1]
            # j_sta, m_sta, m_id = self.ZZKY_Mac(Job_i, Ji)
            #AGV调度
            agv = 0 if j_sta == 0 else CHS3[i]
            self.scheduling(Job_i, self.MNN[m_sta][m_id], agv)
            i = i + 1
        obs = self.Features()  # 更新状态特征
        # 1 工件完工时间 # 2 机器空闲时间 # 3 AGV运输距离
        self.fitness1 = float(obs[5]);self.fitness2 = float(obs[6]);self.fitness3 = float(obs[7])

    def Stage_Decode4(self,CHS1,CHS2,CHS3):#CHS：初始种群
        fitness1, fitness2, fitness3 = [], [], []
        T_r = [0 for _ in range(self.J_num+1)]
        i = 0
        for Job_i in CHS1:#i：工件序号
            T_r[Job_i] = T_r[Job_i] + 1  # 为工件序号计数，算当前阶段
            Ji = self.Jobs[Job_i]  # 当前批次工件类
            #机器调度
            j_sta = T_r[Job_i] - 1
            m_sta = j_sta + 1
            m_id = CHS2[j_sta * self.J_num + self.count_element([T_r],m_sta)-1]
            # j_sta, m_sta, m_id = self.ZZKY_Mac(Job_i, Ji)
            #AGV调度
            agv = 0 if j_sta == 0 else CHS3[i]
            self.scheduling(Job_i, self.MNN[m_sta][m_id], agv)
            i = i + 1
            obs = self.Features()  # 更新状态特征
            # 1 工件完工时间 # 2 机器空闲时间 # 3 AGV运输距离
            fitness1.append(float(obs[5]));fitness2.append(float(obs[6]));fitness3.append(float(obs[7]))
        return fitness1, fitness2, fitness3
    def reStage_Decode3(self,CHS1,CHS2,CHS3):#CHS：初始种群
        T_r = [0 for _ in range(self.J_num+1)]
        i = 0
        for Job_i in CHS1:#i：工件序号
            T_r[Job_i] = T_r[Job_i] + 1  # 为工件序号计数，算当前阶段
            Ji = self.Jobs[Job_i]  # 当前批次工件类
            #机器调度
            j_sta = T_r[Job_i] - 1
            # m_sta, m_id = self.find_element_index(self.MNN, CHS2[j_sta * self.J_num + self.count_element([T_r],j_sta + 1)-1])
            m_sta = j_sta + 1
            m_id = CHS2[j_sta * self.J_num + self.count_element([T_r],m_sta)-1]
            # j_sta, m_sta, m_id = self.ZZKY_Mac(Job_i, Ji)
            #AGV调度
            agv = 0 if j_sta == 0 else CHS3[i]
            self.rescheduling(Job_i, self.MNN[m_sta][m_id], agv)
            i = i + 1
        obs = self.Features()  # 更新状态特征
        # 1 工件完工时间 # 2 机器空闲时间 # 3 AGV运输距离
        self.fitness1 = float(obs[5]);self.fitness2 = float(obs[6]);self.fitness3 = float(obs[7])

    def reStage_Decode4(self,CHS1,CHS2,CHS3):#CHS：初始种群
        fitness1, fitness2, fitness3 = [], [], []
        T_r = [0 for _ in range(self.J_num+1)]
        i = 0
        for Job_i in CHS1:#i：工件序号
            T_r[Job_i] = T_r[Job_i] + 1  # 为工件序号计数，算当前阶段
            Ji = self.Jobs[Job_i]  # 当前批次工件类
            #机器调度
            j_sta = T_r[Job_i] - 1
            # m_sta, m_id = self.find_element_index(self.MNN, CHS2[j_sta * self.J_num + self.count_element([T_r],j_sta + 1)-1])
            m_sta = j_sta + 1
            m_id = CHS2[j_sta * self.J_num + self.count_element([T_r],m_sta)-1]
            # j_sta, m_sta, m_id = self.ZZKY_Mac(Job_i, Ji)
            #AGV调度
            agv = 0 if j_sta == 0 else CHS3[i]
            self.rescheduling(Job_i, self.MNN[m_sta][m_id], agv)
            i = i + 1
            obs = self.Features()  # 更新状态特征
            # 1 工件完工时间 # 2 机器空闲时间 # 3 AGV运输距离
            fitness1.append(float(obs[5]));fitness2.append(float(obs[6]));fitness3.append(float(obs[7]))
        return fitness1, fitness2, fitness3
    def best(self):
        agv_using_time = [[]]
        agv_on = [[]]
        for k in range(1, len(self.AGVs)):
            agv_using_time.append(self.AGVs[k].using_time)
            agv_on.append(self.AGVs[k]._on)
        mac_start = [[]]
        mac_end = [[]]
        mac_on = [[]]
        mac_last_ot = [[]]
        for i in range(1, len(self.M_j)):  # 在第i个阶段
            mac_start1 = []
            mac_end1 = []
            mac_on1 = []
            mac_last_ot1 = []
            for j in range(self.M_j[i]):  # 在第j个机器
                mac_start1.append(self.AL_jk[i][j].start)
                mac_end1.append(self.AL_jk[i][j].end)
                mac_on1.append(self.AL_jk[i][j]._on)
                mac_last_ot1.append(self.AL_jk[i][j].last_ot)
            mac_start.append(mac_start1)
            mac_end.append(mac_end1)
            mac_on.append(mac_on1)
            mac_last_ot.append(mac_last_ot1)
        print(mac_start[1][0], self.AL_jk[1][0].start)
        return agv_using_time, agv_on, mac_start, mac_end, mac_on, mac_last_ot, self.UK, self.yunshu_AGV
    # 画甘特图
    def Gantt(self,suanli, v):
        fig = plt.figure()
        M = ['red', 'blue', 'yellow', 'orange', 'green', 'moccasin', 'purple', 'pink', 'navajowhite', 'Thistle', 'Magenta', 'SlateBlue', 'RoyalBlue', 'Aqua', 'floralwhite', 'ghostwhite', 'goldenrod',
             'mediumslateblue', 'navajowhite', 'navy', 'sandybrown', "cornflowerblue", "sandybrown", "mediumorchid", "cadetblue", "darkolivegreen", "steelblue", "darkslateblue", "cadetblue", "tomato", "mediumpurple"]
        M_num, t, Y_label= 0, 0, [0]
        #7 机器全部空闲时间
        Idle_sum = 0
        for i in range(1, len(self.AL_jk)):
            Idle_time = self.Idle_time_jisuan(i)
            Idle_sum += sum(Idle_time)
        # 8 AGV运输距离
        S_sum = sum(self.yunshu_AGV)
        # AGV甘特图
        for k in range(1,len(self.AGVs)):
            for m in range(len(self.AGVs[k].using_time)):
                if self.AGVs[k].using_time[m][1] - self.AGVs[k].using_time[m][0] != 0:
                    if self.AGVs[k]._on[m] != None:
                        J_variety = 0
                        for a in range(self.n+1):
                            for b in range(len(self.JNN[a])):
                                if self.JNN[a][b] == self.AGVs[k]._on[m]:
                                    J_variety = a
                        plt.barh(k, width=self.AGVs[k].using_time[m][1] - self.AGVs[k].using_time[m][0],
                                 height=0.6, left=self.AGVs[k].using_time[m][0], color=M[J_variety], edgecolor='black')
                    else:
                        plt.barh(k, width=self.AGVs[k].using_time[m][1] - self.AGVs[k].using_time[m][0],
                                 height=0.6, left=self.AGVs[k].using_time[m][0], color='white', edgecolor='black')
            Y_label.append(k)
        # 机器甘特图
        for i in range(1, len(self.M_j)):  # 在第i个阶段
            for j in range(self.M_j[i]):  # 在第j个机器
                for q in range(len(self.AL_jk[i][j].start)):  # 在该机器加工的第q批工件
                    Start_time = self.AL_jk[i][j].start[q]
                    End_time = self.AL_jk[i][j].end[q + 1]
                    Job = self.AL_jk[i][j]._on[q]
                    J_variety, J_batch = 0, 0
                    for a in range(1,self.n+1):
                        for b in range(len(self.JNN[a])):
                            if self.JNN[a][b] == Job:
                                J_variety, J_batch = a, b
                    text = "(%s,%d)" % (J_variety, J_batch + 1)
                    plt.barh(M_num + k + 1, width=End_time - Start_time, height=0.8, left=Start_time, color=M[J_variety], edgecolor='black')
                    # plt.text(x=Start_time + ((End_time - Start_time) / 2 - 0.25), y=M_num + k + 1 - 0.2, s=text, size=24, fontproperties='Times New Roman')
                if self.AL_jk[i][j].last_ot > t:
                    t = self.AL_jk[i][j].last_ot
                M_num += 1
                Y_label.append(M_num)
        Y_label.append(M_num+1)
        title = "调度甘特图：完工:{}，机器空闲:{}，AGV总运输距离:{}".format(round(t,2), round(Idle_sum,2), round(S_sum,2))
        plt.title(title, fontsize=24)
        plt.xlim(0)
        plt.yticks(np.arange(M_num + k + 2), Y_label, size=24, fontproperties='Times New Roman')
        plt.hlines(k + 0.4, xmin=0, xmax=t, color="black")  # 横线
        plt.ylabel("AGV           机器(工件类别，批数)        ", size=24, fontproperties='SimSun')
        plt.xlabel("时间", size=24, fontproperties='SimSun')
        plt.tick_params(labelsize=24)
        # title = "Makespan:{}，Idle time:{}，Transportation distance:{}".format(round(t,2), round(Idle_sum,2), round(S_sum,2))
        # plt.title(title, fontsize=24)
        # plt.xlim(0)
        # plt.yticks(np.arange(M_num + k + 2), Y_label, size=24, fontproperties='Times New Roman')
        # plt.hlines(k + 0.4, xmin=0, xmax=t, color="black")  # 横线
        # plt.ylabel("AGV               Machine     ", size=24, fontproperties='Times New Roman')
        # plt.xlabel("Time", size=24, fontproperties='Times New Roman')
        # plt.tick_params(labelsize=24)
        plt.tick_params(direction='in')
        plt.tight_layout()
        main_window = QtWidgets.QMainWindow(); canvas = FigureCanvas(fig); main_window.setCentralWidget(canvas); main_window.showFullScreen(); main_window.close()
        plt.savefig('data_out/240401/fig/算例{}_v{}甘特图.png'.format(suanli, v), dpi=500)
        # plt.show()
        plt.close()
    def Gantt2(self,suanli, v, faultyMachine, faultyTime_Mac, repairTime_Mac, faultyAGV, fauTimeAGV, repairTime_AGV):
        fig = plt.figure()
        M = ['red', 'blue', 'yellow', 'orange', 'green', 'moccasin', 'purple', 'pink', 'navajowhite', 'Thistle', 'Magenta', 'SlateBlue', 'RoyalBlue', 'Aqua', 'floralwhite', 'ghostwhite', 'goldenrod',
             'mediumslateblue', 'navajowhite', 'navy', 'sandybrown', "cornflowerblue", "sandybrown", "mediumorchid", "cadetblue", "darkolivegreen", "steelblue", "darkslateblue", "cadetblue", "tomato", "mediumpurple"]
        M_num, t, Y_label= 0, 0, [0]
        #7 机器全部空闲时间
        Idle_sum = 0
        for i in range(1, len(self.AL_jk)):
            Idle_time = self.Idle_time_jisuan(i)
            Idle_sum += sum(Idle_time)
        # 8 AGV运输距离
        S_sum = sum(self.yunshu_AGV)
        # AGV甘特图
        fauTimeAGV_min = 99999
        for i in range(len(fauTimeAGV)):
            if fauTimeAGV[i] != 0:
                if fauTimeAGV[i] < fauTimeAGV_min:
                    fauTimeAGV_min = fauTimeAGV[i]
                    fauTAGV_min = i
        faultyTime_Mac2 = faultyTime_Mac; fauTimeAGV_min2 = fauTimeAGV_min
        faultyTime_Mac = min(faultyTime_Mac2, fauTimeAGV_min2); fauTimeAGV_min = min(faultyTime_Mac2, fauTimeAGV_min2)
        for k in range(1,len(self.AGVs)):
            if k == fauTAGV_min and fauTimeAGV_min2 == fauTimeAGV_min:
                plt.vlines(fauTimeAGV_min2, ymin=k - 0.4, ymax=k + 0.4, color='black', linestyle='--')
                plt.annotate('   故障发生', xy=(fauTimeAGV_min2, k - 0.4), xytext=(fauTimeAGV_min, k - 1), arrowprops=dict(facecolor='black', shrink=0.001))
            for m in range(len(self.AGVs[k].using_time)):
                Start_time = self.AGVs[k].using_time[m][0]
                End_time = self.AGVs[k].using_time[m][1]
                if End_time - Start_time != 0:
                    if Start_time < fauTimeAGV_min2 < End_time and k == fauTAGV_min and fauTimeAGV_min2 == fauTimeAGV_min:
                        plt.barh(k, width=End_time - Start_time, height=0.6, left=Start_time, color='red', edgecolor='black')
                    elif Start_time < fauTimeAGV_min:
                        if self.AGVs[k]._on[m] != None:
                            plt.barh(k, width=End_time - Start_time, height=0.6, left=Start_time, color='green', edgecolor='black')
                        else:
                            plt.barh(k, width=End_time - Start_time, height=0.6, left=Start_time, color='white', edgecolor='black')
                    else:
                        plt.barh(k, width=End_time - Start_time, height=0.6, left=Start_time, color='gray', edgecolor='black')
            Y_label.append(k)
        # 机器甘特图
        # if faultyTime_Mac <= min(fauTimeAGV):
        for i in range(1, len(self.M_j)):  # 在第i个阶段
            for j in range(self.M_j[i]):  # 在第j个机器
                if self.MNN[i][j] == faultyMachine and faultyTime_Mac2 == faultyTime_Mac:
                    plt.vlines(faultyTime_Mac, ymin=faultyMachine + k - 0.4, ymax=faultyMachine + k + 0.4, color='black', linestyle='--')
                    plt.annotate('   故障发生', xy=(faultyTime_Mac, faultyMachine + k - 0.4), xytext=(faultyTime_Mac, faultyMachine + k - 1), arrowprops=dict(facecolor='black', shrink=0.001))
                for q in range(len(self.AL_jk[i][j].start)):  # 在该机器加工的第q批工件
                    # if self.MNN[i][j] == faultyMachine:
                    #     M_num += 0.5
                    Start_time = self.AL_jk[i][j].start[q]
                    End_time = self.AL_jk[i][j].end[q + 1]
                    Job = self.AL_jk[i][j]._on[q]
                    J_variety, J_batch = 0, 0
                    for a in range(1,self.n+1):
                        for b in range(len(self.JNN[a])):
                            if self.JNN[a][b] == Job:
                                J_variety, J_batch = a, b
                    text = "(%s,%d)" % (J_variety, J_batch + 1)
                    if Start_time < faultyTime_Mac2 < End_time and self.MNN[i][j] == faultyMachine and faultyTime_Mac2 == faultyTime_Mac:
                        plt.barh(M_num + k + 1, width=End_time - Start_time, height=0.8, left=Start_time, color='red', edgecolor='black')
                    elif Start_time < faultyTime_Mac:
                        plt.barh(M_num + k + 1, width=End_time - Start_time, height=0.8, left=Start_time, color='green', edgecolor='black')
                    else:
                        plt.barh(M_num + k + 1, width=End_time - Start_time, height=0.8, left=Start_time, color='yellow', edgecolor='black')
                    plt.text(x=(Start_time + End_time) / 2 - 20, y=M_num + k + 1 - 0.2, s=text, size=24, fontproperties='Times New Roman')
                if self.AL_jk[i][j].last_ot > t:
                    t = self.AL_jk[i][j].last_ot
                M_num += 1
                Y_label.append(M_num)
        Y_label.append(M_num+1)
        title = "预调度图：完工:{}，机器空闲:{}，AGV总运输距离:{}".format(round(t,2), round(Idle_sum,2), round(S_sum,2))
        plt.title(title, fontsize=24)
        plt.xlim(0)
        plt.yticks(np.arange(M_num + k + 2), Y_label, size=24, fontproperties='Times New Roman')
        plt.hlines(k + 0.4, xmin=0, xmax=t, color="black")  # 横线
        plt.ylabel("AGV       机器(工件类别，批数)     ", size=24, fontproperties='SimSun')
        plt.xlabel("时间", size=24, fontproperties='SimSun')
        plt.tick_params(labelsize=24)
        plt.tick_params(direction='in')
        plt.tight_layout()
        main_window = QtWidgets.QMainWindow(); canvas = FigureCanvas(fig); main_window.setCentralWidget(canvas); main_window.showFullScreen(); main_window.close()
        plt.savefig('data_out/240401/fig/算例{}_v{}预调度甘特图.png'.format(suanli, v), dpi=500)
        # plt.show()
        plt.close()
    def Get_G1(self,faultyTime_Mac, fauTimeAGV):
        G1_job, G1_mac, G1_agv = [], [], []
        for i in range(1, len(self.M_j)):  # 在第i个阶段
            for j in range(self.M_j[i]):  # 在第j个机器
                for q in range(len(self.AL_jk[i][j].start)):  # 在该机器加工的第q批工件
                    if self.AL_jk[i][j].start[q] < min(faultyTime_Mac, min(fauTimeAGV)):
                        G1_job.append(self.AL_jk[i][j]._on[q]);G1_mac.append(self.MNN[i][j]);G1_agv.append(self.AL_jk[i][j]._on_agv[q])
        return G1_job, G1_mac, G1_agv
    def Get_J_start(self):
        J_start = []
        for i in range(1, len(self.Jobs)):
            J_start.append(self.Jobs[i].start)
        return J_start
    def reGantt(self,suanfa, suanli, v, m_sta, faultyMachine, faultyTime_Mac, faultyAGV, faultyTime_AGV, repairTime_Mac, repairTime_AGV):
        fig = plt.figure()
        M = ['red', 'blue', 'yellow', 'orange', 'green', 'moccasin', 'purple', 'pink', 'navajowhite', 'Thistle', 'Magenta', 'SlateBlue', 'RoyalBlue', 'Aqua', 'floralwhite', 'ghostwhite', 'goldenrod',
             'mediumslateblue', 'navajowhite', 'navy', 'sandybrown', "cornflowerblue", "sandybrown", "mediumorchid", "cadetblue", "darkolivegreen", "steelblue", "darkslateblue", "cadetblue", "tomato", "mediumpurple"]
        M_num, t, Y_label= 0, 0, [0]
        #7 机器全部空闲时间
        Idle_sum = 0
        for i in range(1, len(self.AL_jk)):
            Idle_time = self.Idle_time_jisuan(i)
            Idle_sum += sum(Idle_time)
        # 8 AGV运输距离
        S_sum = sum(self.yunshu_AGV)
        fauAGV, fauTime_AGV = [], []
        for i in range(len(faultyAGV)):
            fauTime_AGV.append(faultyTime_AGV[i] * faultyAGV[i])
            fauAGV.append(faultyAGV[i]*i)
            if fauAGV[i] != 0:
                plt.vlines(fauTime_AGV[i], ymin=i-0.3, ymax=i+0.3, color='r', linestyle='--')
                plt.vlines(fauTime_AGV[i]+ repairTime_AGV, ymin=i-0.3, ymax=i + 0.3, color='b', linestyle='--')
        k = len(self.AGVs)-1
        plt.vlines(faultyTime_Mac, ymin=faultyMachine + k - 0.4, ymax=faultyMachine + k + 0.4, color='r', linestyle='--')
        plt.vlines(faultyTime_Mac + repairTime_Mac, ymin=faultyMachine + k - 0.4, ymax=faultyMachine + k + 0.4, color='b', linestyle='--')
        fauAGV = [i for i in fauAGV if i != 0]
        fauTime_AGV = [i for i in fauTime_AGV if i != 0]
        fauTime = [faultyTime_Mac] + fauTime_AGV
        # AGV甘特图
        for k in range(1,len(self.AGVs)):
            for m in range(len(self.AGVs[k].using_time)):
                if self.AGVs[k].using_time[m][1] - self.AGVs[k].using_time[m][0] != 0:
                    if self.AGVs[k]._on[m] != None:
                        if self.AGVs[k]._on[m] == 999:
                            plt.barh(k, width=self.AGVs[k].using_time[m][1] - self.AGVs[k].using_time[m][0],
                                     height=0.6, left=self.AGVs[k].using_time[m][0], color='black', edgecolor='black')
                        else:
                            J_variety = 0
                            for a in range(self.n+1):
                                for b in range(len(self.JNN[a])):
                                    if self.JNN[a][b] == self.AGVs[k]._on[m]:
                                        J_variety = a
                            plt.barh(k, width=self.AGVs[k].using_time[m][1] - self.AGVs[k].using_time[m][0],
                                     height=0.6, left=self.AGVs[k].using_time[m][0], color=M[J_variety], edgecolor='black')
                    else:
                        plt.barh(k, width=self.AGVs[k].using_time[m][1] - self.AGVs[k].using_time[m][0],
                                 height=0.6, left=self.AGVs[k].using_time[m][0], color='white', edgecolor='black')
            Y_label.append(k)
        # 机器甘特图
        m_sta2, m_id2 = self.find_element_index(self.MNN, faultyMachine)  # 下一工序机器所在阶段和序号（阶段内）
        J_m2 = self.AL_jk[m_sta2][m_id2]  # 下一工序机器
        if J_m2.end[-1] < faultyTime_Mac:
            J_m2.start.append(faultyTime_Mac); J_m2.end.append(faultyTime_Mac + repairTime_Mac); J_m2._on.append(999)
        for i in range(1, len(self.M_j)):  # 在第i个阶段
            for j in range(self.M_j[i]):  # 在第j个机器
                for q in range(len(self.AL_jk[i][j].start)):  # 在该机器加工的第q批工件
                    Start_time = self.AL_jk[i][j].start[q]
                    End_time = self.AL_jk[i][j].end[q + 1]
                    Job = self.AL_jk[i][j]._on[q]
                    if Job == 999:
                        plt.barh(M_num + k + 1, width=End_time - Start_time, height=0.8, left=Start_time, color='black', edgecolor='black')
                    else:
                        J_variety, J_batch = 0, 0
                        for a in range(1,self.n+1):
                            for b in range(len(self.JNN[a])):
                                if self.JNN[a][b] == Job:
                                    J_variety, J_batch = a, b
                        text = "(%s,%d)" % (J_variety, J_batch + 1)
                        plt.barh(M_num + k + 1, width=End_time - Start_time, height=0.8, left=Start_time, color=M[J_variety], edgecolor='black')
                        plt.text(x=Start_time + ((End_time - Start_time) / 2 - 0.25), y=M_num + k + 1 - 0.2, s=text, size=15, fontproperties='Times New Roman')
                if self.AL_jk[i][j].last_ot > t:
                    t = self.AL_jk[i][j].last_ot
                M_num += 1
                Y_label.append(M_num)
        Y_label.append(M_num+1)
        title = "{}重调度图：完工:{}，机器空闲:{}，AGV总运输距离:{}\n(阶段{}的机器{}在{}时间发生故障，AGV{}在{}时间发生故障)".format(suanfa, round(t,2), round(Idle_sum,2), round(S_sum,2), m_sta, faultyMachine, faultyTime_Mac, fauAGV, fauTime_AGV)
        plt.title(title, fontsize=24)
        plt.xlim(0)
        plt.yticks(np.arange(M_num + k + 2), Y_label, size=24, fontproperties='Times New Roman')
        plt.hlines(k + 0.4, xmin=0, xmax=t, color="black")  # 横线
        plt.ylabel("AGV       机器(工件类别，批数)     ", size=24, fontproperties='SimSun')
        plt.xlabel("时间", size=24, fontproperties='SimSun')
        plt.tick_params(labelsize=24)
        plt.tick_params(direction='in')
        plt.tight_layout()
        main_window = QtWidgets.QMainWindow(); canvas = FigureCanvas(fig); main_window.setCentralWidget(canvas); main_window.showFullScreen(); main_window.close()
        plt.savefig('data_out/240401/fig/算例{}_v{}{}重调度甘特图.png'.format(suanli, v, suanfa), dpi=500)
        plt.show()
        plt.close()
    def reGantt2(self,suanfa, suanli, v, m_sta, faultyMachine, faultyTime_Mac, faultyAGV, faultyTime_AGV, repairTime_Mac, repairTime_AGV):
        fig = plt.figure()
        M = ['red', 'blue', 'yellow', 'orange', 'green', 'moccasin', 'purple', 'pink', 'navajowhite', 'Thistle', 'Magenta', 'SlateBlue', 'RoyalBlue', 'Aqua', 'floralwhite', 'ghostwhite', 'goldenrod',
             'mediumslateblue', 'navajowhite', 'navy', 'sandybrown', "cornflowerblue", "sandybrown", "mediumorchid", "cadetblue", "darkolivegreen", "steelblue", "darkslateblue", "cadetblue", "tomato", "mediumpurple"]
        M_num, t, Y_label= 0, 0, [0]
        #7 机器全部空闲时间
        Idle_sum = 0
        for i in range(1, len(self.AL_jk)):
            Idle_time = self.Idle_time_jisuan(i)
            Idle_sum += sum(Idle_time)
        # 8 AGV运输距离
        S_sum = sum(self.yunshu_AGV)
        fauAGV, fauTime_AGV = [], []
        for i in range(len(faultyAGV)):
            fauTime_AGV.append(faultyTime_AGV[i] * faultyAGV[i])
            fauAGV.append(faultyAGV[i]*i)
            if fauAGV[i] != 0:
                plt.vlines(fauTime_AGV[i], ymin=i-0.3, ymax=i+0.3, color='r', linestyle='--')
                plt.vlines(fauTime_AGV[i]+ repairTime_AGV, ymin=i-0.3, ymax=i + 0.3, color='b', linestyle='--')
        k = len(self.AGVs)-1
        plt.vlines(faultyTime_Mac, ymin=faultyMachine + k - 0.4, ymax=faultyMachine + k + 0.4, color='r', linestyle='--')
        plt.vlines(faultyTime_Mac + repairTime_Mac, ymin=faultyMachine + k - 0.4, ymax=faultyMachine + k + 0.4, color='b', linestyle='--')
        fauAGV = [i for i in fauAGV if i != 0]
        fauTime_AGV = [i for i in fauTime_AGV if i != 0]
        fauTime = [faultyTime_Mac] + fauTime_AGV
        # AGV甘特图
        fauTimeAGV_min = 99999
        for i in range(len(fauTime_AGV)):
            if fauTime_AGV[i] != 0:
                if fauTime_AGV[i] < fauTimeAGV_min:
                    fauTimeAGV_min = fauTime_AGV[i]
                    fauTAGV_min = i
        faultyTime_Mac2 = faultyTime_Mac; fauTimeAGV_min2 = fauTimeAGV_min
        faultyTime_Mac = min(faultyTime_Mac2, fauTimeAGV_min2); fauTimeAGV_min = min(faultyTime_Mac2, fauTimeAGV_min2)
        for k in range(1,len(self.AGVs)):
            if k == fauTAGV_min and fauTimeAGV_min2 == fauTimeAGV_min:
                plt.vlines(fauTimeAGV_min2, ymin=k + 0.6, ymax=k + 1.4, color='black', linestyle='--')
                plt.annotate('   故障发生', xy=(fauTimeAGV_min2, k + 0.6), xytext=(fauTimeAGV_min, k), arrowprops=dict(facecolor='black', shrink=0.001))
            for m in range(len(self.AGVs[k].using_time)):
                Start_time = self.AGVs[k].using_time[m][0]
                End_time = self.AGVs[k].using_time[m][1]
                if End_time - Start_time != 0:
                    if self.AGVs[k]._on[m] != None:
                        if self.AGVs[k]._on[m] == 999:
                            plt.barh(k, width=End_time - Start_time, height=0.6, left=Start_time, color='black', edgecolor='black')
                        else:
                            if Start_time < fauTimeAGV_min:
                                plt.barh(k, width=End_time - Start_time, height=0.6, left=Start_time, color='green', edgecolor='black')
                            else:
                                plt.barh(k, width=End_time - Start_time, height=0.6, left=Start_time, color='yellow', edgecolor='black')
                    else:
                        plt.barh(k, width=End_time - Start_time, height=0.6, left=Start_time, color='white', edgecolor='black')
            Y_label.append(k)
        # 机器甘特图
        m_sta2, m_id2 = self.find_element_index(self.MNN, faultyMachine)  # 下一工序机器所在阶段和序号（阶段内）
        J_m2 = self.AL_jk[m_sta2][m_id2]  # 下一工序机器
        if J_m2.end[-1] < faultyTime_Mac:
            J_m2.start.append(faultyTime_Mac); J_m2.end.append(faultyTime_Mac + repairTime_Mac); J_m2._on.append(999)
        for i in range(1, len(self.M_j)):  # 在第i个阶段
            for j in range(self.M_j[i]):  # 在第j个机器
                if self.MNN[i][j] == faultyMachine and faultyTime_Mac2 == faultyTime_Mac:
                    plt.vlines(faultyTime_Mac, ymin=faultyMachine + k - 0.4, ymax=faultyMachine + k + 0.4, color='black', linestyle='--')
                    plt.annotate('   故障发生', xy=(faultyTime_Mac, faultyMachine + k - 0.4), xytext=(faultyTime_Mac, faultyMachine + k - 1), arrowprops=dict(facecolor='black', shrink=0.001))
                for q in range(len(self.AL_jk[i][j].start)):  # 在该机器加工的第q批工件
                    Start_time = self.AL_jk[i][j].start[q]
                    End_time = self.AL_jk[i][j].end[q + 1]
                    Job = self.AL_jk[i][j]._on[q]
                    if Job == 999:
                        plt.barh(M_num + k + 1, width=End_time - Start_time, height=0.8, left=Start_time, color='black', edgecolor='black')
                    else:
                        J_variety, J_batch = 0, 0
                        for a in range(1,self.n+1):
                            for b in range(len(self.JNN[a])):
                                if self.JNN[a][b] == Job:
                                    J_variety, J_batch = a, b
                        text = "(%s,%d)" % (J_variety, J_batch + 1)
                        if Start_time < faultyTime_Mac:
                            plt.barh(M_num + k + 1, width=End_time - Start_time, height=0.8, left=Start_time, color='green', edgecolor='black')
                        else:
                            plt.barh(M_num + k + 1, width=End_time - Start_time, height=0.8, left=Start_time, color='yellow', edgecolor='black')
                        plt.text(x=(Start_time + End_time) / 2 - 20, y=M_num + k + 1 - 0.2, s=text, size=15, fontproperties='Times New Roman')
                if self.AL_jk[i][j].last_ot > t:
                    t = self.AL_jk[i][j].last_ot
                M_num += 1
                Y_label.append(M_num)
        Y_label.append(M_num+1)
        title = "{}重调度图：完工:{}，机器空闲:{}，AGV总运输距离:{}\n(阶段{}的机器{}在{}时间发生故障，AGV{}在{}时间发生故障)".format(suanfa, round(t,2), round(Idle_sum,2), round(S_sum,2), m_sta, faultyMachine, faultyTime_Mac2, fauAGV, fauTime_AGV)
        plt.title(title, fontsize=15)
        plt.xlim(0)
        plt.yticks(np.arange(M_num + k + 2), Y_label, size=24, fontproperties='Times New Roman')
        plt.hlines(k + 0.4, xmin=0, xmax=t, color="black")  # 横线
        plt.ylabel("AGV       机器(工件类别，批数)     ", size=15, fontproperties='SimSun')
        plt.xlabel("时间", size=15, fontproperties='SimSun')
        plt.tick_params(labelsize=24)
        plt.tick_params(direction='in')
        plt.tight_layout()
        main_window = QtWidgets.QMainWindow(); canvas = FigureCanvas(fig); main_window.setCentralWidget(canvas); main_window.showFullScreen(); main_window.close()
        plt.savefig('data_out/240401/fig/算例{}_v{}{}重调度甘特图.png'.format(suanli, v, suanfa), dpi=500)
        plt.show()
        plt.close()

    # 画最优解甘特图
    def Gantt_best(self,agv_using_time, agv_on, mac_start, mac_end, mac_on, mac_last_ot, UK, UK_AGV):
        fig = plt.figure()
        M = ['red', 'blue', 'yellow', 'orange', 'green', 'moccasin', 'purple', 'pink', 'navajowhite', 'Thistle', 'Magenta', 'SlateBlue', 'RoyalBlue', 'Aqua', 'floralwhite', 'ghostwhite', 'goldenrod',
             'mediumslateblue', "darkolivegreen", "steelblue", "darkslateblue", "cadetblue", "tomato", "mediumpurple"]
        M_num, t, Y_label = 0, 0, [0]
        # 4 机器平均利用率
        U_ave = self.calculate_sum(UK) / (sum(self.M_j) - 1)
        # 8 AGV平均空闲率
        S_sum = (sum(UK_AGV) - UK_AGV[0]) / self.v
        # AGV甘特图
        for k in range(1, len(self.AGVs)):
            for m in range(len(agv_using_time[k])):
                if agv_using_time[k][m][1] - agv_using_time[k][m][0] != 0:
                    if agv_on[k][m] != None:
                        J_variety = 0
                        for a in range(self.n+1):
                            for b in range(len(self.JNN[a])):
                                if self.JNN[a][b] == agv_on[k][m]:
                                    J_variety = a
                        plt.barh(k, width=agv_using_time[k][m][1] - agv_using_time[k][m][0], height=0.6, left=agv_using_time[k][m][0],
                                 color=M[J_variety], edgecolor='black')
                    else:
                        plt.barh(k, width=agv_using_time[k][m][1] - agv_using_time[k][m][0], height=0.6, left=agv_using_time[k][m][0],
                                 color='white', edgecolor='black')
            Y_label.append(k)
        # 机器甘特图
        for i in range(1, len(self.M_j)):  # 在第i个阶段
            for j in range(self.M_j[i]):  # 在第j个机器
                for q in range(len(mac_start[i][j])):  # 在该机器加工的第q批工件
                    Start_time = mac_start[i][j][q]
                    End_time = mac_end[i][j][q + 1]
                    Job = mac_on[i][j][q]
                    J_variety, J_batch = 0, 0
                    for a in range(1, self.n+1):
                        for b in range(len(self.JNN[a])):
                            if self.JNN[a][b] == Job:
                                J_variety, J_batch = a, b
                    text = "(%s,%d)" % (J_variety, J_batch + 1)
                    plt.barh(M_num + k + 1, width=End_time - Start_time, height=0.8, left=Start_time, color=M[J_variety], edgecolor='black')
                    plt.text(x=Start_time + ((End_time - Start_time) / 2 - 0.25), y=M_num + k + 1 - 0.2, s=text, size=15, fontproperties='Times New Roman')
                if mac_last_ot[i][j] > t:
                    t = mac_last_ot[i][j]
                M_num += 1
                Y_label.append(M_num)
        Y_label.append(M_num + 1)
        title = "最优解的机器与AGV调度甘特图:完工时间：{}，机器平均空闲率： {}%，AGV平均空闲率： {}%".format(t, round((1-U_ave)*100, 2), round((S_sum)*100, 2))
        plt.title(title)
        plt.xlim(0)
        plt.yticks(np.arange(M_num + k + 2), Y_label, size=20, fontproperties='Times New Roman')
        plt.hlines(k + 0.4, xmin=0, xmax=t, color="black")  # 横线
        plt.ylabel("AGV                       机器(工件类别，批数)                )  ", size=20, fontproperties='SimSun')
        plt.xlabel("时间", size=20, fontproperties='SimSun')
        plt.tick_params(labelsize=18)
        plt.tick_params(direction='in')
        plt.show()