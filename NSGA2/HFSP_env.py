import random
import string
import matplotlib.pyplot as plt
from HFSP_Instance import n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,MNN,ST_i
import numpy as np

#定义工件类
class Machines:
    # 初始化机器类
    def __init__(self):
        self.start = []  # 开始时间
        self.end = [0]  # 结束时间
        self._on = []  #
        self.T = []  # 所用时间
        self.last_ot = 0  # 完成时间
        self.Idle = []  # 空闲时间集合

    # 更新机器类
    def update(self, s, e, t, on):
        self.start.append(s)
        self.end.append(e)
        self.T.append(t)
        self.last_ot = e
        self._on.append(on)  # 一共加工过的工件序号

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
        self.using_time=[]
        self.T = []  # 所用时间
        self._on=[]
        self._to=[]
        self.end=0

    def ST(self,s,t1,t2):
        if s > (self.end + t1):  # 工件和AGV那个最晚到当前分拣区，以此时间点为AGV负载开始时间点
            return s - t1, s + t2  # 输出AGV空载开始时间点，负载结束时间点
        else:
            return self.end, self.end + t1 + t2  # 输出AGV空载开始时间点，负载结束时间点

    def update(self,s,e,trans1,trans2,J_site,J_m,_on=None):
        self.using_time.append([s, s + trans1])  # [空载开始，空载结束]
        self.using_time.append([e - trans2, e])  # [负载开始，负载结束]
        self.T.append(trans1 + trans2)
        self._on.append(None)
        self._on.append(_on)#负载工件批次序号
        self._to.extend([J_site,J_m])#[负载开始机器序号，负载结束机器序号]
        self.end=s+trans1+trans2#负载结束时间点
        self.cur_site=J_m#更新当前位置

class Scheduling:
    def __init__(self):
        self.Jobs = []
        for i in range(J_num+1):
            J = job(i)
            self.Jobs.append(J)
        self.AL_jk=[] # j阶段的机器k [[],阶段1[机器1，机器2，。。。],阶段2[],阶段3[]]
        for i in range(len(M_j)):    #突出机器的阶段性，即各阶段有哪些机器
            if i == 0:
                self.AL_jk.append([])
            else:
                State_i=[]
                for j in range(M_j[i]):
                    M=Machines()
                    State_i.append(M)
                self.AL_jk.append(State_i)
        self.AGVs=[]#[0,AGV1,AGV2,。。]
        for k in range(v + 1):
            agv = Agv(k)
            self.AGVs.append(agv)
        self.fitness=0
        self.fitness1 = 0
        self.fitness2 = 0

        self.Done = []
        self.CRJBJL = [[0 for _ in row] for row in JNN]  # 工件批次完成记录表 [[0], [0], [0, 0, 0], [0, 0], [0, 0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0, 0]]
        self.CRJB = [0 for i in range(n + 1)]  # 工件批次完成率
        self.CRS = [0 for i in range(n + 1)]  # 作业完成率
        self.SCSJ = DL_bat  # 工件松弛时间，默认是deadline
        self.SCSJ[0] = 99999
        self.UK = []  # 各机器利用率 [[], [0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0]]
        self.UK_AGV = [-1.0]  # AGV空闲率 [0, 1, 1, 1]
        self.m = 1.0
        self.u = 0.0
        self.a = 1.0
        for i in range(State + 1):
            self.UK.append([0 for i in range(M_j[i])])
        for i in range(v):
            self.UK_AGV.append(1.0)
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
    def find_all_max_indices(self, lst):  # 返回目标列表所有最大值的索引
        max_value = max(lst)
        indices = [idx for idx, value in enumerate(lst) if value == max_value]
        return indices
    def shengyu_jiagong_banyun(self,Job,state,time):# 计算Job工件state阶段之后（不包括state阶段）的剩余加工时间和搬运时间
        Ji = self.Jobs[Job]  # 当前批次工件类
        pts_sum = time
        for s in range(state+1, State):
            pt_sum = 0
            if s == state+1:
                for j in range(len(self.AL_jk[s])):
                    pt_sum += PT[s][j][Job] + D_pq[Ji.cur_site][MNN[s][j]] / V_r
            else:
                d_pq = 0
                for j in range(len(self.AL_jk[s])):
                    pt_sum += PT[s][j][Job]
                    for i in range(len(self.AL_jk[s + 1])):
                        d_pq += D_pq[MNN[s+1][i]][MNN[s][j]] / V_r
                pt_sum += d_pq / len(self.AL_jk[s + 1])
            pts_sum += pt_sum / len(self.AL_jk[s])
        return pts_sum
    #每个阶段的解码
    def Stage_Decode(self,CHS):#CHS：初始种群
        for Job_i in CHS:#i：工件序号
            Ji = self.Jobs[Job_i]  # 当前批次工件类
            last_od=Ji.last_ot#工件i完成时间
            #机器调度
            j_sta, j_mac = self.find_element_index(MNN, Ji.cur_site)  # 工件当前所在机器的阶段（0、1、2...state）和机器序号（阶段内）
            j_arr, j_bat = self.find_element_index(JNN, Job_i)  # 工件种类序号 批次号
            m_sta = j_sta + 1
            last_Md = [self.AL_jk[m_sta][M_i].last_ot for M_i in range(M_j[m_sta])]  # 下一阶段机器的完成时间 m_sta阶段[机器0完成时间,机器1完成时间,。。]
            O_et = [0 for i in range(M_j[m_sta])]  # m_sta阶段[工件i在机器0开始加工时间,工件i在机器1开始加工时间,。。]
            O_uk = self.UK[m_sta]  # m_sta阶段[机器0利用率,机器1利用率,。。]
            if j_sta == 0:  # 工件处于第一阶段前
                for i in range(M_j[m_sta]):
                    O_et[i] = max(last_Md[i], ST_i[j_arr] * JN_ip[j_arr][j_bat])  # max(当前阶段机器的完成时间,工件i个数*该种分拣时间)
                if O_et.count(min(O_et)) > 1:  # 当结束加工时间最小机器数为2个及以上，则从中随机选择一个机器
                    m_id = random.choice(self.find_all_min_indices(O_et))  # 从中随机选择一个机器
                else:  # 只有一台机器结束加工时间最短
                    m_id = O_et.index(min(O_et))  # 就选它
            else:
                for i in range(M_j[m_sta]):
                    O_et[i] = max(last_Md[i], Ji.last_ot + D_pq[Ji.cur_site][MNN[m_sta][i]] / V_r)  # max(当前阶段机器的完成时间,工件i结束当前工序并被AGV送到下一机器时间)
                if O_et.count(min(O_et)) > 1:  # 当结束加工时间最小机器数为2个及以上，则选择运输距离最短的机器
                    mac_min = self.find_all_indices(O_et, min(O_et))
                    TT_min = D_pq[Ji.cur_site][MNN[m_sta][mac_min[0]]]
                    m_id = 0
                    for i in mac_min:
                        if D_pq[Ji.cur_site][MNN[m_sta][i]] < TT_min:
                            TT_min = D_pq[Ji.cur_site][MNN[m_sta][i]]
                            m_id = i
                else:  # 只有一台机器结束加工时间最短
                    m_id = O_et.index(min(O_et))  # 就选它
            J_m = self.AL_jk[m_sta][m_id]  # 下一工序机器
            #AGV调度
            if j_sta == 0:
                agv = 0
            else:
                J_end, J_site, on_m = Ji.get_info() # 加工结束时间，当前位置，在下一工序机器加工时间，各工序加工机器编号（工序内）
                best_agv = None
                min_tf = 99999
                best_s, best_e, t1, t2 = None, None, None, None
                if self.UK_AGV.count(max(self.UK_AGV)) > 1:  # 当空闲率最大AGV数为2个及以上，则从中选择一个FAST
                    for agv in self.find_all_max_indices(self.UK_AGV):
                        trans1 = D_pq[self.AGVs[agv].cur_site][J_site]  # AGV到工件当前位置距离
                        trans2 = D_pq[J_site][MNN[m_sta][m_id]]  # 工件到下一工序机器距离
                        start, end = self.AGVs[agv].ST(J_end[-1], trans1 / V_r,
                                                       trans2 / V_r)  # start：AGV空载开始时间点，end：负载结束时间点
                        if end < min_tf:
                            best_s, best_e, t1, t2 = start, end, trans1 / V_r, trans2 / V_r
                            best_agv = self.AGVs[agv]
                            min_tf = best_e
                    best_agv.update(best_s, best_e, t1, t2, J_site, MNN[m_sta][m_id], Ji.idx)  # 选出最优的AGV
                else:  # 只有一台AGV空闲率最大
                    if min(self.UK_AGV[1:]) == 0.0:
                        for agv in self.find_all_indices(self.UK_AGV, 0.0):
                            trans1 = D_pq[self.AGVs[agv].cur_site][J_site]  # AGV到工件当前位置距离
                            trans2 = D_pq[J_site][MNN[m_sta][m_id]]  # 工件到下一工序机器距离
                            start, end = self.AGVs[agv].ST(J_end[-1], trans1 / V_r,
                                                           trans2 / V_r)  # start：AGV空载开始时间点，end：负载结束时间点
                            if end < min_tf:
                                best_s, best_e, t1, t2 = start, end, trans1 / V_r, trans2 / V_r
                                best_agv = self.AGVs[agv]
                                min_tf = best_e
                    else:
                        agv = self.UK_AGV.index(max(self.UK_AGV))  # 就选它
                        trans1 = D_pq[self.AGVs[agv].cur_site][J_site]  # AGV到工件当前位置距离
                        trans2 = D_pq[J_site][MNN[m_sta][m_id]]  # 工件到下一工序机器距离
                        start, end = self.AGVs[agv].ST(J_end[-1], trans1 / V_r,
                                                       trans2 / V_r)  # start：AGV空载开始时间点，end：负载结束时间点
                        if trans1 < min_tf:
                            best_s, best_e, t1, t2 = start, end, trans1 / V_r, trans2 / V_r
                            best_agv = self.AGVs[agv]
                            min_tf = trans1
                    best_agv.update(best_s, best_e, t1, t2, J_site, MNN[m_sta][m_id], Ji.idx)  # 选出最优的AGV
                agv = best_agv.idx
            A_i = self.AGVs[agv]  # 运输工件到下一工序的AGV
            last_ot = Ji.last_ot  # 工件上道工序加工结束时间
            last_mt = J_m.last_ot  # 工件下一工序机器上道工序加工结束时间
            if j_sta < 1:
                Start_time = max(last_ot + ST_i[j_arr] * JN_ip[j_arr][j_bat], last_mt)
            else:
                Start_time = max(A_i.end, last_mt)  # 以AGV送到和机器结束上一个工件加工时间大者为开始加工时间
                '''AGV状态信息'''
                self.UK_AGV[agv] = 1.0 if A_i.end == 0 else (A_i.end - A_i.using_time[0][0] - sum(A_i.T)) / (A_i.end - A_i.using_time[0][0])  # AGV空闲率
            pt = PT[m_sta][m_id][Job_i]  # 即将加工的工序加工时间
            end_time = Start_time + pt
            J_m.update(Start_time, end_time, pt, Job_i)
            J_m.idle_time()
            Ji.update(Start_time, end_time, pt, MNN[m_sta][m_id])
            '''工件状态信息'''
            j_arr, j_bat = self.find_element_index(JNN, Job_i)  # 找到Job的种类序号和批次号
            self.CRJB[j_arr] = self.CRJBJL[j_arr].count(State) / len(JNN[j_arr])  # 工件批次完成率
            CRJB_sum = 0  # 该类工件阶段完成数
            for i in range(len(JNN[j_arr])):
                CRJB_sum += JN_ip[j_arr][i] * self.CRJBJL[j_arr][i]  # 该类工件每批数量*该批完成阶段数
            self.CRS[j_arr] = CRJB_sum / (sum(JN_ip[j_arr]) * State)  # 该类工件阶段完成率
            self.SCSJ[Job_i] = DL[j_arr] - Ji.last_ot - self.shengyu_jiagong_banyun(Job_i, m_sta, 0)  # 该批工件松弛时间=deadline-当前时间-剩余加工时间-剩余搬运时间
            self.CRJBJL[j_arr][j_bat] += 1  # 该批工件完成阶段数+1
            '''机器状态信息'''
            self.UK[m_sta][m_id] = 0 if J_m.last_ot == 0 else sum(J_m.T) / (J_m.last_ot - J_m.start[0])  # 机器利用率
        # 1 工件完工时间
        self.fitness = self.AL_jk[1][0].last_ot
        for i in range(1, len(self.AL_jk)):
            for j in range(len(self.AL_jk[i])):
                if self.AL_jk[i][j].last_ot > self.fitness:
                    self.fitness = self.AL_jk[i][j].last_ot
        # 2 机器空闲率
        self.fitness1 = 1.0 - self.calculate_sum(self.UK) / (sum(M_j) - 1)
        # 3 AGV空闲率
        x, y = 0, 0
        for i in range(1, v + 1):
            x += 0 if self.AGVs[i].end == 0 else self.AGVs[i].end - self.AGVs[i].using_time[0][0] - sum(self.AGVs[i].T)
            y += 0 if self.AGVs[i].end == 0 else self.AGVs[i].end - self.AGVs[i].using_time[0][0]
        self.fitness2 = 1.0 if y == 0 else x / y
        #self.fitness2 = (sum(self.UK_AGV) - self.UK_AGV[0]) / v

    #解码
    def Decode(self,CHS):
        for i in range(1):
            self.Stage_Decode(CHS)
            Job_end=[self.Jobs[i].last_ot for i in range(1,J_num+1)]#阶段i[工件1完成时间,工件2完成时间,。。]
            CHS = sorted(range(len(Job_end)), key=lambda k: Job_end[k], reverse=False)#按照完成时间升序排列阶段i的工件序号

    #画甘特图
    def Gantt(self):
        fig = plt.figure()
        M = ['red', 'blue', 'yellow', 'orange', 'green', 'moccasin', 'purple', 'pink', 'navajowhite', 'Thistle',
             'Magenta', 'SlateBlue', 'RoyalBlue', 'Aqua', 'floralwhite', 'ghostwhite', 'goldenrod', 'mediumslateblue',
             'navajowhite','navy', 'sandybrown', "cornflowerblue", "sandybrown", "mediumorchid", "cadetblue", "rosybrown", "palevioletred", "darkorange",
        "powderblue", "mediumslateblue", "darkkhaki", "palegreen", "darkorchid", "burlywood", "deepskyblue",
        "lightslategray", "lightcoral", "mediumturquoise", "springgreen", "darkviolet", "khaki", "mediumvioletred",
        "lightseagreen", "saddlebrown", "mediumaquamarine", "lightpink", "cadet", "hotpink", "indianred",
        "lightgreen", "lightblue", "palegoldenrod", "darkturquoise", "limegreen", "mediumseagreen", "purple",
        "slategray", "mediumblue", "darkred", "darkgreen", "darkblue", "darkslategray", "olivedrab", "midnightblue",
        "darksalmon", "darkcyan", "blueviolet", "darkmagenta", "forestgreen", "firebrick", "dimgray", "mediumspringgreen",
        "orangered", "slateblue", "darkolivegreen", "steelblue", "darkslateblue", "cadetblue", "tomato", "mediumpurple",
        "mediumspringgreen", "mediumblue", "springgreen", "darkviolet", "khaki", "mediumvioletred", "lightseagreen",
        "saddlebrown", "mediumaquamarine", "lightpink", "cadet", "hotpink", "indianred", "lightgreen", "lightblue",
        "palegoldenrod", "darkturquoise", "limegreen", "mediumseagreen", "purple", "slategray", "mediumblue",
        "darkred", "darkgreen", "darkblue", "darkslategray", "olivedrab", "midnightblue", "darksalmon", "darkcyan",
        "blueviolet", "darkmagenta", "forestgreen", "firebrick", "dimgray", "mediumspringgreen", "orangered", "slateblue",
        "darkolivegreen", "steelblue", "darkslateblue", "cadetblue", "tomato", "mediumpurple"]

        M_num, t, Y_label= 0, 0, [0]
        #k = 0
        # AGV甘特图
        for k in range(1,len(self.AGVs)):
            for m in range(len(self.AGVs[k].using_time)):
                if self.AGVs[k].using_time[m][1] - self.AGVs[k].using_time[m][0] != 0:
                    if self.AGVs[k]._on[m] != None:
                        J_variety = 0
                        for a in range(n+1):
                            for b in range(len(JNN[a])):
                                if JNN[a][b] == self.AGVs[k]._on[m]:
                                    J_variety = a
                        plt.barh(k, width=self.AGVs[k].using_time[m][1] - self.AGVs[k].using_time[m][0],
                                 height=0.6, left=self.AGVs[k].using_time[m][0], color=M[J_variety], edgecolor='black')
                    else:
                        plt.barh(k, width=self.AGVs[k].using_time[m][1] - self.AGVs[k].using_time[m][0],
                                 height=0.6, left=self.AGVs[k].using_time[m][0], color='white', edgecolor='black')
            Y_label.append(k)
        # 机器甘特图
        for i in range(1, len(M_j)):  # 在第i个阶段
            for j in range(M_j[i]):  # 在第j个机器
                for q in range(len(self.AL_jk[i][j].start)):  # 在该机器加工的第q批工件
                    Start_time = self.AL_jk[i][j].start[q]
                    End_time = self.AL_jk[i][j].end[q + 1]
                    Job = self.AL_jk[i][j]._on[q]
                    J_variety, J_batch = 0, 0
                    for a in range(1,n+1):
                        for b in range(len(JNN[a])):
                            if JNN[a][b] == Job:
                                J_variety, J_batch = a, b
                    text = "(%s,%d)" % (J_variety, J_batch + 1)
                    plt.barh(M_num + k + 1, width=End_time - Start_time, height=0.8, left=Start_time, color=M[J_variety], edgecolor='black')
                    plt.text(x=Start_time + ((End_time - Start_time) / 2 - 0.25), y=M_num + k + 1 - 0.2, s=text, size=15, fontproperties='Times New Roman')
                if self.AL_jk[i][j].last_ot > t:
                    t = self.AL_jk[i][j].last_ot
                M_num += 1
                Y_label.append(M_num)
        Y_label.append(M_num+1)
        title = "最优解的机器与AGV调度甘特图:完工时间：{}，机器空闲时间： {}，AGV运输距离： {}".format(self.fitness, self.fitness1, self.fitness2)
        plt.title(title)
        plt.xlim(0)
        plt.yticks(np.arange(M_num + k + 2), Y_label, size=20, fontproperties='Times New Roman')
        plt.hlines(k + 0.4, xmin=0, xmax=t, color="black")  # 横线
        plt.ylabel("AGV   机器(工件类别，批数)  ", size=20, fontproperties='SimSun')
        plt.xlabel("时间", size=20, fontproperties='SimSun')
        plt.tick_params(labelsize=20)
        plt.tick_params(direction='in')
        plt.show()
#
# Sch=Scheduling(J_num,Machine,State,PT)

