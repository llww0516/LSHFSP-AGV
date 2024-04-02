import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Linear, ReLU, Sequential
import parl
from parl.utils import logger, summary
from parl.utils.utils import check_model_method
from parl.utils.scheduler import LinearDecayScheduler
from parl.algorithms.paddle import DQN, DDQN
from collections import deque, namedtuple
from tqdm import tqdm
import copy
import statistics
import time
import datetime
import numpy as np
import os
import gym
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from NSGA2.Initial import *
from NSGA2.CalFitness import FitDHHFSP,EnergySave_DHHFSP
from NSGA2.GA import NSGA2POXES3, NSGA2POXES4, reNSGA2POXES4, TiHuanQianDuan, split_by_occurrence
from NSGA2.Tool import *
from NSGA2.LocalSearch import *
# from HFSP_AGV_MultiHead_DQN_Parl_240117 import MDDQN_main
# from HFSP_AGV_MultiHead_DQN_Parl_240117 import reMDDQN_main
from HFSP_Env_MDDQN_NSGA2_240224 import Situation
# from HFSP_Instance import J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,suanli,xunlian_id
from PyQt5 import QtWidgets, QtCore
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
from tqdm import trange
import pandas as pd
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
# J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,suanli,xunlian_id = 0,0,0,[],[],[],0,0,[],'1',0
import math
import random
from collections import defaultdict
random.seed(24)
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
def Suanli_Read(xunlian, gongjianzhongshu, jiqishu, AGVshu):
    '''所有工件种类、阶段数、工件批次（所有种类工件一起编号）、机器序号（所有阶段机器一起编号）、AGV序号都是从 1 开始编号'''
    '''所有工件种类内的批数序号p、阶段内的机器序号k都是从 0 开始编号'''
    xunlian_id = xunlian # 训练第几次
    suanli = '_n{}_m{}'.format(gongjianzhongshu, jiqishu)
    v=AGVshu # AGV数量
    path = 'data/Suanli2/Suanli{}.pkl'.format(suanli)
    with open(path, 'rb') as f:
        Suanli = pickle.load(f)
    N_i = Suanli[1] # 种类i的工件数量
    n = Suanli[0] # 工件种类数
    State=3 # 阶段数
    V_r=1 #AGV的运行速度
    M_j = Suanli[5] # 各阶段机器数，0阶段机器数为1，也就是分拣区
    m = sum(M_j) - 1 # 机器数量
    MNN = [] # 按阶段给机器分组[[0], [1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12]]
    current_num = 0
    for num in M_j:
        sub_list = list(range(current_num, current_num + num))
        MNN.append(sub_list)
        current_num += num
    PT,JN_ip,JNN=Generate(State, N_i, M_j)#PT=[阶段][机器序号（阶段内）][批次]工件加工时间，JN_ip 作业i的第p个子批包含多少工件数[种类][批次]，JNN 按工件种类给每批工件分组[[0],[1],[2, 3, 4],[5, 6],[7, 8, 9, 10],[11, 12, 13], [14, 15, 16], [17, 18, 19, 20]]
    print(N_i)
    PT = Suanli[2]
    J_num=len(PT[1][0])-1
    ST_i = Suanli[3]
    D_pq = Suanli[4]
    return n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id

class QNetwork(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(QNetwork, self).__init__()
        hid1_size = 128
        hid2_size = 128
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, act_dim)
    def forward(self, obs):
        h1 = F.relu(self.fc1(obs))
        h2 = F.relu(self.fc2(h1))
        Q = self.fc3(h2)
        return Q
class MultiHeadQNetwork(parl.Model):
    def __init__(self, obs_dim, act_dim):
        super(MultiHeadQNetwork, self).__init__()
        hid1_size = 128
        hid2_size = 128
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3_head1 = nn.Linear(hid2_size, act_dim)
        self.fc3_head2 = nn.Linear(hid2_size, act_dim)
        self.fc3_head3 = nn.Linear(hid2_size, act_dim)
    def forward(self, obs):
        h1 = F.relu(self.fc1(obs))
        h2 = F.relu(self.fc2(h1))
        Q_head1 = self.fc3_head1(h2)
        Q_head2 = self.fc3_head2(h2)
        Q_head3 = self.fc3_head3(h2)

        return [Q_head1, Q_head2, Q_head3]

class AtariModel(parl.Model):
    def __init__(self, act_dim, dueling=False):
        super().__init__()
        self.conv1 = nn.Conv2D(in_channels=4, out_channels=32, kernel_size=5, stride=1, padding=2, weight_attr=nn.initializer.KaimingNormal())
        self.conv2 = nn.Conv2D(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2, weight_attr=nn.initializer.KaimingNormal())
        self.conv3 = nn.Conv2D(in_channels=32, out_channels=64, kernel_size=4, stride=1, padding=1, weight_attr=nn.initializer.KaimingNormal())
        self.conv4 = nn.Conv2D(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, weight_attr=nn.initializer.KaimingNormal())
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2D(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        self.dueling = dueling
        if dueling:
            self.linear_1_adv = nn.Linear(in_features=6400, out_features=512, weight_attr=nn.initializer.KaimingNormal())
            self.linear_2_adv = nn.Linear(in_features=512, out_features=act_dim)
            self.linear_1_val = nn.Linear(in_features=6400, out_features=512, weight_attr=nn.initializer.KaimingNormal())
            self.linear_2_val = nn.Linear(in_features=512, out_features=1)
        else:
            self.linear_1 = nn.Linear(in_features=6400, out_features=act_dim)
    def forward(self, obs):
        obs = obs / 255.0
        out = self.max_pool(self.relu(self.conv1(obs)))
        out = self.max_pool(self.relu(self.conv2(out)))
        out = self.max_pool(self.relu(self.conv3(out)))
        out = self.relu(self.conv4(out))
        out = self.flatten(out)
        if self.dueling:
            As = self.relu(self.linear_1_adv(out))
            As = self.linear_2_adv(As)
            V = self.relu(self.linear_1_val(out))
            V = self.linear_2_val(V)
            Q = As + (V - As.mean(axis=1, keepdim=True))
        else:
            Q = self.linear_1(out)
        return Q

class MultiHeadDQN(parl.Algorithm):
    def __init__(self, model, gamma=None, lr=None):
        # checks
        check_model_method(model, 'forward', self.__class__.__name__)
        assert isinstance(gamma, float)
        assert isinstance(lr, float)

        self.model = model
        self.target_model = copy.deepcopy(model)

        self.gamma = gamma
        self.lr = lr

        self.mse_loss = paddle.nn.MSELoss(reduction='mean')
        self.optimizer = paddle.optimizer.Adam(
            learning_rate=lr, parameters=self.model.parameters())

    def predict(self, obs):
        return self.model(obs)

    def learn(self, obs, action, reward, next_obs, terminal, i):
        # Q
        pred_values = self.model(obs)[i]
        action_dim = pred_values.shape[-1]
        action = paddle.squeeze(action, axis=-1)
        action_onehot = paddle.nn.functional.one_hot(
            action, num_classes=action_dim)
        pred_value = pred_values * action_onehot
        pred_value = paddle.sum(pred_value, axis=1, keepdim=True)

        # target Q
        with paddle.no_grad():
            max_v = self.target_model(next_obs)[i].max(1, keepdim=True)
            target = reward + (1 - terminal) * self.gamma * max_v
        loss = self.mse_loss(pred_value, target)

        # optimize
        self.optimizer.clear_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def sync_target(self):
        self.model.sync_weights_to(self.target_model)

class Agent(parl.Agent):
    def __init__(self, algorithm, update_target_step, act_dim, e_greed=0.1, e_greed_decrement=0):
        super(Agent, self).__init__(algorithm)
        assert isinstance(act_dim, int)
        self.act_dim = act_dim

        self.global_step = 0
        self.update_target_steps = update_target_step
        self.e_greed = e_greed
        self.e_greed_decrement = e_greed_decrement

    def sample(self, obs):
        sample = np.random.random()
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)
        else:
            if np.random.random() < 0.01:
                act = np.random.randint(self.act_dim)
            else:
                act = self.predict(obs)
        self.e_greed = max(0.01, self.e_greed - self.e_greed_decrement)
        return act

    def predict(self, obs):
        obs = paddle.to_tensor(obs, dtype='float32')
        pred_q = self.alg.predict(obs)
        act = int(pred_q.argmax())
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)
        terminal = np.expand_dims(terminal, axis=-1)

        obs = paddle.to_tensor(obs, dtype='float32')
        act = paddle.to_tensor(act, dtype='int32')
        reward = paddle.to_tensor(reward, dtype='float32')
        next_obs = paddle.to_tensor(next_obs, dtype='float32')
        terminal = paddle.to_tensor(terminal, dtype='float32')
        loss = self.alg.learn(obs, act, reward, next_obs, terminal)
        return float(loss)

class MultiHeadAgent(parl.Agent):
    def __init__(self, algorithm, act_dim, e_greed=0.1, e_greed_decrement=0):
        super(MultiHeadAgent, self).__init__(algorithm)
        #assert isinstance(act_dim, int)
        self.act_dim = act_dim

        self.global_step = 0
        self.update_target_steps = 200
        self.e_greed = e_greed
        self.e_greed_decrement = e_greed_decrement

    def sample(self, obs):
        sample = np.random.random()
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)
        else:
            if np.random.random() < 0.01:
                act = np.random.randint(self.act_dim)
            else:
                act = self.predict(obs)
        self.e_greed = max(0.01, self.e_greed - self.e_greed_decrement)
        return act

    def predict(self, obs):
        obs = paddle.to_tensor(obs, dtype='float32')
        pred_q = self.alg.predict(obs)
        act = int(pred_q[0].argmax())
        ma = max(pred_q[0])
        for i in range(len(pred_q)):
            if ma < max(pred_q[i]):
                ma = max(pred_q[i])
                act = int(pred_q[i].argmax())
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        for i in range(len(reward[0])):
            for k in range(len(reward)):
                rewards = reward[k][i]
            act = np.expand_dims(act, axis=-1)
            rewards = np.expand_dims(rewards, axis=-1)
            terminal = np.expand_dims(terminal, axis=-1)

            obs = paddle.to_tensor(obs, dtype='float32')
            act = paddle.to_tensor(act, dtype='int32')
            rewards = paddle.to_tensor(rewards, dtype='float32')
            next_obs = paddle.to_tensor(next_obs, dtype='float32')
            terminal = paddle.to_tensor(terminal, dtype='float32')
            loss = self.alg.learn(obs, act, rewards, next_obs, terminal, i)
        return float(loss)

class AtariAgent(parl.Agent):
    def __init__(self, algorithm, act_dim, start_lr, total_step, update_target_step):
        super().__init__(algorithm)
        self.global_update_step = 0
        self.update_target_step = update_target_step
        self.act_dim = act_dim
        self.curr_ep = 1
        self.ep_end = 0.1
        self.lr_end = 0.00001

        self.ep_scheduler = LinearDecayScheduler(1, total_step)
        self.lr_scheduler = LinearDecayScheduler(start_lr, total_step)
    def sample(self, obs):
        explore = np.random.choice([True, False], p=[self.curr_ep, 1 - self.curr_ep])
        if explore:
            act = np.random.randint(self.act_dim)
        else:
            act = self.predict(obs)
        self.curr_ep = max(self.ep_scheduler.step(1), self.ep_end)
        return act
    def predict(self, obs):
        if obs.ndim == 3:  # if obs is 3 dimensional, we need to expand it to have batch_size = 1
            obs = np.expand_dims(obs, axis=0)
        obs = paddle.to_tensor(obs, dtype='float32')
        pred_q = self.alg.predict(obs).detach().numpy().squeeze()

        best_actions = np.where(pred_q == pred_q.max())[0]
        act = np.random.choice(best_actions)
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        if self.global_update_step % self.update_target_step == 0:
            self.alg.sync_target()
        self.global_update_step += 1

        reward = np.clip(reward, -1, 1)
        act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)
        terminal = np.expand_dims(terminal, axis=-1)

        obs = paddle.to_tensor(obs, dtype='float32')
        act = paddle.to_tensor(act, dtype='int32')
        reward = paddle.to_tensor(reward, dtype='float32')
        next_obs = paddle.to_tensor(next_obs, dtype='float32')
        terminal = paddle.to_tensor(terminal, dtype='float32')

        loss = self.alg.learn(obs, act, reward, next_obs, terminal)

        # learning rate decay
        self.alg.optimizer.set_lr(max(self.lr_scheduler.step(1), self.lr_end))
        return float(loss)

class ReplayMemory(object):
    def __init__(self, max_size, obs_dim, act_dim, MO):
        self.max_size = int(max_size)
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.obs = np.zeros((max_size, obs_dim), dtype='float32')
        if act_dim == 0:  # Discrete control environment
            self.action = np.zeros((max_size,), dtype='int32')
        else:  # Continuous control environment
            self.action = np.zeros((max_size, act_dim), dtype='float32')
        if MO:
            self.reward = [[] for _ in range(max_size)]
        else:
            self.reward = np.zeros((max_size,), dtype='float32')
        self.terminal = np.zeros((max_size,), dtype='bool')
        self.next_obs = np.zeros((max_size, obs_dim), dtype='float32')

        self._curr_size = 0
        self._curr_pos = 0

    def sample_batch(self, batch_size):
        batch_idx = np.random.randint(self._curr_size, size=batch_size)
        obs = self.obs[batch_idx]
        reward = [self.reward[i] for i in batch_idx]
        action = self.action[batch_idx]
        next_obs = self.next_obs[batch_idx]
        terminal = self.terminal[batch_idx]
        return obs, action, reward, next_obs, terminal
    def make_index(self, batch_size):
        batch_idx = np.random.randint(self._curr_size, size=batch_size)
        return batch_idx
    def sample_batch_by_index(self, batch_idx):
        obs = self.obs[batch_idx]
        reward = self.reward[batch_idx]
        action = self.action[batch_idx]
        next_obs = self.next_obs[batch_idx]
        terminal = self.terminal[batch_idx]
        return obs, action, reward, next_obs, terminal
    def append(self, obs, act, reward, next_obs, terminal):
        if self._curr_size < self.max_size:
            self._curr_size += 1
        self.obs[self._curr_pos] = obs
        self.action[self._curr_pos] = act
        self.reward[self._curr_pos] = reward
        self.next_obs[self._curr_pos] = next_obs
        self.terminal[self._curr_pos] = terminal
        self._curr_pos = (self._curr_pos + 1) % self.max_size
    def size(self):
        return self._curr_size
    def __len__(self):
        return self._curr_size
    def save(self, pathname):
        other = np.array([self._curr_size, self._curr_pos], dtype=np.int32)
        np.savez(
            pathname,
            obs=self.obs,
            action=self.action,
            reward=self.reward,
            terminal=self.terminal,
            next_obs=self.next_obs,
            other=other)
    def load(self, pathname):
        data = np.load(pathname)
        other = data['other']
        if int(other[0]) > self.max_size:
            logger.warn('loading from a bigger size rpm!')
        self._curr_size = min(int(other[0]), self.max_size)
        self._curr_pos = min(int(other[1]), self.max_size - 1)

        self.obs[:self._curr_size] = data['obs'][:self._curr_size]
        self.action[:self._curr_size] = data['action'][:self._curr_size]
        self.reward[:self._curr_size] = data['reward'][:self._curr_size]
        self.terminal[:self._curr_size] = data['terminal'][:self._curr_size]
        self.next_obs[:self._curr_size] = data['next_obs'][:self._curr_size]
        logger.info("[load rpm]memory loade from {}".format(pathname))
    def load_from_d4rl(self, dataset):
        logger.info("Dataset Info: ")
        for key in dataset:
            logger.info('key: {},\tshape: {},\tdtype: {}'.format(
                key, dataset[key].shape, dataset[key].dtype))
        assert 'observations' in dataset
        assert 'next_observations' in dataset
        assert 'actions' in dataset
        assert 'rewards' in dataset
        assert 'terminals' in dataset

        self.obs = dataset['observations']
        self.next_obs = dataset['next_observations']
        self.action = dataset['actions']
        self.reward = dataset['rewards']
        self.terminal = dataset['terminals']
        self._curr_size = dataset['terminals'].shape[0]
        assert self._curr_size <= self.max_size, 'please set a proper max_size for ReplayMemory'
        logger.info('Number of terminals on: {}'.format(self.terminal.sum()))
duqu_best_MDDQN = True; duqu_MDDQN = False; duqu_best_NSGA2 = True  # 是否直接读取MDDQN已有结果，是：True，否：False
duqu_reMDDQN = True ; duqu_best_reNSGA2 = True # 是否直接读取reMDDQN已有结果
duqu_MDDQN_model = False ; duqu_reMDDQN_model = False # 是否直接读取训练的模型
Job_num = [5]
Mac_sum = [12]
AGV_num = [3,4,5]
XUNLIAN = 1
# ps=200;Pc=0.1;Pm=0.02
nfe = 1 # NSGA2迭代轮数
diedai = 125 # NSGA2迭代次数
max_episode = 5000  # DDQN训练次数
rule_episode = 200  # 规则训练次数

LEARN_FREQ = 5  # 训练频率
MEMORY_SIZE = 200000
MEMORY_WARMUP_SIZE = 200
BATCH_SIZE = 64
# LEARNING_RATE = 0.005#0.0005
# GAMMA = 0.85#0.99
UPDATE_TARGET_STEP = 200
EVAL_RENDER = False
Experience = namedtuple('Experience', ['obs', 'action', 'reward', 'isOver'])
# ACT_line, JOB_line, MAC_line, AGV_line = [],[],[],[]

# train an episode
def train_episode(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id, agent, env, rpm, NSGA2_fit1, NSGA2_fit2, NSGA2_fit3, MDDQN_fit1, MDDQN_fit2, MDDQN_fit3, make_max, Idle_sum, S_sum, TR, best_MDDQN, MO=True):
    total_reward = 0
    env.reset()
    obs = env.Features()  # 初始状态是否为0？？？
    obs = paddle.to_tensor(obs, dtype='float32')
    act_line, job_line, mac_line, agv_line = [],[],[],[]
    for step in range(State * J_num):  # 对工序遍历，每次选出一个动作和机器
        at = agent.sample(obs)
        #print(at)
        Job_id, Machine_id, Agv_id = Env_action(at+1, env)
        if step == State * J_num - 1:
            done = True
        else:
            done = False
        act_line.append(at); job_line.append(Job_id); mac_line.append(Machine_id); agv_line.append(Agv_id)
        env.scheduling(Job_id, Machine_id, Agv_id)  # 选择机器、工件和AGV后更新调度以及计算状态特征的数据
        obs_t = env.Features()  # 更新状态特征
        next_obs = paddle.to_tensor(obs_t, dtype='float32')
        '''执行动作后，根据新状态获得奖励'''  # 奖励函数根据第6个前后状态进行判断
        reward = env.reward2(step, NSGA2_fit1, NSGA2_fit2, NSGA2_fit3, MDDQN_fit1, MDDQN_fit2, MDDQN_fit3, MO)  # 根据新的状态获得奖励
        rpm.append(obs, at, reward, next_obs, done)
        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            # s,a,r,s',done
            (batch_obs, batch_action, batch_reward, batch_next_obs, batch_done) = rpm.sample_batch(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)
        if MO:
            total_reward += sum(reward)
        else:
            total_reward += reward
        obs = next_obs
        if done:
            break
    make_max.append(float(obs[5])); Idle_sum.append(float(obs[6])); S_sum.append(float(obs[7]))
    Zuiyou = True
    for i in range(1, v + 1):
        if not env.AGVs[i].using_time:
            Zuiyou = False
    if not best_MDDQN[0] and Zuiyou == True:
        best_MDDQN[0].append(float(obs[5])); best_MDDQN[0].append(float(obs[6])); best_MDDQN[0].append(float(obs[7]))
        best_MDDQN[1] = job_line; best_MDDQN[2] = mac_line; best_MDDQN[3] = agv_line
    else:
        if best_MDDQN[0][0]**2+best_MDDQN[0][1]**2+best_MDDQN[0][2]**2 > float(obs[5])**2+float(obs[6])**2+float(obs[7])**2 and Zuiyou == True:
            best_MDDQN[0][0] = float(obs[5]); best_MDDQN[0][1] = float(obs[6]); best_MDDQN[0][2] = float(obs[7])
            best_MDDQN[1] = job_line; best_MDDQN[2] = mac_line; best_MDDQN[3] = agv_line
    # ACT_line.append(act_line);JOB_line.append(job_line);MAC_line.append(mac_line);AGV_line.append(agv_line)
    TR.append(total_reward)
    return total_reward, float(obs[5]), float(obs[6]), float(obs[7]), make_max, Idle_sum, S_sum, TR, best_MDDQN
# train an episode
def retrain_episode(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id, agent, env, rpm, NSGA2_fit1, NSGA2_fit2, NSGA2_fit3, MDDQN_fit1, MDDQN_fit2, MDDQN_fit3, make_max, Idle_sum, S_sum, TR, best_reMDDQN, G1_job,G1_mac,G1_agv, MO=True):
    total_reward = 0
    env.reset()
    obs = env.Features()  # 初始状态是否为0？？？
    obs = paddle.to_tensor(obs, dtype='float32')
    act_line, job_line, mac_line, agv_line = [],[],[],[]
    for step in range(State * J_num):  # 对工序遍历，每次选出一个动作和机器
        at = agent.sample(obs)
        #print(at)
        if step < len(G1_job):
            Job_id, Machine_id, Agv_id = G1_job[step],G1_mac[step],G1_agv[step]
        else:
            Job_id, Machine_id, Agv_id = Env_action(at+1, env)
        if step == State * J_num - 1:
            done = True
        else:
            done = False
        act_line.append(at); job_line.append(Job_id); mac_line.append(Machine_id); agv_line.append(Agv_id)
        env.rescheduling(Job_id, Machine_id, Agv_id)  # 选择机器、工件和AGV后更新调度以及计算状态特征的数据
        obs_t = env.Features()  # 更新状态特征
        next_obs = paddle.to_tensor(obs_t, dtype='float32')
        '''执行动作后，根据新状态获得奖励'''  # 奖励函数根据第6个前后状态进行判断
        reward = env.reward2(step, NSGA2_fit1, NSGA2_fit2, NSGA2_fit3, MDDQN_fit1, MDDQN_fit2, MDDQN_fit3, MO)  # 根据新的状态获得奖励
        rpm.append(obs, at, reward, next_obs, done)
        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            # s,a,r,s',done
            (batch_obs, batch_action, batch_reward, batch_next_obs, batch_done) = rpm.sample_batch(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)
        if MO:
            total_reward += sum(reward)
        else:
            total_reward += reward
        obs = next_obs
        if done:
            break
    make_max.append(float(obs[5])); Idle_sum.append(float(obs[6])); S_sum.append(float(obs[7]))
    Zuiyou = True
    for i in range(1, v + 1):
        if not env.AGVs[i].using_time:
            Zuiyou = False
    if not best_reMDDQN[0] and Zuiyou == True:
        best_reMDDQN[0].append(float(obs[5])); best_reMDDQN[0].append(float(obs[6])); best_reMDDQN[0].append(float(obs[7]))
        best_reMDDQN[1] = job_line; best_reMDDQN[2] = mac_line; best_reMDDQN[3] = agv_line
    else:
        if best_reMDDQN[0][0]**2+best_reMDDQN[0][1]**2+best_reMDDQN[0][2]**2 > float(obs[5])**2+float(obs[6])**2+float(obs[7])**2 and Zuiyou == True:
            best_reMDDQN[0][0] = float(obs[5]); best_reMDDQN[0][1] = float(obs[6]); best_reMDDQN[0][2] = float(obs[7])
            best_reMDDQN[1] = job_line; best_reMDDQN[2] = mac_line; best_reMDDQN[3] = agv_line
    # ACT_line.append(act_line);JOB_line.append(job_line);MAC_line.append(mac_line);AGV_line.append(agv_line)
    TR.append(total_reward)
    return total_reward, float(obs[5]), float(obs[6]), float(obs[7]), make_max, Idle_sum, S_sum, TR, best_reMDDQN

# evaluate 5 episodes
def evaluate_episodes(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id, agent, NSGA2_fit1, NSGA2_fit2, NSGA2_fit3, MDDQN_fit1, MDDQN_fit2, MDDQN_fit3, MO=True):
    eval_reward = []
    for i in range(1):
        env = Situation(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id)
        obs = env.Features()  # 初始状态是否为0？？？
        obs = paddle.to_tensor(obs, dtype='float32')
        episode_reward = 0
        JOB_MACHINE, AT = [], []
        for s in range(State * J_num):  # 对工序遍历，每次选出一个动作和机器
            at = agent.predict(obs)
            Job_id, Machine_id, Agv_id = Env_action(at+1,env)
            JOB_MACHINE.append(Job_id)
            AT.append(at+1)
            '''每经过一次工序（包含工件及机器）更新学习状态和调度进程'''
            if s == State * J_num - 1:
                done = True
                print('选择工件', JOB_MACHINE)
                print('选择规则', AT)
            else:
                done = False
            env.scheduling(Job_id, Machine_id, Agv_id)  # 选择机器、工件和AGV后更新调度以及计算状态特征的数据
            obs_t = env.Features()  # 更新状态特征
            obs_t = paddle.to_tensor(obs_t, dtype='float32')
            '''执行动作后，根据新状态获得奖励'''  # 奖励函数根据第6个前后状态进行判断
            reward = env.reward2(s, NSGA2_fit1, NSGA2_fit2, NSGA2_fit3, MDDQN_fit1, MDDQN_fit2, MDDQN_fit3, MO)  # 根据新的状态获得奖励
            obs = obs_t  # 状态更新
            if MO:
                episode_reward += sum(reward)
            else:
                episode_reward += reward
            if done:
                break
        eval_reward.append(episode_reward)
        logger.info('完工时间:{}    机器空闲时间:{}   agv运输距离:{}'.format(int(obs[5]), int(obs[6]), int(obs[7])))
    return np.mean(eval_reward)
def reevaluate_episodes(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id, agent, NSGA2_fit1, NSGA2_fit2, NSGA2_fit3, MDDQN_fit1, MDDQN_fit2, MDDQN_fit3, G1_job,G1_mac,G1_agv,faultyMachine, faultyTime_Mac, repairTime_Mac, faultyAGV, faultyTime_AGV, repairTime_AGV, MO=True):
    eval_reward = []
    for i in range(1):
        env = Situation(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id)
        env.reschedule(faultyMachine, faultyTime_Mac, repairTime_Mac, faultyAGV, faultyTime_AGV, repairTime_AGV)
        obs = env.Features()  # 初始状态是否为0？？？
        obs = paddle.to_tensor(obs, dtype='float32')
        episode_reward = 0
        JOB_MACHINE, AT = [], []
        for s in range(State * J_num):  # 对工序遍历，每次选出一个动作和机器
            at = agent.predict(obs)
            if s < len(G1_job):
                Job_id, Machine_id, Agv_id = G1_job[s], G1_mac[s], G1_agv[s]
            else:
                Job_id, Machine_id, Agv_id = Env_action(at+1, env)
            JOB_MACHINE.append(Job_id)
            AT.append(at+1)
            '''每经过一次工序（包含工件及机器）更新学习状态和调度进程'''
            if s == State * J_num - 1:
                done = True
                print('选择工件', JOB_MACHINE)
                print('选择规则', AT)
            else:
                done = False
            env.rescheduling(Job_id, Machine_id, Agv_id)  # 选择机器、工件和AGV后更新调度以及计算状态特征的数据
            obs_t = env.Features()  # 更新状态特征
            obs_t = paddle.to_tensor(obs_t, dtype='float32')
            '''执行动作后，根据新状态获得奖励'''  # 奖励函数根据第6个前后状态进行判断
            reward = env.reward2(s, NSGA2_fit1, NSGA2_fit2, NSGA2_fit3, MDDQN_fit1, MDDQN_fit2, MDDQN_fit3, MO)  # 根据新的状态获得奖励
            obs = obs_t  # 状态更新
            if MO:
                episode_reward += sum(reward)
            else:
                episode_reward += reward
            if done:
                break
        eval_reward.append(episode_reward)
        logger.info('完工时间:{}    机器空闲时间:{}   agv运输距离:{}'.format(int(obs[5]), int(obs[6]), int(obs[7])))
    return np.mean(eval_reward)

def Env_action(at, env):
    if at == 1:
        Job_id, Machine_id, Agv_id = env.rule1()
    elif at == 2:
        Job_id, Machine_id, Agv_id = env.rule2()
    elif at == 3:
        Job_id, Machine_id, Agv_id = env.rule3()
    elif at == 4:
        Job_id, Machine_id, Agv_id = env.rule4()
    elif at == 5:
        Job_id, Machine_id, Agv_id = env.rule5()
    elif at == 6:
        Job_id, Machine_id, Agv_id = env.rule6()
    elif at == 7:
        Job_id, Machine_id, Agv_id = env.rule7()
    elif at == 8:
        Job_id, Machine_id, Agv_id = env.rule8()
    elif at == 9:
        Job_id, Machine_id, Agv_id = env.rule9()
    elif at == 10:
        Job_id, Machine_id, Agv_id = env.rule10()
    elif at == 11:
        Job_id, Machine_id, Agv_id = env.rule11()
    elif at == 12:
        Job_id, Machine_id, Agv_id = env.rule12()
    elif at == 13:
        Job_id, Machine_id, Agv_id = env.rule13()
    elif at == 14:
        Job_id, Machine_id, Agv_id = env.rule14()
    elif at == 15:
        Job_id, Machine_id, Agv_id = env.rule15()
    elif at == 16:
        Job_id, Machine_id, Agv_id = env.rule16()
    elif at == 17:
        Job_id, Machine_id, Agv_id = env.rule17()
    elif at == 18:
        Job_id, Machine_id, Agv_id = env.rule18()
    elif at == 19:
        Job_id, Machine_id, Agv_id = env.rule19()
    elif at == 20:
        Job_id, Machine_id, Agv_id = env.rule20()
    elif at == 21:
        Job_id, Machine_id, Agv_id = env.rule21()
    elif at == 22:
        Job_id, Machine_id, Agv_id = env.rule22()
    elif at == 23:
        Job_id, Machine_id, Agv_id = env.rule23()
    elif at == 24:
        Job_id, Machine_id, Agv_id = env.rule24()
    elif at == 25:
        Job_id, Machine_id, Agv_id = env.rule25()
    elif at == 26:
        Job_id, Machine_id, Agv_id = env.rule26()
    elif at == 27:
        Job_id, Machine_id, Agv_id = env.rule27()
    return Job_id, Machine_id, Agv_id

def NSGA(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id, ps, Pc, Pm):
    MaxNFEs = diedai*2*ps # 第一个数字为NSGA2迭代次数
    AP = []
    AFit = []  # 精英归档
    # 针对一定数量的轮次进行迭代
    for rround in range(nfe):
        # 初始化染色体和适应度值
        p_chrom_job = RandomRule2(ps, J_num, State)
        p_chrom_mac = []
        for i in range(ps):
            tmp = []
            for j in range(State):
                tmp.extend([random.randint(0, M_j[j+1]-1) for _ in range(J_num)])
            p_chrom_mac.append(tmp)
        p_chrom_agv = [[random.randint(1, v) for _ in range(State*J_num)] for _ in range(ps)]
        fitness = [[0 for _ in range(3)] for _ in range(ps)]
        NFEs = 0  # 函数评估次数
        # 计算每个解的适应度
        for i in range(ps):
            s = Situation(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id)
            s.Stage_Decode3(p_chrom_job[i],p_chrom_mac[i],p_chrom_agv[i])
            fitness[i][0] = s.fitness1  # 加工时间
            fitness[i][1] = s.fitness2  # 机器空闲时间
            fitness[i][2] = s.fitness3  # AGV运行距离
        i = 1
        # 主要的优化循环
        while NFEs < MaxNFEs:
            print(' 第', rround + 1, '轮，迭代次数', i)
            i = i + 1
            # 应用 NSGA2POXES 算法更新染色体和适应度
            p_chrom_job, p_chrom_mac, p_chrom_agv, fitness = NSGA2POXES4(p_chrom_job, p_chrom_mac, p_chrom_agv, fitness, Pc, Pm, ps, M_j,n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id)
            NFEs += 2 * ps
            fitness_Z = Zhuanzhi(fitness)
            PF = pareto(fitness_Z)  # 包含帕累托前沿解在原适应度值数组中的索引的列表
            # 更新精英归档以保存非支配解
            if len(AFit) == 0:
                AP = copy.copy([p_chrom_job[index] for index in PF])
                AM = copy.copy([p_chrom_mac[index] for index in PF])
                AA = copy.copy([p_chrom_agv[index] for index in PF])
                AFit = copy.copy([fitness[index] for index in PF])
            else:  # 将新的帕累托前沿解添加到 AP AF AFit中
                AP = np.vstack((AP, [p_chrom_job[index] for index in PF]))
                AM = np.vstack((AM, [p_chrom_mac[index] for index in PF]))
                AA = np.vstack((AA, [p_chrom_agv[index] for index in PF]))
                AFit = np.vstack((AFit, [fitness[index] for index in PF]))
    AFit_Z = Zhuanzhi(AFit)
    PF = pareto(AFit_Z)
    AP3D, AM3D, AA3D = [], [], []
    make_span3D, Idle_time3D, Agv_S3D = [], [], []
    for i in PF:
        if AFit[i][1]>5:
            if AFit[i][0] not in make_span3D or AFit[i][1] not in Idle_time3D or AFit[i][2] not in Agv_S3D:
                AP3D.append(AP[i]);AM3D.append(AM[i]);AA3D.append(AA[i])
                make_span3D.append(AFit[i][0])
                Idle_time3D.append(AFit[i][1])
                Agv_S3D.append(AFit[i][2])
    pf = MO_Best(make_span3D, Idle_time3D, Agv_S3D)
    #pf = random.randint(0, len(make_span3D) - 1)
    NSGA2_fit = []
    NSGA2_fit.append(make_span3D)
    NSGA2_fit.append(Idle_time3D)
    NSGA2_fit.append(Agv_S3D)
    # s = Situation(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id)
    # s.Stage_Decode3(AP3D[pf], AM3D[pf], AA3D[pf])
    # s.Gantt()

    return AP3D, AM3D, AA3D, NSGA2_fit, pf, AFit_Z
def reNSGA(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id, ps, Pc, Pm, faultyMachine, faultyTime_Mac, repairTime_Mac, faultyAGV, faultyTime_AGV, repairTime_AGV, G1_job, G1_mac, G1_agv):
    MaxNFEs = diedai*2*ps # 第一个数字为NSGA2迭代次数
    AP = []
    AFit = []  # 精英归档
    # 针对一定数量的轮次进行迭代
    for rround in range(nfe):
        # 初始化染色体和适应度值
        p_chrom_job = RandomRule2(ps, J_num, State)
        for i in range(ps):
            p_chrom_job[i] = TiHuanQianDuan(G1_job, p_chrom_job[i])
        p_chrom_mac = []
        for i in range(ps):
            tmp = []
            for j in range(State):
                tmp.extend([random.randint(0, M_j[j+1]-1) for _ in range(J_num)])
            p_chrom_mac.append(tmp)
        G1_mac_s1, G1_mac_s2, G1_mac_s3 = [],[],[]#split_by_occurrence(G1_mac)
        job1,job2=[],[]
        for i in range(len(G1_job)):
            if G1_job[i] not in job1:
                job1.append(G1_job[i])
                m_sta, m_id = find_element_index(MNN, G1_mac[i])
                G1_mac_s1.append(m_id)
            elif G1_job[i] not in job2:
                job2.append(G1_job[i])
                m_sta, m_id = find_element_index(MNN, G1_mac[i])
                G1_mac_s2.append(m_id)
            else:
                m_sta, m_id = find_element_index(MNN, G1_mac[i])
                G1_mac_s3.append(m_id)
        for i in range(ps):
            p_chrom_mac_s1 = TiHuanQianDuan(G1_mac_s1, p_chrom_mac[i][:J_num])
            p_chrom_mac_s2 = TiHuanQianDuan(G1_mac_s2, p_chrom_mac[i][J_num:J_num * 2])
            p_chrom_mac_s3 = TiHuanQianDuan(G1_mac_s3, p_chrom_mac[i][J_num * 2:J_num * 3])
            p_chrom_mac[i] = p_chrom_mac_s1 + p_chrom_mac_s2 + p_chrom_mac_s3
        p_chrom_agv = [[random.randint(1, v) for _ in range(State*J_num)] for _ in range(ps)]
        for i in range(ps):
            p_chrom_agv[i] = TiHuanQianDuan(G1_agv, p_chrom_agv[i])
        fitness = [[0 for _ in range(3)] for _ in range(ps)]
        NFEs = 0  # 函数评估次数
        # 计算每个解的适应度
        for i in range(ps):
            s = Situation(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id)
            s.reschedule(faultyMachine, faultyTime_Mac, repairTime_Mac, faultyAGV, faultyTime_AGV, repairTime_AGV)
            s.reStage_Decode3(p_chrom_job[i],p_chrom_mac[i],p_chrom_agv[i])
            fitness[i][0] = s.fitness1  # 加工时间
            fitness[i][1] = s.fitness2  # 机器空闲时间
            fitness[i][2] = s.fitness3  # AGV运行距离
        i = 1
        # 主要的优化循环
        while NFEs < MaxNFEs:
            print(' 第', rround + 1, '轮，迭代次数', i)
            i = i + 1
            # 应用 NSGA2POXES 算法更新染色体和适应度
            p_chrom_job, p_chrom_mac, p_chrom_agv, fitness = reNSGA2POXES4(p_chrom_job, p_chrom_mac, p_chrom_agv, fitness, Pc, Pm, ps, M_j,n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id, G1_job, G1_mac, G1_agv)
            NFEs += 2 * ps
            fitness_Z = Zhuanzhi(fitness)
            PF = pareto(fitness_Z)  # 包含帕累托前沿解在原适应度值数组中的索引的列表
            # 更新精英归档以保存非支配解
            if len(AFit) == 0:
                AP = copy.copy([p_chrom_job[index] for index in PF])
                AM = copy.copy([p_chrom_mac[index] for index in PF])
                AA = copy.copy([p_chrom_agv[index] for index in PF])
                AFit = copy.copy([fitness[index] for index in PF])
            else:  # 将新的帕累托前沿解添加到 AP AF AFit中
                AP = np.vstack((AP, [p_chrom_job[index] for index in PF]))
                AM = np.vstack((AM, [p_chrom_mac[index] for index in PF]))
                AA = np.vstack((AA, [p_chrom_agv[index] for index in PF]))
                AFit = np.vstack((AFit, [fitness[index] for index in PF]))
    AFit_Z = Zhuanzhi(AFit)
    PF = pareto(AFit_Z)
    AP3D, AM3D, AA3D = [], [], []
    make_span3D, Idle_time3D, Agv_S3D = [], [], []
    for i in PF:
        if AFit[i][1]>5:
            if AFit[i][0] not in make_span3D or AFit[i][1] not in Idle_time3D or AFit[i][2] not in Agv_S3D:
                AP3D.append(AP[i]);AM3D.append(AM[i]);AA3D.append(AA[i])
                make_span3D.append(AFit[i][0])
                Idle_time3D.append(AFit[i][1])
                Agv_S3D.append(AFit[i][2])
    pf = MO_Best(make_span3D, Idle_time3D, Agv_S3D)
    #pf = random.randint(0, len(make_span3D) - 1)
    NSGA2_fit = [make_span3D, Idle_time3D, Agv_S3D]
    # s = Situation(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id)
    # s.reStage_Decode3(AP3D[pf], AM3D[pf], AA3D[pf])
    # s.Gantt()
    return AP3D, AM3D, AA3D, NSGA2_fit, pf, AFit_Z
def find_element_index(mat, target):# 返回目标元素所在的行和列索引
    for i in range(len(mat)):
        for j in range(len(mat[i])):
            if mat[i][j] == target:
                return i, j
def train_episode_MDDQN(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id,agent, env, rpm, make_max, Idle_sum, S_sum, TR, MO=True):
    total_reward = 0
    env.reset()
    obs = env.Features()  # 初始状态是否为0？？？
    obs = paddle.to_tensor(obs, dtype='float32')
    for step in range(State * J_num):  # 对工序遍历，每次选出一个动作和机器
        at = agent.sample(obs)
        #print(at)
        Job_id, Machine_id, Agv_id = Env_action(at+1, env)
        if step == State * J_num - 1:
            done = True
        else:
            done = False
        env.scheduling(Job_id, Machine_id, Agv_id)  # 选择机器、工件和AGV后更新调度以及计算状态特征的数据
        obs_t = env.Features()  # 更新状态特征
        next_obs = paddle.to_tensor(obs_t, dtype='float32')
        '''执行动作后，根据新状态获得奖励'''  # 奖励函数根据第6个前后状态进行判断
        reward = env.reward(obs, obs_t, 1, MO)  # 根据新的状态获得奖励
        rpm.append(obs, at, reward, next_obs, done)
        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            # s,a,r,s',done
            (batch_obs, batch_action, batch_reward, batch_next_obs, batch_done) = rpm.sample_batch(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)
        if MO:
            total_reward += sum(reward)
        else:
            total_reward += reward
        obs = next_obs
        if done:
            break
    make_max.append(int(obs[5])); Idle_sum.append(int(obs[6])); S_sum.append(int(obs[7])); TR.append(total_reward)
    return total_reward, int(obs[5]), int(obs[6]), int(obs[7]), make_max, Idle_sum, S_sum, TR
def retrain_episode_MDDQN(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id,agent, env, rpm, make_max, Idle_sum, S_sum, TR, MO=True):
    total_reward = 0
    env.reset()
    obs = env.Features()  # 初始状态是否为0？？？
    obs = paddle.to_tensor(obs, dtype='float32')
    for step in range(State * J_num):  # 对工序遍历，每次选出一个动作和机器
        at = agent.sample(obs)
        #print(at)
        Job_id, Machine_id, Agv_id = Env_action(at+1, env)
        if step == State * J_num - 1:
            done = True
        else:
            done = False
        env.rescheduling(Job_id, Machine_id, Agv_id)  # 选择机器、工件和AGV后更新调度以及计算状态特征的数据
        obs_t = env.Features()  # 更新状态特征
        next_obs = paddle.to_tensor(obs_t, dtype='float32')
        '''执行动作后，根据新状态获得奖励'''  # 奖励函数根据第6个前后状态进行判断
        reward = env.reward(obs, obs_t, 1, MO)  # 根据新的状态获得奖励
        rpm.append(obs, at, reward, next_obs, done)
        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            # s,a,r,s',done
            (batch_obs, batch_action, batch_reward, batch_next_obs, batch_done) = rpm.sample_batch(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)
        if MO:
            total_reward += sum(reward)
        else:
            total_reward += reward
        obs = next_obs
        if done:
            break
    make_max.append(int(obs[5])); Idle_sum.append(int(obs[6])); S_sum.append(int(obs[7])); TR.append(total_reward)
    return total_reward, int(obs[5]), int(obs[6]), int(obs[7]), make_max, Idle_sum, S_sum, TR
def MDDQN_main(algoname, max_episode, obs_dim, act_dim, xunlian, gongjianzhongshu, jiqishu, AGVshu, ps, Pc, Pm, LEARNING_RATE, GAMMA, tanlan, Multi_O=True, baocun=True):
    n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id = Suanli_Read(xunlian, gongjianzhongshu, jiqishu, AGVshu)
    MO = Multi_O; BC = baocun
    env = Situation(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id)
    make_max, Idle_sum, S_sum, TR = [], [], [], []  # 3个目标
    all, make_span3D, Idle_time3D, Agv_S3D = [], [], [], []
    # set action_shape = 0 while in discrete control environment
    rpm = ReplayMemory(MEMORY_SIZE, obs_dim, 0, MO)
    # if os.path.exists('model/MDDQN{}_v{}.ckpt'.format(suanli, v)):
    #     agent.restore('model/MDDQN{}_v{}.ckpt'.format(suanli, v))
    # build an agent
    if MO:
        model = MultiHeadQNetwork(obs_dim=obs_dim, act_dim=act_dim)
        alg = MultiHeadDQN(model, gamma=GAMMA, lr=LEARNING_RATE) if algoname == 'MDQN' else MultiHeadDDQN(model, gamma=GAMMA, lr=LEARNING_RATE)
        agent = MultiHeadAgent(alg, act_dim=act_dim, e_greed=tanlan, e_greed_decrement=1e-6)
    else:
        model = QNetwork(obs_dim=obs_dim, act_dim=act_dim)
        alg = DQN(model, gamma=GAMMA, lr=LEARNING_RATE) if algoname == 'MDQN' else DDQN(model, gamma=GAMMA, lr=LEARNING_RATE)
        agent = Agent(alg, UPDATE_TARGET_STEP, act_dim=act_dim, e_greed=tanlan, e_greed_decrement=1e-6)
    if os.path.exists('model/MDDQN{}_v{}.ckpt'.format(suanli, v)) and duqu_MDDQN_model == True:
        agent.restore('model/MDDQN{}_v{}.ckpt'.format(suanli, v))
    # warmup memory
    time_start = time.time()
    while len(rpm) < MEMORY_WARMUP_SIZE:
        train_episode_MDDQN(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id,agent, env, rpm, make_max, Idle_sum, S_sum, TR, MO)
    # start training
    episode = 0
    while episode < max_episode:
        # train part
        ALL = [[], [], []]
        for i in trange(50):
            total_reward, a, b, c, make_max, Idle_sum, S_sum, TR = train_episode_MDDQN(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id,agent, env, rpm, make_max, Idle_sum, S_sum, TR, MO)
            ALL[0].append(a); ALL[1].append(b); ALL[2].append(c)
            episode += 1
        PF = pareto(ALL)
        make_span3D.extend([ALL[0][index] for index in PF])
        Idle_time3D.extend([ALL[1][index] for index in PF])
        Agv_S3D.extend([ALL[2][index] for index in PF])
        logger.info('episode:{}    e_greed:{}'.format(episode, agent.e_greed))

    # # save the parameters to ./model.ckpt
    # save_path = './model.ckpt'
    # agent.save(save_path)
    # # save the model and parameters of policy network for inference
    # save_inference_path = './inference_model'
    # input_shapes = [[None, obs_dim]]
    # input_dtypes = ['float32']
    # agent.save_inference_model(save_inference_path, input_shapes, input_dtypes)
    time_end2 = time.time()
    logger.info('{}花费时间:{}分钟{}秒'.format(algoname, int((time_end2-time_start)/60), round((time_end2-time_start)%60,1)))
    all.append(make_span3D); all.append(Idle_time3D); all.append(Agv_S3D); PF = pareto(all)
    make_span3D, Idle_time3D, Agv_S3D = [], [], []
    for i in PF:
        make_span3D.append(all[0][i])
        Idle_time3D.append(all[1][i])
        Agv_S3D.append(all[2][i])
    make_span3D, Idle_time3D, Agv_S3D = Zuixiaojuli_Top20(make_span3D, Idle_time3D, Agv_S3D)
    MDDQN_all = [make_span3D,Idle_time3D,Agv_S3D]; MDDQN_ALL = [make_max,Idle_sum,S_sum,TR]
    if BC:
        with open('data/MDDQN/AGV{}/MDDQN{}_v{}.pkl'.format(v, suanli, v), 'wb') as f:
            pickle.dump(MDDQN_all, f)
        with open('data/MDDQN/AGV{}/MDDQN_ALL{}_v{}.pkl'.format(v, suanli, v), 'wb') as f:
            pickle.dump(MDDQN_ALL, f)
    return MDDQN_all,MDDQN_ALL
def reMDDQN_main(algoname, max_episode, obs_dim, act_dim, xunlian, gongjianzhongshu, jiqishu, AGVshu, ps, Pc, Pm, LEARNING_RATE, GAMMA, tanlan, faultyMachine, faultyTime_Mac, repairTime_Mac, faultyAGV, faultyTime_AGV, repairTime_AGV, Multi_O=True, baocun=True):
    n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id = Suanli_Read(xunlian, gongjianzhongshu, jiqishu, AGVshu)
    MO = Multi_O; BC = baocun
    env = Situation(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id)
    env.reschedule(faultyMachine, faultyTime_Mac, repairTime_Mac, faultyAGV, faultyTime_AGV, repairTime_AGV)
    make_max, Idle_sum, S_sum, TR = [], [], [], []  # 3个目标
    all, make_span3D, Idle_time3D, Agv_S3D = [], [], [], []
    # set action_shape = 0 while in discrete control environment
    rpm = ReplayMemory(MEMORY_SIZE, obs_dim, 0, MO)

    # build an agent
    if MO:
        model = MultiHeadQNetwork(obs_dim=obs_dim, act_dim=act_dim)
        alg = MultiHeadDQN(model, gamma=GAMMA, lr=LEARNING_RATE) if algoname == 'MDQN' else MultiHeadDDQN(model, gamma=GAMMA, lr=LEARNING_RATE)
        agent = MultiHeadAgent(alg, act_dim=act_dim, e_greed=tanlan, e_greed_decrement=1e-6)
    else:
        model = QNetwork(obs_dim=obs_dim, act_dim=act_dim)
        alg = DQN(model, gamma=GAMMA, lr=LEARNING_RATE) if algoname == 'MDQN' else DDQN(model, gamma=GAMMA, lr=LEARNING_RATE)
        agent = Agent(alg, UPDATE_TARGET_STEP, act_dim=act_dim, e_greed=tanlan, e_greed_decrement=1e-6)
    if os.path.exists('model/reMDDQN{}_v{}.ckpt'.format(suanli, v)) and duqu_reMDDQN_model == True:
        agent.restore('model/reMDDQN{}_v{}.ckpt'.format(suanli, v))
    # warmup memory
    time_start = time.time()
    while len(rpm) < MEMORY_WARMUP_SIZE:
        retrain_episode_MDDQN(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id,agent, env, rpm, make_max, Idle_sum, S_sum, TR, MO)
    # start training
    episode = 0
    while episode < max_episode:
        # train part
        ALL = [[], [], []]
        for i in trange(50):
            total_reward, a, b, c, make_max, Idle_sum, S_sum, TR = retrain_episode_MDDQN(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id,agent, env, rpm, make_max, Idle_sum, S_sum, TR, MO)
            ALL[0].append(a); ALL[1].append(b); ALL[2].append(c)
            episode += 1
        PF = pareto(ALL)
        make_span3D.extend([ALL[0][index] for index in PF])
        Idle_time3D.extend([ALL[1][index] for index in PF])
        Agv_S3D.extend([ALL[2][index] for index in PF])
        logger.info('episode:{}    e_greed:{}'.format(episode, agent.e_greed))

    time_end2 = time.time()
    logger.info('{}花费时间:{}分钟{}秒'.format(algoname, int((time_end2-time_start)/60), round((time_end2-time_start)%60,1)))
    all.append(make_span3D); all.append(Idle_time3D); all.append(Agv_S3D); PF = pareto(all)
    make_span3D, Idle_time3D, Agv_S3D = [], [], []
    for i in PF:
        make_span3D.append(all[0][i])
        Idle_time3D.append(all[1][i])
        Agv_S3D.append(all[2][i])
    make_span3D, Idle_time3D, Agv_S3D = Zuixiaojuli_Top20(make_span3D, Idle_time3D, Agv_S3D)
    reMDDQN_all = [make_span3D,Idle_time3D,Agv_S3D]; reMDDQN_ALL = [make_max,Idle_sum,S_sum,TR]
    if BC:
        with open('data/reMDDQN/AGV{}/reMDDQN{}_v{}.pkl'.format(v, suanli, v), 'wb') as f:
            pickle.dump(reMDDQN_all, f)
        with open('data/reMDDQN/AGV{}/reMDDQN_ALL{}_v{}.pkl'.format(v, suanli, v), 'wb') as f:
            pickle.dump(reMDDQN_ALL, f)
    return reMDDQN_all,reMDDQN_ALL
def Get_J_s(env, J_num, JNN, suanli, v, suanfa):
    J_start = env.Get_J_start()
    data_J_start = pd.DataFrame({'J_num': [], 'state1': [], 'state2': [], 'state3': []})
    for i in range(1,J_num+1):
        j_arr, j_bat = env.find_element_index(JNN, i)  # 工件种类序号 批次号
        data_J_start.loc['O_{}{}'.format(j_arr, j_bat+1), :] = ['O_{}{}'.format(j_arr, j_bat+1), J_start[i-1][0], J_start[i-1][1], J_start[i-1][2]]
    loc_name = '{}{}_v{}'.format(suanfa, suanli, v); data_out = 'data_out/240401/J_start.xlsx'
    if os.path.exists(data_out):
        excel_file = pd.ExcelFile(data_out)
        # 获取所有表单名
        sheet_names = excel_file.sheet_names
        if loc_name in sheet_names:
            # 删除指定表单
            df = excel_file.parse(loc_name)
            df.drop(df.index, inplace=True)
            # 保存修政后的excel文件
            writer = pd.ExcelWriter(data_out)
            for sheet_name in sheet_names:
                if sheet_name != loc_name:
                    excel_file.parse(sheet_name).to_excel(writer, sheet_name=sheet_name, index=False)
            writer.close()
        with pd.ExcelWriter(data_out, mode='a', engine='openpyxl') as writer:
            data_J_start.to_excel(writer, sheet_name=loc_name, index=False)
    else:
        data_J_start.to_excel(data_out, sheet_name=loc_name, index=False)

def main(xunlian, gongjianzhongshu, jiqishu, AGVshu, ps, Pc, Pm, LEARNING_RATE, GAMMA, tanlan, Multi_O=True):
    n, J_num, State, M_j, PT, D_pq, JNN, v, V_r, JN_ip, ST_i, MNN, suanli, xunlian_id = Suanli_Read(xunlian, gongjianzhongshu, jiqishu, AGVshu)
    env = Situation(n, J_num, State, M_j, PT, D_pq, JNN, v, V_r, JN_ip, ST_i, MNN, suanli, xunlian_id)
    MO = Multi_O
    obs_dim = 8
    act_dim = 27
    algo_name = args.algo; max_episode = args.max_episode
    make_max, Idle_sum, S_sum, TR = [], [], [], []  # 3个目标
    all, make_span3D, Idle_time3D, Agv_S3D = [], [], [], []
    best_MDDQN = [[], [], [], []]

    '''————————————————未发生故障下调度算法————————————————'''
    if os.path.exists('data/best_MDDQN/best_MDDQN{}_v{}.pkl'.format(suanli, v)) and duqu_best_MDDQN == True:
        with open('data/best_MDDQN/best_MDDQN{}_v{}.pkl'.format(suanli, v), 'rb') as f:
            best_MDDQN_load = pickle.load(f)
    else:
        rpm = ReplayMemory(MEMORY_SIZE, obs_dim, 0, MO)
        # build an agent
        if MO:
            model = MultiHeadQNetwork(obs_dim=obs_dim, act_dim=act_dim)
            alg = MultiHeadDQN(model, gamma=GAMMA, lr=LEARNING_RATE)
            agent = MultiHeadAgent(alg, act_dim=act_dim, e_greed=tanlan, e_greed_decrement=1e-6)
        else:
            model = QNetwork(obs_dim=obs_dim, act_dim=act_dim)
            alg = DQN(model, gamma=GAMMA, lr=LEARNING_RATE) if algo_name == 'NSGA2_MDQN' else DDQN(model, gamma=GAMMA, lr=LEARNING_RATE)
            agent = Agent(alg, UPDATE_TARGET_STEP, act_dim=act_dim, e_greed=tanlan, e_greed_decrement=1e-6)
        # warmup memory
        if os.path.exists('model/MDDQN{}_v{}.ckpt'.format(suanli, v)) and duqu_MDDQN_model == True:
            agent.restore('model/MDDQN{}_v{}.ckpt'.format(suanli, v))
            max_episode = 50
        if os.path.exists('data/MDDQN/AGV{}/MDDQN{}_v{}.pkl'.format(v, suanli, v)) and duqu_MDDQN == True:
            with open('data/MDDQN/AGV{}/MDDQN{}_v{}.pkl'.format(v, suanli, v), 'rb') as f:
                MDDQN_all = pickle.load(f)
            with open('data/MDDQN/AGV{}/MDDQN_ALL{}_v{}.pkl'.format(v, suanli, v), 'rb') as f:
                MDDQN_ALL = pickle.load(f)
        else:
            MDDQN_all, MDDQN_ALL = MDDQN_main('DDQN', max_episode, obs_dim, act_dim, xunlian, gongjianzhongshu, jiqishu, AGVshu, ps, Pc, Pm, LEARNING_RATE, GAMMA, tanlan, Multi_O, True)
        time_start = time.time()
        if os.path.exists('data/best_NSGA/best_NSGA{}_v{}.pkl'.format(suanli, v)) and duqu_best_NSGA2 == True:
            with open('data/best_NSGA/best_NSGA{}_v{}.pkl'.format(suanli, v), 'rb') as f:
                best_NSGA = pickle.load(f)
            AP, AM, AA, NSGA2_fit, pf, NSGA2_ALL = best_NSGA[0], best_NSGA[1], best_NSGA[2], best_NSGA[3], best_NSGA[4], best_NSGA[5],
        else:
            AP, AM, AA, NSGA2_fit, pf, NSGA2_ALL = NSGA(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id, ps, Pc, Pm)
            best_NSGA = [AP, AM, AA, NSGA2_fit, pf, NSGA2_ALL]
            with open('data/best_NSGA/best_NSGA{}_v{}.pkl'.format(suanli, v), 'wb') as f:
                pickle.dump(best_NSGA, f)
        time_end1 = time.time()
        s = Situation(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id)
        NSGA2_fit1, NSGA2_fit2, NSGA2_fit3 = s.Stage_Decode4(AP[pf], AM[pf], AA[pf])
        while len(rpm) < MEMORY_WARMUP_SIZE:
            if len(rpm) == 0:
                MDDQN_fit1, MDDQN_fit2, MDDQN_fit3 = NSGA2_fit1, NSGA2_fit2, NSGA2_fit3
            else:
                MDDQN_fit1, MDDQN_fit2, MDDQN_fit3, best_MDDQN = Huigun(n, J_num, State, M_j, PT, D_pq, JNN, v, V_r, JN_ip, ST_i, MNN, suanli, xunlian_id, best_MDDQN)
            train_episode(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id,agent, env, rpm, NSGA2_fit1, NSGA2_fit2, NSGA2_fit3, MDDQN_fit1, MDDQN_fit2, MDDQN_fit3, make_max, Idle_sum, S_sum, TR, best_MDDQN, MO)
        # start training
        episode = 0
        while episode < max_episode:
            # train part
            ALL = [[], [], []]
            for i in trange(50):
                MDDQN_fit1, MDDQN_fit2, MDDQN_fit3, best_MDDQN = Huigun(n,J_num,State,M_j,PT,D_pq,JNN, v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id, best_MDDQN)
                total_reward, a, b, c, make_max, Idle_sum, S_sum, TR, best_MDDQN = train_episode(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id,agent, env, rpm, NSGA2_fit1, NSGA2_fit2, NSGA2_fit3, MDDQN_fit1, MDDQN_fit2, MDDQN_fit3, make_max, Idle_sum, S_sum, TR, best_MDDQN, MO)
                ALL[0].append(a); ALL[1].append(b); ALL[2].append(c)
                episode += 1
            PF = pareto(ALL)
            make_span3D.extend([ALL[0][index] for index in PF])
            Idle_time3D.extend([ALL[1][index] for index in PF])
            Agv_S3D.extend([ALL[2][index] for index in PF])
            # test part
            eval_reward = evaluate_episodes(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id,agent, NSGA2_fit1, NSGA2_fit2, NSGA2_fit3, MDDQN_fit1, MDDQN_fit2, MDDQN_fit3, MO)
            logger.info('episode:{}    e_greed:{}   Test reward:{}'.format(episode, agent.e_greed, eval_reward))
        if max_episode >= 5000:
            agent.save('model/MDDQN{}_v{}.ckpt'.format(suanli, v))
            with open('data/best_MDDQN/best_MDDQN{}_v{}.pkl'.format(suanli, v), 'wb') as f:
                pickle.dump(best_MDDQN, f)
        time_end2 = time.time()
        logger.info('NSGA2花费时间:{}分钟{}秒  MDDQN花费时间:{}分钟{}秒  总花费时间:{}分钟{}秒'.format(int((time_end1-time_start)/60), round((time_end1-time_start)%60,1), int((time_end2-time_end1)/60), round((time_end2-time_end1)%60,1), int((time_end2-time_start)/60), round((time_end2-time_start)%60,1)))
        plt.figure(1); plt.plot(make_max[:],label=algo_name);plt.plot(MDDQN_ALL[0][:],label='MDDQN');plt.legend(); plt.title("{}算法({}_v{}算例)完工时间趋势图".format(algo_name, suanli, v))
        plt.tight_layout();plt.savefig('data_out/240401/fig/算例{}_v{}加工完工时间趋势图.png'.format(suanli, v), dpi=500)
        plt.figure(2); plt.plot(Idle_sum[:],label=algo_name);plt.plot(MDDQN_ALL[1][:],label='MDDQN');plt.legend(); plt.title("{}算法({}_v{}算例)机器空闲时间趋势图".format(algo_name, suanli, v))
        plt.tight_layout();plt.savefig('data_out/240401/fig/算例{}_v{}机器空闲时间趋势图.png'.format(suanli, v), dpi=500)
        plt.figure(3); plt.plot(S_sum[:],label=algo_name);plt.plot(MDDQN_ALL[2][:],label='MDDQN');   plt.legend(); plt.title("{}算法({}_v{}算例)AGV运输距离趋势图".format(algo_name, suanli, v))
        plt.tight_layout();plt.savefig('data_out/240401/fig/算例{}_v{}AGV运输距离趋势图.png'.format(suanli, v), dpi=500)
        plt.figure(4); plt.plot(TR[:],label=algo_name);plt.plot(MDDQN_ALL[3][:],label='MDDQN');      plt.legend(); plt.title("{}算法({}_v{}算例)奖励趋势图".format(algo_name, suanli, v))
        plt.tight_layout();plt.savefig('data_out/240401/fig/算例{}_v{}奖励趋势图.png'.format(suanli, v), dpi=500)
        # plt.show()
        all.append(make_span3D); all.append(Idle_time3D); all.append(Agv_S3D); PF = pareto(all)
        make_span3D, Idle_time3D, Agv_S3D = [], [], []
        for i in PF:
            make_span3D.append(all[0][i]); Idle_time3D.append(all[1][i]); Agv_S3D.append(all[2][i])
        make_span3D, Idle_time3D, Agv_S3D = Zuixiaojuli_Top20(make_span3D, Idle_time3D, Agv_S3D)
        fig = plt.figure(); ax = fig.add_subplot(111, projection='3d')
        best = MO_Best(make_span3D, Idle_time3D, Agv_S3D)
        HV_R = []
        HV_R.append(max(max(make_span3D),max(NSGA2_fit[0]),max(MDDQN_all[0])))
        HV_R.append(max(max(Idle_time3D),max(NSGA2_fit[1]),max(MDDQN_all[1])))
        HV_R.append(max(max(Agv_S3D),max(NSGA2_fit[2]),max(MDDQN_all[2])))
        # 规则比较
        data_pj = pd.DataFrame({'Site': [], 'spacing': [], 'HV': []})
        data_out = 'data_out/240401/result{}_v{}.xlsx'.format(suanli, v)
        # data_out = 'data_out/tiankou5/{}_v{}/result{}_{}_{}_{}_{}_{}_{}_{}_{}.xlsx'.format(suanli, v, suanli, v, xunlian_id, ps, Pc, Pm, LEARNING_RATE, GAMMA, tanlan)
        ax.scatter(make_span3D[:], Idle_time3D[:], Agv_S3D[:], c='r', label=algo_name, marker='s', alpha=0.5)
        Guizebijiao(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id, data_out, act_dim, ax, HV_R, data_pj)
        ax.legend(loc='upper right', ncol=3); ax.set_title('{}算法({}_v{}算例)与单规则结果比较图'.format(algo_name, suanli, v))
        ax.set_xlabel('Makespan'); ax.set_ylabel('Mac_idle'); ax.set_zlabel('AGV_distance'); ax.view_init(elev=30, azim=-118);plt.tight_layout()
        main_window = QtWidgets.QMainWindow(); canvas = FigureCanvas(fig); main_window.setCentralWidget(canvas); main_window.showFullScreen(); main_window.close()
        fig.savefig('data_out/240401/fig/算例{}_v{}规则对比.png'.format(suanli, v), dpi=500)
        # NSGA2_fit[0], NSGA2_fit[1], NSGA2_fit[2] = Zuixiaojuli_Top20(NSGA2_fit[0], NSGA2_fit[1], NSGA2_fit[2])
        # MDDQN_all[0], MDDQN_all[1], MDDQN_all[2] = Zuixiaojuli_Top20(MDDQN_all[0], MDDQN_all[1], MDDQN_all[2])
        Data_loc(data_out, 'NSGA2', NSGA2_fit[0], NSGA2_fit[1], NSGA2_fit[2])
        Data_loc(data_out, 'MDDQN', MDDQN_all[0], MDDQN_all[1], MDDQN_all[2])
        Data_loc(data_out, algo_name, make_span3D, Idle_time3D, Agv_S3D)
        fig2 = plt.figure(); ax2 = fig2.add_subplot(111, projection='3d')
        # Shuchu(ax2, make_span3D, Idle_time3D, Agv_S3D, 't{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(suanli, v, xunlian_id, ps, Pc, Pm, LEARNING_RATE, GAMMA, tanlan), HV_R, data_pj)
        Shuchu(ax2, NSGA2_fit[0], NSGA2_fit[1], NSGA2_fit[2], 'NSGA2', HV_R, data_pj)
        Shuchu(ax2, MDDQN_all[0], MDDQN_all[1], MDDQN_all[2], 'MDDQN', HV_R, data_pj)
        Shuchu(ax2, make_span3D, Idle_time3D, Agv_S3D, algo_name, HV_R, data_pj)
        ax2.legend(loc='upper right'); ax2.set_title('{}算法({}_v{}算例)与NSGA2、MDDQN结果比较图'.format(algo_name, suanli, v));
        ax2.set_xlabel('Makespan'); ax2.set_ylabel('Mac_idle'); ax2.set_zlabel('AGV_distance'); ax2.view_init(elev=30, azim=-118);plt.tight_layout()
        main_window = QtWidgets.QMainWindow(); canvas = FigureCanvas(fig2); main_window.setCentralWidget(canvas); main_window.showFullScreen(); main_window.close()
        fig2.savefig('data_out/240401/fig/算例{}_v{}算法对比.png'.format(suanli, v), dpi=500)
        with pd.ExcelWriter(data_out, mode='a', engine='openpyxl') as writer:
            data_pj.to_excel(writer, sheet_name='评价指标', index=False)
        # plt.show()
        plt.close('all')
        best_MDDQN_load = best_MDDQN
    env.reset()
    for step in range(State * J_num):
        env.scheduling(best_MDDQN_load[1][step], best_MDDQN_load[2][step], best_MDDQN_load[3][step])  # 选择机器、工件和AGV后更新调度以及计算状态特征的数据
    env.Gantt(suanli, v)
    # 保存工件开始加工时间
    Get_J_s(env, J_num, JNN, suanli, v, '静态')
    '''————————————————发生故障下调度算法——故障详情————————————————'''

    if os.path.exists('data/Suanli2/Guzhang{}_v{}.pkl'.format(suanli, v)):
        with open('data/Suanli2/Guzhang{}_v{}.pkl'.format(suanli, v), 'rb') as f:
            Guzhang = pickle.load(f)
        faultyMachine, faultyTime_Mac, repairTime_Mac, faultyAGV, faultyTime_AGV, fauTimeAGV, repairTime_AGV = Guzhang[0], Guzhang[1], Guzhang[2], Guzhang[3], Guzhang[4], Guzhang[5], Guzhang[6]
        m_sta, m_mac = env.find_element_index(MNN, faultyMachine)
        print('警告！您车间位于{}阶段的机器{}在{}时间点发生故障，AGV{}在{}时间点发生故障'.format(m_sta, faultyMachine, faultyTime_Mac, faultyAGV, faultyTime_AGV))
    else:
        faultyMachine = random.randint(1, sum(M_j) - 1)  # 随机选择一个故障机器序号
        m_sta, m_mac = env.find_element_index(MNN, faultyMachine)
        faultyTime_Mac = round(random.uniform(env.AL_jk[m_sta][m_mac].start[0], env.AL_jk[m_sta][m_mac].last_ot),1)  # 随机选择一个故障开始点
        repairTime_Mac = 500  # 选择一个故障修复所用时间
        # faultyAGV = random.randint(1, v)  # 随机选择一个故障AGV序号
        # faultyTime_AGV = random.randint(env.AGVs[faultyAGV].using_time[0][0], env.AGVs[faultyAGV].end)  # 随机选择一个故障开始点
        faultyAGV = [random.randint(0, 1) for _ in range(v)]  # 随机一个AGV是否故障的列表，0为正常，1为故障
        faultyAGV[random.randint(0, v - 1)] = 1  # 保证有一个AGV是故障的
        faultyAGV = [0] + faultyAGV
        faultyTime_AGV = [0] + [round(random.uniform(env.AGVs[_].using_time[1][0], env.AGVs[_].end),1) for _ in range(1, v + 1)]  # 随机生成一个故障开始点列表
        repairTime_AGV = 500  # 选择一个故障修复所用时间
        # faultyTime_Mac, faultyAGV, faultyTime_AGV = 1200,[0,1,0,0],[0,1300,0,0]
        print('警告！您车间位于{}阶段的机器{}在{}时间点发生故障，AGV{}在{}时间点发生故障'.format(m_sta, faultyMachine, faultyTime_Mac, faultyAGV, faultyTime_AGV))
        fauTimeAGV = []
        for i in range(len(faultyAGV)):
            fauTimeAGV.append(faultyTime_AGV[i] * faultyAGV[i])
        Guzhang = [faultyMachine, faultyTime_Mac, repairTime_Mac, faultyAGV, faultyTime_AGV, fauTimeAGV, repairTime_AGV]
        with open('data/Suanli2/Guzhang{}_v{}.pkl'.format(suanli, v), 'wb') as f:
            pickle.dump(Guzhang, f)
    # 输出原调度方案分组示意图
    env.Gantt2(suanli, v, faultyMachine, faultyTime_Mac, repairTime_Mac, faultyAGV, fauTimeAGV, repairTime_AGV)
    fauTimeAGV = [i for i in fauTimeAGV if i != 0]
    G1_job, G1_mac, G1_agv = env.Get_G1(faultyTime_Mac, fauTimeAGV) # 记录原调度方案G1 G2作业序列
    env.reschedule(faultyMachine, faultyTime_Mac, repairTime_Mac, faultyAGV, faultyTime_AGV, repairTime_AGV)

    '''————————————————发生故障下调度算法——重调度————————————————'''
    max_episode = args.max_episode
    make_max, Idle_sum, S_sum, TR = [], [], [], []  # 3个目标
    all, make_span3D, Idle_time3D, Agv_S3D = [], [], [], []
    best_reMDDQN = [[], [], [], []]
    # build an agent
    rpm = ReplayMemory(MEMORY_SIZE, obs_dim, 0, MO)
    if MO:
        model = MultiHeadQNetwork(obs_dim=obs_dim, act_dim=act_dim)
        alg = MultiHeadDQN(model, gamma=GAMMA, lr=LEARNING_RATE)
        agent = MultiHeadAgent(alg, act_dim=act_dim, e_greed=tanlan, e_greed_decrement=1e-6)
    else:
        model = QNetwork(obs_dim=obs_dim, act_dim=act_dim)
        alg = DQN(model, gamma=GAMMA, lr=LEARNING_RATE) if algo_name == 'NSGA2_MDQN' else DDQN(model, gamma=GAMMA, lr=LEARNING_RATE)
        agent = Agent(alg, UPDATE_TARGET_STEP, act_dim=act_dim, e_greed=tanlan, e_greed_decrement=1e-6)
    if os.path.exists('model/reMDDQN{}_v{}.ckpt'.format(suanli, v)) and duqu_reMDDQN_model == True:
        agent.restore('model/reMDDQN{}_v{}.ckpt'.format(suanli, v))
        max_episode = 500
    if os.path.exists('data/reMDDQN/AGV{}/reMDDQN{}_v{}.pkl'.format(v, suanli, v)) and duqu_reMDDQN == True:
        with open('data/reMDDQN/AGV{}/reMDDQN{}_v{}.pkl'.format(v, suanli, v), 'rb') as f:
            reMDDQN_all = pickle.load(f)
        with open('data/reMDDQN/AGV{}/reMDDQN_ALL{}_v{}.pkl'.format(v, suanli, v), 'rb') as f:
            reMDDQN_ALL = pickle.load(f)
    else:
        reMDDQN_all, reMDDQN_ALL = reMDDQN_main('DDQN', max_episode, obs_dim, act_dim, xunlian, gongjianzhongshu, jiqishu, AGVshu, ps, Pc, Pm, LEARNING_RATE, GAMMA, tanlan, faultyMachine, faultyTime_Mac, repairTime_Mac, faultyAGV, faultyTime_AGV, repairTime_AGV, Multi_O, True)
    time_start = time.time()
    if os.path.exists('data/best_NSGA/best_reNSGA{}_v{}.pkl'.format(suanli, v)) and duqu_best_reNSGA2 == True:
        with open('data/best_NSGA/best_reNSGA{}_v{}.pkl'.format(suanli, v), 'rb') as f:
            best_reNSGA = pickle.load(f)
        AP, AM, AA, NSGA2_fit, pf, NSGA2_ALL = best_reNSGA[0], best_reNSGA[1], best_reNSGA[2], best_reNSGA[3], best_reNSGA[4], best_reNSGA[5],
    else:
        AP, AM, AA, NSGA2_fit, pf, NSGA2_ALL = reNSGA(n, J_num, State, M_j, PT, D_pq, JNN, v, V_r, JN_ip, ST_i, MNN, suanli, xunlian_id, ps, Pc, Pm, faultyMachine, faultyTime_Mac, repairTime_Mac, faultyAGV, faultyTime_AGV, repairTime_AGV, G1_job, G1_mac, G1_agv)
        best_reNSGA = [AP, AM, AA, NSGA2_fit, pf, NSGA2_ALL]
        with open('data/best_NSGA/best_reNSGA{}_v{}.pkl'.format(suanli, v), 'wb') as f:
            pickle.dump(best_reNSGA, f)
    time_end1 = time.time()
    s = Situation(n, J_num, State, M_j, PT, D_pq, JNN, v, V_r, JN_ip, ST_i, MNN, suanli, xunlian_id)
    s.reschedule(faultyMachine, faultyTime_Mac, repairTime_Mac, faultyAGV, faultyTime_AGV, repairTime_AGV)
    NSGA2_fit1, NSGA2_fit2, NSGA2_fit3 = s.reStage_Decode4(AP[pf], AM[pf], AA[pf])
    while len(rpm) < MEMORY_WARMUP_SIZE:
        if len(rpm) == 0:
            MDDQN_fit1, MDDQN_fit2, MDDQN_fit3 = NSGA2_fit1, NSGA2_fit2, NSGA2_fit3
        else:
            MDDQN_fit1, MDDQN_fit2, MDDQN_fit3, best_reMDDQN = reHuigun(n, J_num, State, M_j, PT, D_pq, JNN, v, V_r, JN_ip, ST_i, MNN, suanli, xunlian_id, faultyMachine, faultyTime_Mac, repairTime_Mac, faultyAGV, faultyTime_AGV, repairTime_AGV, best_reMDDQN)
        retrain_episode(n, J_num, State, M_j, PT, D_pq, JNN, v, V_r, JN_ip, ST_i, MNN, suanli, xunlian_id, agent, env, rpm, NSGA2_fit1, NSGA2_fit2, NSGA2_fit3, MDDQN_fit1, MDDQN_fit2, MDDQN_fit3, make_max, Idle_sum, S_sum, TR, best_reMDDQN, G1_job,G1_mac,G1_agv, MO)
    # start training
    episode = 0
    while episode < max_episode:
        ALL = [[], [], []]
        for i in trange(50):
            MDDQN_fit1, MDDQN_fit2, MDDQN_fit3, best_reMDDQN = reHuigun(n, J_num, State, M_j, PT, D_pq, JNN, v, V_r, JN_ip, ST_i, MNN, suanli, xunlian_id, faultyMachine, faultyTime_Mac, repairTime_Mac, faultyAGV, faultyTime_AGV, repairTime_AGV, best_reMDDQN)
            total_reward, a, b, c, make_max, Idle_sum, S_sum, TR, best_reMDDQN = retrain_episode(n, J_num, State, M_j, PT, D_pq, JNN, v, V_r, JN_ip, ST_i, MNN, suanli, xunlian_id, agent, env, rpm, NSGA2_fit1, NSGA2_fit2, NSGA2_fit3, MDDQN_fit1, MDDQN_fit2, MDDQN_fit3, make_max, Idle_sum, S_sum, TR, best_reMDDQN, G1_job,G1_mac,G1_agv, MO)
            ALL[0].append(a); ALL[1].append(b); ALL[2].append(c)
            episode += 1
        PF = pareto(ALL)
        make_span3D.extend([ALL[0][index] for index in PF])
        Idle_time3D.extend([ALL[1][index] for index in PF])
        Agv_S3D.extend([ALL[2][index] for index in PF])
        # test part
        eval_reward = reevaluate_episodes(n, J_num, State, M_j, PT, D_pq, JNN, v, V_r, JN_ip, ST_i, MNN, suanli, xunlian_id, agent, NSGA2_fit1, NSGA2_fit2, NSGA2_fit3, MDDQN_fit1, MDDQN_fit2, MDDQN_fit3, G1_job,G1_mac,G1_agv,faultyMachine, faultyTime_Mac, repairTime_Mac, faultyAGV, faultyTime_AGV, repairTime_AGV, MO)
        logger.info('episode:{}    e_greed:{}   Test reward:{}'.format(episode, agent.e_greed, eval_reward))
    if max_episode >= 5000:
        agent.save('model/reMDDQN{}_v{}.ckpt'.format(suanli, v))
    time_end2 = time.time()
    logger.info('NSGA2花费时间:{}分钟{}秒  MDDQN花费时间:{}分钟{}秒  总花费时间:{}分钟{}秒'.format(int((time_end1 - time_start) / 60), round((time_end1 - time_start) % 60, 1), int((time_end2 - time_end1) / 60), round((time_end2 - time_end1) % 60, 1), int((time_end2 - time_start) / 60), round((time_end2 - time_start) % 60, 1)))
    plt.figure(1); plt.plot(make_max[:], label=algo_name); plt.plot(reMDDQN_ALL[0][:], label='MDDQN'); plt.legend(); plt.title("重调度{}算法({}_v{}算例)完工时间趋势图".format(algo_name, suanli, v)); plt.tight_layout()
    plt.savefig('data_out/240401/fig/算例{}_v{}重调度加工完工时间趋势图.png'.format(suanli, v), dpi=500)
    plt.figure(2); plt.plot(Idle_sum[:], label=algo_name); plt.plot(reMDDQN_ALL[1][:], label='MDDQN'); plt.legend(); plt.title("重调度{}算法({}_v{}算例)机器空闲时间趋势图".format(algo_name, suanli, v)); plt.tight_layout()
    plt.savefig('data_out/240401/fig/算例{}_v{}重调度机器空闲时间趋势图.png'.format(suanli, v), dpi=500)
    plt.figure(3); plt.plot(S_sum[:], label=algo_name); plt.plot(reMDDQN_ALL[2][:], label='MDDQN'); plt.legend(); plt.title("重调度{}算法({}_v{}算例)AGV运输距离趋势图".format(algo_name, suanli, v)); plt.tight_layout()
    plt.savefig('data_out/240401/fig/算例{}_v{}重调度AGV运输距离趋势图.png'.format(suanli, v), dpi=500)
    plt.figure(4); plt.plot(TR[:], label=algo_name); plt.plot(reMDDQN_ALL[3][:], label='MDDQN'); plt.legend(); plt.title("重调度{}算法({}_v{}算例)奖励趋势图".format(algo_name, suanli, v)); plt.tight_layout()
    plt.savefig('data_out/240401/fig/算例{}_v{}重调度奖励趋势图.png'.format(suanli, v), dpi=500)
    # plt.show()
    all.append(make_span3D); all.append(Idle_time3D); all.append(Agv_S3D); PF = pareto(all)
    make_span3D, Idle_time3D, Agv_S3D = [], [], []
    for i in PF:
        make_span3D.append(all[0][i]); Idle_time3D.append(all[1][i]); Agv_S3D.append(all[2][i])
    make_span3D, Idle_time3D, Agv_S3D = Zuixiaojuli_Top20(make_span3D, Idle_time3D, Agv_S3D)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    best = MO_Best(make_span3D, Idle_time3D, Agv_S3D)
    HV_R = []
    HV_R.append(max(max(make_span3D), max(NSGA2_fit[0]), max(reMDDQN_all[0])))
    HV_R.append(max(max(Idle_time3D), max(NSGA2_fit[1]), max(reMDDQN_all[1])))
    HV_R.append(max(max(Agv_S3D), max(NSGA2_fit[2]), max(reMDDQN_all[2])))
    # 规则比较
    data_pj = pd.DataFrame({'Site': [], 'spacing': [], 'HV': []})
    data_out = 'data_out/240401/REresult{}_v{}.xlsx'.format(suanli, v)
    # data_out = 'data_out/tiankou5/{}_v{}/result{}_{}_{}_{}_{}_{}_{}_{}_{}.xlsx'.format(suanli, v, suanli, v, xunlian_id, ps, Pc, Pm, LEARNING_RATE, GAMMA, tanlan)
    ax.scatter(make_span3D[:], Idle_time3D[:], Agv_S3D[:], c='r', label=algo_name, marker='s', alpha=0.5)
    # ax.scatter(reMDDQN_all[0], reMDDQN_all[1], reMDDQN_all[2], c='cadetblue', label=algo_name, marker='s', alpha=0.5)
    reGuizebijiao(n, J_num, State, M_j, PT, D_pq, JNN, v, V_r, JN_ip, ST_i, MNN, suanli, xunlian_id, data_out, act_dim, ax, HV_R, data_pj, faultyMachine, faultyTime_Mac, repairTime_Mac, faultyAGV, faultyTime_AGV, repairTime_AGV, G1_job,G1_mac,G1_agv)
    ax.legend(loc='upper right', ncol=3); ax.set_title('重调度{}算法({}_v{}算例)与单规则结果比较图'.format(algo_name, suanli, v))
    ax.set_xlabel('Makespan'); ax.set_ylabel('Mac_idle'); ax.set_zlabel('AGV_distance')
    ax.view_init(elev=30, azim=-118); plt.tight_layout()
    main_window = QtWidgets.QMainWindow(); canvas = FigureCanvas(fig); main_window.setCentralWidget(canvas); main_window.showFullScreen(); main_window.close()
    fig.savefig('data_out/240401/fig/算例{}_v{}重调度规则对比.png'.format(suanli, v), dpi=500)
    # NSGA2_fit[0], NSGA2_fit[1], NSGA2_fit[2] = Zuixiaojuli_Top20(NSGA2_fit[0], NSGA2_fit[1], NSGA2_fit[2])
    # reMDDQN_all[0], reMDDQN_all[1], reMDDQN_all[2] = Zuixiaojuli_Top20(reMDDQN_all[0], reMDDQN_all[1], reMDDQN_all[2])
    Data_loc(data_out, 'NSGA2', NSGA2_fit[0], NSGA2_fit[1], NSGA2_fit[2])
    Data_loc(data_out, 'MDDQN', reMDDQN_all[0], reMDDQN_all[1], reMDDQN_all[2])
    Data_loc(data_out, algo_name, make_span3D, Idle_time3D, Agv_S3D)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    # Shuchu(ax2, make_span3D, Idle_time3D, Agv_S3D, 't{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(suanli, v, xunlian_id, ps, Pc, Pm, LEARNING_RATE, GAMMA, tanlan), HV_R, data_pj)
    Shuchu(ax2, NSGA2_fit[0], NSGA2_fit[1], NSGA2_fit[2], 'NSGA2', HV_R, data_pj)
    Shuchu(ax2, reMDDQN_all[0], reMDDQN_all[1], reMDDQN_all[2], 'MDDQN', HV_R, data_pj)
    Shuchu(ax2, make_span3D, Idle_time3D, Agv_S3D, algo_name, HV_R, data_pj)
    ax2.legend(loc='upper right'); ax2.set_title('重调度{}算法({}_v{}算例)与NSGA2、MDDQN结果比较图'.format(algo_name, suanli, v))
    ax2.set_xlabel('Makespan'); ax2.set_ylabel('Mac_idle'); ax2.set_zlabel('AGV_distance')
    ax2.view_init(elev=30, azim=-118); plt.tight_layout()
    main_window = QtWidgets.QMainWindow(); canvas = FigureCanvas(fig2); main_window.setCentralWidget(canvas); main_window.showFullScreen(); main_window.close()
    fig2.savefig('data_out/240401/fig/算例{}_v{}重调度算法对比.png'.format(suanli, v), dpi=500)
    with pd.ExcelWriter(data_out, mode='a', engine='openpyxl') as writer:
        data_pj.to_excel(writer, sheet_name='评价指标', index=False)
    # plt.show()
    plt.close('all')
    env.reset()
    for step in range(State * J_num):
        env.rescheduling(best_reMDDQN[1][step], best_reMDDQN[2][step], best_reMDDQN[3][step])  # 选择机器、工件和AGV后更新调度以及计算状态特征的数据
    env.reGantt(algo_name, suanli, v, m_sta, faultyMachine, faultyTime_Mac, faultyAGV, faultyTime_AGV, repairTime_Mac, repairTime_AGV)
    s.reGantt('NSGA2', suanli, v, m_sta, faultyMachine, faultyTime_Mac, faultyAGV, faultyTime_AGV, repairTime_Mac, repairTime_AGV)
    # 保存工件开始加工时间
    # Get_J_s(env, J_num, JNN, suanli, v, '动态')
def Huigun(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id, best_MDDQN):
    fit1, fit2, fit3 = [], [], []
    s = Situation(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id)
    # pf = MO_Best(make_max, Idle_sum, S_sum)
    for step in range(State * J_num):
        # s.scheduling(JOB_line[pf][step], MAC_line[pf][step], AGV_line[pf][step])  # 选择机器、工件和AGV后更新调度以及计算状态特征的数据
        s.scheduling(best_MDDQN[1][step], best_MDDQN[2][step], best_MDDQN[3][step])  # 选择机器、工件和AGV后更新调度以及计算状态特征的数据
        obs_new = s.Features()  # 更新状态特征
        fit1.append(float(obs_new[5])); fit2.append(float(obs_new[6])); fit3.append(float(obs_new[7]))
    return fit1, fit2, fit3, best_MDDQN
def reHuigun(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id,faultyMachine, faultyTime_Mac, repairTime_Mac, faultyAGV, faultyTime_AGV, repairTime_AGV, best_reMDDQN):
    fit1, fit2, fit3 = [], [], []
    s = Situation(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id)
    s.reschedule(faultyMachine, faultyTime_Mac, repairTime_Mac, faultyAGV, faultyTime_AGV, repairTime_AGV)
    # pf = MO_Best(make_max, Idle_sum, S_sum)
    for step in range(State * J_num):
        # s.rescheduling(JOB_line[pf][step], MAC_line[pf][step], AGV_line[pf][step])  # 选择机器、工件和AGV后更新调度以及计算状态特征的数据
        s.rescheduling(best_reMDDQN[1][step], best_reMDDQN[2][step], best_reMDDQN[3][step])  # 选择机器、工件和AGV后更新调度以及计算状态特征的数据
        obs_new = s.Features()  # 更新状态特征
        fit1.append(float(obs_new[5])); fit2.append(float(obs_new[6])); fit3.append(float(obs_new[7]))
    return fit1, fit2, fit3, best_reMDDQN
def Shuchu(ax2, O1, O2, O3, suanfa, HV_R, data_pj):
    if suanfa == 'NSGA2':
        ax2.scatter(O1[:], O2[:], O3[:], c='b', label=suanfa, marker='o', alpha=0.5)
    elif suanfa == 'MDDQN':
        ax2.scatter(O1[:], O2[:], O3[:], c='g', label=suanfa, marker='p', alpha=0.5)
    else:
        ax2.scatter(O1[:], O2[:], O3[:], c='r', label=suanfa, marker='s', alpha=0.5)
    logger.info('{}： Spacing:{}  HV:{}'.format(suanfa, round(Spacing_P(O1, O2, O3), 2),round(HV(O1, O2, O3, HV_R), 2)))
    data_pj.loc['{}'.format(suanfa), :] = [suanfa, round(Spacing_P(O1, O2, O3), 2),round(HV(O1, O2, O3, HV_R), 2)]

def Data_loc(data_out, loc_name, make_span3D, Idle_time3D, Agv_S3D):
    data = pd.DataFrame([['完工时间'],['空闲时间'],['运输距离']], columns=['Site'])
    data = pd.DataFrame({'Site': [], '完工时间': [], '空闲时间': [], '运输距离': []})
    for i in range(len(make_span3D)):
        data.loc['解{}'.format(i+1), :] = ['解{}'.format(i+1), make_span3D[i],Idle_time3D[i],Agv_S3D[i]]
    data.loc['平均值', :] = ['平均值', round(statistics.mean(make_span3D), 2),round(statistics.mean(Idle_time3D), 2),round(statistics.mean(Agv_S3D), 2)]
    data.loc['标准差', :] = ['标准差', Biaozhuncha(make_span3D), Biaozhuncha(Idle_time3D), Biaozhuncha(Agv_S3D)]
    # data.to_excel(data_out, sheet_name=loc_name, index=False)
    if loc_name == 'rule1':
        data.to_excel(data_out, sheet_name=loc_name, index=False)
    else:
        excel_file = pd.ExcelFile(data_out)
        # 获取所有表单名
        sheet_names = excel_file.sheet_names
        if loc_name in sheet_names:
            # 删除指定表单
            df = excel_file.parse(loc_name)
            df.drop(df.index, inplace=True)
            # 保存修政后的excel文件
            writer = pd.ExcelWriter(data_out)
            for sheet_name in sheet_names:
                if sheet_name != loc_name:
                    excel_file.parse(sheet_name).to_excel(writer, sheet_name=sheet_name, index=False)
            writer.close()
        with pd.ExcelWriter(data_out, mode='a', engine='openpyxl') as writer:
            data.to_excel(writer, sheet_name=loc_name, index=False)

def calc_reward_to_go(reward_list, gamma=1.0):
    for i in range(len(reward_list) - 2, -1, -1):
        # G_i = r_i + γ·G_i+1
        reward_list[i] += gamma * reward_list[i + 1]  # Gt
    return np.array(reward_list)
def MO_Best(make_span3D, Idle_time3D, Agv_S3D):
    pf, minmin = 0, float('inf')
    for i in range(len(make_span3D)):
        if (make_span3D[i]-min(make_span3D))**2 + (Idle_time3D[i]-min(Idle_time3D))**2 + (Agv_S3D[i]-min(Agv_S3D))**2 < minmin:
            minmin = (make_span3D[i]-min(make_span3D))**2 + (Idle_time3D[i]-min(Idle_time3D))**2 + (Agv_S3D[i]-min(Agv_S3D))**2
            pf = i
    return pf
def Zuixiaojuli_Top20(make_span3D, Idle_time3D, Agv_S3D):
    lst = []
    for i in range(len(make_span3D)):
        lst.append((make_span3D[i]-min(make_span3D))**2 + (Idle_time3D[i]-min(Idle_time3D))**2 + (Agv_S3D[i]-min(Agv_S3D))**2)
    # sorted_lst = sorted(lst)
    # top_20 = sorted_lst[:20]
    # top_20_indices = [lst.index(x) for x in top_20]

    top_20_indices = []
    indexed_lst = list(enumerate(lst))
    sorted_lst = sorted(indexed_lst, key=lambda x: x[1])
    top_20_elements = sorted_lst[:20]
    for index, value in top_20_elements:
        top_20_indices.append(index)

    top_20_make_span3D = [make_span3D[x] for x in top_20_indices]
    top_20_Idle_time3D = [Idle_time3D[x] for x in top_20_indices]
    top_20_Agv_S3D = [Agv_S3D[x] for x in top_20_indices]
    return top_20_make_span3D, top_20_Idle_time3D, top_20_Agv_S3D
def Guizebijiao(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id, data_out, act_dim, ax, HV_R, data_pj):
    rgb_colors = ['#000000','#FF0000','#00FF00','#0000FF','#FFFF00','#FF00FF','#00FFFF','#FFA500','#800080','#008000','#800000','#008080','#000080','#FFC0CB','#ADD8E6','#F0E68C','#FA8072','#32CD32','#2E8B57']
                             # 红色    # 绿色     # 蓝色     # 黄色    # 品红色    # 青色     # 橙色     # 紫色    # 深绿色   # 深红色   # 深青色   # 深蓝色    # 粉色    # 亮蓝色    # 卡其色   # 洋红色   # 酸橙绿   # 海绿色
    ALL = []
    for i in range(1,18+1):
        print('{}_v{}rule{}'.format(suanli, v, i))
        make_span_rule3D, Idle_time_rule3D, Agv_S_rule3D = [], [], []
        ALL_rule = [[], [], []]
        for j in trange(rule_episode):
            env = Situation(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id)
            for s in range(State * J_num):  # 对工序遍历，每次选出一个动作和机器
                Job_id, Machine_id, Agv_id = Env_action(i, env)
                done = True if s == State * J_num - 1 else False
                env.scheduling(Job_id, Machine_id, Agv_id)  # 选择机器、工件和AGV后更新调度以及计算状态特征的数据
                if done:
                    obs = env.Features()  # 更新状态特征
                    ALL_rule[0].append(float(obs[5]))
                    ALL_rule[1].append(float(obs[6]))
                    ALL_rule[2].append(float(obs[7]))
                    break
        PF = pareto(ALL_rule)
        for iii in PF:
            if ALL_rule[0][iii] not in make_span_rule3D or ALL_rule[1][iii] not in Idle_time_rule3D or ALL_rule[2][iii] not in Agv_S_rule3D:
                make_span_rule3D.append(ALL_rule[0][iii]); Idle_time_rule3D.append(ALL_rule[1][iii]); Agv_S_rule3D.append(ALL_rule[2][iii])
        ALL.append([make_span_rule3D, Idle_time_rule3D, Agv_S_rule3D])
        # make_span_rule3D, Idle_time_rule3D, Agv_S_rule3D = Zuixiaojuli_Top20(make_span_rule3D, Idle_time_rule3D, Agv_S_rule3D)
        HV_R[0]=max(HV_R[0],max(make_span_rule3D))
        HV_R[1]=max(HV_R[1],max(Idle_time_rule3D))
        HV_R[2]=max(HV_R[2],max(Agv_S_rule3D))
        rulelabel = 'rule{}'.format(i)
        ax.scatter(make_span_rule3D[:], Idle_time_rule3D[:], Agv_S_rule3D[:], c=rgb_colors[i], label=rulelabel, marker='o', alpha=0.5)
        Data_loc(data_out, 'rule{}'.format(i), make_span_rule3D, Idle_time_rule3D, Agv_S_rule3D)
    for i in range(18):
        logger.info('{}  Spacing:{}  HV:{}'.format('rule{}'.format(i+1), round(Spacing_P(ALL[i][0], ALL[i][1], ALL[i][2]),2), round(HV(ALL[i][0], ALL[i][1], ALL[i][2], HV_R),2)))
        data_pj.loc['rule{}'.format(i+1), :] = ['rule{}'.format(i+1), round(Spacing_P(ALL[i][0], ALL[i][1], ALL[i][2]),2), round(HV(ALL[i][0], ALL[i][1], ALL[i][2], HV_R),2)]
def reGuizebijiao(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id, data_out, act_dim, ax, HV_R, data_pj, faultyMachine, faultyTime_Mac, repairTime_Mac, faultyAGV, faultyTime_AGV, repairTime_AGV, G1_job,G1_mac,G1_agv):
    rgb_colors = ['#000000','#FF0000','#00FF00','#0000FF','#FFFF00','#FF00FF','#00FFFF','#FFA500','#800080','#008000','#800000','#008080','#000080','#FFC0CB','#ADD8E6','#F0E68C','#FA8072','#32CD32','#2E8B57']
                             # 红色    # 绿色     # 蓝色     # 黄色    # 品红色    # 青色     # 橙色     # 紫色    # 深绿色   # 深红色   # 深青色   # 深蓝色    # 粉色    # 亮蓝色    # 卡其色   # 洋红色   # 酸橙绿   # 海绿色
    ALL = []
    for i in range(1,18+1):
        print('{}_v{}rule{}'.format(suanli, v, i))
        make_span_rule3D, Idle_time_rule3D, Agv_S_rule3D = [], [], []
        ALL_rule = [[], [], []]
        for j in trange(rule_episode):
            env = Situation(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id)
            env.reschedule(faultyMachine, faultyTime_Mac, repairTime_Mac, faultyAGV, faultyTime_AGV, repairTime_AGV)
            for s in range(State * J_num):  # 对工序遍历，每次选出一个动作和机器
                if s < len(G1_job):
                    Job_id, Machine_id, Agv_id = G1_job[s], G1_mac[s], G1_agv[s]
                else:
                    Job_id, Machine_id, Agv_id = Env_action(i, env)
                done = True if s == State * J_num - 1 else False
                env.rescheduling(Job_id, Machine_id, Agv_id)  # 选择机器、工件和AGV后更新调度以及计算状态特征的数据
                if done:
                    obs = env.Features()  # 更新状态特征
                    ALL_rule[0].append(float(obs[5]))
                    ALL_rule[1].append(float(obs[6]))
                    ALL_rule[2].append(float(obs[7]))
                    break
            # if i == 10:
            #     env.reGantt('规则', suanli, v, 2, faultyMachine, faultyTime_Mac, faultyAGV, faultyTime_AGV, repairTime_Mac, repairTime_AGV)
        PF = pareto(ALL_rule)
        for iii in PF:
            if ALL_rule[0][iii] not in make_span_rule3D or ALL_rule[1][iii] not in Idle_time_rule3D or ALL_rule[2][iii] not in Agv_S_rule3D:
                make_span_rule3D.append(ALL_rule[0][iii]); Idle_time_rule3D.append(ALL_rule[1][iii]); Agv_S_rule3D.append(ALL_rule[2][iii])
        ALL.append([make_span_rule3D, Idle_time_rule3D, Agv_S_rule3D])
        # make_span_rule3D, Idle_time_rule3D, Agv_S_rule3D = Zuixiaojuli_Top20(make_span_rule3D, Idle_time_rule3D, Agv_S_rule3D)
        HV_R[0]=max(HV_R[0],max(make_span_rule3D))
        HV_R[1]=max(HV_R[1],max(Idle_time_rule3D))
        HV_R[2]=max(HV_R[2],max(Agv_S_rule3D))
        rulelabel = 'rule{}'.format(i)
        ax.scatter(make_span_rule3D[:], Idle_time_rule3D[:], Agv_S_rule3D[:], c=rgb_colors[i], label=rulelabel, marker='o', alpha=0.5)
        Data_loc(data_out, 'rule{}'.format(i), make_span_rule3D, Idle_time_rule3D, Agv_S_rule3D)
    for i in range(18):
        logger.info('{}  Spacing:{}  HV:{}'.format('rule{}'.format(i+1), round(Spacing_P(ALL[i][0], ALL[i][1], ALL[i][2]),2), round(HV(ALL[i][0], ALL[i][1], ALL[i][2], HV_R),2)))
        data_pj.loc['rule{}'.format(i+1), :] = ['rule{}'.format(i+1), round(Spacing_P(ALL[i][0], ALL[i][1], ALL[i][2]),2), round(HV(ALL[i][0], ALL[i][1], ALL[i][2], HV_R),2)]
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
        hypervolume += (HV_R[0]+100-make_span_rule3D[i])*(HV_R[1]+100-Idle_time_rule3D[i])*(HV_R[2]+100-Agv_S_rule3D[i])
    return hypervolume/len(make_span_rule3D)

def pareto(fitness):
    PF = []  # 存放帕累托前沿解的索引
    L = len(fitness[0])  # 解的总数（数组行数）
    pn = np.zeros(L, dtype=int)  # 存放每个解被支配的次数
    for i in range(L):
        for j in range(L):
            if i != j and dominates(fitness, fitness, len(fitness), i, j):
                pn[j] += 1
        if pn[i] == 0:
            PF.append(i)
    return PF
def dominates(x, y, num_objectives, i, j):
    dominates_x = False
    dominates_y = False
    for k in range(num_objectives):
        if x[k][i] < y[k][j]:
            dominates_x = True
        elif x[k][i] > y[k][j]:
            dominates_y = True
    return dominates_x and not dominates_y
def Zhuanzhi(AFit):
    fit = [[] for _ in range(len(AFit[0]))]
    for a in range(len(AFit)):
        for i in range(len(fit)):
            fit[i].append(AFit[a][i])
    return fit
def Biaozhuncha(A):
    K = 0
    for i in A:
        K += np.square(i - sum(A)/len(A))
    a = np.sqrt(K / len(A))
    return a
def Rule_re(xunlian, gongjianzhongshu, jiqishu, AGVshu, ps, Pc, Pm, LEARNING_RATE, GAMMA, tanlan, Multi_O=True):
    n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id = Suanli_Read(xunlian, gongjianzhongshu, jiqishu, AGVshu)
    act_dim = 18
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    data_out2 = 'data_out/result{}_v{}_{}.xlsx'.format(suanli, v, xunlian_id)
    excel_file = pd.ExcelFile(data_out2)
    # 获取所有表单名
    sheet_names = excel_file.sheet_names
    for sheet_name in sheet_names:
        if sheet_name == 'NSGA2_MDDQN':
            make_span3D, Idle_time3D, Agv_S3D = Rule_re_du(suanli, v, xunlian_id, data_out2, sheet_name)
        elif sheet_name == 'NSGA2':
            NSGA2_fit1, NSGA2_fit2, NSGA2_fit3 = Rule_re_du(suanli, v, xunlian_id, data_out2, sheet_name)
        elif sheet_name == 'MDDQN':
            MDDQN_all1, MDDQN_all2, MDDQN_all3 = Rule_re_du(suanli, v, xunlian_id, data_out2, sheet_name)
    HV_R = []
    HV_R.append(max(max(make_span3D),max(NSGA2_fit1),max(MDDQN_all1)))
    HV_R.append(max(max(Idle_time3D),max(NSGA2_fit2),max(MDDQN_all2)))
    HV_R.append(max(max(Agv_S3D),max(NSGA2_fit3),max(MDDQN_all3)))
    data_pj = pd.DataFrame({'Site': [], 'spacing': [], 'HV': []})
    data_out = 'data_out/re/result{}_v{}_{}.xlsx'.format(suanli, v, xunlian_id)
    Guizebijiao(n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id, data_out, act_dim, ax, HV_R, data_pj)
    Data_loc(data_out, 'NSGA2', NSGA2_fit1, NSGA2_fit2, NSGA2_fit3)
    Data_loc(data_out, 'MDDQN', MDDQN_all1, MDDQN_all2, MDDQN_all3)
    Data_loc(data_out, 'NSGA2_MDDQN', make_span3D, Idle_time3D, Agv_S3D)
    Shuchu(ax2, NSGA2_fit1, NSGA2_fit2, NSGA2_fit3, 'NSGA2', HV_R, data_pj)
    Shuchu(ax2, MDDQN_all1, MDDQN_all2, MDDQN_all3, 'MDDQN', HV_R, data_pj)
    Shuchu(ax2, make_span3D, Idle_time3D, Agv_S3D, 'NSGA2_MDDQN', HV_R, data_pj)
    with pd.ExcelWriter(data_out, mode='a', engine='openpyxl') as writer:
        data_pj.to_excel(writer, sheet_name='评价指标', index=False)

def Rule_re_du(suanli, v, xunlian_id, data_out, sheet_name):
    df = pd.read_excel(data_out, sheet_name=sheet_name)
    make_span3D, Idle_time3D, Agv_S3D = [], [], []
    # 将每一列数据存储到列表中
    for column_name in df.columns:
        if column_name == '完工时间':
            make_span3D = list(df[column_name])
            make_span3D = make_span3D[:-2]
        elif column_name == '空闲时间':
            Idle_time3D = list(df[column_name])
            Idle_time3D = Idle_time3D[:-2]
        elif column_name == '运输距离':
            Agv_S3D = list(df[column_name])
            Agv_S3D = Agv_S3D[:-2]
    return make_span3D, Idle_time3D, Agv_S3D
def Rule_re_du2(data_out, sheet_name):
    df = pd.read_excel(data_out, sheet_name=sheet_name)
    make_span3D, Idle_time3D, Agv_S3D = [], [], []
    # 将每一列数据存储到列表中
    for column_name in df.columns:
        if column_name == '完工时间':
            make_span3D = list(df[column_name])
            make_span3D = round(sum(make_span3D[:-2])/len(make_span3D[:-2]),2)
        elif column_name == '空闲时间':
            Idle_time3D = list(df[column_name])
            Idle_time3D = round(sum(Idle_time3D[:-2])/len(Idle_time3D[:-2]),2)
        elif column_name == '运输距离':
            Agv_S3D = list(df[column_name])
            Agv_S3D = round(sum(Agv_S3D[:-2])/len(Agv_S3D[:-2]),2)
    return make_span3D, Idle_time3D, Agv_S3D
def Shuchu_re(xunlian, gongjianzhongshu, jiqishu, AGVshu):
    n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id = Suanli_Read(xunlian, gongjianzhongshu, jiqishu, AGVshu)
    act_dim = 18
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    data_out = 'data_out/240401gai/REresult/REresult{}_v{}.xlsx'.format(suanli, v)
    excel_file = pd.ExcelFile(data_out)
    # 获取所有表单名
    sheet_names = excel_file.sheet_names
    for sheet_name in sheet_names:
        if sheet_name == 'NSGA2_MDDQN':
            make_span3D, Idle_time3D, Agv_S3D = Rule_re_du(suanli, v, xunlian_id, data_out, sheet_name)
        elif sheet_name == 'NSGA2':
            NSGA2_fit1, NSGA2_fit2, NSGA2_fit3 = Rule_re_du(suanli, v, xunlian_id, data_out, sheet_name)
        elif sheet_name == 'MDDQN':
            MDDQN_all1, MDDQN_all2, MDDQN_all3 = Rule_re_du(suanli, v, xunlian_id, data_out, sheet_name)
    ax.scatter(make_span3D[:], Idle_time3D[:], Agv_S3D[:], c='r', label='NSGA2_MDDQN', marker='s', alpha=0.5)
    HV_R = []
    HV_R.append(max(max(make_span3D),max(NSGA2_fit1),max(MDDQN_all1)))
    HV_R.append(max(max(Idle_time3D),max(NSGA2_fit2),max(MDDQN_all2)))
    HV_R.append(max(max(Agv_S3D),max(NSGA2_fit3),max(MDDQN_all3)))
    data_pj = pd.DataFrame({'Site': [], 'spacing': [], 'HV': []})
    rgb_colors = ['#808080','#00FF00','#0000FF','#FFFF00','#FF00FF','#00FFFF','#FFA500','#800080','#008000','#800000','#008080','#000080','#FFC0CB','#ADD8E6','#F0E68C','#FA8072','#32CD32','#2E8B57']
    ii, ALL = 0, []
    for sheet_name in sheet_names:
        if sheet_name[:4] == 'rule':
            make_span_rule3D, Idle_time_rule3D, Agv_S_rule3D = Rule_re_du(suanli, v, xunlian_id, data_out, sheet_name)
            ALL.append([make_span_rule3D, Idle_time_rule3D, Agv_S_rule3D])
            HV_R[0] = max(HV_R[0], max(make_span_rule3D))
            HV_R[1] = max(HV_R[1], max(Idle_time_rule3D))
            HV_R[2] = max(HV_R[2], max(Agv_S_rule3D))
            ax.scatter(make_span_rule3D[:], Idle_time_rule3D[:], Agv_S_rule3D[:], c=rgb_colors[ii], label=sheet_name, marker='o', alpha=0.5)
            ii = ii + 1
    for i in range(len(ALL)):
        logger.info('{}  Spacing:{}  HV:{}'.format('rule{}'.format(i+1), round(Spacing_P(ALL[i][0], ALL[i][1], ALL[i][2]),2), round(HV(ALL[i][0], ALL[i][1], ALL[i][2], HV_R),2)))
        data_pj.loc['rule{}'.format(i+1), :] = ['rule{}'.format(i+1), round(Spacing_P(ALL[i][0], ALL[i][1], ALL[i][2]),2), round(HV(ALL[i][0], ALL[i][1], ALL[i][2], HV_R),2)]
    Shuchu(ax2, NSGA2_fit1, NSGA2_fit2, NSGA2_fit3, 'NSGA2', HV_R, data_pj)
    Shuchu(ax2, MDDQN_all1, MDDQN_all2, MDDQN_all3, 'MDDQN', HV_R, data_pj)
    Shuchu(ax2, make_span3D, Idle_time3D, Agv_S3D, 'NSGA2_MDDQN', HV_R, data_pj)
    with pd.ExcelWriter(data_out, mode='a', engine='openpyxl') as writer:
        data_pj.to_excel(writer, sheet_name='评价指标2', index=False)
    ax.legend(loc='upper right', ncol=3); ax.set_title('重调度NSGA2_MDDQN算法({}_v{}算例)与单规则结果比较图'.format(suanli, v))
    ax.set_xlabel('Makespan'); ax.set_ylabel('Mac_idle'); ax.set_zlabel('AGV_distance')
    ax.view_init(elev=25, azim=-106)
    plt.tight_layout()
    fig.savefig('data_out/240401gai/fig/算例{}_v{}重调度规则对比.png'.format(suanli, v), dpi=500)
    ax2.legend(loc='upper right'); ax2.set_title('重调度NSGA2_MDDQN算法({}_v{}算例)与NSGA2、MDDQN结果比较图'.format(suanli, v));
    ax2.set_xlabel('Makespan'); ax2.set_ylabel('Mac_idle'); ax2.set_zlabel('AGV_distance')
    ax2.view_init(elev=25, azim=-106)
    plt.tight_layout()
    fig2.savefig('data_out/240401gai/fig/算例{}_v{}重调度算法对比.png'.format(suanli, v), dpi=500)
    # plt.show()
    plt.close('all')
    print('n{}m{}v{}指标计算完成'.format(gongjianzhongshu, jiqishu, AGVshu))

def Shuchu_re2(xunlian, gongjianzhongshu, jiqishu, AGVshu, data_pj):
    n,J_num,State,M_j,PT,D_pq,JNN,v,V_r,JN_ip,ST_i,MNN,suanli,xunlian_id = Suanli_Read(xunlian, gongjianzhongshu, jiqishu, AGVshu)
    data_out = 'data_out/240401gai/REresult/REresult{}_v{}.xlsx'.format(suanli, v)
    excel_file = pd.ExcelFile(data_out)
    # 获取所有表单名
    sheet_names = excel_file.sheet_names
    make_span, Idle_time, Agv_S = ['n{}m{}v{}加工时间平均值'.format(gongjianzhongshu, jiqishu, AGVshu)],['n{}m{}v{}空闲时间平均值'.format(gongjianzhongshu, jiqishu, AGVshu)],['n{}m{}v{}运输距离平均值'.format(gongjianzhongshu, jiqishu, AGVshu)]
    for sheet_name in sheet_names:
        if sheet_name == '评价指标2':
            df = pd.read_excel(data_out, sheet_name=sheet_name)
            spacing, HV = ['n{}m{}v{}spacing'.format(gongjianzhongshu, jiqishu, AGVshu)], ['n{}m{}v{}HV'.format(gongjianzhongshu, jiqishu, AGVshu)]
            # 将每一列数据存储到列表中
            for column_name in df.columns:
                if column_name == 'spacing':
                    spacing.extend(list(df[column_name]))
                elif column_name == 'HV':
                    HV.extend(list(df[column_name]))
        elif sheet_name != '评价指标':
            make_span_rule3D, Idle_time_rule3D, Agv_S_rule3D = Rule_re_du2(data_out, sheet_name)
            make_span.append(make_span_rule3D); Idle_time.append(Idle_time_rule3D); Agv_S.append(Agv_S_rule3D)
    data_pj.loc['n{}m{}v{}加工时间平均值'.format(gongjianzhongshu, jiqishu, AGVshu),:] = make_span;data_pj.loc['n{}m{}v{}空闲时间平均值'.format(gongjianzhongshu, jiqishu, AGVshu),:] = Idle_time;data_pj.loc['n{}m{}v{}运输距离平均值'.format(gongjianzhongshu, jiqishu, AGVshu),:] = Agv_S
    data_pj.loc['n{}m{}v{}spacing'.format(gongjianzhongshu, jiqishu, AGVshu), :] = spacing;data_pj.loc['n{}m{}v{}HV'.format(gongjianzhongshu, jiqishu, AGVshu),:] = HV
    print('n{}m{}v{}整合完成'.format(gongjianzhongshu, jiqishu, AGVshu))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_episode',
        type=int,
        default=max_episode,
        help='stop condition: number of max episode')
    parser.add_argument(
        '--algo',
        default='NSGA2_MDDQN',
        type=str,
        help='DQN/DDQN, represent DQN, double DQN respectively')
    parser.add_argument(
        '--dueling',
        default=False,
        type=bool,
        help=
        'if True, represent dueling DQN or dueling DDQN, else ord DQN or DDQN')
    parser.add_argument(
        '--warmup_size',
        type=int,
        default=50000,
        help='warmup size for agent to learn')
    parser.add_argument(
        '--test_every_steps',
        type=int,
        default=100000,
        help='the step interval between two consecutive evaluations')
    args = parser.parse_args()

    data_pj = pd.DataFrame(
        {'Site': [], 'rule1': [], 'rule2': [], 'rule3': [], 'rule4': [], 'rule5': [], 'rule6': [], 'rule7': [],
         'rule8': [], 'rule9': [], 'rule10': [], 'rule11': [], 'rule12': [], 'rule13': [], 'rule14': [], 'rule15': [],
         'rule16': [], 'rule17': [], 'rule18': [], 'NSGA2': [], 'MDDQN': [], 'NSGA2_MDDQN': [], })
    for i in Job_num:
       for j in Mac_sum:
           for k in AGV_num:
               '''田口实验'''
               # TIANKOU = [
               #            [100,0.1,0.02,0.0001,0.7,0.1],
               #            [100,0.2,0.04,0.005,0.75,0.2],
               #            [100,0.3,0.06,0.001,0.8,0.3],
               #            [100,0.4,0.08,0.05,0.85,0.4],
               #            [100,0.5,0.1,0.01,0.9,0.5],
               #            [125, 0.1, 0.04, 0.001, 0.85, 0.5],
               #            [125, 0.2, 0.06, 0.05, 0.9, 0.1],
               #            [125, 0.3, 0.08, 0.01, 0.7, 0.2],
               #            [125, 0.4, 0.1, 0.0001, 0.75, 0.3],
               #            [125, 0.5, 0.02, 0.005, 0.8, 0.4],
               #            [150, 0.1, 0.06, 0.01, 0.75, 0.4],
               #            [150, 0.2, 0.08, 0.0001, 0.8, 0.5],
               #            [150, 0.3, 0.10, 0.005, 0.85, 0.1],
               #            [150, 0.4, 0.02, 0.001, 0.9, 0.2],
               #            [150, 0.5, 0.04, 0.05, 0.7, 0.3],
               #            [175, 0.1, 0.08, 0.005, 0.9, 0.3],
               #            [175, 0.2, 0.10, 0.001, 0.7, 0.4],
               #            [175, 0.3, 0.02, 0.05, 0.75, 0.5],
               #            [175, 0.4, 0.04, 0.01, 0.8, 0.1],
               #            [175, 0.5, 0.06, 0.0001, 0.85, 0.2],
               #            [200, 0.1, 0.10, 0.05, 0.8, 0.2],
               #            [200, 0.2, 0.02, 0.01, 0.85, 0.3],
               #            [200, 0.3, 0.04, 0.0001, 0.9, 0.4],
               #            [200, 0.4, 0.06, 0.005, 0.7, 0.5],
               #            [200, 0.5, 0.08, 0.001, 0.75, 0.1]]
               # data_pj = pd.DataFrame({'Site': [], 'spacing': [], 'HV': []})
               # data_out_pj = 'data_out/tiankou5/result_n{}_m{}_v{}.xlsx'.format(i, j, k)
               # for ii in range(len(TIANKOU)):
               #     ps, Pc, Pm, LEARNING_RATE, GAMMA, tanlan = TIANKOU[ii][0],TIANKOU[ii][1],TIANKOU[ii][2],TIANKOU[ii][3],TIANKOU[ii][4],TIANKOU[ii][5]
               #     main(1, i, j, k, ps, Pc, Pm, LEARNING_RATE, GAMMA, tanlan, data_pj, False)
               '''正常运行混合算法'''
               main(XUNLIAN, i, j, k, 125, 0.3, 0.03, 0.001, 0.8, 0.1, False)
               # main(XUNLIAN, i, j, k, 100, 0.1, 0.02, 0.005, 0.85, 0.1, False)
               '''单独跑规则'''
               # Rule_re(XUNLIAN, i, j, k, 100, 0.1, 0.02, 0.005, 0.85, 0.1, False)
               '''根据调整数据输出3D图和重新计算评价指标'''
               # Shuchu_re(XUNLIAN, i, j, k)
               # for ps in [100,125,150,175,200]:
               #     main(1, i, j, k, ps, 0.1, 0.02, 0.005, 0.8, 0.1, data_pj, False)
               # for Pc in [0.1,0.3,0.5,0.7,0.9]:
               #     main(1, i, j, k, 100, Pc, 0.02, 0.005, 0.8, 0.1, data_pj, False)
               # for Pm in [0.02,0.04,0.06,0.08,0.1]:
               #     main(1, i, j, k, 100, 0.1, Pm, 0.005, 0.8, 0.1, data_pj, False)
               # for LEARNING_RATE in [0.0001,0.005,0.001,0.05,0.01]:
               #     main(1, i, j, k, 100, 0.1, 0.02, LEARNING_RATE, 0.8, 0.1, data_pj, False)
               # for GAMMA in [0.5,0.6,0.7,0.8,0.9]:
               #     main(1, i, j, k, 100, 0.1, 0.02, 0.005, GAMMA, 0.1, data_pj, False)
               # for tanlan in [0.1,0.2,0.3,0.4,0.5]:
               #     main(1, i, j, k, 100, 0.1, 0.02, 0.005, 0.8, tanlan, data_pj, False)
               # data_pj.to_excel(data_out_pj, sheet_name='评价指标', index=False)
               '''将数据整合到一个sheet'''
    #            Shuchu_re2(XUNLIAN, i, j, k, data_pj)
    # data_pj.to_excel('data_out/240401gai/REresult/REresult_all.xlsx', sheet_name='整合', index=False)