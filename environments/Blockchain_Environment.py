import copy
import random
from collections import namedtuple
import gym
import torch
from gym import wrappers
import numpy as np
import matplotlib as mpl
from gym import spaces
from gym.utils import seeding
from matplotlib import pyplot
from random import randint
from scipy.optimize import fminbound

class Blockchain_Environment(gym.Env):
    environment_name = "Smart Contract in blockchain"

    def __init__(self, num_vehicles=50, task_num=30):
        self.num_vehicles = num_vehicles
        self.task_num_per_episode = task_num
        self.vehicle_count = 0
        self.maxR = 500 #m, max relative distance between request vehicle and other vehicles
        self.maxV = 33 #km/h, max relative velocity between requst vehicle and other vehicles
        self.max_v = 50 # maximum vehicles in the communication range of request vehicle
        self.bandwidth = 10 # MHz
        self.snr_ref = 1 # reference SNR, which is used to compute rate by B*log2(1+snr_ref*d^-a) 
        self.snr_alpha = 2
        self.vehicle_F = range(3,8)  #GHz
        self.data_size = [0.2, 0.5] #MBytes
        self.comp_size = [0.2, 0.4] #GHz
        self.tau = [0.5, 1, 2, 4] #s
        self.max_datasize = max(self.data_size)
        self.max_compsize = max(self.comp_size)
        self.max_tau = max(self.tau)
        self.price_level = 10
        self.kappa = 0.05
        self.penalty = -np.log(1+self.max_tau)

        self.action_space = spaces.Discrete(self.num_vehicles*self.price_level)
        self.observation_space = spaces.Dict({
            "snr":spaces.Box(0,self.snr_ref,shape=(self.max_v,),dtype='float32'),
            "time_remain":spaces.Box(0,100,shape=(self.max_v,),dtype='float32'),
            "freq_remain":spaces.Box(0,max(self.vehicle_F),shape=(self.max_v,),dtype='float32'),
            "reliability":spaces.Box(0,1,shape=(self.max_v,),dtype='float32'),
            "task":spaces.Box(0,max(self.max_datasize,self.max_compsize,self.max_tau),shape=(3,),dtype='float32')})
        self.seed()
        self.reward_threshold = 0.0
        self.trials = 100
        self.max_episode_steps = 100
        self._max_episode_steps = 100
        self.id = "Offloading"
        self.count = [0,0,0,0]
        self.delay = [0,0,0,0]
        self.choice = [0]*self.num_vehicles
        self.serv_reward = [0]*self.num_vehicles
        self.count_file = "blockchain.txt"
        self.utility = 0
        self.vehicles = [] #vehicles in the range
        self.tasks = [] #tasks for offloading
        self.init_vehicles()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Resets the environment and returns the start state"""
        self.move_vehicles()
        # self.add_vehicle()
        self.step_count = 0
        self.next_state = None
        self.reward = None
        self.done = False
        self.move_vehicles()
        for v in self.vehicles:
            v["freq_remain"] = v["freq_init"]
            v["position"] = v["position_init"]
        task = self.tasks[0]
        self.s = {
            "snr":np.array([min(self.snr_ref*(abs(v["position"])/200)**-2, 1) for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles)),
            "time_remain":np.array([min(-v["position"]/v["velocity"]+500/abs(v["velocity"]), 100) for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles)),
            "freq_remain":np.array([v["freq_remain"] for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles)),
            "reliability":np.array([self.compute_reliability(v) for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles)),
            "task":np.array(task)}

        with open(self.count_file,'a') as f:
            f.write(str(self.utility)+' '+' '.join([str(i) for i in self.count])+' '+' '.join([str(i) for i in self.delay])+' '
            +' '.join([str(i) for i in self.choice])+' '+' '.join([str(i) for i in self.serv_reward])+' '
            +' '.join([str(i) for i in self.s["reliability"][:self.num_vehicles]])+'\n')
        self.count = [0,0,0,0]
        self.delay = [0,0,0,0]
        self.choice = [0]*self.num_vehicles
        self.serv_reward = [0]*self.num_vehicles
        self.utility = 0

        return spaces.flatten(self.observation_space, self.s)

    def step(self, action):
        self.step_count += 1
        self.reward = self.compute_reward(action)
        self.utility += self.reward
        self.move_vehicles()
        if self.step_count >= self.task_num_per_episode: 
            self.done = True
        else: 
            self.done = False
            self.s["snr"] = np.array([min(self.snr_ref*(abs(v["position"])/200)**-2, 1) for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles))
            self.s["freq_remain"] = np.array([v["freq_remain"] for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles))
            self.s["time_remain"] = np.array([min(-v["position"]/v["velocity"]+500/abs(v["velocity"]), 100) for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles))
            self.s["reliability"] = np.array([self.compute_reliability(v) for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles))
            task = self.tasks[self.step_count]
            self.s["task"] = np.array(task)
        return spaces.flatten(self.observation_space, self.s), self.reward, self.done, {}

    def compute_reward(self, action):
        """Computes the reward we would have got with this achieved goal and desired goal. Must be of this exact
        interface to fit with the open AI gym specifications"""
        task = self.s["task"]
        reward, utility_n, is_finish, v_id, freq_alloc = self.compute_utility(action, task)
        if v_id==self.num_vehicles:
            return reward
        v = self.vehicles[v_id]
        v["freq_remain"] -= freq_alloc
        v["task_num"]+=1
        v["finish_task"].pop(0)
        v["finish_task"].append(is_finish)
        v["utility"].pop(0)
        v["utility"].append(utility_n)
        return reward

    def init_vehicles(self):
        for i in range(self.num_vehicles):
            self.vehicle_count += 1
            # v_f = random.choice(self.vehicle_F)
            v_f = (i%3)*2+3
            v_p = random.uniform(-self.maxR*0.9,self.maxR*0.9)
            v_v = np.random.normal(0, self.maxV/2)
            v_v = v_v if v_v!=0 else random.choice([-0.1, 0.1])
            # v_r = random.choice(range(1,11))/10
            v_r = (i%10+1)/10
            self.vehicles.append({"id":self.vehicle_count, "position":v_p, "position_init":v_p, "velocity":v_v, "reliability":v_r,
            "freq_init":v_f, "freq_remain":v_f, "task_num":0, "finish_task":[0.0]*1000, "utility":[0.0]*1000})
            # np.random.shuffle(self.vehicles)

    def add_vehicle(self):
        if len(self.vehicles) <= self.num_vehicles:
            self.vehicle_count += 1
            v_f = np.random.choice(self.vehicle_F)
            v_v = np.random.normal(0,self.maxV/2)
            v_v = v_v if v_v!=0 else random.choice([-0.1, 0.1])
            v_p = -self.maxR if v_v>0 else self.maxR
            v_r = min(max(0, np.random.normal(0.5,0.2)),1)
            self.vehicles.append({"id":self.vehicle_count, "position":v_p, "position_init":v_p, "velocity":v_v, "reliability":v_r,
            "freq_init":v_f, "freq_remain":v_f, "task_num":0, "finish_task":[0.0]*1000, "utility":[0.0]*1000})

    def move_vehicles(self):
        for i in range(len(self.vehicles)):
            self.vehicles[i]["position"] += self.vehicles[i]["velocity"]/3.6*0.5
            if abs(self.vehicles[i]["position"]) >= self.maxR:
                self.vehicles[i]["position"] = -self.vehicles[i]["position"]
    
    def generate_offload_tasks(self, file, group_num):
        with open(file,'w+') as f:
            for _ in range(group_num):
                f.write("tasks:\n")
                tasks = []
                for j in range(len(self.tau)):
                    for _ in range(8):
                        max_t = self.tau[j]
                        data_size = random.uniform(self.data_size[0]*max_t*2,self.data_size[1]*max_t*2)
                        compute_size = random.uniform(self.comp_size[0]*max_t*2,self.comp_size[1]*max_t*2)
                        tasks.append(str(data_size)+' '+str(compute_size)+' '+str(max_t))
                np.random.shuffle(tasks)
                f.write('\n'.join(tasks)+'\n')

    def produce_action(self, action_type):
        if action_type=="random":
            v_id = np.random.choice(range(self.num_vehicles))
            fraction = np.random.choice(range(self.price_level-1))
        if action_type=="greedy":
            v_id = np.argmax(self.s["freq_remain"])
            task = self.s["task"]
            fraction = np.argmax([self.compute_utility(v_id*self.price_level+i, task, False)[0] for i in range(1,self.price_level)])
        action = v_id*self.price_level + fraction + 1
        return action

    def load_offloading_tasks(self, file, index):
        a = []
        self.tasks = []
        with open(file) as f:
            a = f.read().split("tasks:\n")[index].split('\n')
        for i in a[:-1]:
            tmp = i.split(' ')
            self.tasks.append([float(k) for k in tmp])

    def get_resouce_allocation(self, v, price, task, rate):
        F = v["freq_remain"]
        fmin = 0.001
        fmax = min(F,np.sqrt(price/self.kappa))
        if fmin > fmax:
            return 0.00001
        r = v["reliability"]
        f = lambda x:-r*np.log(1+max(task[2]-task[0]/rate-task[1]/x, 0.00001)) + (1-r)*self.kappa*x*x*task[1]
        try:
            alloc = fminbound(f, fmin, fmax)
        except:
            return 0.00001
        return alloc

    def compute_utility(self, action, task, is_count=True):
        is_finish = 0
        utility_n = 0
        v_id = action//self.price_level
        if v_id==self.num_vehicles:
            return 0, 0, 0, v_id, 0
        utility = -np.log(1+self.max_tau)
        v = self.vehicles[v_id]
        rate = self.bandwidth*np.log2(1+self.s["snr"][v_id])
        price = (action%self.price_level+1)/self.price_level*np.log(1+task[2])/task[1]
        self.choice[v_id]+=1
        freq_alloc = self.get_resouce_allocation(v, price, task, rate)
        self.serv_reward[v_id]+=(price*task[1]-self.kappa*freq_alloc*freq_alloc*task[1])
        t_total = task[0]/rate + task[1]/freq_alloc
        time_remain = max(-v["position"]/v["velocity"]+500/abs(v["velocity"]), 0.00001)
        if t_total <=min(task[2], time_remain):
            utility = np.log(1+task[2]-t_total) - price*task[1]
            if is_count:
                self.count[int(np.log2(task[2]))+1] += 1
                self.delay[int(np.log2(task[2]))+1] += t_total
            is_finish = 1
            utility_n = np.log(1+task[2]-t_total)/np.log(1+task[2])
        else:
            utility = self.penalty
        return utility, utility_n, is_finish, v_id, freq_alloc

    def compute_reliability(self, v):
        num = min(100,v["task_num"])
        if num==0:
            return 0
        average_r = sum(v["finish_task"][-num:])/num
        if sum(v["finish_task"][-num:])==0:
            average_u = 0
        else:
            average_u = sum(v["utility"][-num:])/sum(v["finish_task"][-num:])
        return 0.2*average_u + 0.8*average_r


class Consensus_Environment(gym.Env):
    environment_name = "Consensus in blockchain"

    def __init__(self, num_cons_nodes=20):
        self.num_BS = 50
        self.num_cons_nodes = num_cons_nodes
        self.BS_count = 0
        self.rate = 12.5 # MB/s
        self.BS_F = range(20,30)  #GHz
        self.rho = 0.1
        self.trans_num = 0
        self.batch_size = 0
        self.trans_size = 0.002
        self.trans_factor = 1
        self.block_interval = 5
        self.delta = 1.6
        self.xi = 0.02
        self.eps_1 = 0.1
        self.eps_2 = 0.001
        self.comp_a = 0.002
        self.comp_b = 0.008
        self.comp_c = 0.0005
        self.actions = []

        self.action_space = spaces.Discrete(self.num_BS)
        self.observation_space = spaces.Dict({
            "freq_remain":spaces.Box(0,max(self.BS_F),shape=(self.num_BS,),dtype='float32'),
            "reliability":spaces.Box(0,1,shape=(self.num_BS,),dtype='float32'),
            "trans_num":spaces.Box(0,5000,shape=(1,), dtype='float32')})
        self.seed()
        self.reward_threshold = 0.0
        self.trials = 100
        self.max_episode_steps = 100
        self._max_episode_steps = 100
        self.step_num_per_episode = self.num_cons_nodes + 1
        self.id = "Consensus"
        self.consensus_delay = 0
        self.count_file = "consensus.txt"
        self.utility = 0
        self.init_nodes()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Resets the environment and returns the start state"""
        self.step_count = 0
        self.next_state = None
        self.reward = None
        self.done = False
        with open(self.count_file,'a') as f:
            # f.write(str(self.utility/self.step_num_per_episode)+' '+str(self.consensus_delay/self.step_num_per_episode)+' '+'\n')
            f.write(' '.join([str(i) for i in self.actions])+'\n')
        self.actions=[]
        self.utility = 0
        self.consensus_delay = 0
        self.init_nodes()
        self.s = {
            "freq_remain":np.array([b["freq_remain"] for b in self.nodes]),
            "reliability":np.array([b["reliability"] for b in self.nodes]),
            "trans_num":np.array(self.trans_num)}
        return spaces.flatten(self.observation_space, self.s)

    def step(self, action):
        self.step_count += 1
        self.reward = self.compute_reward(action)
        self.utility += self.reward
        self.actions.append(action)
        if self.step_count == self.step_num_per_episode: 
            self.done = True
        else: 
            self.done = False
            self.s["freq_remain"] = np.array([b["freq_remain"] for b in self.nodes])
            self.s["reliability"] = np.array([b["reliability"] for b in self.nodes])
            self.s["trans_num"] = self.trans_num
        return spaces.flatten(self.observation_space, self.s), self.reward, self.done, {}

    def compute_reward(self, action):
        """Computes the reward we would have got with this achieved goal and desired goal. Must be of this exact
        interface to fit with the open AI gym specifications"""
        if self.step_count == 1:
            reward = 0
            self.batch_size = int((action+1)/self.num_BS*self.trans_num)
        else:
            reward, delay = self.compute_utility(action)
            self.consensus_delay += delay
            self.nodes[action]["freq_remain"] = 0
        return reward

    def init_nodes(self):
        self.nodes=[]
        self.trans_num = 0
        for _ in range(self.num_BS):
            self.BS_count += 1
            freq = random.choice(self.BS_F)
            num_vehicles = max(0,int(np.random.normal(90,30)))
            self.trans_num+=int(num_vehicles*self.trans_factor)
            freq_r=max(0,freq-self.rho*num_vehicles)
            r = min(max(0,np.random.normal(0.5, 0.2)),1)
            self.nodes.append({"id":self.BS_count, "freq_init":freq, "freq_remain":freq_r, "reliability":r})

    def change_nodes(self, action):
        pass
            
    def produce_action(self, num_nodes, action_type):
        if action_type=="random":
            selection = np.random.choice(list(range(self.num_BS)),size=num_nodes,replace=False)
            block_size = random.random()
        if action_type=="greedy":
            selection = np.argsort([n["freq_remain"] for n in self.nodes])[-num_nodes:]
            block_size = random.random()
        return selection, block_size

    def compute_utility(self, action):
        utility = 0
        block_size = self.batch_size*self.trans_size
        T_d = 50*block_size/self.rate
        N = self.num_cons_nodes
        f = (N-1)//3
        if self.step_count == 2:
            comp = self.batch_size*(self.comp_b+self.comp_c) + self.comp_a + (2*N+4*f)*self.comp_c
        else:
            comp = self.batch_size*(self.comp_b+self.comp_c) + (2*N+4*f)*self.comp_c
        if self.nodes[action]["freq_remain"]==0:
            T_v = 1e6
        else:
            T_v = comp/self.nodes[action]["freq_remain"]
        delay = T_d + T_v
        if delay <= self.delta*self.block_interval:
            utility = self.eps_1*self.nodes[action]["reliability"]+self.eps_2*self.batch_size
            utility = np.log(1 + self.delta*self.block_interval - delay) * utility
        else:
            utility = -20*np.log(1+self.delta*self.block_interval)
        return utility, delay

    def produce_utility(self, selection, block_size):
        N = self.num_cons_nodes
        f = (N-1)//3
        comp_p = self.batch_size*(self.comp_b+self.comp_c) + self.comp_a + (2*N+4*f)*self.comp_c
        comp_r = self.batch_size*(self.comp_b+self.comp_c) + (2*N+4*f)*self.comp_c
        freq_p = self.nodes[selection[-1]]["freq_remain"]
        freq_r = min([self.nodes[i]["freq_remain"] for i in selection[:-1]])
        T_d = 50*block_size/self.rate
        T_v = max(comp_p/freq_p,comp_r/freq_r)
        delay = T_d + T_v
        utility = 0
        for i in range(len(selection)):
            utility += (self.eps_1*self.nodes[selection[i]]["reliability"]+self.eps_2*self.batch_size)*np.log(1 + self.delta*self.block_interval - delay)
        return utility, delay
