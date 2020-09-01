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
        self.maxV = 30 #km/h, max relative velocity between requst vehicle and other vehicles
        self.max_v = 50 # maximum vehicles in the communication range of request vehicle
        self.bandwidth = 10 # MHz
        self.snr_ref = 1 # reference SNR, which is used to compute rate by B*log2(1+snr_ref*d^-a) 
        self.snr_alpha = 2
        self.vehicle_F = range(5,11)  #GHz
        self.data_size = [0.1, 0.2] #MBytes
        self.comp_size = [0.2, 0.4] #GHz
        self.tau = [0.5, 1, 2, 4] #s
        self.max_datasize = max(self.data_size)
        self.max_compsize = max(self.comp_size)
        self.max_tau = max(self.tau)
        self.ref_price = 0.1
        self.price = 0.1
        self.price_level = 10
        self.kappa = 0.1
        self.distance_factor = 1
        self.penalty = -np.log(1+self.max_tau)

        self.action_space = spaces.Discrete(self.num_vehicles*self.price_level)
        self.observation_space = spaces.Dict({
            "snr":spaces.Box(0,self.snr_ref,shape=(self.max_v,),dtype='float32'),
            "time_remain":spaces.Box(0,100,shape=(self.max_v,),dtype='float32'),
            "freq_remain":spaces.Box(0,max(self.vehicle_F),shape=(self.max_v,),dtype='float32'),
            "reliability":spaces.Box(0,1,shape=(self.max_v,),dtype='float32'),
            "task":spaces.Box(0,max(self.max_datasize,self.max_compsize,self.max_tau),shape=(3,),dtype='float32')})
        self.seed()
        self.trials = 100
        self.max_episode_steps = 100
        self._max_episode_steps = 100
        self.id = "Offloading"
        self.count = [0,0,0,0]
        self.delay = [0,0,0,0]
        self.count_file = "blockchain.txt"
        self.utility = 0
        self.vehicles = [] #vehicles in the range
        self.tasks = [] #tasks for offloading
        self.init_vehicles()
        # self.generate_offload_tasks()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Resets the environment and returns the start state"""
        # for _ in range(random.randint(1,10)):
        #     self.add_vehicle()
        self.move_vehicles()
        # self.add_vehicle()
        # self.generate_offload_tasks()
        self.step_count = 0
        self.next_state = None
        self.reward = None
        self.done = False
        for v in self.vehicles:
            v["freq"] = v["freq_init"]
            v["freq_remain"] = max(0, v["freq_init"] - sum([i[1]/i[2] for i in v["tasks"]]))
            v["position"] = v["position_init"]
            alpha_max = v["freq_remain"]/v["freq"]
            v["u_max"] = sum([np.log(1+alpha_max*i[2]) for i in v["tasks"]])
        with open(self.count_file,'a') as f:
            f.write(str(self.utility)+' '+' '.join([str(i) for i in self.count])+' '
            +' '.join([str(i) for i in self.delay])+' '+'\n')
        self.count = [0,0,0,0]
        self.delay = [0,0,0,0]
        self.utility = 0
        task = self.tasks[0]
        self.s = {
            "snr":np.array([min(self.snr_ref*(abs(v["position"])/200)**-2, 1) for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles)),
            "time_remain":np.array([min(-v["position"]/v["velocity"]+500/abs(v["velocity"]), 100) for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles)),
            "freq_remain":np.array([v["freq_remain"] for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles)),
            "reliability":np.array([v["u_max"] for v in self.vehicles] + [0]*(self.max_v-self.num_vehicles)),
            "task":np.array(task)}
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
            task = self.tasks[self.step_count]
            self.s["task"] = np.array(task)
        return spaces.flatten(self.observation_space, self.s), self.reward, self.done, {}

    def compute_reward(self, action):
        """Computes the reward we would have got with this achieved goal and desired goal. Must be of this exact
        interface to fit with the open AI gym specifications"""
        task = self.s["task"]
        reward, v_id, freq_alloc = self.compute_utility(action, task)
        if v_id==self.num_vehicles:
            return reward
        v = self.vehicles[v_id]
        v["freq"] -= freq_alloc
        v["freq_remain"] = max(0, v["freq"] - sum([i[1]/i[2] for i in v["tasks"]]))
        alpha_max = v["freq_remain"]/v["freq"]
        v["u_max"] = sum([np.log(1+alpha_max*i[2]) for i in v["tasks"]])
        return reward

    def init_vehicles(self):
        for _ in range(self.num_vehicles):
            self.vehicle_count += 1
            v_f = random.choice(self.vehicle_F)
            v_p = random.uniform(-self.maxR*0.9,self.maxR*0.9)
            v_v = np.random.normal(0, self.maxV/2)
            v_v = v_v if v_v!=0 else random.choice([-0.1, 0.1])
            self.vehicles.append({"id":self.vehicle_count, "position":v_p, "position_init":v_p, "velocity":v_v, "freq_init":v_f, "freq":v_f, "freq_remain":0, "tasks":[]})

    def add_vehicle(self):
        if len(self.vehicles) <= self.num_vehicles:
            self.vehicle_count += 1
            v_f = np.random.choice(self.vehicle_F)
            v_v = np.random.normal(0,self.maxV/2)
            v_v = v_v if v_v!=0 else random.choice([-0.1, 0.1])
            v_p = -self.maxR if v_v>0 else self.maxR
            self.vehicles.append({"id":self.vehicle_count, "position":v_p, "position_init":v_p, "velocity":v_v, "freq_init":v_f, "freq":v_f, "freq_remain":0, "tasks":[]})

    def move_vehicles(self):
        for i in range(len(self.vehicles)):
            self.vehicles[i]["position"] += self.vehicles[i]["velocity"]/3.6*0.1
            # if abs(self.vehicles[i]["position"]) >= self.maxR:
            #     self.vehicles.pop(i)
            #     self.add_vehicle()
    
    def generate_offload_tasks(self, file, task_num, group_num):
        with open(file,'w+') as f:
            for _ in range(group_num):
                f.write("tasks:\n")
                for _ in range(task_num):
                    max_t = random.choice(self.tau)
                    data_size = random.uniform(self.data_size[0]*max_t*2,self.data_size[1]*max_t*2)
                    compute_size = random.uniform(self.comp_size[0]*max_t*2,self.comp_size[1]*max_t*2)
                    task = [str(data_size), str(compute_size), str(max_t)]
                    f.write(' '.join(task)+'\n')

    def produce_action(self, action_type):
        if action_type=="random":
            v_id = np.random.choice(range(self.num_vehicles))
            fraction = np.random.choice(range(self.price_level-1))
        if action_type=="greedy":
            v_id = np.argmax(self.s["freq_remain"])
            task = self.s["task"]
            fraction = np.argmax([self.compute_utility(v_id*self.price_level+i, task)[0] for i in range(1,self.price_level)])
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
        fmin = task[1]/task[2]
        fmax = min(F,np.sqrt(price/self.kappa))
        if fmin > fmax:
            return 0
        r = v["reliability"]
        f = lambda x:r*np.log(1+task[2]-task[0]/rate-task[1]/x) - (1-r)*self.kappa*x*x*task[1]
        alloc = fminbound(f, fmin, fmax)
        return alloc

    def compute_utility(self, action, task):
        v_id = action//self.price_level
        if v_id==self.num_vehicles:
            return 0, v_id, 0
        utility = -np.log(1+self.max_tau)
        v = self.vehicles[v_id]
        rate = self.bandwidth*np.log2(1+self.s["snr"][v_id])
        freq_alloc = self.get_resouce_allocation(v, price, task, rate)
        cost = self.kappa*freq_alloc*freq_alloc*task[1]
        # fraction = (action%self.price_level)/self.price_level
        # freq_alloc = fraction*v["freq_remain"]
        if freq_alloc <= 0:
            return utility, v_id, 0
        t_total = task[0]/rate + task[1]/freq_alloc
        time_remain = max(-v["position"]/v["velocity"]+500/abs(v["velocity"]), 0.00001)
        # cost = self.ref_price*task[1]
        if t_total <=min(task[2], time_remain):
            utility = np.log(1+task[2]-t_total) - cost
            self.count[int(np.log2(task[2]))+1] += 1
            self.delay[int(np.log2(task[2]))+1] += t_total
        else:
            utility = self.penalty - cost
        return utility, v_id, freq_alloc



class Consensus_Environment(gym.Env):
    environment_name = "Consensus in blockchain"

    def __init__(self, num_nodes=20):
        self.num_BS = 50
        self.BS_count = 0
        self.rate = 12.5 # MB/s
        self.BS_F = range(20,30)  #GHz
        self.rho = 0.3
        self.trans_num = 0
        self.trans_factor = 0.5
        self.action_space = spaces.Box(-1,1,shape=(self.num_BS+1,), dtype='float32')
        self.observation_space = spaces.Dict({
            "freq_remain":spaces.Box(0,max(self.BS_F),shape=(self.num_BS,),dtype='float32'),
            "reliability":spaces.Box(0,1,shape=(self.num_BS,),dtype='float32'),
            "trans_num":spaces.Box(0,1,dtype='float32')})
        self.seed()
        self.trials = 100
        self.max_episode_steps = 100
        self._max_episode_steps = 100
        self.id = "Consensus"
        self.consensus_delay = 0
        self.count_file = "consensus.txt"
        self.utility = 0
        self.nodes = [] #vehicles in the range
        self.init_nodes()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Resets the environment and returns the start state"""
        self.change_nodes()
        self.step_count = 0
        self.next_state = None
        self.reward = None
        self.done = False
        with open(self.count_file,'a') as f:
            f.write(str(self.utility)+' '+self.consensus_delay+' '+'\n')
        self.utility = 0
        self.s = {
            "freq_remain":np.array([b["freq_remain"] for b in self.nodes]),
            "reliability":np.array([b["reliability"] for b in self.nodes]),
            "trans_num":np.array(self.trans_num)}
        return spaces.flatten(self.observation_space, self.s)

    def step(self, action):
        self.step_count += 1
        self.reward = self.compute_reward(action)
        self.utility += self.reward
        self.change_nodes()
        if self.step_count >= self.task_num_per_episode: 
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
        reward = self.compute_utility(action)
        return reward

    def init_nodes(self):
        for _ in range(self.num_BS):
            self.BS_count += 1
            freq = random.choice(self.BS_F)
            r = random.choice(range(51)/50)
            self.nodes.append({"id":self.BS_count, "freq_init":freq, "freq_remain":freq, "reliability":r})

    def change_nodes(self):
        self.trans_num
        for b in range(self.num_BS):
            num_vehicles = np.random.poisson(30)
            self.nodes[b]["freq_remain"]=self.nodes[b]["freq_init"]-self.rho*num_vehicles
            self.trans_num+=num_vehicles
        self.trans_num=self.trans_num*self.trans_factor
            

    def produce_action(self, action_type):
        if action_type=="random":
            action = 0
        if action_type=="greedy":
            action = 0
        return action

    def compute_utility(self, action):
        utility = 0
        return utility