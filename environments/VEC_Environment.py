import copy
import random
from collections import namedtuple
import gym
from gym import wrappers
import numpy as np
import matplotlib as mpl
from gym import spaces
from gym.utils import seeding
from matplotlib import pyplot
from random import randint
from scipy.optimize import fsolve

class VEC_Environment(gym.Env):
    environment_name = "Vehicular Edge Computing"

    def __init__(self, num_vehicles=50, task_num=30):
        self.num_vehicles = num_vehicles
        self.task_num_per_episode = task_num
        self.vehicle_count = 0
        self.maxR = 500 #m, max relative distance between request vehicle and other vehicles
        self.maxV = 30 #km/h, max relative velocity between requst vehicle and other vehicles
        self.max_v = 80 # maximum vehicles in the communication range of request vehicle
        self.max_local_task = 30
        self.bandwidth = 6 # MHz
        self.snr_ref = 1 # reference SNR, which is used to compute rate by B*log2(1+snr_ref*d^-a) 
        self.snr_alpha = 2
        self.vehicle_F = range(2,7)  #GHz
        self.max_datasize = 10 #MBytes
        self.max_compsize = 1 #GHz
        self.max_tau = 10 # s
        self.price = 0.1

        self.action_space = spaces.Tuple((spaces.Discrete(self.max_v),spaces.Box(self.price,10*np.log(1+self.max_tau)/self.max_compsize,(1,))))
        self.observation_space = spaces.Dict({"num_vehicles":spaces.Discrete(self.max_v),
        "position":spaces.Box(0,self.maxR,shape=(self.max_v,),dtype='float32'),
        "velocity":spaces.Box(0,self.maxV,shape=(self.max_v,),dtype='float32'),
        "freq_remain":spaces.Box(0,6,shape=(self.max_v,),dtype='float32'),
        "task":spaces.Box(0,max(self.max_datasize,self.max_compsize,self.max_tau),shape=(3,),dtype='float32')})
        self.seed()
        self.reward_threshold = 0.0
        self.trials = 100
        self.max_episode_steps = 100
        self.id = "VEC"
        
        self.vehicles = [] #vehicles in the range
        self.tasks = [] #tasks for offloading
        self.init_vehicles()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Resets the environment and returns the start state"""
        for _ in range(random.randint(1,10)):
            self.add_vehicle()
        self.move_vehicles()
        self.generate_local_tasks()
        self.generate_offload_tasks()
        self.step_count = 0
        self.next_state = None
        self.reward = None
        self.done = False
        task = self.tasks.pop()
        self.s = {"num_vehicles":np.array(len(self.vehicles)),
        "position":np.array([v["position"] for v in self.vehicles]+[0]*(self.max_v-len(self.vehicles))),
        "velocity":np.array([v["velocity"] for v in self.vehicles]+[0]*(self.max_v-len(self.vehicles))),
        "freq_remain":np.array([v["freq_remain"] for v in self.vehicles]+[0]*(self.max_v-len(self.vehicles))),
        "task":np.array([task["data_size"],task["compute_size"],task["max_t"]])}
        return self.s

    def step(self, action):
        self.step_count += 1
        self.reward = self.compute_reward(action)
        if action[0] < len(self.vehicles):
            self.s["freq_remain"][action[0]] = self.vehicles[action[0]]["freq_remain"]
        task = self.tasks.pop()
        self.s["task"] = [task["data_size"],task["compute_size"],task["max_t"]]
        if self.step_count >= self.task_num_per_episode: 
            self.done = True
        else: 
            self.done = False
        return self.s, self.reward, self.done, {}

    def compute_reward(self, action):
        """Computes the reward we would have got with this achieved goal and desired goal. Must be of this exact
        interface to fit with the open AI gym specifications"""
        v_id = action[0]
        cost = action[1][0]
        task = self.s["task"]
        reward = -np.log(1+self.max_tau)
        if v_id >= len(self.vehicles):
            return reward
        v = self.vehicles[v_id]
        if v["freq_remain"]==0:
            return reward
        alpha_max = v["freq_remain"]/v["freq"]
        u_max = sum([np.log(1+alpha_max*i["max_t"]) for i in v["tasks"]])
        alpha = fsolve(lambda a:sum([np.log(1+a*i["max_t"]) for i in v["tasks"]])-u_max+(cost-self.price)*task[1],0.1)[0]
        if alpha <=0:
            return reward
        freq_alloc = v["freq"]-(v["freq"]-v["freq_remain"])/(1-alpha)
        if freq_alloc <= 0:
            return reward
        snr = self.snr_ref if abs(v["position"])<50 else self.snr_ref*(abs(v["position"])/50)**-2
        t_total = task[0]/(self.bandwidth*np.log2(1+snr)) + task[1]/freq_alloc
        if t_total <= task[2]:
            reward = np.log(1+task[2]-t_total) - cost*task[1]
            v["freq"] -= freq_alloc
            v["freq_remain"] = v["freq"] - sum([i["compute_size"]/i["max_t"] for i in v["tasks"]])
            v["freq_remain"] = v["freq_remain"] if v["freq_remain"]>0 else 0
        return reward

    def init_vehicles(self):
        for _ in range(self.num_vehicles):
            self.vehicle_count += 1
            v_f = random.choice(self.vehicle_F)
            v_p = random.uniform(-self.maxR,self.maxR)
            v_v = random.uniform(-self.maxV,self.maxV)
            v_v = v_v if v_v!=0 else random.choice([-0.1, 0.1])
            self.vehicles.append({"id":self.vehicle_count, "freq":v_f, "position":v_p, "velocity":v_v, "freq_remain":v_f, "tasks":[]})

    def add_vehicle(self):
        if len(self.vehicles) <= self.max_v:
            self.vehicle_count += 1
            v_f = np.random.choice(self.vehicle_F)
            v_v = random.uniform(-self.maxV,self.maxV)
            v_v = v_v if v_v!=0 else random.choice([-0.1, 0.1])
            v_p = -self.maxR if v_v>0 else self.maxR
            self.vehicles.append({"id":self.vehicle_count, "freq":v_f, "position":v_p, "velocity":v_v, "freq_remain":v_f, "tasks":[]})

    def move_vehicles(self):
        for i in range(len(self.vehicles)-1,-1,-1):
            self.vehicles[i]["position"] += self.vehicles[i]["velocity"]*self.max_tau
            if abs(self.vehicles[i]["position"]) >= self.maxR:
                self.vehicles.pop(i)

    def generate_local_tasks(self):
        for v in self.vehicles:
            v["tasks"] = []
            for _ in range(random.randint(round(self.max_local_task/10),self.max_local_task)):
                data_size = random.uniform(self.max_datasize/10,self.max_datasize)
                compute_size = random.uniform(self.max_compsize/10,self.max_compsize)
                max_t = random.uniform(self.max_tau/10,self.max_tau)
                v["tasks"].append({"data_size":data_size, "compute_size":compute_size, "max_t":max_t})
            v["freq_remain"] = v["freq"] - sum([i["compute_size"]/i["max_t"] for i in v["tasks"]])
            v["freq_remain"] = v["freq_remain"] if v["freq_remain"]>0 else 0
    
    def generate_offload_tasks(self):
        for _ in range(self.task_num_per_episode):
            data_size = random.uniform(self.max_datasize/10,self.max_datasize)
            compute_size = random.uniform(self.max_compsize/10,self.max_compsize)
            max_t = random.uniform(self.max_tau/10,self.max_tau)
            self.tasks.append({"data_size":data_size, "compute_size":compute_size, "max_t":max_t})