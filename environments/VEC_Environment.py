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

    def __init__(self, num_vehicles=80, task_num=50):
        self.num_vehicles = num_vehicles
        self.task_num_per_episode = task_num
        self.vehicle_count = 0
        self.maxR = 500 #m, max relative distance between request vehicle and other vehicles
        self.maxV = 30 #km/h, max relative velocity between requst vehicle and other vehicles
        self.max_v = 80 # maximum vehicles in the communication range of request vehicle
        self.max_local_task = 1
        self.bandwidth = 6 # MHz
        self.snr_ref = 1 # reference SNR, which is used to compute rate by B*log2(1+snr_ref*d^-a) 
        self.snr_alpha = 2
        self.vehicle_F = range(2,7)  #GHz
        self.max_datasize = 2 #MBytes
        self.max_compsize = 0.2 #GHz
        self.max_tau = 5 # s
        self.price = 0.1
        self.max_price = np.log(1+self.max_tau)

        self.action_space = spaces.Discrete(self.num_vehicles*100)
        self.observation_space = spaces.Dict({
            "snr":spaces.Box(0,self.snr_ref,shape=(self.max_v,),dtype='float32'),
            "time_remain":spaces.Box(0,100,shape=(self.max_v,),dtype='float32'),
            "freq_remain":spaces.Box(0,6,shape=(self.max_v,),dtype='float32'),
            "task":spaces.Box(0,max(self.max_datasize,self.max_compsize,self.max_tau),shape=(3,),dtype='float32')})
        self.seed()
        self.reward_threshold = 0.0
        self.trials = 100
        self.max_episode_steps = 100
        self._max_episode_steps = 100
        self.id = "VEC"
        
        self.vehicles = [] #vehicles in the range
        self.tasks = [] #tasks for offloading
        self.init_vehicles()
        self.generate_offload_tasks()
        self.generate_local_tasks()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Resets the environment and returns the start state"""
        # for _ in range(random.randint(1,10)):
        #     self.add_vehicle()
        # self.move_vehicles()
        # self.generate_local_tasks()
        self.step_count = 0
        self.next_state = None
        self.reward = None
        self.done = False
        task = self.tasks[0]
        self.s = {
            "snr":np.array([min(self.snr_ref*(abs(v["position"])/200)**-2, 1) for v in self.vehicles]),
            "time_remain":np.array([min(-v["position"]/v["velocity"]+500/abs(v["velocity"]), 100) for v in self.vehicles]),
            "freq_remain":np.array([v["freq_remain"] for v in self.vehicles]),
            "task":np.array([task["data_size"],task["compute_size"],task["max_t"]])}
        return spaces.flatten(self.observation_space, self.s)

    def step(self, action):
        self.step_count += 1
        self.reward = self.compute_reward(action)
        self.s["freq_remain"][action//100] = self.vehicles[action//100]["freq_remain"]
        if self.step_count >= self.task_num_per_episode: 
            self.done = True
        else: 
            self.done = False
            task = self.tasks[self.step_count]
            self.s["task"] = [task["data_size"],task["compute_size"],task["max_t"]]
        return spaces.flatten(self.observation_space, self.s), self.reward, self.done, {}

    def compute_reward(self, action):
        """Computes the reward we would have got with this achieved goal and desired goal. Must be of this exact
        interface to fit with the open AI gym specifications"""
        task = self.s["task"]
        v_id = action//100
        cost = (action/100-v_id)*self.max_price + self.price*task[1]
        reward = -np.log(1+self.max_tau)
        v = self.vehicles[v_id]
        if v["freq_remain"]==0:
            return reward
        alpha_max = v["freq_remain"]/v["freq"]
        u_max = sum([np.log(1+alpha_max*i["max_t"]) for i in v["tasks"]])
        alpha = fsolve(lambda a:sum([np.log(1+a*i["max_t"]) for i in v["tasks"]])-u_max+ cost -self.price*task[1],0.1)[0]
        if alpha <=0:
            return reward
        freq_alloc = v["freq"]-(v["freq"]-v["freq_remain"])/(1-alpha)
        if freq_alloc <= 0:
            return reward
        snr = self.s["snr"][v_id]
        t_total = task[0]/(self.bandwidth*np.log2(1+snr)) + task[1]/freq_alloc
        # print("computing time=",task[1]/freq_alloc,"freq_min=",task[1]/task[2])
        if t_total <= task[2]:
            reward = np.log(1+task[2]-t_total) - cost
            v["freq"] -= freq_alloc
            v["freq_remain"] = v["freq"] - sum([i["compute_size"]/i["max_t"] for i in v["tasks"]])
            v["freq_remain"] = v["freq_remain"] if v["freq_remain"]>0 else 0
            # print("t_total=",t_total,"reward=",reward)
        return reward

    def init_vehicles(self):
        for _ in range(self.num_vehicles):
            self.vehicle_count += 1
            v_f = random.choice(self.vehicle_F)
            v_p = random.uniform(-self.maxR,self.maxR)
            v_v = random.uniform(-self.maxV,self.maxV)
            v_v = v_v if v_v!=0 else random.choice([-0.1, 0.1])
            self.vehicles.append({"id":self.vehicle_count, "position":v_p, "velocity":v_v, "freq_init":v_f, "freq":v_f, "freq_remain":0, "tasks":[]})

    def add_vehicle(self):
        if len(self.vehicles) <= self.max_v:
            self.vehicle_count += 1
            v_f = np.random.choice(self.vehicle_F)
            v_v = random.uniform(-self.maxV,self.maxV)
            v_v = v_v if v_v!=0 else random.choice([-0.1, 0.1])
            v_p = -self.maxR if v_v>0 else self.maxR
            self.vehicles.append({"id":self.vehicle_count, "position":v_p, "velocity":v_v, "freq_init":v_f, "freq":v_f, "freq_remain":0, "tasks":[]})

    def move_vehicles(self):
        for i in range(len(self.vehicles)-1,-1,-1):
            self.vehicles[i]["position"] += self.vehicles[i]["velocity"]*self.max_tau
            if abs(self.vehicles[i]["position"]) >= self.maxR:
                self.vehicles.pop(i)
                self.add_vehicle()

    def generate_local_tasks(self):
        for v in self.vehicles:
            v["tasks"] = []
            for _ in range(random.randint(0,self.max_local_task)):
                data_size = random.uniform(self.max_datasize/5,self.max_datasize)
                compute_size = random.uniform(self.max_compsize/5,self.max_compsize)
                max_t = random.randint(1, self.max_tau)
                v["tasks"].append({"data_size":data_size, "compute_size":compute_size, "max_t":max_t})
            v["freq_remain"] = v["freq_init"] - sum([i["compute_size"]/i["max_t"] for i in v["tasks"]])
            v["freq_remain"] = v["freq_remain"] if v["freq_remain"]>0 else 0
    
    def generate_offload_tasks(self):
        for _ in range(self.task_num_per_episode):
            data_size = random.uniform(self.max_datasize/5,self.max_datasize)
            compute_size = random.uniform(self.max_compsize/5,self.max_compsize)
            max_t = random.randint(1, self.max_tau)
            self.tasks.append({"data_size":data_size, "compute_size":compute_size, "max_t":max_t})