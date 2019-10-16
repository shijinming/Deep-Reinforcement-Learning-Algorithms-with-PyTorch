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

class VEC_Environment(gym.Env):
    environment_name = "Vehicular Edge Computing"

    def __init__(self, num_vehicles=50, task_num=30):
        self.num_vehicles = num_vehicles
        self.task_num_per_episode = task_num
        self.vehicle_count = 0
        self.maxR = 500 #m, max relative distance between request vehicle and other vehicles
        self.maxV = 30 #km/h, max relative velocity between requst vehicle and other vehicles
        self.max_v = 80 # maximum vehicles in the communication range of request vehicle
        self.max_tau = 10 # s
        self.bandwidth = 6 # MHz
        self.snr_ref = 1 # reference SNR, which is used to compute rate by B*log2(1+snr_ref*d^-a) 
        self.snr_alpha = 2
        self.vehicle_F = range(2,7)  #GHz
        self.init_vehicles()

        self.action_space = spaces.Tuple((spaces.Discrete(self.max_v),spaces.Box(0,1,(1,))))
        self.observation_space = spaces.Dict({"num_vehicles":spaces.Discrete(self.max_v),
        "position":spaces.Box(0,self.maxR,shape=(0,self.max_v,)),
        "velocity":spaces.Box(0,self.maxV,shape=(self.max_v,)),
        "freq_remain":spaces.Box(0,6,shape=(self.max_v,))})
        self.seed()
        self.reward_threshold = 0.0
        self.trials = 100
        self.max_episode_steps = 100
        self.id = "VEC"
        self.vehicles = [] #vehicles in the range
        self.tasks = [] #tasks for offloading

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Resets the environment and returns the start state"""
        self.add_vehicle()
        self.move_vehicles()
        self.generate_local_tasks()
        self.generate_offload_tasks()
        self.step_count = 0
        self.next_state = None
        self.reward = None
        self.done = False
        self.s = {"num_vehicles":np.array(len(self.vehicles)),
        "position":np.array([v["position"] for v in self.vehicles]+[0]*(self.max_v-len(self.vehicles))),
        "velocity":np.array([v["velocity"] for v in self.vehicles]+[0]*(self.max_v-len(self.vehicles))),
        "freq_remain":np.array([v["freq_remain"] for v in self.vehicles]+[0]*(self.max_v-len(self.vehicles)))}
        return self.s

    def step(self, action):
        self.step_count += 1
        self.next_state = self.s
        self.next_state["freq_remain"][action[1]]
        if self.step_count >= self.task_num_per_episode: 
            self.done = True
        else: 
            self.done = False
        return self.next_state, self.reward, self.done, {}

    def compute_reward(self):
        """Computes the reward we would have got with this achieved goal and desired goal. Must be of this exact
        interface to fit with the open AI gym specifications"""
        reward = 0

        return reward

    def init_vehicles(self):
        for _ in range(self.num_vehicles):
            self.vehicle_count += 1
            v_f = random.choice(self.vehicle_F)
            v_p = random.uniform(-self.maxR,self.maxR)
            v_v = random.uniform(-self.maxV,self.maxV)
            v_v = v_v if v_v!=0 else random.choice([-0.1, 0.1])
            self.vehicles.append({"id":self.vehicle_count, "freq":v_f, "position":v_p, "velocity":v_v, "tasks":[], "freq_remain":v_f})

    def add_vehicle(self):
        if len(self.vehicles) <= self.max_v:
            self.vehicle_count += 1
            v_f = np.random.choice(self.vehicle_F)
            v_v = random.uniform(-self.maxV,self.maxV)
            v_v = v_v if v_v!=0 else random.choice([-0.1, 0.1])
            v_p = -self.maxR if v_v>0 else self.maxR
            self.vehicles.append({"id":self.vehicle_count, "freq":v_f, "position":v_p, "velocity":v_v, "tasks":[], "freq_remain":v_f})

    def move_vehicles(self):
        for i in range(len(self.vehicles)-1,-1,-1):
            self.vehicles[i]["position"] += self.vehicles[i]["velocity"]*self.max_tau
            if abs(self.vehicles[i]["position"]) >= self.maxR:
                self.vehicles.pop(i)

    def generate_local_tasks(self):
        for v in self.vehicles:
            v["tasks"] = []
            for _ in range(random.randint(1,10)):
                data_size = random.randint(1,10)
                compute_size = random.randint(1,10)
                max_t = random.randint(1,10)
                v["tasks"].append({"data_size":data_size, "compute_size":compute_size, "max_t":max_t})
            v["freq_remain"] = sum([i["compute_size"]/i["max_t"] for i in v["tasks"]])
    
    def generate_offload_tasks(self):
        for _ in range(self.task_num_per_episode):
            data_size = random.randint(1,10)
            compute_size = random.randint(1,10)
            max_t = random.randint(1,10)
            self.tasks.append({"data_size":data_size, "compute_size":compute_size, "max_t":max_t})
