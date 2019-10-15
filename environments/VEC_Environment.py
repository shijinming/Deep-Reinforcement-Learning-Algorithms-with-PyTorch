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

    def __init__(self, num_vehicles=500):
        self.num_vehicles = num_vehicles
        self.vehicle_count = 0
        self.snr_level = 100
        self.freq_level = 100
        self.cost_level = 100
        self.mu = 0.1 #ms
        self.maxL = 10000 #m, max length of road
        self.max_v = 30 # maximum vehicles in a communication range
        self.max_dur = 10 #s, max duration of a communication link
        self.velocity = np.arange(15, 30, 0.1)
        self.slot = 100 # ms
        self.bandwidth = 6 # MHz
        self.snr_ref = 1 # reference SNR, which is used to compute rate by B*log2(1+snr_ref*d^-a) 
        self.snr_alpha = 2
        self.vehicle_F = range(2,6)  #GHz

        self.action_space = spaces.Tuple((spaces.Discrete(self.max_v),spaces.Box(0,1,(1,))))
        self.observation_space = spaces.Dict({"num_vehicles":spaces.Discrete(self.max_v),
        "position":spaces.Box(0,self.maxL,shape=(0,self.max_v,)),
        "duration":spaces.Box(0,self.max_dur,shape=(self.max_v,)),
        "freq_remain":spaces.Box(0,6,shape=(self.max_v,))})
        self.seed()
        self.reward_threshold = 0.0
        self.trials = 100
        self.max_episode_steps = 100
        self.id = "VEC"
        self.vehicles = [] #vehicle element template [id, frequence, position, task_list]
        self.tasks = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Resets the environment and returns the start state"""
        self.step_count = 0
        self.next_state = None
        self.reward = None
        self.done = False
        self.s = {"num_vehicles":np.array([0]),
        "position":np.array([0]*self.max_v),
        "duration":np.array([0]*self.max_v),
        "freq_remain":np.array([0]*self.max_v)}
        return self.s

    def step(self, action):
        self.step_count += 1
        if np.random.choice(range(100)) < 10:
            self.add_vehicle()
        self.generate_tasks()
        self.next_state,self.reward = self.step_next_state(action)
        if len(self.vehicles)==0: 
            self.done = True
        else: 
            self.done = False
        return self.next_state, self.reward, self.done, {}

    def step_next_state(self, action):
        state = [0]*len(self.possible_states)
        self.move_vehicles()
        self.tasks_update()
        for i in range(len(self.tasks)):
            self.tasks[i]["cost"] = action[i+self.max_num_of_task_generate_per_step]
            self.vehicles[action[i]]["new_tasks"].append(self.tasks[i])
        reward = self.compute_reward()
        for v in self.vehicles:
            state[v["id"]] = round(v["position"])
            freq_r = v["freq"] - sum([i["freq"] for i in v["tasks"]])
            state[v["id"]+self.vehicles_for_offload] = freq_r if freq_r>0 else 0
        return np.array(state), reward

    def compute_reward(self):
        """Computes the reward we would have got with this achieved goal and desired goal. Must be of this exact
        interface to fit with the open AI gym specifications"""
        reward = 0

        return reward

    def add_vehicle(self):
        if self.vehicle_count > self.num_vehicles:
            return
        v_f = np.random.choice(self.vehicle_F)
        v_p = 0
        v_v = np.random.choice(self.velocity)
        self.vehicles.append({"id":self.vehicle_count, "freq":v_f, "position":v_p, "velocity":v_v, "tasks":[], "freq_remain":v_f})

    def move_vehicles(self):
        for i in range(len(self.vehicles)):
            self.vehicles[i]["position"]+=self.vehicles[i]["velocity"]*self.slot/1000.0
            if self.vehicles[i]["position"] >self. maxL:
                self.vehicles.pop(i)

    def generate_tasks(self):
        for v in self.vehicles:
            v["tasks"] = []
            for _ in range(random.randint(1,10)):
                data_size = random.randint(1,10)
                compute_size = random.randint(1,10)
                max_t = random.randint(1,10)
                v["tasks"].append({"data_size":data_size, "compute_size":compute_size, "max_t":max_t})
            v["freq_remain"] = sum([i["compute_size"]/i["max_t"] for i in v["tasks"]])



