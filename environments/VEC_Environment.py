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

    def __init__(self, num_vehicles=500, num_tasks=100, with_V2V=True, with_V2R=True):
        self.max_num_of_task_generate_per_step = num_tasks
        self.vehicles_for_offload = 1
        if with_V2V:
            self.vehicles_for_offload = num_vehicles
        self.RSU_for_offload = 0
        if with_V2R:
            self.RSU_for_offload = 10
        self.vehicle_count = 0
        self.snr_level = 100
        self.freq_level = 100
        self.cost_level = 100
        self.mu = 0.1 #ms
        self.maxL = 10000 #m, max length of road
        self.maxV = 30
        self.velocity = np.arange(15, self.maxV, 0.1)
        self.slot = 100 # ms
        self.bandwidth = 6 # MHz
        self.snr_ref = 1 # reference SNR, which is used to compute rate by B*log2(1+snr_ref*d^-a) 
        self.snr_alpha = 2
        self.vehicle_F = range(2,6)  #GHz
        self.RSU_F = 20  #GHz

        self.possible_states = [self.maxL]*(self.vehicles_for_offload) + [self.freq_level]*(self.vehicles_for_offload+self.RSU_for_offload) 
        self.possible_actions = [self.vehicles_for_offload+self.RSU_for_offload]*self.max_num_of_task_generate_per_step + [self.cost_level]*self.max_num_of_task_generate_per_step
        self.action_space = spaces.MultiDiscrete(self.possible_actions)
        self.observation_space = spaces.Tuple(spaces.Discrete(max_v),spaces.Box(0,))
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
        self.s = np.array([0]*self.vehicles_for_offload+[0]*self.vehicles_for_offload+[self.RSU_F]*self.RSU_for_offload)
        return self.s

    def step(self, action):
        self.step_count += 1
        self.tasks = []
        self.generate_tasks(self.max_num_of_task_generate_per_step)
        self.next_state,self.reward = self.step_next_state(action)
        if len(self.vehicles)==0: 
            self.done = True
        else: 
            self.done = False
        if np.random.choice(range(100)) < 10:
            self.add_vehicle()
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
        for v in self.vehicles:
            freq_remain = v["freq"] - sum([i["freq"] for i in v["tasks"]])
            if freq_remain <= 0:
                reward += sum([-np.log(1+i["max_t"]) for i in v["new_tasks"]])
            else:
                own_tasks = [i for i in v["new_tasks"] if i["source"]==v["id"]]
                other_tasks = [i for i in v["new_tasks"] if i["source"]!=v["id"]]
                for i in own_tasks:
                    freq = i["compute_size"]/(i["max_t"]-self.mu)
                    if freq > freq_remain:
                        reward += -np.log(1+i["max_t"])
                    else:
                        i["freq"] = freq
                        freq_remain -= freq
                        reward += np.log(1+self.mu)
                        v["tasks"].append(i)
                if freq_remain <= 0:
                    reward += sum([-np.log(1+task["max_t"]) for task in other_tasks])
                else:
                    total_cost = [task["cost"] for task in other_tasks]
                    for i in other_tasks:
                        freq = i["cost"]/total_cost*freq_remain
                        dp = abs(v["position"] - i["source"]["position"])
                        snr = self.snr_ref*((dp/50)**-2) if dp>50 else self.snr_ref
                        delay = i["data_size"]/(self.bandwidth*np.log2(1+snr)) + i["compute_size"]/freq
                        if delay <= i["max_t"]:
                            i["freq"] = freq
                            reward += np.log(1+(i["max_t"]-delay))
                            v["tasks"].append(i)
                        else:
                            reward += -np.log(1+i["max_t"])
        return reward

    def add_vehicle(self):
        if self.vehicle_count > self.vehicles_for_offload:
            return
        v_f = np.random.choice(self.vehicle_F)
        v_p = 0
        v_v = np.random.choice(self.velocity)
        self.vehicles.append({"id":self.vehicle_count, "freq":v_f, "position":v_p, "velocity":v_v, "tasks":[], "new_tasks":[]})

    def move_vehicles(self):
        for i in range(len(self.vehicles)):
            self.vehicles[i]["position"]+=self.vehicles[i]["velocity"]*self.slot/1000.0
            if self.vehicles[i]["position"] >self. maxL:
                self.vehicles.pop(i)

    def generate_tasks(self, task_num):
        v_id = [v["id"] for v in self.vehicles]
        v_w = [x%20+1 for x in v_id]
        for _ in range(task_num):
            data_size = random.randint(1,10)
            compute_size = random.randint(1,10)
            max_t = random.randint(1,10)
            start_time = self.step_count * self.slot
            source = random.choices(v_id, weights=v_w, k=1)[0]
            freq = 0
            cost = 0
            self.tasks.append({"data_size":data_size, "compute_size":compute_size, "max_t":max_t, "start_time":start_time, "source":source, "freq":freq, "cost":cost})

    def tasks_update(self):
        for v in self.vehicles:
            tmp=[]
            for i in v["tasks"]:
                if self.step_count*self.slot - i[0] < i[1]:
                    tmp.append(i)
            v["tasks"] = tmp



