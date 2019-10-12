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

    def __init__(self):
        self.numVehicles = 500
        self.num_of_task_generate_per_step = 100
        self.vehicle_count = 0
        self.snr_level = 100
        self.freq_level = 100
        self.mu = 0.1 #ms
        self.actions = set(range(4))
        self.reward_for_achieving_goal = (self.grid_width + self.grid_height) * 3.0
        self.step_reward_for_not_achieving_goal = -1.0
        self.state_only_dimension = 1
        self.possible_states = [snr_level]*(self.numVehicles+1) + [freq_level]*(self.actionsnumVehicles+1) 
        self.possible_actions = [self.numVehicles+1, 10]
        self.action_space = spaces.MultiDiscrete(self.possible_actions)
        self.observation_space = spaces.MultiDiscrete(self.possible_states)

        self.seed()
        self.reward_threshold = 0.0
        self.trials = 100
        self.max_episode_steps = 100
        self.id = "VEC"
        self.vehicles = [] #vehicle element template [id, frequence, position, task_list]
        self.velocity = range(60,110)
        self.slot = 100 # ms
        self.bandwidth = 6 # MHz
        self.snr_ref = 0 # reference SNR, which is used to compute rate by B*log2(1+snr_ref*d^-a) 
        self.snr_alpha = 3
        self.maxL = 10000 #m, max length of RSU range 
        self.vehicle_F = range(2,6)  #GHz
        self.RSU_F = 20  #GHz
        self.tasks = []

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Resets the environment and returns the start state"""
        self.desired_goal = [self.location_to_state(self.current_goal_location)]
        self.achieved_goal = [self.location_to_state(self.current_user_location)]
        self.step_count = 0
        self.state = [self.location_to_state(self.current_user_location), self.location_to_state(self.current_goal_location)]
        self.next_state = None
        self.reward = None
        self.done = False
        self.achieved_goal = self.state[:self.state_only_dimension]
        self.s = np.array(self.state[:self.state_only_dimension])
        return self.s

    def step(self, action):
        self.step_count += 1
        self.next_state = self.calculate_next_state(action)
        self.reward = self.compute_reward()
        if len(self.vehicles)==0: 
            self.done = True
        else: 
            self.done = False
        self.state = self.next_state
        self.s = np.array(self.next_state[:self.state_only_dimension])
        if np.random.choice(range(100)) < 10:
            self.add_vehicle()

        return self.s, self.reward, self.done, {}

    def compute_reward(self):
        """Computes the reward we would have got with this achieved goal and desired goal. Must be of this exact
        interface to fit with the open AI gym specifications"""
        if (achieved_goal == desired_goal).all():
            reward = self.reward_for_achieving_goal
        else:
            reward = self.step_reward_for_not_achieving_goal
        return reward

    def add_vehicle(self):
        if self.vehicle_count > self.numVehicles:
            return
        v_f = np.random.choice(vehicle_F)
        v_p = 0
        v_v = np.random.choice(self.velocity)
        vehicles.append({"id":self.vehicle_count, "freq":v_f, "position":v_p, "tasks":[]})

    def generate_tasks(self, task_num):
        for i in range(task_num):
            data_size = random.randint()
            compute_size = random.randint()
            max_delay = random.randint()
            t_total = 0
            self.tasks.append({"data_size":data_size, "compute_size":compute_size, "max_t":max_t, "start_time":start_time, "source":source})

    def compute_delay(self, task, action):
        delay = 0
        if task[3]==action["device"]:
            delay = self.mu
        else:
            dp = abs(self.vehicles[task[3]][2] - self.vehicles[action["device"]][2])
            delay += task[0]/(self.snr_ref*(dp**self.snr_alpha))
            delay += task[1]/get_freq_allocation(vehicles[action["device"]],task)
        return delay

    def get_freq_allocation(self, vehicle, task，price):
        for t in vehicle

    def finish_tasks(self):
        for v in range(len(self.vehicles)):
            if 
    def move_vehicles(self):
        for i in range(len(vehicles)):
            vehicles[i][2]+=vehicles[i][3]*slot/1000.0
            if vehicles[i][2] > maxL:
                vehicles.pop(i)