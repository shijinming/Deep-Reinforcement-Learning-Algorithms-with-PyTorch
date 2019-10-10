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
        self.snr_level=100
        self.freq_level=100
        self.actions = set(range(4))
        self.reward_for_achieving_goal = (self.grid_width + self.grid_height) * 3.0
        self.step_reward_for_not_achieving_goal = -1.0
        self.state_only_dimension = 1
        self.possible_states = [snr_level]*(numVehicles+1) + [freq_level]*(numVehicles+1) 
        self.action_space = spaces.Dict({"device":spaces.Discrete(self.numVehicles+1), "cost":spaces.Discrete(10)})
        self.observation_space = spaces.MultiDiscrete(self.possible_states)

        self.seed()
        self.reward_threshold = 0.0
        self.trials = 100
        self.max_episode_steps = 100
        self.id = "VEC"
        self.vehicles = []
        self.vehicle_id = 0
        self.velocity = range(60,110)
        self.slot = 100 # ms
        self.bandwidth = 6 # MHz
        self.snr_ref = 0 # reference SNR, which is used to compute rate by B*log2(1+snr_ref*d^-a) 
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

    def step(self, desired_action):
        if type(desired_action) is np.ndarray:
            assert desired_action.shape[0] == 1
            assert len(desired_action.shape) == 1
            desired_action = desired_action[0]
        self.step_count += 1
        action = self.determine_which_action_will_actually_occur(desired_action)
        desired_new_state = self.calculate_desired_new_state(action)
        self.next_state = [self.location_to_state(self.current_user_location), self.desired_goal[0]]

        if self.step_count >= self.max_episode_steps: 
            self.done = True
        else: 
            self.done = False
        self.achieved_goal = self.next_state[:self.state_only_dimension]
        self.state = self.next_state
        self.s = np.array(self.next_state[:self.state_only_dimension])
        if np.random.choice(range(100)) < 10:
            self.add_vehicle()

        return self.s, self.reward, self.done, {}

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Computes the reward we would have got with this achieved goal and desired goal. Must be of this exact
        interface to fit with the open AI gym specifications"""
        if (achieved_goal == desired_goal).all():
            reward = self.reward_for_achieving_goal
        else:
            reward = self.step_reward_for_not_achieving_goal
        return reward

    def add_vehicle(self):
        v_f = np.random.choice(vehicle_F)
        v_p = 0
        v_v = np.random.choice(self.velocity)
        tasks = []
        vehicles.append([self.vehicle_id, v_f, v_p, v_v, tasks])
        vehicle_id+=1

    def generate_tasks(self):
        for v in vehicles:
            data_size = random.randint()
            compute_size = random.randint()
            max_delay = random.randint()
            t_total = 0
            self.tasks.append([data_size, compute_size, max_t, source])

    def compute_delay(self, task, action):
        delay = 0
        if task[3]==action["device"]:
            delay += task[1]/self.vehicles[action["device"]][]

    def move_vehicles(self):
        for v in range(len(vehicles)):
            vehicles[i][2]+=vehicles[i][3]*slot/1000.0
            if vehicles[i][2] > maxL:
                vehicles.pop(i)