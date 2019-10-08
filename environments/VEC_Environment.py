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
        self.numVehicles = 
        self.actions = set(range(4))
        self.reward_for_achieving_goal = (self.grid_width + self.grid_height) * 3.0
        self.step_reward_for_not_achieving_goal = -1.0
        self.state_only_dimension = 1
        self.num_possible_states = self.grid_height * self.grid_width
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.num_possible_states)

        self.seed()
        self.reward_threshold = 0.0
        self.trials = 100
        self.max_episode_steps = 100
        self.id = "VEC"

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Resets the environment and returns the start state"""
        self.grid = self.create_grid()
        self.place_goal()
        self.place_agent()
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

        if self.user_at_goal_location():
            self.reward = self.reward_for_achieving_goal
            self.done = True
        else:
            self.reward = self.step_reward_for_not_achieving_goal
            if self.step_count >= self.max_episode_steps: self.done = True
            else: self.done = False
        self.achieved_goal = self.next_state[:self.state_only_dimension]
        self.state = self.next_state
        self.s = np.array(self.next_state[:self.state_only_dimension])

        return self.s, self.reward, self.done, {}

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Computes the reward we would have got with this achieved goal and desired goal. Must be of this exact
        interface to fit with the open AI gym specifications"""
        if (achieved_goal == desired_goal).all():
            reward = self.reward_for_achieving_goal
        else:
            reward = self.step_reward_for_not_achieving_goal
        return reward