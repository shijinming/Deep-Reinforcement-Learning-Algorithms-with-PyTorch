import random
import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
from environments.VEC_Environment import VEC_Environment


num_vehicles = 20
task_num = 50
num_episode = 2000
trials = 100
action_type = ["random","greedy"]

for i in action_type:
    plt.figure()
    plt.title("VEC_{}".format(i))
    for price_level in range(1,100,10):
        results = []
        rollings = []
        env = VEC_Environment(num_vehicles=num_vehicles, task_num=task_num)
        for _ in range(num_episode):
            env.reset()
            reward = 0
            for _ in range(task_num):
                _,r,_,_=env.step(env.produce_action(i),price_level)
                reward+=r
            results.append(reward)
            rollings.append(np.mean(results[-trials:]))
        plt.plot(rollings,label=str(price_level))
    plt.legend()
    plt.savefig("results/data_and_graphs/VEC_{}.png".format(i))