from environments.Blockchain_Environment import Blockchain_Environment
import numpy as np


action_type = ["random","greedy"]
task_num = 32
task_file = "../blockchain/tasks.txt"

for iter in range(200):
    print("iter =",iter)
    for num_vehicles in range(5,41,5):
        environment = Blockchain_Environment(num_vehicles=num_vehicles, task_num=task_num)
        environment.load_offloading_tasks(task_file, iter%5+1)
        for i in action_type:
            print(i)
            results = []
            rollings = []
            if i=="greedy":
                num_episode = 5
                environment.count_file = "../blockchain/greedy.txt"
            elif i=="random":
                num_episode = 1000
                environment.count_file = "../blockchain/random.txt"
            with open(environment.count_file,'a') as f:
                f.write('num_vehicles='+str(num_vehicles)+'\n')
            for _ in range(num_episode):
                environment.reset()
                reward = 0
                for _ in range(task_num):
                    _,r,_,_=environment.step(environment.produce_action(i))
                    reward+=r
                results.append(reward)
            print("mean_reward=", np.mean(results),"max_reward=",max(results))