from environments.Blockchain_Environment import Consensus_Environment
import numpy as np


for num_cons_nodes in range(5,41,5):
    environment = Consensus_Environment(num_cons_nodes=num_cons_nodes)
    for i in action_type:
        results = []
        rollings = []
        if i=="greedy":
            num_episode = 5000
            environment.count_file = "../blockchain/consensus_greedy.txt"
        elif i=="random":
            num_episode = 5000
            environment.count_file = "../blockchain/consensus_random.txt"
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