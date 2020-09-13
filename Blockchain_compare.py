from environments.Blockchain_Environment import Consensus_Environment
import numpy as np


action_type = ["random","greedy"]
with open("../blockchain/consensus_greedy.txt",'w+') as f:
    f.write("")
with open("../blockchain/consensus_random.txt",'w+') as f:
    f.write("")
for num_cons_nodes in range(5,41,5):
    print("num_cons_nodes=",num_cons_nodes)
    environment = Consensus_Environment(num_cons_nodes=num_cons_nodes)
    for i in action_type:
        results = []
        if i=="greedy":
            num_episode = 5000
            environment.count_file = "../blockchain/consensus_greedy.txt"
        elif i=="random":
            num_episode = 5000
            environment.count_file = "../blockchain/consensus_random.txt"
        with open(environment.count_file,'a') as f:
            f.write('num_cons_nodes='+str(num_cons_nodes)+'\n')
        for _ in range(num_episode):
            environment.reset()
            reward = 0
            selection, batch_size = environment.produce_action(i)
            reward, delay = environment.produce_utility(selection, batch_size)
            results.append(reward)
            with open(environment.count_file,'a') as f:
                f.write(str(reward)+' '+str(delay)+' '+str(batch_size)+'\n')
        print("mean_reward=", np.mean(results),"max_reward=",max(results))