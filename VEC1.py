from agents.DQN_agents.DDQN import DDQN
from agents.actor_critic_agents.DDPG import DDPG
from agents.actor_critic_agents.SAC import SAC
from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
from agents.actor_critic_agents.A3C import A3C 
from agents.DQN_agents.DDQN import DDQN
from agents.DQN_agents.Dueling_DDQN import Dueling_DDQN
from environments.VEC_Environment1 import VEC_Environment1
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config
import matplotlib.pyplot as plt
import numpy as np

config = Config()
config.seed = 1
    
config.num_episodes_to_run = 20000
config.file_to_save_data_results = "results/data_and_graphs/VEC.pkl"
config.file_to_save_results_graph = "results/data_and_graphs/VEC.png"
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = False
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = True
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False
config.device = "cuda:0"

config.hyperparameters = {
    "DQN_Agents": {
        "learning_rate": 0.00002,
        "batch_size": 256,
        "buffer_size": 100000,
        "epsilon_decay_rate_denominator": 150,
        "discount_rate": 0.99,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 1,
        "linear_hidden_units": [512, 256, 256, 128],
        "final_layer_activation": None,
        "batch_norm": False,
        "gradient_clipping_norm": 5,
        "HER_sample_proportion": 0.8,
        "learning_iterations": 1,
        "clip_rewards": False,
        "tau":0.01
    },
        "Actor_Critic_Agents": {  # hyperparameters taken from https://arxiv.org/pdf/1802.09477.pdf
        "Actor": {
            "learning_rate": 0.0001,
            "linear_hidden_units": [1200, 800],
            "final_layer_activation": None,
            "batch_norm": False,
            "tau": 0.005,
            "gradient_clipping_norm": 5
        },

        "Critic": {
            "learning_rate": 0.0002,
            "linear_hidden_units": [800, 500],
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 100000,
            "tau": 0.005,
            "gradient_clipping_norm": 5
        },

        "min_steps_before_learning": 5000,
        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0, #for O-H noise
        "theta": 0.15, #for O-H noise
        "sigma": 0.25, #for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 1,
        "learning_updates_per_learning_session": 1,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": False,
        "do_evaluation_iterations": True,
        "clip_rewards":False 
    }
}

count_file = "../sac.txt"
with open(count_file,'w+') as f:
    f.write("")
num_episode = 10
trials = 100
action_type = ["random","greedy"]
task_num = 30
task_file = "../tasks.txt"
# config.environment = VEC_Environment1(num_vehicles=50, task_num=task_num)
# config.environment.generate_offload_tasks(task_file, task_num, 10)
for group in range(1,2):
    print("group =",group)
    for num_vehicles in [30]:
        config.environment = VEC_Environment1(num_vehicles=num_vehicles, task_num=task_num)
        config.environment.load_offloading_tasks(task_file, group)
        config.environment.count_file = count_file
        for i in action_type:
            print(i)
            with open(count_file,'a') as f:
                f.write(i+'\n')
            results = []
            rollings = []
            if i=="greedy":
                num_episode = 5
            elif i=="random":
                num_episode = 1000
            for _ in range(num_episode):
                config.environment.reset()
                reward = 0
                for _ in range(task_num):
                    _,r,_,_=config.environment.step(config.environment.produce_action(i))
                    reward+=r
                results.append(reward)
                rollings.append(np.mean(results[-trials:]))
            print("mean_reward=", np.mean(results),"max_reward=",max(results))
        with open(count_file,'a') as f:
            f.write('SAC\n')
        AGENTS = [SAC] 
        trainer = Trainer(config, AGENTS)
        trainer.run_games_for_agents()