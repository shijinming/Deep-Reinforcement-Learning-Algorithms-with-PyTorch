from agents.DQN_agents.DDQN import DDQN
from agents.actor_critic_agents.DDPG import DDPG
from agents.actor_critic_agents.SAC_Discrete import SAC_Discrete
from agents.actor_critic_agents.A3C import A3C 
from agents.DQN_agents.DDQN import DDQN
from environments.VEC_Environment import VEC_Environment
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config

config = Config()
config.seed = 1
    
num_vehicles = 20
task_num = 50
# embedding_dimensions = [[num_possible_states, 20]]
# print("Num possible states ", num_possible_states)
embedding_dimensions = [[num_vehicles*3+3, 50]]
config.environment = VEC_Environment(num_vehicles=num_vehicles, task_num=task_num)

config.num_episodes_to_run = 5000
config.file_to_save_data_results = "results/data_and_graphs/VEC.pkl"
config.file_to_save_results_graph = "results/data_and_graphs/VEC.png"
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 1
config.use_GPU = True
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False
config.device = "cuda:0"

config.hyperparameters = {
    "DQN_Agents": {
        "learning_rate": 0.0001,
        "batch_size": 256,
        "buffer_size": 100000,
        "epsilon_decay_rate_denominator": 150,
        "discount_rate": 0.999,
        "incremental_td_error": 1e-8,
        "update_every_n_steps": 1,
        "linear_hidden_units": [32, 32],
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
            "learning_rate": 0.0002,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": "Softmax",
            "batch_norm": False,
            "tau": 0.01,
            "gradient_clipping_norm": 5
        },

        "Critic": {
            "learning_rate": 0.0005,
            "linear_hidden_units": [64, 64],
            "final_layer_activation": None,
            "batch_norm": False,
            "buffer_size": 100000,
            "tau": 0.01,
            "gradient_clipping_norm": 5
        },

        "min_steps_before_learning": 0,
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
        "do_evaluation_iterations": False,
        "clip_rewards":False 

    }
}

AGENTS = [SAC_Discrete] #DIAYN] # A3C] #SNN_HRL] #, DDQN]
trainer = Trainer(config, AGENTS)
trainer.run_games_for_agents()