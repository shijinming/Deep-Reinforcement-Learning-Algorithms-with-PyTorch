from agents.DQN_agents.DDQN import DDQN
from agents.actor_critic_agents.DDPG import DDPG
from agents.actor_critic_agents.SAC import SAC
from environments.VEC_Environment import VEC_Environment
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config

config = Config()
config.seed = 1
    
# embedding_dimensions = [[num_possible_states, 20]]
# print("Num possible states ", num_possible_states)

config.environment = VEC_Environment(num_vehicles=80, task_num=50)

config.num_episodes_to_run = 1000
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
config.save_model = True

# config.hyperparameters = {
#     "DQN_Agents": {
#         "linear_hidden_units": [30, 10],
#         "learning_rate": 0.01,
#         "buffer_size": 40000,
#         "batch_size": 256,
#         "final_layer_activation": "None",
#         "columns_of_data_to_be_embedded": [0],
#         # "embedding_dimensions": embedding_dimensions,
#         "batch_norm": False,
#         "gradient_clipping_norm": 5,
#         "update_every_n_steps": 1,
#         "epsilon_decay_rate_denominator": 10,
#         "discount_rate": 0.99,
#         "learning_iterations": 1,
#         "tau": 0.01,
#         "exploration_cycle_episodes_length": None,
#         "clip_rewards": False
#     },

#     "SNN_HRL": {
#         "SKILL_AGENT": {
#             "num_skills": 20,
#             "regularisation_weight": 1.5,
#             "visitations_decay": 0.9999,
#             "episodes_for_pretraining": 300,
#             "batch_size": 256,
#             "learning_rate": 0.001,
#             "buffer_size": 40000,
#             "linear_hidden_units": [20, 10],
#             "final_layer_activation": "None",
#             "columns_of_data_to_be_embedded": [0, 1],
#             # "embedding_dimensions": [embedding_dimensions[0], [20, 6]],
#             "batch_norm": False,
#             "gradient_clipping_norm": 2,
#             "update_every_n_steps": 1,
#             "epsilon_decay_rate_denominator": 500,
#             "discount_rate": 0.999,
#             "learning_iterations": 1,
#             "tau": 0.01,
#             "clip_rewards": False
#         },

#         "MANAGER": {
#             "timesteps_before_changing_skill": 6,
#             "linear_hidden_units": [10, 5],
#             "learning_rate": 0.01,
#             "buffer_size": 40000,
#             "batch_size": 256,
#             "final_layer_activation": "None",
#             "columns_of_data_to_be_embedded": [0],
#             # "embedding_dimensions": embedding_dimensions,
#             "batch_norm": False,
#             "gradient_clipping_norm": 5,
#             "update_every_n_steps": 1,
#             "epsilon_decay_rate_denominator": 50,
#             "discount_rate": 0.99,
#             "learning_iterations": 1,
#             "tau": 0.01,
#             "clip_rewards": False

#         }

#     },

#     "Actor_Critic_Agents": {

#         "learning_rate": 0.005,
#         "linear_hidden_units": [20, 10],

#         "columns_of_data_to_be_embedded": [0],
#         # "embedding_dimensions": embedding_dimensions,
#         "final_layer_activation": ["SOFTMAX", None],
#         "gradient_clipping_norm": 5.0,
#         "discount_rate": 0.99,
#         "epsilon_decay_rate_denominator": 50.0,
#         "normalise_rewards": True,
#         "clip_rewards": False

#     },


#     "DIAYN": {

#         "num_skills": 5,
#         "DISCRIMINATOR": {
#             "learning_rate": 0.01,
#             "linear_hidden_units": [20, 10],
#             "columns_of_data_to_be_embedded": [0],
#             # "embedding_dimensions": embedding_dimensions,
#         },

#         "AGENT": {
#             "learning_rate": 0.01,
#             "linear_hidden_units": [20, 10],
#         }
#     },


#     "HRL": {
#         "linear_hidden_units": [10, 5],
#         "learning_rate": 0.01,
#         "buffer_size": 40000,
#         "batch_size": 256,
#         "final_layer_activation": "None",
#         "columns_of_data_to_be_embedded": [0],
#         # "embedding_dimensions": embedding_dimensions,
#         "batch_norm": False,
#         "gradient_clipping_norm": 5,
#         "update_every_n_steps": 1,
#         "epsilon_decay_rate_denominator": 400,
#         "discount_rate": 0.99,
#         "learning_iterations": 1,
#         "tau": 0.01

#     }
# }
config.hyperparameters = {
    "Policy_Gradient_Agents": {
            "learning_rate": 0.05,
            "linear_hidden_units": [30, 15],
            "final_layer_activation": "TANH",
            "learning_iterations_per_round": 10,
            "discount_rate": 0.9,
            "batch_norm": False,
            "clip_epsilon": 0.2,
            "episodes_per_learning_round": 10,
            "normalise_rewards": True,
            "gradient_clipping_norm": 5,
            "mu": 0.0,
            "theta": 0.15,
            "sigma": 0.2,
            "epsilon_decay_rate_denominator": 1,
            "clip_rewards": False
        },

    "Actor_Critic_Agents": {
            "Actor": {
                "learning_rate": 0.003,
                "linear_hidden_units": [20, 20],
                "final_layer_activation": None,
                "batch_norm": False,
                "tau": 0.005,
                "gradient_clipping_norm": 5,
                "initialiser": "Xavier"
            },

            "Critic": {
                "learning_rate": 0.02,
                "linear_hidden_units": [20, 20],
                "final_layer_activation": None,
                "batch_norm": False,
                "buffer_size": 1000000,
                "tau": 0.005,
                "gradient_clipping_norm": 5,
                "initialiser": "Xavier"
            },

        "min_steps_before_learning": 1000, #for SAC only
        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0,  # for O-H noise
        "theta": 0.15,  # for O-H noise
        "sigma": 0.25,  # for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 20,
        "learning_updates_per_learning_session": 10,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": True,
        "do_evaluation_iterations": True,
        "clip_rewards": False

    }

}

AGENTS = [DDPG] #DIAYN] # A3C] #SNN_HRL] #, DDQN]
trainer = Trainer(config, AGENTS)
trainer.run_games_for_agents()