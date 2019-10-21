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
    
# embedding_dimensions = [[num_possible_states, 20]]
# print("Num possible states ", num_possible_states)

config.environment = VEC_Environment(num_vehicles=30, task_num=20)

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
config.overwrite_action_size = 100 + 70



AGENTS = [A3C] #DIAYN] # A3C] #SNN_HRL] #, DDQN]
trainer = Trainer(config, AGENTS)
trainer.run_games_for_agents()