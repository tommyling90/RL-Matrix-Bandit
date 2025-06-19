import numpy as np

from agentSpace import AgentSpace
from learningAlgo import LearningAlgo
from agent import Agent
from environment import Environnement
from utils import normalizeMatrix

class Execute:
    def __init__(self, n_instance, T, n_agents, const, title, n_actions):
        self.n_instance = n_instance
        self.T = T
        self.n_agents = n_agents
        self.const = const
        self.title = title
        self.n_actions = n_actions

    def runOnePDExperiment(self, matrices, algo, noise_dist, noise_params):
        env = Environnement(matrices, noise_dist, noise_params)
        for agent in range(0, self.n_agents):
            a_space = AgentSpace(self.n_actions)
            learning_algo = LearningAlgo(self.const[agent], algo[agent], a_space, noise_params[1])
            env.ajouter_agents(Agent(a_space, learning_algo))
        
        plays = []
        exploration_list = []
        for time_step in range(0, self.T):
            actions, explorations = env.step()
            plays.append(actions)
            exploration_list.append(explorations)

        actions_played, explorations_done = [], []
        for j in range(0, self.n_agents):
            actions_played.append([i[j] for i in plays])
            explorations_done.append([i[j] for i in exploration_list])
        regrets = [env.agents[k].regret for k in range(self.n_agents)]
        rewards = [env.agents[k].reward for k in range(self.n_agents)]

        return actions_played, regrets, rewards, explorations_done

    def getPDResult(self, matrices, algo, noise_dist='normal', noise_params=(0, 0.05)):
        matrices_norm = [normalizeMatrix(mat,0) for mat in matrices]
        all_rewards = []
        all_regrets = []
        all_plays = []
        all_explorations = []

        for realisation in range(0, self.n_instance):
            plays, regrets, rewards, explorations = self.runOnePDExperiment(matrices_norm, algo, noise_dist, noise_params)
            all_plays.append(np.array(plays))
            all_rewards.append(np.array(rewards))
            all_regrets.append(np.array(regrets))
            all_explorations.append(np.array(explorations))

        return np.array(all_plays), np.array(all_regrets), np.array(all_rewards), np.array(all_explorations)
