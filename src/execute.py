import numpy as np

from agentSpace import AgentSpace
from learningAlgo import LearningAlgo
from agent import Agent
from environment import Environnement
from utils import normalizeMatrix
from scipy import stats

class Execute:
    def __init__(self, n_instance, T, n_agents, const, title):
        self.n_instance = n_instance
        self.T = T
        self.n_agents = n_agents
        self.const = const
        self.title = title

    def runOnePDExperiment(self, matrices, algo, noise_dist, noise_params):
        env = Environnement(matrices, noise_dist, noise_params)
        for agent in range(0, self.n_agents):
            a_space = AgentSpace(len(matrices[0][0]))
            learning_algo = LearningAlgo(self.const[agent], algo[agent], a_space, noise_params[1])
            env.ajouter_agents(Agent(a_space, learning_algo))
        
        plays = []
        exploration_list = []
        for time_step in range(0, self.T):
            actions, explorations = env.step()
            plays.append(actions)
            exploration_list.append(explorations)

        actions_played = [[i[0] for i in plays],[i[1] for i in plays]]
        regrets = [env.agents[k].regret for k in range(self.n_agents)]
        rewards = [env.agents[k].reward for k in range(self.n_agents)]
        explorations_done = [[i[0] for i in exploration_list], [i[1] for i in exploration_list]]
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
            all_rewards.append(np.array(rewards).T)
            all_regrets.append(np.array(regrets).T)
            all_explorations.append(np.array(explorations))

        plays_arr = np.stack(all_plays, axis=2)
        rewards_arr  = np.stack(all_rewards, axis=2)
        regrets_arr = np.stack(all_regrets, axis=2)
        cum_regrets_arr = regrets_arr.cumsum(axis=0)
        explorations_arr = np.stack(all_explorations, axis=0).T
        explorations_conjointe_arr = np.min(explorations_arr, axis=1)

        mean_r     = rewards_arr.mean(axis=2)
        std_r      = rewards_arr.std (axis=2)
        mean_reg    = cum_regrets_arr.mean(axis=2)
        std_reg     = cum_regrets_arr.std (axis=2)
        mean_exploration = explorations_arr.mean(axis=2)
        mean_exploration_conjointe = explorations_conjointe_arr.mean(axis=1)

        results = {
        'experiment':   self.title,
        'algo':         algo,
        'noise_dist':   noise_dist,
        'noise_params': noise_params,
        'metrics': {}
    }

        for name, arr in [
            ('mean_reward',     mean_r),
            ('std_reward',      std_r),
            ('mean_cum_regret', mean_reg),
            ('std_cum_regret',  std_reg),
            ('mean_exploration', mean_exploration),
        ]:
            results['metrics'][name] = {
                f'agent_{i}': arr[:, i]
                for i in range(self.n_agents)
            }
        results['metrics']['mean_exploration_conjointe'] = mean_exploration_conjointe

        props = {}
        for a in range(len(matrices[0][0])):
            prop = np.mean(plays_arr == a, axis=2)
            props[f'action_{a}'] = {
                f'agent_{i}': prop[:, i]
                for i in range(self.n_agents)
            }
        results['metrics']['prop_action'] = props

        paire_action = np.ones((plays_arr.shape[1], plays_arr.shape[2]))
        for i in range(plays_arr.shape[1]):
            for j in range(plays_arr.shape[2]):
                jj = 0
                for a1 in range(len(matrices[0][0])):
                    for a2 in range(len(matrices[0][0])):
                        jj += 1
                        if plays_arr[0, i, j] == a1 and plays_arr[1, i, j] == a2:
                            paire_action[i, j] = jj

        counts = paire_action
        types = np.unique(counts)
        n_types = types.size
        vecteur_de_compte = np.zeros((counts.shape[0], n_types), dtype=int)
        for j in range(counts.shape[0]):
            for idx, t in enumerate(types):
                vecteur_de_compte[j, idx] = np.count_nonzero(counts[j] == t)
        results["metrics"]["vecteur_de_props"] = vecteur_de_compte/self.n_instance

        mode_result = stats.mode(counts[-50:, :], axis=0, keepdims=True)
        mode_calcul = mode_result.mode[0]
        vecteur_de_compte_mode = np.zeros(len(types), dtype=int)
        for i in range(len(types)+1):
            for j in mode_calcul:
                if j == i:
                    vecteur_de_compte_mode[i-1] += 1

        results["metrics"]["vecteur_de_proportion_mode"] = vecteur_de_compte_mode/self.n_instance
        print(vecteur_de_compte_mode)
        print(len(mode_calcul))

        return results
