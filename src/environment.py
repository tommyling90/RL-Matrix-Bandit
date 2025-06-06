import numpy as np

class Environnement:
    def __init__(self, matrices, noise_dist, noise_params):
        self.agents = []
        self.matrices = matrices
        self.noise_dist = noise_dist
        self.noise_params = noise_params

    def ajouter_agents(self, agent):
        self.agents.append(agent)

    def sample_noise(self):
        if self.noise_dist == 'normal':
            mean, var = self.noise_params
            std = np.sqrt(var)
            return np.random.normal(mean, std)
        else:
            raise ValueError(f"Unknown noise distribution: {self.noise_dist}")

    def updateStep(self, a1, a2):
        r1 = self.matrices[0][a1, a2] + self.sample_noise()
        r2 = self.matrices[1][a1, a2] + self.sample_noise()

        min_matrix = np.minimum(self.matrices[0], self.matrices[1])
        max_val = np.max(min_matrix)
        regret_matrix = max_val - min_matrix
        regret = regret_matrix[a1, a2]

        self.agents[0].update(a1, r1, regret)
        self.agents[1].update(a2, r2, regret)

    def step(self):
        action1, exploration1 = self.agents[0].train()
        action2, exploration2 = self.agents[1].train()
        self.updateStep(action1, action2)
        return [action1, action2], [exploration1, exploration2]