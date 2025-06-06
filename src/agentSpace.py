import numpy as np
class AgentSpace:
    def __init__(self, n_arms):
        self.target_plays = np.zeros(n_arms, dtype=int)
        self.plays = np.zeros(n_arms, dtype=int)
        self.avg_reward = np.zeros(n_arms)
        self.sums = np.zeros(n_arms)
        self.t = 0
        self.n_arms = n_arms
