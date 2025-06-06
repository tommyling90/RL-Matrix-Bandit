import numpy as np
from agentSpace import AgentSpace

class Agent:
    def __init__(self, a_space: AgentSpace, algo):

        self.learning_algo = algo
        self.a_space = a_space
        self.regret = []
        self.reward = []

    def update(self, action, step_reward, step_regret):
        self.a_space.plays[action] += 1
        self.a_space.sums[action] += step_reward
        self.a_space.avg_reward = np.divide(
            self.a_space.sums,
            self.a_space.plays,
            out=np.zeros_like(self.a_space.sums, dtype=float),
            where=self.a_space.plays != 0
        )
        self.regret.append(step_regret)
        self.reward.append(step_reward)

    def train(self):
        self.a_space.t += 1
        action, exploration = self.learning_algo.getAction()

        return action, exploration
