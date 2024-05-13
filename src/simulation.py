from typing import List

import numpy as np


class PBMSimulator:
    def __init__(self, position_bias: float):
        self.position_bias = position_bias

    def __call__(self, relevance: List[float]):
        n_actions = len(relevance)
        examination = self.get_position_bias(n_actions)
        p = examination * relevance
        return np.random.binomial(n=1, p=p, size=(n_actions,))

    def get_position_bias(self, n_actions):
        return (1 / np.arange(1, n_actions + 1)) ** self.position_bias


class CascadeSimulator:

    def __call__(self, relevance: List[float]):
        clicks = np.zeros_like(relevance)

        for i, r in enumerate(relevance):
            if np.random.binomial(n=1, p=r) == 1:
                clicks[i] = 1.0
                break

        return clicks


def run_simulation(bandit, simulator, n_rounds):
    results = []

    for i in range(n_rounds):
        actions = bandit.get_actions(top_k)
        clicks = simulator(actions)
        bandit.update(actions=actions, reward=clicks)

    predicted_relevance = bandit.rewards / bandit.pulls

    result_df = pd.DataFrame(
        {
            "action": np.arange(n_actions),
            "relevance": relevance,
            "predicted_relevance": predicted_relevance,
        }
    )

