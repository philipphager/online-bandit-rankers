import numpy as np
from typing import List


class PBMSimulator:
    def __init__(self, position_bias: float):
        self.position_bias = position_bias

    def __call__(self, relevance: List[float]):
        n_actions = len(relevance)
        examination = self.get_position_bias(n_actions)
        p = examination * relevance

        clicks = np.random.binomial(n=1, p=p, size=(n_actions,))
        impressions = np.zeros_like(clicks)
        clicked_items = np.nonzero(clicks)[0]

        if len(clicked_items) > 0:
            last_clicked_item = int(clicked_items[-1])
            impressions[:last_clicked_item] = 1

        return clicks, impressions

    def get_position_bias(self, n_actions):
        return (1 / np.arange(1, n_actions + 1)) ** self.position_bias


class CascadeSimulator:

    def __call__(self, relevance: List[float]):
        clicks = np.zeros_like(relevance)
        impressions = np.zeros_like(relevance)

        for i, r in enumerate(relevance):
            impressions[i] = 1.0

            if np.random.binomial(n=1, p=r) == 1:
                clicks[i] = 1.0
                break

        return clicks, impressions


class GeometricSimulator:
    def __init__(self, position_bias: float):
        self.position_bias = position_bias

    def __call__(self, relevance: List[float]):
        n_actions = len(relevance)

        examined_ranks = self.sampled_examined_ranks(n_actions)
        p = relevance.copy()
        p[examined_ranks:] = 0

        clicks = np.random.binomial(n=1, p=p, size=(n_actions,))
        impressions = np.zeros_like(clicks)
        impressions[:examined_ranks] = 1

        return clicks, impressions

    def sampled_examined_ranks(self, n_actions: int):
        """
        Draws a rank k ~ Geometric(p). All items <= k have been examined by the user.
        """
        p = self.get_position_bias(n_actions)
        return np.random.choice(np.arange(1, n_actions + 1), p=p)

    def get_position_bias(self, n_actions):
        ranks = np.arange(1, n_actions + 1)
        probabilities = (1 - self.position_bias) ** (ranks - 1) * self.position_bias
        probabilities /= probabilities.sum()

        return probabilities
