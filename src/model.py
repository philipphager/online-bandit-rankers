import numpy as np
from typing import Optional, List


class RankingUCBBandit:
    def __init__(self, actions: int):
        self.actions = actions
        self.pulls = np.ones(actions)
        self.rewards = np.ones(actions)

    def get_actions(self, top_k: int):
        mean_rewards = self.rewards / self.pulls.clip(min=1)
        confidence_bound = np.sqrt(1.5 * np.log((self.pulls.sum()) / self.pulls))
        return np.argsort(-(mean_rewards + confidence_bound))[:top_k]

    def update(self, actions: np.ndarray, reward: np.ndarray):
        self.pulls[actions] += 1
        self.rewards[actions] += reward

    def get_relevance(self, actions: Optional[np.ndarray] = None):
        actions = np.arange(self.actions) if actions is None else actions
        return self.rewards[actions] / self.pulls[actions]


class PBMRankingUCBBandit:
    def __init__(self, actions: int, examination: np.ndarray, delta: float):
        self.actions = actions
        self.examination = examination
        self.delta = delta
        self.rewards = np.ones(actions)
        self.pulls = np.ones(actions)
        self.total_examination = np.ones(actions)

    def get_actions(self, top_k: int):
        mean_rewards = self.rewards / self.total_examination
        confidence_bound = np.sqrt(self.pulls / self.total_examination) * np.sqrt(
            self.delta / 2 * self.total_examination
        )
        return np.argsort(-(mean_rewards + confidence_bound))[:top_k]

    def update(self, actions: np.ndarray, reward: np.ndarray):
        self.rewards[actions] += reward
        self.pulls[actions] += 1
        self.total_examination[actions] += self.examination

    def get_relevance(self, actions: Optional[np.ndarray] = None):
        actions = np.arange(self.actions) if actions is None else actions
        return self.rewards[actions] / self.total_examination[actions]


class CascadingRankingUCBBandit:
    def __init__(self, actions: int):
        self.actions = actions
        self.rewards = np.ones(actions)
        self.pulls = np.ones(actions)

    def get_actions(self, top_k: int):
        mean_rewards = self.rewards / self.pulls
        confidence_bound = np.sqrt(1.5 * np.log((self.pulls.sum()) / self.pulls))
        return np.argsort(-(mean_rewards + confidence_bound))[:top_k]

    def update(self, actions: np.ndarray, reward: np.ndarray):
        if reward.sum() > 0:
            # All items before and including the clicked items were inspected:
            first_reward_idx = np.argmax(reward)
            rewarded_action = actions[first_reward_idx]
            examined_actions = actions[: (first_reward_idx + 1)]
            self.rewards[rewarded_action] += 1
            self.pulls[examined_actions] += 1
        else:
            # In case of no clicks, assume all displayed items were inspected:
            self.pulls[actions] += 1

    def get_relevance(self, actions: Optional[np.ndarray] = None):
        actions = np.arange(self.actions) if actions is None else actions
        return self.rewards[actions] / self.pulls[actions]
