"""Random recommender: a uniform baseline used as a lower bound."""

import numpy as np

from .base import Recommender


class RandomRecommender(Recommender):
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def recommend(self, profile, pois):
        ids = pois["poi_id"].to_numpy()
        self.rng.shuffle(ids)
        return ids.tolist()
