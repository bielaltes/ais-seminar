"""Context-aware post-filtering.

This follows the post-filtering idea from the context-aware seminar: a base
recommender produces a ranking, and the ranking is then adjusted to the current
context. Here the context is the weather. On a rainy day the outdoor places are
pushed to the back of the list, so the tourist is steered toward indoor options.
"""

from .base import OUTDOOR, Recommender


class ContextAwareRecommender(Recommender):
    name = "context"

    def __init__(self, base):
        self.base = base

    def recommend(self, profile, pois):
        ranked = self.base.recommend(profile, pois)
        if profile.weather != "rainy":
            return ranked
        category = dict(zip(pois["poi_id"], pois["category"]))
        indoor = [pid for pid in ranked if category[pid] not in OUTDOOR]
        outdoor = [pid for pid in ranked if category[pid] in OUTDOOR]
        return indoor + outdoor
