"""Interests recommender: rank POIs by how well they match the tourist."""

from .base import Recommender, affordable_mask, preference_fit


class InterestsRecommender(Recommender):
    name = "interests"

    def recommend(self, profile, pois):
        candidates = pois[affordable_mask(profile, pois)].copy()
        candidates["fit"] = preference_fit(profile, candidates)
        ranked = candidates.sort_values(["fit", "popularity"], ascending=False)
        return ranked["poi_id"].tolist()
