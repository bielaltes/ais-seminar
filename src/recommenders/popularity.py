"""Popularity recommender: rank POIs by popularity and ignore the profile."""

from .base import Recommender


class PopularityRecommender(Recommender):
    name = "popularity"

    def recommend(self, profile, pois):
        ranked = pois.sort_values("popularity", ascending=False)
        return ranked["poi_id"].tolist()
