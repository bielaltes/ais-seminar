"""Interests recommender: rank POIs by how well they match the tourist."""

from .base import Recommender

OUTDOOR = {"park_nature", "beach", "viewpoint"}


class InterestsRecommender(Recommender):
    name = "interests"

    def recommend(self, profile, pois):
        fee = pois["entrance_fee_eur"].fillna(0.0).astype(float)
        candidates = pois[fee <= profile.budget].copy()
        if candidates.empty:
            candidates = pois.copy()
        interest = candidates["category"].map(lambda c: profile.interests.get(c, 0.0))
        outdoor = candidates["category"].isin(OUTDOOR).astype(float)
        outdoor_match = profile.outdoor_preference * outdoor + (1 - profile.outdoor_preference) * (1 - outdoor)
        candidates["fit"] = 0.8 * interest + 0.2 * outdoor_match
        ranked = candidates.sort_values(["fit", "popularity"], ascending=False)
        return ranked["poi_id"].tolist()
