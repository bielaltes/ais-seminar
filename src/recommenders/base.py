"""Common interface and helpers shared by every recommender."""

import numpy as np

OUTDOOR = {"park_nature", "beach", "viewpoint"}


class Recommender:
    """A recommender turns a tourist profile into a ranked list of POI ids."""

    name = "base"

    def recommend(self, profile, pois):
        """Return a list of poi_id ordered from the best to the worst suggestion."""
        raise NotImplementedError


def preference_fit(profile, pois):
    """Score every POI by how well it matches the tourist's interests."""
    interest = pois["category"].map(lambda c: profile.interests.get(c, 0.0))
    outdoor = pois["category"].isin(OUTDOOR).astype(float)
    outdoor_match = profile.outdoor_preference * outdoor + (1 - profile.outdoor_preference) * (1 - outdoor)
    return (0.8 * interest + 0.2 * outdoor_match).astype(float)


def affordable_mask(profile, pois):
    """A POI is affordable when its entrance fee fits the tourist's budget."""
    fee = pois["entrance_fee_eur"].fillna(0.0).astype(float)
    mask = fee <= profile.budget
    return mask if mask.any() else np.ones(len(pois), dtype=bool)
