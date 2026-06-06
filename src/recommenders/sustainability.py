"""Sustainability-aware recommender based on ELECTRE-III.

The recommender balances the tourist's preferences with five sustainability
criteria. Following the multi-criteria seminar, the alternatives are ranked with
the ELECTRE-III outranking method; a weighted average is kept as a cross-check.
"""

import numpy as np
from pyDecision.algorithm import electre_iii

from ..constants import CRITERIA
from .base import Recommender, preference_fit

CANDIDATE_K = 30
Q = np.full(len(CRITERIA), 0.05)   # indifference thresholds
P = np.full(len(CRITERIA), 0.15)   # preference thresholds
V = np.full(len(CRITERIA), 0.40)   # veto thresholds


def criteria_weights(sustainability_sensitivity):
    """Split the weight between the preference and the five sustainability criteria."""
    s = sustainability_sensitivity
    w_pref = 1.0 - 0.6 * s
    w_each = (0.6 * s) / 5.0
    return np.array([w_pref] + [w_each] * 5)


def _matrix(profile, pois):
    fit = preference_fit(profile, pois).to_numpy()
    rest = pois[CRITERIA[1:]].to_numpy(dtype=float)
    return np.column_stack([fit, rest])


def _parse_rank(rank_blocks, n):
    """Turn ELECTRE's descending distillation into a rank position per alternative."""
    position = np.full(n, n, dtype=float)
    for place, block in enumerate(rank_blocks):
        for token in block.split(";"):
            idx = int(token.strip()[1:]) - 1
            position[idx] = place
    return position


class SustainabilityRecommender(Recommender):
    name = "sustainability"

    def recommend(self, profile, pois):
        weights = criteria_weights(profile.sustainability_sensitivity)
        matrix = _matrix(profile, pois)
        score = matrix @ weights

        order = np.argsort(-score)
        head = order[:CANDIDATE_K]
        tail = order[CANDIDATE_K:]

        sub = matrix[head]
        blocks = electre_iii(sub, P=P, Q=Q, V=V, W=weights, graph=False)[2]
        position = _parse_rank(blocks, len(head))
        head_sorted = head[np.lexsort((-score[head], position))]

        ranked = list(head_sorted) + list(tail)
        ids = pois["poi_id"].to_numpy()
        return [ids[i] for i in ranked]

    def weighted_average_ranking(self, profile, pois):
        """The simpler weighted-average ranking, used as a cross-check."""
        weights = criteria_weights(profile.sustainability_sensitivity)
        score = _matrix(profile, pois) @ weights
        order = np.argsort(-score)
        ids = pois["poi_id"].to_numpy()
        return [ids[i] for i in order]

    def explain(self, profile, pois, poi_id):
        """Per-criterion contribution of one POI to its weighted score."""
        weights = criteria_weights(profile.sustainability_sensitivity)
        row = pois.index[pois["poi_id"] == poi_id][0]
        values = _matrix(profile, pois.loc[[row]])[0]
        return {c: float(w * v) for c, w, v in zip(CRITERIA, weights, values)}
