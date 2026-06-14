"""Beyond-accuracy metrics: intra-list diversity, novelty and catalogue coverage."""

import numpy as np

from ..dataio import haversine_km

MAX_CITY_KM = 12.0  # rough span of Barcelona, used to scale geographic distance


def _dissimilarity(a, b, poi):
    category = 0.0 if poi["category"][a] == poi["category"][b] else 1.0
    distance = haversine_km(*poi["coords"][a], *poi["coords"][b]) / MAX_CITY_KM
    return 0.5 * category + 0.5 * min(distance, 1.0)


def _intra_list_diversity(top_k, poi):
    if len(top_k) < 2:
        return 0.0
    pairs = [(top_k[i], top_k[j]) for i in range(len(top_k)) for j in range(i + 1, len(top_k))]
    return float(np.mean([_dissimilarity(a, b, poi) for a, b in pairs]))


def beyond_accuracy_metrics(result, pois, k=10):
    poi = {
        "category": dict(zip(pois["poi_id"], pois["category"])),
        "coords": {r.poi_id: (float(r.lat), float(r.lon)) for r in pois.itertuples()},
    }
    prob = pois["popularity"].to_numpy() + 1e-6
    prob = prob / prob.sum()
    novelty_of = {pid: -np.log2(p) for pid, p in zip(pois["poi_id"], prob)}

    diversity, novelty, recommended, top_sets = [], [], set(), []
    for t in result["tourists"]:
        top_k = t["recommended"][:k]
        recommended.update(top_k)
        top_sets.append(set(top_k))
        diversity.append(_intra_list_diversity(top_k, poi))
        novelty.append(float(np.mean([novelty_of[pid] for pid in top_k])))
    return {
        "diversity": float(np.mean(diversity)),
        "novelty": float(np.mean(novelty)),
        "coverage": len(recommended) / len(pois),
        "personalisation": _personalisation(top_sets, k),
    }


def _personalisation(top_sets, k, n_pairs=4000, seed=123):
    """Average dissimilarity between the recommendation lists of different tourists."""
    rng = np.random.default_rng(seed)
    n = len(top_sets)
    if n < 2:
        return 0.0
    overlaps = []
    for _ in range(n_pairs):
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i != j:
            overlaps.append(len(top_sets[i] & top_sets[j]) / k)
    return float(1.0 - np.mean(overlaps))
