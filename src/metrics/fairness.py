"""Fairness and distribution metrics: how evenly visits and exposure are spread."""

import numpy as np

from ..simulation.crowding import overcrowding_count

TOP_SHARE_FRACTION = 0.10  # the most popular tenth of the catalogue


def gini(values):
    """Gini coefficient of a list of non-negative values (0 = perfectly even)."""
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * (index * x).sum() - (n + 1) * x.sum()) / (n * x.sum()))


def fairness_metrics(result, pois, k=10):
    visits = result["visits"]
    capacities = {r.poi_id: float(r.capacity) for r in pois.itertuples()}

    exposure = {pid: 0 for pid in pois["poi_id"]}
    for t in result["tourists"]:
        for pid in t["recommended"][:k]:
            exposure[pid] += 1

    n_top = max(1, int(len(pois) * TOP_SHARE_FRACTION))
    top_popular = set(pois.sort_values("popularity", ascending=False)["poi_id"].head(n_top))
    total_visits = sum(visits.values()) or 1
    top_share = sum(visits[pid] for pid in top_popular) / total_visits

    return {
        "poi_visit_gini": gini(list(visits.values())),
        "district_visit_gini": gini(list(result["district_visits"].values())),
        "exposure_gini": gini(list(exposure.values())),
        "overcrowded_pois": overcrowding_count(visits, capacities),
        "top_popular_visit_share": float(top_share),
    }
