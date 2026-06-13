"""Accuracy metrics: precision, recall, F1 and NDCG at k.

A POI is relevant to a tourist when it belongs to a category the tourist clearly
likes, following the category-based relevance used in the points-of-interest seminar.
"""

import numpy as np

REL_THRESHOLD = 0.5


def _relevant_ids(profile, poi_category):
    liked = {c for c, w in profile.interests.items() if w >= REL_THRESHOLD}
    return {pid for pid, cat in poi_category.items() if cat in liked}


def _ndcg(top_k, relevant):
    dcg = sum(1.0 / np.log2(i + 2) for i, pid in enumerate(top_k) if pid in relevant)
    ideal = min(len(relevant), len(top_k))
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def accuracy_metrics(result, pois, k=10):
    poi_category = dict(zip(pois["poi_id"], pois["category"]))
    precision, recall, ndcg = [], [], []
    for t in result["tourists"]:
        relevant = _relevant_ids(t["profile"], poi_category)
        if not relevant:
            continue
        top_k = t["recommended"][:k]
        hits = sum(1 for pid in top_k if pid in relevant)
        precision.append(hits / k)
        recall.append(hits / len(relevant))
        ndcg.append(_ndcg(top_k, relevant))
    p, r = float(np.mean(precision)), float(np.mean(recall))
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1, "ndcg": float(np.mean(ndcg))}
