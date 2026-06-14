"""Tourist segmentation with k-means.

This follows the profiling idea from the tourism seminar, where tourists are
grouped into segments before their behaviour is studied. We cluster the
population on its interests and a few attitude features, then label each segment
by the category its members care about most.
"""

import numpy as np
from sklearn.cluster import KMeans

from .constants import CATEGORIES


def _features(population):
    rows = []
    for p in population:
        interests = [p.interests[c] for c in CATEGORIES]
        rows.append(interests + [p.crowd_aversion, p.sustainability_sensitivity, p.outdoor_preference])
    return np.array(rows)


def cluster_population(population, k=5, seed=0):
    """Return a cluster label for every tourist in the population."""
    features = _features(population)
    return KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(features)


def segment_labels(population, labels):
    """Name each cluster by the category its members like most, on average."""
    names = {}
    for cluster in sorted(set(labels)):
        members = [p for p, c in zip(population, labels) if c == cluster]
        mean_interest = {cat: np.mean([m.interests[cat] for m in members]) for cat in CATEGORIES}
        names[cluster] = max(mean_interest, key=mean_interest.get)
    return names
