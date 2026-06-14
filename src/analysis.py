"""Extra analyses that go beyond the main four-way comparison.

This script runs three focused studies grounded in the seminars:
  1. context-aware recommendation (weather post-filtering),
  2. tourist segmentation with k-means,
  3. a comparison of ELECTRE-III against a weighted average, with explanations.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .dataio import load_pois
from .experiment import all_metrics
from .profiles import Profile, generate_population
from .recommenders.base import OUTDOOR, preference_fit
from .recommenders.contextual import ContextAwareRecommender
from .recommenders.sustainability import SustainabilityRecommender, WeightedAverageRecommender
from .segments import cluster_population, segment_labels
from .simulation.model import CityModel

REL_THRESHOLD = 0.5


def _run(pois, population, recommender, seed):
    return CityModel(pois, population, recommender, seed=seed).run()


def _relevant(profile, poi_category):
    liked = {c for c, w in profile.interests.items() if w >= REL_THRESHOLD}
    return {pid for pid, cat in poi_category.items() if cat in liked}


def _precision(tourist, poi_category, k=10):
    relevant = _relevant(tourist["profile"], poi_category)
    if not relevant:
        return None
    top = tourist["recommended"][:k]
    return sum(1 for pid in top if pid in relevant) / k


def context_analysis(pois, res_sustain, res_context):
    """Share of outdoor visits among rainy-weather tourists, with and without context."""
    category = dict(zip(pois["poi_id"], pois["category"]))

    def rainy_outdoor_share(result):
        outdoor, total = 0, 0
        for t in result["tourists"]:
            if t["profile"].weather != "rainy":
                continue
            for pid in t["visited"]:
                total += 1
                outdoor += int(category[pid] in OUTDOOR)
        return outdoor / total if total else 0.0

    return pd.DataFrame([
        {"recommender": "sustainability", "rainy_outdoor_visit_share": rainy_outdoor_share(res_sustain)},
        {"recommender": "sustainability+context", "rainy_outdoor_visit_share": rainy_outdoor_share(res_context)},
    ]).set_index("recommender")


def segment_analysis(pois, population, res_sustain, k_clusters=5, seed=0):
    """Per-segment outcomes under the sustainability recommender."""
    labels = cluster_population(population, k=k_clusters, seed=seed)
    names = segment_labels(population, labels)
    segment_of = {id(p): int(c) for p, c in zip(population, labels)}

    poi_category = dict(zip(pois["poi_id"], pois["category"]))
    low_pressure = dict(zip(pois["poi_id"], pois["low_pressure"]))

    rows = {c: {"segment": names[c], "size": 0, "precision": [], "visited_low_pressure": []} for c in names}
    for t in res_sustain["tourists"]:
        c = segment_of[id(t["profile"])]
        rows[c]["size"] += 1
        p = _precision(t, poi_category)
        if p is not None:
            rows[c]["precision"].append(p)
        if t["visited"]:
            rows[c]["visited_low_pressure"].append(np.mean([low_pressure[pid] for pid in t["visited"]]))

    out = []
    for c, r in rows.items():
        out.append({
            "cluster": c,
            "segment": r["segment"],
            "size": r["size"],
            "precision": float(np.mean(r["precision"])) if r["precision"] else 0.0,
            "visited_low_pressure": float(np.mean(r["visited_low_pressure"])) if r["visited_low_pressure"] else 0.0,
        })
    return pd.DataFrame(out).set_index("cluster").sort_values("size", ascending=False)


def aggregation_analysis(pois, res_sustain, res_wa, k=10):
    """Compare ELECTRE-III against the weighted average on key metrics and rank agreement."""
    keep = ["precision", "top_popular_visit_share", "district_visit_gini", "low_pressure_district_share"]
    electre = {m: all_metrics(res_sustain, pois, k)[m] for m in keep}
    wa = {m: all_metrics(res_wa, pois, k)[m] for m in keep}

    a = {id(t["profile"]): set(t["recommended"][:k]) for t in res_sustain["tourists"]}
    b = {id(t["profile"]): set(t["recommended"][:k]) for t in res_wa["tourists"]}
    overlap = float(np.mean([len(a[i] & b[i]) / k for i in a if i in b]))

    table = pd.DataFrame({"ELECTRE-III": electre, "weighted_average": wa})
    table.loc["top10_overlap"] = [overlap, overlap]
    return table


def explanation_cases(pois):
    """Show, for a few illustrative tourists, the recommended place and the criterion
    values that justify it. This is the transparency the trustworthy systems seminar asks for."""
    mcda = SustainabilityRecommender()

    def profile(label, interest):
        weights = {c: 0.1 for c in pois["category"].unique()}
        weights[interest] = 0.95
        return label, Profile(interests=weights, budget=80.0, mobility_mode="walk",
                              walking_tolerance=2.0, crowd_aversion=0.6, sustainability_sensitivity=0.7,
                              outdoor_preference=0.5, travels_with_kids=False, group_size=2,
                              trip_length=5, home_district="Eixample", weather="sunny")

    def values(p, poi_id):
        row = pois[pois["poi_id"] == poi_id]
        pref = float(preference_fit(p, row).iloc[0])
        r = row.iloc[0]
        return {"preference_fit": round(pref, 2), "low_pressure": round(float(r.low_pressure), 2),
                "environmental": round(float(r.environmental), 2), "accessibility": round(float(r.accessibility), 2),
                "cultural": round(float(r.cultural), 2), "local_economy": round(float(r.local_economy), 2)}

    cases = [profile("museum lover", "museum"), profile("religious-site lover", "religious"),
             profile("nature lover", "park_nature")]
    rows = []
    for label, p in cases:
        top = mcda.recommend(p, pois)[0]
        r = pois[pois["poi_id"] == top].iloc[0]
        rows.append({"tourist": label, "top_recommendation": r["name"], "category": r["category"],
                     "district": r["district"], **values(p, top)})
    return pd.DataFrame(rows).set_index("tourist")


def main():
    parser = argparse.ArgumentParser(description="Extra seminar-grounded analyses.")
    parser.add_argument("--pois", default="data/pois_barcelona.csv")
    parser.add_argument("--districts", default="data/nbhd_indicators.csv")
    parser.add_argument("--tourists", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    pois = load_pois(args.pois, args.districts)
    population = generate_population(args.tourists, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    res_sustain = _run(pois, population, SustainabilityRecommender(), args.seed)
    res_context = _run(pois, population, ContextAwareRecommender(SustainabilityRecommender()), args.seed)
    res_wa = _run(pois, population, WeightedAverageRecommender(), args.seed)

    context = context_analysis(pois, res_sustain, res_context)
    segments = segment_analysis(pois, population, res_sustain, seed=args.seed)
    aggregation = aggregation_analysis(pois, res_sustain, res_wa)
    explanations = explanation_cases(pois)

    context.to_csv(out / "context.csv")
    segments.to_csv(out / "segments.csv")
    aggregation.to_csv(out / "aggregation.csv")
    explanations.to_csv(out / "explanations.csv")

    print("=== Context (rainy-day outdoor visit share) ===")
    print(context.round(3).to_string())
    print("\n=== Segments (under the sustainability recommender) ===")
    print(segments.round(3).to_string())
    print("\n=== Aggregation: ELECTRE-III vs weighted average ===")
    print(aggregation.round(3).to_string())
    print("\n=== Explanation case studies ===")
    print(explanations.to_string())


if __name__ == "__main__":
    main()
