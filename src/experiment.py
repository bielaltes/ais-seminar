"""Run the four recommenders on one tourist population and export the results."""

import argparse
from pathlib import Path

import pandas as pd

from .dataio import load_pois
from .metrics.accuracy import accuracy_metrics
from .metrics.beyond_accuracy import beyond_accuracy_metrics
from .metrics.fairness import fairness_metrics
from .metrics.sustainability import sustainability_metrics
from .profiles import generate_population
from .recommenders.interests import InterestsRecommender
from .recommenders.popularity import PopularityRecommender
from .recommenders.random_rec import RandomRecommender
from .recommenders.sustainability import SustainabilityRecommender
from .simulation.model import CityModel


def all_metrics(result, pois, k):
    row = {"strategy": result["strategy"]}
    row.update(accuracy_metrics(result, pois, k))
    row.update(beyond_accuracy_metrics(result, pois, k))
    row.update(fairness_metrics(result, pois, k))
    row.update(sustainability_metrics(result, pois))
    return row


def build_recommenders(seed):
    return [
        PopularityRecommender(),
        InterestsRecommender(),
        SustainabilityRecommender(),
        RandomRecommender(seed),
    ]


def run_experiment(pois_path, districts_path, n_tourists, seed, out_dir, k):
    pois = load_pois(pois_path, districts_path)
    population = generate_population(n_tourists, seed)

    rows, poi_visits, district_visits = [], {}, {}
    for rec in build_recommenders(seed):
        result = CityModel(pois, population, rec, seed=seed).run()
        rows.append(all_metrics(result, pois, k))
        poi_visits[f"visits_{rec.name}"] = result["visits"]
        district_visits[rec.name] = result["district_visits"]

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame(rows).set_index("strategy")
    metrics.to_csv(out / "metrics.csv")

    meta = pois.set_index("poi_id")[["name", "category", "lat", "lon", "district", "popularity", "low_pressure", "capacity"]]
    meta.join(pd.DataFrame(poi_visits)).to_csv(out / "poi_visits.csv")
    pd.DataFrame(district_visits).rename_axis("district").to_csv(out / "district_visits.csv")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate POI recommenders with an agent-based simulation.")
    parser.add_argument("--pois", default="data/pois_barcelona.csv")
    parser.add_argument("--districts", default="data/nbhd_indicators.csv")
    parser.add_argument("--tourists", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    metrics = run_experiment(args.pois, args.districts, args.tourists, args.seed, args.out, args.k)
    print(metrics.round(3).to_string())


if __name__ == "__main__":
    main()
