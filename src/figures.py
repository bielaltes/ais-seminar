"""Turn the exported result tables into the figures used by the report."""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STRATEGY_ORDER = ["popularity", "interests", "sustainability", "random"]


def _grouped_bar(metrics, columns, title, path):
    data = metrics.loc[STRATEGY_ORDER, columns]
    ax = data.plot(kind="bar", figsize=(8, 4.5), rot=0)
    ax.set_title(title)
    ax.set_xlabel("recommender")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _district_shares(district_visits, path):
    shares = district_visits[STRATEGY_ORDER] / district_visits[STRATEGY_ORDER].sum()
    ax = shares.plot(kind="bar", figsize=(10, 4.5), rot=45)
    ax.set_title("Share of visits per district")
    ax.set_ylabel("share of all visits")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _visit_maps(poi_visits, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, strategy in zip(axes.ravel(), STRATEGY_ORDER):
        column = f"visits_{strategy}"
        sizes = 8 + 0.05 * poi_visits[column]
        ax.scatter(poi_visits["lon"], poi_visits["lat"], s=sizes, alpha=0.6, c=poi_visits[column], cmap="viridis")
        ax.set_title(strategy)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
    fig.suptitle("Where tourists end up, by recommender")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def make_figures(results_dir):
    out = Path(results_dir)
    metrics = pd.read_csv(out / "metrics.csv", index_col="strategy")
    district_visits = pd.read_csv(out / "district_visits.csv", index_col="district")
    poi_visits = pd.read_csv(out / "poi_visits.csv")

    _grouped_bar(metrics, ["precision", "recall", "f1", "ndcg"], "Accuracy metrics", out / "fig_accuracy.png")
    _grouped_bar(metrics, ["diversity", "coverage"], "Diversity and coverage", out / "fig_beyond.png")
    _grouped_bar(metrics, ["poi_visit_gini", "district_visit_gini", "exposure_gini", "top_popular_visit_share"],
                 "Concentration and fairness", out / "fig_fairness.png")
    _grouped_bar(metrics, ["visited_low_pressure", "visited_environmental", "visited_accessibility",
                           "visited_cultural", "visited_local_economy"],
                 "Sustainability of the visited POIs", out / "fig_sustainability.png")
    _district_shares(district_visits, out / "fig_districts.png")
    _visit_maps(poi_visits, out / "fig_maps.png")


def main():
    parser = argparse.ArgumentParser(description="Build the report figures from the results.")
    parser.add_argument("--results", default="results")
    args = parser.parse_args()
    make_figures(args.results)


if __name__ == "__main__":
    main()
