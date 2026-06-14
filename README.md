# Sustainable POI recommender — agent-based evaluation (Barcelona)

This repository contains an agent-based simulator that **evaluates** a multi-criteria,
sustainability-aware Point-of-Interest (POI) recommender for Barcelona. It compares four
recommendation strategies and measures whether the sustainability-aware one reduces overcrowding at the
famous spots and spreads tourists more evenly across the city.

The four strategies compared are:

- **Popularity** — ranks POIs by their popularity, ignoring the tourist profile.
- **Interests** — ranks POIs by how well their category matches the tourist's interests.
- **Sustainability-aware** — a multi-criteria recommender (ELECTRE-III) that balances the tourist's
  preferences with environmental, social, cultural and economic criteria.
- **Random** — a uniform baseline used as a lower bound.

## Layout

```
data/        the Barcelona POI dataset and district indicators
src/         profiles, data loading, recommenders, simulation, metrics, experiment runner
demo/        a Streamlit dashboard that shows the simulation results
results/     generated tables and figures
```

## How to run

```bash
pip install -r requirements.txt
python -m src.experiment        # runs the four strategies and writes results/
python -m src.analysis          # context, segmentation and aggregation studies
python -m src.figures           # builds the figures in results/
streamlit run demo/streamlit_app.py
```

On top of the main comparison, `src/analysis.py` adds three seminar-grounded
studies: a context-aware post-filter that reacts to the weather, a k-means
segmentation of the tourists, and a comparison of ELECTRE-III against a weighted
average together with per-criterion explanations.

## Data

The dataset is built from public, citable sources (Open Data BCN, Wikimedia Pageviews, the Observatori del
Turisme a Barcelona, OpenStreetMap and the Generalitat heritage registry). The full list of references is
given in the written report.
