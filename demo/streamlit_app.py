"""Streamlit dashboard that shows the simulation results for each recommender."""

from pathlib import Path

import pandas as pd
import pydeck as pdk
import streamlit as st

RESULTS = Path(__file__).resolve().parent.parent / "results"
STRATEGIES = ["popularity", "interests", "sustainability", "random"]
ACCURACY = ["precision", "recall", "f1", "ndcg"]
BEYOND = ["diversity", "coverage"]
FAIRNESS = ["poi_visit_gini", "district_visit_gini", "exposure_gini", "top_popular_visit_share"]
SUSTAINABILITY = ["visited_low_pressure", "visited_environmental", "visited_accessibility",
                  "visited_cultural", "visited_local_economy"]


@st.cache_data
def load_results():
    metrics = pd.read_csv(RESULTS / "metrics.csv", index_col="strategy")
    poi_visits = pd.read_csv(RESULTS / "poi_visits.csv")
    district_visits = pd.read_csv(RESULTS / "district_visits.csv", index_col="district")
    return metrics, poi_visits, district_visits


def crowding_map(poi_visits, strategy):
    column = f"visits_{strategy}"
    data = poi_visits.copy()
    top = data[column].max() or 1
    data["radius"] = 40 + 300 * data[column] / top
    data["red"] = (255 * data[column] / top).astype(int)
    data["blue"] = 255 - data["red"]
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position="[lon, lat]",
        get_radius="radius",
        get_fill_color="[red, 60, blue, 160]",
        pickable=True,
    )
    view = pdk.ViewState(latitude=41.39, longitude=2.17, zoom=11)
    tooltip = {"text": "{name}\n" + column + ": {" + column + "}"}
    return pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip)


st.set_page_config(page_title="Sustainable POI recommender — evaluation", layout="wide")
st.title("Agent-based evaluation of a sustainable POI recommender")
st.caption("Barcelona — comparing popularity, interests, sustainability-aware and random recommenders.")

if not (RESULTS / "metrics.csv").exists():
    st.warning("No results yet. Run `python -m src.experiment` first.")
    st.stop()

metrics, poi_visits, district_visits = load_results()

st.header("Where tourists end up")
strategy = st.selectbox("Recommender", STRATEGIES, index=2)
st.pydeck_chart(crowding_map(poi_visits, strategy))
st.caption("Larger, redder dots are more crowded POIs under the chosen recommender.")

st.header("How the recommenders compare")
left, right = st.columns(2)
with left:
    st.subheader("Accuracy")
    st.bar_chart(metrics.loc[STRATEGIES, ACCURACY])
    st.subheader("Diversity and coverage")
    st.bar_chart(metrics.loc[STRATEGIES, BEYOND])
with right:
    st.subheader("Concentration and fairness")
    st.bar_chart(metrics.loc[STRATEGIES, FAIRNESS])
    st.subheader("Sustainability of the visited POIs")
    st.bar_chart(metrics.loc[STRATEGIES, SUSTAINABILITY])

st.header("Spread of visits across districts")
shares = district_visits[STRATEGIES] / district_visits[STRATEGIES].sum()
st.bar_chart(shares)

st.header("All metrics")
st.dataframe(metrics.round(3))
