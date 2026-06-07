"""The city model: it holds the POIs, the crowding state and the tourist agents."""

import numpy as np
import pandas as pd
from mesa import Model

from ..constants import CITY_CORE
from .agent import Tourist


class CityModel(Model):
    def __init__(self, pois, population, recommender, seed=0):
        super().__init__(rng=seed)
        self.pois = pois.reset_index(drop=True)
        self.recommender = recommender
        self.np_rng = np.random.default_rng(seed)

        self.coords = {r.poi_id: (float(r.lat), float(r.lon)) for r in self.pois.itertuples()}
        self.capacity = {r.poi_id: float(r.capacity) for r in self.pois.itertuples()}
        self.fee = {r.poi_id: (0.0 if pd.isna(r.entrance_fee_eur) else float(r.entrance_fee_eur)) for r in self.pois.itertuples()}
        self.district = {r.poi_id: r.district for r in self.pois.itertuples()}

        centroids = self.pois.groupby("district")[["lat", "lon"]].mean()
        self._centroids = {d: (float(row.lat), float(row.lon)) for d, row in centroids.iterrows()}

        self.visits = {pid: 0 for pid in self.pois["poi_id"]}
        self.district_visits = {d: 0 for d in self.pois["district"].unique()}

        self.max_steps = max(p.trip_length for p in population)
        for profile in population:
            Tourist(self, profile)

    def district_centroid(self, district):
        return self._centroids.get(district, CITY_CORE)

    def record_visit(self, poi_id):
        self.visits[poi_id] += 1
        self.district_visits[self.district[poi_id]] += 1

    def run(self):
        for _ in range(self.max_steps):
            self.agents.shuffle_do("step")
        return self.collect()

    def collect(self):
        tourists = [
            {"profile": a.profile, "recommended": a.recommendations, "visited": a.visited}
            for a in self.agents
        ]
        return {
            "strategy": self.recommender.name,
            "tourists": tourists,
            "visits": dict(self.visits),
            "district_visits": dict(self.district_visits),
        }
