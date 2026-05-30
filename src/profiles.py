"""Tourist profiles and the generator that builds a population of them."""

from dataclasses import dataclass, field

import numpy as np

from .constants import CATEGORIES, DISTRICTS


@dataclass
class Profile:
    interests: dict          # weight in [0, 1] for each POI category
    budget: float            # euros available for entrance fees during the trip
    mobility_mode: str       # "walk" or "transit"
    walking_tolerance: float # kilometres the tourist accepts to walk per leg
    crowd_aversion: float    # in [0, 1]; how strongly crowding discourages a visit
    sustainability_sensitivity: float  # in [0, 1]; weight given to sustainability
    outdoor_preference: float          # in [0, 1]; preference for outdoor places
    travels_with_kids: bool
    group_size: int
    trip_length: int         # number of POIs the tourist plans to visit
    home_district: str       # district of the hotel / entry point


def _sample_interests(rng):
    """Give one to three categories a high weight and the rest a low one."""
    weights = {c: float(rng.uniform(0.0, 0.25)) for c in CATEGORIES}
    n_pref = int(rng.integers(1, 4))
    for c in rng.choice(CATEGORIES, size=n_pref, replace=False):
        weights[c] = float(rng.uniform(0.6, 1.0))
    return weights


def generate_population(n, seed=0):
    """Return a list of ``n`` tourist profiles sampled with a fixed seed."""
    rng = np.random.default_rng(seed)
    population = []
    for _ in range(n):
        with_kids = bool(rng.random() < 0.30)
        profile = Profile(
            interests=_sample_interests(rng),
            budget=float(rng.uniform(20.0, 150.0)),
            mobility_mode=str(rng.choice(["walk", "transit"], p=[0.45, 0.55])),
            walking_tolerance=float(rng.uniform(0.5, 3.0)),
            crowd_aversion=float(rng.beta(2.0, 2.0)),
            sustainability_sensitivity=float(rng.beta(2.0, 3.0)),
            outdoor_preference=float(rng.beta(2.0, 2.0)),
            travels_with_kids=with_kids,
            group_size=int(rng.integers(1, 6)),
            trip_length=int(rng.integers(3, 9)),
            home_district=str(rng.choice(DISTRICTS)),
        )
        population.append(profile)
    return population
