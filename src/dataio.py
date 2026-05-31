"""Loading and validation of the Barcelona POI dataset, plus the derived criteria."""

import numpy as np
import pandas as pd

from .constants import (
    ACCESSIBILITY_SCORE,
    CATEGORIES,
    CITY_CORE,
    DISTRICTS,
    HERITAGE_SCORE,
)

REQUIRED_COLUMNS = [
    "poi_id", "name", "category", "lat", "lon", "district", "barri",
    "wiki_pageviews_12m", "annual_visitors", "capacity_daily",
    "entrance_fee_eur", "is_green", "wheelchair_accessible", "heritage_protected",
]
NO_NAN_COLUMNS = ["name", "category", "lat", "lon", "district"]


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in kilometres between two coordinates."""
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _minmax(series):
    values = series.astype(float)
    lo, hi = values.min(), values.max()
    if hi - lo < 1e-9:
        return pd.Series(np.full(len(values), 0.5), index=values.index)
    return (values - lo) / (hi - lo)


def _match_to_canonical(values, canonical, label):
    """Map free-text names onto a canonical list, ignoring case and spacing."""
    lookup = {name.casefold().strip(): name for name in canonical}
    matched, unknown = [], set()
    for value in values:
        key = str(value).casefold().strip()
        if key in lookup:
            matched.append(lookup[key])
        else:
            matched.append(value)
            unknown.add(value)
    if unknown:
        raise ValueError(f"Unknown {label} values: {sorted(unknown)}")
    return matched


def load_districts(path):
    """Load the per-district indicators and normalise the district names."""
    df = pd.read_csv(path)
    df["district"] = _match_to_canonical(df["district"], DISTRICTS, "district")
    return df


def load_pois(pois_path, districts_path):
    """Load the POI dataset, validate it and attach the derived criteria columns."""
    df = pd.read_csv(pois_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(df) < 50:
        raise ValueError(f"The dataset must hold at least 50 POIs, found {len(df)}.")

    for column in NO_NAN_COLUMNS:
        if df[column].isna().any():
            raise ValueError(f"Column '{column}' has empty values.")

    bad_categories = sorted(set(df["category"]) - set(CATEGORIES))
    if bad_categories:
        raise ValueError(f"Unknown categories: {bad_categories}")
    df["district"] = _match_to_canonical(df["district"], DISTRICTS, "district")

    districts = load_districts(districts_path)
    df = df.merge(districts[["district", "tourist_density_per_km2"]], on="district", how="left")

    df["popularity"] = _build_popularity(df)
    df["low_pressure"] = 1.0 - _minmax(df["tourist_density_per_km2"].fillna(df["tourist_density_per_km2"].median()))
    df["environmental"] = df["is_green"].fillna(0).astype(float).clip(0, 1)
    df["accessibility"] = df["wheelchair_accessible"].fillna("unknown").map(ACCESSIBILITY_SCORE).fillna(0.5)
    df["cultural"] = df["heritage_protected"].fillna("unknown").map(HERITAGE_SCORE).fillna(0.3)
    df["local_economy"] = _minmax(haversine_km(df["lat"], df["lon"], CITY_CORE[0], CITY_CORE[1]))
    df["capacity"] = _build_capacity(df)

    return df


def _build_popularity(df):
    """Popularity from Wikipedia pageviews, falling back to annual visitors."""
    source = df["wiki_pageviews_12m"].astype(float)
    fallback = df["annual_visitors"].astype(float)
    source = source.fillna(fallback)
    source = source.fillna(source.min() if source.notna().any() else 0.0)
    return _minmax(source)


def _build_capacity(df):
    """Daily capacity: the official figure when known, else scaled by popularity."""
    official = df["capacity_daily"].astype(float)
    scaled = 200.0 + 4800.0 * df["popularity"]
    return official.fillna(scaled)
