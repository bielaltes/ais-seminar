"""Sustainability-outcome metrics measured on the POIs the tourists actually visited."""

import numpy as np


def _visit_weighted_mean(visits, value_of):
    total = sum(visits.values())
    if total == 0:
        return 0.0
    return float(sum(visits[pid] * value_of[pid] for pid in visits) / total)


def sustainability_metrics(result, pois):
    visits = result["visits"]
    criteria = ["low_pressure", "environmental", "accessibility", "cultural", "local_economy"]
    metrics = {}
    for c in criteria:
        value_of = {pid: float(v) for pid, v in zip(pois["poi_id"], pois[c])}
        metrics[f"visited_{c}"] = _visit_weighted_mean(visits, value_of)

    pressure = pois.groupby("district")["district_pressure"].first()
    low_pressure_districts = set(pressure[pressure <= pressure.median()].index)
    total_visits = sum(result["district_visits"].values()) or 1
    share = sum(v for d, v in result["district_visits"].items() if d in low_pressure_districts) / total_visits
    metrics["low_pressure_district_share"] = float(share)
    return metrics
