"""Small helpers to reason about crowding from visit counts and capacities."""


def crowd_ratio(visits, capacity):
    """Visits divided by capacity; zero when the capacity is unknown."""
    return visits / capacity if capacity > 0 else 0.0


def overcrowding_count(visit_counts, capacities):
    """Number of POIs whose visits exceed their daily capacity."""
    return sum(1 for pid, v in visit_counts.items() if v > capacities[pid])
