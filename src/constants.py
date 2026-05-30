"""Shared vocabulary for POIs, districts and the sustainability criteria."""

# The closed set of POI categories used across the dataset and the recommenders.
CATEGORIES = [
    "monument",
    "museum",
    "religious",
    "park_nature",
    "beach",
    "market",
    "viewpoint",
    "cultural_venue",
    "architecture",
    "square_landmark",
    "leisure",
]

# The ten official districts of Barcelona.
DISTRICTS = [
    "Ciutat Vella",
    "Eixample",
    "Sants-Montjuïc",
    "Les Corts",
    "Sarrià-Sant Gervasi",
    "Gràcia",
    "Horta-Guinardó",
    "Nou Barris",
    "Sant Andreu",
    "Sant Martí",
]

# The historic tourist core of the city, used to measure how far a POI spreads
# visitors away from the saturated centre. Coordinates of Plaça de Catalunya.
CITY_CORE = (41.3870, 2.1701)

# How each accessibility label maps to a score in [0, 1].
ACCESSIBILITY_SCORE = {"yes": 1.0, "limited": 0.5, "no": 0.0, "unknown": 0.5}

# How each heritage-protection label maps to a cultural-value score in [0, 1].
HERITAGE_SCORE = {"BCIN": 1.0, "BCIL": 0.7, "none": 0.2, "unknown": 0.3}

# The six criteria the sustainability-aware recommender balances.
CRITERIA = [
    "preference_fit",
    "low_pressure",
    "environmental",
    "accessibility",
    "cultural",
    "local_economy",
]
