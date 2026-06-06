"""Common interface shared by every recommender."""


class Recommender:
    """A recommender turns a tourist profile into a ranked list of POI ids."""

    name = "base"

    def recommend(self, profile, pois):
        """Return a list of poi_id ordered from the best to the worst suggestion."""
        raise NotImplementedError
