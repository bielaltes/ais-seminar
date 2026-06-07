"""The tourist agent: it follows a recommendation and reacts to crowding."""

import numpy as np
from mesa import Agent

from ..dataio import haversine_km

CHOICE_POOL = 8      # how many of the top suggestions the tourist looks at each step
W_RANK = 2.0         # weight of the recommended order
W_CROWD = 1.0        # weight of the crowding penalty
W_TRAVEL = 0.7       # weight of the travel penalty
TEMPERATURE = 0.5    # randomness of the choice (lower is greedier)


def _softmax(values):
    shifted = values - values.max()
    weights = np.exp(shifted / TEMPERATURE)
    return weights / weights.sum()


class Tourist(Agent):
    def __init__(self, model, profile):
        super().__init__(model)
        self.profile = profile
        self.budget = profile.budget
        self.steps_left = profile.trip_length
        self.location = model.district_centroid(profile.home_district)
        self.recommendations = model.recommender.recommend(profile, model.pois)
        self.visited = []
        self._visited_set = set()
        self.done = False

    def _candidates(self):
        pool = []
        for pid in self.recommendations:
            if pid in self._visited_set:
                continue
            if self.model.fee[pid] <= self.budget:
                pool.append(pid)
            if len(pool) == CHOICE_POOL:
                break
        return pool

    def _utility(self, position, pool_size, pid):
        rank_score = 1.0 - position / pool_size
        ratio = self.model.visits[pid] / self.model.capacity[pid]
        crowd_penalty = self.profile.crowd_aversion * min(ratio, 3.0)
        reach = self.profile.walking_tolerance * (1.0 if self.profile.mobility_mode == "walk" else 5.0)
        distance = haversine_km(self.location[0], self.location[1], *self.model.coords[pid])
        travel_penalty = min(distance / reach, 3.0)
        return W_RANK * rank_score - W_CROWD * crowd_penalty - W_TRAVEL * travel_penalty

    def step(self):
        if self.done:
            return
        pool = self._candidates()
        if not pool:
            self.done = True
            return
        utilities = np.array([self._utility(i, len(pool), pid) for i, pid in enumerate(pool)])
        choice = int(self.model.np_rng.choice(len(pool), p=_softmax(utilities)))
        pid = pool[choice]

        self.visited.append(pid)
        self._visited_set.add(pid)
        self.budget -= self.model.fee[pid]
        self.location = self.model.coords[pid]
        self.model.record_visit(pid)

        self.steps_left -= 1
        if self.steps_left <= 0:
            self.done = True
