from scenario_gym.state import State

from .base import Metric


# ---------- Collision Check ----------#
class CollisionTimestamp(Metric):
    """Checks if the ego vehicle collided with any object in the scenario."""

    name = "collision_timestamp"

    def _reset(self, state: State) -> None:
        self.ego = state.scenario.ego
        self.collision_check_and_timestamp = False
        self.t = 0.0

    def _step(self, state: State) -> None:
        if not self.collision_check_and_timestamp:
            if len(state.collisions()[state.scenario.entities[0]]) > 0:
                self.collision_check_and_timestamp = state.t

    def get_state(self) -> float | bool:
        """Return collision check and timestamp."""
        return self.collision_check_and_timestamp