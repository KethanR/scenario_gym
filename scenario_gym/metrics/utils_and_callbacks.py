import numpy as np

from scenario_gym.entity.pedestrian import Pedestrian
from scenario_gym.state import State
from scenario_gym.callback import StateCallback

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
    

# ---------- Callback -> Lane and Lane Center for Distance Metrics ----------#
class LaneAndLaneCenter(StateCallback):
    """
    Callback to retrieve the current lane and lane center index. 
    
    Calculates the lane center index closest to an entity (ego vehicle) for a given road/lane. 
    """

    def _reset(self, state: State) -> None:
        """Reset callback and declares variables."""
        self.ego = state.scenario.ego
        self.entities = state.scenario.entities
        # Initialise default callback parameters
        self.min_distance = float("inf")
        self.final_lane_center_index = None
        self.final_lane_index = None


    def __call__(self, state: State) -> None:
        # Initialize variables to deduce the closest lane center coordinates
        min_distance = float("inf")
        final_lane_center_index = None
        final_lane_index = None

        # Get ego vehicle pose, velocity, and lane.
        entity_pos = state.poses[self.ego][:2]
        # vel_ego = state.velocities[self.ego][:2]
        entity_road_network_information = state.get_road_info_at_entity(self.ego)
        # # Get non_ego vehicle pose, velocity, and lane.
        # pos_non_ego = state.poses[entity][:2]
        # vel_non_ego = state.velocities[entity][:2]
        # non_ego_road_network_info = state.get_road_info_at_entity(entity)

        road_network_types = entity_road_network_information[0]
        road_network_ids = entity_road_network_information[1]

        if "Lane" not in road_network_types:
            return 0, 0

        for lane_index, road_network_type in enumerate(road_network_types):
            if road_network_type == "Lane":
                lane = road_network_ids[lane_index]
                # Loop through lane center points to find the closest one to entity
                for lane_center_index, coords in enumerate(lane.center.coords):
                    # Compute the Euclidean distance between ego and lane center
                    distance = abs(np.linalg.norm(entity_pos - coords))
                    # Update the closest lane center index
                    if distance < min_distance:
                        min_distance = distance
                        final_lane_center_index = lane_center_index
                        final_lane_index = lane_index

        self.current_lane = road_network_ids[final_lane_index]
        self.final_lane_center_index = final_lane_center_index