import numpy as np
from shapely.geometry import Polygon

from scenario_gym.entity.pedestrian import Pedestrian
from scenario_gym.entity.vehicle import Vehicle
from scenario_gym.state import State

from .base import Metric


# ---------- Severity Metrics ----------#
class DeltaV(Metric):
    """
    Calculates Delta-V metric.

    Defined as the change in velocity between pre-collision
    and post-collision trajectories of a vehicle.
    Will leverage inelastic collision kinematics to predict delta-v
    during runtime IF entities are on a collision course.
    Dev Note: This metric only considers traffic particpants/objects that
    are "pedestrians" (scenario_gym.entity.pedestrian.Pedestrian)
    or "vehicles" (scenario_gym.entity.vehicle.Vehicle) for now.
    """

    name = "delta_v"

    def _reset(self, state: State) -> None:
        """Reset the delta_v measure for each entity."""
        self.ego = state.scenario.ego
        self.delta_v_dict = {}
        self.predefined_masses = {"Vehicle": 1500, "Pedestrian": 80}

    def _translate_polygon(self, polygon, dx, dy) -> Polygon:
        return Polygon([(p[0] + dx, p[1] + dy) for p in polygon.exterior.coords])

    def _will_collide_check(
        self,
        pos1,
        vel1,
        box1,
        pos2,
        vel2,
        box2,
        distance_tolerance,
        time_horizon=5,
        time_steps=50,
    ) -> list:
        x1, y1 = pos1
        vx1, vy1 = vel1
        x2, y2 = pos2
        vx2, vy2 = vel2

        times = np.linspace(0, time_horizon, time_steps)

        for t in times:
            # Project bounding boxes to their positions at time t.
            box1_at_t = self._translate_polygon(
                box1, x1 + vx1 * t - x1, y1 + vy1 * t - y1
            )
            box2_at_t = self._translate_polygon(
                box2, x2 + vx2 * t - x2, y2 + vy2 * t - y2
            )
            # Check if the projected bounding boxes
            # intersect (projected to actually overlap).
            # If no intersection, check if they're projected to be
            # within the distance tolerance (i.e. a near miss).
            if (
                box1_at_t.intersects(box2_at_t)
                or box1_at_t.distance(box2_at_t) <= distance_tolerance
            ):
                return [True, t]
        return [False, None]

    def _step(self, state: State) -> None:
        """
        Delta-V for every timestep.

        Predict the delta_v for each entity potentially involved
        in a collision with ego vehicle.
        """
        # Get entities within 10m of the ego, if entities exist...
        if len(state.get_entities_in_radius(*state.poses[self.ego][:2], 10)) > 1:
            # Then loop through non_ego entities and check if they will reach
            # intersection point at approx. the same time as the ego (function
            # _will_collide does this).
            for entity in state.get_entities_in_radius(
                *state.poses[self.ego][:2], 10
            ):
                if entity == self.ego:
                    continue

                # Get ego vehicle pose, velocity, and bounding box.
                pos_ego = state.poses[self.ego][:2]
                vel_ego = state.velocities[self.ego][:2]
                box_ego = self.ego.get_bounding_box_geom(state.poses[self.ego])
                # Get non_ego vehicle pose, velocity, and bounding box.
                pos_non_ego = state.poses[entity][:2]
                vel_non_ego = state.velocities[entity][:2]
                box_non_ego = entity.get_bounding_box_geom(state.poses[entity])

                if isinstance(entity, Pedestrian):
                    distance_tolerance = 1
                elif isinstance(entity, Vehicle):
                    distance_tolerance = 0.1

                # Check if ego will collide with other entity.
                self.ego_time_to_collision_bool_and_value = (
                    self._will_collide_check(
                        pos_ego,
                        vel_ego,
                        box_ego,
                        pos_non_ego,
                        vel_non_ego,
                        box_non_ego,
                        distance_tolerance,
                    )
                )

                # If they will collide, then retrieve time-to-collision.
                if not self.ego_time_to_collision_bool_and_value[0]:
                    continue

                elif not isinstance(entity, Pedestrian) and not isinstance(
                    entity, Vehicle
                ):
                    continue

                elif self.ego_time_to_collision_bool_and_value[1] < 1:
                    ego_class = self.ego.__class__.__name__
                    entity_class = entity.__class__.__name__
                    entity_delta_v = (
                        self.predefined_masses[ego_class]
                        / (
                            self.predefined_masses[ego_class]
                            + self.predefined_masses[entity_class]
                        )
                    ) * (np.linalg.norm(vel_ego) - np.linalg.norm(vel_non_ego))

                    # state.t inclusion in the dictionary (line below). To
                    # track/timestamp TTC in simulation.
                    if state.t not in self.delta_v_dict:
                        self.delta_v_dict[state.t] = []
                    self.delta_v_dict[state.t].append(
                        {"delta_v": {f"{entity}": entity_delta_v}}
                    )
                else:
                    continue

    def get_state(self) -> dict:
        """Return delta_v dictionary."""
        return self.delta_v_dict