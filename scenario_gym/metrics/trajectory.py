import numpy as np
from shapely.geometry import Polygon

from scenario_gym.entity.pedestrian import Pedestrian
from scenario_gym.state import State

from .base import Metric


# ---------- Speed/Velocity Metrics ----------#
class EgoAvgSpeed(Metric):
    """Measure the average speed of the ego."""

    name = "ego_avg_speed"

    def _reset(self, state: State) -> None:
        """Reset the average speed."""
        self.ego = state.scenario.ego
        self.ego_avg_speed = np.linalg.norm(state.velocities[self.ego][:3])
        self.t = 0.0

    def _step(self, state: State) -> None:
        """Update the average speed."""
        speed = np.linalg.norm(state.velocities[self.ego][:3])
        w = self.t / state.t
        self.ego_avg_speed += (1.0 - w) * (speed - self.ego_avg_speed)
        self.t = state.t

    def get_state(self) -> float:
        """Return the current average speed."""
        return self.ego_avg_speed


class EgoMaxSpeed(Metric):
    """Measure the maximum speed of the ego."""

    name = "ego_max_speed"

    def _reset(self, state: State) -> None:
        """Reset the maximum speed."""
        self.ego = state.scenario.ego
        self.ego_max_speed = np.linalg.norm(state.velocities[self.ego][:3])

    def _step(self, state: State) -> None:
        """Update the maximum speed."""
        speed = np.linalg.norm(state.velocities[self.ego][:3])
        self.ego_max_speed = np.maximum(speed, self.ego_max_speed)

    def get_state(self) -> float:
        """Return the current max speed."""
        return self.ego_max_speed


# ---------- Distance Metrics ----------#
class EgoDistanceTravelled(Metric):
    """Measure the distance travelled by the ego."""

    name = "ego_distance_travelled"

    def _reset(self, state: State) -> None:
        """Find the ego."""
        self.ego = state.scenario.ego

    def _step(self, state: State) -> None:
        """Pass as entity will update its distance."""
        self.dist = state.distances[self.ego]

    def get_state(self) -> float:
        """Return the current distance travelled."""
        return self.dist


class DistanceToEgo(Metric):
    """Measure distance to ego vehicle."""

    name = "distance_to_ego"

    def _reset(self, state: State) -> None:
        self.ego = state.scenario.ego
        self.distance_to_ego_dict = {}

    def _step(self, state: State) -> None:
        for entity in state.get_entities_in_radius(*state.poses[self.ego][:2], 10):
            if entity == self.ego:
                continue

            # Get ego vehicle position.
            pos_ego = state.poses[self.ego][:2]
            # Get non_ego vehicle position.
            pos_non_ego = state.poses[entity][:2]

            difference_vector = pos_ego - pos_non_ego
            distance = np.linalg.norm(difference_vector)

            # state.t inclusion in the dictionary (line below). To track/timestamp
            # TTC in simulation.
            if state.t not in self.distance_to_ego_dict:
                self.distance_to_ego_dict[state.t] = []
            self.distance_to_ego_dict[state.t].append(
                {"distance_to_ego": {f"{entity}": distance}}
            )

    def get_state(self) -> float:
        return self.distance_to_ego_dict


class SafeLongDistance(Metric):
    """Determines safe longitudinal distance.
    All speed and acceleration inputs must be for the longitudinal axis
    responseTime: time it takes rear car to react and begin braking
    speedFront: current velocity of front car
    accFront: current acceleration of front car
    accFrontMaxBrake: max braking of front car
    speedRear: current velocity of rear car
    accRear: current acceleration of rear car
    accRearMaxResp: max acceleration of rear car during response time
    accRearMinBrake: min braking of rear car
    """

    name = "safe_longitudinal_distance"

    def _reset(self, state: State) -> None:
        self.ego = state.scenario.ego
        self.safe_long_dist_dict = {}
        self.safe_long_dist_brake_dict = {}
        self.long_dist_dict = {}
        self.long_risk_dict = {}
        # Time it takes rear car to react and begin braking (both longitudinal and lateral).
        # https://trl.co.uk/sites/default/files/PPR313_new.pdf
        # Unit: seconds (s)
        self.responseTime = 1.5
        # Max braking of front car (longitudinal)
        # Bokare, P. S., & Maurya, A. K. (2017). Acceleration-deceleration behaviour of various vehicle types. Transportation research procedia, 25, 4733-4749.
        # Unit: m/s^2
        self.accFrontMaxBrake = 2.5
        # Max acceleration of rear car during response time (longitudinal)
        # Bokare, P. S., & Maurya, A. K. (2017).Acceleration-deceleration behaviour of various vehicle types. Transportation research procedia, 25, 4733-4749.
        # Unit: m/s^2
        self.accRearMaxResp = 2.5
        # Min braking of rear car (longitudinal).
        # Unit: m/s^2
        self.accRearMinBrake = 1
        # Max braking capability of rear car (longitudinal) replace for ACC_REAR_MIN_BRAKE when calculating 'safeDistanceBrake'.
        # Bokare, P. S., & Maurya, A. K. (2017). Acceleration-deceleration behaviour of various vehicle types. Transportation research procedia, 25, 4733-4749.
        # Unit: m/s^2
        self.accRearMaxBrake = 4

    def _get_lane_and_lane_center_index(
        self, entity_pos, entity_road_network_information
    ):
        # Initialize variables to deduce the closest lane center coordinates to
        # entity
        min_distance = float("inf")
        final_lane_center_index = None
        final_lane_index = None

        if "Lane" not in entity_road_network_information[0]:
            # print("SafeLongDist: Silly Goose! Vehicle is out-of-bounds (no lane access).")
            return 0, 0

        for lane_index, road_network_type in enumerate(
            entity_road_network_information[0]
        ):
            if road_network_type == "Lane":
                lane = entity_road_network_information[1][lane_index]
                # Loop through lane center points to find the closest one to entity
                for lane_center_index, coords in enumerate(lane.center.coords):
                    # Compute the Euclidean distance between ego position and lane
                    # center
                    distance = abs(np.linalg.norm(entity_pos - coords))
                    # Update the closest lane center index if the distance is
                    # smaller
                    if distance < min_distance:
                        min_distance = distance
                        final_lane_center_index = lane_center_index
                        final_lane_index = lane_index

        current_lane = entity_road_network_information[1][final_lane_index]

        return current_lane, final_lane_center_index

    def _is_ego_leading_following_or_side(self, pos1, roadinfo1, pos2, roadinfo2):
        """
        Dev Note: Safe longitudinal distance is currently limited to instances where both vehicles are in the same lane.
        """
        (
            current_lane_vehicle_1,
            final_lane_center_index_vehicle_1,
        ) = self._get_lane_and_lane_center_index(pos1, roadinfo1)
        (
            current_lane_vehicle_2,
            final_lane_center_index_vehicle_2,
        ) = self._get_lane_and_lane_center_index(pos2, roadinfo2)

        if current_lane_vehicle_1 == 0 or current_lane_vehicle_2 == 0:
            return "out-of-bounds"

        elif current_lane_vehicle_1 == current_lane_vehicle_2:
            current_lane_center_point = np.array(
                current_lane_vehicle_1.center.coords[
                    final_lane_center_index_vehicle_1
                ]
            )
            next_lane_center_point = np.array(
                current_lane_vehicle_1.center.coords[
                    final_lane_center_index_vehicle_1 - 1
                ]
            )
            longitudinal_direction = -(
                next_lane_center_point - current_lane_center_point
            )
            # Project relative position vectors onto the direction vector using dot
            # product
            proj1 = np.dot(pos1, longitudinal_direction)
            proj2 = np.dot(pos2, longitudinal_direction)
            # Determine which position vector is ahead
            if proj1 > proj2:
                return "leading"
            else:
                return "following"

        else:
            return "side-by-side"

    def _longitudinal_speed(
        self, entity_pos, entity_vel, entity_road_network_information
    ):
        (
            current_lane,
            final_lane_center_index,
        ) = self._get_lane_and_lane_center_index(
            entity_pos, entity_road_network_information
        )

        # Convert closest lane center point and next-nearest lane center to array
        # Does not matter if the next lane center point is 'ahead' or 'behind' direction of travel...
        # Because we only need the projection of the entity's velocity onto the
        # longitudinal direction (no sign needed for speed)
        current_lane_center_point = np.array(
            current_lane.center.coords[final_lane_center_index]
        )
        next_lane_center_point = np.array(
            current_lane.center.coords[final_lane_center_index - 1]
        )

        longitudinal_direction = next_lane_center_point - current_lane_center_point
        normalised_longitudinal_direction = longitudinal_direction / np.linalg.norm(
            longitudinal_direction
        )

        # Calculate longitudinal speed
        longitudinal_speed = np.dot(entity_vel, normalised_longitudinal_direction)
        # Take the absolute value
        longitudinal_speed = np.abs(longitudinal_speed)

        return longitudinal_speed

    def _longitudinal_distance(
        self, pos_ego, pos_non_ego, entity_road_network_information
    ):
        (
            current_lane,
            final_lane_center_index,
        ) = self._get_lane_and_lane_center_index(
            pos_ego, entity_road_network_information
        )

        # Convert closest lane center point and next-nearest lane center to array
        # Does not matter if the next lane center point is 'ahead' or 'behind' direction of travel...
        # Because we only need the projection of the entity's velocity onto the
        # longitudinal direction (no sign needed for speed)
        current_lane_center_point = np.array(
            current_lane.center.coords[final_lane_center_index]
        )
        next_lane_center_point = np.array(
            current_lane.center.coords[final_lane_center_index - 1]
        )

        longitudinal_direction = next_lane_center_point - current_lane_center_point
        normalised_longitudinal_direction = longitudinal_direction / np.linalg.norm(
            longitudinal_direction
        )

        # Calculate the vector between ego and non-ego vehicle positions
        relative_position = pos_non_ego - pos_ego
        # Project the relative position onto the longitudinal direction
        longitudinal_distance = np.dot(
            relative_position, normalised_longitudinal_direction
        )
        # Take the absolute value of the longitudinal distance
        longitudinal_distance = np.abs(longitudinal_distance)

        return longitudinal_distance

    def _calculate_safe_long_dist(self, speed_rear, speed_front):
        sign = [1, -1][speed_rear + self.responseTime * self.accRearMaxResp < 0]
        safeLonDis = (
            0.5 * np.power(speed_front, 2) / self.accFrontMaxBrake
            - speed_rear * self.responseTime
            - 0.5 * np.power(self.responseTime, 2) * self.accRearMaxResp
            - 0.5
            * sign
            * np.power(speed_rear + self.responseTime * self.accRearMaxResp, 2)
            / self.accRearMinBrake
        )
        safeLonDisBrake = (
            0.5 * np.power(speed_front, 2) / self.accFrontMaxBrake
            - speed_rear * self.responseTime
            - 0.5 * np.power(self.responseTime, 2) * self.accRearMaxResp
            - 0.5
            * sign
            * np.power(speed_rear + self.responseTime * self.accRearMaxResp, 2)
            / self.accRearMaxBrake
        )

        return safeLonDis, safeLonDisBrake

    # Function to calculate the longitudinal or lateral risk index [0,1]
    def _long_risk_index(self, safe_distance, safe_distance_brake, distance):
        """
        All inputs must me either longitudinal or lateral safeDistance: safe longitudinal/lateral distance (use
        function SafeLonDistance/SafeLatDistance) safeDistanceBrake: safe longitudinal/lateral distance under max
        braking capacity (use function SafeLonDistance/SafeLatDistance with max braking acceleration) distance: current
        longitudinal/lateral distance between cars
        """
        if safe_distance + distance > 0:
            r = 0
        elif safe_distance_brake + distance <= 0:
            r = 1
        else:
            r = 1 - (safe_distance_brake + distance) / (
                safe_distance_brake - safe_distance
            )
        return r

    def _step(self, state: State) -> None:
        """Remember, all speed and acceleration inputs must be for the longitudinal axis."""
        # Get entities within 10m of the ego, if entities exist...
        if len(state.get_entities_in_radius(*state.poses[self.ego][:2], 10)) > 1:
            for entity in state.get_entities_in_radius(
                *state.poses[self.ego][:2], 10
            ):
                if entity == self.ego:
                    continue
                if isinstance(entity, Pedestrian):
                    continue

                # Get ego vehicle pose, velocity, and lane.
                pos_ego = state.poses[self.ego][:2]
                vel_ego = state.velocities[self.ego][:2]
                ego_road_network_info = state.get_road_info_at_entity(self.ego)
                # Get non_ego vehicle pose, velocity, and lane.
                pos_non_ego = state.poses[entity][:2]
                vel_non_ego = state.velocities[entity][:2]
                non_ego_road_network_info = state.get_road_info_at_entity(entity)

                ego_relative_position_string = (
                    self._is_ego_leading_following_or_side(
                        pos_ego,
                        ego_road_network_info,
                        pos_non_ego,
                        non_ego_road_network_info,
                    )
                )

                if ego_relative_position_string == "following":
                    speed_rear = self._longitudinal_speed(
                        pos_ego, vel_ego, ego_road_network_info
                    )
                    speed_front = self._longitudinal_speed(
                        pos_non_ego, vel_non_ego, non_ego_road_network_info
                    )
                elif ego_relative_position_string == "leading":
                    speed_rear = self._longitudinal_speed(
                        pos_non_ego, vel_non_ego, non_ego_road_network_info
                    )
                    speed_front = self._longitudinal_speed(
                        pos_ego, vel_ego, ego_road_network_info
                    )
                else:
                    continue

                safeLonDis, safeLonDisBrake = self._calculate_safe_long_dist(
                    speed_rear, speed_front
                )
                long_dist = self._longitudinal_distance(
                    pos_ego, pos_non_ego, ego_road_network_info
                )

                long_risk = self._long_risk_index(
                    safeLonDis, safeLonDisBrake, long_dist
                )

                # state.t inclusion in the dictionary (line below). To
                # track/timestamp TTC in simulation.
                if state.t not in self.safe_long_dist_dict:
                    self.safe_long_dist_dict[state.t] = []
                self.safe_long_dist_dict[state.t].append(
                    {"safe_longitudinal_distance": {f"{entity}": safeLonDis}}
                )

                if state.t not in self.safe_long_dist_brake_dict:
                    self.safe_long_dist_brake_dict[state.t] = []
                self.safe_long_dist_brake_dict[state.t].append(
                    {
                        "safe_longitudinal_distance_brake": {
                            f"{entity}": safeLonDisBrake
                        }
                    }
                )

                if state.t not in self.long_dist_dict:
                    self.long_dist_dict[state.t] = []
                self.long_dist_dict[state.t].append(
                    {"longitudinal_distance": {f"{entity}": long_dist}}
                )

                if state.t not in self.long_risk_dict:
                    self.long_risk_dict[state.t] = []
                self.long_risk_dict[state.t].append(
                    {"longitudinal_risk": {f"{entity}": long_risk}}
                )

    def get_state(self) -> float:
        """Return the current distance travelled."""
        return (
            self.safe_long_dist_dict,
            self.safe_long_dist_brake_dict,
            self.long_dist_dict,
            self.long_risk_dict,
        )


class SafeLatDistance(Metric):
    """Determines safe latitudinal distance.
    All speed and acceleration inputs must be for the lateral axis
    responseTime: time it takes rear car to react and begin braking
    speedLeft: current velocity of left car
    speedRight: current velocity of right car
    accMaxResp: max acceleration of both cars towards each other during response time
    accMinBrake: min braking of both cars
    """

    name = "safe_latitudinal_distance"

    def _reset(self, state: State) -> None:
        """Find the ego."""
        self.ego = state.scenario.ego
        self.safe_lat_dist_dict = {}
        self.safe_lat_dist_brake_dict = {}
        self.lat_dist_dict = {}
        self.lat_risk_dict = {}
        # Time it takes rear car to react and begin braking (both longitudinal and lateral).
        # https://trl.co.uk/sites/default/files/PPR313_new.pdf
        # Unit: seconds (s)
        self.responseTime = 1.5
        # Max acceleration of both cars towards each other during response time (lateral).
        # Unit: m/s^2
        self.accMaxResp = 1
        # Min braking of both cars (lateral).
        # https://doi.org/10.1007/s12544-013-0120-2
        # Unit: m/s^2
        self.accMinBrake = 1.5
        # Max braking capability of both cars (lateral).
        # Replace for ACC_MIN_BRAKE when calculating 'safeDistanceBrake'.
        # https://doi.org/10.1007/s12544-013-0120-2
        # Unit: m/s^2
        self.accMaxBrake = 4

    def _get_lane_and_lane_center_index(
        self, entity_pos, entity_road_network_information
    ):
        # Initialize variables to deduce the closest lane center coordinates to
        # entity
        min_distance = float("inf")
        final_lane_center_index = None
        final_lane_index = None

        if "Lane" not in entity_road_network_information[0]:
            # print("SafeLatDist: Silly Goose! Vehicle is out-of-bounds (no lane access).")
            return 0, 0

        for lane_index, road_network_type in enumerate(
            entity_road_network_information[0]
        ):
            if road_network_type == "Lane":
                lane = entity_road_network_information[1][lane_index]
                # Loop through lane center points to find the closest one to entity
                for lane_center_index, coords in enumerate(lane.center.coords):
                    # Compute the Euclidean distance between ego position and lane
                    # center
                    distance = abs(np.linalg.norm(entity_pos - coords))
                    # Update the closest lane center index if the distance is
                    # smaller
                    if distance < min_distance:
                        min_distance = distance
                        final_lane_center_index = lane_center_index
                        final_lane_index = lane_index

        current_lane = entity_road_network_information[1][final_lane_index]

        return current_lane, final_lane_center_index

    def _is_ego_left_or_right(self, pos1, roadinfo1, pos2, roadinfo2):
        """
        Dev Note: Safe latitudinal distance is currently limited to instances where both vehicles are in separate lanes.
        """
        (
            current_lane_vehicle_1,
            final_lane_center_index_vehicle_1,
        ) = self._get_lane_and_lane_center_index(pos1, roadinfo1)
        (
            current_lane_vehicle_2,
            final_lane_center_index_vehicle_2,
        ) = self._get_lane_and_lane_center_index(pos2, roadinfo2)

        if current_lane_vehicle_1 == 0 or current_lane_vehicle_2 == 0:
            return "out-of-bounds"

        elif current_lane_vehicle_1 != current_lane_vehicle_2:
            current_lane_center_point = np.array(
                current_lane_vehicle_1.center.coords[
                    final_lane_center_index_vehicle_1
                ]
            )
            next_lane_center_point = np.array(
                current_lane_vehicle_1.center.coords[
                    final_lane_center_index_vehicle_1 - 1
                ]
            )
            longitudinal_direction = -(
                next_lane_center_point - current_lane_center_point
            )
            relative_position = pos2 - pos1
            cross_product = np.cross(longitudinal_direction, relative_position)
            if cross_product < 0:
                return "right"
            else:
                return "left"

        else:
            return "inline"

    def _latitudinal_speed(
        self, entity_pos, entity_vel, entity_road_network_information
    ):
        (
            current_lane,
            final_lane_center_index,
        ) = self._get_lane_and_lane_center_index(
            entity_pos, entity_road_network_information
        )

        # Convert closest lane center point and next-nearest lane center to array
        # Does not matter if the next lane center point is 'ahead' or 'behind' direction of travel...
        # Because we only need the projection of the entity's velocity onto the
        # latitudinal direction (no sign needed for speed)
        current_lane_center_point = np.array(
            current_lane.center.coords[final_lane_center_index]
        )
        next_lane_center_point = np.array(
            current_lane.center.coords[final_lane_center_index - 1]
        )

        longitudinal_direction = -(
            next_lane_center_point - current_lane_center_point
        )

        latitudinal_direction = np.array(
            [-longitudinal_direction[1], longitudinal_direction[0]]
        )
        normalised_latitudinal_direction = latitudinal_direction / np.linalg.norm(
            latitudinal_direction
        )

        # Calculate longitudinal speed
        latitudinal_speed = np.dot(entity_vel, normalised_latitudinal_direction)
        # Take the absolute value
        latitudinal_speed = np.abs(latitudinal_speed)

        return latitudinal_speed

    def _latitudinal_distance(
        self, pos_ego, pos_non_ego, entity_road_network_information
    ):
        (
            current_lane,
            final_lane_center_index,
        ) = self._get_lane_and_lane_center_index(
            pos_ego, entity_road_network_information
        )

        # Convert closest lane center point and next-nearest lane center to array
        # Does not matter if the next lane center point is 'ahead' or 'behind' direction of travel...
        # Because we only need the projection of the entity's velocity onto the
        # latitudinal direction (no sign needed for speed)
        current_lane_center_point = np.array(
            current_lane.center.coords[final_lane_center_index]
        )
        next_lane_center_point = np.array(
            current_lane.center.coords[final_lane_center_index - 1]
        )

        longitudinal_direction = -(
            next_lane_center_point - current_lane_center_point
        )

        latitudinal_direction = np.array(
            [-longitudinal_direction[1], longitudinal_direction[0]]
        )
        normalised_latitudinal_direction = latitudinal_direction / np.linalg.norm(
            latitudinal_direction
        )

        # Calculate the vector between ego and non-ego vehicle positions
        relative_position = pos_non_ego - pos_ego
        # Project the relative position onto the latitudinal direction
        latitudinal_distance = np.dot(
            relative_position, normalised_latitudinal_direction
        )
        # Take the absolute value of the latitudinal distance
        latitudinal_distance = np.abs(latitudinal_distance)

        return latitudinal_distance

    def _calculate_safe_lat_dist(self, speed_left, speed_right):
        # assuming vehicles travel towards each other during response time for
        # boundary condition
        vLeft = speed_left - self.responseTime * self.accMaxResp
        vRight = speed_right + self.responseTime * self.accMaxResp
        # left vehicle move to left
        if vLeft >= 0:
            # right vehicle move to left
            if vRight >= 0:
                acc_left_min_gap = acc_right_max_gap = self.accMaxBrake
                acc_left_max_gap = acc_right_min_gap = self.accMinBrake
            # right vehicle move to right
            else:
                acc_left_min_gap = acc_right_min_gap = self.accMaxBrake
                acc_left_max_gap = acc_right_max_gap = self.accMinBrake

        # left vehicle move to right
        else:
            # right vehicle move to left
            if vRight >= 0:
                acc_left_min_gap = acc_right_min_gap = self.accMinBrake
                acc_left_max_gap = acc_right_max_gap = self.accMaxBrake
            # right vehicle move to right
            else:
                acc_left_min_gap = acc_right_max_gap = self.accMinBrake
                acc_left_max_gap = acc_right_min_gap = self.accMaxBrake

        sign_left = [1, -1][vLeft < 0]
        sign_right = [1, -1][vRight < 0]

        safeLatDis = (
            0.5 * (speed_left + vLeft) * self.responseTime
            + 0.5 * sign_left * np.power(vLeft, 2) / acc_left_min_gap
            - (
                0.5 * (speed_right + vRight) * self.responseTime
                + 0.5 * sign_right * np.power(vRight, 2) / acc_right_min_gap
            )
        )
        safeLatDisBrake = (
            0.5 * (speed_left + vLeft) * self.responseTime
            + 0.5 * sign_left * np.power(vLeft, 2) / acc_left_max_gap
            - (
                0.5 * (speed_right + vRight) * self.responseTime
                + 0.5 * sign_right * np.power(vRight, 2) / acc_right_max_gap
            )
        )

        return safeLatDis, safeLatDisBrake

    # Function to calculate the longitudinal or lateral risk index [0,1]
    def _lat_risk_index(self, safe_distance, safe_distance_brake, distance):
        """
        All inputs must me either longitudinal or lateral safeDistance: safe longitudinal/lateral distance (use
        function SafeLonDistance/SafeLatDistance) safeDistanceBrake: safe longitudinal/lateral distance under max
        braking capacity (use function SafeLonDistance/SafeLatDistance with max braking acceleration) distance: current
        longitudinal/lateral distance between cars
        """
        if safe_distance + distance > 0:
            r = 0
        elif safe_distance_brake + distance <= 0:
            r = 1
        else:
            r = 1 - (safe_distance_brake + distance) / (
                safe_distance_brake - safe_distance
            )
        return r

    def _step(self, state: State) -> None:
        """Remember, all speed and acceleration inputs must be for the longitudinal axis."""
        # Get entities within 10m of the ego, if entities exist...
        if len(state.get_entities_in_radius(*state.poses[self.ego][:2], 10)) > 1:
            for entity in state.get_entities_in_radius(
                *state.poses[self.ego][:2], 10
            ):
                if entity == self.ego:
                    continue
                if isinstance(entity, Pedestrian):
                    continue

                # Get ego vehicle pose, velocity, and lane.
                pos_ego = state.poses[self.ego][:2]
                vel_ego = state.velocities[self.ego][:2]
                ego_lane = state.get_road_info_at_entity(self.ego)
                # Get non_ego vehicle pose, velocity, and lane.
                pos_non_ego = state.poses[entity][:2]
                vel_non_ego = state.velocities[entity][:2]
                non_ego_lane = state.get_road_info_at_entity(entity)

                check = self._is_ego_left_or_right(
                    pos_ego, ego_lane, pos_non_ego, non_ego_lane
                )

                if check is None or check == "inline":
                    continue
                elif check == "right":
                    speed_right = self._latitudinal_speed(
                        pos_ego, vel_ego, ego_lane
                    )
                    speed_left = self._latitudinal_speed(
                        pos_non_ego, vel_non_ego, non_ego_lane
                    )
                elif check == "left":
                    speed_right = self._latitudinal_speed(
                        pos_non_ego, vel_non_ego, non_ego_lane
                    )
                    speed_left = self._latitudinal_speed(pos_ego, vel_ego, ego_lane)
                else:
                    continue

                safeLatDis, safeLatDisBrake = self._calculate_safe_lat_dist(
                    speed_left, speed_right
                )
                lat_dist = self._latitudinal_distance(
                    pos_ego, pos_non_ego, ego_lane
                )

                lat_risk = self._lat_risk_index(
                    safeLatDis, safeLatDisBrake, lat_dist
                )

                # state.t inclusion in the dictionary (line below). To
                # track/timestamp TTC in simulation.
                if state.t not in self.safe_lat_dist_dict:
                    self.safe_lat_dist_dict[state.t] = []
                self.safe_lat_dist_dict[state.t].append(
                    {"safe_latitudinal_distance": {f"{entity}": safeLatDis}}
                )

                if state.t not in self.safe_lat_dist_brake_dict:
                    self.safe_lat_dist_brake_dict[state.t] = []
                self.safe_lat_dist_brake_dict[state.t].append(
                    {
                        "safe_latitudinal_distance_brake": {
                            f"{entity}": safeLatDisBrake
                        }
                    }
                )

                if state.t not in self.lat_dist_dict:
                    self.lat_dist_dict[state.t] = []
                self.lat_dist_dict[state.t].append(
                    {"latitudinal_distance": {f"{entity}": lat_dist}}
                )

                if state.t not in self.lat_dist_dict:
                    self.lat_dist_dict[state.t] = []
                self.lat_dist_dict[state.t].append(
                    {"latitudinal_risk": {f"{entity}": lat_risk}}
                )

    def get_state(self) -> float:
        """Return the current distance travelled."""
        return (
            self.safe_lat_dist_dict,
            self.safe_lat_dist_brake_dict,
            self.lat_dist_dict,
            self.lat_risk_dict,
        )


# ---------- Time-Based Metrics ----------#
class TimeToCollision(Metric):
    """Measure the time-to-collision to nearest traffic participant (ego perspective)."""

    name = "time_to_collision"

    def _reset(self, state: State) -> None:
        """Reset the average speed."""
        self.ego = state.scenario.ego
        self.t = 0.0
        self.time_to_collision_dict = {}

    def _translate_polygon(self, polygon, dx, dy):
        return Polygon([(p[0] + dx, p[1] + dy) for p in polygon.exterior.coords])

    def _will_collide_check_and_ttc(
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
    ):
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
            # Check if the projected bounding boxes intersect (projected to actually overlap).
            # If no intersection, check if they're within the distance tolerance (projected to be within distance_tolerance, i.e. a near miss).
            # If either of these conditions are true, then 't' is the
            # time-to-collision (TTC).
            if (
                box1_at_t.intersects(box2_at_t)
                or box1_at_t.distance(box2_at_t) <= distance_tolerance
            ):
                return [True, t]
        return [False, None]

    def _step(self, state: State) -> None:
        """Update the average speed."""
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

                if entity.__class__.__name__ == "Pedestrian":
                    distance_tolerance = 1
                elif entity.__class__.__name__ == "Vehicle":
                    distance_tolerance = 0.1

                # Check if ego will collide with other entity.
                self.ego_time_to_collision_bool_and_value = (
                    self._will_collide_check_and_ttc(
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
                if self.ego_time_to_collision_bool_and_value[0]:
                    # state.t inclusion in the dictionary (line below). To
                    # track/timestamp TTC in simulation.
                    if state.t not in self.time_to_collision_dict:
                        self.time_to_collision_dict[state.t] = []
                    self.time_to_collision_dict[state.t].append(
                        {
                            "time_to_collision": {
                                f"{entity}": self.ego_time_to_collision_bool_and_value[
                                    1
                                ]
                            }
                        }
                    )
                else:
                    # Line below is commented-out to omit tracking TTC for entities outside certain radius of ego.
                    # self.time_to_collision_dict[state.t] = {entity: self.ego_time_to_collision_bool_and_value[0]}
                    continue

    def get_state(self) -> dict:
        """Return the current average speed."""
        return self.time_to_collision_dict


# ---------- Index-scale Criticality Metrics ----------#
class SpaceOccupancyIndex(Metric):
    """The Space Occupancy Index (SOI) defines a personal space for the ego vehicle and counts violations by other participants."""

    name = "space_occupancy_index"

    def _reset(self, state: State) -> None:
        """Reset the space occupancy index."""
        self.ego = state.scenario.ego
        self.space_occupancy_index = {}
        self.predefined_radius = 15

    def _step(self, state: State) -> None:
        """Capture the number of personal space incursions experienced by each actor."""
        # Personal space is defined by existing within a set radius.
        if (
            len(
                state.get_entities_in_radius(
                    *state.poses[self.ego][:2], self.predefined_radius
                )
            )
            > 1
        ):
            for entity in state.get_entities_in_radius(
                *state.poses[self.ego][:2], self.predefined_radius
            ):
                if entity == self.ego:
                    continue

                # state.t inclusion in the dictionary (line below). To
                # track/timestamp TTC in simulation.
                if state.t not in self.space_occupancy_index:
                    self.space_occupancy_index[state.t] = []
                self.space_occupancy_index[state.t].append(
                    {"space_occupancy_index": {f"{entity}": 1}}
                )

    def get_state(self):
        """Return the space occupancy index ."""
        return self.space_occupancy_index


# ---------- Severity Metrics ----------#
class DeltaV(Metric):
    """
    Defined as the change in velocity between pre-collision and post-collision trajectories of a vehicle.
    Will leverage inelastic collision kinematics to predict delta-v during runtime IF entities are on a collision course.
    Dev Note: This metric only considers traffic particpants/objects that are "pedestrians" (scenario_gym.entity.pedestrian.Pedestrian) or "vehicles" (scenario_gym.entity.vehicle.Vehicle) for now.
    """

    name = "delta_v"

    def _reset(self, state: State) -> None:
        """Reset the delta_v measure for each entity."""
        self.ego = state.scenario.ego
        self.delta_v_dict = {}
        self.predefined_masses = {"Vehicle": 1500, "Pedestrian": 80}

    def _translate_polygon(self, polygon, dx, dy):
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
    ):
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
            # Check if the projected bounding boxes intersect (projected to actually overlap).
            # If no intersection, check if they're within the distance tolerance
            # (projected to be within distance_tolerance, i.e. a near miss).
            if (
                box1_at_t.intersects(box2_at_t)
                or box1_at_t.distance(box2_at_t) <= distance_tolerance
            ):
                return [True, t]
        return [False, None]

    def _step(self, state: State) -> None:
        """Predict the delta_v for each entity potentially involved in a collision with ego vehicle."""
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

                if entity.__class__.__name__ == "Pedestrian":
                    distance_tolerance = 1
                elif entity.__class__.__name__ == "Vehicle":
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

                elif (
                    entity.__class__.__name__ != "Pedestrian"
                    and entity.__class__.__name__ != "Vehicle"
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

    def get_state(self):
        """Return delta_v dictionary."""
        return self.delta_v_dict


# TODOASAP - CONFLICT INDEX
class ConflictIndex(Metric):
    """
    A metric to estimate the conflict index between two agents, incorporating both
    collision probability and severity factors.

    The conflict index (CI) is calculated based on the predicted change in kinetic energy
    (ΔKe) in a hypothetical collision scenario between two agents, A1 and A2, at the time
    they enter and exit a designated conflict area. The formula for CI is:

        CI(A1, A2, CA, α, β) = (α * ΔKe) / (e^(β * PET(A1, A2, CA)))

    where:
        - PET(A1, A2, CA): The Post Encroachment Time (PET) between the agents within
          the conflict area, representing the time until potential collision.
        - α ∈ [0, 1]: A calibration factor representing the proportion of energy transfer
          from vehicle body to passengers, used to quantify collision severity.
        - β (s⁻¹): A calibration factor dependent on scenario factors (e.g., country, road
          geometry, visibility) that adjusts collision probability estimation.
        - ΔKe: The absolute change in kinetic energy of the agents before and after the
          predicted collision, based on masses, velocities, and angles.

    This metric provides an estimation of the collision likelihood and severity in a given
    conflict scenario.
    """

    name = "conflict_index"

    def _reset(self, state: State) -> None:
        self.ego = state.scenario.ego
        self.t = 0.0

    def calculate(self) -> NotImplementedError:
        """
        Placeholder method for calculating the conflict index.
        """
        raise NotImplementedError(
            "The conflict_index metric hasn't been implemented yet. Still a WIP."
        )

    def _step(self, state: State) -> None:
        self.calculate

    def get_state(self) -> None:
        return None


# ---------- Collision Check ----------#


class CollisionCheck(Metric):
    """Checks if the ego vehicle collided with any object in the scenario."""

    name = "collision_timestamp"

    def _reset(self, state: State) -> None:
        self.ego = state.scenario.ego
        self.collision_check_and_timestamp = False
        self.t = 0.0

    def _step(self, state: State) -> None:
        if not self.collision_check_and_timestamp:
            # print("Collision check is false")
            if len(state.collisions()[state.scenario.entities[0]]) > 0:
                self.collision_check_and_timestamp = state.t
                # print("collision check is true at time", state.t)

    def get_state(self) -> float:
        return self.collision_check_and_timestamp
