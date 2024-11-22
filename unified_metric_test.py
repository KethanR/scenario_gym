import os
import json
import numpy as np
import pandas as pd
import re
from scenario_gym import ScenarioGym
from scenario_gym.metrics import (
    DistanceToEgo,
    EgoAvgSpeed,
    EgoMaxSpeed,
    TimeToCollision,
    SafeLongDistance,
    SafeLatDistance,
    SpaceOccupancyIndex,
    DeltaV,
    CollisionTimestamp,
)
from scenario_gym.xosc_interface import import_scenario
# from scenario_gym.metrics.utils_and_callbacks import LaneAndLaneCenter
# state_callbacks=[LaneAndLaneCenter()]


# Instantiating variables
# openscenario_file = "C:\\Users\\ketha\\Desktop\\IITS_TSL\\Repos\\scenario_gym\\tests\\input_files\\Scenarios\\1518e754-318f-4847-8a30-2dce552b4504.xosc"
openscenario_file = "C:\\Users\\ketha\\Desktop\\IITS_TSL\\Repos\\scenario_gym\\tests\\input_files\\Scenarios\\a2281876-e0b4-4048-a08a-1ce69f94c085.xosc"
# openscenario_file = "C:\\Users\\ketha\\Desktop\\IITS_TSL\\Repos\\scenario_gym\\tests\\input_files\\Scenarios\\5c5188e0-715a-4dd2-a6b2-b3c96b52d608.xosc"
default_save_recordings_path = (
    "C:\\Users\\ketha\\Desktop\\IITS_TSL\\Repos\\scenario_gym\\saved_recordings\\"
)
default_save_metrics_path = "C:\\Users\\ketha\\Desktop\\IITS_TSL\\Repos\\scenario_gym\\saved_metric_evaluations\\"
plot_bool = False
render_bool = True
print_metrics_bool = True
unify_metrics_bool = True
loop_bool = True


def load_and_rollout_scenario(
    openscenario_file,
    default_save_recordings_path,
    default_save_metrics_path,
    render_bool,
    print_metrics_bool,
    unify_metrics_bool,
    plot_bool,
) -> None:
    # String handling for saved .mp4
    file_name, file_extension = os.path.splitext(openscenario_file)

    # Simulating scenario
    gym = ScenarioGym(
        metrics=[
            TimeToCollision(),
            DistanceToEgo(),
            SafeLongDistance(),
            SafeLatDistance(),
            SpaceOccupancyIndex(),
            DeltaV(),
            EgoAvgSpeed(),
            EgoMaxSpeed(),
            CollisionTimestamp(),
        ]
    )
    # gym = ScenarioGym()
    gym.load_scenario(openscenario_file)
    gym.rollout(
        render=render_bool,
        video_path=default_save_recordings_path + os.path.basename(file_name) + ".mp4",
    ) 

    # Print risk metric values
    if print_metrics_bool:
        metrics = gym.get_metrics()
        metrics = cleanup_and_lagging_metrics(metrics)

        if unify_metrics_bool:
            if metrics["Lagging Metrics Post-Runtime"]["collision_timestamp"] != False:
                unified_risk_value = 1
                _continuous_unified_risk_metric(metrics)
            else:
                unified_risk_value = _unified_risk_metric(metrics, False, None)
                _continuous_unified_risk_metric(metrics)

            metrics["Lagging Metrics Post-Runtime"][
                "unified_risk_value"
            ] = unified_risk_value

        # Saving metrics json file and storing in saved_metric_evaluations
        save_metrics_path = (
            default_save_metrics_path + os.path.basename(file_name) + ".json"
        )
        with open(save_metrics_path, "w") as json_file:
            json.dump(metrics, json_file, indent=4)

    # Show static trajectory plot
    if plot_bool:
        scenario = import_scenario(openscenario_file)
        scenario.plot()


def _get_metric_dataframe(json_data):
    # Dictionary to hold dataframes for each metric
    metric_dataframes = {}

    # Loop through the JSON entries, ignoring the "Lagging Metrics Post-Runtime"
    for timestamp, metrics_list in json_data.items():
        if timestamp == "Lagging Metrics Post-Runtime":
            continue  # Skip lagging metrics

        # Process each metric for this timestamp
        for metric in metrics_list:
            for metric_name, vehicle_data in metric.items():
                # Ensure the dataframe for this metric exists
                if metric_name not in metric_dataframes:
                    metric_dataframes[metric_name] = []

                # Create a dictionary to hold the data for this timestamp
                row = {"timestamp": timestamp}

                # For each vehicle, add the value to the row
                if isinstance(vehicle_data, dict):
                    for vehicle, value in vehicle_data.items():
                        column_name = f"{vehicle}"  # Column named by vehicle
                        row[column_name] = value

                # Append this row to the corresponding metric dataframe
                metric_dataframes[metric_name].append(row)

    # Convert lists of rows to DataFrames, setting 'timestamp' as index
    for metric_name, rows in metric_dataframes.items():
        df = pd.DataFrame(rows)
        df.set_index("timestamp", inplace=True)
        metric_dataframes[metric_name] = df

    return metric_dataframes


def _unified_risk_metric(metrics, continuous_bool, timestep):
    # Convert the JSON to separate Pandas DataFrames for each metric
    # Does not handle "Lagging metrics".
    # This will be handled after the loop below.
    dataframes_dict = _get_metric_dataframe(metrics)
    metrics_list = []

    # Loop through each metric and its corresponding DataFrame
    for metric_name, df in dataframes_dict.items():
        if continuous_bool == True:
            df = df[df.index <= timestep]

        if metric_name == "delta_v":
            delta_v_term = _delta_v_term(df)
            metrics_list.append(delta_v_term)
            print("delta_v_term:", delta_v_term)

        if metric_name == "space_occupancy_index":
            space_occupancy_index_term = _space_occupancy_index_term(df)
            metrics_list.append(space_occupancy_index_term)
            print("space_occupancy_index_term:", space_occupancy_index_term)

        if metric_name == "time_to_collision":
            time_to_collision_term = _time_to_collision_term(df)
            metrics_list.append(time_to_collision_term)
            print("time_to_collision_term:", time_to_collision_term)

        if metric_name == "distance_to_ego":
            distance_to_ego_term = _distance_to_ego_term(df)
            metrics_list.append(distance_to_ego_term)
            print("distance_to_ego_term", distance_to_ego_term)

        if metric_name == "latitudinal_risk":
            latitudinal_risk_term = _latitudinal_or_longitudinal_risk_term(df)
            metrics_list.append(latitudinal_risk_term)
            print("latitudinal_risk_term:", latitudinal_risk_term)

        if metric_name == "longitudinal_risk":
            longitudinal_risk_term = _latitudinal_or_longitudinal_risk_term(df)
            metrics_list.append(longitudinal_risk_term)
            print("longitudinal_risk_term:", longitudinal_risk_term)

    # Handling "lagging metrics" below
    if continuous_bool != True:
        lagging_metrics_term = _lagging_metrics_term(metrics)
        metrics_list.append(lagging_metrics_term)

    weights = np.ones(len(metrics_list))
    weights = weights / len(weights)

    unified_risk_value = np.dot(weights, np.array(metrics_list))
    if np.isnan(unified_risk_value):
        unified_risk_value = 0
    print("Unified_Risk_Value", unified_risk_value)

    return unified_risk_value


def _continuous_unified_risk_metric(metrics):
    for timestep in metrics.keys():
        # Ensure we're not adding the metric to non-time-step keys like "Lagging Metrics Post-Runtime"
        if isinstance(metrics[timestep], list):
            continuous_unified_risk_metric_value = _unified_risk_metric(
                metrics, True, timestep
            )
            # Append the "continuous_unified_risk_metric" entry at the end of each timestep's metrics list
            metrics[timestep].append(
                {"continuous_unified_risk_metric": continuous_unified_risk_metric_value}
            )


def _delta_v_term(term_specific_dataframe):
    # Sample threshold values
    pedestrian_threshold = 5
    vehicle_threshold = 10

    # Create a new DataFrame to store the normalized values
    normalized_values = {}

    for column in term_specific_dataframe.columns:
        if "pedestrian" in column:
            # Apply pedestrian threshold
            normalized_values[column] = _proportion_above_threshold(
                term_specific_dataframe[column], pedestrian_threshold
            )
        elif "vehicle" in column:
            # Apply vehicle threshold
            normalized_values[column] = _proportion_above_threshold(
                term_specific_dataframe[column], vehicle_threshold
            )
        else:
            normalized_values[column] = 0

    # Convert the normalized values to a DataFrame for easier handling
    normalized_df = pd.DataFrame(
        list(normalized_values.items()), columns=["Entity", "Normalized Value"]
    )

    if len(normalized_df) > 1:
        delta_v_term = normalized_df["Normalized Value"].mean()
    else:
        delta_v_term = normalized_df["Normalized Value"].iloc[0]

    return delta_v_term


def _space_occupancy_index_term(term_specific_dataframe):
    # Check if "vehicle" or "pedestrian" are in the list
    has_vehicle = any("vehicle" in item for item in term_specific_dataframe.columns)
    has_pedestrian = any(
        "pedestrian" in item for item in term_specific_dataframe.columns
    )

    if has_vehicle and has_pedestrian:
        if len(term_specific_dataframe.columns) > 5:
            space_occupancy_index_term = 1.0
        else:
            space_occupancy_index_term = 0.5

    else:
        if len(term_specific_dataframe.columns) > 5:
            space_occupancy_index_term = 0.5
        else:
            space_occupancy_index_term = 0.0

    return space_occupancy_index_term


def _time_to_collision_term(term_specific_dataframe):
    # Sample threshold values
    pedestrian_threshold = 1.5
    vehicle_threshold = 1
    # Create a new DataFrame to store the normalized values
    normalized_values = {}

    for column in term_specific_dataframe.columns:
        if "pedestrian" in column:
            # Apply pedestrian threshold
            normalized_values[column] = _proportion_below_threshold(
                term_specific_dataframe[column], pedestrian_threshold
            )
        elif "vehicle" in column:
            # Apply vehicle threshold
            normalized_values[column] = _proportion_below_threshold(
                term_specific_dataframe[column], vehicle_threshold
            )
        else:
            normalized_values[column] = 0

    # Convert the normalized values to a DataFrame for easier handling
    normalized_df = pd.DataFrame(
        list(normalized_values.items()), columns=["Entity", "Normalized Value"]
    )
    if len(normalized_df) > 1:
        time_to_collision_term = normalized_df["Normalized Value"].mean()
    else:
        time_to_collision_term = normalized_df["Normalized Value"].iloc[0]

    return time_to_collision_term


def _distance_to_ego_term(term_specific_dataframe):
    results = {}
    gradient_df = _calculate_gradients(term_specific_dataframe)
    # Distance and gradient thresholds for vehicles and pedestrians
    vehicle_thresholds = (4, 2)
    pedestrian_thresholds = (10, 1)

    # Iterate over each original column (excluding the gradient columns)
    for col in gradient_df.columns:
        if not col.endswith("_gradient"):
            # Determine the thresholds based on whether it's a vehicle or pedestrian column
            if "vehicle" in col.lower():
                value_threshold, gradient_threshold = vehicle_thresholds
            elif "pedestrian" in col.lower():
                value_threshold, gradient_threshold = pedestrian_thresholds
            else:
                # Skip if the column doesn't match the naming pattern for vehicle or pedestrian
                continue

            # Get the corresponding gradient column
            gradient_col = f"{col}_gradient"

            # Check if the gradient column exists in the DataFrame
            if gradient_col in gradient_df.columns:
                # Find the indices where the conditions are met
                condition = (gradient_df[col] < value_threshold) & (
                    gradient_df[gradient_col].abs() > gradient_threshold
                )

                # Get the timestamps (index values) where the condition is met
                matching_timestamps = gradient_df.index[condition].tolist()

                # If there are any matching timestamps, add them to the results dictionary
                if matching_timestamps:
                    results[col] = matching_timestamps

    has_vehicle = any("vehicle" in key.lower() for key in results)
    has_pedestrian = any("pedestrian" in key.lower() for key in results)

    if has_vehicle and has_pedestrian:
        distance_to_ego_term = 1  # Both vehicle and pedestrian results present
    elif has_vehicle:
        distance_to_ego_term = 0.5  # Only vehicle results present
    elif has_pedestrian:
        distance_to_ego_term = 1  # Only pedestrian results present
    else:
        distance_to_ego_term = 0

    return distance_to_ego_term


def _latitudinal_or_longitudinal_risk_term(term_specific_dataframe):
    # Initialize a list to store the average of non-zero values for each column
    average_non_zero_list = []

    # Iterate over each column in the DataFrame
    for column in term_specific_dataframe.columns:
        # Filter out the zero values for the current column
        non_zero_values = term_specific_dataframe[column][
            term_specific_dataframe[column] != 0
        ]

        # Calculate the average of non-zero values for the current column
        if len(non_zero_values) > 0:
            average_non_zero = non_zero_values.mean()
            average_non_zero_list.append(average_non_zero)

    # Calculate the average of the averages
    if len(average_non_zero_list) > 0:
        latitudinal_or_longitudinal_risk_term = sum(average_non_zero_list) / len(
            average_non_zero_list
        )
    else:
        latitudinal_or_longitudinal_risk_term = 0

    return latitudinal_or_longitudinal_risk_term


# TODOASAP
def _lagging_metrics_term(metrics):
    # Initialisation
    TET_term = 0
    TIT_term = 0
    CSO_term = 0
    avg_term = 0
    max_term = 0
    lagging_metrics_term = 0

    # Thresholds (set semi-arbitrary for now)
    threshold_avg_speed = 7
    threshold_max_speed = 10
    threshold_TET = 0.4
    threshold_TIT = 1
    threshold_CSO = 5

    # Retrieve lagging metrics
    ego_avg_speed = metrics["Lagging Metrics Post-Runtime"]["ego_avg_speed"]
    ego_max_speed = metrics["Lagging Metrics Post-Runtime"]["ego_max_speed"]
    time_exposed_time_to_collision_dict = metrics["Lagging Metrics Post-Runtime"][
        "time_exposed_time_to_collision"
    ]
    time_integrated_time_to_collision_dict = metrics["Lagging Metrics Post-Runtime"][
        "time_integrated_time_to_collision"
    ]
    cumulative_space_occupancy_dict = metrics["Lagging Metrics Post-Runtime"][
        "cumulative_space_occupancy"
    ]

    # Handle avg and max speed
    if ego_avg_speed > threshold_avg_speed:
        avg_term = 1
    else:
        avg_term = 0
    if ego_max_speed > threshold_max_speed:
        max_term = 1
    else:
        max_term = 0

    # Count values above TET and TIT threshold, and handle TET+TIT
    TET_count_above_threshold = sum(
        1
        for value in time_exposed_time_to_collision_dict.values()
        if value > threshold_TET
    )
    TIT_count_above_threshold = sum(
        1
        for value in time_integrated_time_to_collision_dict.values()
        if value > threshold_TIT
    )
    if TET_count_above_threshold > 1:
        TET_term = 1
    else:
        TET_term = 0
    if TIT_count_above_threshold > 1:
        TIT_term = 1
    else:
        TIT_term = 0

    # Handle CSO
    CSO_count_above_threshold = sum(
        1 for value in cumulative_space_occupancy_dict.values() if value > threshold_CSO
    )
    if CSO_count_above_threshold > 1:
        CSO_term = 1
    else:
        CSO_term = 0

    lagging_metrics_term = (TET_term + TIT_term + CSO_term + avg_term + max_term) / 5

    return lagging_metrics_term


# Function to compute the proportion of values above a threshold
def _proportion_above_threshold(lst, threshold):
    if len(lst) == 0:
        return 0
    return sum(1 for x in lst if x > threshold) / len(lst)


# Function to compute the proportion of values above a threshold
def _proportion_below_threshold(lst, threshold):
    if len(lst) == 0:
        return 0
    return sum(1 for x in lst if x < threshold) / len(lst)


def _calculate_gradients(df):
    df = df.copy()
    # Access the index as the timestamp
    timestamp = df.index
    # Iterate over each column in the DataFrame
    for col in df.columns:
        # Compute the difference between consecutive values in the column
        col_diff = df[col].diff()
        # Compute the difference between consecutive timestamps from the index
        timestamp_diff = timestamp.to_series().diff()
        # Calculate the gradient by dividing column difference by timestamp difference
        gradient = col_diff / timestamp_diff
        # Append the gradient as a new column in the original DataFrame
        df[f"{col}_gradient"] = gradient

    return df


def cleanup_and_lagging_metrics(data):
    # Initialize a dictionary to store TET and TIT for each entity.
    entity_tet_duration = {}
    entity_tit_duration = {}
    space_occupancy_index_cumulative = {}

    # Extract and sort the keys that are valid float timestamps.
    timestamps = sorted([float(ts) for ts in data if isinstance(ts, (int, float))])

    # Iterate over the sorted timestamps.
    for i in range(1, len(timestamps)):
        current_timestamp = timestamps[i]
        previous_timestamp = timestamps[i - 1]

        # Calculate the time difference between consecutive timestamps.
        time_diff = current_timestamp - previous_timestamp

        # Extract the corresponding simulation data for the current timestamp.
        current_simulation_data = data[current_timestamp]

        # Iterate through the dictionaries in the list for each timestamp.
        for metric_entry in current_simulation_data:
            if "time_to_collision" in metric_entry:
                # Extract the time_to_collision dictionary
                ttc_data = metric_entry["time_to_collision"]

                # Iterate through the TTC entries and extract their TTC values.
                for entity, ttc in ttc_data.items():
                    # Split the string by spaces and dots
                    parts = entity.split(".")

                    # Get the second-to-last part before 'object'
                    if len(parts) > 1:
                        entity_type = parts[-2]
                        class_string = entity_type.lower()  # Return in lowercase

                    if class_string == "pedestrian":
                        threshold_ttc = 1
                    elif class_string == "vehicle":
                        threshold_ttc = 0.5
                    else:
                        # Placeholder value for now.
                        threshold_ttc = 0.5

                    if (
                        0 < ttc < threshold_ttc
                    ):  # Check if TTC is less than the threshold.
                        # Calculate TIT and TET for this entity
                        if entity not in entity_tet_duration:
                            entity_tet_duration[entity] = 0
                        entity_tet_duration[entity] += time_diff
                        if entity not in entity_tit_duration:
                            entity_tit_duration[entity] = 0
                        entity_tit_duration[entity] += (
                            (1 / ttc) - (1 / threshold_ttc)
                        ) * time_diff

            if "space_occupancy_index" in metric_entry:
                space_occupancy_index_data = metric_entry["space_occupancy_index"]

                for entity, _blank in space_occupancy_index_data.items():
                    if entity not in space_occupancy_index_cumulative:
                        space_occupancy_index_cumulative[entity] = 0
                    space_occupancy_index_cumulative[entity] += time_diff

    # Check if "Lagging Metrics Post-Runtime" exists in the data
    if "Lagging Metrics Post-Runtime" not in data:
        # If not, create it as an empty dictionary
        data["Lagging Metrics Post-Runtime"] = {}

    # Merge TET, TIT, and Cumulative Space Occupancy values into "Lagging Metrics Post-Runtime"
    data["Lagging Metrics Post-Runtime"].update(
        {
            "time_exposed_time_to_collision": entity_tet_duration,
            "time_integrated_time_to_collision": entity_tit_duration,
            "cumulative_space_occupancy": space_occupancy_index_cumulative,
        }
    )

    return data


if loop_bool:
    # Loop through all files in the directory
    for filename in os.listdir(
        "C:\\Users\\ketha\\Desktop\\IITS_TSL\\Repos\\scenario_gym\\tests\\input_files\\Scenarios"
    ):
        file_path = os.path.join(
            "C:\\Users\\ketha\\Desktop\\IITS_TSL\\Repos\\scenario_gym\\tests\\input_files\\Scenarios",
            filename,
        )
        if os.path.isfile(file_path):
            load_and_rollout_scenario(
                file_path,
                default_save_recordings_path,
                default_save_metrics_path,
                render_bool,
                print_metrics_bool,
                unify_metrics_bool,
                plot_bool,
            )
            print("Rolled out:", filename)
else:
    load_and_rollout_scenario(
        openscenario_file,
        default_save_recordings_path,
        default_save_metrics_path,
        render_bool,
        print_metrics_bool,
        unify_metrics_bool,
        plot_bool,
    )
