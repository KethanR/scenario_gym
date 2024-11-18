import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Initialize session state for toggle functionality
if "show_plots" not in st.session_state:
    st.session_state.show_plots = False

# Paths for JSON files and video files
json_dir = "C:\\Users\\ketha\\Desktop\\IITS_TSL\\Repos\\scenario_gym\\saved_metric_evaluations\\"
video_dir = (
    "C:\\Users\\ketha\\Desktop\\IITS_TSL\\Repos\\scenario_gym\\saved_recordings\\"
)

# Get list of available JSON files
json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

# Streamlit dropdown to select JSON file
selected_file = st.selectbox("Select a JSON file", json_files)

# Load selected JSON file
json_file_path = os.path.join(json_dir, selected_file)
with open(json_file_path, "r") as f:
    data = json.load(f)

# Display video associated with the selected JSON file (always visible)
video_file = selected_file.replace(".json", ".mp4")
video_file_path = os.path.join(video_dir, video_file)

# Check if the video file exists before displaying
if os.path.exists(video_file_path):
    st.video(video_file_path)
else:
    st.write(f"Video for {selected_file} not found.")

# Toggle button to show/hide metric plots
if st.button("Show/Hide Metric Plots"):
    st.session_state.show_plots = not st.session_state.show_plots  # Toggle state

# Check if plots should be shown
if st.session_state.show_plots:
    # Extract and display the values of "ego_avg_speed" and "ego_max_speed"
    lagging_metrics = data.get("Lagging Metrics Post-Runtime", {})

    # Define the list of metrics to plot (existing functionality)
    # metrics_to_plot = ["time_to_collision", "longitudinal_distance", "safe_longitudinal_distance", "safe_longitudinal_distance_brake", "longitudinal_risk", "latitudinal_distance", "safe_latitudinal_distance", "safe_latitudinal_distance_brake", "latitudinal_risk", "delta_v"]
    metrics_to_plot = [
        "time_to_collision",
        "distance_to_ego",
        "longitudinal_risk",
        "latitudinal_risk",
        "delta_v",
    ]

    # Function to extract and plot each metric
    def plot_metric(metric_name):
        vehicles_data = {}
        time_stamps_set = set()  # To track unique timestamps across all vehicles

        # Extract time stamps and metric values for all vehicles
        for time, metrics in data.items():
            for metric in metrics:
                if metric_name in metric:
                    try:
                        time_stamps_set.add(
                            float(time)
                        )  # Collect all unique timestamps
                    except:
                        continue

                    # Loop through all vehicles/pedestrians under the current metric
                    for vehicle, value in metric[metric_name].items():
                        # Simplify the vehicle/pedestrian label
                        if "vehicle" in vehicle:
                            short_vehicle_label = vehicle.split(
                                "scenario_gym.entity.vehicle."
                            )[-1].replace(">", "")
                        elif "pedestrian" in vehicle:
                            short_vehicle_label = vehicle.split(
                                "scenario_gym.entity.pedestrian."
                            )[-1].replace(">", "")
                        else:
                            short_vehicle_label = (
                                vehicle  # Fallback for other object types
                            )

                        if short_vehicle_label not in vehicles_data:
                            vehicles_data[
                                short_vehicle_label
                            ] = {}  # Store time-value pairs for each vehicle
                        vehicles_data[short_vehicle_label][
                            float(time)
                        ] = value  # Store the value for this vehicle at this time
                    break  # Exit loop once the metric is found for this time step

        # Sort time stamps
        time_stamps = sorted(list(time_stamps_set))

        # Create a DataFrame with time as the index
        df = pd.DataFrame({"Time": time_stamps})

        # Add each vehicle's values to the DataFrame, filling missing values with None
        for vehicle, time_value_dict in vehicles_data.items():
            df[vehicle] = [time_value_dict.get(t, None) for t in time_stamps]

        # Plot the values for all vehicles
        fig, ax = plt.subplots()
        for vehicle in vehicles_data.keys():
            ax.plot(
                df["Time"],
                df[vehicle],
                marker="o",
                label=f"{metric_name} for {vehicle}",
            )

        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{metric_name} Value")
        ax.set_title(f"{metric_name} Over Time for All Vehicles")
        ax.legend(
            loc="upper right", title="Legend", fontsize="small"
        )  # Set the legend title to "Legend"
        ax.grid(True)

        # Display the chart in Streamlit
        st.pyplot(fig)

    # Function to extract and plot each metric as a bar chart, with color legend instead of x-axis labels
    def plot_bar_chart(metric_name):
        vehicles_data = {}

        # Extract the values for each vehicle under the bar chart metric
        if metric_name in lagging_metrics:
            for vehicle, value in lagging_metrics[metric_name].items():
                # Simplify the vehicle/pedestrian label
                if "vehicle" in vehicle:
                    short_vehicle_label = vehicle.split("scenario_gym.entity.vehicle.")[
                        -1
                    ].replace(">", "")
                elif "pedestrian" in vehicle:
                    short_vehicle_label = vehicle.split(
                        "scenario_gym.entity.pedestrian."
                    )[-1].replace(">", "")
                else:
                    short_vehicle_label = vehicle  # Fallback for other object types

                vehicles_data[short_vehicle_label] = value

        # Convert the vehicle data to a DataFrame
        df = pd.DataFrame(
            list(vehicles_data.items()), columns=["Vehicle", f"{metric_name} Value"]
        )

        # Plot the bar chart with consistent structure
        fig, ax = plt.subplots()

        # Check if there's only one vehicle to adjust the presentation
        if len(df) == 1:
            bars = ax.bar(
                range(len(df)),
                df[f"{metric_name} Value"],
                color=plt.cm.tab10.colors[: len(df)],
                width=0.2,
            )
            ax.set_xlim(-0.5, 0.5)  # Center the single bar
        else:
            bars = ax.bar(
                range(len(df)),
                df[f"{metric_name} Value"],
                color=plt.cm.tab10.colors[: len(df)],
            )

        # Add labels on top of the bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval,
                round(yval, 2),
                va="bottom",
                ha="center",
            )

        # Set the x-axis label and add a legend with vehicle names and colors
        ax.set_xlabel("Vehicles")
        ax.set_xticks([])  # Remove x-axis labels
        ax.set_ylabel(f"{metric_name} Value")
        ax.set_title(f'{metric_name.replace("_", " ").capitalize()} for All Vehicles')

        # Always show the legend for vehicle/pedestrian colors
        ax.legend(
            bars, df["Vehicle"], loc="upper right", title="Legend", fontsize="small"
        )  # Set the legend title to "Legend"

        # Display the bar chart in Streamlit
        st.pyplot(fig)

    # First plot the cumulative space occupancy bar chart at the top
    st.write(f"## Cumulative Space Occupancy Bar Chart")
    plot_bar_chart("cumulative_space_occupancy")

    # Loop through each metric and create a plot for each (original functionality)
    for metric in metrics_to_plot:
        st.write(f"## {metric.capitalize()} Plot")
        plot_metric(metric)

    # Extract timesteps and continuous_unified_risk_metric values
    timesteps = []
    risk_metrics = []

    for time, metrics in data.items():
        # Only process if it is a dictionary of metrics (exclude other keys like 'Lagging Metrics Post-Runtime')
        if isinstance(metrics, list):
            # Find 'continuous_unified_risk_metric' in each timestep's metrics
            for metric in metrics:
                if "continuous_unified_risk_metric" in metric:
                    timesteps.append(float(time))  # Convert timestamp to float
                    risk_metrics.append(metric["continuous_unified_risk_metric"])
                    break  # Exit once found for this timestep

    # Create a DataFrame from the extracted data
    df = pd.DataFrame(
        {"Timestep": timesteps, "Continuous Unified Risk Metric": risk_metrics}
    )

    # Plot the continuous unified risk metric over timesteps
    fig, ax = plt.subplots()
    ax.plot(df["Timestep"], df["Continuous Unified Risk Metric"], marker="o", color="b")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Continuous Unified Risk Metric")
    ax.set_title("Continuous Unified Risk Metric Over Time")
    ax.grid(True)

    # Display the plot in Streamlit
    st.write("## Continuous Unified Risk Metric Over Time")
    st.pyplot(fig)

    # Define the remaining bar charts to show (for lagging metrics)
    # bar_chart_metrics = ["time_exposed_time_to_collision", "time_integrated_time_to_collision"]

    # Loop through the remaining lagging metrics and create a bar chart for each
    # for metric in bar_chart_metrics:
    #     st.write(f"## {metric.replace('_', ' ').capitalize()} Bar Chart")
    #     plot_bar_chart(metric)
