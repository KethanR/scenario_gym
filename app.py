#######################
# Import libraries
import os
import json
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import base64
import xml.etree.ElementTree as ET

#######################
# Page configuration
st.set_page_config(
    page_title="Scenario Metrics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

alt.themes.enable("dark")

#######################
# CSS styling
st.markdown(
    """
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state for toggle functionality
if "show_plots" not in st.session_state:
    st.session_state.show_plots = False

# Set the title as the main header
st.markdown(
    "<h1 style='text-align: center;'>Scenario Metrics Dashboard</h1>",
    unsafe_allow_html=True
)

#######################

# Paths for JSON files and video files
# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Define paths relative to the current directory
json_dir = os.path.join(current_dir, "saved_metric_evaluations")
video_dir = os.path.join(current_dir, "saved_recordings")
description_dir = os.path.join(current_dir, "tests/input_files/Scenarios/")
# Get list of available JSON files
json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

#######################
# Add a logo in the sidebar
logo = os.path.join(current_dir, "saved_images/Dept_Civ_Env_Eng_sub_brand_large_RGB_White_safe_area.png")

# # Optional: Add other sidebar elements
# st.sidebar.header("Navigation")
# st.sidebar.radio("Go to", ["Home", "Metrics", "About"])

# Sidebar
with st.sidebar:
    # st.title("📊 Scenario Metrics Dashboard")
    st.image(logo, use_container_width=True)
    selected_file = st.selectbox("Select a Scenario", json_files)

    # Load selected JSON file
    json_file_path = os.path.join(json_dir, selected_file)
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Convert the dictionary to a JSON string
    json_data = json.dumps(data, indent=4)

    # Add a download button
    st.download_button(
        label="Download JSON file",
        data=json_data,  # Pass the serialized JSON string
        file_name=selected_file,  # Name of the file
        mime="application/json"  # MIME type for JSON
    )

#######################
# Dashboard Main Panel
col = st.columns((2, 3.5, 4.2), gap="medium")


with col[0]:
#---------- Capture Unified Risk Value (Continuous and Post-Runtime) ----------# 
    st.markdown("###### Unified Risk Metric")

    # Continuous unified risk metric 
    # Initialize variables to track the maximum risk metric and associated timestep
    max_risk_value = float('-inf')
    max_timestep = None

    # Iterate through the timesteps in the data
    for timestep, metrics in data.items():
        # Ensure the key is not "Lagging Metrics Post-Runtime"
        if timestep != "Lagging Metrics Post-Runtime":
            for metric in metrics:
                if "continuous_unified_risk_metric" in metric:
                    value = metric["continuous_unified_risk_metric"]
                    if value > max_risk_value:
                        max_risk_value = value
                        max_timestep = timestep

    rounded_max_continuous_risk_value = round(max_risk_value, 2)

    st.metric(
        label="Continuous", value=rounded_max_continuous_risk_value
    )

    # Post-runtime unified risk metric 
    unified_metric_value = data["Lagging Metrics Post-Runtime"]["unified_risk_value"]
    rounded_unified_metric_value = round(unified_metric_value, 2)
    # Calculate difference between highest continuous unified risk 
    # And post-runtime unified risk
    unified_risk_difference = rounded_unified_metric_value - rounded_max_continuous_risk_value

    rounded_unified_risk_difference = round(unified_risk_difference, 2)

    st.metric(
        label="Post-Runtime", value=rounded_unified_metric_value, delta=rounded_unified_risk_difference
    )

#---------- END: Capture Unified Risk Value (Continuous and Post-Runtime) ----------#



#---------- Scenario Specifications ----------#
    
    collision_timestamp = data["Lagging Metrics Post-Runtime"]["collision_timestamp"]
    if collision_timestamp != False:
        collision_timestamp = round(collision_timestamp, 2)

    maximum_speed = data["Lagging Metrics Post-Runtime"]["ego_max_speed"]
    rounded_maximum_speed = round(maximum_speed, 2)
    average_speed = data["Lagging Metrics Post-Runtime"]["ego_avg_speed"]
    rounded_average_speed = round(average_speed, 2)

    st.markdown("###### Scenario Specifications")
    st.metric(
        label="Collision Timestamp", value=collision_timestamp
    )
    st.metric(
        label="Maximum Speed", value=rounded_maximum_speed
    )
    st.metric(
        label="Average Speed", value=rounded_average_speed
    )

#---------- END: Collision Specifications ----------#



with col[1]:
#---------- Scenario Playback Video ----------# 

    st.markdown(f"###### Scenario Playback ({selected_file.replace(".json", "")})")

    # Display video associated with the selected JSON file (always visible)
    video_file = selected_file.replace(".json", ".mp4")
    video_file_path = os.path.join(video_dir, video_file)

    # Check if the video file exists before displaying
    if os.path.exists(video_file_path):
        st.video(video_file_path)
    else:
        st.write(f"Video for {video_file_path} not found.")

#---------- END: Scenario Playback Video ----------# 


#---------- Continuous Risk Metric Plot ----------# 

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
    st.markdown("###### Unified Risk Metric Plot")
    st.pyplot(fig)

#---------- END: Continuous Risk Metric Plot ----------#  


#---------- Additional Metric Plots ----------#

    # Extract and display the values of "ego_avg_speed" and "ego_max_speed"
    lagging_metrics = data.get("Lagging Metrics Post-Runtime", {})

    # Toggle button to show/hide metric plots
    if st.button("Show/Hide Metric Plots"):
        st.session_state.show_plots = not st.session_state.show_plots  # Toggle state

    # Check if plots should be shown
    if st.session_state.show_plots:
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
                            elif "misc" in vehicle:
                                short_vehicle_label = vehicle.split(
                                    "scenario_gym.entity.misc."
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
                    label=f"{vehicle}",
                )

            ax.set_xlabel("Time (s)")
            ax.set_ylabel(f"{metric_name} Value")
            ax.set_title(f"{metric_name} over Time")
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
                    elif "misc" in vehicle:
                        short_vehicle_label = vehicle.split(
                            "scenario_gym.entity.misc."
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
            ax.set_title(f'{metric_name.replace("_", " ").capitalize()}')

            # Always show the legend for vehicle/pedestrian colors
            ax.legend(
                bars, df["Vehicle"], loc="upper right", title="Legend", fontsize="small"
            )  # Set the legend title to "Legend"

            # Display the bar chart in Streamlit
            st.pyplot(fig)

        # First plot the cumulative space occupancy bar chart at the top
        st.markdown(f"##### Cumulative Space Occupancy Bar Chart")
        plot_bar_chart("cumulative_space_occupancy")

        # Loop through each metric and create a plot for each (original functionality)
        for metric in metrics_to_plot:
            st.markdown(f"#### {metric.capitalize()} Plot")
            plot_metric(metric)

#---------- END: Additional Metric Plots ----------#


with col[2]:
#---------- Extract Max Metric Values ----------# 

    # Dictionary to store the maximum value and associated timestamp for each metric
    max_metrics = {}

    # Iterate through the timesteps in the data
    for timestep, metrics in data.items():
        if timestep != "Lagging Metrics Post-Runtime":
            for metric_dict in metrics:
                for metric_name, metric_values in metric_dict.items():
                    if isinstance(metric_values, (float, int)):  # For scalar values
                        if metric_name not in max_metrics or metric_values > max_metrics[metric_name][0]:
                            max_metrics[metric_name] = (metric_values, timestep)
                    elif isinstance(metric_values, dict):  # For values in a dictionary
                        max_value = max(metric_values.values())
                        if metric_name not in max_metrics or max_value > max_metrics[metric_name][0]:
                            max_metrics[metric_name] = (max_value, timestep)

    # Create a DataFrame from the max_metrics dictionary
    df_max = pd.DataFrame(
        [(metric, value, timestamp) for metric, (value, timestamp) in max_metrics.items()],
        columns=["Metric", "Max Value", "Timestamp"]
    )
    # Create a copy for severity metrics
    df_severity = df_max.copy()

    # List of metrics to be removed
    metrics_to_remove_max = ["delta_v", "distance_to_ego", "safe_latitudinal_distance", "safe_latitudinal_distance_brake", "latitudinal_distance", "safe_longitudinal_distance", "safe_longitudinal_distance_brake", "longitudinal_distance", "time_to_collision", "continuous_unified_risk_metric"]
    # Remove the specific metrics from the DataFrame
    df_max = df_max[~df_max["Metric"].isin(metrics_to_remove_max)]
    df_max = df_max.sort_values(by="Metric")
    
#---------- END: Extract Max Metric Values ----------#


#---------- Extract Min Metric Values ----------# 

    # Dictionary to store the minimum value and associated timestamp for each metric
    min_metrics = {}

    # Iterate through the timesteps in the data
    for timestep, metrics in data.items():
        if timestep != "Lagging Metrics Post-Runtime":
            for metric_dict in metrics:
                for metric_name, metric_values in metric_dict.items():
                    if isinstance(metric_values, (float, int)):  # For scalar values
                        if metric_name not in min_metrics or metric_values < min_metrics[metric_name][0]:
                            min_metrics[metric_name] = (metric_values, timestep)
                    elif isinstance(metric_values, dict):  # For values in a dictionary
                        min_value = min(metric_values.values())
                        if metric_name not in min_metrics or min_value < min_metrics[metric_name][0]:
                            min_metrics[metric_name] = (min_value, timestep)

    # Create a DataFrame from the min_metrics dictionary
    df_min = pd.DataFrame(
        [(metric, value, timestamp) for metric, (value, timestamp) in min_metrics.items()],
        columns=["Metric", "Min Value", "Timestamp"]
    )
    # List of metrics to be removed
    metrics_to_remove_min = ["delta_v", "space_occupancy_index", "latitudinal_risk", "longitudinal_risk", "continuous_unified_risk_metric"]
    # Remove the specific metrics from the DataFrame
    df_min = df_min[~df_min["Metric"].isin(metrics_to_remove_min)]
    df_min = df_min.sort_values(by="Metric")

#---------- END: Extract Min Metric Values ----------# 


#---------- Visualise Metric Values ----------# 

    st.markdown('###### Criticality Metrics (Max-Extreme)')
    # Visualize the DataFrame using Streamlit
    st.dataframe(
        df_max,
        column_order=("Metric", "Max Value", "Timestamp"),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Metric": st.column_config.TextColumn("Metric"),
            "Max Value": st.column_config.ProgressColumn(
                "Max Value",
                format="%.2f",
                min_value=0,
                max_value=1,
            )
        }
    )

    st.markdown('###### Criticality Metrics (Min-Extreme)')
    # Visualize the DataFrame using Streamlit
    st.dataframe(
        df_min,
        column_order=("Metric", "Min Value", "Timestamp"),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Metric": st.column_config.TextColumn("Metric"),
            "Min Value": st.column_config.ProgressColumn(
                "Min Value",
                format="%.2f",
                min_value=-50,
                max_value=10,
            )
        }
    )

    # List of metrics to be removed
    metrics_to_remove_severity = ["latitudinal_risk", "longitudinal_risk", "space_occupancy_index", "distance_to_ego", "safe_latitudinal_distance", "safe_latitudinal_distance_brake", "latitudinal_distance", "safe_longitudinal_distance", "safe_longitudinal_distance_brake", "longitudinal_distance", "time_to_collision", "continuous_unified_risk_metric"]
    # Remove the specific metrics from the DataFrame
    df_severity = df_severity[~df_severity["Metric"].isin(metrics_to_remove_severity)]

    st.markdown('###### Severity Metrics')
    # Visualize the DataFrame using Streamlit
    st.dataframe(
        df_severity,
        column_order=("Metric", "Max Value", "Timestamp"),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Metric": st.column_config.TextColumn("Metric"),
            "Max Value": st.column_config.ProgressColumn(
                "Max Value",
                format="%.2f",
                min_value=0,
                max_value=10,
            )
        }
    )


    # # Sample bounds for different categories of metrics
    # max_metrics_bounds = {
    #     "latitudinal_risk": (0, 1),
    #     "longitudinal_risk": (0, 1),
    #     "space_occupancy_index": (0, 1)
    # }

    # min_metrics_bounds = {
    #     "distance_to_ego": (0, 10),
    #     "latitudinal_distance": (0, 10),
    #     "longitudinal_distance": (0, 10),
    #     "safe_latitudinal_distance": (-50, 0),
    #     "safe_latitudinal_distance_brake": (-50, 0),
    #     "safe_longitudinal_distance": (-50, 0),
    #     "safe_longitudinal_distance_brake": (-50, 0),
    #     "time_to_collision": (0, 1)
    # }

    # severity_metrics_bounds = {
    #     "delta_v": (0, 10)
    # }

#---------- END: Visualise Metric Values ----------#
 


#---------- About Stub ----------#

    input_file = os.path.join(description_dir, selected_file.replace(".json", ".xosc"))
    # Parse the .xosc file
    tree = ET.parse(input_file)
    root = tree.getroot()

    # Find the <FileHeader> tag
    file_header = root.find('FileHeader')

    # Check if 'flavor' attribute exists
    if file_header is not None:
        # Extract flavor text, if present
        flavor_text = file_header.get('flavor', 'No flavor text found')
    
    with st.expander('Scenario Description', expanded=False):
        st.write(f'''
            - :orange[**General Description and Points of Interest**]: {flavor_text} 
            ''')

#---------- END: About Stub ----------#



#---------- About Stub ----------#

    with st.expander('About', expanded=False):
        st.write('''
            - :orange[**Criticality Metrics**]: Quantifies the probability of a collision occurring. 
            - :orange[**Severity Metrics**]: Captures the extent of damage or injury caused by a collision. 
            ''')

#---------- END: About Stub ----------#