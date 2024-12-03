import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go



def plot_spi_over_time(filtered_df, selected_teams):
    """
    Plot the Soccer Power Index (SPI) over time for selected teams.

    Args:
        filtered_df (DataFrame): The filtered dataset.
        selected_teams (list): List of selected teams to include in the visualization.
    """
    if filtered_df.empty or not selected_teams:
        st.warning("No data matches the selected filters or no teams selected.")
        return

    # Filter for the selected teams
    team_df = filtered_df[
        (filtered_df["team1"].isin(selected_teams)) | (filtered_df["team2"].isin(selected_teams))
    ]

    # Melt SPI columns for easier plotting
    melted_spi = pd.melt(
        team_df,
        id_vars=["Match Date", "team1", "team2"],
        value_vars=["spi1", "spi2"],
        var_name="SPI Type",
        value_name="SPI",
    )
    melted_spi["Team"] = melted_spi.apply(
        lambda row: row["team1"] if row["SPI Type"] == "spi1" else row["team2"], axis=1
    )
    melted_spi = melted_spi[melted_spi["Team"].isin(selected_teams)]

    # Plot SPI over time
    fig = px.scatter(
        melted_spi,
        x="Match Date",
        y="SPI",
        color="Team",
        trendline="ols",
        title="SPI Over Time",
        labels={"SPI": "Soccer Power Index", "Match Date": "Date", "Team": "Team"},
    )
    st.plotly_chart(fig, use_container_width=True)
    
    
    
def plot_importance_over_time(filtered_df, selected_teams):
    """
    Plot the importance of matches over time for selected teams.

    Args:
        filtered_df (DataFrame): The filtered dataset.
        selected_teams (list): List of selected teams to include in the visualization.
    """
    if filtered_df.empty or not selected_teams:
        st.warning("No data matches the selected filters or no teams selected.")
        return

    # Filter for the selected teams
    team_df = filtered_df[
        (filtered_df["team1"].isin(selected_teams)) | (filtered_df["team2"].isin(selected_teams))
    ]

    # Add a unified "importance" column
    team_df["importance"] = team_df.apply(
        lambda row: row["importance1"] if row["team1"] in selected_teams else row["importance2"], axis=1
    )
    team_df["Team"] = team_df.apply(
        lambda row: row["team1"] if row["team1"] in selected_teams else row["team2"], axis=1
    )

    # Aggregate importance over time
    team_df["Match Month"] = team_df["Match Date"].dt.to_period("M")
    importance_data = team_df.groupby(["Match Month", "Team"])["importance"].mean().reset_index()
    importance_data["Match Month"] = importance_data["Match Month"].astype(str)

    # Plot importance trends
    fig = px.line(
        importance_data,
        x="Match Month",
        y="importance",
        color="Team",
        title="Match Importance Over Time",
        labels={"importance": "Importance (0-100)", "Match Month": "Month", "Team": "Team"},
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_prediction_accuracy_over_time(filtered_df, selected_teams):
    """
    Plot the accuracy of goal predictions over time as the difference between projected and actual goals.
    """
    if filtered_df.empty or not selected_teams:
        st.warning("No data matches the selected filters or no teams selected.")
        return

    # Sidebar toggle for trend line
    with st.sidebar.expander("Prediction Accuracy Graph Options"):
        show_trendline = st.checkbox("Show Trend Line", value=True, key="accuracy_trendline")

    # Prepare data for selected teams
    team_data = pd.DataFrame({
        "Match Date": pd.concat([filtered_df["Match Date"], filtered_df["Match Date"]]),
        "Team": pd.concat([filtered_df["team1"], filtered_df["team2"]]),
        "Prediction Difference": pd.concat(
            [filtered_df["proj_score1"] - filtered_df["score1"], 
             filtered_df["proj_score2"] - filtered_df["score2"]]
        ),
        "Team Type": ["Home"] * len(filtered_df) + ["Away"] * len(filtered_df)
    }).reset_index(drop=True)

    # Filter to selected teams only
    team_data = team_data[team_data["Team"].isin(selected_teams)]

    # Scatter plot of prediction accuracy over time
    trendline_option = "ols" if show_trendline else None
    fig = px.scatter(
        team_data,
        x="Match Date",
        y="Prediction Difference",
        color="Team",
        facet_col="Team Type",
        trendline=trendline_option,
        labels={"Prediction Difference": "Prediction Accuracy (Difference in goals)", "Match Date": "Date"},
        title="Prediction Accuracy Over Time",
        hover_data=["Team", "Team Type", "Prediction Difference"],
        height=600,
    )

    st.plotly_chart(fig, use_container_width=True)



def plot_win_percentage_vs_spi(filtered_df, selected_teams):
    """
    Scatter plot showing win percentage vs. SPI for home and away teams.
    """
    if filtered_df.empty or not selected_teams:
        st.warning("No data matches the selected filters or no teams selected.")
        return

    # Prepare data
    win_data = pd.DataFrame({
        "SPI": pd.concat([filtered_df["spi1"], filtered_df["spi2"]]),
        "Win Percentage": pd.concat([
            (filtered_df["score1"] > filtered_df["score2"]).astype(int) * 100,
            (filtered_df["score2"] > filtered_df["score1"]).astype(int) * 100
        ]),
        "Team": pd.concat([filtered_df["team1"], filtered_df["team2"]]),
        "Team Type": ["Home"] * len(filtered_df) + ["Away"] * len(filtered_df)
    }).reset_index(drop=True)

    # Filter to selected teams only
    win_data = win_data[win_data["Team"].isin(selected_teams)]

    # Scatter plot
    fig = px.scatter(
        win_data,
        x="SPI",
        y="Win Percentage",
        color="Team",
        facet_col="Team Type",
        labels={"SPI": "SPI Rating", "Win Percentage": "Win %"},
        title="Win % vs. SPI Factor (Home & Away)",
        hover_data=["Team", "Team Type", "SPI", "Win Percentage"],
        height=600,
    )

    st.plotly_chart(fig, use_container_width=True)
    
    
def plot_win_percentage_vs_importance(filtered_df, selected_teams):
    """
    Scatter plot showing win percentage vs. importance for home and away teams.
    """
    if filtered_df.empty or not selected_teams:
        st.warning("No data matches the selected filters or no teams selected.")
        return

    # Prepare data
    filtered_df = filtered_df[
        (filtered_df["team1"].isin(selected_teams)) | (filtered_df["team2"].isin(selected_teams))
    ]

    # Calculate win percentage by importance levels
    home_data = filtered_df.groupby("importance1").apply(
        lambda group: (group["score1"] > group["score2"]).mean() * 100
    ).reset_index(name="Win Percentage")

    away_data = filtered_df.groupby("importance2").apply(
        lambda group: (group["score2"] > group["score1"]).mean() * 100
    ).reset_index(name="Win Percentage")

    home_data["Importance"] = home_data["importance1"]
    away_data["Importance"] = away_data["importance2"]

    combined_data = pd.concat([home_data, away_data], ignore_index=True)

    # Scatter plot
    fig = px.scatter(
        combined_data,
        x="Importance",
        y="Win Percentage",
        trendline="ols",
        labels={"Importance": "Match Importance", "Win Percentage": "Win %"},
        title="Win % vs. Importance",
        hover_data={"Importance": True, "Win Percentage": True},
        height=600,
    )

    st.plotly_chart(fig, use_container_width=True)




def create_gauge_chart(label, value, color, width=150, height=150):
    """
    Create a single gauge chart with Plotly and allow dynamic sizing.

    Args:
        label (str): The label for the gauge.
        value (float): The value to display on the gauge.
        color (str): The color of the gauge.
        width (int): Width of the gauge in pixels. Default is 150.
        height (int): Height of the gauge in pixels. Default is 150.

    Returns:
        go.Figure: A Plotly gauge chart figure.
    """
    return go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            title={"text": label, "font": {"size": 10}},  # Smaller font for compact display
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, value], "color": color},
                    {"range": [value, 100], "color": "lightgray"},
                ],
            },
        )
    ).update_layout(
        height=height,  # Adjustable height
        width=width,    # Adjustable width
        margin={"t": 5, "b": 5, "l": 5, "r": 5},  # Reduced margins for compactness
        showlegend=False,
    )

def plot_team_performance_gauges(filtered_df, selected_teams):
    """
    Display visual gauges for Win %, Draw %, and Loss % for each selected team.
    """
    if filtered_df.empty or not selected_teams:
        st.warning("No data matches the selected filters or no teams selected.")
        return

    for team in selected_teams:
        st.subheader(f"Performance Metrics for {team}")
        team_data = filtered_df[
            (filtered_df["team1"] == team) | (filtered_df["team2"] == team)
        ]

        total_matches = len(team_data)
        if total_matches == 0:
            st.write(f"No matches available for {team}.")
            continue

        # Calculate percentages
        win_percentage = ((team_data["team1"] == team) & (team_data["score1"] > team_data["score2"])).sum()
        win_percentage += ((team_data["team2"] == team) & (team_data["score2"] > team_data["score1"])).sum()
        win_percentage = (win_percentage / total_matches) * 100

        draw_percentage = (team_data["score1"] == team_data["score2"]).sum() / total_matches * 100
        loss_percentage = 100 - win_percentage - draw_percentage

        # Create gauges
        win_gauge = create_gauge_chart("", win_percentage, "green")
        draw_gauge = create_gauge_chart("", draw_percentage, "orange")
        loss_gauge = create_gauge_chart("", loss_percentage, "red")

        # Display gauges with labels
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Win %**")
            st.plotly_chart(win_gauge, use_container_width=True)
        with col2:
            st.markdown("**Draw %**")
            st.plotly_chart(draw_gauge, use_container_width=True)
        with col3:
            st.markdown("**Loss %**")
            st.plotly_chart(loss_gauge, use_container_width=True)
            
            
            
            
            
def plot_cumulative_goal_difference(filtered_df, selected_teams):
    """
    Line plot showing cumulative goal difference over time for selected teams.
    So the sum of the difference in goals between the team we are looking at and their opponent
    """
    if filtered_df.empty or not selected_teams:
        st.warning("No data matches the selected filters or no teams selected.")
        return

    # Filter only for selected teams
    filtered_df = filtered_df[
        (filtered_df["team1"].isin(selected_teams)) | (filtered_df["team2"].isin(selected_teams))
    ]
    filtered_df["Goal Difference"] = filtered_df["score1"] - filtered_df["score2"]

    # Cumulative goal difference calculation
    goal_diff_data = pd.concat([
        filtered_df[["Match Date", "team1", "Goal Difference"]].rename(columns={"team1": "Team"}),
        filtered_df[["Match Date", "team2", "Goal Difference"]]
        .assign(Goal_Difference=lambda df: -df["Goal Difference"])  # Reverse goal difference for the away team
        .rename(columns={"team2": "Team"}),
    ])
    goal_diff_data = goal_diff_data[goal_diff_data["Team"].isin(selected_teams)]  # Ensure only selected teams
    goal_diff_data = goal_diff_data.groupby(["Match Date", "Team"]).sum().groupby("Team").cumsum().reset_index()

    # Improved plot
    fig = px.line(
        goal_diff_data,
        x="Match Date",
        y="Goal Difference",
        color="Team",
        labels={"Goal Difference": "Cumulative Goal Difference", "Match Date": "Date"},
        title="Cumulative Goal Difference Over Time for Selected Teams",
        hover_data={"Goal Difference": ":.2f", "Match Date": "|%B %d, %Y"},
        height=500,
    )

    st.plotly_chart(fig, use_container_width=True)







def display_comparison_table(filtered_df, selected_teams):
    # Collect metrics for each team
    metrics = []
    for team in selected_teams:
        team_df = filtered_df[(filtered_df["team1"] == team) | (filtered_df["team2"] == team)]

        if team_df.empty:
            continue

        metrics.append({
            "Team": team,
            "Total Matches": len(team_df),
            "Avg Goals per Match": f"{team_df['Total Goals'].mean():.2f}",
            "Match Quality (Avg)": f"{team_df['Match Quality'].mean():.2f}",
            "High Stakes Matches": team_df["High Stakes"].sum(),
            "Home Win %": f"{((team_df['team1'] == team) & (team_df['score1'] > team_df['score2'])).sum() / len(team_df) * 100:.2f}%",
            "Away Win %": f"{((team_df['team2'] == team) & (team_df['score2'] > team_df['score1'])).sum() / len(team_df) * 100:.2f}%",
        })

    # Convert to DataFrame and Display
    metrics_df = pd.DataFrame(metrics)
    st.subheader("Team Comparison Table")
    st.dataframe(metrics_df, use_container_width=True)




def plot_team_comparison(filtered_df, selected_teams):
    if not selected_teams:
        st.warning("No teams selected for comparison.")
        return

    team_data = pd.concat([
        filtered_df[filtered_df["team1"].isin(selected_teams)][["Match Date", "team1", "Total Goals"]].rename(columns={"team1": "Team"}),
        filtered_df[filtered_df["team2"].isin(selected_teams)][["Match Date", "team2", "Total Goals"]].rename(columns={"team2": "Team"}),
    ])

    # Plot comparison
    fig = px.line(
        team_data,
        x="Match Date",
        y="Total Goals",
        color="Team",
        title="Goals Over Time per Team",
        labels={"Total Goals": "Goals", "Match Date": "Date", "Team": "Team"},
    )
    st.plotly_chart(fig, use_container_width=True)