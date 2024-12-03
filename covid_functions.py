import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from visualisations import create_gauge_chart

def render_covid_effect_tab(df: pd.DataFrame, selected_teams):
    st.title("Effects of COVID on Football")

    # Sub-tabs for Metrics and Visualizations
    tab1, tab2 = st.tabs(["Metrics", "Visualizations"])

    # Metrics Tab
    with tab1:
        display_covid_metrics(df, selected_teams)

    # Visualizations Tab
    with tab2:
        display_covid_visualizations(df, selected_teams)

    
        
        
def calculate_outcome_percentages(df, selected_teams=None):
    """
    Calculate win, loss, and draw percentages.
    
    Args:
        df (DataFrame): Filtered dataset.
        selected_teams (list or None): Teams to compute metrics for. If None or "All", calculate league-wide metrics.

    Returns:
        dict: Dictionary with percentages for wins, losses, and draws.
    """
    total_matches = len(df)
    if total_matches == 0:
        return {"Win %": 0, "Loss %": 0, "Draw %": 0}

    if not selected_teams or "All" in selected_teams:
        # League-wide metrics
        home_wins = (df["score1"] > df["score2"]).sum()
        draws = (df["score1"] == df["score2"]).sum()
        away_wins = (df["score2"] > df["score1"]).sum()

        return {
            "Away Win %": (away_wins / total_matches) * 100,
            "Draw %": (draws / total_matches) * 100,
            "Home Win %": (home_wins / total_matches) * 100,
        }
    else:
        # Metrics for selected teams
        team_df = df[
            (df["team1"].isin(selected_teams)) | (df["team2"].isin(selected_teams))
        ]

        total_team_matches = len(team_df)
        if total_team_matches == 0:
            return {"Win %": 0, "Draw %": 0,"Loss %": 0}

        wins = ((team_df["team1"].isin(selected_teams)) & (team_df["score1"] > team_df["score2"])).sum()
        wins += ((team_df["team2"].isin(selected_teams)) & (team_df["score2"] > team_df["score1"])).sum()

        draws = (team_df["score1"] == team_df["score2"]).sum()
        losses = total_team_matches - wins - draws

        return {
            "Win %": (wins / total_team_matches) * 100,
            "Draw %": (draws / total_team_matches) * 100,
            "Loss %": (losses / total_team_matches) * 100,
        }
        
        
        
def display_covid_visualizations(df, selected_teams=None):
    st.subheader("COVID Impact Visualizations")

    # Ensure 'date' is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Goals Scored Per Month Across Periods
    st.write("### Goals Scored Per Month Across COVID Periods")
    goals_data = []
    periods = ["Pre-COVID", "During COVID", "Post-COVID"]
    period_data = [
        df[df['date'] < "2019-12-01"],
        df[(df['date'] >= "2019-12-01") & (df['date'] <= "2021-06-30")],
        df[df['date'] > "2021-06-30"]
    ]

    for period, data in zip(periods, period_data):
        if selected_teams and "All" not in selected_teams:
            data = data[(data["team1"].isin(selected_teams)) | (data["team2"].isin(selected_teams))]
        total_goals = data["score1"].sum() + data["score2"].sum()
        months = data['date'].dt.to_period('M').nunique()  # Number of unique months in the period
        goals_per_month = round(total_goals / months, 2) if months > 0 else 0
        goals_data.append({"Period": period, "Goals Per Month": goals_per_month})

    goals_df = pd.DataFrame(goals_data)
    fig_goals_per_month = px.bar(
        goals_df,
        x="Period",
        y="Goals Per Month",
        title="Average Goals Scored Per Month Across COVID Periods",
        labels={"Goals Per Month": "Goals Per Month", "Period": "COVID Period"},
        color="Period",
        text="Goals Per Month"
    )

    # Ensure text values are rounded in the chart
    fig_goals_per_month.update_traces(texttemplate='%{text:.2f}')

    st.plotly_chart(fig_goals_per_month, use_container_width=True)

    # SPI vs. COVID Infection Rates for Selected Teams
    st.write("### Combined SPI vs. COVID Infection Rates for Selected Teams")
    if selected_teams and "All" not in selected_teams:
        # Filter data for selected teams
        team_data = df[
            (df["team1"].isin(selected_teams)) | (df["team2"].isin(selected_teams))
        ].copy()

        # Combine SPI values for the selected team(s)
        team_data["Team SPI"] = team_data.apply(
            lambda row: row["spi1"] if row["team1"] in selected_teams else row["spi2"], axis=1
        )
        team_data["Team"] = team_data.apply(
            lambda row: row["team1"] if row["team1"] in selected_teams else row["team2"], axis=1
        )
        team_data["COVID Metric"] = team_data["new_cases_per_million"]

        # Plot SPI vs COVID infection rates
        fig_spi_vs_covid = px.scatter(
            team_data,
            x="COVID Metric",
            y="Team SPI",
            color="Team",  # Different colors for different teams
            title=f"Combined SPI vs. COVID Infection Rates for Selected Teams",
            labels={
                "COVID Metric": "COVID Infection Rates (Cases per Million)",
                "Team SPI": "Combined SPI (Team Performance)",
                "Team": "Team"
            },
            trendline="ols",
            hover_data=["team1", "team2", "league"]
        )
        st.plotly_chart(fig_spi_vs_covid, use_container_width=True)
    else:
        st.warning("Please select specific team(s) to view SPI changes against COVID infection rates.")
        
def display_covid_metrics(df, selected_teams=None):
    st.subheader("COVID Impact Metrics")

    # Ensure 'date' is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Segment data into periods
    pre_covid = df[df['date'] < "2019-12-01"]
    during_covid = df[(df['date'] >= "2019-12-01") & (df['date'] <= "2021-06-30")]
    post_covid = df[df['date'] > "2021-06-30"]

    def calculate_league_metrics(data):
        total_matches = len(data)
        if total_matches == 0:
            return {"Win %": 0, "Draw %": 0, "Loss %": 0}

        win_percentage = (data["score1"] > data["score2"]).sum() / total_matches * 100
        draw_percentage = (data["score1"] == data["score2"]).sum() / total_matches * 100
        loss_percentage = (data["score1"] < data["score2"]).sum() / total_matches * 100

        return {
            "Win %": win_percentage,
            "Draw %": draw_percentage,
            "Loss %": loss_percentage,
        }

    def calculate_team_metrics(data, team):
        total_matches = len(data)
        if total_matches == 0:
            return {"Win %": 0, "Draw %": 0, "Loss %": 0}

        wins = ((data["team1"] == team) & (data["score1"] > data["score2"])).sum()
        wins += ((data["team2"] == team) & (data["score2"] > data["score1"])).sum()

        draws = (data["score1"] == data["score2"]).sum()
        losses = total_matches - wins - draws

        return {
            "Win %": (wins / total_matches) * 100,
            "Draw %": (draws / total_matches) * 100,
            "Loss %": (losses / total_matches) * 100,
        }

    if not selected_teams or "All" in selected_teams:
        # League-wide metrics
        st.write("### League-Wide Performance Metrics")
        st.subheader("Displaying league-wide metrics for all matches (team1).")
        league_metrics = {
            "Pre-COVID": calculate_league_metrics(pre_covid),
            "During COVID": calculate_league_metrics(during_covid),
            "Post-COVID": calculate_league_metrics(post_covid),
        }

        for period, metrics in league_metrics.items():
            with st.container():
                st.subheader(f"{period} Performance")
                col1, col2, col3 = st.columns([1, 1, 1])  # Adjust column width
                col1.plotly_chart(
                    create_gauge_chart("", metrics["Win %"], "green", width=80, height=80),
                    use_container_width=False,
                    key=f"league_{period}_win"
                )
                col2.plotly_chart(
                    create_gauge_chart("", metrics["Draw %"], "orange", width=80, height=80),
                    use_container_width=False,
                    key=f"league_{period}_draw"
                )
                col3.plotly_chart(
                    create_gauge_chart("", metrics["Loss %"], "red", width=80, height=80),
                    use_container_width=False,
                    key=f"league_{period}_loss"
                )
    else:
        # Metrics for each selected team
        if len(selected_teams) == 1:
            st.subheader(f"Displaying metrics for {selected_teams[0]}.")
        else:
            st.subheader("Displaying metrics for multiple selected teams.")

        for team in selected_teams:
            with st.expander(f"Performance Metrics for {team}"):
                team_metrics = {
                    "Pre-COVID": calculate_team_metrics(pre_covid, team),
                    "During COVID": calculate_team_metrics(during_covid, team),
                    "Post-COVID": calculate_team_metrics(post_covid, team),
                }

                for period, metrics in team_metrics.items():
                    with st.container():
                        st.subheader(f"{period} Performance")
                        col1, col2, col3 = st.columns([1, 1, 1])  # Adjust column width
                        col1.plotly_chart(
                            create_gauge_chart("", metrics["Win %"], "green", width=80, height=80),
                            use_container_width=False,
                            key=f"{team}_{period}_win"
                        )
                        col2.plotly_chart(
                            create_gauge_chart("", metrics["Draw %"], "orange", width=80, height=80),
                            use_container_width=False,
                            key=f"{team}_{period}_draw"
                        )
                        col3.plotly_chart(
                            create_gauge_chart("", metrics["Loss %"], "red", width=80, height=80),
                            use_container_width=False,
                            key=f"{team}_{period}_loss"
                        )

    # Visual Explanation
    st.write("""
    These gauges show the win, draw, and loss percentages based on the selected teams or all teams during the three COVID-related periods:
    - Pre-COVID (Before Dec 2019)
    - During COVID (Dec 2019 to Jun 2021)
    - Post-COVID (After Jun 2021)

    If no teams are selected, these metrics represent league-wide performance for team1. Otherwise, they reflect the performance of the selected teams.
    """)