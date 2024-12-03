import streamlit as st
import pandas as pd
import plotly.express as px


from visualisations import (
    plot_spi_over_time,
    plot_importance_over_time,
    plot_win_percentage_vs_importance,
    plot_win_percentage_vs_spi,
    plot_team_performance_gauges,
    plot_cumulative_goal_difference
)



@st.cache_data
def load_data():
    df = pd.read_csv("../data.csv")

    # Drop unnecessary columns
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Convert dates
    df["Match Date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)


    return df


def render_sidebar(df):
    """Render dynamic sidebar filters based on selected options."""
    st.sidebar.header("Filters")

    # Default values for filtering
    leagues_default = ["All"] + sorted(df["league"].dropna().unique())
    seasons_default = ["All"] + sorted(df["season"].dropna().unique(), reverse=True)
    teams_default = ["All"] + sorted(pd.concat([df["team1"], df["team2"]]).dropna().unique())

    # Sidebar multiselects
    selected_leagues = st.sidebar.multiselect(
        "Select League(s):",
        options=leagues_default,
        default=["All"]
    )

    # Filter data for leagues to limit other options
    if "All" not in selected_leagues:
        df = df[df["league"].isin(selected_leagues)]

    selected_seasons = st.sidebar.multiselect(
        "Select Season(s):",
        options=["All"] + sorted(df["season"].dropna().unique(), reverse=True),
        default=["All"]
    )

    # Filter data for seasons to limit teams
    if "All" not in selected_seasons:
        df = df[df["season"].isin(selected_seasons)]

    selected_teams = st.sidebar.multiselect(
        "Select Team(s):",
        options=["All"] + sorted(pd.concat([df["team1"], df["team2"]]).dropna().unique()),
        default=["All"]
    )

    # Filter data for teams to limit other options
    if "All" not in selected_teams:
        df = df[
            (df["team1"].isin(selected_teams)) |
            (df["team2"].isin(selected_teams))
        ]

    # Return filters
    return {
        "leagues": selected_leagues,
        "seasons": selected_seasons,
        "teams": selected_teams
    }

def check_missing_data(df):
    """Check for missing data in the filtered DataFrame."""
    missing_data = df.isnull().sum()
    total_missing = missing_data.sum()
    missing_summary = pd.DataFrame({
        "Column": missing_data[missing_data > 0].index,
        "Missing Values": missing_data[missing_data > 0]
    }).reset_index(drop=True)

    return total_missing, missing_summary

def apply_filters(df, filters):
    filtered_df = df.copy()

    # Apply league filter
    if "All" not in filters["leagues"] and filters["leagues"]:
        filtered_df = filtered_df[filtered_df["league"].isin(filters["leagues"])]

    # Apply season filter
    if "All" not in filters["seasons"] and filters["seasons"]:
        filtered_df = filtered_df[filtered_df["season"].isin(filters["seasons"])]

    # Apply team filter
    if "All" not in filters["teams"] and filters["teams"]:
        filtered_df = filtered_df[
            (filtered_df["team1"].isin(filters["teams"]))
            | (filtered_df["team2"].isin(filters["teams"]))
        ]

    # Check for missing data in the filtered dataset
    check_missing_data(filtered_df)

    return filtered_df


def calculate_percentages(team_df, team):
    total_matches = len(team_df)
    if total_matches == 0:
        return "N/A", "N/A", "N/A"  # No matches available

    # Calculate the number of wins, losses, and draws
    home_wins = ((team_df["team1"] == team) & (team_df["score1"] > team_df["score2"])).sum()
    away_wins = ((team_df["team2"] == team) & (team_df["score2"] > team_df["score1"])).sum()
    wins = home_wins + away_wins

    home_losses = ((team_df["team1"] == team) & (team_df["score1"] < team_df["score2"])).sum()
    away_losses = ((team_df["team2"] == team) & (team_df["score2"] < team_df["score1"])).sum()
    losses = home_losses + away_losses

    draws = (team_df["score1"] == team_df["score2"]).sum()

    # Compute percentages
    win_percentage = (wins / total_matches) * 100
    loss_percentage = (losses / total_matches) * 100
    draw_percentage = 100 - win_percentage - loss_percentage

    # Return formatted values
    return f"{win_percentage:.2f}%", f"{loss_percentage:.2f}%", f"{draw_percentage:.2f}%"




def display_metrics(filtered_df, filters):
    st.header("Metrics Overview")
    selected_teams = filters["teams"]
    selected_leagues = filters["leagues"]
    selected_seasons = filters["seasons"]

    # Initialize Tabs
    tabs = st.tabs(["Key Metrics", "Performance Metrics", "Efficiency Metrics", "Advanced Metrics"])
    
    # Global Metrics (for "All Teams" or when no teams are selected)
    if not selected_teams or "All" in selected_teams:
        with tabs[0]:
            st.subheader("Key Metrics for All Teams")
            col1, col2, col3 = st.columns(3)

            with col1:
                total_matches = len(filtered_df)
                avg_goals_per_match = f"{filtered_df['Total Goals'].mean():.2f}" if total_matches > 0 else "N/A"
                avg_match_quality = f"{filtered_df['Match Quality'].mean():.2f}" if total_matches > 0 else "N/A"
                st.metric("Total Matches", total_matches)
                st.metric("Avg Goals per Match", avg_goals_per_match)
                st.metric("Avg Match Quality", avg_match_quality)

            with col2:
                home_wins = (filtered_df["score1"] > filtered_df["score2"]).sum()
                away_wins = (filtered_df["score1"] < filtered_df["score2"]).sum()
                draws = (filtered_df["score1"] == filtered_df["score2"]).sum()
                total_matches = len(filtered_df)
                home_win_pct = f"{(home_wins / total_matches) * 100:.2f}%" if total_matches > 0 else "N/A"
                away_win_pct = f"{(away_wins / total_matches) * 100:.2f}%" if total_matches > 0 else "N/A"
                draw_pct = f"{(draws / total_matches) * 100:.2f}%" if total_matches > 0 else "N/A"
                st.metric("Home Win %", home_win_pct)
                st.metric("Away Win %", away_win_pct)
                st.metric("Draw %", draw_pct)

            with col3:
                high_stakes_matches = filtered_df["High Stakes"].sum()
                avg_importance_gap = f"{filtered_df['Importance Gap'].mean():.2f}" if total_matches > 0 else "N/A"
                avg_pressure_index = f"{filtered_df['Match Pressure Index'].mean():.2f}" if total_matches > 0 else "N/A"
                st.metric("High Stakes Matches", high_stakes_matches)
                st.metric("Avg Importance Gap", avg_importance_gap)
                st.metric("Avg Match Pressure Index", avg_pressure_index)
        with tabs[0]:
            st.subheader("Key Metrics for All Teams")
        
        # Metric Descriptions
        with st.expander("What do these metrics mean?"):
            st.write("""
            - **Total Matches**: The total number of matches played in the filtered data.
            - **Avg Goals per Match**: The average number of goals scored per match.
            - **Avg Match Quality**: The average match quality score, based on various parameters.
            - **Home Win %**: Percentage of matches won by home teams.
            - **Away Win %**: Percentage of matches won by away teams.
            - **Draw %**: Percentage of matches that ended in a draw.
            - **High Stakes Matches**: Number of matches classified as high stakes (Importance above 0.8 for one or both of the teams).
            - **Avg Importance Gap**: Average difference in importance levels between the two teams in matches.
            - **Avg Match Pressure Index**: Average pressure experienced during matches, based on importance and stakes.
            """)

        col1, col2, col3 = st.columns(3)
############################### PERFORMANCE METRICS FOR ALL TEAMS

        with tabs[1]:
            st.subheader("Performance Metrics for All Teams")
            col1, col2 = st.columns(2)

            with col1:
                avg_home_spi_change = f"{filtered_df['Home SPI Change'].mean():.2f}" if "Home SPI Change" in filtered_df else "N/A"
                avg_away_spi_change = f"{filtered_df['Away SPI Change'].mean():.2f}" if "Away SPI Change" in filtered_df else "N/A"
                st.metric("Avg Home SPI Change", avg_home_spi_change)
                st.metric("Avg Away SPI Change", avg_away_spi_change)

            with col2:
                top_scorer = filtered_df.groupby("team1")["Total Goals"].sum().idxmax() if not filtered_df.empty else "N/A"
                best_defensive_team = (
                    filtered_df.groupby("team2")["Total Goals"].sum().idxmin() if not filtered_df.empty else "N/A"
                )
                st.metric("Top Scoring Team", top_scorer)
                st.metric("Best Defensive Team", best_defensive_team)

############################### FILTERED TEAM METRICS

    else:
        for team in selected_teams:
            team_df = filtered_df[(filtered_df["team1"] == team) | (filtered_df["team2"] == team)]

            if team_df.empty:
                st.warning(f"No data available for {team}")
                continue

            with tabs[0]:
                st.subheader(f"Key Metrics for {team}")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Matches", len(team_df))
                    st.metric("Top Match Quality", f"{team_df['Match Quality'].max():.2f}")
                with col2:
                    st.metric("Avg Goals per Match", f"{team_df['Total Goals'].mean():.2f}")
                    st.metric("High Stakes Matches", team_df["High Stakes"].sum())
                with col3:
                    st.metric("Match Quality (Avg)", f"{team_df['Match Quality'].mean():.2f}")

            with tabs[1]:
                st.subheader(f"Performance Metrics for {team}")
                col1, col2, col3 = st.columns(3)
                win_pct, loss_pct, draw_pct = calculate_percentages(team_df, team)
                with col1:
                    home_wins = ((team_df["team1"] == team) & (team_df["score1"] > team_df["score2"])).sum()
                    away_wins = ((team_df["team2"] == team) & (team_df["score2"] > team_df["score1"])).sum()
                    draws = (team_df["score1"] == team_df["score2"]).sum()
                    st.metric("Home Wins", home_wins)
                    st.metric("Total Wins",home_wins + away_wins)
                    st.metric("Win %", win_pct)
                    

                with col2:
                    st.metric("Away Wins", away_wins)
                    st.metric("Draws", draws)
                    st.metric("Draw %", draw_pct)
                    

                with col3:
                    avg_spi_change = (
                        team_df["Home SPI Change"].mean() if team_df["team1"].iloc[0] == team else team_df["Away SPI Change"].mean()
                    )
                    st.metric("Avg SPI Change", f"{avg_spi_change:.2f}" if avg_spi_change else "N/A")
                    st.metric("Total Losses", len(team_df)-draws - home_wins - away_wins)
                    st.metric("Loss %", loss_pct)
                    
            with tabs[2]:
                st.subheader(f"Efficiency stats for {team}")
                col1, col2, col3 = st.columns(3)

                with col1:
                    goals_per_match = team_df["Total Goals"].mean()
                    st.metric("Avg Goals per Match", f"{goals_per_match:.2f}" if pd.notnull(goals_per_match) else "N/A")

                    avg_home_efficiency = team_df["Home Efficiency"].mean()
                    st.metric("Avg Home Efficiency", f"{avg_home_efficiency:.2f}" if pd.notnull(avg_home_efficiency) else "N/A")

                with col2:
                    avg_away_efficiency = team_df["Away Efficiency"].mean()
                    st.metric("Avg Away Efficiency", f"{avg_away_efficiency:.2f}" if pd.notnull(avg_away_efficiency) else "N/A")

                    overall_efficiency = (team_df["Home Efficiency"].sum() + team_df["Away Efficiency"].sum()) / (len(team_df) * 2)
                    st.metric("Overall Efficiency", f"{overall_efficiency:.2f}" if pd.notnull(overall_efficiency) else "N/A")

                with col3:
                    avg_home_xg_diff = team_df["Home xG Difference"].mean()
                    st.metric("Home xG Difference", f"{avg_home_xg_diff:.2f}" if pd.notnull(avg_home_xg_diff) else "N/A")

                    avg_away_xg_diff = team_df["Away xG Difference"].mean()
                    st.metric("Away xG Difference", f"{avg_away_xg_diff:.2f}" if pd.notnull(avg_away_xg_diff) else "N/A")

            with tabs[3]:
                st.subheader(f"More advanced stats for {team}")
                col1, col2, col3 = st.columns(3)

                with col1:
                    avg_match_pressure_index = team_df["Match Pressure Index"].mean()
                    st.metric("Avg Match Pressure Index", f"{avg_match_pressure_index:.2f}" if pd.notnull(avg_match_pressure_index) else "N/A")

                    avg_importance_gap = team_df["Importance Gap"].mean()
                    st.metric("Avg Importance Gap", f"{avg_importance_gap:.2f}" if pd.notnull(avg_importance_gap) else "N/A")

                with col2:
                    avg_home_spi_change = team_df["Home SPI Change"].mean()
                    st.metric("Avg Home SPI Change", f"{avg_home_spi_change:.2f}" if pd.notnull(avg_home_spi_change) else "N/A")

                    avg_away_spi_change = team_df["Away SPI Change"].mean()
                    st.metric("Avg Away SPI Change", f"{avg_away_spi_change:.2f}" if pd.notnull(avg_away_spi_change) else "N/A")

                with col3:
                    home_overperformance = team_df["Home Overperformance"].mean()
                    st.metric("Home Overperformance", f"{home_overperformance:.2f}" if pd.notnull(home_overperformance) else "N/A")

                    away_overperformance = team_df["Away Overperformance"].mean()
                    st.metric("Away Overperformance", f"{away_overperformance:.2f}" if pd.notnull(away_overperformance) else "N/A")

                    high_stakes_matches = team_df["High Stakes"].sum()
                    st.metric("High Stakes Matches", high_stakes_matches)
                            
                # If multiple teams are selected, show comparison table
                if len(selected_teams) > 1:
                    st.subheader("Team Comparison Table")
                    comparison_metrics = []
                    for team in selected_teams:
                        team_df = filtered_df[(filtered_df["team1"] == team) | (filtered_df["team2"] == team)]
                        if team_df.empty:
                            continue
                        metrics = {
                            "Team": team,
                            "Total Matches": len(team_df),
                            "Avg Goals per Match": f"{team_df['Total Goals'].mean():.2f}",
                            "High Stakes Matches": team_df["High Stakes"].sum(),
                            "Home Win %": f"{((team_df['team1'] == team) & (team_df['score1'] > team_df['score2'])).sum() / len(team_df) * 100:.2f}%",
                            "Away Win %": f"{((team_df['team2'] == team) & (team_df['score2'] > team_df['score1'])).sum() / len(team_df) * 100:.2f}%",
                            "Draw %": f"{(team_df['score1'] == team_df['score2']).sum() / len(team_df) * 100:.2f}%",
                        }
                        comparison_metrics.append(metrics)
                    comparison_df = pd.DataFrame(comparison_metrics)
                    st.dataframe(comparison_df, use_container_width=True)




def display_visualizations(filtered_df, selected_teams):
    st.subheader("Visualizations")

    if not selected_teams or "All" in selected_teams:
        st.header("Visualizations for All Teams")
        
        # League-Wide SPI Trends
        st.subheader("Average SPI Trends Over Time")
        spi_trends = filtered_df.groupby(filtered_df["Match Date"].dt.to_period("M"))[["spi1", "spi2"]].mean().reset_index()
        spi_trends["Match Date"] = spi_trends["Match Date"].astype(str)

        fig = px.line(
            spi_trends,
            x="Match Date",
            y=["spi1", "spi2"],
            labels={"value": "Average SPI", "Match Date": "Date"},
            title="Average SPI for Home and Away Teams Over Time",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Win/Draw/Loss Distribution
        st.subheader("Win/Draw/Loss Distribution")
        total_matches = len(filtered_df)
        wins = (filtered_df["score1"] > filtered_df["score2"]).sum()
        losses = (filtered_df["score1"] < filtered_df["score2"]).sum()
        draws = total_matches - wins - losses

        fig = px.pie(
            names=["Wins", "Draws", "Losses"],
            values=[wins, draws, losses],
            title="Win/Draw/Loss Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Goal Distribution
        st.subheader("Goal Distribution")
        fig = px.histogram(
            filtered_df,
            x="Total Goals",
            nbins=20,
            title="Distribution of Total Goals Scored per Match",
            labels={"Total Goals": "Goals"},
        )
        st.plotly_chart(fig, use_container_width=True)

        # Importance vs. Match Quality
        st.subheader("Importance vs. Match Quality")
        fig = px.scatter(
            filtered_df,
            x="importance1",
            y="Match Quality",
            title="Importance vs. Match Quality",
            labels={"importance1": "Importance", "Match Quality": "Match Quality"},
            trendline="ols",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Average Performance Metrics by Season
        st.subheader("Average Performance Metrics by Season")
        performance_by_season = (
            filtered_df.groupby("season")[["Home Efficiency", "Away Efficiency", "Match Quality"]]
            .mean()
            .reset_index()
        )
        fig = px.bar(
            performance_by_season,
            x="season",
            y=["Home Efficiency", "Away Efficiency", "Match Quality"],
            barmode="group",
            title="Average Performance Metrics by Season",
            labels={"value": "Average Metric Value", "season": "Season"},
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        # Existing logic for selected teams
        team_names = ", ".join(selected_teams)
        st.header(f"Graphs for {team_names}")
        plot_team_performance_gauges(filtered_df, selected_teams)
        plot_spi_over_time(filtered_df, selected_teams)
        plot_importance_over_time(filtered_df, selected_teams)
        plot_cumulative_goal_difference(filtered_df, selected_teams)

