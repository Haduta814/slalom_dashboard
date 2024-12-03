import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot_calibration_curve(df):
    # Actual outcomes as binary
    actual_win = (df["score1"] > df["score2"]).astype(int)
    predicted_prob_win = df["prob1"]

    # Generate calibration curve
    prob_true, prob_pred = calibration_curve(actual_win, predicted_prob_win, n_bins=10)

    # Create Plotly figure
    fig = go.Figure()

    # Add calibration curve
    fig.add_trace(go.Scatter(
        x=prob_pred,
        y=prob_true,
        mode="markers+lines",
        name="Calibration Curve",
        marker=dict(size=8, color="blue"),
        line=dict(color="blue", width=2),
    ))

    # Add perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Perfect Calibration",
        line=dict(color="green", dash="dash"),
    ))

    # Update layout
    fig.update_layout(
        title="Probability Calibration Curve for Predicted Wins",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
        template="plotly_white",
    )

    # Display the figure in Streamlit
    st.plotly_chart(fig, use_container_width=True)
def render_prediction_accuracy_tab(filtered_df: pd.DataFrame, filters):
    st.title("Prediction Accuracy")


    # Tabs for general metrics and visualization
    tab1, tab2 = st.tabs(["General Metrics", "Visuals"])

    # General Metrics Tab
    with tab1:
        display_general_prediction_metrics(filtered_df)

    # Accuracy Over Time Tab
    with tab2:
        st.write("Accuracy Over Time")
        display_accuracy_over_time(filtered_df)
        st.write("### Probability Calibration Curve")
        plot_calibration_curve(filtered_df)

def display_general_prediction_metrics(df):
    st.subheader("General Prediction Metrics")

    # Calculate Outcome Prediction Accuracy
    df["Outcome Correct"] = (
        ((df["score1"] > df["score2"]) & (df["prob1"] > df["prob2"])) |
        ((df["score2"] > df["score1"]) & (df["prob2"] > df["prob1"])) |
        ((df["score1"] == df["score2"]) & (df["probtie"] > df[["prob1", "prob2"]].max(axis=1)))
    ).astype(int)

    outcome_accuracy = df["Outcome Correct"].mean() * 100

    # Score Prediction Accuracy
    df["Score Difference"] = (
        (df["proj_score1"] - df["score1"]).abs() +
        (df["proj_score2"] - df["score2"]).abs()
    )
    avg_score_diff = df["Score Difference"].mean()

    # Precision and Recall for Predicted Wins
    df["Predicted Win"] = (df["prob1"] > df["prob2"]) & (df["prob1"] > df["probtie"])
    df["Actual Win"] = df["score1"] > df["score2"]

    true_positives = ((df["Predicted Win"]) & (df["Actual Win"])).sum()
    predicted_positives = df["Predicted Win"].sum()
    actual_positives = df["Actual Win"].sum()

    precision = (true_positives / predicted_positives * 100) if predicted_positives > 0 else 0
    recall = (true_positives / actual_positives * 100) if actual_positives > 0 else 0

    # Brier Score for Probability Calibration
    df["Brier Score"] = (
        ((df["prob1"] - (df["score1"] > df["score2"]).astype(int)) ** 2) +
        ((df["prob2"] - (df["score2"] > df["score1"]).astype(int)) ** 2) +
        ((df["probtie"] - (df["score1"] == df["score2"]).astype(int)) ** 2)
    )
    avg_brier_score = df["Brier Score"].mean()

    # Log Loss for Probabilistic Predictions
    import numpy as np
    df["Log Loss"] = -np.log(
        np.where(df["score1"] > df["score2"], df["prob1"],
                 np.where(df["score2"] > df["score1"], df["prob2"], df["probtie"]))
    )
    avg_log_loss = df["Log Loss"].mean()

    # Horizontally stack metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Outcome Prediction Accuracy", f"{outcome_accuracy:.2f}%")
        st.metric("Precision (Predicted Wins)", f"{precision:.2f}%")

    with col2:
        st.metric("Average Score Prediction Difference", f"{avg_score_diff:.2f}")
        st.metric("Recall (Actual Wins)", f"{recall:.2f}%")

    with col3:
        st.metric("Brier Score (Lower is Better)", f"{avg_brier_score:.4f}")
        st.metric("Log Loss (Lower is Better)", f"{avg_log_loss:.4f}")

    # Expandable section explaining metrics
    with st.expander("What do these metrics mean?"):
        st.markdown("""
        - **Outcome Prediction Accuracy**: The percentage of matches where the model correctly predicted the match outcome (win/loss/tie) based on probabilities.
        - **Precision (Predicted Wins)**: Of the matches predicted as wins, the percentage that were correct.
        - **Recall (Actual Wins)**: Of the matches that were actual wins, the percentage the model predicted correctly.
        - **Average Score Prediction Difference**: The average absolute difference between predicted scores and actual scores for both teams.
        - **Brier Score**: Measures the accuracy of probabilistic predictions; lower values indicate better calibration and accuracy.
        - **Log Loss**: Penalizes incorrect predictions more heavily than correct ones. Lower values indicate better accuracy and calibration.
        """)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
def display_accuracy_over_time(df):
    st.subheader("Prediction Accuracy Over Time")

    # Ensure 'date' column has valid datetime values
    df = df.dropna(subset=["date"])  # Drop rows with NaT in 'date'

    # Calculate weekly accuracy
    df["Week"] = df["date"].dt.to_period("W").apply(lambda x: x.start_time)
    accuracy_over_time = df.groupby("Week")["Outcome Correct"].mean().reset_index()
    accuracy_over_time["Outcome Accuracy (%)"] = accuracy_over_time["Outcome Correct"] * 100

    # Plot line chart
    fig = px.line(
        accuracy_over_time,
        x="Week",
        y="Outcome Accuracy (%)",
        title="Prediction Accuracy Over Time",
        labels={"Week": "Date", "Outcome Accuracy (%)": "Accuracy (%)"},
    )
    st.plotly_chart(fig, use_container_width=True)
