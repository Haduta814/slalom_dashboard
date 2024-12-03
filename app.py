import streamlit as st
import pandas as pd
import plotly.express as px


from helper_functions import render_sidebar,display_metrics,display_visualizations,load_data,apply_filters
from covid_functions import render_covid_effect_tab
from accuracy import render_prediction_accuracy_tab

def main():
    st.set_page_config(
        page_title="Football Data Dashboard", page_icon="⚽", layout="wide"
    )
    
    st.title("Football Data Dashboard")

    # Load the data
    df = load_data()
    filters = render_sidebar(df)  # Collect filters from the sidebar
    filtered_df = apply_filters(df, filters)  # Apply filters to the data

    if not filtered_df.empty:
        st.write(f"Games recorded: {len(filtered_df)}")
    else:
        st.warning("Filtered data is empty. Please adjust your filters.")

    # Tabs
    tabs = st.tabs(["Metrics", "Visualizations", "Data Preview", "Effects of COVID", "Accuracy of model"])

    # Metrics Tab
    with tabs[0]:
        display_metrics(filtered_df, filters)

    # Visualizations Tab
    with tabs[1]:
        display_visualizations(filtered_df, filters["teams"])

    # Data Preview Tab
    with tabs[2]:
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)

    # Effects of COVID Tab
    with tabs[3]:
        render_covid_effect_tab(filtered_df,filters["teams"])
    # Prediction Accuracy Tab
    with tabs[4]:
        render_prediction_accuracy_tab(filtered_df,filters )

if __name__ == "__main__":
    main()
