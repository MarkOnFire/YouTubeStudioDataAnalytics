"""
Archival Performance Panel.
Visualizes old videos that continue to perform well over time.
"""

import streamlit as st
import plotly.express as px
import pandas as pd


def render_archival_performance_panel(api_loader):
    """
    Render the archival performance analytics panel.

    Args:
        api_loader: YouTubeAPIDataLoader instance with data loaded
    """
    st.header("Archival Performance")

    threshold = st.slider(
        "Archival threshold (months)",
        min_value=6,
        max_value=36,
        value=12,
        help="Videos older than this threshold are considered archival content",
    )

    archival_df = api_loader.get_archival_performance(months_threshold=threshold)

    if archival_df.empty:
        st.info(f"No archival videos found older than {threshold} months.")
        return

    # --- Metric row ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Archival Videos", f"{len(archival_df):,}")
    with col2:
        st.metric("Total Archival Views", f"{int(archival_df['Views'].sum()):,}")
    with col3:
        st.metric("Avg Views / Day", f"{archival_df['Views per Day'].mean():,.1f}")

    # --- Scatter: Age vs Views per Day ---
    st.subheader("Age vs Daily Velocity")
    scatter_fig = px.scatter(
        archival_df,
        x="Age (months)",
        y="Views per Day",
        color="Show Name",
        hover_data=["Title", "Views"],
        labels={
            "Age (months)": "Age (months)",
            "Views per Day": "Views per Day",
        },
    )
    scatter_fig.update_layout(height=500)
    st.plotly_chart(scatter_fig, use_container_width=True)

    # --- Bar: Top 10 by daily velocity ---
    st.subheader("Top 10 Archival Videos by Daily Velocity")
    top10 = archival_df.head(10)
    bar_fig = px.bar(
        top10,
        x="Views per Day",
        y="Title",
        orientation="h",
        color="Show Name",
        labels={"Views per Day": "Views per Day", "Title": ""},
    )
    bar_fig.update_layout(yaxis={"autorange": "reversed"}, height=450)
    st.plotly_chart(bar_fig, use_container_width=True)

    # --- Data table: Top 20 ---
    st.subheader("Top 20 Archival Videos")
    st.dataframe(
        archival_df.head(20).reset_index(drop=True),
        use_container_width=True,
    )
