"""
Show Breakdown Panel.
Visualizes performance breakdown by show/series across the channel.
"""

import streamlit as st
import plotly.express as px
import pandas as pd


def render_show_breakdown_panel(api_loader):
    """
    Render the show breakdown analytics panel.

    Args:
        api_loader: YouTubeAPIDataLoader instance with data loaded
    """
    st.header("Show Breakdown")

    show_df = api_loader.get_show_breakdown()

    if show_df.empty:
        st.info("No show data available. Ensure videos have been loaded.")
        return

    # Reset index so Show Name is a column for plotting
    show_df = show_df.reset_index()

    # --- Bar chart: Views by show ---
    st.subheader("Views by Show")
    bar_fig = px.bar(
        show_df.sort_values("Views", ascending=True),
        x="Views",
        y="Show Name",
        orientation="h",
        labels={"Views": "Total Views", "Show Name": ""},
    )
    bar_fig.update_layout(height=max(400, len(show_df) * 30))
    st.plotly_chart(bar_fig, use_container_width=True)

    # --- Treemap: Video count by show ---
    st.subheader("Catalog Size by Show")
    treemap_fig = px.treemap(
        show_df,
        path=["Show Name"],
        values="Video Count",
        labels={"Video Count": "Videos"},
    )
    treemap_fig.update_layout(height=500)
    st.plotly_chart(treemap_fig, use_container_width=True)

    # --- Data table: Full metrics ---
    st.subheader("All Show Metrics")
    st.dataframe(
        show_df.sort_values("Views", ascending=False).reset_index(drop=True),
        use_container_width=True,
    )
