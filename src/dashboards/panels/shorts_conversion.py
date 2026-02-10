"""
Shorts vs Longform Analysis Panel.
Compares performance metrics between YouTube Shorts and longform content.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def render_shorts_analysis_panel(api_loader):
    """
    Render the Shorts vs Longform content analysis panel.

    Args:
        api_loader: YouTubeAPIDataLoader instance with loaded data
    """
    st.header("Shorts vs Longform Analysis")

    summary = api_loader.get_shorts_summary()
    shorts = summary['shorts']
    longform = summary['longform']

    if shorts['count'] == 0:
        st.info("No Shorts detected in this channel's content. "
                "Videos 60 seconds or shorter are classified as Shorts.")
        return

    # --- Side-by-side metric comparison ---
    st.subheader("Performance Comparison")
    col_shorts, col_longform = st.columns(2)

    with col_shorts:
        st.markdown("**Shorts**")
        st.metric("Count", f"{shorts['count']:,}")
        st.metric("Total Views", f"{shorts['total_views']:,}")
        st.metric("Avg Views", f"{shorts['avg_views']:,.0f}")
        st.metric("Avg Engagement Rate", f"{shorts['avg_engagement']:.2f}%")

    with col_longform:
        st.markdown("**Longform**")
        st.metric("Count", f"{longform['count']:,}")
        st.metric("Total Views", f"{longform['total_views']:,}")
        st.metric("Avg Views", f"{longform['avg_views']:,.0f}")
        st.metric("Avg Engagement Rate", f"{longform['avg_engagement']:.2f}%")

    st.markdown("---")

    # --- Charts side by side ---
    chart_left, chart_right = st.columns(2)

    # Grouped bar chart: avg views and avg engagement
    with chart_left:
        st.subheader("Avg Views & Engagement")
        bar_fig = go.Figure(data=[
            go.Bar(
                name="Avg Views",
                x=["Shorts", "Longform"],
                y=[shorts['avg_views'], longform['avg_views']],
                yaxis="y",
                offsetgroup=0,
            ),
            go.Bar(
                name="Avg Engagement (%)",
                x=["Shorts", "Longform"],
                y=[shorts['avg_engagement'], longform['avg_engagement']],
                yaxis="y2",
                offsetgroup=1,
            ),
        ])
        bar_fig.update_layout(
            yaxis=dict(title="Avg Views", side="left"),
            yaxis2=dict(title="Avg Engagement (%)", side="right", overlaying="y"),
            barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(bar_fig, use_container_width=True)

    # Pie chart: content mix
    with chart_right:
        st.subheader("Content Mix")
        pie_fig = px.pie(
            names=["Shorts", "Longform"],
            values=[shorts['count'], longform['count']],
            hole=0.4,
        )
        pie_fig.update_traces(textinfo="label+percent+value")
        pie_fig.update_layout(
            margin=dict(t=40, b=40),
            showlegend=False,
        )
        st.plotly_chart(pie_fig, use_container_width=True)

    st.markdown("---")

    # --- Per-show breakdown table ---
    st.subheader("Per-Show Breakdown")
    show_df = api_loader.get_show_breakdown()

    if show_df.empty:
        st.info("No show breakdown data available.")
        return

    # Format for display
    display_df = show_df.copy()
    display_df.index.name = "Show Name"
    st.dataframe(display_df, use_container_width=True)
