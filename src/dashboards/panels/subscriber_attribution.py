"""
Subscriber Attribution Panel.
Shows subscriber acquisition breakdown by content type (Shorts vs Longform).
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


def render_subscriber_attribution_panel(api_loader):
    """
    Render the subscriber attribution analytics panel.

    Args:
        api_loader: YouTubeAPIDataLoader instance with data loaded
    """
    st.header("Subscriber Attribution")

    try:
        data = api_loader.get_subscriber_sources_by_content_type()
    except Exception:
        st.warning(
            "Subscriber attribution data is not available. "
            "This requires YouTube Analytics API access."
        )
        return

    # --- Metric row ---
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Attributed Subscribers", f"{data['total']:,}")
    with col2:
        st.metric("From Shorts", f"{data['from_shorts']:,}")
    with col3:
        st.metric("From Longform", f"{data['from_longform']:,}")
    with col4:
        st.metric("Shorts Share", f"{data['shorts_percentage']:.1f}%")

    # --- Donut chart: Shorts vs Longform split ---
    labels = ["Shorts", "Longform"]
    values = [data["from_shorts"], data["from_longform"]]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.4,
                marker=dict(colors=["#FF6B6B", "#4ECDC4"]),
                textinfo="label+percent",
                hovertemplate="%{label}: %{value:,} subscribers<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        title_text="Subscriber Acquisition by Content Type",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Subscriber attribution data is based on a 90-day rolling window "
        "from the YouTube Analytics API."
    )
