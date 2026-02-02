# Planning

## Current State

**Mode**: Development (Active - YouTube API integration in progress)
**Last Session**: 2026-02-02 by The Fixer (modernization)
**Branch**: main

## Current Focus

YouTube API integration module complete but uncommitted. Next steps: integrate API data loader with existing dashboard infrastructure and implement PBS Wisconsin-specific dashboard panels.

## Quick Links

- Progress log: `progress.md`
- Scratchpad: `backlog.md` (session scratchpad → GitHub Issues)
- Upstream project: [YouTubeStudioDataAnalytics](https://github.com/DuongCaoNhan/YouTubeStudioDataAnalytics)

## PBS Wisconsin Context

This is a custom fork for PBS Wisconsin's YouTube channel analytics with specialized features:
- **Title pattern parsing** for show extraction
- **Shorts detection** and analysis (videos ≤60s)
- **Archival content** performance tracking
- **Show-based aggregation** for multi-series reporting

## Upstream Relationship

**Fork of**: [DuongCaoNhan/YouTubeStudioDataAnalytics](https://github.com/DuongCaoNhan/YouTubeStudioDataAnalytics)
**Fork purpose**: Replace CSV workflows with live YouTube API integration + PBS-specific analytics
**Upstream sync**: Not planned (significant PBS customizations make upstream incompatible)

## Development Environment

- Python 3.9+ virtual environment at `venv/`
- OAuth credentials in `credentials/` (git-ignored)
- SQLite database at `data/youtube_analytics.db`
- Dashboard runs on Streamlit (port 8501) or Dash (port 8050)
