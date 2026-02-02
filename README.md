# PBS Wisconsin YouTube Analytics

**Custom fork of [YouTubeStudioDataAnalytics](https://github.com/DuongCaoNhan/YouTubeStudioDataAnalytics)** with direct YouTube API integration and PBS Wisconsin-specific analytics features.

## What Makes This Fork Different

This fork replaces CSV-based workflows with live YouTube Data API v3 and Analytics API integration, adding:

- **Title Pattern Parser**: Extracts show names from PBS Wisconsin video titles
- **Shorts Detection**: Automatically flags videos ≤60 seconds as Shorts
- **Archival Content Tracking**: Surfaces older videos gaining new traction
- **Show-Based Analytics**: Aggregates metrics by PBS Wisconsin show/series
- **SQLite Persistence**: Caches API data locally to minimize quota usage

## Quick Start

```bash
# Clone and setup
git clone git@github.com:MarkOnFire/pbswi-youtube-analytics.git
cd pbswi-youtube-analytics
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure YouTube API credentials (see Setup section)
python -m src.youtube_api.auth

# Run the dashboard
python main.py --streamlit
```

## Setup

### 1. Google Cloud Project Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable these APIs:
   - YouTube Data API v3
   - YouTube Analytics API
4. Go to **Credentials > Create Credentials > OAuth Client ID**
5. Select **Desktop application**
6. Download the JSON credentials
7. Save as `credentials/credentials.json`

### 2. OAuth Authentication

```bash
source venv/bin/activate
python -m src.youtube_api.auth
```

This opens a browser for Google OAuth consent. After approval, tokens are saved to `credentials/token.json`.

### 3. Configure Channels

Edit `config/channels.yaml` to add your channel IDs:

```yaml
channels:
  - id: "UCxxxxxxxxxx"
    name: "PBS Wisconsin"
    type: "main"
```

## Features

### YouTube API Integration (`src/youtube_api/`)

- **OAuth2 Flow** (`auth.py`): Handles Google authentication with token refresh
- **API Client** (`client.py`): Wraps YouTube Data API v3 and Analytics API
- **Data Loader** (`data_loader.py`): Drop-in replacement for CSV loading
- **Database** (`database.py`): SQLite persistence for historical data
- **Models** (`models.py`): Pydantic data validation and serialization

### PBS Wisconsin Customizations

#### Title Pattern Parsing
Extracts show names from video titles:
- Standard format: `"Video Title | SHOW NAME"` → "SHOW NAME"
- Exception: `"Wisconsin Life | Video Title"` → "Wisconsin Life"

#### Shorts vs. Longform
Automatically categorizes videos:
- **Shorts**: ≤60 seconds
- **Longform**: >60 seconds

Track conversion metrics: which Shorts drive traffic to longform content.

#### Archival Content Analytics
Identify videos older than 12 months that are gaining new views/engagement.

#### Show-Based Aggregation
Group analytics by show for multi-series reporting and comparison.

## Usage

### API Data Loader

```python
from src.youtube_api import YouTubeAPIDataLoader

# Initialize (uses authenticated user's channel)
loader = YouTubeAPIDataLoader()

# Load video data (replaces CSV loading)
videos_df = loader.load_videos_data()
subscribers_df = loader.load_subscribers_data()

# PBS-specific methods
archival = loader.get_archival_performance(months_threshold=12)
shorts_summary = loader.get_shorts_summary()
show_breakdown = loader.get_show_breakdown()
```

### Database Queries

```python
from src.youtube_api import AnalyticsDatabase

db = AnalyticsDatabase()

# Store videos
db.upsert_videos_bulk(videos)

# Query
archival = db.get_archival_videos(months_threshold=12)
shows = db.get_show_summary()
shorts_vs_long = db.get_shorts_vs_longform()
```

### Running Dashboards

```bash
# Streamlit (recommended)
python main.py --streamlit

# Dash alternative
python main.py --dash

# Data analysis only (no UI)
python main.py --data-only
```

## Project Structure

```
pbswi-youtube-analytics/
├── CLAUDE.md             # AI agent instructions
├── README.md             # This file
├── credentials/          # OAuth tokens (git-ignored)
│   ├── credentials.json  # From Google Cloud Console
│   └── token.json        # Generated after OAuth
├── config/
│   └── channels.yaml     # Channel configuration
├── data/
│   └── youtube_analytics.db  # SQLite database
├── planning/             # Session tracking and backlog
│   ├── README.md         # Current state and quick links
│   ├── progress.md       # Session handoff log
│   └── backlog.md        # Session scratchpad (items → GitHub Issues)
├── src/
│   ├── youtube_api/      # YouTube API integration (NEW)
│   │   ├── auth.py       # OAuth flow
│   │   ├── client.py     # API client
│   │   ├── data_loader.py # DataFrame loader
│   │   ├── database.py   # SQLite persistence
│   │   └── models.py     # Pydantic models
│   ├── analytics/        # Core analytics (from upstream)
│   ├── dashboards/       # Streamlit/Dash apps
│   └── utils/            # Utilities
└── main.py               # Entry point
```

## API Rate Limits

**YouTube Data API v3**:
- 10,000 quota units per day
- List operations: 1 unit per request
- Search: 100 units per request

**YouTube Analytics API**:
- 200 queries per day per user

**Strategy**: Cache aggressively in SQLite, only fetch new/updated data.

## Roadmap

See `planning/backlog.md` for current priorities. Key items:

- [ ] Dashboard integration with API data loader
- [ ] PBS-specific dashboard panels (Shorts conversion, archival content, show breakdown)
- [ ] Automated daily data refresh (launchd job)
- [ ] Historical backfill script
- [ ] Multi-channel comparison view

## Development

For AI agents working in this repo, see `CLAUDE.md` for:
- Agent workflow guidance
- Available MCP servers (The Library, Airtable, Readwise, Obsidian)
- Development commands and conventions

## Upstream

This fork is based on [YouTubeStudioDataAnalytics](https://github.com/DuongCaoNhan/YouTubeStudioDataAnalytics) by [@DuongCaoNhan](https://github.com/DuongCaoNhan). The original project provides CSV-based analytics; this fork adds live API integration and PBS Wisconsin customizations.

**Upstream sync**: Not planned due to significant architectural divergence (API vs. CSV workflows).

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contact

- **Maintainer**: [@MarkOnFire](https://github.com/MarkOnFire)
- **Organization**: PBS Wisconsin
- **Original Project**: [@DuongCaoNhan](https://github.com/DuongCaoNhan)
