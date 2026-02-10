"""
Tests for the YouTube API integration module.

Covers:
- extract_show_name() title parsing logic
- YouTubeAPIDataLoader interface compatibility and data quality validation
- AnalyticsDatabase CRUD operations (using temp SQLite files)
- Pydantic model validation
"""

import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

# Add project root to path so 'src' package imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.youtube_api.data_loader import extract_show_name, YouTubeAPIDataLoader
from src.youtube_api.database import AnalyticsDatabase
from src.youtube_api.models import Video, ChannelStats, DailyAnalytics, VideoAnalytics


# ---------------------------------------------------------------------------
# extract_show_name() tests
# ---------------------------------------------------------------------------

class TestExtractShowName:
    """Tests for PBS Wisconsin title pattern parsing."""

    def test_standard_pattern(self):
        """Standard: 'Video Title | SHOW NAME' -> 'SHOW NAME'."""
        assert extract_show_name("Great Episode | PBS Wisconsin") == "PBS Wisconsin"

    def test_wisconsin_life_exception(self):
        """Exception: 'Wisconsin Life | ...' -> 'Wisconsin Life'."""
        assert extract_show_name("Wisconsin Life | A Day in Madison") == "Wisconsin Life"

    def test_no_pipe(self):
        """No pipe character -> 'Uncategorized'."""
        assert extract_show_name("Just a Title") == "Uncategorized"

    def test_multiple_pipes(self):
        """Multiple pipes: takes last segment as show name."""
        assert extract_show_name("Part 1 | Part 2 | SHOW NAME") == "SHOW NAME"

    def test_empty_string(self):
        """Empty string -> 'Uncategorized'."""
        assert extract_show_name("") == "Uncategorized"

    def test_pipe_with_whitespace(self):
        """Pipe surrounded by spaces is the delimiter; plain pipe is not."""
        assert extract_show_name("No|Spaces") == "Uncategorized"

    def test_wisconsin_life_multiple_pipes(self):
        """Wisconsin Life exception still works with extra pipe segments."""
        assert extract_show_name("Wisconsin Life | Segment | Extra") == "Wisconsin Life"

    def test_trailing_whitespace(self):
        """Show name trailing whitespace is stripped."""
        assert extract_show_name("Title | SHOW NAME  ") == "SHOW NAME"


# ---------------------------------------------------------------------------
# YouTubeAPIDataLoader interface tests
# ---------------------------------------------------------------------------

class TestDataLoaderInterface:
    """Verify YouTubeAPIDataLoader exposes the methods that
    YouTubeAnalytics.run_complete_analysis() expects."""

    def test_has_load_all_data(self):
        assert hasattr(YouTubeAPIDataLoader, 'load_all_data')
        assert callable(getattr(YouTubeAPIDataLoader, 'load_all_data'))

    def test_has_get_data_summary(self):
        assert hasattr(YouTubeAPIDataLoader, 'get_data_summary')
        assert callable(getattr(YouTubeAPIDataLoader, 'get_data_summary'))

    def test_has_validate_data_quality(self):
        assert hasattr(YouTubeAPIDataLoader, 'validate_data_quality')
        assert callable(getattr(YouTubeAPIDataLoader, 'validate_data_quality'))

    def test_has_export_processed_data(self):
        assert hasattr(YouTubeAPIDataLoader, 'export_processed_data')
        assert callable(getattr(YouTubeAPIDataLoader, 'export_processed_data'))

    def test_has_load_videos_data(self):
        assert hasattr(YouTubeAPIDataLoader, 'load_videos_data')

    def test_has_load_subscribers_data(self):
        assert hasattr(YouTubeAPIDataLoader, 'load_subscribers_data')


# ---------------------------------------------------------------------------
# validate_data_quality() structure tests
# ---------------------------------------------------------------------------

class TestValidateDataQuality:
    """Ensure validate_data_quality() returns the expected dict shape."""

    @patch('src.youtube_api.data_loader.YouTubeAPIClient')
    def test_quality_report_structure_no_data(self, mock_client_cls):
        """With no loaded data, report has correct top-level keys and defaults."""
        loader = YouTubeAPIDataLoader.__new__(YouTubeAPIDataLoader)
        loader.videos_df = None
        loader.subscribers_df = None
        loader.client = mock_client_cls.return_value

        report = loader.validate_data_quality()

        assert 'videos' in report
        assert 'subscribers' in report
        assert report['videos']['issues'] == []
        assert report['videos']['quality_score'] == 100
        assert report['subscribers']['issues'] == []
        assert report['subscribers']['quality_score'] == 100

    @patch('src.youtube_api.data_loader.YouTubeAPIClient')
    def test_quality_report_with_clean_data(self, mock_client_cls):
        """Clean data should yield quality_score 100 (or 95 if outliers)."""
        loader = YouTubeAPIDataLoader.__new__(YouTubeAPIDataLoader)
        loader.client = mock_client_cls.return_value

        loader.videos_df = pd.DataFrame({
            'Title': ['A | Show', 'B | Show'],
            'Views': [1000, 1200],
            'Likes': [50, 60],
            'Comments': [5, 6],
        })
        loader.subscribers_df = pd.DataFrame({
            'Subscribers Gained': [10, 20],
            'Subscribers Lost': [1, 2],
        })

        report = loader.validate_data_quality()

        assert isinstance(report['videos']['issues'], list)
        assert isinstance(report['videos']['quality_score'], int)
        assert isinstance(report['subscribers']['issues'], list)
        assert isinstance(report['subscribers']['quality_score'], int)

    @patch('src.youtube_api.data_loader.YouTubeAPIClient')
    def test_quality_report_negative_subscribers(self, mock_client_cls):
        """Negative subscriber values should produce quality issues."""
        loader = YouTubeAPIDataLoader.__new__(YouTubeAPIDataLoader)
        loader.client = mock_client_cls.return_value
        loader.videos_df = None

        loader.subscribers_df = pd.DataFrame({
            'Subscribers Gained': [-5, 20],
            'Subscribers Lost': [1, 2],
        })

        report = loader.validate_data_quality()

        assert len(report['subscribers']['issues']) > 0
        assert report['subscribers']['quality_score'] < 100


# ---------------------------------------------------------------------------
# AnalyticsDatabase CRUD tests
# ---------------------------------------------------------------------------

class TestAnalyticsDatabase:
    """Tests for AnalyticsDatabase using temporary SQLite files."""

    @pytest.fixture
    def db(self, tmp_path):
        """Create a fresh database in a temp directory."""
        db_path = tmp_path / "test_analytics.db"
        return AnalyticsDatabase(db_path=db_path)

    def _make_video(self, video_id="vid_001", **overrides):
        """Helper to build a video dict with sensible defaults."""
        data = {
            'video_id': video_id,
            'title': 'Test Video | Test Show',
            'description': 'A test video',
            'published_at': datetime(2024, 6, 15),
            'channel_id': 'UC_test',
            'channel_title': 'Test Channel',
            'show_name': 'Test Show',
            'duration_minutes': 5.0,
            'is_short': False,
            'views': 1000,
            'likes': 50,
            'comments': 10,
            'engagement_rate': 6.0,
            'views_per_day': 5.0,
            'days_since_publication': 200,
        }
        data.update(overrides)
        return data

    # -- upsert_video / get_all_videos --

    def test_upsert_and_retrieve_single_video(self, db):
        """Insert one video and retrieve it."""
        video = self._make_video()
        db.upsert_video(video)

        results = db.get_all_videos()
        assert len(results) == 1
        assert results[0]['video_id'] == 'vid_001'
        assert results[0]['title'] == 'Test Video | Test Show'

    def test_upsert_updates_existing_video(self, db):
        """Upserting with same video_id updates the record."""
        db.upsert_video(self._make_video(views=1000))
        db.upsert_video(self._make_video(views=2000))

        results = db.get_all_videos()
        assert len(results) == 1
        assert results[0]['views'] == 2000

    def test_get_all_videos_channel_filter(self, db):
        """get_all_videos filters by channel_id when provided."""
        db.upsert_video(self._make_video('v1', channel_id='UC_a'))
        db.upsert_video(self._make_video('v2', channel_id='UC_b'))

        results = db.get_all_videos(channel_id='UC_a')
        assert len(results) == 1
        assert results[0]['channel_id'] == 'UC_a'

    # -- upsert_videos_bulk --

    def test_bulk_upsert(self, db):
        """Bulk insert multiple videos at once."""
        videos = [
            self._make_video('v1', title='First | Show A'),
            self._make_video('v2', title='Second | Show B'),
            self._make_video('v3', title='Third | Show A'),
        ]
        count = db.upsert_videos_bulk(videos)

        assert count == 3
        assert len(db.get_all_videos()) == 3

    def test_bulk_upsert_updates_existing(self, db):
        """Bulk upsert updates records that already exist."""
        db.upsert_video(self._make_video('v1', views=100))

        updated = [self._make_video('v1', views=999)]
        db.upsert_videos_bulk(updated)

        results = db.get_all_videos()
        assert len(results) == 1
        assert results[0]['views'] == 999

    # -- add_daily_stats --

    def test_add_and_query_daily_stats(self, db):
        """Add daily stats and verify they persist."""
        date = datetime(2024, 8, 1)
        db.add_daily_stats('vid_001', date, {
            'views': 500,
            'likes': 25,
            'comments': 3,
            'watch_time_minutes': 120.5,
            'subscribers_gained': 2,
        })

        # Query via session to verify
        from src.youtube_api.database import DailyStatsTable
        with db.get_session() as session:
            row = session.query(DailyStatsTable).filter_by(video_id='vid_001').first()
            assert row is not None
            assert row.views == 500
            assert row.watch_time_minutes == 120.5

    def test_daily_stats_upsert(self, db):
        """Inserting same video_id+date updates existing stats."""
        date = datetime(2024, 8, 1)
        db.add_daily_stats('vid_001', date, {'views': 100})
        db.add_daily_stats('vid_001', date, {'views': 200})

        from src.youtube_api.database import DailyStatsTable
        with db.get_session() as session:
            rows = session.query(DailyStatsTable).filter_by(video_id='vid_001').all()
            assert len(rows) == 1
            assert rows[0].views == 200

    # -- add_channel_snapshot --

    def test_add_channel_snapshot(self, db):
        """Add a channel snapshot and retrieve it."""
        date = datetime.now() - timedelta(days=10)
        db.add_channel_snapshot('UC_test', date, {
            'subscriber_count': 50000,
            'video_count': 300,
            'view_count': 10000000,
            'daily_views': 5000,
            'daily_watch_time_minutes': 2500.0,
            'daily_subscribers_gained': 15,
            'daily_subscribers_lost': 3,
        })

        history = db.get_channel_history('UC_test', days=365)
        assert len(history) == 1
        assert history[0]['subscriber_count'] == 50000

    # -- get_archival_videos --

    def test_get_archival_videos(self, db):
        """Videos older than threshold appear in archival results."""
        old_date = datetime.now() - timedelta(days=400)
        recent_date = datetime.now() - timedelta(days=30)

        db.upsert_video(self._make_video('old', published_at=old_date))
        db.upsert_video(self._make_video('new', published_at=recent_date))

        archival = db.get_archival_videos(months_threshold=12)
        ids = [v['video_id'] for v in archival]
        assert 'old' in ids
        assert 'new' not in ids

    # -- get_show_summary --

    def test_get_show_summary(self, db):
        """Show summary aggregates correctly across videos."""
        db.upsert_videos_bulk([
            self._make_video('v1', show_name='Show A', views=1000, likes=50),
            self._make_video('v2', show_name='Show A', views=2000, likes=100),
            self._make_video('v3', show_name='Show B', views=500, likes=20),
        ])

        summary = db.get_show_summary()
        shows = {s['show_name']: s for s in summary}

        assert 'Show A' in shows
        assert 'Show B' in shows
        assert shows['Show A']['video_count'] == 2
        assert shows['Show A']['total_views'] == 3000
        assert shows['Show B']['video_count'] == 1

    # -- get_shorts_vs_longform --

    def test_shorts_vs_longform(self, db):
        """Shorts and longform stats are separated correctly."""
        db.upsert_videos_bulk([
            self._make_video('s1', is_short=True, views=5000),
            self._make_video('s2', is_short=True, views=3000),
            self._make_video('l1', is_short=False, views=10000),
        ])

        result = db.get_shorts_vs_longform()
        assert result['shorts']['count'] == 2
        assert result['shorts']['total_views'] == 8000
        assert result['longform']['count'] == 1
        assert result['longform']['total_views'] == 10000


# ---------------------------------------------------------------------------
# Pydantic model tests
# ---------------------------------------------------------------------------

class TestPydanticModels:
    """Test Pydantic data model validation and computed fields."""

    def test_video_is_short_true(self):
        """Videos <= 1.0 minute are Shorts."""
        v = Video(
            video_id='abc',
            title='Short Clip',
            published_at=datetime.now(),
            channel_id='UC_test',
            channel_title='Test',
            duration_minutes=0.5,
            duration_iso='PT30S',
        )
        assert v.is_short is True

    def test_video_is_short_false(self):
        """Videos > 1.0 minute are not Shorts."""
        v = Video(
            video_id='abc',
            title='Long Video',
            published_at=datetime.now(),
            channel_id='UC_test',
            channel_title='Test',
            duration_minutes=10.0,
            duration_iso='PT10M',
        )
        assert v.is_short is False

    def test_video_show_name_standard(self):
        """Video model extracts show name from standard title."""
        v = Video(
            video_id='abc',
            title='Great Content | PBS Wisconsin',
            published_at=datetime.now(),
            channel_id='UC_test',
            channel_title='Test',
            duration_minutes=5.0,
            duration_iso='PT5M',
        )
        assert v.show_name == 'PBS Wisconsin'

    def test_video_show_name_wisconsin_life(self):
        """Video model handles Wisconsin Life exception."""
        v = Video(
            video_id='abc',
            title='Wisconsin Life | Madison Farmers Market',
            published_at=datetime.now(),
            channel_id='UC_test',
            channel_title='Test',
            duration_minutes=5.0,
            duration_iso='PT5M',
        )
        assert v.show_name == 'Wisconsin Life'

    def test_video_engagement_rate_zero_views(self):
        """Engagement rate is 0.0 when views are zero."""
        v = Video(
            video_id='abc',
            title='No Views',
            published_at=datetime.now(),
            channel_id='UC_test',
            channel_title='Test',
            duration_minutes=5.0,
            duration_iso='PT5M',
            views=0,
            likes=0,
            comments=0,
        )
        assert v.engagement_rate == 0.0

    def test_video_engagement_rate_calculation(self):
        """Engagement rate = (likes + comments) / views * 100."""
        v = Video(
            video_id='abc',
            title='Popular',
            published_at=datetime.now(),
            channel_id='UC_test',
            channel_title='Test',
            duration_minutes=5.0,
            duration_iso='PT5M',
            views=1000,
            likes=80,
            comments=20,
        )
        assert v.engagement_rate == pytest.approx(10.0)

    def test_daily_analytics_net_subscribers(self):
        """DailyAnalytics computes net subscribers correctly."""
        d = DailyAnalytics(
            date=datetime.now(),
            subscribers_gained=50,
            subscribers_lost=12,
        )
        assert d.net_subscribers == 38

    def test_channel_stats_creation(self):
        """ChannelStats model instantiates with required fields."""
        cs = ChannelStats(
            channel_id='UC_test',
            title='Test Channel',
            uploads_playlist_id='UU_test',
            published_at='2020-01-01T00:00:00Z',
        )
        assert cs.channel_id == 'UC_test'
        assert cs.subscriber_count == 0

    def test_video_analytics_defaults(self):
        """VideoAnalytics defaults numeric fields to 0."""
        va = VideoAnalytics(video_id='abc')
        assert va.views == 0
        assert va.watch_time_minutes == 0.0
        assert va.subscribers_gained == 0
