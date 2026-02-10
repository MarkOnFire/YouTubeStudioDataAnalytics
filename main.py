"""
YouTube Analytics - Main Entry Point
A comprehensive analytics toolkit for YouTube Studio data.

This is the main entry point for the refactored YouTube Analytics project.
It provides a simple interface to run complete analytics or specific components.
"""

import sys
import argparse
import logging
from pathlib import Path
import os

# Add src to path for imports - Multiple methods for robustness
project_root = Path(__file__).parent
src_path = project_root / 'src'

# Method 1: Add to sys.path
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Method 2: Set PYTHONPATH environment variable for subprocess compatibility
current_pythonpath = os.environ.get('PYTHONPATH', '')
if str(src_path) not in current_pythonpath:
    os.environ['PYTHONPATH'] = f"{src_path}{os.pathsep}{current_pythonpath}"

# Now import our modules
try:
    from analytics import YouTubeAnalytics
    from utils import load_config
    from dashboards import StreamlitDashboard, DashDashboard
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(f"📁 Current working directory: {os.getcwd()}")
    print(f"📁 Project root: {project_root}")
    print(f"📁 Src path: {src_path}")
    print(f"📁 Src exists: {src_path.exists()}")
    print(f"🐍 Python path: {sys.path[:3]}...")  # Show first 3 entries
    raise

# Optional YouTube API integration
try:
    from youtube_api import YouTubeAPIDataLoader
    API_AVAILABLE = True
except ImportError:
    API_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_complete_analysis(config_path=None, output_dir="data/exports",
                          use_api=False, lookback_days=540):
    """
    Run the complete analytics pipeline.

    Args:
        config_path: Path to configuration file
        output_dir: Directory to save results
        use_api: Use YouTube API instead of CSV files
        lookback_days: Days of data to fetch when using API
    """
    print("🚀 Starting YouTube Analytics Pipeline...")

    try:
        # Load configuration
        config = load_config(config_path)

        # Initialize analytics system
        if use_api:
            if not API_AVAILABLE:
                print("❌ YouTube API module not available. Install google-api-python-client.")
                return None
            api_loader = YouTubeAPIDataLoader(lookback_days=lookback_days)
            analytics = YouTubeAnalytics(
                config=config.to_dict(),
                data_loader=api_loader
            )
        else:
            analytics = YouTubeAnalytics(
                videos_file=config.get('data.default_videos_file'),
                subscribers_file=config.get('data.default_subscribers_file'),
                config=config.to_dict()
            )

        # Run complete analysis
        results = analytics.run_complete_analysis(
            save_results=True,
            output_dir=output_dir
        )

        # Display summary
        analytics.display_summary_stats()

        print(f"✅ Analysis completed successfully!")
        print(f"📁 Results saved to: {output_dir}")

        return results

    except Exception as e:
        logger.error(f"Error in analysis pipeline: {e}")
        print(f"❌ Analysis failed: {e}")
        return None

def run_streamlit_dashboard():
    """Run the Streamlit dashboard."""
    print("🚀 Starting Streamlit Dashboard...")
    
    try:
        dashboard = StreamlitDashboard()
        dashboard.run()
        
    except Exception as e:
        logger.error(f"Error starting Streamlit dashboard: {e}")
        print(f"❌ Dashboard failed to start: {e}")

def run_dash_dashboard():
    """Run the Dash dashboard."""
    print("🚀 Starting Dash Dashboard...")
    
    try:
        dashboard = DashDashboard()
        dashboard.run()
        
    except Exception as e:
        logger.error(f"Error starting Dash dashboard: {e}")
        print(f"❌ Dashboard failed to start: {e}")

def run_data_analysis_only(config_path=None, use_api=False, lookback_days=540):
    """
    Run only data analysis without ML or export.

    Args:
        config_path: Path to configuration file
        use_api: Use YouTube API instead of CSV files
        lookback_days: Days of data to fetch when using API
    """
    print("🔍 Running Data Analysis Only...")

    try:
        # Load configuration
        config = load_config(config_path)

        # Initialize analytics system
        if use_api:
            if not API_AVAILABLE:
                print("❌ YouTube API module not available. Install google-api-python-client.")
                return None
            api_loader = YouTubeAPIDataLoader(lookback_days=lookback_days)
            analytics = YouTubeAnalytics(data_loader=api_loader)
        else:
            analytics = YouTubeAnalytics(
                videos_file=config.get('data.default_videos_file'),
                subscribers_file=config.get('data.default_subscribers_file')
            )

        # Load data and generate basic statistics
        analytics.load_data()
        summary = analytics.generate_summary_statistics()

        # Display results
        analytics.display_summary_stats()

        print("✅ Data analysis completed!")

        return summary

    except Exception as e:
        logger.error(f"Error in data analysis: {e}")
        print(f"❌ Data analysis failed: {e}")
        return None

def run_ml_prediction_demo(config_path=None, use_api=False, lookback_days=540):
    """
    Run ML prediction demonstration.

    Args:
        config_path: Path to configuration file
        use_api: Use YouTube API instead of CSV files
        lookback_days: Days of data to fetch when using API
    """
    print("🤖 Running ML Prediction Demo...")

    try:
        # Load configuration
        config = load_config(config_path)

        # Initialize analytics system
        if use_api:
            if not API_AVAILABLE:
                print("❌ YouTube API module not available. Install google-api-python-client.")
                return None
            api_loader = YouTubeAPIDataLoader(lookback_days=lookback_days)
            analytics = YouTubeAnalytics(data_loader=api_loader)
        else:
            analytics = YouTubeAnalytics(
                videos_file=config.get('data.default_videos_file'),
                subscribers_file=config.get('data.default_subscribers_file')
            )
        
        # Load data and train model
        analytics.load_data()
        training_results = analytics.train_prediction_model(hyperparameter_tuning=True)
        
        print(f"🎯 Model Performance:")
        print(f"   R² Score: {training_results['test_r2']:.3f}")
        print(f"   MAE: {training_results['test_mae']:.0f} views")
        print(f"   RMSE: {training_results['test_rmse']:.0f} views")
        
        # Demo prediction
        sample_features = {
            'Duration (minutes)': 15,
            'Likes': 150,
            'Comments': 25,
            'Like Rate (%)': 3.5,
            'Comment Rate (%)': 0.8,
            'Engagement Rate (%)': 4.3
        }
        
        prediction = analytics.predict_video_performance(sample_features)
        print(f"\n🔮 Sample Prediction:")
        print(f"   Input: 15-min video with 150 likes, 25 comments")
        print(f"   Predicted Views: {prediction['predicted_views']:,.0f}")
        
        print("✅ ML demo completed!")
        
        return training_results
        
    except Exception as e:
        logger.error(f"Error in ML demo: {e}")
        print(f"❌ ML demo failed: {e}")
        return None

def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description="YouTube Analytics - Comprehensive analytics for YouTube Studio data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --analysis           # Run complete analysis
  python main.py --streamlit          # Start Streamlit dashboard
  python main.py --dash               # Start Dash dashboard
  python main.py --data-only          # Data analysis only
  python main.py --ml-demo            # ML prediction demo
  python main.py --config config.json # Use custom config
        """
    )
    
    # Command options
    parser.add_argument('--analysis', action='store_true',
                       help='Run complete analytics pipeline')
    parser.add_argument('--streamlit', action='store_true',
                       help='Start Streamlit dashboard')
    parser.add_argument('--dash', action='store_true',
                       help='Start Dash dashboard')
    parser.add_argument('--data-only', action='store_true',
                       help='Run data analysis only (no ML or export)')
    parser.add_argument('--ml-demo', action='store_true',
                       help='Run ML prediction demonstration')
    
    # Data source options
    parser.add_argument('--api', action='store_true',
                       help='Use YouTube API instead of CSV files')
    parser.add_argument('--lookback-days', type=int, default=540,
                       help='Days of data to fetch from API (default: 540)')

    # Configuration options
    parser.add_argument('--config', type=str,
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default='data/exports',
                       help='Output directory for results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Display banner
    print("=" * 60)
    print("📺 YOUTUBE ANALYTICS - MODULAR EDITION")
    print("=" * 60)
    print("A comprehensive analytics toolkit for YouTube Studio data")
    print()
    
    # Execute based on arguments
    if args.analysis:
        run_complete_analysis(args.config, args.output,
                              use_api=args.api, lookback_days=args.lookback_days)
    elif args.streamlit:
        run_streamlit_dashboard()
    elif args.dash:
        run_dash_dashboard()
    elif args.data_only:
        run_data_analysis_only(args.config,
                               use_api=args.api, lookback_days=args.lookback_days)
    elif args.ml_demo:
        run_ml_prediction_demo(args.config,
                               use_api=args.api, lookback_days=args.lookback_days)
    else:
        # Default: show help and run interactive mode
        parser.print_help()
        print()
        
        # Interactive mode
        print("🎮 Interactive Mode")
        print("1. Complete Analysis")
        print("2. Streamlit Dashboard")
        print("3. Dash Dashboard") 
        print("4. Data Analysis Only")
        print("5. ML Demo")
        print("0. Exit")
        
        try:
            choice = input("\nSelect option (0-5): ").strip()
            
            if choice == '1':
                run_complete_analysis(args.config, args.output)
            elif choice == '2':
                run_streamlit_dashboard()
            elif choice == '3':
                run_dash_dashboard()
            elif choice == '4':
                run_data_analysis_only(args.config)
            elif choice == '5':
                run_ml_prediction_demo(args.config)
            elif choice == '0':
                print("👋 Goodbye!")
            else:
                print("❌ Invalid choice")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
