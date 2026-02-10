"""
Core YouTube Analytics Module
Main orchestrator class that coordinates all analytics operations.
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import logging
from pathlib import Path

from .data_loader import DataLoader
from .visualizations import ChartGenerator
from .ml_predictor import MLPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class YouTubeAnalytics:
    """
    Main YouTube Analytics class that orchestrates all analysis operations.
    """
    
    def __init__(self,
                 videos_file: str = "data/sample/videos.csv",
                 subscribers_file: str = "data/sample/subscribers.csv",
                 config: Optional[Dict[str, Any]] = None,
                 data_loader=None):
        """
        Initialize the YouTube Analytics system.

        Args:
            videos_file: Path to videos CSV file
            subscribers_file: Path to subscribers CSV file
            config: Configuration dictionary
            data_loader: Optional pre-configured data loader (e.g. YouTubeAPIDataLoader).
                         If provided, videos_file and subscribers_file are ignored.
        """
        self.config = config or {}

        # Initialize components
        if data_loader is not None:
            self.data_loader = data_loader
        else:
            self.data_loader = DataLoader(videos_file, subscribers_file)
        self.chart_generator = ChartGenerator(
            theme_colors=self.config.get('colors', None)
        )
        self.ml_predictor = MLPredictor(
            model_type=self.config.get('ml_model_type', 'linear')
        )
        
        # Data storage
        self.videos_df: Optional[pd.DataFrame] = None
        self.subscribers_df: Optional[pd.DataFrame] = None
        
        # Analysis results
        self.analysis_results: Dict[str, Any] = {}
        
        logger.info("YouTube Analytics system initialized")
    
    def load_data(self) -> None:
        """Load all data using the data loader."""
        try:
            logger.info("Loading YouTube analytics data...")
            
            self.videos_df, self.subscribers_df = self.data_loader.load_all_data()
            
            # Store data summary
            self.analysis_results['data_summary'] = self.data_loader.get_data_summary()
            
            logger.info("Data loading completed successfully")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def generate_summary_statistics(self) -> Dict[str, Any]:
        """
        Generate comprehensive summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        if self.videos_df is None:
            raise ValueError("No data loaded. Please run load_data() first.")
        
        try:
            summary = {
                'overview': {
                    'total_videos': len(self.videos_df),
                    'total_views': int(self.videos_df['Views'].sum()),
                    'total_likes': int(self.videos_df['Likes'].sum()),
                    'total_comments': int(self.videos_df['Comments'].sum()),
                    'date_range': {
                        'start': self.videos_df['Publish Date'].min().strftime('%Y-%m-%d'),
                        'end': self.videos_df['Publish Date'].max().strftime('%Y-%m-%d')
                    }
                },
                'engagement_metrics': {
                    'average_like_rate': float(self.videos_df['Like Rate (%)'].mean()),
                    'average_comment_rate': float(self.videos_df['Comment Rate (%)'].mean()),
                    'average_engagement_rate': float(self.videos_df['Engagement Rate (%)'].mean()),
                    'median_views': float(self.videos_df['Views'].median()),
                    'std_views': float(self.videos_df['Views'].std())
                },
                'top_performers': {
                    'most_viewed': {
                        'title': self.videos_df.loc[self.videos_df['Views'].idxmax(), 'Title'],
                        'views': int(self.videos_df['Views'].max())
                    },
                    'highest_like_rate': {
                        'title': self.videos_df.loc[self.videos_df['Like Rate (%)'].idxmax(), 'Title'],
                        'rate': float(self.videos_df['Like Rate (%)'].max())
                    },
                    'most_comments': {
                        'title': self.videos_df.loc[self.videos_df['Comments'].idxmax(), 'Title'],
                        'comments': int(self.videos_df['Comments'].max())
                    }
                }
            }
            
            # Add subscriber summary if available
            if self.subscribers_df is not None:
                summary['subscriber_metrics'] = {
                    'total_gained': int(self.subscribers_df['Subscribers Gained'].sum()),
                    'total_lost': int(self.subscribers_df['Subscribers Lost'].sum()),
                    'net_growth': int(self.subscribers_df['Net Subscribers'].sum()),
                    'average_daily_gain': float(self.subscribers_df['Subscribers Gained'].mean()),
                    'best_day': {
                        'date': self.subscribers_df.loc[
                            self.subscribers_df['Subscribers Gained'].idxmax(), 'Date'
                        ].strftime('%Y-%m-%d'),
                        'gained': int(self.subscribers_df['Subscribers Gained'].max())
                    }
                }
            
            self.analysis_results['summary_statistics'] = summary
            logger.info("Summary statistics generated successfully")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary statistics: {e}")
            raise
    
    def create_all_visualizations(self, save_charts: bool = False,
                                output_dir: str = "data/exports/charts") -> Dict[str, Any]:
        """
        Create all standard visualizations.
        
        Args:
            save_charts: Whether to save charts to files
            output_dir: Directory to save charts
            
        Returns:
            Dictionary with chart objects
        """
        if self.videos_df is None:
            raise ValueError("No data loaded. Please run load_data() first.")
        
        try:
            logger.info("Creating all visualizations...")
            
            charts = {}
            
            # Views timeline
            charts['views_timeline'] = self.chart_generator.create_views_timeline(self.videos_df)
            
            # Engagement comparison
            charts['engagement_comparison'] = self.chart_generator.create_engagement_comparison(self.videos_df)
            
            # Engagement rates
            charts['engagement_rates'] = self.chart_generator.create_engagement_rates_chart(self.videos_df)
            
            # Correlation heatmap
            charts['correlation_heatmap'] = self.chart_generator.create_correlation_heatmap(self.videos_df)
            
            # Performance scatter
            charts['performance_scatter'] = self.chart_generator.create_performance_scatter(self.videos_df)
            
            # Top performers
            charts['top_performers'] = self.chart_generator.create_top_performers_chart(
                self.videos_df, metric='Views', chart_type='horizontal_bar'
            )
            
            # Distribution histograms
            charts['views_distribution'] = self.chart_generator.create_distribution_histogram(
                self.videos_df, 'Views'
            )
            charts['engagement_distribution'] = self.chart_generator.create_distribution_histogram(
                self.videos_df, 'Engagement Rate (%)'
            )
            
            # Subscriber activity (if available)
            if self.subscribers_df is not None:
                charts['subscriber_activity'] = self.chart_generator.create_subscriber_activity_chart(
                    self.subscribers_df
                )
            
            # Multi-metric dashboard
            charts['dashboard'] = self.chart_generator.create_multi_metric_dashboard(
                self.videos_df, self.subscribers_df
            )
            
            # Save charts if requested
            if save_charts:
                self._save_all_charts(charts, output_dir)
            
            self.analysis_results['visualizations'] = charts
            logger.info(f"Created {len(charts)} visualizations successfully")
            
            return charts
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            raise
    
    def train_prediction_model(self, 
                             feature_columns: Optional[List[str]] = None,
                             model_type: Optional[str] = None,
                             hyperparameter_tuning: bool = False) -> Dict[str, Any]:
        """
        Train machine learning model for view prediction.
        
        Args:
            feature_columns: Columns to use as features
            model_type: Type of model to train
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with training results
        """
        if self.videos_df is None:
            raise ValueError("No data loaded. Please run load_data() first.")
        
        try:
            logger.info("Training prediction model...")
            
            # Update model type if specified
            if model_type:
                self.ml_predictor = MLPredictor(model_type)
            
            # Train the model
            training_results = self.ml_predictor.train_model(
                self.videos_df,
                feature_columns=feature_columns,
                perform_hyperparameter_tuning=hyperparameter_tuning
            )
            
            # Analyze feature importance
            feature_analysis = self.ml_predictor.analyze_feature_importance()
            training_results['feature_analysis'] = feature_analysis
            
            self.analysis_results['ml_training'] = training_results
            logger.info("Model training completed successfully")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Error training prediction model: {e}")
            raise
    
    def predict_video_performance(self, video_features: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict performance for a new video.
        
        Args:
            video_features: Dictionary with video features
            
        Returns:
            Prediction results
        """
        try:
            if not self.ml_predictor.is_trained:
                logger.warning("Model not trained, training with default settings...")
                self.train_prediction_model()
            
            prediction = self.ml_predictor.predict_views(video_features)
            logger.info(f"Prediction made: {prediction['predicted_views']:.0f} views")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def run_complete_analysis(self, 
                            save_results: bool = True,
                            output_dir: str = "data/exports") -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.
        
        Args:
            save_results: Whether to save results
            output_dir: Directory to save results
            
        Returns:
            Complete analysis results
        """
        try:
            logger.info("Starting complete YouTube analytics pipeline...")
            
            # Load data
            self.load_data()
            
            # Generate summary statistics
            self.generate_summary_statistics()
            
            # Create visualizations
            self.create_all_visualizations(save_charts=save_results, output_dir=f"{output_dir}/charts")
            
            # Train ML model
            try:
                self.train_prediction_model(hyperparameter_tuning=True)
            except Exception as e:
                logger.warning(f"ML training failed: {e}")
                self.analysis_results['ml_training'] = {'error': str(e)}
            
            # Data quality analysis
            quality_report = self.data_loader.validate_data_quality()
            self.analysis_results['data_quality'] = quality_report
            
            # Generate insights
            insights = self.generate_insights()
            self.analysis_results['insights'] = insights
            
            # Save results if requested
            if save_results:
                self.export_results(output_dir)
            
            logger.info("Complete analysis pipeline finished successfully")
            return self.analysis_results
            
        except Exception as e:
            logger.error(f"Error in complete analysis: {e}")
            raise
    
    def generate_insights(self) -> Dict[str, Any]:
        """
        Generate actionable insights from the analysis.
        
        Returns:
            Dictionary with insights and recommendations
        """
        if self.videos_df is None:
            return {'error': 'No data available for insights'}
        
        try:
            insights = {
                'content_strategy': [],
                'performance_optimization': [],
                'audience_engagement': [],
                'growth_opportunities': []
            }
            
            # Content strategy insights
            avg_duration = self.videos_df['Duration (minutes)'].mean()
            top_duration = self.videos_df.nlargest(3, 'Views')['Duration (minutes)'].mean()
            
            if abs(top_duration - avg_duration) > 2:
                if top_duration > avg_duration:
                    insights['content_strategy'].append(
                        f"Consider creating longer videos (top performers avg {top_duration:.1f} min vs channel avg {avg_duration:.1f} min)"
                    )
                else:
                    insights['content_strategy'].append(
                        f"Consider creating shorter videos (top performers avg {top_duration:.1f} min vs channel avg {avg_duration:.1f} min)"
                    )
            
            # Engagement insights
            avg_like_rate = self.videos_df['Like Rate (%)'].mean()
            top_like_rate = self.videos_df['Like Rate (%)'].max()
            
            if top_like_rate > avg_like_rate * 1.5:
                best_video = self.videos_df.loc[self.videos_df['Like Rate (%)'].idxmax(), 'Title']
                insights['audience_engagement'].append(
                    f"Analyze '{best_video}' for engagement best practices (like rate: {top_like_rate:.2f}%)"
                )
            
            # Performance optimization
            correlation_matrix = self.videos_df[['Views', 'Likes', 'Comments', 'Duration (minutes)']].corr()
            duration_views_corr = correlation_matrix.loc['Duration (minutes)', 'Views']
            
            if abs(duration_views_corr) > 0.3:
                correlation_direction = "positively" if duration_views_corr > 0 else "negatively"
                insights['performance_optimization'].append(
                    f"Video duration is {correlation_direction} correlated with views (r={duration_views_corr:.2f})"
                )
            
            # Growth opportunities
            if self.subscribers_df is not None:
                recent_growth = self.subscribers_df.tail(7)['Net Subscribers'].mean()
                overall_growth = self.subscribers_df['Net Subscribers'].mean()
                
                if recent_growth > overall_growth * 1.2:
                    insights['growth_opportunities'].append(
                        "Recent subscriber growth is accelerating - consider increasing upload frequency"
                    )
                elif recent_growth < overall_growth * 0.8:
                    insights['growth_opportunities'].append(
                        "Recent subscriber growth is slowing - review content strategy and engagement tactics"
                    )
            
            # Upload timing insights
            self.videos_df['Upload_Day'] = self.videos_df['Publish Date'].dt.day_name()
            day_performance = self.videos_df.groupby('Upload_Day')['Views'].mean().sort_values(ascending=False)
            
            if len(day_performance) > 1:
                best_day = day_performance.index[0]
                worst_day = day_performance.index[-1]
                performance_diff = (day_performance.iloc[0] - day_performance.iloc[-1]) / day_performance.iloc[-1] * 100
                
                if performance_diff > 20:
                    insights['content_strategy'].append(
                        f"Consider uploading more on {best_day} (avg {day_performance.iloc[0]:.0f} views vs {worst_day} {day_performance.iloc[-1]:.0f} views)"
                    )
            
            logger.info("Insights generated successfully")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {'error': str(e)}
    
    def export_results(self, output_dir: str = "data/exports") -> None:
        """
        Export all analysis results to files.
        
        Args:
            output_dir: Directory to save results
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Export processed data
            self.data_loader.export_processed_data(output_dir)
            
            # Export analysis results as JSON
            import json
            
            # Convert numpy types to Python types for JSON serialization
            serializable_results = self._make_json_serializable(self.analysis_results)
            
            with open(output_path / "analysis_results.json", 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            # Export to Excel
            self._export_to_excel(output_path / "youtube_analytics_report.xlsx")
            
            # Save ML model if trained
            if self.ml_predictor.is_trained:
                self.ml_predictor.save_model(str(output_path / "ml_model.joblib"))
            
            logger.info(f"Results exported to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            raise
    
    def _save_all_charts(self, charts: Dict[str, Any], output_dir: str) -> None:
        """Save all charts to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for chart_name, chart_fig in charts.items():
            if chart_fig is not None:
                chart_path = output_path / f"{chart_name}.html"
                self.chart_generator.save_chart(chart_fig, str(chart_path))
    
    def _export_to_excel(self, filepath: str) -> None:
        """Export analysis results to Excel."""
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Video data
            if self.videos_df is not None:
                self.videos_df.to_excel(writer, sheet_name='Video Analytics', index=False)
            
            # Subscriber data
            if self.subscribers_df is not None:
                self.subscribers_df.to_excel(writer, sheet_name='Subscriber Activity', index=False)
            
            # Summary statistics
            if 'summary_statistics' in self.analysis_results:
                summary_data = self._flatten_dict(self.analysis_results['summary_statistics'])
                summary_df = pd.DataFrame(list(summary_data.items()), columns=['Metric', 'Value'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # ML results
            if 'ml_training' in self.analysis_results and 'error' not in self.analysis_results['ml_training']:
                ml_data = self._flatten_dict(self.analysis_results['ml_training'])
                ml_df = pd.DataFrame(list(ml_data.items()), columns=['Metric', 'Value'])
                ml_df.to_excel(writer, sheet_name='ML Results', index=False)
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary for export."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def display_summary_stats(self) -> None:
        """Display formatted summary statistics."""
        if 'summary_statistics' not in self.analysis_results:
            self.generate_summary_statistics()
        
        summary = self.analysis_results['summary_statistics']
        
        print("\n" + "="*60)
        print("📈 YOUTUBE CHANNEL ANALYTICS SUMMARY")
        print("="*60)
        
        # Overview
        overview = summary['overview']
        print(f"📺 Total Videos: {overview['total_videos']:,}")
        print(f"👀 Total Views: {overview['total_views']:,}")
        print(f"👍 Total Likes: {overview['total_likes']:,}")
        print(f"💬 Total Comments: {overview['total_comments']:,}")
        
        # Engagement metrics
        engagement = summary['engagement_metrics']
        print(f"📊 Average Like Rate: {engagement['average_like_rate']:.2f}%")
        print(f"💭 Average Comment Rate: {engagement['average_comment_rate']:.2f}%")
        
        # Top performers
        top = summary['top_performers']
        print(f"\n🏆 TOP PERFORMERS:")
        print(f"   Most Viewed: {top['most_viewed']['title'][:50]}... - {top['most_viewed']['views']:,} views")
        print(f"   Highest Like Rate: {top['highest_like_rate']['title'][:50]}... - {top['highest_like_rate']['rate']:.2f}%")
        
        # Subscriber metrics if available
        if 'subscriber_metrics' in summary:
            subs = summary['subscriber_metrics']
            print(f"\n👥 SUBSCRIBER METRICS:")
            print(f"   Total Gained: {subs['total_gained']:,}")
            print(f"   Total Lost: {subs['total_lost']:,}")
            print(f"   Net Growth: {subs['net_growth']:,}")
            print(f"   Best Day: {subs['best_day']['date']} ({subs['best_day']['gained']} gained)")
    
    def get_analysis_results(self) -> Dict[str, Any]:
        """Get all analysis results."""
        return self.analysis_results
