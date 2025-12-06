"""
Enhanced JSON output module for NHL predictions API
Generates comprehensive JSON files suitable for web app consumption
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PredictionJSONExporter:
    """Export predictions to comprehensive JSON format for API consumption"""
    
    def __init__(self, output_dir: str = 'predictions_json'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def export_predictions(self, 
                          predictions_df: pd.DataFrame,
                          betting_engine_portfolio: Dict,
                          metadata: Optional[Dict] = None) -> Dict[str, str]:
        """
        Export predictions to multiple JSON files for different API endpoints
        
        Returns:
            Dict of endpoint names to file paths
        """
        if len(predictions_df) == 0:
            logger.warning("No predictions to export")
            return {}
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        date_str = datetime.now().strftime('%Y-%m-%d')
        
        # Generate comprehensive JSON structures
        endpoints = {
            'summary': self._generate_summary_json(predictions_df, betting_engine_portfolio, date_str),
            'games': self._generate_games_json(predictions_df, date_str),
            'betting': self._generate_betting_json(predictions_df, betting_engine_portfolio, date_str),
            'analytics': self._generate_analytics_json(predictions_df, date_str),
            'portfolio': self._generate_portfolio_json(predictions_df, betting_engine_portfolio, date_str)
        }
        
        # Add metadata if provided
        if metadata:
            endpoints['metadata'] = metadata
        
        # Save each endpoint
        saved_files = {}
        for endpoint_name, data in endpoints.items():
            filename = f"{endpoint_name}_{date_str}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
            saved_files[endpoint_name] = str(filepath)
            logger.info(f"Saved {endpoint_name} endpoint: {filepath}")
        
        # Also save a "latest" version without timestamp for easy API access
        for endpoint_name, data in endpoints.items():
            latest_filename = f"{endpoint_name}_latest.json"
            latest_filepath = self.output_dir / latest_filename
            
            with open(latest_filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
            
            logger.info(f"Saved latest {endpoint_name}: {latest_filepath}")
        
        return saved_files
    
    def _generate_summary_json(self, 
                               predictions_df: pd.DataFrame,
                               portfolio: Dict,
                               date: str) -> Dict:
        """Generate high-level summary for dashboard"""
        bet_games = predictions_df[predictions_df['action'] == 'BET']
        
        return {
            'api_version': '1.0',
            'generated_at': datetime.now().isoformat(),
            'date': date,
            'summary': {
                'total_games': len(predictions_df),
                'games_with_bets': len(bet_games),
                'games_to_skip': len(predictions_df) - len(bet_games),
                'total_stake': float(bet_games['bet_size'].sum()) if len(bet_games) > 0 else 0,
                'total_expected_value': float(bet_games['expected_value'].sum()) if len(bet_games) > 0 else 0,
                'average_edge': float(bet_games['edge'].mean()) if len(bet_games) > 0 else 0,
                'portfolio_exposure_pct': float(portfolio.get('exposure_pct', 0)),
                'portfolio_sharpe': float(portfolio.get('portfolio_sharpe', 0))
            },
            'top_picks': self._get_top_picks(bet_games, n=3),
            'edge_distribution': self._calculate_edge_distribution(bet_games),
            'confidence_levels': self._calculate_confidence_levels(predictions_df)
        }
    
    def _generate_games_json(self, predictions_df: pd.DataFrame, date: str) -> Dict:
        """Generate detailed game-by-game predictions"""
        games = []
        
        for idx, pred in predictions_df.iterrows():
            game_data = {
                'game_id': str(pred['game_id']),
                'matchup': {
                    'away_team': pred['away_team'],
                    'home_team': pred['home_team'],
                    'venue': pred['venue'],
                    'game_time_utc': pred['game_time'],
                    'display_text': pred['matchup']
                },
                'prediction': {
                    'predicted_winner': pred['predicted_winner'],
                    'predicted_team': pred['predicted_team'],
                    'home_win_probability': float(pred['home_win_prob']),
                    'away_win_probability': float(pred['away_win_prob']),
                    'model_confidence': float(pred['model_probability']),
                    'calibrated_accuracy': float(pred['empirical_accuracy']),
                    'calibration_adjustment': float(pred['empirical_accuracy'] - pred['model_probability'])
                },
                'odds': {
                    'decimal': float(pred['decimal_odds']),
                    'american': int(pred['american_odds']),
                    'implied_probability': float(pred['implied_probability'])
                },
                'value_analysis': {
                    'edge': float(pred['edge']),
                    'edge_classification': pred['edge_class'],
                    'expected_value': float(pred['expected_value']),
                    'expected_roi': float(pred['expected_roi']),
                    'sharpe_ratio': float(pred['sharpe_ratio'])
                },
                'recommendation': {
                    'action': pred['action'],
                    'bet_size': float(pred['bet_size']),
                    'bet_percentage': float(pred['bet_pct_bankroll']),
                    'kelly_fraction': float(pred['kelly_fraction']),
                    'reasoning': pred['reasoning']
                },
                'risk_metrics': {
                    'portfolio_correlation': float(pred['portfolio_correlation']),
                    'current_exposure': float(pred['current_portfolio_exposure']),
                    'accuracy_std_error': float(pred['accuracy_std_error'])
                }
            }
            
            games.append(game_data)
        
        return {
            'api_version': '1.0',
            'generated_at': datetime.now().isoformat(),
            'date': date,
            'total_games': len(games),
            'games': games
        }
    
    def _generate_betting_json(self, 
                               predictions_df: pd.DataFrame,
                               portfolio: Dict,
                               date: str) -> Dict:
        """Generate betting-focused view with recommendations"""
        bet_games = predictions_df[predictions_df['action'] == 'BET'].copy()
        bet_games = bet_games.sort_values('bet_size', ascending=False)
        
        recommendations = []
        for idx, bet in bet_games.iterrows():
            rec = {
                'priority': int(idx + 1),
                'game_id': str(bet['game_id']),
                'matchup': bet['matchup'],
                'game_time_utc': bet['game_time'],
                'pick': {
                    'team': bet['predicted_team'],
                    'side': bet['predicted_winner'],
                    'confidence': float(bet['model_probability']),
                    'calibrated_accuracy': float(bet['empirical_accuracy'])
                },
                'bet_details': {
                    'recommended_stake': float(bet['bet_size']),
                    'stake_percentage': float(bet['bet_pct_bankroll']),
                    'odds_american': int(bet['american_odds']),
                    'odds_decimal': float(bet['decimal_odds']),
                    'kelly_fraction': float(bet['kelly_fraction'])
                },
                'value_metrics': {
                    'edge': float(bet['edge']),
                    'edge_class': bet['edge_class'],
                    'expected_value': float(bet['expected_value']),
                    'expected_roi_pct': float(bet['expected_roi']),
                    'sharpe_ratio': float(bet['sharpe_ratio'])
                },
                'reasoning': bet['reasoning'],
                'strength': self._classify_bet_strength(bet)
            }
            recommendations.append(rec)
        
        return {
            'api_version': '1.0',
            'generated_at': datetime.now().isoformat(),
            'date': date,
            'portfolio_summary': {
                'total_positions': len(recommendations),
                'total_stake': float(bet_games['bet_size'].sum()) if len(bet_games) > 0 else 0,
                'total_expected_value': float(bet_games['expected_value'].sum()) if len(bet_games) > 0 else 0,
                'average_edge': float(bet_games['edge'].mean()) if len(bet_games) > 0 else 0,
                'portfolio_roi': float((bet_games['expected_value'].sum() / bet_games['bet_size'].sum() * 100)) 
                                if len(bet_games) > 0 and bet_games['bet_size'].sum() > 0 else 0,
                'exposure_percentage': float(portfolio.get('exposure_pct', 0)),
                'portfolio_sharpe': float(portfolio.get('portfolio_sharpe', 0))
            },
            'recommendations': recommendations
        }
    
    def _generate_analytics_json(self, predictions_df: pd.DataFrame, date: str) -> Dict:
        """Generate analytics and statistical insights"""
        return {
            'api_version': '1.0',
            'generated_at': datetime.now().isoformat(),
            'date': date,
            'probability_analysis': {
                'average_model_probability': float(predictions_df['model_probability'].mean()),
                'average_calibrated_accuracy': float(predictions_df['empirical_accuracy'].mean()),
                'average_calibration_adjustment': float(
                    (predictions_df['empirical_accuracy'] - predictions_df['model_probability']).mean()
                ),
                'probability_distribution': self._calculate_probability_distribution(predictions_df)
            },
            'edge_analysis': {
                'total_positive_edge_games': len(predictions_df[predictions_df['edge'] > 0]),
                'average_edge': float(predictions_df['edge'].mean()),
                'max_edge': float(predictions_df['edge'].max()),
                'edge_by_classification': self._calculate_edge_by_class(predictions_df)
            },
            'value_metrics': {
                'total_expected_value': float(predictions_df['expected_value'].sum()),
                'average_sharpe_ratio': float(predictions_df['sharpe_ratio'].mean()),
                'value_concentration': self._calculate_value_concentration(predictions_df)
            },
            'odds_analysis': {
                'average_odds_decimal': float(predictions_df['decimal_odds'].mean()),
                'average_implied_probability': float(predictions_df['implied_probability'].mean()),
                'odds_distribution': self._calculate_odds_distribution(predictions_df)
            },
            'team_analysis': {
                'home_favorites': len(predictions_df[predictions_df['home_win_prob'] > 0.5]),
                'away_favorites': len(predictions_df[predictions_df['away_win_prob'] > 0.5]),
                'picks_by_side': {
                    'home': len(predictions_df[predictions_df['predicted_winner'] == 'HOME']),
                    'away': len(predictions_df[predictions_df['predicted_winner'] == 'AWAY'])
                }
            }
        }
    
    def _generate_portfolio_json(self, 
                                 predictions_df: pd.DataFrame,
                                 portfolio: Dict,
                                 date: str) -> Dict:
        """Generate portfolio-level risk and exposure analysis"""
        bet_games = predictions_df[predictions_df['action'] == 'BET']
        
        return {
            'api_version': '1.0',
            'generated_at': datetime.now().isoformat(),
            'date': date,
            'portfolio_metrics': {
                'total_positions': len(bet_games),
                'total_exposure': float(portfolio.get('total_exposure', 0)),
                'exposure_percentage': float(portfolio.get('exposure_pct', 0)),
                'available_capital': float(portfolio.get('available_capital', 0)),
                'position_count': int(portfolio.get('position_count', 0))
            },
            'expected_performance': {
                'portfolio_expected_value': float(portfolio.get('portfolio_expected_value', 0)),
                'portfolio_sharpe_ratio': float(portfolio.get('portfolio_sharpe', 0)),
                'average_edge': float(portfolio.get('avg_edge', 0)),
                'weighted_average_probability': float(bet_games['model_probability'].mean()) if len(bet_games) > 0 else 0
            },
            'risk_metrics': {
                'average_correlation': float(bet_games['portfolio_correlation'].mean()) if len(bet_games) > 0 else 0,
                'concentration_risk': self._calculate_concentration_risk(bet_games),
                'kelly_criterion_compliance': self._calculate_kelly_compliance(bet_games)
            },
            'position_breakdown': self._generate_position_breakdown(bet_games),
            'diversification': {
                'unique_teams': len(set(bet_games['predicted_team'].tolist())) if len(bet_games) > 0 else 0,
                'home_away_split': {
                    'home_bets': len(bet_games[bet_games['predicted_winner'] == 'HOME']),
                    'away_bets': len(bet_games[bet_games['predicted_winner'] == 'AWAY'])
                }
            }
        }
    
    # Helper methods for detailed analytics
    
    def _get_top_picks(self, bet_games: pd.DataFrame, n: int = 3) -> List[Dict]:
        """Get top N picks by expected value"""
        if len(bet_games) == 0:
            return []
        
        top = bet_games.nlargest(n, 'expected_value')
        picks = []
        
        for idx, pick in top.iterrows():
            picks.append({
                'rank': len(picks) + 1,
                'team': pick['predicted_team'],
                'matchup': pick['matchup'],
                'edge': float(pick['edge']),
                'expected_value': float(pick['expected_value']),
                'bet_size': float(pick['bet_size'])
            })
        
        return picks
    
    def _calculate_edge_distribution(self, bet_games: pd.DataFrame) -> Dict:
        """Calculate distribution of edges across classifications"""
        if len(bet_games) == 0:
            return {}
        
        dist = bet_games.groupby('edge_class').agg({
            'bet_size': ['count', 'sum'],
            'edge': 'mean',
            'expected_value': 'sum'
        }).round(4)
        
        result = {}
        for edge_class in dist.index:
            result[edge_class] = {
                'count': int(dist.loc[edge_class, ('bet_size', 'count')]),
                'total_stake': float(dist.loc[edge_class, ('bet_size', 'sum')]),
                'average_edge': float(dist.loc[edge_class, ('edge', 'mean')]),
                'total_ev': float(dist.loc[edge_class, ('expected_value', 'sum')])
            }
        
        return result
    
    def _calculate_confidence_levels(self, predictions_df: pd.DataFrame) -> Dict:
        """Categorize predictions by confidence level"""
        def classify_confidence(prob):
            if prob >= 0.65:
                return 'high'
            elif prob >= 0.55:
                return 'medium'
            else:
                return 'low'
        
        predictions_df['confidence_level'] = predictions_df['model_probability'].apply(classify_confidence)
        
        counts = predictions_df['confidence_level'].value_counts().to_dict()
        
        return {
            'high_confidence': counts.get('high', 0),
            'medium_confidence': counts.get('medium', 0),
            'low_confidence': counts.get('low', 0)
        }
    
    def _calculate_probability_distribution(self, predictions_df: pd.DataFrame) -> Dict:
        """Calculate probability distribution statistics"""
        return {
            'min': float(predictions_df['model_probability'].min()),
            'max': float(predictions_df['model_probability'].max()),
            'mean': float(predictions_df['model_probability'].mean()),
            'median': float(predictions_df['model_probability'].median()),
            'std': float(predictions_df['model_probability'].std()),
            'quartiles': {
                '25th': float(predictions_df['model_probability'].quantile(0.25)),
                '50th': float(predictions_df['model_probability'].quantile(0.50)),
                '75th': float(predictions_df['model_probability'].quantile(0.75))
            }
        }
    
    def _calculate_edge_by_class(self, predictions_df: pd.DataFrame) -> Dict:
        """Calculate average edge by classification"""
        if 'edge_class' not in predictions_df.columns or len(predictions_df) == 0:
            return {}
        
        edge_stats = predictions_df.groupby('edge_class')['edge'].agg(['count', 'mean', 'std']).round(4)
        
        result = {}
        for edge_class in edge_stats.index:
            result[edge_class] = {
                'count': int(edge_stats.loc[edge_class, 'count']),
                'average_edge': float(edge_stats.loc[edge_class, 'mean']),
                'std_dev': float(edge_stats.loc[edge_class, 'std']) if not pd.isna(edge_stats.loc[edge_class, 'std']) else 0
            }
        
        return result
    
    def _calculate_value_concentration(self, predictions_df: pd.DataFrame) -> Dict:
        """Calculate how concentrated value is in top picks"""
        bet_games = predictions_df[predictions_df['action'] == 'BET'].copy()
        
        if len(bet_games) == 0:
            return {'top_3_ev_percentage': 0, 'top_5_ev_percentage': 0}
        
        bet_games = bet_games.sort_values('expected_value', ascending=False)
        total_ev = bet_games['expected_value'].sum()
        
        if total_ev <= 0:
            return {'top_3_ev_percentage': 0, 'top_5_ev_percentage': 0}
        
        top_3_ev = bet_games.head(3)['expected_value'].sum()
        top_5_ev = bet_games.head(5)['expected_value'].sum()
        
        return {
            'top_3_ev_percentage': float((top_3_ev / total_ev) * 100),
            'top_5_ev_percentage': float((top_5_ev / total_ev) * 100)
        }
    
    def _calculate_odds_distribution(self, predictions_df: pd.DataFrame) -> Dict:
        """Categorize games by odds ranges"""
        def classify_odds(american_odds):
            if american_odds >= 150:
                return 'heavy_underdog'
            elif american_odds >= 100:
                return 'moderate_underdog'
            elif american_odds >= -110:
                return 'slight_underdog'
            elif american_odds >= -150:
                return 'slight_favorite'
            elif american_odds >= -200:
                return 'moderate_favorite'
            else:
                return 'heavy_favorite'
        
        predictions_df['odds_category'] = predictions_df['american_odds'].apply(classify_odds)
        
        counts = predictions_df['odds_category'].value_counts().to_dict()
        
        return {k: int(v) for k, v in counts.items()}
    
    def _calculate_concentration_risk(self, bet_games: pd.DataFrame) -> Dict:
        """Calculate position concentration metrics"""
        if len(bet_games) == 0:
            return {'largest_position_pct': 0, 'top_3_positions_pct': 0}
        
        total_stake = bet_games['bet_size'].sum()
        
        if total_stake == 0:
            return {'largest_position_pct': 0, 'top_3_positions_pct': 0}
        
        sorted_bets = bet_games.sort_values('bet_size', ascending=False)
        
        largest_pos = sorted_bets.iloc[0]['bet_size'] / total_stake * 100
        top_3_pos = sorted_bets.head(3)['bet_size'].sum() / total_stake * 100
        
        return {
            'largest_position_pct': float(largest_pos),
            'top_3_positions_pct': float(top_3_pos)
        }
    
    def _calculate_kelly_compliance(self, bet_games: pd.DataFrame) -> Dict:
        """Calculate Kelly criterion compliance metrics"""
        if len(bet_games) == 0:
            return {'average_kelly_fraction': 0, 'kelly_range': {}}
        
        avg_kelly = bet_games['kelly_fraction'].mean()
        
        # Categorize Kelly fractions
        kelly_ranges = {
            'conservative': len(bet_games[bet_games['kelly_fraction'] < 0.02]),
            'moderate': len(bet_games[(bet_games['kelly_fraction'] >= 0.02) & (bet_games['kelly_fraction'] < 0.05)]),
            'aggressive': len(bet_games[bet_games['kelly_fraction'] >= 0.05])
        }
        
        return {
            'average_kelly_fraction': float(avg_kelly),
            'kelly_range': kelly_ranges
        }
    
    def _generate_position_breakdown(self, bet_games: pd.DataFrame) -> List[Dict]:
        """Generate detailed breakdown of each position"""
        if len(bet_games) == 0:
            return []
        
        positions = []
        for idx, bet in bet_games.iterrows():
            positions.append({
                'game_id': str(bet['game_id']),
                'team': bet['predicted_team'],
                'stake': float(bet['bet_size']),
                'edge': float(bet['edge']),
                'expected_value': float(bet['expected_value']),
                'kelly_fraction': float(bet['kelly_fraction'])
            })
        
        return positions
    
    def _classify_bet_strength(self, bet: pd.Series) -> str:
        """Classify the strength of a betting recommendation"""
        edge = bet['edge']
        ev = bet['expected_value']
        sharpe = bet['sharpe_ratio']
        
        # Multi-factor strength classification
        if edge >= 0.08 and ev >= 20 and sharpe >= 0.5:
            return 'EXCEPTIONAL'
        elif edge >= 0.05 and ev >= 10 and sharpe >= 0.3:
            return 'STRONG'
        elif edge >= 0.03 and ev >= 5:
            return 'GOOD'
        elif edge >= 0.01:
            return 'MODERATE'
        else:
            return 'MARGINAL'


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return super(NumpyEncoder, self).default(obj)