"""
Backtesting framework for evaluating prediction models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict

from models.base import PropPrediction, Parlay, ParlayLeg
from models.config import PARLAY_CONFIG


@dataclass
class BetResult:
    """Result of a single bet"""
    prediction: PropPrediction
    actual_value: float
    line: float
    bet_type: str  # 'over' or 'under'
    won: bool
    profit: float  # At -110 odds: +0.91 for win, -1.0 for loss
    
    @property
    def edge_realized(self) -> float:
        """Actual edge vs line"""
        if self.bet_type == 'over':
            return self.actual_value - self.line
        return self.line - self.actual_value


@dataclass
class BacktestResults:
    """Complete backtest results"""
    total_bets: int
    wins: int
    losses: int
    win_rate: float
    total_profit: float
    roi: float  # Return on investment
    
    # By confidence tier
    results_by_confidence: Dict[str, Dict] = field(default_factory=dict)
    
    # By prop type
    results_by_prop: Dict[str, Dict] = field(default_factory=dict)
    
    # By edge size
    results_by_edge: Dict[str, Dict] = field(default_factory=dict)
    
    # Time series
    daily_results: pd.DataFrame = None
    
    def __str__(self):
        return (
            f"Backtest Results\n"
            f"================\n"
            f"Total Bets: {self.total_bets}\n"
            f"Record: {self.wins}-{self.losses} ({self.win_rate:.1%})\n"
            f"Total Profit: {self.total_profit:+.2f} units\n"
            f"ROI: {self.roi:+.1%}\n"
        )


class PropBacktester:
    """Backtest player and team prop predictions"""
    
    def __init__(
        self,
        player_logs: pd.DataFrame,
        team_logs: pd.DataFrame
    ):
        self.player_logs = player_logs.copy()
        self.team_logs = team_logs.copy()
        
        # Ensure date columns are datetime
        self.player_logs['game_date'] = pd.to_datetime(self.player_logs['game_date'])
        self.team_logs['game_date'] = pd.to_datetime(self.team_logs['game_date'])
        
        # Build lookup indices
        self._build_indices()
    
    def _build_indices(self):
        """Build lookup indices for fast access"""
        # Player game lookup
        self.player_game_lookup = {}
        for _, row in self.player_logs.iterrows():
            key = (row['player_id'], row['game_id'])
            self.player_game_lookup[key] = row
        
        # Team game lookup
        self.team_game_lookup = {}
        for _, row in self.team_logs.iterrows():
            key = (row['team_id'], row['game_id'])
            self.team_game_lookup[key] = row
    
    def get_actual_value(
        self,
        prediction: PropPrediction
    ) -> Optional[float]:
        """Get actual value for a prediction"""
        
        if prediction.player_id:
            # Player prop
            key = (prediction.player_id, prediction.game_id)
            if key not in self.player_game_lookup:
                return None
            
            game = self.player_game_lookup[key]
            prop = prediction.prop_type
            
            # Handle combo stats
            if prop == 'pra':
                return game['pts'] + game['reb'] + game['ast']
            elif prop == 'pr':
                return game['pts'] + game['reb']
            elif prop == 'pa':
                return game['pts'] + game['ast']
            elif prop == 'ra':
                return game['reb'] + game['ast']
            elif prop == 'stocks':
                return game['stl'] + game['blk']
            elif prop in game:
                return game[prop]
            
        else:
            # Team prop - need both teams
            if prediction.prop_type == 'spread':
                # Find both team results for this game
                home_key = (prediction.team_id, prediction.game_id)
                if home_key not in self.team_game_lookup:
                    return None
                
                home_game = self.team_game_lookup[home_key]
                
                # Find away team
                for key, game in self.team_game_lookup.items():
                    if key[1] == prediction.game_id and key[0] != prediction.team_id:
                        away_game = game
                        break
                else:
                    return None
                
                return home_game['pts'] - away_game['pts']
            
            elif prediction.prop_type in ['total', 'total_pts']:
                # Sum both teams
                total = 0
                games_found = 0
                for key, game in self.team_game_lookup.items():
                    if key[1] == prediction.game_id:
                        total += game['pts']
                        games_found += 1
                
                if games_found == 2:
                    return total
        
        return None
    
    def evaluate_bet(
        self,
        prediction: PropPrediction,
        odds: float = -110
    ) -> Optional[BetResult]:
        """Evaluate a single bet"""
        
        if prediction.line is None:
            return None
        
        actual = self.get_actual_value(prediction)
        if actual is None:
            return None
        
        # Determine bet type
        bet_type = prediction.recommended_bet or ('over' if prediction.edge > 0 else 'under')
        
        # Determine if won
        if bet_type == 'over':
            won = actual > prediction.line
        elif bet_type == 'under':
            won = actual < prediction.line
        elif bet_type == 'home':
            won = actual > prediction.line  # spread
        elif bet_type == 'away':
            won = actual < prediction.line
        else:
            won = actual > prediction.line
        
        # Calculate profit
        if won:
            if odds < 0:
                profit = 100 / abs(odds)
            else:
                profit = odds / 100
        else:
            profit = -1.0
        
        return BetResult(
            prediction=prediction,
            actual_value=actual,
            line=prediction.line,
            bet_type=bet_type,
            won=won,
            profit=profit
        )
    
    def run_backtest(
        self,
        predictions: List[PropPrediction],
        min_confidence: float = 0.5,
        min_edge: float = 0.0
    ) -> BacktestResults:
        """
        Run backtest on a list of predictions
        
        Args:
            predictions: List of predictions to evaluate
            min_confidence: Minimum confidence to include
            min_edge: Minimum edge to include
            
        Returns:
            BacktestResults object
        """
        results = []
        
        for pred in predictions:
            # Filter by confidence
            if pred.prediction.confidence < min_confidence:
                continue
            
            # Filter by edge
            if pred.edge is not None and abs(pred.edge) < min_edge:
                continue
            
            result = self.evaluate_bet(pred)
            if result:
                results.append(result)
        
        if not results:
            return BacktestResults(
                total_bets=0, wins=0, losses=0,
                win_rate=0, total_profit=0, roi=0
            )
        
        # Aggregate results
        total = len(results)
        wins = sum(1 for r in results if r.won)
        losses = total - wins
        win_rate = wins / total
        total_profit = sum(r.profit for r in results)
        roi = total_profit / total
        
        # Results by confidence tier
        conf_tiers = {
            'low (50-60%)': (0.5, 0.6),
            'medium (60-70%)': (0.6, 0.7),
            'high (70%+)': (0.7, 1.0)
        }
        
        results_by_conf = {}
        for tier_name, (low, high) in conf_tiers.items():
            tier_results = [r for r in results 
                          if low <= r.prediction.prediction.confidence < high]
            if tier_results:
                tier_wins = sum(1 for r in tier_results if r.won)
                tier_profit = sum(r.profit for r in tier_results)
                results_by_conf[tier_name] = {
                    'count': len(tier_results),
                    'wins': tier_wins,
                    'win_rate': tier_wins / len(tier_results),
                    'profit': tier_profit,
                    'roi': tier_profit / len(tier_results)
                }
        
        # Results by prop type
        results_by_prop = defaultdict(list)
        for r in results:
            results_by_prop[r.prediction.prop_type].append(r)
        
        prop_summary = {}
        for prop, prop_results in results_by_prop.items():
            prop_wins = sum(1 for r in prop_results if r.won)
            prop_profit = sum(r.profit for r in prop_results)
            prop_summary[prop] = {
                'count': len(prop_results),
                'wins': prop_wins,
                'win_rate': prop_wins / len(prop_results),
                'profit': prop_profit,
                'roi': prop_profit / len(prop_results)
            }
        
        # Results by edge size
        edge_tiers = {
            'small (0-2)': (0, 2),
            'medium (2-4)': (2, 4),
            'large (4+)': (4, 100)
        }
        
        results_by_edge = {}
        for tier_name, (low, high) in edge_tiers.items():
            tier_results = [r for r in results 
                          if r.prediction.edge and low <= abs(r.prediction.edge) < high]
            if tier_results:
                tier_wins = sum(1 for r in tier_results if r.won)
                tier_profit = sum(r.profit for r in tier_results)
                results_by_edge[tier_name] = {
                    'count': len(tier_results),
                    'wins': tier_wins,
                    'win_rate': tier_wins / len(tier_results),
                    'profit': tier_profit,
                    'roi': tier_profit / len(tier_results)
                }
        
        # Daily results for time series
        daily_data = defaultdict(lambda: {'bets': 0, 'wins': 0, 'profit': 0})
        for r in results:
            date = r.prediction.game_date.date()
            daily_data[date]['bets'] += 1
            daily_data[date]['wins'] += 1 if r.won else 0
            daily_data[date]['profit'] += r.profit
        
        daily_df = pd.DataFrame([
            {'date': d, **v} for d, v in sorted(daily_data.items())
        ])
        if not daily_df.empty:
            daily_df['cumulative_profit'] = daily_df['profit'].cumsum()
            daily_df['cumulative_bets'] = daily_df['bets'].cumsum()
            daily_df['cumulative_roi'] = daily_df['cumulative_profit'] / daily_df['cumulative_bets']
        
        return BacktestResults(
            total_bets=total,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            total_profit=total_profit,
            roi=roi,
            results_by_confidence=results_by_conf,
            results_by_prop=prop_summary,
            results_by_edge=results_by_edge,
            daily_results=daily_df
        )
    
    def print_backtest_report(self, results: BacktestResults):
        """Print formatted backtest report"""
        print("\n" + "="*60)
        print("BACKTEST REPORT")
        print("="*60)
        
        print(f"\nOverall Performance:")
        print(f"  Total Bets: {results.total_bets}")
        print(f"  Record: {results.wins}-{results.losses} ({results.win_rate:.1%})")
        print(f"  Total Profit: {results.total_profit:+.2f} units")
        print(f"  ROI: {results.roi:+.1%}")
        
        print(f"\nBy Confidence Level:")
        for tier, data in results.results_by_confidence.items():
            print(f"  {tier}: {data['count']} bets, "
                  f"{data['win_rate']:.1%} win rate, "
                  f"{data['roi']:+.1%} ROI")
        
        print(f"\nBy Prop Type:")
        for prop, data in sorted(results.results_by_prop.items()):
            print(f"  {prop}: {data['count']} bets, "
                  f"{data['win_rate']:.1%} win rate, "
                  f"{data['roi']:+.1%} ROI")
        
        print(f"\nBy Edge Size:")
        for tier, data in results.results_by_edge.items():
            print(f"  {tier}: {data['count']} bets, "
                  f"{data['win_rate']:.1%} win rate, "
                  f"{data['roi']:+.1%} ROI")
        
        # Breakeven analysis
        # At -110 odds, need ~52.4% to break even
        breakeven = 0.524
        print(f"\nBreakeven Analysis:")
        print(f"  Required win rate at -110: {breakeven:.1%}")
        print(f"  Actual win rate: {results.win_rate:.1%}")
        if results.win_rate > breakeven:
            print(f"  ✅ PROFITABLE - {(results.win_rate - breakeven)*100:.1f}% above breakeven")
        else:
            print(f"  ❌ NOT PROFITABLE - {(breakeven - results.win_rate)*100:.1f}% below breakeven")