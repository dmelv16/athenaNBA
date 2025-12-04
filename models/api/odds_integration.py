"""
Odds Integration for NBA Predictions
Integrates betting lines with prediction system for edge calculation

Usage:
    from models.api.odds_integration import OddsIntegration
    
    odds = OddsIntegration()
    predictions = odds.enrich_player_predictions(predictions, game_date)
"""

import sys
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional
import os

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.api.odds_uploader import OddsUploader
from models.api.odds_client import OddsPAPIClient


class OddsIntegration:
    """
    Integrates odds data with prediction system
    """
    
    def __init__(
        self,
        api_key: str = None,
        bookmaker: str = 'pinnacle',
        use_live: bool = False
    ):
        """
        Initialize odds integration
        
        Args:
            api_key: OddsPAPI key (or from ODDSPAPI_KEY env var)
            bookmaker: Preferred bookmaker
            use_live: Whether to fetch live odds (vs database only)
        """
        self.bookmaker = bookmaker
        self.use_live = use_live
        
        # Database for stored odds
        self.db = OddsUploader()
        
        # API client for live odds
        self.client = None
        if use_live:
            api_key = api_key or os.environ.get('ODDSPAPI_KEY')
            if api_key:
                self.client = OddsPAPIClient(api_key, bookmaker)
        
        # Caches
        self._line_cache: Dict[str, float] = {}
        self._game_lines_cache: Dict[str, Dict] = {}
    
    # ==========================================
    # Player Prop Lines
    # ==========================================
    
    def get_player_line(
        self,
        player_name: str,
        prop_type: str,
        game_date: date = None
    ) -> Optional[float]:
        """Get player prop line"""
        game_date = game_date or date.today()
        cache_key = f"{player_name}_{prop_type}_{game_date}"
        
        if cache_key in self._line_cache:
            return self._line_cache[cache_key]
        
        line = self.db.get_player_prop_line(
            player_name, prop_type, game_date, self.bookmaker
        )
        
        if line is not None:
            self._line_cache[cache_key] = line
        
        return line
    
    def get_player_all_lines(
        self,
        player_name: str,
        game_date: date = None
    ) -> Dict[str, float]:
        """Get all prop lines for a player"""
        game_date = game_date or date.today()
        
        props = self.db.get_player_props(
            game_date, player_name=player_name, bookmaker=self.bookmaker
        )
        
        return {p['prop_type']: float(p['line']) for p in props}
    
    # ==========================================
    # Game Lines
    # ==========================================
    
    def get_game_lines(
        self,
        home_team_abbrev: str,
        game_date: date = None
    ) -> Optional[Dict]:
        """Get game lines (spread, total, moneyline)"""
        game_date = game_date or date.today()
        cache_key = f"{home_team_abbrev}_{game_date}"
        
        if cache_key in self._game_lines_cache:
            return self._game_lines_cache[cache_key]
        
        odds = self.db.get_game_odds(game_date, self.bookmaker)
        
        for game in odds:
            if game.get('home_team_abbrev') == home_team_abbrev:
                result = {
                    'spread_line': float(game['spread_line']) if game.get('spread_line') else None,
                    'total_line': float(game['total_line']) if game.get('total_line') else None,
                    'home_ml': game.get('home_ml'),
                    'away_ml': game.get('away_ml'),
                }
                self._game_lines_cache[cache_key] = result
                return result
        
        return None
    
    def get_spread(self, home_team_abbrev: str, game_date: date = None) -> Optional[float]:
        """Get spread line for home team"""
        return self.db.get_game_spread(home_team_abbrev, game_date or date.today(), self.bookmaker)
    
    def get_total(self, home_team_abbrev: str, game_date: date = None) -> Optional[float]:
        """Get total (over/under) line"""
        return self.db.get_game_total(home_team_abbrev, game_date or date.today(), self.bookmaker)
    
    # ==========================================
    # Prediction Enrichment
    # ==========================================
    
    def enrich_player_predictions(
        self,
        predictions: List[Dict],
        game_date: date = None
    ) -> List[Dict]:
        """Add lines and edge to player predictions"""
        game_date = game_date or date.today()
        
        for pred in predictions:
            player = pred.get('player_name')
            prop_type = pred.get('prop_type')
            predicted = pred.get('predicted_value')
            
            if not all([player, prop_type, predicted]):
                continue
            
            line = self.get_player_line(player, prop_type, game_date)
            
            if line is not None:
                pred['line'] = line
                pred['edge'] = round(float(predicted) - line, 2)
                pred['recommended_bet'] = 'over' if pred['edge'] > 0 else 'under'
            else:
                pred['line'] = None
                pred['edge'] = None
                pred['recommended_bet'] = None
        
        return predictions
    
    def enrich_team_predictions(
        self,
        predictions: List[Dict],
        game_date: date = None
    ) -> List[Dict]:
        """Add lines and edge to team predictions"""
        game_date = game_date or date.today()
        
        for pred in predictions:
            home = pred.get('home_team_abbrev')
            prop_type = pred.get('prop_type')
            predicted = pred.get('predicted_value')
            
            if not all([home, prop_type, predicted is not None]):
                continue
            
            if prop_type == 'spread':
                line = self.get_spread(home, game_date)
                if line is not None:
                    pred['line'] = line
                    pred['edge'] = round(float(predicted) - line, 2)
                    pred['recommended_bet'] = 'home' if pred['edge'] > 0 else 'away'
            
            elif prop_type in ['total', 'total_pts']:
                line = self.get_total(home, game_date)
                if line is not None:
                    pred['line'] = line
                    pred['edge'] = round(float(predicted) - line, 2)
                    pred['recommended_bet'] = 'over' if pred['edge'] > 0 else 'under'
        
        return predictions
    
    # ==========================================
    # Edge Analysis
    # ==========================================
    
    def find_edges(
        self,
        predictions: List[Dict],
        min_edge: float = 2.0,
        min_confidence: float = 0.6
    ) -> List[Dict]:
        """Find predictions with significant edge"""
        edges = []
        
        for pred in predictions:
            edge = pred.get('edge')
            conf = pred.get('confidence', 0)
            
            if edge is None:
                continue
            
            if abs(edge) >= min_edge and conf >= min_confidence:
                edges.append(pred)
        
        edges.sort(key=lambda x: abs(x['edge']), reverse=True)
        return edges
    
    # ==========================================
    # Reporting
    # ==========================================
    
    def generate_betting_report(
        self,
        player_predictions: List[Dict],
        team_predictions: List[Dict],
        game_date: date = None,
        min_edge: float = 2.0
    ) -> str:
        """Generate formatted betting report"""
        game_date = game_date or date.today()
        
        player_predictions = self.enrich_player_predictions(player_predictions, game_date)
        team_predictions = self.enrich_team_predictions(team_predictions, game_date)
        
        player_edges = self.find_edges(player_predictions, min_edge)
        team_edges = self.find_edges(team_predictions, min_edge)
        
        lines = [
            "=" * 60,
            f"NBA BETTING REPORT - {game_date}",
            "=" * 60,
            f"\nBookmaker: {self.bookmaker}",
            f"Minimum Edge: {min_edge}",
        ]
        
        # Team bets
        lines.append(f"\n{'='*40}")
        lines.append("TEAM PROPS")
        lines.append("="*40)
        
        for pred in team_edges[:10]:
            lines.append(
                f"\n{pred.get('home_team_abbrev', '')} vs {pred.get('away_team_abbrev', '')}"
                f"\n  {pred['prop_type'].upper()}: Pred {pred['predicted_value']:.1f} | "
                f"Line {pred.get('line', 'N/A')} | Edge {pred.get('edge', 0):+.1f}"
                f"\n  Bet: {pred.get('recommended_bet', 'N/A').upper()} "
                f"[{pred.get('confidence', 0):.0%}]"
            )
        
        if not team_edges:
            lines.append("\n  No significant edges found")
        
        # Player props
        lines.append(f"\n{'='*40}")
        lines.append("PLAYER PROPS")
        lines.append("="*40)
        
        for pred in player_edges[:20]:
            lines.append(
                f"\n{pred['player_name']} - {pred['prop_type'].upper()}"
                f"\n  Pred: {pred['predicted_value']:.1f} | "
                f"Line: {pred.get('line', 'N/A')} | Edge: {pred.get('edge', 0):+.1f}"
                f"\n  Bet: {pred.get('recommended_bet', 'N/A').upper()} "
                f"[{pred.get('confidence', 0):.0%}]"
            )
        
        if not player_edges:
            lines.append("\n  No significant edges found")
        
        lines.append(f"\n{'='*60}")
        
        return '\n'.join(lines)
    
    def clear_cache(self):
        """Clear line caches"""
        self._line_cache.clear()
        self._game_lines_cache.clear()
    
    def close(self):
        """Clean up connections"""
        if self.client:
            self.client.close()
        if self.db:
            self.db.close()