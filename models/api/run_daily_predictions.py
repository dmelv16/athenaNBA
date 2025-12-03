"""
Daily NBA Prediction Runner
Loads models, generates predictions for today's games, uploads to database

Usage:
    python -m models.api.run_daily_predictions
    python -m models.api.run_daily_predictions --date 2025-12-03
"""

import argparse
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from nba_api.stats.endpoints import scoreboardv2
from nba_api.stats.static import teams as nba_teams
import time

from models.data.data_loader import NBADataLoader
from models.features.player_features import PlayerFeatureEngineer
from models.features.team_features import TeamFeatureEngineer
from models.trainers.player_props_trainer import PlayerPropsTrainer
from models.trainers.team_props_trainer import TeamPropsTrainer
from models.predictors.game_predictor import GamePredictor, GameInfo, PlayerInfo
from models.predictors.parlay_builder import ParlayBuilder
from models.api.db_uploader import NBAPredictionUploader
from models.config import PATH_CONFIG, PARLAY_CONFIG


class DailyPredictionRunner:
    """Orchestrates daily prediction generation"""
    
    def __init__(self):
        self.data_loader = None
        self.player_fe = None
        self.team_fe = None
        self.player_trainer = None
        self.team_trainer = None
        self.predictor = None
        self.parlay_builder = None
        self.uploader = None
        
        # Team mappings
        self.team_id_to_abbrev = {}
        self.abbrev_to_team_id = {}
    
    def initialize(self):
        """Load data and models"""
        print("\n" + "=" * 60)
        print("INITIALIZING NBA PREDICTION SYSTEM")
        print("=" * 60)
        
        # Load data
        print("\n📊 Loading data from database...")
        self.data_loader = NBADataLoader(min_date='2023-01-01')
        player_logs, team_logs, players, teams_df = self.data_loader.load_all_data()
        print(f"  Player logs: {len(player_logs):,}")
        print(f"  Team logs: {len(team_logs):,}")
        
        # Build team mappings
        self._build_team_mappings(teams_df)
        
        # Initialize feature engineers
        print("\n🔧 Initializing feature engineers...")
        self.player_fe = PlayerFeatureEngineer(player_logs, team_logs, players, teams_df)
        self.team_fe = TeamFeatureEngineer(team_logs, teams_df)
        
        # Load trained models
        print("\n🤖 Loading trained models...")
        self.player_trainer = PlayerPropsTrainer(self.player_fe)
        self.team_trainer = TeamPropsTrainer(self.team_fe)
        
        try:
            self.player_trainer.load_all_models()
            self.team_trainer.load_all_models()
            print("  ✓ Models loaded successfully")
        except Exception as e:
            print(f"  ✗ Error loading models: {e}")
            print("  → Training new models...")
            self.player_trainer.train_all_models(verbose=False)
            self.team_trainer.train_all_models(verbose=False)
            self.player_trainer.save_all_models()
            self.team_trainer.save_all_models()
        
        # Initialize predictor
        self.predictor = GamePredictor(
            self.player_fe, self.team_fe,
            self.player_trainer, self.team_trainer
        )
        
        # Initialize parlay builder
        self.parlay_builder = ParlayBuilder()
        
        # Initialize database uploader
        print("\n💾 Connecting to prediction database...")
        self.uploader = NBAPredictionUploader()
        self.uploader.init_schema()
        
        print("\n✓ Initialization complete!")
    
    def _build_team_mappings(self, teams_df: pd.DataFrame):
        """Build team ID to abbreviation mappings"""
        all_teams = nba_teams.get_teams()
        for team in all_teams:
            self.team_id_to_abbrev[team['id']] = team['abbreviation']
            self.abbrev_to_team_id[team['abbreviation']] = team['id']
    
    def fetch_todays_games(self, target_date: date = None) -> List[Dict]:
        """Fetch today's NBA games from API"""
        target_date = target_date or date.today()
        print(f"\n📅 Fetching games for {target_date}...")
        
        try:
            time.sleep(1)  # Rate limiting
            scoreboard = scoreboardv2.ScoreboardV2(
                game_date=target_date.strftime('%Y-%m-%d'),
                league_id='00'
            )
            
            games_df = scoreboard.get_data_frames()[0]  # GameHeader
            
            if games_df.empty:
                print("  No games found for this date")
                return []
            
            games = []
            for _, row in games_df.iterrows():
                game = {
                    'game_id': row['GAME_ID'],
                    'game_date': target_date,
                    'game_status': row.get('GAME_STATUS_TEXT', ''),
                    'home_team_id': row['HOME_TEAM_ID'],
                    'away_team_id': row['VISITOR_TEAM_ID'],
                    'home_team_abbrev': self.team_id_to_abbrev.get(row['HOME_TEAM_ID'], 'UNK'),
                    'away_team_abbrev': self.team_id_to_abbrev.get(row['VISITOR_TEAM_ID'], 'UNK'),
                }
                games.append(game)
            
            print(f"  Found {len(games)} games")
            for g in games:
                print(f"    • {g['away_team_abbrev']} @ {g['home_team_abbrev']}")
            
            return games
            
        except Exception as e:
            print(f"  ✗ Error fetching games: {e}")
            return []
    
    def get_players_for_game(self, game: Dict) -> List[PlayerInfo]:
        """Get active players for a game"""
        players = []
        
        # Get players from both teams
        for team_id, is_home in [(game['home_team_id'], True), (game['away_team_id'], False)]:
            team_abbrev = self.team_id_to_abbrev.get(team_id, 'UNK')
            opponent_abbrev = game['away_team_abbrev'] if is_home else game['home_team_abbrev']
            
            # Get active players for this team from our database
            query = """
                SELECT DISTINCT p.player_id, p.full_name
                FROM players p
                JOIN player_game_logs pgl ON p.player_id = pgl.player_id
                WHERE p.is_active = true
                  AND pgl.matchup LIKE %s
                  AND pgl.game_date >= %s
                ORDER BY p.full_name
            """
            
            # Look for recent games with this team abbreviation
            with self.data_loader.db.get_cursor() as cur:
                cur.execute(query, (f"%{team_abbrev}%", date.today() - timedelta(days=60)))
                team_players = cur.fetchall()
            
            for player_id, player_name in team_players:
                players.append(PlayerInfo(
                    player_id=player_id,
                    player_name=player_name,
                    team_id=team_id,
                    team_abbrev=team_abbrev,
                    opponent_abbrev=opponent_abbrev,
                    is_home=is_home,
                    rest_days=2,  # Default - could fetch actual rest days
                    lines=None  # No lines by default - could integrate odds API
                ))
        
        return players
    
    def generate_predictions(self, target_date: date = None) -> Dict:
        """Generate all predictions for target date"""
        target_date = target_date or date.today()
        
        print("\n" + "=" * 60)
        print(f"GENERATING PREDICTIONS FOR {target_date}")
        print("=" * 60)
        
        # Fetch games
        games = self.fetch_todays_games(target_date)
        
        if not games:
            return {'games': [], 'player_predictions': [], 'team_predictions': [], 'parlays': []}
        
        all_player_predictions = []
        all_team_predictions = []
        all_prop_predictions = []  # For parlay builder
        
        for game in games:
            print(f"\n🏀 {game['away_team_abbrev']} @ {game['home_team_abbrev']}")
            
            # Create GameInfo object
            game_info = GameInfo(
                game_id=game['game_id'],
                game_date=datetime.combine(game['game_date'], datetime.min.time()),
                home_team_id=game['home_team_id'],
                home_team_name=game['home_team_abbrev'],
                home_team_abbrev=game['home_team_abbrev'],
                away_team_id=game['away_team_id'],
                away_team_name=game['away_team_abbrev'],
                away_team_abbrev=game['away_team_abbrev']
            )
            
            # Team predictions
            print("  📈 Team predictions...")
            team_preds = self.predictor.predict_team_props(game_info)
            
            for pred in team_preds:
                pred_dict = {
                    'game_id': game['game_id'],
                    'game_date': target_date,
                    'home_team_id': game['home_team_id'],
                    'home_team_abbrev': game['home_team_abbrev'],
                    'away_team_id': game['away_team_id'],
                    'away_team_abbrev': game['away_team_abbrev'],
                    'prop_type': pred.prop_type,
                    'predicted_value': pred.prediction.pred_value,
                    'confidence': pred.prediction.confidence,
                    'lower_bound': pred.prediction.lower_bound,
                    'upper_bound': pred.prediction.upper_bound,
                    'line': pred.line,
                    'edge': pred.edge,
                    'recommended_bet': pred.recommended_bet
                }
                all_team_predictions.append(pred_dict)
                all_prop_predictions.append(pred)
                
                print(f"    {pred.prop_type}: {pred.prediction.pred_value:.1f} "
                      f"[{pred.prediction.confidence:.0%}]")
            
            # Player predictions
            print("  👤 Player predictions...")
            players = self.get_players_for_game(game)
            print(f"    Found {len(players)} players")
            
            player_count = 0
            for player in players:
                try:
                    player_preds = self.predictor.predict_player_props(player, game_info)
                    
                    for pred in player_preds:
                        # Skip low-accuracy props
                        if pred.prop_type in ['stl', 'blk']:
                            continue
                        
                        pred_dict = {
                            'game_id': game['game_id'],
                            'game_date': target_date,
                            'player_id': player.player_id,
                            'player_name': player.player_name,
                            'team_abbrev': player.team_abbrev,
                            'opponent_abbrev': player.opponent_abbrev,
                            'is_home': player.is_home,
                            'prop_type': pred.prop_type,
                            'predicted_value': pred.prediction.pred_value,
                            'confidence': pred.prediction.confidence,
                            'lower_bound': pred.prediction.lower_bound,
                            'upper_bound': pred.prediction.upper_bound,
                            'line': pred.line,
                            'edge': pred.edge,
                            'recommended_bet': pred.recommended_bet
                        }
                        all_player_predictions.append(pred_dict)
                        all_prop_predictions.append(pred)
                    
                    if player_preds:
                        player_count += 1
                        
                except Exception as e:
                    # Skip players we can't predict
                    continue
            
            print(f"    Generated predictions for {player_count} players")
        
        # Build parlays
        print("\n🎰 Building parlays...")
        daily_parlays = self.parlay_builder.build_daily_parlays(
            all_prop_predictions,
            num_standard=5,
            num_sgp_per_game=2,
            risk_level='moderate'
        )
        
        # Convert parlays to dict format
        parlay_dicts = []
        for candidate in daily_parlays.get('standard', []):
            parlay_dicts.append(candidate.parlay.to_dict())
            parlay_dicts[-1]['score'] = candidate.score
            parlay_dicts[-1]['parlay_type'] = 'standard'
        
        for game_id, sgps in daily_parlays.get('same_game', {}).items():
            for candidate in sgps:
                parlay_dict = candidate.parlay.to_dict()
                parlay_dict['score'] = candidate.score
                parlay_dict['parlay_type'] = 'same_game'
                parlay_dict['game_id'] = game_id
                parlay_dicts.append(parlay_dict)
        
        print(f"  Created {len(parlay_dicts)} parlays")
        
        return {
            'games': games,
            'player_predictions': all_player_predictions,
            'team_predictions': all_team_predictions,
            'parlays': parlay_dicts
        }
    
    def upload_predictions(self, predictions: Dict, target_date: date = None):
        """Upload predictions to database"""
        target_date = target_date or date.today()
        
        print("\n" + "=" * 60)
        print("UPLOADING PREDICTIONS TO DATABASE")
        print("=" * 60)
        
        # Upload player predictions
        player_count = self.uploader.upload_player_predictions(
            predictions['player_predictions'],
            prediction_date=target_date
        )
        
        # Upload team predictions
        team_count = self.uploader.upload_team_predictions(
            predictions['team_predictions'],
            prediction_date=target_date
        )
        
        # Upload parlays
        parlay_count = self.uploader.upload_parlays(
            predictions['parlays'],
            prediction_date=target_date
        )
        
        print(f"\n✓ Upload complete!")
        print(f"  Player predictions: {player_count}")
        print(f"  Team predictions: {team_count}")
        print(f"  Parlays: {parlay_count}")
    
    def run(self, target_date: date = None):
        """Run full prediction pipeline"""
        target_date = target_date or date.today()
        
        print("\n" + "=" * 60)
        print("🏀 NBA DAILY PREDICTION RUNNER")
        print("=" * 60)
        print(f"Date: {target_date}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Initialize
        self.initialize()
        
        # Generate predictions
        predictions = self.generate_predictions(target_date)
        
        if predictions['player_predictions'] or predictions['team_predictions']:
            # Upload to database
            self.upload_predictions(predictions, target_date)
            
            # Summary
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"Games: {len(predictions['games'])}")
            print(f"Player predictions: {len(predictions['player_predictions'])}")
            print(f"Team predictions: {len(predictions['team_predictions'])}")
            print(f"Parlays: {len(predictions['parlays'])}")
        else:
            print("\n⚠️  No predictions generated (no games or insufficient data)")
        
        print("\n✅ Done!")
        
        # Cleanup
        if self.uploader:
            self.uploader.close()
        if self.data_loader:
            self.data_loader.close()


def main():
    parser = argparse.ArgumentParser(description='NBA Daily Prediction Runner')
    parser.add_argument(
        '--date',
        type=str,
        help='Target date (YYYY-MM-DD). Defaults to today.'
    )
    
    args = parser.parse_args()
    
    target_date = None
    if args.date:
        target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
    
    runner = DailyPredictionRunner()
    runner.run(target_date)


if __name__ == "__main__":
    main()