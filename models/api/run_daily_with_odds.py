"""
Daily NBA Prediction Runner with Odds Integration
Generates predictions, fetches odds, calculates edges, and uploads to database

Usage:
    python -m models.api.run_daily_with_odds
    python -m models.api.run_daily_with_odds --date 2025-12-03
    python -m models.api.run_daily_with_odds --fetch-odds  # Also fetch fresh odds
"""

import argparse
import os
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Optional

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.data.data_loader import NBADataLoader
from models.features.player_features import PlayerFeatureEngineer
from models.features.team_features import TeamFeatureEngineer
from models.trainers.player_props_trainer import PlayerPropsTrainer
from models.trainers.team_props_trainer import TeamPropsTrainer
from models.predictors.game_predictor import GamePredictor, GameInfo, PlayerInfo
from models.predictors.parlay_builder import ParlayBuilder
from models.api.db_uploader import NBAPredictionUploader
from models.api.odds_integration import OddsIntegration
from models.api.fetch_odds import NBAOddsFetcher


class DailyPredictionRunnerWithOdds:
    """Run predictions with odds integration"""
    
    def __init__(self, bookmaker: str = 'fanduel'):
        self.bookmaker = bookmaker
        
        # Prediction components
        self.data_loader = None
        self.player_fe = None
        self.team_fe = None
        self.player_trainer = None
        self.team_trainer = None
        self.predictor = None
        self.parlay_builder = None
        
        # Database uploaders
        self.pred_uploader = None
        
        # Odds integration
        self.odds = None
        self.odds_fetcher = None
        
        # Mappings
        self.team_id_to_abbrev = {}
        self.players_with_data = set()
    
    def initialize(self, api_key: str = None):
        """Initialize all components"""
        print("\n" + "=" * 60)
        print("INITIALIZING PREDICTION SYSTEM WITH ODDS")
        print("=" * 60)
        
        # Load data
        print("\n📊 Loading data...")
        self.data_loader = NBADataLoader(min_date='2023-01-01')
        player_logs, team_logs, players, teams_df = self.data_loader.load_all_data()
        
        self.players_with_data = set(player_logs['player_id'].unique())
        self._build_team_mappings(teams_df)
        
        # Feature engineers
        print("\n🔧 Initializing features...")
        self.player_fe = PlayerFeatureEngineer(player_logs, team_logs, players, teams_df)
        self.team_fe = TeamFeatureEngineer(team_logs, teams_df)
        
        # Load models
        print("\n🤖 Loading models...")
        self.player_trainer = PlayerPropsTrainer(self.player_fe)
        self.team_trainer = TeamPropsTrainer(self.team_fe)
        
        try:
            self.player_trainer.load_all_models()
            self.team_trainer.load_all_models()
            print("  ✓ Models loaded")
        except:
            print("  Training new models...")
            self.player_trainer.train_all_models(verbose=False)
            self.team_trainer.train_all_models(verbose=False)
            self.player_trainer.save_all_models()
            self.team_trainer.save_all_models()
        
        # Predictor
        self.predictor = GamePredictor(
            self.player_fe, self.team_fe,
            self.player_trainer, self.team_trainer
        )
        self.parlay_builder = ParlayBuilder()
        
        # Uploaders
        self.pred_uploader = NBAPredictionUploader()
        self.pred_uploader.init_schema()
        
        # Odds integration
        api_key = api_key or os.environ.get('ODDSPAPI_KEY')
        self.odds = OddsIntegration(api_key, self.bookmaker)
        
        if api_key:
            self.odds_fetcher = NBAOddsFetcher(api_key, self.bookmaker)
        
        print("\n✓ Initialization complete")
    
    def _build_team_mappings(self, teams_df):
        """Build team ID to abbreviation mappings"""
        try:
            from nba_api.stats.static import teams as nba_teams
            for team in nba_teams.get_teams():
                self.team_id_to_abbrev[team['id']] = team['abbreviation']
        except:
            self.team_id_to_abbrev = dict(zip(teams_df['team_id'], teams_df['abbreviation']))
    
    def fetch_odds_if_needed(self, target_date: date):
        """Fetch odds if not already in database"""
        if not self.odds_fetcher:
            print("  ⚠️ No API key, skipping odds fetch")
            return
        
        # Check if we have odds for this date
        existing = self.odds.db.get_player_props(target_date, bookmaker=self.bookmaker)
        
        if existing:
            print(f"  ✓ Already have {len(existing)} props for {target_date}")
            return
        
        print(f"\n📈 Fetching odds for {target_date}...")
        self.odds_fetcher.fetch_daily_odds(target_date)
    
    def run(self, target_date: date = None, fetch_odds: bool = False):
        """Run full prediction pipeline with odds"""
        target_date = target_date or date.today()
        
        print("\n" + "=" * 60)
        print(f"🏀 NBA PREDICTIONS WITH ODDS - {target_date}")
        print("=" * 60)
        
        # Optionally fetch fresh odds
        if fetch_odds:
            self.fetch_odds_if_needed(target_date)
        
        # Generate predictions (using existing run_daily_predictions flow)
        from models.api.run_daily_predictions import DailyPredictionRunner
        
        # Create runner and use its methods
        runner = DailyPredictionRunner()
        runner.data_loader = self.data_loader
        runner.player_fe = self.player_fe
        runner.team_fe = self.team_fe
        runner.player_trainer = self.player_trainer
        runner.team_trainer = self.team_trainer
        runner.predictor = self.predictor
        runner.parlay_builder = self.parlay_builder
        runner.uploader = self.pred_uploader
        runner.team_id_to_abbrev = self.team_id_to_abbrev
        runner.abbrev_to_team_id = {v: k for k, v in self.team_id_to_abbrev.items()}
        runner.players_with_data = self.players_with_data
        runner.roster_fetcher = None
        
        # Import roster fetcher
        from models.api.run_daily_predictions import RosterFetcher
        runner.roster_fetcher = RosterFetcher()
        
        predictions = runner.generate_predictions(target_date)
        
        if not predictions['player_predictions'] and not predictions['team_predictions']:
            print("\n⚠️ No predictions generated")
            return
        
        # Enrich with odds
        print("\n📊 Enriching predictions with betting lines...")
        
        predictions['player_predictions'] = self.odds.enrich_player_predictions(
            predictions['player_predictions'], target_date
        )
        predictions['team_predictions'] = self.odds.enrich_team_predictions(
            predictions['team_predictions'], target_date
        )
        
        # Count enriched
        with_lines = sum(1 for p in predictions['player_predictions'] if p.get('line'))
        print(f"  Player predictions with lines: {with_lines}/{len(predictions['player_predictions'])}")
        
        with_team_lines = sum(1 for p in predictions['team_predictions'] if p.get('line'))
        print(f"  Team predictions with lines: {with_team_lines}/{len(predictions['team_predictions'])}")
        
        # Upload enriched predictions
        print("\n💾 Uploading predictions...")
        
        player_count = self.pred_uploader.upload_player_predictions(
            predictions['player_predictions'], target_date
        )
        team_count = self.pred_uploader.upload_team_predictions(
            predictions['team_predictions'], target_date
        )
        parlay_count = self.pred_uploader.upload_parlays(
            predictions['parlays'], target_date
        )
        
        print(f"  ✓ Player predictions: {player_count}")
        print(f"  ✓ Team predictions: {team_count}")
        print(f"  ✓ Parlays: {parlay_count}")
        
        # Generate betting report
        report = self.odds.generate_betting_report(
            predictions['player_predictions'],
            predictions['team_predictions'],
            target_date,
            min_edge=2.0
        )
        print(report)
        
        # Save report
        report_dir = Path('predictions')
        report_dir.mkdir(exist_ok=True)
        report_path = report_dir / f"betting_report_{target_date}.txt"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\n📄 Report saved: {report_path}")
        
        print("\n✅ Complete!")
    
    def close(self):
        if self.odds:
            self.odds.close()
        if self.pred_uploader:
            self.pred_uploader.close()
        if self.data_loader:
            self.data_loader.close()
        if self.odds_fetcher:
            self.odds_fetcher.close()


def main():
    parser = argparse.ArgumentParser(description='Daily Predictions with Odds')
    parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD)')
    parser.add_argument('--bookmaker', type=str, default='fanduel')
    parser.add_argument('--fetch-odds', action='store_true', help='Fetch fresh odds')
    
    args = parser.parse_args()
    
    target_date = None
    if args.date:
        target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
    
    api_key = os.environ.get('ODDSPAPI_KEY')
    
    runner = DailyPredictionRunnerWithOdds(args.bookmaker)
    
    try:
        runner.initialize(api_key)
        runner.run(target_date, fetch_odds=args.fetch_odds)
    finally:
        runner.close()


if __name__ == '__main__':
    main()