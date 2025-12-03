"""
Backfill NBA Predictions - DEBUG VERSION
"""

import argparse
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from etl.database.connection import get_db_connection
from models.data.data_loader import NBADataLoader
from models.features.player_features import PlayerFeatureEngineer
from models.features.team_features import TeamFeatureEngineer
from models.trainers.player_props_trainer import PlayerPropsTrainer
from models.trainers.team_props_trainer import TeamPropsTrainer
from models.api.db_uploader import NBAPredictionUploader


class OptimizedPredictionRunner:
    
    PLAYER_PROPS = ['pts', 'reb', 'ast', 'pra', 'pr', 'pa', 'ra']
    
    def __init__(self):
        self.db = get_db_connection()
        self.player_fe = None
        self.team_fe = None
        self.player_trainer = None
        self.team_trainer = None
        self.uploader = None
        
        self.team_id_to_abbrev = {}
        self.abbrev_to_team_id = {}
        self.player_id_to_name = {}
        
        self.player_game_logs_df = None
        self.games_by_date = {}
        
        self.all_player_features = None
        self.feature_cols_by_prop = {}
        self.player_feature_index = {}
    
    def initialize(self):
        print("\n" + "=" * 60)
        print("INITIALIZING")
        print("=" * 60)
        
        print("\n📊 Loading data...")
        data_loader = NBADataLoader(min_date='2023-01-01')
        player_logs, team_logs, players_df, teams_df = data_loader.load_all_data()
        print(f"  Player logs: {len(player_logs):,}")
        print(f"  Team logs: {len(team_logs):,}")
        
        self.player_game_logs_df = player_logs.copy()
        self.player_id_to_name = dict(zip(players_df['player_id'], players_df['full_name']))
        
        self.team_id_to_abbrev = dict(zip(teams_df['team_id'], teams_df['abbreviation']))
        self.abbrev_to_team_id = {v: k for k, v in self.team_id_to_abbrev.items()}
        
        print("  Indexing games by date...")
        team_logs['game_date'] = pd.to_datetime(team_logs['game_date']).dt.date
        for game_date, group in team_logs.groupby('game_date'):
            self.games_by_date[game_date] = group
        print(f"  Indexed {len(self.games_by_date)} dates")
        
        self.player_game_logs_df['game_date'] = pd.to_datetime(
            self.player_game_logs_df['game_date']
        ).dt.date
        
        print("\n🔧 Building features...")
        self.player_fe = PlayerFeatureEngineer(player_logs, team_logs, players_df, teams_df)
        self.team_fe = TeamFeatureEngineer(team_logs, teams_df)
        
        print("  Storing full feature cache...")
        self.all_player_features = self.player_fe._feature_cache.copy()
        self.all_player_features['game_date'] = pd.to_datetime(
            self.all_player_features['game_date']
        ).dt.date
        print(f"  Feature rows: {len(self.all_player_features):,}")
        
        min_date = self.all_player_features['game_date'].min()
        max_date = self.all_player_features['game_date'].max()
        print(f"  📅 Feature date range: {min_date} to {max_date}")
        
        print("  Building player feature index...")
        for idx, row in self.all_player_features.iterrows():
            player_id = row['player_id']
            game_date = row['game_date']
            if player_id not in self.player_feature_index:
                self.player_feature_index[player_id] = []
            self.player_feature_index[player_id].append((game_date, idx))
        
        for player_id in self.player_feature_index:
            self.player_feature_index[player_id].sort(key=lambda x: x[0])
        print(f"  Indexed {len(self.player_feature_index)} players")
        
        print("\n🤖 Loading models...")
        self.player_trainer = PlayerPropsTrainer(self.player_fe)
        self.team_trainer = TeamPropsTrainer(self.team_fe)
        
        try:
            self.player_trainer.load_all_models()
            self.team_trainer.load_all_models()
            print(f"  Player models: {list(self.player_trainer.models.keys())}")
            print(f"  Team models: {list(self.team_trainer.models.keys())}")
        except Exception as e:
            print(f"  Training new models: {e}")
            self.player_trainer.train_all_models(verbose=False)
            self.team_trainer.train_all_models(verbose=False)
            self.player_trainer.save_all_models()
            self.team_trainer.save_all_models()
        
        for prop in self.PLAYER_PROPS:
            if prop in self.player_trainer.models:
                self.feature_cols_by_prop[prop] = self.player_trainer.models[prop].feature_cols
        
        self.uploader = NBAPredictionUploader()
        self.uploader.init_schema()
        
        data_loader.close()
        print(f"\n✓ Initialization complete")
    
    def get_games_for_date(self, target_date):
        if target_date not in self.games_by_date:
            return []
        
        df = self.games_by_date[target_date]
        games_dict = {}
        
        for _, row in df.iterrows():
            game_id = row['game_id']
            matchup = row['matchup'] or ''
            is_home = ' vs. ' in matchup
            team_abbrev = self.team_id_to_abbrev.get(row['team_id'], 'UNK')
            
            if game_id not in games_dict:
                games_dict[game_id] = {
                    'game_id': game_id,
                    'game_date': target_date,
                    'home_team_id': None, 'away_team_id': None,
                    'home_team_abbrev': None, 'away_team_abbrev': None,
                }
            
            if is_home:
                games_dict[game_id]['home_team_id'] = row['team_id']
                games_dict[game_id]['home_team_abbrev'] = team_abbrev
            else:
                games_dict[game_id]['away_team_id'] = row['team_id']
                games_dict[game_id]['away_team_abbrev'] = team_abbrev
        
        return [g for g in games_dict.values() if g['home_team_id'] and g['away_team_id']]
    
    def get_player_features_for_date(self, player_id, game_date):
        if player_id not in self.player_feature_index:
            return None
        
        games = self.player_feature_index[player_id]
        
        left, right = 0, len(games) - 1
        result_idx = None
        
        while left <= right:
            mid = (left + right) // 2
            if games[mid][0] < game_date:
                result_idx = games[mid][1]
                left = mid + 1
            else:
                right = mid - 1
        
        if result_idx is None:
            return None
        
        return self.all_player_features.loc[result_idx]
    
    def get_players_for_game(self, game_id, game, game_date, debug=False):
        mask = self.player_game_logs_df['game_id'] == game_id
        game_players = self.player_game_logs_df.loc[mask, ['player_id', 'matchup']].drop_duplicates()
        
        if debug:
            print(f"\n    DEBUG: game_id={game_id}, game_date={game_date}")
            print(f"    DEBUG: Found {len(game_players)} players in game logs")
        
        players = []
        not_in_index = 0
        no_games_before = 0
        
        for _, row in game_players.iterrows():
            player_id = row['player_id']
            
            if player_id not in self.player_feature_index:
                not_in_index += 1
                continue
            
            features = self.get_player_features_for_date(player_id, game_date)
            if features is None:
                no_games_before += 1
                continue
            
            matchup = row['matchup'] or ''
            is_home = ' vs. ' in matchup
            
            players.append({
                'player_id': player_id,
                'player_name': self.player_id_to_name.get(player_id, f"Player {player_id}"),
                'team_abbrev': game['home_team_abbrev'] if is_home else game['away_team_abbrev'],
                'opponent_abbrev': game['away_team_abbrev'] if is_home else game['home_team_abbrev'],
                'is_home': is_home,
                'features': features,
            })
        
        if debug:
            print(f"    DEBUG: {len(players)} with features, {not_in_index} not in index, {no_games_before} no history")
        
        return players
    
    def predict_player_batch(self, players, game, target_date, debug=False):
        predictions = []
        
        if debug and players:
            print(f"    DEBUG predict_player_batch: {len(players)} players")
        
        for player in players:
            player_id = player['player_id']
            features = player['features']
            
            for prop in self.PLAYER_PROPS:
                if prop not in self.player_trainer.models:
                    continue
                
                try:
                    model = self.player_trainer.models[prop]
                    feature_cols = self.feature_cols_by_prop.get(prop, model.feature_cols)
                    
                    if debug and player == players[0] and prop == 'pts':
                        print(f"    DEBUG: features type = {type(features)}")
                        print(f"    DEBUG: feature_cols count = {len(feature_cols)}")
                        missing = [c for c in feature_cols if c not in features.index]
                        if missing:
                            print(f"    DEBUG: MISSING {len(missing)} cols: {missing[:5]}...")
                    
                    feature_vals = pd.DataFrame([features])[feature_cols].fillna(0)
                    preds = model.model.predict(feature_vals)
                    pred_value = float(preds[0])
                    
                    predictions.append({
                        'game_id': game['game_id'],
                        'game_date': target_date,
                        'player_id': int(player_id),
                        'player_name': player['player_name'],
                        'team_abbrev': player['team_abbrev'],
                        'opponent_abbrev': player['opponent_abbrev'],
                        'is_home': player['is_home'],
                        'prop_type': prop,
                        'predicted_value': pred_value,
                        'confidence': float(model._base_confidence),
                        'lower_bound': pred_value - 1.5 * model.metrics.mae,
                        'upper_bound': pred_value + 1.5 * model.metrics.mae,
                        'line': None,
                        'edge': None,
                        'recommended_bet': None
                    })
                except Exception as e:
                    if debug:
                        print(f"    DEBUG ERROR: {prop} for {player['player_name']}: {e}")
                    continue
        
        if debug:
            print(f"    DEBUG: Generated {len(predictions)} total predictions")
        
        return predictions
    
    def predict_team_props(self, game, target_date):
        predictions = []
        
        features = self.team_fe.prepare_game_prediction(
            home_team_id=game['home_team_id'],
            away_team_id=game['away_team_id'],
            game_date=pd.Timestamp(target_date)
        )
        
        if features is None:
            return predictions
        
        for prop in ['spread', 'total_pts']:
            if prop not in self.team_trainer.models:
                continue
            
            try:
                model = self.team_trainer.models[prop]
                pred_results = model.predict_with_confidence(features)
                
                if pred_results:
                    pred = pred_results[0]
                    prop_name = 'total' if prop == 'total_pts' else prop
                    
                    predictions.append({
                        'game_id': game['game_id'],
                        'game_date': target_date,
                        'home_team_id': int(game['home_team_id']),
                        'home_team_abbrev': game['home_team_abbrev'],
                        'away_team_id': int(game['away_team_id']),
                        'away_team_abbrev': game['away_team_abbrev'],
                        'prop_type': prop_name,
                        'predicted_value': float(pred.pred_value),
                        'confidence': float(pred.confidence),
                        'lower_bound': float(pred.lower_bound),
                        'upper_bound': float(pred.upper_bound),
                        'line': None,
                        'edge': None,
                        'recommended_bet': None
                    })
            except:
                continue
        
        return predictions
    
    def process_date(self, target_date, debug=False):
        games = self.get_games_for_date(target_date)
        
        if not games:
            return 0, 0, 0
        
        all_player_preds = []
        all_team_preds = []
        
        for idx, game in enumerate(games):
            team_preds = self.predict_team_props(game, target_date)
            all_team_preds.extend(team_preds)
            
            do_debug = debug and (idx == 0)
            players = self.get_players_for_game(game['game_id'], game, target_date, debug=do_debug)
            player_preds = self.predict_player_batch(players, game, target_date, debug=do_debug)
            all_player_preds.extend(player_preds)
            
            print(f"    {game['away_team_abbrev']}@{game['home_team_abbrev']}: "
                  f"{len(player_preds)} player, {len(team_preds)} team")
        
        p_count = self.uploader.upload_player_predictions(all_player_preds, target_date)
        t_count = self.uploader.upload_team_predictions(all_team_preds, target_date)
        
        return p_count, t_count, 0
    
    def check_existing(self, target_date):
        existing = self.uploader.get_predictions_by_date(target_date)
        return bool(existing.get('player_predictions'))
    
    def close(self):
        if self.uploader:
            self.uploader.close()
        if self.db:
            self.db.close()


def backfill_predictions(start_date, end_date, skip_existing=True, debug=False):
    print("\n" + "=" * 60)
    print("🏀 NBA PREDICTION BACKFILL")
    print("=" * 60)
    print(f"Date Range: {start_date} to {end_date}")
    print(f"Debug: {debug}")
    
    total_days = (end_date - start_date).days
    
    runner = OptimizedPredictionRunner()
    runner.initialize()
    
    processed = 0
    skipped = 0
    errors = 0
    total_player = 0
    total_team = 0
    
    start_time = time.time()
    current_date = start_date
    
    while current_date < end_date:
        day_num = processed + skipped + errors + 1
        print(f"\n📅 {current_date} ({day_num}/{total_days})")
        
        try:
            if skip_existing and runner.check_existing(current_date):
                print("  ⏭️ Exists, skipping")
                skipped += 1
                current_date += timedelta(days=1)
                continue
            
            day_start = time.time()
            p, t, _ = runner.process_date(current_date, debug=debug)
            day_time = time.time() - day_start
            
            if p or t:
                total_player += p
                total_team += t
                processed += 1
                print(f"  ✓ {p} player, {t} team ({day_time:.1f}s)")
            else:
                print("  ⚠️ No predictions")
                skipped += 1
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            errors += 1
        
        current_date += timedelta(days=1)
    
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Processed: {processed} | Skipped: {skipped} | Errors: {errors}")
    print(f"Predictions: {total_player} player, {total_team} team")
    print(f"Time: {elapsed/60:.1f}min")
    
    runner.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=str)
    parser.add_argument('--end', type=str)
    parser.add_argument('--days', type=int)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    
    if args.days:
        end_date = date.today()
        start_date = end_date - timedelta(days=args.days)
    elif args.start:
        start_date = datetime.strptime(args.start, '%Y-%m-%d').date()
        end_date = datetime.strptime(args.end, '%Y-%m-%d').date() if args.end else date.today()
    else:
        print("Error: Specify --start or --days")
        sys.exit(1)
    
    backfill_predictions(start_date, end_date, skip_existing=not args.overwrite, debug=args.debug)


if __name__ == "__main__":
    main()