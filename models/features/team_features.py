"""
Feature engineering for team prop predictions (spread, totals)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from models.config import FEATURE_CONFIG


class TeamFeatureEngineer:
    """Engineer features for team-level predictions"""
    
    def __init__(
        self,
        team_game_logs: pd.DataFrame,
        teams_df: Optional[pd.DataFrame] = None
    ):
        """
        Initialize with team game log data
        
        Args:
            team_game_logs: DataFrame from team_game_logs table
            teams_df: Optional teams reference table
        """
        self.team_logs = self._prepare_team_logs(team_game_logs)
        self.teams = teams_df
        self._build_game_matchups()
    
    def _prepare_team_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare team game logs"""
        df = df.copy()
        df['game_date'] = pd.to_datetime(df['game_date'])
        df = df.sort_values(['team_id', 'game_date']).reset_index(drop=True)
        
        # Parse matchup
        df['is_home'] = df['matchup'].str.contains(' vs\\. ', na=False).astype(int)
        df['opponent_abbrev'] = df['matchup'].apply(self._extract_opponent)
        df['team_abbrev'] = df['matchup'].apply(self._extract_team)
        
        # Calculate advanced metrics
        df['pace'] = df['fga'] + 0.44 * df['fta'] - df['oreb'] + df['tov']
        df['efg_pct'] = (df['fgm'] + 0.5 * df['fg3m']) / df['fga'].clip(lower=1)
        df['ts_pct'] = df['pts'] / (2 * (df['fga'] + 0.44 * df['fta'])).clip(lower=1)
        df['tov_pct'] = df['tov'] / (df['fga'] + 0.44 * df['fta'] + df['tov']).clip(lower=1)
        df['oreb_pct'] = df['oreb'] / (df['oreb'] + df['dreb']).clip(lower=1)
        
        return df
    
    def _extract_opponent(self, matchup: str) -> str:
        """Extract opponent from matchup string"""
        if pd.isna(matchup):
            return None
        if ' vs. ' in matchup:
            return matchup.split(' vs. ')[1].strip()
        elif ' @ ' in matchup:
            return matchup.split(' @ ')[1].strip()
        return None
    
    def _extract_team(self, matchup: str) -> str:
        """Extract team abbreviation from matchup string"""
        if pd.isna(matchup):
            return None
        return matchup.split()[0].strip()
    
    def _build_game_matchups(self):
        """Build matched game records (home vs away in same row)"""
        # Build team mappings first
        team_info = self.team_logs[['team_id', 'team_abbrev']].drop_duplicates()
        self.team_id_to_abbrev = dict(zip(team_info['team_id'], team_info['team_abbrev']))
        self.abbrev_to_team_id = {v: k for k, v in self.team_id_to_abbrev.items() if v}
        
        # Group by game_id to pair teams - build games DataFrame directly
        games_list = []
        
        for game_id, group in self.team_logs.groupby('game_id'):
            if len(group) != 2:
                continue
            
            home = group[group['is_home'] == 1]
            away = group[group['is_home'] == 0]
            
            if home.empty or away.empty:
                continue
            
            home = home.iloc[0]
            away = away.iloc[0]
            
            games_list.append({
                'game_id': game_id,
                'game_date': home['game_date'],
                'season': home['season'],
                'home_team_id': home['team_id'],
                'away_team_id': away['team_id'],
                'home_team': home['team_abbrev'],
                'away_team': away['team_abbrev'],
                'home_pts': home['pts'],
                'away_pts': away['pts'],
                'total_pts': home['pts'] + away['pts'],
                'spread': home['pts'] - away['pts'],  # Positive = home won
                'home_win': int(home['wl'] == 'W'),
            })
        
        self.games = pd.DataFrame(games_list)
    
    def calculate_team_rolling_stats(
        self,
        windows: List[int] = None
    ) -> pd.DataFrame:
        """Calculate rolling team statistics"""
        if windows is None:
            windows = FEATURE_CONFIG.rolling_windows
        
        df = self.team_logs.copy()
        stat_cols = FEATURE_CONFIG.team_stats + ['pace', 'efg_pct', 'ts_pct', 'tov_pct']
        
        new_cols = {}
        
        for col in stat_cols:
            if col not in df.columns:
                continue
            
            for window in windows:
                # Rolling averages
                new_cols[f'{col}_avg_{window}'] = (
                    df.groupby('team_id')[col]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
                )
                
                # Rolling std
                new_cols[f'{col}_std_{window}'] = (
                    df.groupby('team_id')[col]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=3).std())
                )
            
            # Season average
            new_cols[f'{col}_season_avg'] = (
                df.groupby(['team_id', 'season'])[col]
                .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
            )
        
        # Add all columns at once
        result = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        
        return result
    
    def calculate_opponent_adjusted_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate opponent-adjusted statistics"""
        result = df.copy()
        
        # Get league averages
        league_avg = df.groupby('game_date')[['pts', 'pace']].mean()
        league_avg.columns = ['league_pts_avg', 'league_pace_avg']
        result = result.merge(league_avg, on='game_date', how='left')
        
        # Points vs league average
        result['pts_vs_league'] = result['pts'] - result['league_pts_avg']
        
        return result
    
    def add_team_contextual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add context features for team games"""
        result = df.copy()
        new_cols = {}
        
        # Rest days
        new_cols['rest_days'] = (
            result.groupby('team_id')['game_date']
            .transform(lambda x: x.diff().dt.days)
        ).fillna(3).clip(0, 10)
        
        new_cols['is_b2b'] = (new_cols['rest_days'] <= 1).astype(int)
        
        # Road trip length (consecutive away games)
        def calc_road_trip(group):
            trip = []
            current = 0
            for is_home in group:
                if is_home == 0:
                    current += 1
                else:
                    current = 0
                trip.append(current)
            return pd.Series(trip, index=group.index)
        
        new_cols['road_trip_game'] = (
            result.groupby('team_id')['is_home']
            .transform(calc_road_trip)
        )
        
        # Recent record
        new_cols['recent_wins'] = (
            result.groupby('team_id')['wl']
            .transform(lambda x: (x.shift(1) == 'W').rolling(10, min_periods=1).sum())
        )
        
        new_cols['win_pct_10'] = new_cols['recent_wins'] / 10
        
        # Scoring trend
        new_cols['pts_trend'] = (
            result.groupby('team_id')['pts']
            .transform(lambda x: x.shift(1).rolling(3).mean() - x.shift(1).rolling(10).mean())
        )
        
        # Add all at once
        result = pd.concat([result, pd.DataFrame(new_cols, index=result.index)], axis=1)
        
        return result
    
    def build_matchup_features(self) -> pd.DataFrame:
        """Build features for game matchups (for spread/total prediction)"""
        if self.games.empty:
            return pd.DataFrame()
        
        # Start with paired games
        games = self.games.copy()
        
        # Get team rolling stats
        team_stats = self.calculate_team_rolling_stats()
        team_stats = self.add_team_contextual_features(team_stats)
        
        # Get the feature columns we want to pull
        feature_cols = [c for c in team_stats.columns if '_avg_' in c or '_std_' in c or 
                       c in ['rest_days', 'is_b2b', 'win_pct_10', 'pts_trend', 'road_trip_game']]
        
        # Create a lookup: (team_id, game_date) -> features
        team_stats_indexed = team_stats.set_index(['team_id', 'game_date'])[feature_cols]
        
        # Build home and away features
        home_features = []
        away_features = []
        
        for _, game in games.iterrows():
            # Get home team features
            home_key = (game['home_team_id'], game['game_date'])
            if home_key in team_stats_indexed.index:
                h_feats = team_stats_indexed.loc[home_key].to_dict()
            else:
                h_feats = {c: np.nan for c in feature_cols}
            home_features.append(h_feats)
            
            # Get away team features
            away_key = (game['away_team_id'], game['game_date'])
            if away_key in team_stats_indexed.index:
                a_feats = team_stats_indexed.loc[away_key].to_dict()
            else:
                a_feats = {c: np.nan for c in feature_cols}
            away_features.append(a_feats)
        
        # Convert to DataFrames
        home_df = pd.DataFrame(home_features)
        away_df = pd.DataFrame(away_features)
        
        # Prefix columns
        home_df.columns = ['home_' + str(c) for c in home_df.columns]
        away_df.columns = ['away_' + str(c) for c in away_df.columns]
        
        # Combine
        result = pd.concat([games.reset_index(drop=True), 
                          home_df.reset_index(drop=True),
                          away_df.reset_index(drop=True)], axis=1)
        
        # Add differential features
        new_cols = {}
        for stat in ['pts', 'pace', 'efg_pct']:
            for window in [5, 10]:
                home_col = f'home_{stat}_avg_{window}'
                away_col = f'away_{stat}_avg_{window}'
                if home_col in result.columns and away_col in result.columns:
                    new_cols[f'{stat}_diff_{window}'] = result[home_col] - result[away_col]
        
        # Rest advantage
        if 'home_rest_days' in result.columns and 'away_rest_days' in result.columns:
            new_cols['rest_advantage'] = result['home_rest_days'] - result['away_rest_days']
        
        if new_cols:
            result = pd.concat([result, pd.DataFrame(new_cols, index=result.index)], axis=1)
        
        # Remove duplicate columns
        result = result.loc[:, ~result.columns.duplicated()]
        
        return result
    
    def build_spread_features(self) -> Tuple[pd.DataFrame, str]:
        """Build features for spread prediction"""
        df = self.build_matchup_features()
        target = 'spread'
        return df, target
    
    def build_total_features(self) -> Tuple[pd.DataFrame, str]:
        """Build features for total points prediction"""
        df = self.build_matchup_features()
        target = 'total_pts'
        return df, target
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get feature columns for model training"""
        exclude = {
            'game_id', 'game_date', 'season', 'home_team_id', 'away_team_id',
            'home_team', 'away_team', 'home_pts', 'away_pts', 'total_pts',
            'spread', 'home_win'
        }
        
        # Remove duplicate columns first
        df = df.loc[:, ~df.columns.duplicated()]
        
        feature_cols = []
        for c in df.columns:
            if c in exclude:
                continue
            col_dtype = df[c].dtype
            if col_dtype in ['float64', 'int64', 'float32', 'int32', 'int', 'float']:
                feature_cols.append(c)
        
        return feature_cols
    
    def prepare_game_prediction(
        self,
        home_team_id: int,
        away_team_id: int,
        game_date: pd.Timestamp = None
    ) -> Optional[pd.DataFrame]:
        """
        Prepare features for predicting an upcoming game
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID  
            game_date: Date of game (defaults to today)
            
        Returns:
            Single-row DataFrame with features
        """
        if game_date is None:
            game_date = pd.Timestamp.now()
        
        team_stats = self.calculate_team_rolling_stats()
        team_stats = self.add_team_contextual_features(team_stats)
        
        # Get feature columns
        feature_cols = [c for c in team_stats.columns if '_avg_' in c or '_std_' in c or 
                       c in ['rest_days', 'is_b2b', 'win_pct_10', 'pts_trend']]
        
        # Get home team's latest stats
        home_data = team_stats[
            (team_stats['team_id'] == home_team_id) & 
            (team_stats['game_date'] < game_date)
        ]
        if len(home_data) < 5:
            return None
        home_feats = home_data.iloc[-1][feature_cols].to_dict()
        
        # Get away team's latest stats
        away_data = team_stats[
            (team_stats['team_id'] == away_team_id) & 
            (team_stats['game_date'] < game_date)
        ]
        if len(away_data) < 5:
            return None
        away_feats = away_data.iloc[-1][feature_cols].to_dict()
        
        # Build feature row
        features = {}
        
        for col in feature_cols:
            features[f'home_{col}'] = home_feats.get(col)
            features[f'away_{col}'] = away_feats.get(col)
        
        # Differentials
        for stat in ['pts', 'pace']:
            for w in [5, 10]:
                h_col = f'{stat}_avg_{w}'
                if h_col in home_feats and h_col in away_feats:
                    features[f'{stat}_diff_{w}'] = home_feats[h_col] - away_feats[h_col]
        
        return pd.DataFrame([features])