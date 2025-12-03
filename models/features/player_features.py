"""
Feature engineering for player prop predictions
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from models.config import FEATURE_CONFIG


class PlayerFeatureEngineer:
    """Engineer features for player prop predictions"""
    
    def __init__(
        self, 
        player_game_logs: pd.DataFrame,
        team_game_logs: pd.DataFrame,
        players_df: Optional[pd.DataFrame] = None,
        teams_df: Optional[pd.DataFrame] = None
    ):
        """
        Initialize with game log data
        
        Args:
            player_game_logs: DataFrame from player_game_logs table
            team_game_logs: DataFrame from team_game_logs table
            players_df: Optional players reference table
            teams_df: Optional teams reference table
        """
        self.player_logs = self._prepare_player_logs(player_game_logs)
        self.team_logs = self._prepare_team_logs(team_game_logs)
        self.players = players_df
        self.teams = teams_df
        
        # Precompute team mappings and stats
        self._build_team_mappings()
        self._precompute_team_defensive_stats()
    
    def _prepare_player_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare player game logs"""
        df = df.copy()
        df['game_date'] = pd.to_datetime(df['game_date'])
        df = df.sort_values(['player_id', 'game_date']).reset_index(drop=True)
        
        # Add combined stats
        df['pra'] = df['pts'] + df['reb'] + df['ast']
        df['pr'] = df['pts'] + df['reb']
        df['pa'] = df['pts'] + df['ast']
        df['ra'] = df['reb'] + df['ast']
        df['stocks'] = df['stl'] + df['blk']  # Steals + blocks
        
        # Parse matchup for opponent and home/away
        df['is_home'] = df['matchup'].str.contains(' vs\\. ', na=False).astype(int)
        df['opponent_abbrev'] = df['matchup'].apply(self._extract_opponent)
        
        return df
    
    def _prepare_team_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare team game logs"""
        df = df.copy()
        df['game_date'] = pd.to_datetime(df['game_date'])
        df = df.sort_values(['team_id', 'game_date']).reset_index(drop=True)
        
        # Calculate pace proxy
        df['pace'] = df['fga'] + 0.44 * df['fta'] - df['oreb'] + df['tov']
        
        # Offensive/defensive indicators
        df['is_home'] = df['matchup'].str.contains(' vs\\. ', na=False).astype(int)
        df['opponent_abbrev'] = df['matchup'].apply(self._extract_opponent)
        
        return df
    
    def _extract_opponent(self, matchup: str) -> str:
        """Extract opponent abbreviation from matchup string"""
        if pd.isna(matchup):
            return None
        if ' vs. ' in matchup:
            return matchup.split(' vs. ')[1].strip()
        elif ' @ ' in matchup:
            return matchup.split(' @ ')[1].strip()
        return None
    
    def _build_team_mappings(self):
        """Build team abbreviation to ID mappings"""
        # Extract from team logs
        team_abbrevs = self.team_logs.groupby('team_id').apply(
            lambda x: x['matchup'].iloc[0].split()[0] if len(x) > 0 else None,
            include_groups=False
        )
        self.team_id_to_abbrev = team_abbrevs.to_dict()
        self.abbrev_to_team_id = {v: k for k, v in self.team_id_to_abbrev.items() if v}
    
    def _precompute_team_defensive_stats(self):
        """Precompute rolling team defensive statistics"""
        df = self.team_logs.copy()
        
        # Build all features at once using a dictionary
        new_cols = {}
        
        for window in FEATURE_CONFIG.rolling_windows:
            for stat in ['pts', 'fg_pct', 'fg3_pct', 'pace']:
                if stat in df.columns:
                    new_cols[f'team_{stat}_avg_{window}'] = (
                        df.groupby('team_id')[stat]
                        .transform(lambda x: x.shift(1).rolling(window, min_periods=3).mean())
                    )
        
        # Add all columns at once
        self.team_stats = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    
    def calculate_rolling_stats(
        self,
        df: pd.DataFrame,
        stat_cols: List[str],
        windows: List[int] = None,
        group_col: str = 'player_id'
    ) -> pd.DataFrame:
        """
        Calculate rolling statistics for given columns
        
        Args:
            df: DataFrame with stats
            stat_cols: Columns to calculate rolling stats for
            windows: Window sizes (defaults to config)
            group_col: Column to group by
        """
        if windows is None:
            windows = FEATURE_CONFIG.rolling_windows
        
        result = df.copy()
        
        # Build all new columns in a dictionary first
        new_cols = {}
        
        for col in stat_cols:
            if col not in result.columns:
                continue
            
            for window in windows:
                # Rolling mean (shifted to avoid leakage)
                new_cols[f'{col}_avg_{window}'] = (
                    result.groupby(group_col)[col]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=2).mean())
                )
                
                # Rolling std (variance indicator)
                new_cols[f'{col}_std_{window}'] = (
                    result.groupby(group_col)[col]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=3).std())
                )
                
                # Rolling max (ceiling indicator)
                new_cols[f'{col}_max_{window}'] = (
                    result.groupby(group_col)[col]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=2).max())
                )
                
                # Rolling min (floor indicator)
                new_cols[f'{col}_min_{window}'] = (
                    result.groupby(group_col)[col]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=2).min())
                )
            
            # Season-to-date average
            new_cols[f'{col}_season_avg'] = (
                result.groupby([group_col, 'season'])[col]
                .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
            )
        
        # Games played this season
        new_cols['games_this_season'] = result.groupby([group_col, 'season']).cumcount()
        
        # Concatenate all new columns at once
        result = pd.concat([result, pd.DataFrame(new_cols, index=result.index)], axis=1)
        
        return result
    
    def add_trend_features(self, df: pd.DataFrame, stat_cols: List[str]) -> pd.DataFrame:
        """Add trend indicators (recent vs longer term)"""
        result = df.copy()
        new_cols = {}
        
        for col in stat_cols:
            if col not in df.columns:
                continue
            
            # Short-term vs long-term trend
            avg_3_col = f'{col}_avg_3'
            avg_10_col = f'{col}_avg_10'
            if avg_3_col in result.columns and avg_10_col in result.columns:
                new_cols[f'{col}_trend'] = result[avg_3_col] - result[avg_10_col]
            
            # Momentum (difference from 2 games ago)
            new_cols[f'{col}_momentum'] = (
                result.groupby('player_id')[col]
                .transform(lambda x: x.shift(1) - x.shift(3))
            )
            
            # Consistency score (inverse of coefficient of variation)
            avg_10_col = f'{col}_avg_10'
            std_10_col = f'{col}_std_10'
            if avg_10_col in result.columns and std_10_col in result.columns:
                new_cols[f'{col}_consistency'] = (
                    result[avg_10_col] / (result[std_10_col] + 0.1)
                )
        
        # Add all at once
        result = pd.concat([result, pd.DataFrame(new_cols, index=result.index)], axis=1)
        
        return result
    
    def add_contextual_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add game context features"""
        result = df.copy()
        new_cols = {}
        
        # Rest days
        new_cols['rest_days'] = (
            result.groupby('player_id')['game_date']
            .transform(lambda x: x.diff().dt.days)
        ).fillna(3).clip(0, 10)
        
        # Back-to-back
        new_cols['is_b2b'] = (new_cols['rest_days'] <= 1).astype(int)
        
        # Day of week (some players perform differently)
        new_cols['day_of_week'] = result['game_date'].dt.dayofweek
        
        # Month (track seasonal patterns)
        new_cols['month'] = result['game_date'].dt.month
        
        # Recent win/loss impact
        new_cols['recent_wins'] = (
            result.groupby('player_id')['wl']
            .transform(lambda x: (x.shift(1) == 'W').rolling(5, min_periods=1).sum())
        )
        
        # Streak (consecutive W or L)
        def calc_streak(group):
            streak = []
            current = 0
            for wl in group:
                if pd.isna(wl):
                    streak.append(0)
                elif wl == 'W':
                    current = max(1, current + 1) if current >= 0 else 1
                    streak.append(current)
                else:
                    current = min(-1, current - 1) if current <= 0 else -1
                    streak.append(current)
            return pd.Series(streak, index=group.index)
        
        new_cols['team_streak'] = result.groupby('player_id')['wl'].transform(calc_streak).shift(1)
        
        # Add all at once
        result = pd.concat([result, pd.DataFrame(new_cols, index=result.index)], axis=1)
        
        return result
    
    def add_opponent_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add opponent-related features"""
        result = df.copy()
        
        # Get opponent team stats
        opp_cols = [c for c in self.team_stats.columns if 'team_' in c and '_avg_' in c]
        opponent_stats = self.team_stats[['team_id', 'game_date'] + opp_cols].copy()
        
        # Map opponent abbreviation to team_id
        result['opponent_team_id'] = result['opponent_abbrev'].map(self.abbrev_to_team_id)
        
        # For each game, get opponent's recent defensive stats
        # This is approximate - ideally we'd join on game_date
        opponent_recent = (
            opponent_stats.groupby('team_id')
            .last()
            .reset_index()
        )
        opponent_recent.columns = ['opponent_team_id'] + ['opp_' + c if c != 'opponent_team_id' else c for c in opponent_recent.columns[1:]]
        
        result = result.merge(opponent_recent, on='opponent_team_id', how='left')
        
        # Historical performance vs this opponent
        new_cols = {}
        for stat in ['pts', 'reb', 'ast', 'pra']:
            if stat in df.columns:
                new_cols[f'{stat}_vs_opp_avg'] = (
                    result.groupby(['player_id', 'opponent_team_id'])[stat]
                    .transform(lambda x: x.shift(1).expanding().mean())
                )
                new_cols[f'{stat}_vs_opp_games'] = (
                    result.groupby(['player_id', 'opponent_team_id']).cumcount()
                )
        
        if new_cols:
            result = pd.concat([result, pd.DataFrame(new_cols, index=result.index)], axis=1)
        
        return result
    
    def add_minutes_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add minutes-related features (critical for props)"""
        result = df.copy()
        
        if 'min' not in result.columns:
            return result
        
        new_cols = {}
        
        # Minutes trend
        new_cols['min_trend'] = (
            result.groupby('player_id')['min']
            .transform(lambda x: x.shift(1).rolling(3).mean() - x.shift(1).rolling(10).mean())
        )
        
        # Minutes stability
        new_cols['min_stability'] = (
            result.groupby('player_id')['min']
            .transform(lambda x: x.shift(1).rolling(5).std())
        )
        
        # Per-minute rates (efficiency)
        for stat in ['pts', 'reb', 'ast', 'stl', 'blk']:
            if stat in result.columns:
                per_min_col = f'{stat}_per_min'
                result[per_min_col] = result[stat] / result['min'].clip(lower=1)
                new_cols[f'{stat}_per_min_avg_5'] = (
                    result.groupby('player_id')[per_min_col]
                    .transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
                )
        
        result = pd.concat([result, pd.DataFrame(new_cols, index=result.index)], axis=1)
        
        return result
    
    def add_usage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add usage and opportunity features"""
        result = df.copy()
        new_cols = {}
        
        # Shot attempts as usage proxy
        result['fga_rate'] = result['fga'] / result['min'].clip(lower=1) * 36
        new_cols['fga_rate_avg_5'] = (
            result.groupby('player_id')['fga_rate']
            .transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
        )
        
        # True shooting attempts
        result['tsa'] = result['fga'] + 0.44 * result['fta']
        new_cols['tsa_avg_5'] = (
            result.groupby('player_id')['tsa']
            .transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
        )
        
        # Assist ratio
        if 'ast' in result.columns and 'tov' in result.columns:
            result['ast_to_tov'] = result['ast'] / result['tov'].clip(lower=0.5)
            new_cols['ast_to_tov_avg_5'] = (
                result.groupby('player_id')['ast_to_tov']
                .transform(lambda x: x.shift(1).rolling(5, min_periods=2).mean())
            )
        
        result = pd.concat([result, pd.DataFrame(new_cols, index=result.index)], axis=1)
        
        return result
    
    def build_feature_set(
        self,
        target_stat: str,
        include_all: bool = True
    ) -> pd.DataFrame:
        """
        Build complete feature set for a target stat
        
        Args:
            target_stat: The stat to predict ('pts', 'reb', 'ast', etc.)
            include_all: Whether to include all feature types
            
        Returns:
            DataFrame with all features and target
        """
        df = self.player_logs.copy()
        
        # Core stats to build features for
        stat_cols = FEATURE_CONFIG.player_stats + ['pra', 'pr', 'pa', 'ra', 'stocks']
        stat_cols = [s for s in stat_cols if s in df.columns]
        
        # Rolling statistics
        print("  Building rolling stats...")
        df = self.calculate_rolling_stats(df, stat_cols)
        
        # Trend features
        print("  Building trend features...")
        df = self.add_trend_features(df, stat_cols)
        
        # Context features
        print("  Building context features...")
        df = self.add_contextual_features(df)
        
        # Opponent features  
        print("  Building opponent features...")
        df = self.add_opponent_features(df)
        
        # Minutes features
        print("  Building minutes features...")
        df = self.add_minutes_features(df)
        
        # Usage features
        print("  Building usage features...")
        df = self.add_usage_features(df)
        
        # Drop rows without enough history
        min_games = FEATURE_CONFIG.min_games_required
        df = df[df['games_this_season'] >= min_games].copy()
        
        # Ensure target exists
        if target_stat not in df.columns:
            raise ValueError(f"Target stat '{target_stat}' not found in data")
        
        # Remove duplicate columns (keep first occurrence)
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Defragment the dataframe
        df = df.copy()
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame, target: str) -> List[str]:
        """Get list of feature columns (excluding target and metadata)"""
        exclude_cols = {
            target, 'player_id', 'game_id', 'game_date', 'season', 
            'matchup', 'wl', 'opponent_abbrev', 'opponent_team_id',
            'id', 'player_name', 'is_active', 'created_at'
        }
        
        # Also exclude raw stats (we want derived features)
        raw_stats = FEATURE_CONFIG.player_stats + ['pra', 'pr', 'pa', 'ra', 'stocks']
        exclude_cols.update(raw_stats)
        
        # Also exclude per-minute raw calculations
        exclude_cols.update([f'{s}_per_min' for s in raw_stats])
        exclude_cols.update(['fga_rate', 'tsa', 'ast_to_tov'])
        
        # Remove duplicate columns first
        df = df.loc[:, ~df.columns.duplicated()]
        
        feature_cols = []
        for c in df.columns:
            if c in exclude_cols:
                continue
            if c.startswith('opp_game_date'):
                continue
            
            # Check dtype safely
            col_dtype = df[c].dtype
            if col_dtype in ['float64', 'int64', 'float32', 'int32', 'int', 'float']:
                feature_cols.append(c)
        
        return feature_cols
    
    def prepare_prediction_features(
        self,
        player_id: int,
        opponent_abbrev: str,
        is_home: bool,
        rest_days: int = 2,
        target_stat: str = 'pts'
    ) -> pd.DataFrame:
        """
        Prepare features for a single upcoming game prediction
        
        Args:
            player_id: Player to predict for
            opponent_abbrev: Opponent team abbreviation
            is_home: Whether home game
            rest_days: Days since last game
            target_stat: Stat being predicted
            
        Returns:
            Single-row DataFrame with features
        """
        # Get player's recent games
        player_data = self.player_logs[self.player_logs['player_id'] == player_id].copy()
        
        if len(player_data) < FEATURE_CONFIG.min_games_required:
            return None
        
        # Build features on historical data
        df = self.build_feature_set(target_stat)
        player_features = df[df['player_id'] == player_id].copy()
        
        if player_features.empty:
            return None
        
        # Get most recent feature values
        latest = player_features.iloc[-1:].copy()
        
        # Override context features for upcoming game
        latest['is_home'] = int(is_home)
        latest['rest_days'] = rest_days
        latest['is_b2b'] = int(rest_days <= 1)
        latest['opponent_abbrev'] = opponent_abbrev
        
        # Update opponent features if we have them
        opp_team_id = self.abbrev_to_team_id.get(opponent_abbrev)
        if opp_team_id:
            latest['opponent_team_id'] = opp_team_id
        
        return latest