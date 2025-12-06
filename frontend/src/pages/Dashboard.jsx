// src/pages/Dashboard.jsx
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import api from '../services/api';

const StatCard = ({ label, value, sub, color = 'white', icon, trend }) => (
  <div className="card p-4">
    <div className="flex items-start justify-between">
      <div>
        <p className="stat-label mb-1">{label}</p>
        <p className={`stat-lg ${color === 'green' ? 'text-emerald-400' : color === 'red' ? 'text-red-400' : color === 'amber' ? 'text-amber-400' : 'text-white'}`}>{value}</p>
        {sub && <p className="text-xs text-slate-500 mt-1">{sub}</p>}
      </div>
      <div className="flex flex-col items-end">
        {icon && <span className="text-2xl opacity-60">{icon}</span>}
        {trend !== undefined && (
          <span className={`text-xs mt-1 ${trend >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {trend >= 0 ? '↑' : '↓'} {Math.abs(trend).toFixed(1)}%
          </span>
        )}
      </div>
    </div>
  </div>
);

const NHLGameCard = ({ game }) => {
  const isBet = game.action === 'BET' || (game.bet_pct_bankroll && game.bet_pct_bankroll > 0);
  const betPct = game.bet_pct_bankroll || game.bet_size / 1000 || 0;
  const edge = game.edge || 0;
  const prob = game.model_probability || game.home_win_probability || 0.5;
  
  return (
    <div className={`card p-4 ${isBet ? 'border-emerald-500/30 bg-emerald-500/5' : ''}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          {isBet && <span className="badge-bet">BET</span>}
        </div>
        <span className="text-xs text-slate-500">
          {game.game_time ? new Date(game.game_time).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) : ''}
        </span>
      </div>
      
      <div className="flex items-center justify-between mb-4">
        <div className="text-center flex-1">
          <p className="font-bold text-lg">{game.away_team}</p>
          <p className="text-[10px] text-slate-500 uppercase">Away</p>
        </div>
        <span className="text-slate-600 text-lg px-3">@</span>
        <div className="text-center flex-1">
          <p className="font-bold text-lg">{game.home_team}</p>
          <p className="text-[10px] text-slate-500 uppercase">Home</p>
        </div>
      </div>
      
      <div className="grid grid-cols-3 gap-2 text-center text-sm">
        <div className="bg-slate-800/50 rounded-lg py-2">
          <p className="text-[10px] text-slate-500 mb-0.5">Pick</p>
          <p className="font-bold text-emerald-400">{game.predicted_winner === 'HOME' ? game.home_team : game.away_team}</p>
        </div>
        <div className="bg-slate-800/50 rounded-lg py-2">
          <p className="text-[10px] text-slate-500 mb-0.5">Edge</p>
          <p className={`font-bold ${edge > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {edge > 0 ? '+' : ''}{(edge * 100).toFixed(1)}%
          </p>
        </div>
        <div className="bg-slate-800/50 rounded-lg py-2">
          <p className="text-[10px] text-slate-500 mb-0.5">Prob</p>
          <p className="font-bold">{(prob * 100).toFixed(0)}%</p>
        </div>
      </div>
      
      {isBet && (
        <div className="mt-3 pt-3 border-t border-white/5 flex justify-between items-center">
          <span className="text-xs text-slate-400">Suggested Stake</span>
          <span className="font-bold text-emerald-400">{(betPct).toFixed(1)}% bankroll</span>
        </div>
      )}
    </div>
  );
};

const RecentBetCard = ({ bet }) => {
  const isWin = bet.bet_result === 'WIN';
  return (
    <div className={`flex items-center justify-between p-3 rounded-lg ${isWin ? 'bg-emerald-500/10' : 'bg-red-500/10'}`}>
      <div className="flex items-center gap-3">
        <span className={`text-lg ${isWin ? 'text-emerald-400' : 'text-red-400'}`}>
          {isWin ? '✓' : '✗'}
        </span>
        <div>
          <p className="text-sm font-medium">{bet.away_team} @ {bet.home_team}</p>
          <p className="text-xs text-slate-400">Bet: {bet.predicted_team}</p>
        </div>
      </div>
      <div className="text-right">
        <p className={`font-bold ${isWin ? 'text-emerald-400' : 'text-red-400'}`}>
          {bet.pnl >= 0 ? '+' : ''}${parseFloat(bet.pnl).toFixed(2)}
        </p>
        <p className="text-xs text-slate-500">{bet.game_date}</p>
      </div>
    </div>
  );
};

const Dashboard = () => {
  const [loading, setLoading] = useState(true);
  const [health, setHealth] = useState(null);
  const [nbaData, setNbaData] = useState(null);
  const [nhlData, setNhlData] = useState(null);
  const [trackingStats, setTrackingStats] = useState(null);
  const [recentBets, setRecentBets] = useState([]);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [h, nba, nhl, stats, bets] = await Promise.allSettled([
        api.getHealth(),
        api.getNBATodayPredictions(),
        api.getNHLTodayPredictions(),
        api.getTrackingStats(),
        api.getBetResults(7, 10),
      ]);
      if (h.status === 'fulfilled') setHealth(h.value);
      if (nba.status === 'fulfilled') setNbaData(nba.value);
      if (nhl.status === 'fulfilled') setNhlData(nhl.value);
      if (stats.status === 'fulfilled') setTrackingStats(stats.value);
      if (bets.status === 'fulfilled') setRecentBets(bets.value.bets || []);
    } catch (e) { console.error(e); }
    setLoading(false);
  };

  const nhlGames = nhlData?.games || [];
  const nhlBets = nhlGames.filter(g => g.action === 'BET' || (g.bet_pct_bankroll && g.bet_pct_bankroll > 0));

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="spinner w-10 h-10"></div>
      </div>
    );
  }

  const overall = trackingStats?.overall || {};
  const currentBankroll = trackingStats?.current_bankroll || 1000;
  const startingBankroll = trackingStats?.starting_bankroll || 1000;
  const totalReturnPct = trackingStats?.total_return_pct || 0;

  return (
    <div className="space-y-6 animate-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Dashboard</h1>
          <p className="text-slate-400 text-sm mt-0.5">{new Date().toLocaleDateString('en-US', {weekday: 'long', month: 'long', day: 'numeric', year: 'numeric'})}</p>
        </div>
        <button onClick={loadData} className="btn-secondary text-sm flex items-center gap-2">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Refresh
        </button>
      </div>

      {/* Bankroll Hero Card */}
      <div className="card p-6 border-emerald-500/20 bg-gradient-to-r from-emerald-500/10 to-transparent">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
          <div>
            <p className="text-slate-400 text-sm">Current Bankroll</p>
            <p className="text-4xl font-bold text-emerald-400">${currentBankroll.toFixed(2)}</p>
            <p className="text-sm text-slate-500 mt-1">Started: ${startingBankroll.toFixed(2)}</p>
          </div>
          <div className="flex gap-6">
            <div className="text-center">
              <p className={`text-2xl font-bold ${totalReturnPct >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {totalReturnPct >= 0 ? '+' : ''}{totalReturnPct.toFixed(1)}%
              </p>
              <p className="text-xs text-slate-500">Total Return</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold">{overall.total_bets || 0}</p>
              <p className="text-xs text-slate-500">Total Bets</p>
            </div>
            <div className="text-center">
              <p className={`text-2xl font-bold ${(overall.win_rate || 0) > 0.52 ? 'text-emerald-400' : 'text-amber-400'}`}>
                {((overall.win_rate || 0) * 100).toFixed(1)}%
              </p>
              <p className="text-xs text-slate-500">Win Rate</p>
            </div>
          </div>
        </div>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard 
          label="NHL Games Today" 
          value={nhlGames.length} 
          sub={`${nhlBets.length} recommended bets`} 
          icon="🏒" 
        />
        <StatCard 
          label="NBA Games Today" 
          value={nbaData?.total_games || 0} 
          sub={`${nbaData?.total_player_predictions || 0} player props`} 
          icon="🏀" 
        />
        <StatCard 
          label="Total P&L" 
          value={`${(overall.total_pnl || 0) >= 0 ? '+' : ''}$${Math.abs(overall.total_pnl || 0).toFixed(0)}`}
          color={(overall.total_pnl || 0) >= 0 ? 'green' : 'red'}
          sub={`ROI: ${(overall.roi_pct || 0).toFixed(1)}%`}
          icon="💰" 
        />
        <StatCard 
          label="Record" 
          value={`${overall.wins || 0}W - ${overall.losses || 0}L`}
          sub={`Avg Edge: ${((overall.avg_edge || 0) * 100).toFixed(1)}%`}
          icon="📊" 
        />
      </div>

      {/* Current Streak & Recent Results */}
      {trackingStats?.current_streak && (
        <div className="card p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold flex items-center gap-2">
              <span>{trackingStats.current_streak.type === 'WIN' ? '🔥' : '❄️'}</span>
              Current Streak
            </h3>
            <span className={`text-lg font-bold ${trackingStats.current_streak.type === 'WIN' ? 'text-emerald-400' : 'text-red-400'}`}>
              {trackingStats.current_streak.length} {trackingStats.current_streak.type === 'WIN' ? 'Wins' : 'Losses'}
            </span>
          </div>
          <div className="flex gap-1">
            {(trackingStats.recent_results || []).slice(0, 10).map((result, i) => (
              <div 
                key={i}
                className={`w-8 h-8 rounded flex items-center justify-center text-xs font-bold ${
                  result === 'WIN' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
                }`}
              >
                {result === 'WIN' ? 'W' : 'L'}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Bets */}
        <section>
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold flex items-center gap-2">📜 Recent Bets</h3>
            <Link to="/history" className="text-emerald-400 hover:text-emerald-300 text-sm">View All →</Link>
          </div>
          <div className="card p-4 space-y-2">
            {recentBets.length > 0 ? (
              recentBets.slice(0, 5).map((bet, i) => (
                <RecentBetCard key={i} bet={bet} />
              ))
            ) : (
              <p className="text-slate-500 text-center py-4">No recent bets</p>
            )}
          </div>
        </section>

        {/* Top Performing Teams */}
        {trackingStats?.top_teams && trackingStats.top_teams.length > 0 && (
          <section>
            <h3 className="font-semibold flex items-center gap-2 mb-3">🏆 Top Performing Teams</h3>
            <div className="card p-4">
              <div className="space-y-2">
                {trackingStats.top_teams.slice(0, 5).map((team, i) => (
                  <div key={i} className="flex items-center justify-between p-2 bg-slate-800/30 rounded-lg">
                    <div className="flex items-center gap-3">
                      <span className="text-lg font-bold text-slate-500">#{i + 1}</span>
                      <span className="font-medium">{team.team}</span>
                    </div>
                    <div className="flex items-center gap-4 text-sm">
                      <span className="text-slate-400">{team.wins}W/{team.bets}B</span>
                      <span className={`font-bold ${team.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {team.pnl >= 0 ? '+' : ''}${team.pnl.toFixed(0)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}
      </div>

      {/* NHL Section */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <span className="text-xl">🏒</span>
            <h2 className="text-lg font-semibold">NHL Predictions</h2>
          </div>
          <Link to="/nhl" className="text-emerald-400 hover:text-emerald-300 text-sm font-medium">View All →</Link>
        </div>
        
        {nhlGames.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {nhlGames.slice(0, 6).map((game, i) => (
              <NHLGameCard key={game.game_id || i} game={game} />
            ))}
          </div>
        ) : (
          <div className="card p-12 text-center">
            <span className="text-4xl mb-3 block">🏒</span>
            <p className="text-slate-400">No NHL games scheduled today</p>
          </div>
        )}
      </section>

      {/* Performance by Edge Class */}
      {trackingStats?.by_edge_class && trackingStats.by_edge_class.length > 0 && (
        <section>
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-3">Performance by Edge Class</h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {trackingStats.by_edge_class.map((edge, idx) => (
              <div key={idx} className="card p-3">
                <p className="text-xs text-slate-500 uppercase mb-1">{edge.edge_class}</p>
                <p className={`text-xl font-bold ${edge.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {edge.pnl >= 0 ? '+' : ''}${edge.pnl.toFixed(0)}
                </p>
                <p className="text-[10px] text-slate-500 mt-1">
                  {edge.wins}/{edge.bets} ({(edge.win_rate * 100).toFixed(0)}%)
                </p>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* NBA Quick Access */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <span className="text-xl">🏀</span>
            <h2 className="text-lg font-semibold">NBA Games</h2>
          </div>
          <Link to="/nba" className="text-emerald-400 hover:text-emerald-300 text-sm font-medium">View Props →</Link>
        </div>
        
        {nbaData?.games?.length > 0 ? (
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
            {nbaData.games.slice(0, 6).map(game => (
              <Link key={game.game_id} to={`/nba/game/${game.game_id}`} className="card-hover p-3 text-center">
                <p className="font-semibold text-sm">{game.away_team_abbrev || game.away_team}</p>
                <p className="text-slate-500 text-xs">@</p>
                <p className="font-semibold text-sm">{game.home_team_abbrev || game.home_team}</p>
                <p className="text-xs text-slate-500 mt-1">{game.player_predictions?.length || 0} props</p>
              </Link>
            ))}
          </div>
        ) : (
          <div className="card p-8 text-center">
            <p className="text-slate-400">No NBA games today</p>
          </div>
        )}
      </section>

      {/* Status */}
      <div className="card p-3 flex items-center justify-between text-xs">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${health?.status === 'healthy' ? 'bg-emerald-500' : 'bg-red-500'}`}></div>
          <span className="text-slate-400">System {health?.status || 'Unknown'}</span>
        </div>
        <div className="flex gap-4 text-slate-500">
          <span>NBA: <span className={health?.sports?.nba === 'available' ? 'text-emerald-400' : 'text-red-400'}>{health?.sports?.nba}</span></span>
          <span>NHL: <span className={health?.sports?.nhl === 'available' ? 'text-emerald-400' : 'text-red-400'}>{health?.sports?.nhl}</span></span>
          <span>Tracking: <span className={health?.tracking === 'available' ? 'text-emerald-400' : 'text-red-400'}>{health?.tracking}</span></span>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;