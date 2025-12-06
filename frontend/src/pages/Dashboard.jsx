// src/pages/Dashboard.jsx
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import api from '../services/api';

const StatCard = ({ label, value, sub, color = 'white', icon }) => (
  <div className="card p-4">
    <div className="flex items-start justify-between">
      <div>
        <p className="stat-label mb-1">{label}</p>
        <p className={`stat-lg ${color === 'green' ? 'text-emerald-400' : color === 'red' ? 'text-red-400' : 'text-white'}`}>{value}</p>
        {sub && <p className="text-xs text-slate-500 mt-1">{sub}</p>}
      </div>
      {icon && <span className="text-2xl opacity-60">{icon}</span>}
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
          <span className="font-bold text-emerald-400">{(betPct * 100).toFixed(1)}% bankroll</span>
        </div>
      )}
    </div>
  );
};

const Dashboard = () => {
  const [loading, setLoading] = useState(true);
  const [health, setHealth] = useState(null);
  const [nbaData, setNbaData] = useState(null);
  const [nhlData, setNhlData] = useState(null);
  const [nhlAccuracy, setNhlAccuracy] = useState(null);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [h, nba, nhl, nhlAcc] = await Promise.allSettled([
        api.getHealth(),
        api.getNBATodayPredictions(),
        api.getNHLTodayPredictions(),
        api.getNHLAccuracy(30),
      ]);
      if (h.status === 'fulfilled') setHealth(h.value);
      if (nba.status === 'fulfilled') setNbaData(nba.value);
      if (nhl.status === 'fulfilled') setNhlData(nhl.value);
      if (nhlAcc.status === 'fulfilled') setNhlAccuracy(nhlAcc.value);
    } catch (e) { console.error(e); }
    setLoading(false);
  };

  const nhlGames = nhlData?.games || [];
  const nhlBets = nhlGames.filter(g => g.action === 'BET' || (g.bet_pct_bankroll && g.bet_pct_bankroll > 0));
  const nhlAccPct = nhlAccuracy?.overall_accuracy ? (nhlAccuracy.overall_accuracy * 100).toFixed(1) : 'N/A';

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="spinner w-10 h-10"></div>
      </div>
    );
  }

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

      {/* Stats Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="NHL Games Today" value={nhlGames.length} sub={`${nhlBets.length} recommended bets`} icon="🏒" />
        <StatCard label="NBA Games Today" value={nbaData?.total_games || 0} sub={`${nbaData?.total_player_predictions || 0} player props`} icon="🏀" />
        <StatCard label="NHL 30-Day Accuracy" value={`${nhlAccPct}%`} color={parseFloat(nhlAccPct) > 52 ? 'green' : 'red'} icon="📊" />
        <StatCard 
          label="Total Bets Today" 
          value={nhlBets.length} 
          sub={nhlBets.length > 0 ? `Avg edge: ${(nhlBets.reduce((s,g) => s + (g.edge||0), 0) / nhlBets.length * 100).toFixed(1)}%` : 'No bets'} 
          icon="🎯" 
        />
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

      {/* NHL Accuracy by Confidence */}
      {nhlAccuracy?.by_confidence && (
        <section>
          <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wide mb-3">NHL Performance by Confidence</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {nhlAccuracy.by_confidence.map(bucket => (
              <div key={bucket.bucket} className="card p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-slate-300 capitalize">{bucket.bucket} Confidence</span>
                  <span className="text-xs text-slate-500">{bucket.count} games</span>
                </div>
                <p className={`text-2xl font-bold ${bucket.accuracy > 0.55 ? 'text-emerald-400' : bucket.accuracy > 0.50 ? 'text-amber-400' : 'text-red-400'}`}>
                  {(bucket.accuracy * 100).toFixed(1)}%
                </p>
                <div className="mt-2 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                  <div 
                    className={`h-full rounded-full ${bucket.accuracy > 0.55 ? 'bg-emerald-500' : bucket.accuracy > 0.50 ? 'bg-amber-500' : 'bg-red-500'}`}
                    style={{width: `${bucket.accuracy * 100}%`}}
                  />
                </div>
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
        </div>
      </div>
    </div>
  );
};

export default Dashboard;