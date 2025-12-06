// src/pages/Dashboard.jsx - Professional Dashboard
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import api from '../services/api';

const StatCard = ({ title, value, subtitle, trend, trendValue, icon, color = 'emerald' }) => {
  const colorClasses = {
    emerald: 'from-emerald-500/20 to-emerald-600/5 border-emerald-500/20',
    blue: 'from-blue-500/20 to-blue-600/5 border-blue-500/20',
    amber: 'from-amber-500/20 to-amber-600/5 border-amber-500/20',
    red: 'from-red-500/20 to-red-600/5 border-red-500/20',
  };
  
  const iconColors = {
    emerald: 'text-emerald-400',
    blue: 'text-blue-400',
    amber: 'text-amber-400',
    red: 'text-red-400',
  };

  return (
    <div className={`glass-card p-6 bg-gradient-to-br ${colorClasses[color]}`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="stat-label">{title}</p>
          <p className={`stat-value mt-1 ${iconColors[color]}`}>{value}</p>
          {subtitle && <p className="text-sm text-slate-400 mt-1">{subtitle}</p>}
          {trend && (
            <div className={`flex items-center gap-1 mt-2 text-sm ${trend === 'up' ? 'text-emerald-400' : 'text-red-400'}`}>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                  d={trend === 'up' ? "M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" : "M13 17h8m0 0V9m0 8l-8-8-4 4-6-6"} />
              </svg>
              <span>{trendValue}</span>
            </div>
          )}
        </div>
        <div className={`w-12 h-12 rounded-xl bg-slate-800/50 flex items-center justify-center ${iconColors[color]}`}>
          {icon}
        </div>
      </div>
    </div>
  );
};

const NHLGameCard = ({ game }) => {
  const edgeClass = game.edge > 0.05 ? 'edge-positive' : game.edge > 0 ? 'text-slate-300' : 'edge-negative';
  const confidenceClass = game.model_probability >= 0.65 ? 'confidence-high' : 
                         game.model_probability >= 0.55 ? 'confidence-medium' : 'confidence-low';
  
  return (
    <div className="glass-card-hover p-5">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <span className="text-2xl">🏒</span>
          <span className="text-xs text-slate-500 font-medium">NHL</span>
        </div>
        {game.action === 'BET' && (
          <span className="badge-success">RECOMMENDED</span>
        )}
      </div>
      
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-lg font-semibold">{game.away_team}</p>
            <p className="text-xs text-slate-500">Away</p>
          </div>
          <span className="text-slate-600 font-bold">@</span>
          <div className="text-right">
            <p className="text-lg font-semibold">{game.home_team}</p>
            <p className="text-xs text-slate-500">Home</p>
          </div>
        </div>
        
        <div className="h-px bg-slate-700/50"></div>
        
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <p className="text-xs text-slate-500 mb-1">Pick</p>
            <p className="font-bold text-emerald-400">{game.predicted_winner}</p>
          </div>
          <div>
            <p className="text-xs text-slate-500 mb-1">Edge</p>
            <p className={`font-bold ${edgeClass}`}>
              {game.edge > 0 ? '+' : ''}{(game.edge * 100).toFixed(1)}%
            </p>
          </div>
          <div>
            <p className="text-xs text-slate-500 mb-1">Confidence</p>
            <p className={`font-bold ${confidenceClass}`}>
              {(game.model_probability * 100).toFixed(0)}%
            </p>
          </div>
        </div>
        
        {game.action === 'BET' && (
          <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-lg p-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-slate-400">Suggested Bet</span>
              <span className="font-bold text-emerald-400">
                {(game.bet_pct_bankroll * 100).toFixed(1)}% of bankroll
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const NBAGameCard = ({ game }) => (
  <Link 
    to={`/nba/game/${game.game_id}`}
    className="glass-card-hover p-4 block"
  >
    <div className="flex items-center justify-between mb-3">
      <span className="text-2xl">🏀</span>
      <span className="text-xs bg-slate-700 px-2 py-1 rounded-full text-slate-300">
        {game.player_predictions?.length || 0} props
      </span>
    </div>
    <p className="font-semibold">
      {game.away_team || game.away_team_abbrev} @ {game.home_team || game.home_team_abbrev}
    </p>
    <p className="text-sm text-slate-500 mt-1">{game.game_date}</p>
  </Link>
);

const Dashboard = () => {
  const [loading, setLoading] = useState(true);
  const [health, setHealth] = useState(null);
  const [nbaData, setNbaData] = useState(null);
  const [nhlData, setNhlData] = useState(null);
  const [nhlAccuracy, setNhlAccuracy] = useState(null);
  const [nbaAccuracy, setNbaAccuracy] = useState(null);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    setLoading(true);
    try {
      const [healthRes, nbaRes, nhlRes, nhlAccRes, nbaAccRes] = await Promise.allSettled([
        api.getHealth(),
        api.getNBATodayPredictions(),
        api.getNHLTodayPredictions(),
        api.getNHLAccuracy(30),
        api.getNBAAccuracy(30),
      ]);

      if (healthRes.status === 'fulfilled') setHealth(healthRes.value);
      if (nbaRes.status === 'fulfilled') setNbaData(nbaRes.value);
      if (nhlRes.status === 'fulfilled') setNhlData(nhlRes.value);
      if (nhlAccRes.status === 'fulfilled') setNhlAccuracy(nhlAccRes.value);
      if (nbaAccRes.status === 'fulfilled') setNbaAccuracy(nbaAccRes.value);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const nbaAvgAccuracy = nbaAccuracy?.accuracy_by_prop?.reduce((sum, p) => 
    sum + (p.line_accuracy || 0), 0) / (nbaAccuracy?.accuracy_by_prop?.length || 1) * 100;

  const nhlOverallAccuracy = nhlAccuracy?.overall_accuracy ? nhlAccuracy.overall_accuracy * 100 : null;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="spinner w-12 h-12"></div>
      </div>
    );
  }

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold">Dashboard</h1>
          <p className="text-slate-400 mt-1">Today's predictions overview</p>
        </div>
        <button 
          onClick={loadDashboardData}
          className="btn-primary flex items-center gap-2 self-start"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Refresh
        </button>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard 
          title="NBA Games Today" 
          value={nbaData?.total_games || 0}
          subtitle={`${nbaData?.total_player_predictions || 0} player props`}
          icon={<span className="text-2xl">🏀</span>}
          color="amber"
        />
        <StatCard 
          title="NHL Games Today" 
          value={nhlData?.games?.length || 0}
          subtitle="Moneyline predictions"
          icon={<span className="text-2xl">🏒</span>}
          color="blue"
        />
        <StatCard 
          title="NBA 30-Day Accuracy" 
          value={nbaAvgAccuracy ? `${nbaAvgAccuracy.toFixed(1)}%` : 'N/A'}
          subtitle="vs betting lines"
          trend={nbaAvgAccuracy > 52.4 ? 'up' : 'down'}
          trendValue={nbaAvgAccuracy > 52.4 ? 'Profitable' : 'Below breakeven'}
          icon={<span className="text-2xl">🎯</span>}
          color={nbaAvgAccuracy > 52.4 ? 'emerald' : 'red'}
        />
        <StatCard 
          title="NHL 30-Day Accuracy" 
          value={nhlOverallAccuracy ? `${nhlOverallAccuracy.toFixed(1)}%` : 'N/A'}
          subtitle="Game predictions"
          trend={nhlOverallAccuracy > 52 ? 'up' : 'down'}
          trendValue={nhlOverallAccuracy > 52 ? 'Profitable' : 'Improving'}
          icon={<span className="text-2xl">📊</span>}
          color={nhlOverallAccuracy > 52 ? 'emerald' : 'amber'}
        />
      </div>

      {/* NHL Best Bets - Featured Section */}
      {nhlData?.games?.length > 0 && (
        <div className="glass-card p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-3">
              <span className="text-3xl">🏒</span>
              <div>
                <h2 className="text-xl font-bold">NHL Predictions</h2>
                <p className="text-sm text-slate-400">Today's game picks with betting recommendations</p>
              </div>
            </div>
            <Link to="/nhl" className="text-emerald-400 hover:text-emerald-300 text-sm font-medium flex items-center gap-1">
              View All
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </Link>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {nhlData.games.slice(0, 6).map((game, idx) => (
              <NHLGameCard key={idx} game={game} />
            ))}
          </div>
        </div>
      )}

      {/* Two Column Layout for NBA */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* NBA Games */}
        <div className="glass-card p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <span className="text-2xl">🏀</span>
              <h2 className="text-lg font-semibold">NBA Games Today</h2>
            </div>
            <Link to="/nba" className="text-emerald-400 hover:text-emerald-300 text-sm">
              View All →
            </Link>
          </div>
          
          {nbaData?.games?.length > 0 ? (
            <div className="grid grid-cols-2 gap-3">
              {nbaData.games.slice(0, 6).map(game => (
                <NBAGameCard key={game.game_id} game={game} />
              ))}
            </div>
          ) : (
            <div className="text-center py-12 text-slate-500">
              <span className="text-4xl mb-3 block">🏀</span>
              <p>No NBA games today</p>
            </div>
          )}
        </div>

        {/* Performance Overview */}
        <div className="glass-card p-6">
          <div className="flex items-center gap-3 mb-4">
            <span className="text-2xl">📈</span>
            <h2 className="text-lg font-semibold">Performance Overview</h2>
          </div>
          
          {nbaAccuracy?.accuracy_by_prop && (
            <div className="space-y-4">
              {nbaAccuracy.accuracy_by_prop.map(prop => {
                const accuracy = (prop.line_accuracy || 0) * 100;
                const isProfitable = accuracy > 52.4;
                
                return (
                  <div key={prop.prop_type}>
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-sm font-medium uppercase text-slate-400">{prop.prop_type}</span>
                      <span className={`text-sm font-bold ${isProfitable ? 'text-emerald-400' : 'text-red-400'}`}>
                        {accuracy.toFixed(1)}%
                      </span>
                    </div>
                    <div className="progress-bar">
                      <div 
                        className={`progress-bar-fill ${isProfitable ? 'bg-emerald-500' : 'bg-red-500'}`}
                        style={{ width: `${Math.min(accuracy, 100)}%` }}
                      />
                    </div>
                    <p className="text-xs text-slate-500 mt-1">{prop.total_predictions} predictions</p>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* System Status */}
      <div className="glass-card p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`w-3 h-3 rounded-full ${health?.status === 'healthy' ? 'bg-emerald-500 animate-pulse' : 'bg-red-500'}`}></div>
            <span className="text-sm text-slate-400">System Status: {health?.status || 'Unknown'}</span>
          </div>
          <div className="flex items-center gap-4 text-sm text-slate-500">
            <span>NBA: <span className={health?.sports?.nba === 'available' ? 'text-emerald-400' : 'text-red-400'}>{health?.sports?.nba}</span></span>
            <span>NHL: <span className={health?.sports?.nhl === 'available' ? 'text-emerald-400' : 'text-red-400'}>{health?.sports?.nhl}</span></span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;