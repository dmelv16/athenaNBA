// src/pages/Dashboard.jsx
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import api from '../services/api';

const StatCard = ({ title, value, subtitle, icon, color = 'green' }) => (
  <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
    <div className="flex items-center justify-between">
      <div>
        <p className="text-gray-400 text-sm">{title}</p>
        <p className={`text-3xl font-bold text-${color}-400 mt-1`}>{value}</p>
        {subtitle && <p className="text-gray-500 text-sm mt-1">{subtitle}</p>}
      </div>
      <span className="text-4xl">{icon}</span>
    </div>
  </div>
);

const GameCard = ({ game, sport }) => (
  <Link 
    to={`/${sport}/game/${game.game_id}`}
    className="block bg-gray-800 rounded-lg p-4 border border-gray-700 hover:border-green-500 transition-colors"
  >
    <div className="flex justify-between items-center mb-2">
      <span className="text-sm text-gray-400">{game.game_date}</span>
      <span className="text-xs bg-green-600 px-2 py-1 rounded">
        {game.team_predictions?.length || 0} team / {game.player_predictions?.length || 0} player
      </span>
    </div>
    <div className="text-lg font-semibold">
      {game.away_team || game.away_team_abbrev} @ {game.home_team || game.home_team_abbrev}
    </div>
  </Link>
);

const Dashboard = () => {
  const [loading, setLoading] = useState(true);
  const [health, setHealth] = useState(null);
  const [nbaData, setNbaData] = useState(null);
  const [nhlData, setNhlData] = useState(null);
  const [accuracy, setAccuracy] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    setLoading(true);
    try {
      const [healthRes, nbaRes, nhlRes, accRes] = await Promise.allSettled([
        api.getHealth(),
        api.getNBATodayPredictions(),
        api.getNHLTodayPredictions(),
        api.getNBAAccuracy(30),
      ]);

      if (healthRes.status === 'fulfilled') setHealth(healthRes.value);
      if (nbaRes.status === 'fulfilled') setNbaData(nbaRes.value);
      if (nhlRes.status === 'fulfilled') setNhlData(nhlRes.value);
      if (accRes.status === 'fulfilled') setAccuracy(accRes.value);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-green-500"></div>
      </div>
    );
  }

  const avgAccuracy = accuracy?.accuracy_by_prop?.reduce((sum, p) => 
    sum + (p.line_accuracy || 0), 0) / (accuracy?.accuracy_by_prop?.length || 1) * 100;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold">Dashboard</h1>
          <p className="text-gray-400">Today's predictions overview</p>
        </div>
        <button 
          onClick={loadDashboardData}
          className="bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg transition-colors"
        >
          Refresh
        </button>
      </div>

      {error && (
        <div className="bg-red-900/50 border border-red-500 rounded-lg p-4">
          <p className="text-red-400">{error}</p>
        </div>
      )}

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard 
          title="NBA Games Today" 
          value={nbaData?.total_games || 0}
          subtitle={`${nbaData?.total_player_predictions || 0} player predictions`}
          icon="🏀"
        />
        <StatCard 
          title="NHL Games Today" 
          value={nhlData?.games?.length || 0}
          icon="🏒"
        />
        <StatCard 
          title="30-Day Accuracy" 
          value={avgAccuracy ? `${avgAccuracy.toFixed(1)}%` : 'N/A'}
          subtitle="vs betting lines"
          icon="🎯"
          color={avgAccuracy > 52.4 ? 'green' : 'red'}
        />
        <StatCard 
          title="API Status" 
          value={health?.status === 'healthy' ? 'Online' : 'Offline'}
          subtitle={`NBA: ${health?.sports?.nba || 'unknown'}`}
          icon="⚡"
          color={health?.status === 'healthy' ? 'green' : 'red'}
        />
      </div>

      {/* Today's Games */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* NBA Games */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold flex items-center gap-2">
              🏀 NBA Games Today
            </h2>
            <Link to="/nba" className="text-green-400 hover:underline text-sm">
              View All →
            </Link>
          </div>
          
          {nbaData?.games?.length > 0 ? (
            <div className="space-y-3">
              {nbaData.games.slice(0, 5).map(game => (
                <GameCard key={game.game_id} game={game} sport="nba" />
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-center py-8">No NBA games today</p>
          )}
        </div>

        {/* NHL Games */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold flex items-center gap-2">
              🏒 NHL Games Today
            </h2>
            <Link to="/nhl" className="text-green-400 hover:underline text-sm">
              View All →
            </Link>
          </div>
          
          {nhlData?.games?.length > 0 ? (
            <div className="space-y-3">
              {nhlData.games.slice(0, 5).map((game, idx) => (
                <div key={idx} className="bg-gray-700 rounded-lg p-4">
                  <p className="font-semibold">{game.away_team} @ {game.home_team}</p>
                  <p className="text-sm text-gray-400">{game.game_date}</p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-center py-8">No NHL games today</p>
          )}
        </div>
      </div>

      {/* Accuracy by Prop Type */}
      {accuracy?.accuracy_by_prop && (
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h2 className="text-xl font-semibold mb-4">30-Day Accuracy by Prop Type</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
            {accuracy.accuracy_by_prop.map(prop => (
              <div key={prop.prop_type} className="bg-gray-700 rounded-lg p-4 text-center">
                <p className="text-sm text-gray-400 uppercase">{prop.prop_type}</p>
                <p className={`text-2xl font-bold ${
                  (prop.line_accuracy || 0) > 0.524 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {prop.line_accuracy ? `${(prop.line_accuracy * 100).toFixed(1)}%` : 'N/A'}
                </p>
                <p className="text-xs text-gray-500">{prop.total_predictions} picks</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;