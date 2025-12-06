// src/pages/NHLPredictions.jsx
import React, { useState, useEffect } from 'react';
import api from '../services/api';

const NHLPredictions = () => {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [summary, setSummary] = useState(null);
  const [activeTab, setActiveTab] = useState('today');

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [todayRes, summaryRes] = await Promise.allSettled([
        api.getNHLTodayPredictions(),
        api.getNHLSummary(30),
      ]);
      
      if (todayRes.status === 'fulfilled') setData(todayRes.value);
      if (summaryRes.status === 'fulfilled') setSummary(summaryRes.value);
    } catch (err) {
      console.error(err);
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

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            🏒 NHL Predictions
          </h1>
          <p className="text-gray-400">
            {data?.games?.length || 0} games today
          </p>
        </div>
        
        <button
          onClick={loadData}
          className="bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg"
        >
          Refresh
        </button>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-gray-700 pb-2">
        <button
          onClick={() => setActiveTab('today')}
          className={`px-4 py-2 rounded-t-lg transition-colors ${
            activeTab === 'today' 
              ? 'bg-green-600 text-white' 
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          Today's Games
        </button>
        <button
          onClick={() => setActiveTab('summary')}
          className={`px-4 py-2 rounded-t-lg transition-colors ${
            activeTab === 'summary' 
              ? 'bg-green-600 text-white' 
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
          }`}
        >
          30-Day Summary
        </button>
      </div>

      {/* Today's Games */}
      {activeTab === 'today' && (
        <div className="space-y-4">
          {data?.games?.length > 0 ? (
            data.games.map((game, idx) => (
              <div key={idx} className="bg-gray-800 rounded-xl p-6 border border-gray-700">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-xl font-semibold">
                    {game.away_team} @ {game.home_team}
                  </h3>
                  <span className="text-gray-400">{game.game_date}</span>
                </div>
                
                {/* Game predictions would go here */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {game.predicted_winner && (
                    <div className="bg-gray-700 rounded-lg p-4">
                      <p className="text-sm text-gray-400">Predicted Winner</p>
                      <p className="text-lg font-semibold text-green-400">{game.predicted_winner}</p>
                    </div>
                  )}
                  {game.win_probability && (
                    <div className="bg-gray-700 rounded-lg p-4">
                      <p className="text-sm text-gray-400">Win Probability</p>
                      <p className="text-lg font-semibold">{(game.win_probability * 100).toFixed(1)}%</p>
                    </div>
                  )}
                  {game.predicted_total && (
                    <div className="bg-gray-700 rounded-lg p-4">
                      <p className="text-sm text-gray-400">Predicted Total</p>
                      <p className="text-lg font-semibold">{game.predicted_total}</p>
                    </div>
                  )}
                  {game.confidence && (
                    <div className="bg-gray-700 rounded-lg p-4">
                      <p className="text-sm text-gray-400">Confidence</p>
                      <p className="text-lg font-semibold">{(game.confidence * 100).toFixed(0)}%</p>
                    </div>
                  )}
                </div>
              </div>
            ))
          ) : (
            <div className="bg-gray-800 rounded-xl p-12 border border-gray-700 text-center">
              <p className="text-gray-500 text-lg">No NHL games scheduled for today</p>
              <p className="text-gray-600 text-sm mt-2">Check back later or view the 30-day summary</p>
            </div>
          )}
        </div>
      )}

      {/* Summary Tab */}
      {activeTab === 'summary' && summary && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <p className="text-gray-400 text-sm">Total Predictions</p>
              <p className="text-3xl font-bold text-white">{summary.total_predictions || 0}</p>
            </div>
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <p className="text-gray-400 text-sm">Win Rate</p>
              <p className={`text-3xl font-bold ${
                (summary.win_rate || 0) > 0.5 ? 'text-green-400' : 'text-red-400'
              }`}>
                {((summary.win_rate || 0) * 100).toFixed(1)}%
              </p>
            </div>
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <p className="text-gray-400 text-sm">ROI</p>
              <p className={`text-3xl font-bold ${
                (summary.roi || 0) > 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {(summary.roi || 0) > 0 ? '+' : ''}{((summary.roi || 0) * 100).toFixed(1)}%
              </p>
            </div>
          </div>
          
          {/* Additional summary data */}
          {summary.by_bet_type && (
            <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
              <h3 className="text-lg font-semibold mb-4">Performance by Bet Type</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {Object.entries(summary.by_bet_type).map(([type, stats]) => (
                  <div key={type} className="bg-gray-700 rounded-lg p-4">
                    <p className="text-sm text-gray-400 uppercase">{type}</p>
                    <p className="text-xl font-bold">{stats.wins}-{stats.losses}</p>
                    <p className="text-sm text-gray-500">
                      {((stats.wins / (stats.wins + stats.losses)) * 100).toFixed(0)}%
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default NHLPredictions;