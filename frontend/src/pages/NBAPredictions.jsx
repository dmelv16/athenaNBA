// src/pages/NBAPredictions.jsx
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import api from '../services/api';

const PredictionRow = ({ pred }) => {
  const confidence = (pred.confidence * 100).toFixed(0);
  const confColor = confidence >= 70 ? 'text-green-400' : confidence >= 60 ? 'text-yellow-400' : 'text-gray-400';
  
  return (
    <tr className="border-b border-gray-700 hover:bg-gray-750">
      <td className="py-3 px-4">{pred.player_name}</td>
      <td className="py-3 px-4">{pred.team_abbrev}</td>
      <td className="py-3 px-4">vs {pred.opponent_abbrev}</td>
      <td className="py-3 px-4 uppercase font-semibold">{pred.prop_type}</td>
      <td className="py-3 px-4 text-right font-mono">{parseFloat(pred.predicted_value).toFixed(1)}</td>
      <td className="py-3 px-4 text-right font-mono">{pred.line || '-'}</td>
      <td className={`py-3 px-4 text-right font-mono ${pred.edge > 0 ? 'text-green-400' : 'text-red-400'}`}>
        {pred.edge ? (pred.edge > 0 ? '+' : '') + parseFloat(pred.edge).toFixed(1) : '-'}
      </td>
      <td className={`py-3 px-4 text-right font-semibold ${confColor}`}>
        {confidence}%
      </td>
      <td className="py-3 px-4 text-center">
        {pred.recommended_bet && (
          <span className={`px-2 py-1 rounded text-xs font-bold ${
            pred.recommended_bet === 'over' ? 'bg-green-600' : 'bg-red-600'
          }`}>
            {pred.recommended_bet.toUpperCase()}
          </span>
        )}
      </td>
    </tr>
  );
};

const NBAPredictions = () => {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);
  const [filterProp, setFilterProp] = useState('all');
  const [filterTeam, setFilterTeam] = useState('all');
  const [sortBy, setSortBy] = useState('confidence');
  const [sortDir, setSortDir] = useState('desc');

  useEffect(() => {
    loadPredictions();
  }, [selectedDate]);

  const loadPredictions = async () => {
    setLoading(true);
    try {
      const today = new Date().toISOString().split('T')[0];
      const res = selectedDate === today 
        ? await api.getNBATodayPredictions()
        : await api.getNBAPredictionsByDate(selectedDate);
      setData(res);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Get all predictions flattened
  const allPredictions = data?.games?.flatMap(g => g.player_predictions || []) || 
                        data?.player_predictions || [];

  // Get unique values for filters
  const propTypes = [...new Set(allPredictions.map(p => p.prop_type))].sort();
  const teams = [...new Set(allPredictions.map(p => p.team_abbrev))].sort();

  // Filter and sort
  let filtered = allPredictions.filter(p => {
    if (filterProp !== 'all' && p.prop_type !== filterProp) return false;
    if (filterTeam !== 'all' && p.team_abbrev !== filterTeam) return false;
    return true;
  });

  filtered.sort((a, b) => {
    let aVal, bVal;
    switch (sortBy) {
      case 'confidence': aVal = a.confidence; bVal = b.confidence; break;
      case 'edge': aVal = Math.abs(a.edge || 0); bVal = Math.abs(b.edge || 0); break;
      case 'player': aVal = a.player_name; bVal = b.player_name; break;
      case 'prediction': aVal = a.predicted_value; bVal = b.predicted_value; break;
      default: aVal = a.confidence; bVal = b.confidence;
    }
    if (sortDir === 'desc') return bVal > aVal ? 1 : -1;
    return aVal > bVal ? 1 : -1;
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            🏀 NBA Predictions
          </h1>
          <p className="text-gray-400">
            {data?.total_games || 0} games, {allPredictions.length} predictions
          </p>
        </div>
        
        <div className="flex items-center gap-4">
          <input
            type="date"
            value={selectedDate}
            onChange={(e) => setSelectedDate(e.target.value)}
            className="bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white"
          />
          <button
            onClick={loadPredictions}
            className="bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Games Overview */}
      {data?.games && data.games.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
          {data.games.map(game => (
            <Link
              key={game.game_id}
              to={`/nba/game/${game.game_id}`}
              className="bg-gray-800 rounded-lg p-3 border border-gray-700 hover:border-green-500 transition-colors text-center"
            >
              <p className="font-semibold text-sm">
                {game.away_team || 'Away'} @ {game.home_team || 'Home'}
              </p>
              <p className="text-xs text-gray-400 mt-1">
                {(game.player_predictions?.length || 0)} picks
              </p>
            </Link>
          ))}
        </div>
      )}

      {/* Filters */}
      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
        <div className="flex flex-wrap gap-4 items-center">
          <div>
            <label className="text-sm text-gray-400 block mb-1">Prop Type</label>
            <select
              value={filterProp}
              onChange={(e) => setFilterProp(e.target.value)}
              className="bg-gray-700 border border-gray-600 rounded px-3 py-2"
            >
              <option value="all">All Props</option>
              {propTypes.map(p => (
                <option key={p} value={p}>{p.toUpperCase()}</option>
              ))}
            </select>
          </div>
          
          <div>
            <label className="text-sm text-gray-400 block mb-1">Team</label>
            <select
              value={filterTeam}
              onChange={(e) => setFilterTeam(e.target.value)}
              className="bg-gray-700 border border-gray-600 rounded px-3 py-2"
            >
              <option value="all">All Teams</option>
              {teams.map(t => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </div>
          
          <div>
            <label className="text-sm text-gray-400 block mb-1">Sort By</label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="bg-gray-700 border border-gray-600 rounded px-3 py-2"
            >
              <option value="confidence">Confidence</option>
              <option value="edge">Edge</option>
              <option value="player">Player Name</option>
              <option value="prediction">Prediction</option>
            </select>
          </div>
          
          <div>
            <label className="text-sm text-gray-400 block mb-1">Order</label>
            <select
              value={sortDir}
              onChange={(e) => setSortDir(e.target.value)}
              className="bg-gray-700 border border-gray-600 rounded px-3 py-2"
            >
              <option value="desc">Highest First</option>
              <option value="asc">Lowest First</option>
            </select>
          </div>
          
          <div className="ml-auto text-sm text-gray-400">
            Showing {filtered.length} of {allPredictions.length} predictions
          </div>
        </div>
      </div>

      {/* Predictions Table */}
      <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-green-500"></div>
          </div>
        ) : filtered.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-700">
                <tr>
                  <th className="py-3 px-4 text-left">Player</th>
                  <th className="py-3 px-4 text-left">Team</th>
                  <th className="py-3 px-4 text-left">Opponent</th>
                  <th className="py-3 px-4 text-left">Prop</th>
                  <th className="py-3 px-4 text-right">Prediction</th>
                  <th className="py-3 px-4 text-right">Line</th>
                  <th className="py-3 px-4 text-right">Edge</th>
                  <th className="py-3 px-4 text-right">Confidence</th>
                  <th className="py-3 px-4 text-center">Rec</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((pred, idx) => (
                  <PredictionRow key={`${pred.player_id}-${pred.prop_type}-${idx}`} pred={pred} />
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-12 text-gray-500">
            No predictions found for selected filters
          </div>
        )}
      </div>
    </div>
  );
};

export default NBAPredictions;