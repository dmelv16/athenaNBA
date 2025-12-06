// src/pages/NBAPredictions.jsx - Enhanced NBA Predictions Page
import React, { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import api from '../services/api';

const PlayerHistoryDropdown = ({ playerId, playerName, onClose }) => {
  const [history, setHistory] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadHistory = async () => {
      try {
        const data = await api.getNBAPlayerHistory(playerId, 5);
        setHistory(data.history);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    loadHistory();
  }, [playerId]);

  return (
    <div className="absolute left-0 mt-2 w-96 z-50 glass-card p-4 shadow-2xl animate-fade-in">
      <div className="flex items-center justify-between mb-3">
        <h4 className="font-semibold text-emerald-400">{playerName} - Last 5 Predictions</h4>
        <button onClick={onClose} className="text-slate-400 hover:text-white">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
      
      {loading ? (
        <div className="flex justify-center py-4">
          <div className="spinner w-6 h-6"></div>
        </div>
      ) : history?.length > 0 ? (
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {history.map((h, idx) => (
            <div key={idx} className={`p-3 rounded-lg ${h.hit === true ? 'bg-emerald-500/10 border border-emerald-500/20' : h.hit === false ? 'bg-red-500/10 border border-red-500/20' : 'bg-slate-700/50'}`}>
              <div className="flex items-center justify-between text-sm">
                <span className="text-slate-400">{h.prediction_date}</span>
                <span className="font-medium uppercase">{h.prop_type}</span>
              </div>
              <div className="flex items-center justify-between mt-1">
                <div>
                  <span className="text-slate-400 text-xs">vs {h.opponent_abbrev}</span>
                </div>
                <div className="text-right">
                  <div className="flex items-center gap-2">
                    <span className="text-slate-400">Pred: <span className="text-white">{parseFloat(h.predicted_value).toFixed(1)}</span></span>
                    <span className="text-slate-600">|</span>
                    <span className="text-slate-400">Actual: <span className={h.hit === true ? 'text-emerald-400' : h.hit === false ? 'text-red-400' : 'text-white'}>{h.actual_value?.toFixed(1) || '-'}</span></span>
                  </div>
                  {h.line && (
                    <div className="text-xs text-slate-500 mt-0.5">
                      Line: {h.line} | Call: <span className={h.recommended_bet === 'over' ? 'text-emerald-400' : 'text-red-400'}>{h.recommended_bet?.toUpperCase()}</span>
                    </div>
                  )}
                </div>
              </div>
              {h.hit !== null && (
                <div className="mt-1 text-right">
                  <span className={`badge ${h.hit ? 'badge-success' : 'badge-danger'}`}>
                    {h.hit ? '✓ HIT' : '✗ MISS'}
                  </span>
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <p className="text-slate-400 text-center py-4">No history available</p>
      )}
    </div>
  );
};

const PredictionRow = ({ pred, showHistory, onToggleHistory }) => {
  const confidence = (pred.confidence * 100).toFixed(0);
  const confClass = confidence >= 70 ? 'confidence-high' : confidence >= 60 ? 'confidence-medium' : 'confidence-low';
  
  return (
    <tr className="table-row relative">
      <td className="table-cell">
        <button 
          onClick={() => onToggleHistory(pred.player_id, pred.player_name)}
          className="text-left hover:text-emerald-400 transition-colors font-medium"
        >
          {pred.player_name}
          <svg className="w-3 h-3 inline ml-1 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        {showHistory === pred.player_id && (
          <PlayerHistoryDropdown 
            playerId={pred.player_id} 
            playerName={pred.player_name}
            onClose={() => onToggleHistory(null)}
          />
        )}
      </td>
      <td className="table-cell text-slate-400">{pred.team_abbrev}</td>
      <td className="table-cell text-slate-400">vs {pred.opponent_abbrev}</td>
      <td className="table-cell uppercase font-semibold text-slate-300">{pred.prop_type}</td>
      <td className="table-cell text-right font-mono">{parseFloat(pred.predicted_value).toFixed(1)}</td>
      <td className="table-cell text-right font-mono text-slate-400">{pred.line || '-'}</td>
      <td className={`table-cell text-right font-mono ${pred.edge > 0 ? 'edge-positive' : pred.edge < 0 ? 'edge-negative' : ''}`}>
        {pred.edge ? (pred.edge > 0 ? '+' : '') + parseFloat(pred.edge).toFixed(1) : '-'}
      </td>
      <td className={`table-cell text-right font-semibold ${confClass}`}>
        {confidence}%
      </td>
      <td className="table-cell text-center">
        {pred.recommended_bet && (
          <span className={`badge ${pred.recommended_bet === 'over' ? 'badge-success' : 'badge-danger'}`}>
            {pred.recommended_bet.toUpperCase()}
          </span>
        )}
      </td>
    </tr>
  );
};

const GameCard = ({ game, isSelected, onClick }) => (
  <button
    onClick={onClick}
    className={`w-full text-left p-4 rounded-xl transition-all duration-200 ${
      isSelected 
        ? 'bg-emerald-500/20 border border-emerald-500/50 shadow-lg shadow-emerald-500/10' 
        : 'bg-slate-800/50 border border-slate-700/50 hover:bg-slate-800 hover:border-slate-600'
    }`}
  >
    <div className="flex items-center justify-between">
      <div>
        <p className="font-semibold text-sm">
          {game.away_team || 'Away'} @ {game.home_team || 'Home'}
        </p>
        <p className="text-xs text-slate-500 mt-1">
          {game.player_predictions?.length || 0} props
        </p>
      </div>
      {isSelected && (
        <div className="w-2 h-2 rounded-full bg-emerald-500"></div>
      )}
    </div>
  </button>
);

const NBAPredictions = () => {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);
  const [selectedGame, setSelectedGame] = useState(null);
  const [filterProp, setFilterProp] = useState('all');
  const [filterTeam, setFilterTeam] = useState('all');
  const [sortBy, setSortBy] = useState('confidence');
  const [sortDir, setSortDir] = useState('desc');
  const [showHistory, setShowHistory] = useState(null);

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
      if (res?.games?.length > 0) {
        setSelectedGame(res.games[0].game_id);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleToggleHistory = useCallback((playerId) => {
    setShowHistory(prev => prev === playerId ? null : playerId);
  }, []);

  // Get predictions for selected game or all
  const getDisplayPredictions = () => {
    if (!data) return [];
    
    if (selectedGame && data.games) {
      const game = data.games.find(g => g.game_id === selectedGame);
      return game?.player_predictions || [];
    }
    
    return data?.games?.flatMap(g => g.player_predictions || []) || 
           data?.player_predictions || [];
  };

  const allPredictions = getDisplayPredictions();
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
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <span className="text-4xl">🏀</span>
            NBA Predictions
          </h1>
          <p className="text-slate-400 mt-1">
            {data?.total_games || 0} games, {allPredictions.length} player props
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          <input
            type="date"
            value={selectedDate}
            onChange={(e) => setSelectedDate(e.target.value)}
            className="input-field"
          />
          <button onClick={loadPredictions} className="btn-primary">
            Refresh
          </button>
        </div>
      </div>

      {/* Games Selector */}
      {data?.games?.length > 0 && (
        <div className="glass-card p-4">
          <h3 className="text-sm font-semibold text-slate-400 mb-3">SELECT GAME</h3>
          <div className="flex gap-3 overflow-x-auto pb-2">
            <button
              onClick={() => setSelectedGame(null)}
              className={`flex-shrink-0 px-4 py-2 rounded-lg transition-all ${
                selectedGame === null 
                  ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/50' 
                  : 'bg-slate-800/50 text-slate-400 hover:bg-slate-700'
              }`}
            >
              All Games
            </button>
            {data.games.map(game => (
              <GameCard
                key={game.game_id}
                game={game}
                isSelected={selectedGame === game.game_id}
                onClick={() => setSelectedGame(game.game_id)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="glass-card p-4">
        <div className="flex flex-wrap gap-4 items-end">
          <div className="flex-1 min-w-[150px]">
            <label className="text-xs text-slate-500 block mb-1.5">Prop Type</label>
            <select
              value={filterProp}
              onChange={(e) => setFilterProp(e.target.value)}
              className="input-field w-full"
            >
              <option value="all">All Props</option>
              {propTypes.map(p => (
                <option key={p} value={p}>{p.toUpperCase()}</option>
              ))}
            </select>
          </div>
          
          <div className="flex-1 min-w-[150px]">
            <label className="text-xs text-slate-500 block mb-1.5">Team</label>
            <select
              value={filterTeam}
              onChange={(e) => setFilterTeam(e.target.value)}
              className="input-field w-full"
            >
              <option value="all">All Teams</option>
              {teams.map(t => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </div>
          
          <div className="flex-1 min-w-[150px]">
            <label className="text-xs text-slate-500 block mb-1.5">Sort By</label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="input-field w-full"
            >
              <option value="confidence">Confidence</option>
              <option value="edge">Edge</option>
              <option value="player">Player Name</option>
              <option value="prediction">Prediction</option>
            </select>
          </div>
          
          <div className="flex-1 min-w-[150px]">
            <label className="text-xs text-slate-500 block mb-1.5">Order</label>
            <select
              value={sortDir}
              onChange={(e) => setSortDir(e.target.value)}
              className="input-field w-full"
            >
              <option value="desc">Highest First</option>
              <option value="asc">Lowest First</option>
            </select>
          </div>
          
          <div className="text-sm text-slate-500 whitespace-nowrap">
            Showing {filtered.length} of {allPredictions.length}
          </div>
        </div>
      </div>

      {/* Predictions Table */}
      <div className="glass-card overflow-hidden">
        {loading ? (
          <div className="flex items-center justify-center h-64">
            <div className="spinner w-12 h-12"></div>
          </div>
        ) : filtered.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th className="table-header table-cell text-left">Player</th>
                  <th className="table-header table-cell text-left">Team</th>
                  <th className="table-header table-cell text-left">Opponent</th>
                  <th className="table-header table-cell text-left">Prop</th>
                  <th className="table-header table-cell text-right">Prediction</th>
                  <th className="table-header table-cell text-right">Line</th>
                  <th className="table-header table-cell text-right">Edge</th>
                  <th className="table-header table-cell text-right">Confidence</th>
                  <th className="table-header table-cell text-center">Call</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((pred, idx) => (
                  <PredictionRow 
                    key={`${pred.player_id}-${pred.prop_type}-${idx}`} 
                    pred={pred}
                    showHistory={showHistory}
                    onToggleHistory={handleToggleHistory}
                  />
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-16 text-slate-500">
            <span className="text-5xl mb-4 block">🏀</span>
            <p className="text-lg">No predictions found</p>
            <p className="text-sm mt-1">Try adjusting your filters</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default NBAPredictions;