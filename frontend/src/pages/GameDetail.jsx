// src/pages/GameDetail.jsx - Enhanced Game Detail Page
import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import api from '../services/api';

const PlayerHistoryModal = ({ player, onClose }) => {
  const [history, setHistory] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadHistory = async () => {
      try {
        const data = await api.getNBAPlayerHistory(player.player_id, 5);
        setHistory(data.history);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    loadHistory();
  }, [player.player_id]);

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="glass-card max-w-2xl w-full max-h-[80vh] overflow-hidden" onClick={e => e.stopPropagation()}>
        <div className="p-6 border-b border-slate-700/50">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold">{player.player_name}</h2>
              <p className="text-slate-400 text-sm">{player.team_abbrev} - Last 5 Predictions</p>
            </div>
            <button onClick={onClose} className="p-2 hover:bg-slate-700 rounded-lg transition-colors">
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
        
        <div className="p-6 overflow-y-auto max-h-[60vh]">
          {loading ? (
            <div className="flex justify-center py-8">
              <div className="spinner w-8 h-8"></div>
            </div>
          ) : history?.length > 0 ? (
            <div className="space-y-3">
              {history.map((h, idx) => (
                <div 
                  key={idx} 
                  className={`p-4 rounded-xl border ${
                    h.hit === true 
                      ? 'bg-emerald-500/10 border-emerald-500/30' 
                      : h.hit === false 
                        ? 'bg-red-500/10 border-red-500/30' 
                        : 'bg-slate-800/50 border-slate-700/50'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-3">
                      <span className="text-sm text-slate-400">{h.prediction_date}</span>
                      <span className="badge badge-info uppercase">{h.prop_type}</span>
                    </div>
                    {h.hit !== null && (
                      <span className={`badge ${h.hit ? 'badge-success' : 'badge-danger'}`}>
                        {h.hit ? '✓ HIT' : '✗ MISS'}
                      </span>
                    )}
                  </div>
                  
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400">vs {h.opponent_abbrev}</span>
                    <div className="flex items-center gap-4 text-sm">
                      <div>
                        <span className="text-slate-500">Predicted:</span>
                        <span className="ml-1 font-semibold">{parseFloat(h.predicted_value).toFixed(1)}</span>
                      </div>
                      {h.line && (
                        <div>
                          <span className="text-slate-500">Line:</span>
                          <span className="ml-1">{h.line}</span>
                        </div>
                      )}
                      <div>
                        <span className="text-slate-500">Actual:</span>
                        <span className={`ml-1 font-semibold ${
                          h.hit === true ? 'text-emerald-400' : h.hit === false ? 'text-red-400' : ''
                        }`}>
                          {h.actual_value?.toFixed(1) || 'Pending'}
                        </span>
                      </div>
                    </div>
                  </div>
                  
                  {h.recommended_bet && (
                    <div className="mt-2 text-sm">
                      <span className="text-slate-500">Call:</span>
                      <span className={`ml-1 font-medium ${h.recommended_bet === 'over' ? 'text-emerald-400' : 'text-red-400'}`}>
                        {h.recommended_bet.toUpperCase()}
                      </span>
                      <span className="text-slate-500 ml-2">({(h.confidence * 100).toFixed(0)}% confidence)</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <p className="text-center text-slate-400 py-8">No prediction history available</p>
          )}
        </div>
      </div>
    </div>
  );
};

const GameDetail = ({ sport = 'nba' }) => {
  const { gameId } = useParams();
  const [loading, setLoading] = useState(true);
  const [game, setGame] = useState(null);
  const [error, setError] = useState(null);
  const [selectedPlayer, setSelectedPlayer] = useState(null);

  useEffect(() => {
    loadGameData();
  }, [gameId]);

  const loadGameData = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getNBAGameDetail(gameId);
      setGame(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="spinner w-12 h-12"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="glass-card p-8 text-center">
        <span className="text-5xl mb-4 block">⚠️</span>
        <h2 className="text-xl font-bold text-red-400">Error Loading Game</h2>
        <p className="text-slate-400 mt-2">{error}</p>
        <Link to={`/${sport}`} className="btn-primary mt-4 inline-block">
          ← Back to Predictions
        </Link>
      </div>
    );
  }

  const homeTeam = game?.team_predictions?.[0]?.home_team_abbrev || 'Home';
  const awayTeam = game?.team_predictions?.[0]?.away_team_abbrev || 'Away';
  
  const playersByTeam = (game?.player_predictions || []).reduce((acc, pred) => {
    const team = pred.team_abbrev;
    if (!acc[team]) acc[team] = [];
    acc[team].push(pred);
    return acc;
  }, {});

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Breadcrumb */}
      <nav className="flex items-center gap-2 text-sm text-slate-400">
        <Link to="/" className="hover:text-white transition-colors">Dashboard</Link>
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
        <Link to={`/${sport}`} className="hover:text-white transition-colors">{sport.toUpperCase()}</Link>
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
        </svg>
        <span className="text-white">Game {gameId}</span>
      </nav>

      {/* Game Header */}
      <div className="glass-card p-8">
        <div className="flex flex-col md:flex-row items-center justify-center gap-8">
          <div className="text-center">
            <p className="text-4xl font-bold">{awayTeam}</p>
            <p className="text-slate-500 mt-1">Away</p>
          </div>
          <div className="flex flex-col items-center">
            <span className="text-3xl text-slate-600">@</span>
            <span className="text-xs text-slate-500 mt-1">Game ID: {gameId}</span>
          </div>
          <div className="text-center">
            <p className="text-4xl font-bold">{homeTeam}</p>
            <p className="text-slate-500 mt-1">Home</p>
          </div>
        </div>
      </div>

      {/* Team Predictions */}
      {game?.team_predictions?.length > 0 && (
        <div className="glass-card p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <span className="text-xl">📊</span>
            Team Predictions
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {game.team_predictions.map((pred, idx) => (
              <div key={idx} className="bg-slate-800/50 rounded-xl p-5">
                <div className="flex justify-between items-center mb-3">
                  <span className="text-slate-400 uppercase text-sm">{pred.prop_type}</span>
                  <span className={`badge ${pred.confidence >= 0.7 ? 'badge-success' : 'badge-warning'}`}>
                    {(pred.confidence * 100).toFixed(0)}% confidence
                  </span>
                </div>
                <p className="text-3xl font-bold">
                  {parseFloat(pred.predicted_value).toFixed(1)}
                </p>
                {pred.line && (
                  <p className="text-sm text-slate-400 mt-2">
                    Line: {pred.line} | Edge: {pred.edge > 0 ? '+' : ''}{parseFloat(pred.edge).toFixed(1)}
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Player Predictions by Team */}
      {Object.entries(playersByTeam).map(([team, predictions]) => (
        <div key={team} className="glass-card overflow-hidden">
          <div className="bg-slate-800/70 px-6 py-4 border-b border-slate-700/50">
            <h2 className="text-lg font-semibold flex items-center gap-2">
              <span className="text-xl">🏀</span>
              {team} Players
              <span className="text-sm font-normal text-slate-400">({predictions.length} props)</span>
            </h2>
          </div>
          
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th className="table-header table-cell text-left">Player</th>
                  <th className="table-header table-cell text-left">Prop</th>
                  <th className="table-header table-cell text-right">Prediction</th>
                  <th className="table-header table-cell text-right">Line</th>
                  <th className="table-header table-cell text-right">Edge</th>
                  <th className="table-header table-cell text-right">Confidence</th>
                  <th className="table-header table-cell text-center">Call</th>
                  <th className="table-header table-cell text-center">History</th>
                </tr>
              </thead>
              <tbody>
                {predictions
                  .sort((a, b) => b.confidence - a.confidence)
                  .map((pred, idx) => (
                    <tr key={idx} className="table-row">
                      <td className="table-cell font-semibold">{pred.player_name}</td>
                      <td className="table-cell uppercase text-slate-300">{pred.prop_type}</td>
                      <td className="table-cell text-right font-mono">
                        {parseFloat(pred.predicted_value).toFixed(1)}
                      </td>
                      <td className="table-cell text-right font-mono text-slate-400">
                        {pred.line || '-'}
                      </td>
                      <td className={`table-cell text-right font-mono ${
                        pred.edge > 0 ? 'edge-positive' : pred.edge < 0 ? 'edge-negative' : ''
                      }`}>
                        {pred.edge ? (pred.edge > 0 ? '+' : '') + parseFloat(pred.edge).toFixed(1) : '-'}
                      </td>
                      <td className={`table-cell text-right ${
                        pred.confidence >= 0.7 ? 'confidence-high' : 
                        pred.confidence >= 0.6 ? 'confidence-medium' : 'confidence-low'
                      }`}>
                        {(pred.confidence * 100).toFixed(0)}%
                      </td>
                      <td className="table-cell text-center">
                        {pred.recommended_bet && (
                          <span className={`badge ${
                            pred.recommended_bet === 'over' ? 'badge-success' : 'badge-danger'
                          }`}>
                            {pred.recommended_bet.toUpperCase()}
                          </span>
                        )}
                      </td>
                      <td className="table-cell text-center">
                        <button
                          onClick={() => setSelectedPlayer(pred)}
                          className="text-emerald-400 hover:text-emerald-300 transition-colors"
                        >
                          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                        </button>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}

      {/* No Predictions */}
      {(!game?.player_predictions || game.player_predictions.length === 0) && (
        <div className="glass-card p-16 text-center">
          <span className="text-5xl mb-4 block">🏀</span>
          <p className="text-lg text-slate-400">No player predictions available for this game</p>
        </div>
      )}

      {/* Player History Modal */}
      {selectedPlayer && (
        <PlayerHistoryModal 
          player={selectedPlayer} 
          onClose={() => setSelectedPlayer(null)} 
        />
      )}
    </div>
  );
};

export default GameDetail;