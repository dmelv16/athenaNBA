// src/pages/GameDetail.jsx
import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import api from '../services/api';

const PlayerHistoryPanel = ({ playerId, onClose }) => {
  const [history, setHistory] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const data = await api.getNBAPlayerHistory(playerId, 5);
        setHistory(data.history || []);
      } catch (e) { console.error(e); }
      setLoading(false);
    };
    load();
  }, [playerId]);

  if (loading) {
    return (
      <td colSpan="8" className="bg-slate-800/30 p-4">
        <div className="flex justify-center"><div className="spinner w-6 h-6"></div></div>
      </td>
    );
  }

  return (
    <td colSpan="8" className="bg-slate-800/30 p-4 border-t border-white/5">
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-semibold text-emerald-400">Last 5 Predictions</span>
        <button onClick={onClose} className="text-slate-400 hover:text-white text-xs">Close ✕</button>
      </div>
      {history && history.length > 0 ? (
        <div className="grid grid-cols-5 gap-2">
          {history.map((h, i) => (
            <div key={i} className={`p-3 rounded-lg text-center ${h.hit === true ? 'bg-emerald-500/10 border border-emerald-500/20' : h.hit === false ? 'bg-red-500/10 border border-red-500/20' : 'bg-slate-700/30'}`}>
              <p className="text-[10px] text-slate-500 mb-1">{h.prediction_date}</p>
              <p className="text-xs text-slate-400 mb-1">vs {h.opponent_abbrev}</p>
              <p className="text-xs uppercase font-semibold text-slate-300 mb-1">{h.prop_type}</p>
              <div className="text-sm space-y-0.5">
                <p className="text-slate-400">Line: {h.line}</p>
                <p className={h.hit === true ? 'text-emerald-400' : h.hit === false ? 'text-red-400' : ''}>
                  Actual: {h.actual_value?.toFixed(1) || '-'}
                </p>
              </div>
              {h.hit !== null && (
                <span className={`inline-block mt-1 text-[10px] px-1.5 py-0.5 rounded ${h.hit ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'}`}>
                  {h.hit ? 'HIT' : 'MISS'}
                </span>
              )}
            </div>
          ))}
        </div>
      ) : (
        <p className="text-slate-500 text-sm text-center">No history available</p>
      )}
    </td>
  );
};

const GameDetail = ({ sport = 'nba' }) => {
  const { gameId } = useParams();
  const [loading, setLoading] = useState(true);
  const [game, setGame] = useState(null);
  const [error, setError] = useState(null);
  const [expandedPlayer, setExpandedPlayer] = useState(null);

  useEffect(() => {
    loadGame();
  }, [gameId]);

  const loadGame = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getNBAGameDetail(gameId);
      setGame(data);
    } catch (e) {
      setError(e.message);
    }
    setLoading(false);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="spinner w-10 h-10"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card p-8 text-center">
        <span className="text-4xl mb-3 block">⚠️</span>
        <h2 className="text-lg font-bold text-red-400">Error Loading Game</h2>
        <p className="text-slate-400 mt-1">{error}</p>
        <Link to="/nba" className="btn-primary mt-4 inline-block text-sm">← Back to NBA</Link>
      </div>
    );
  }

  const homeTeam = game?.team_predictions?.[0]?.home_team_abbrev || 'Home';
  const awayTeam = game?.team_predictions?.[0]?.away_team_abbrev || 'Away';
  
  const playersByTeam = (game?.player_predictions || []).reduce((acc, pred) => {
    const team = pred.team_abbrev || 'Unknown';
    if (!acc[team]) acc[team] = [];
    acc[team].push(pred);
    return acc;
  }, {});

  return (
    <div className="space-y-5 animate-in">
      {/* Breadcrumb */}
      <nav className="flex items-center gap-2 text-sm text-slate-400">
        <Link to="/" className="hover:text-white">Dashboard</Link>
        <span>/</span>
        <Link to="/nba" className="hover:text-white">NBA</Link>
        <span>/</span>
        <span className="text-white">Game</span>
      </nav>

      {/* Game Header */}
      <div className="card p-6">
        <div className="flex items-center justify-center gap-8">
          <div className="text-center">
            <p className="text-3xl font-bold">{awayTeam}</p>
            <p className="text-xs text-slate-500 mt-1">Away</p>
          </div>
          <div className="text-center">
            <span className="text-slate-600 text-2xl">@</span>
            <p className="text-[10px] text-slate-500 mt-1">Game {gameId}</p>
          </div>
          <div className="text-center">
            <p className="text-3xl font-bold">{homeTeam}</p>
            <p className="text-xs text-slate-500 mt-1">Home</p>
          </div>
        </div>
      </div>

      {/* Team Predictions */}
      {game?.team_predictions?.length > 0 && (
        <div className="card p-4">
          <h3 className="font-semibold mb-3 flex items-center gap-2">📊 Team Predictions</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {game.team_predictions.map((pred, idx) => (
              <div key={idx} className="bg-slate-800/50 rounded-lg p-4">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-slate-400 uppercase text-xs">{pred.prop_type}</span>
                </div>
                <p className="text-2xl font-bold">{parseFloat(pred.predicted_value).toFixed(1)}</p>
                {pred.line && (
                  <p className="text-xs text-slate-500 mt-1">Line: {pred.line} | Edge: {pred.edge > 0 ? '+' : ''}{parseFloat(pred.edge).toFixed(1)}</p>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Player Predictions by Team */}
      {Object.entries(playersByTeam).map(([team, predictions]) => (
        <div key={team} className="card overflow-hidden">
          <div className="bg-slate-800/50 px-4 py-3 border-b border-white/5">
            <h3 className="font-semibold flex items-center gap-2">
              🏀 {team}
              <span className="text-xs font-normal text-slate-400">({predictions.length} props)</span>
            </h3>
          </div>
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Player</th>
                  <th>Prop</th>
                  <th className="text-right">Predicted</th>
                  <th className="text-right">Line</th>
                  <th className="text-right">Edge</th>
                  <th className="text-center">Call</th>
                  <th className="text-center">History</th>
                </tr>
              </thead>
              <tbody>
                {predictions.sort((a, b) => Math.abs(b.edge || 0) - Math.abs(a.edge || 0)).map((pred, idx) => (
                  <React.Fragment key={idx}>
                    <tr className={expandedPlayer === pred.player_id ? 'bg-white/[0.02]' : ''}>
                      <td className="font-medium">{pred.player_name}</td>
                      <td className="uppercase text-xs text-slate-300">{pred.prop_type}</td>
                      <td className="text-right font-mono">{parseFloat(pred.predicted_value).toFixed(1)}</td>
                      <td className="text-right font-mono text-slate-400">{pred.line || '-'}</td>
                      <td className={`text-right font-mono ${(pred.edge || 0) > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {pred.edge ? (pred.edge > 0 ? '+' : '') + parseFloat(pred.edge).toFixed(1) : '-'}
                      </td>
                      <td className="text-center">
                        {pred.recommended_bet && (
                          <span className={`badge ${pred.recommended_bet === 'over' ? 'badge-bet' : 'bg-red-500/20 text-red-400 border border-red-500/30'}`}>
                            {pred.recommended_bet.toUpperCase()}
                          </span>
                        )}
                      </td>
                      <td className="text-center">
                        <button
                          onClick={() => setExpandedPlayer(expandedPlayer === pred.player_id ? null : pred.player_id)}
                          className="text-emerald-400 hover:text-emerald-300 text-xs px-2 py-1 rounded hover:bg-white/5"
                        >
                          {expandedPlayer === pred.player_id ? 'Hide' : 'View'}
                        </button>
                      </td>
                    </tr>
                    {expandedPlayer === pred.player_id && (
                      <tr>
                        <PlayerHistoryPanel playerId={pred.player_id} onClose={() => setExpandedPlayer(null)} />
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}

      {(!game?.player_predictions || game.player_predictions.length === 0) && (
        <div className="card p-12 text-center">
          <span className="text-4xl block mb-2">🏀</span>
          <p className="text-slate-400">No player predictions for this game</p>
        </div>
      )}
    </div>
  );
};

export default GameDetail;