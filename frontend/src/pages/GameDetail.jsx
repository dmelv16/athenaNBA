// src/pages/GameDetail.jsx
import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import api from '../services/api';

const GameDetail = ({ sport = 'nba' }) => {
  const { gameId } = useParams();
  const [loading, setLoading] = useState(true);
  const [game, setGame] = useState(null);
  const [error, setError] = useState(null);

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
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-green-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-900/50 border border-red-500 rounded-lg p-6">
        <h2 className="text-xl font-bold text-red-400">Error Loading Game</h2>
        <p className="text-gray-300 mt-2">{error}</p>
        <Link to={`/${sport}`} className="text-green-400 hover:underline mt-4 inline-block">
          ← Back to Predictions
        </Link>
      </div>
    );
  }

  // Group player predictions by team
  const homeTeam = game?.team_predictions?.[0]?.home_team_abbrev;
  const awayTeam = game?.team_predictions?.[0]?.away_team_abbrev;
  
  const playersByTeam = (game?.player_predictions || []).reduce((acc, pred) => {
    const team = pred.team_abbrev;
    if (!acc[team]) acc[team] = [];
    acc[team].push(pred);
    return acc;
  }, {});

  return (
    <div className="space-y-6">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2 text-sm text-gray-400">
        <Link to="/" className="hover:text-white">Dashboard</Link>
        <span>/</span>
        <Link to={`/${sport}`} className="hover:text-white">{sport.toUpperCase()}</Link>
        <span>/</span>
        <span className="text-white">Game {gameId}</span>
      </div>

      {/* Game Header */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <div className="text-center">
          <h1 className="text-3xl font-bold">
            {awayTeam || 'Away'} @ {homeTeam || 'Home'}
          </h1>
          <p className="text-gray-400 mt-2">Game ID: {gameId}</p>
        </div>
      </div>

      {/* Team Predictions */}
      {game?.team_predictions?.length > 0 && (
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h2 className="text-xl font-semibold mb-4">Team Predictions</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {game.team_predictions.map((pred, idx) => (
              <div key={idx} className="bg-gray-700 rounded-lg p-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-400 uppercase">{pred.prop_type}</span>
                  <span className={`px-2 py-1 rounded text-xs ${
                    pred.confidence >= 0.7 ? 'bg-green-600' : 'bg-gray-600'
                  }`}>
                    {(pred.confidence * 100).toFixed(0)}% confidence
                  </span>
                </div>
                <p className="text-2xl font-bold mt-2">
                  {parseFloat(pred.predicted_value).toFixed(1)}
                </p>
                {pred.line && (
                  <p className="text-sm text-gray-400 mt-1">
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
        <div key={team} className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
          <div className="bg-gray-700 px-6 py-3">
            <h2 className="text-xl font-semibold">{team} Players</h2>
          </div>
          
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-750 text-sm text-gray-400">
                <tr>
                  <th className="py-3 px-4 text-left">Player</th>
                  <th className="py-3 px-4 text-left">Prop</th>
                  <th className="py-3 px-4 text-right">Prediction</th>
                  <th className="py-3 px-4 text-right">Line</th>
                  <th className="py-3 px-4 text-right">Edge</th>
                  <th className="py-3 px-4 text-right">Confidence</th>
                  <th className="py-3 px-4 text-center">Call</th>
                </tr>
              </thead>
              <tbody>
                {predictions
                  .sort((a, b) => b.confidence - a.confidence)
                  .map((pred, idx) => (
                    <tr key={idx} className="border-t border-gray-700 hover:bg-gray-750">
                      <td className="py-3 px-4 font-semibold">{pred.player_name}</td>
                      <td className="py-3 px-4 uppercase">{pred.prop_type}</td>
                      <td className="py-3 px-4 text-right font-mono">
                        {parseFloat(pred.predicted_value).toFixed(1)}
                      </td>
                      <td className="py-3 px-4 text-right font-mono">
                        {pred.line || '-'}
                      </td>
                      <td className={`py-3 px-4 text-right font-mono ${
                        pred.edge > 0 ? 'text-green-400' : pred.edge < 0 ? 'text-red-400' : ''
                      }`}>
                        {pred.edge ? (pred.edge > 0 ? '+' : '') + parseFloat(pred.edge).toFixed(1) : '-'}
                      </td>
                      <td className={`py-3 px-4 text-right ${
                        pred.confidence >= 0.7 ? 'text-green-400' : 
                        pred.confidence >= 0.6 ? 'text-yellow-400' : 'text-gray-400'
                      }`}>
                        {(pred.confidence * 100).toFixed(0)}%
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
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      ))}

      {/* No Predictions */}
      {(!game?.player_predictions || game.player_predictions.length === 0) && (
        <div className="bg-gray-800 rounded-xl p-12 border border-gray-700 text-center">
          <p className="text-gray-500">No player predictions available for this game</p>
        </div>
      )}
    </div>
  );
};

export default GameDetail;