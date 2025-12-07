// src/pages/GameDetail.jsx - Enhanced with Game Odds
import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import api from '../services/api';

const EDGE_COLORS = {
  EXCEPTIONAL: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
  STRONG: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  GOOD: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  MODERATE: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  MARGINAL: 'bg-slate-500/20 text-slate-400 border-slate-500/30',
  NEGATIVE: 'bg-red-500/20 text-red-400 border-red-500/30',
};

const OddsCard = ({ label, odds, sublabel }) => {
  if (!odds) return null;
  const isPositive = odds.odds_american && !odds.odds_american.startsWith('-');
  
  return (
    <div className="bg-slate-800/50 rounded-lg p-3 text-center">
      <p className="text-[10px] text-slate-500 uppercase mb-1">{label}</p>
      <p className={`text-xl font-bold font-mono ${isPositive ? 'text-emerald-400' : 'text-white'}`}>
        {odds.odds_american || '-'}
      </p>
      {sublabel && <p className="text-xs text-slate-400 mt-1">{sublabel}</p>}
      {odds.line && <p className="text-xs text-slate-500">Line: {odds.line > 0 ? '+' : ''}{odds.line}</p>}
    </div>
  );
};

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
      <td colSpan="9" className="bg-slate-800/30 p-4">
        <div className="flex justify-center"><div className="spinner w-6 h-6"></div></div>
      </td>
    );
  }

  return (
    <td colSpan="9" className="bg-slate-800/30 p-4 border-t border-white/5">
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
  const [gameOdds, setGameOdds] = useState(null);
  const [error, setError] = useState(null);
  const [expandedPlayer, setExpandedPlayer] = useState(null);
  const [activeTab, setActiveTab] = useState('props');

  useEffect(() => {
    loadGame();
  }, [gameId]);

  const loadGame = async () => {
    setLoading(true);
    setError(null);
    try {
      const [gameData, oddsData] = await Promise.allSettled([
        api.getNBAGameDetail(gameId),
        api.getNBAGameOdds(gameId)
      ]);
      
      if (gameData.status === 'fulfilled') setGame(gameData.value);
      if (oddsData.status === 'fulfilled') setGameOdds(oddsData.value);
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

  const homeTeam = game?.team_predictions?.[0]?.home_team_abbrev || gameOdds?.game?.home_team || 'Home';
  const awayTeam = game?.team_predictions?.[0]?.away_team_abbrev || gameOdds?.game?.away_team || 'Away';
  const gameLines = gameOdds?.game_lines || {};
  
  const playersByTeam = (game?.player_predictions || []).reduce((acc, pred) => {
    const team = pred.team_abbrev || 'Unknown';
    if (!acc[team]) acc[team] = [];
    acc[team].push(pred);
    return acc;
  }, {});

  // Count props with positive edge
  const betsCount = (game?.player_predictions || []).filter(p => p.is_bet_recommended).length;

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

      {/* Game Header with Odds */}
      <div className="card p-6">
        <div className="flex items-center justify-center gap-8 mb-6">
          <div className="text-center">
            <p className="text-3xl font-bold">{awayTeam}</p>
            <p className="text-xs text-slate-500 mt-1">Away</p>
          </div>
          <div className="text-center">
            <span className="text-slate-600 text-2xl">@</span>
            <p className="text-[10px] text-slate-500 mt-1">
              {gameOdds?.game?.start_time_mt 
                ? new Date(gameOdds.game.start_time_mt).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})
                : ''}
            </p>
          </div>
          <div className="text-center">
            <p className="text-3xl font-bold">{homeTeam}</p>
            <p className="text-xs text-slate-500 mt-1">Home</p>
          </div>
        </div>

        {/* Game Lines */}
        {(gameLines.moneyline?.home || gameLines.spread?.home || gameLines.total?.over) && (
          <div className="border-t border-white/5 pt-4">
            <h3 className="text-sm font-semibold text-slate-400 uppercase mb-3">Game Lines</h3>
            <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
              <OddsCard 
                label={`${awayTeam} ML`} 
                odds={gameLines.moneyline?.away} 
              />
              <OddsCard 
                label={`${homeTeam} ML`} 
                odds={gameLines.moneyline?.home} 
              />
              <OddsCard 
                label={`${awayTeam} Spread`} 
                odds={gameLines.spread?.away}
                sublabel={gameLines.spread?.away?.line ? `${gameLines.spread.away.line > 0 ? '+' : ''}${gameLines.spread.away.line}` : ''}
              />
              <OddsCard 
                label={`${homeTeam} Spread`} 
                odds={gameLines.spread?.home}
                sublabel={gameLines.spread?.home?.line ? `${gameLines.spread.home.line > 0 ? '+' : ''}${gameLines.spread.home.line}` : ''}
              />
              <OddsCard 
                label="Over" 
                odds={gameLines.total?.over}
                sublabel={gameLines.total?.over?.line ? `O ${gameLines.total.over.line}` : ''}
              />
              <OddsCard 
                label="Under" 
                odds={gameLines.total?.under}
                sublabel={gameLines.total?.under?.line ? `U ${gameLines.total.under.line}` : ''}
              />
            </div>
          </div>
        )}
      </div>

      {/* Stats Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <div className="card p-3">
          <p className="text-[10px] text-slate-500 uppercase">Total Props</p>
          <p className="text-xl font-bold">{game?.player_predictions?.length || 0}</p>
        </div>
        <div className="card p-3">
          <p className="text-[10px] text-slate-500 uppercase">Recommended Bets</p>
          <p className="text-xl font-bold text-emerald-400">{betsCount}</p>
        </div>
        <div className="card p-3">
          <p className="text-[10px] text-slate-500 uppercase">DK Props Available</p>
          <p className="text-xl font-bold">{gameOdds?.prop_count || 0}</p>
        </div>
        <div className="card p-3">
          <p className="text-[10px] text-slate-500 uppercase">Players</p>
          <p className="text-xl font-bold">{gameOdds?.player_count || Object.keys(playersByTeam).reduce((s, k) => s + playersByTeam[k].length, 0)}</p>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-white/5">
        <button
          onClick={() => setActiveTab('props')}
          className={`px-4 py-2 text-sm font-medium transition-colors ${
            activeTab === 'props' ? 'text-emerald-400 border-b-2 border-emerald-400' : 'text-slate-400 hover:text-white'
          }`}
        >
          Player Props ({game?.player_predictions?.length || 0})
        </button>
        <button
          onClick={() => setActiveTab('odds')}
          className={`px-4 py-2 text-sm font-medium transition-colors ${
            activeTab === 'odds' ? 'text-emerald-400 border-b-2 border-emerald-400' : 'text-slate-400 hover:text-white'
          }`}
        >
          DraftKings Odds ({gameOdds?.prop_count || 0})
        </button>
      </div>

      {/* Props Tab */}
      {activeTab === 'props' && Object.entries(playersByTeam).map(([team, predictions]) => (
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
                  <th className="text-center">Direction</th>
                  <th className="text-right">Odds</th>
                  <th className="text-center">Edge</th>
                  <th className="text-center">History</th>
                </tr>
              </thead>
              <tbody>
                {predictions.sort((a, b) => (b.edge || 0) - (a.edge || 0)).map((pred, idx) => (
                  <React.Fragment key={idx}>
                    <tr className={`${expandedPlayer === pred.player_id ? 'bg-white/[0.02]' : ''} ${pred.is_bet_recommended ? 'border-l-2 border-l-emerald-500' : ''}`}>
                      <td className="font-medium">{pred.player_name}</td>
                      <td className="uppercase text-xs text-slate-300">{pred.prop_type}</td>
                      <td className="text-right font-mono">{parseFloat(pred.predicted_value).toFixed(1)}</td>
                      <td className="text-right font-mono text-slate-400">{pred.dk_line || pred.line || '-'}</td>
                      <td className="text-center">
                        {pred.bet_direction ? (
                          <span className={`badge ${pred.bet_direction === 'over' ? 'badge-bet' : 'bg-red-500/20 text-red-400 border border-red-500/30'}`}>
                            {pred.bet_direction.toUpperCase()}
                          </span>
                        ) : pred.recommended_bet ? (
                          <span className={`badge ${pred.recommended_bet === 'over' ? 'badge-bet' : 'bg-red-500/20 text-red-400 border border-red-500/30'}`}>
                            {pred.recommended_bet.toUpperCase()}
                          </span>
                        ) : '-'}
                      </td>
                      <td className="text-right font-mono text-slate-400">{pred.odds_american || '-'}</td>
                      <td className="text-center">
                        {pred.edge_class ? (
                          <span className={`badge border ${EDGE_COLORS[pred.edge_class] || EDGE_COLORS.MARGINAL}`}>
                            {pred.edge ? (pred.edge > 0 ? '+' : '') + (pred.edge * 100).toFixed(1) + '%' : '-'}
                          </span>
                        ) : pred.edge ? (
                          <span className={`font-mono ${pred.edge > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                            {pred.edge > 0 ? '+' : ''}{(pred.edge * 100).toFixed(1)}%
                          </span>
                        ) : '-'}
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

      {/* DraftKings Odds Tab */}
      {activeTab === 'odds' && gameOdds?.player_props && (
        <div className="space-y-4">
          {Object.entries(gameOdds.player_props).map(([playerName, props]) => (
            <div key={playerName} className="card overflow-hidden">
              <div className="bg-slate-800/50 px-4 py-2 border-b border-white/5">
                <h4 className="font-medium">{playerName}</h4>
              </div>
              <div className="p-4">
                <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2">
                  {props.map((prop, idx) => (
                    <div key={idx} className="bg-slate-700/30 rounded-lg p-2 text-center">
                      <p className="text-[10px] text-slate-500 uppercase">{prop.prop_type}</p>
                      <p className="text-sm font-mono">{prop.line}</p>
                      <p className="text-xs text-slate-400">{prop.label}</p>
                      <p className={`text-sm font-mono ${prop.odds_american?.startsWith('-') ? 'text-white' : 'text-emerald-400'}`}>
                        {prop.odds_american}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {(!game?.player_predictions || game.player_predictions.length === 0) && activeTab === 'props' && (
        <div className="card p-12 text-center">
          <span className="text-4xl block mb-2">🏀</span>
          <p className="text-slate-400">No player predictions for this game</p>
        </div>
      )}
    </div>
  );
};

export default GameDetail;