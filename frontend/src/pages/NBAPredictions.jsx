// src/pages/NBAPredictions.jsx - Enhanced with Odds Integration
import React, { useState, useEffect, useMemo, useCallback } from 'react';
import api from '../services/api';

// Edge class colors
const EDGE_COLORS = {
  EXCEPTIONAL: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
  STRONG: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  GOOD: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  MODERATE: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
  MARGINAL: 'bg-slate-500/20 text-slate-400 border-slate-500/30',
  NEGATIVE: 'bg-red-500/20 text-red-400 border-red-500/30',
};

const EdgeBadge = ({ edge, edgeClass }) => {
  const colorClass = EDGE_COLORS[edgeClass] || EDGE_COLORS.MARGINAL;
  return (
    <span className={`badge border ${colorClass}`}>
      {edge > 0 ? '+' : ''}{(edge * 100).toFixed(1)}% {edgeClass}
    </span>
  );
};

const OddsDisplay = ({ prediction }) => {
  if (!prediction.has_odds) {
    return <span className="text-slate-500 text-xs">No odds</span>;
  }
  
  const direction = prediction.bet_direction;
  const isOver = direction === 'over';
  
  return (
    <div className="space-y-1">
      <div className="flex items-center gap-2">
        <span className={`text-xs font-bold ${isOver ? 'text-emerald-400' : 'text-red-400'}`}>
          {direction.toUpperCase()}
        </span>
        <span className="text-xs text-slate-400">
          {prediction.dk_line}
        </span>
        <span className="text-xs font-mono text-white">
          {prediction.odds_american}
        </span>
      </div>
      <div className="text-[10px] text-slate-500">
        Implied: {((prediction.implied_probability || 0) * 100).toFixed(0)}% | 
        Model: {((prediction.model_probability || 0) * 100).toFixed(0)}%
      </div>
    </div>
  );
};

const PlayerHistoryPanel = ({ playerId, playerName, onClose }) => {
  const [history, setHistory] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        const data = await api.getNBAPlayerHistory(playerId, 20);
        setHistory(data.history || []);
      } catch (e) { console.error(e); }
      setLoading(false);
    };
    load();
  }, [playerId]);

  const groupedHistory = useMemo(() => {
    if (!history || history.length === 0) return [];
    const grouped = {};
    history.forEach(h => {
      const key = `${h.prediction_date}-${h.opponent_abbrev}`;
      if (!grouped[key]) {
        grouped[key] = { date: h.prediction_date, opponent: h.opponent_abbrev, props: [] };
      }
      grouped[key].props.push(h);
    });
    return Object.values(grouped).sort((a, b) => new Date(b.date) - new Date(a.date)).slice(0, 5);
  }, [history]);

  if (loading) {
    return (
      <td colSpan="10" className="bg-slate-800/50 p-4">
        <div className="flex justify-center"><div className="spinner w-6 h-6"></div></div>
      </td>
    );
  }

  return (
    <td colSpan="10" className="bg-slate-800/50 p-4">
      <div className="flex items-center justify-between mb-4">
        <span className="text-sm font-semibold text-emerald-400">{playerName} - Last 5 Games</span>
        <button onClick={onClose} className="text-slate-400 hover:text-white text-xs">Close ✕</button>
      </div>
      
      {groupedHistory.length > 0 ? (
        <div className="space-y-3">
          {groupedHistory.map((game, i) => (
            <div key={i} className="bg-slate-700/30 rounded-lg p-3">
              <div className="flex items-center gap-3 mb-2 pb-2 border-b border-slate-600/50">
                <span className="text-xs text-slate-500">{game.date}</span>
                <span className="text-xs text-slate-400">vs {game.opponent}</span>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                {game.props.map((prop, j) => {
                  const hit = prop.hit;
                  return (
                    <div key={j} className={`p-2 rounded text-xs ${
                      hit === true ? 'bg-emerald-500/10 border border-emerald-500/20' : 
                      hit === false ? 'bg-red-500/10 border border-red-500/20' : 'bg-slate-600/30'
                    }`}>
                      <div className="flex justify-between items-center mb-1">
                        <span className="uppercase font-semibold text-slate-300">{prop.prop_type}</span>
                        {hit !== null && (
                          <span className={`text-[10px] px-1.5 py-0.5 rounded ${
                            hit ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
                          }`}>
                            {hit ? 'HIT' : 'MISS'}
                          </span>
                        )}
                      </div>
                      <div className="space-y-0.5 text-[11px]">
                        <div className="flex justify-between">
                          <span className="text-slate-500">Line:</span>
                          <span className="text-slate-400">{prop.line}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">Pred:</span>
                          <span className="text-slate-300">{parseFloat(prop.predicted_value).toFixed(1)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">Actual:</span>
                          <span className={hit === true ? 'text-emerald-400' : hit === false ? 'text-red-400' : 'text-white'}>
                            {prop.actual_value !== null ? prop.actual_value.toFixed(1) : '-'}
                          </span>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-slate-500 text-sm text-center py-4">No history available</p>
      )}
    </td>
  );
};

const NBAPredictions = () => {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState(null);
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);
  const [filterMatchup, setFilterMatchup] = useState('all');
  const [filterTeam, setFilterTeam] = useState('all');
  const [filterProp, setFilterProp] = useState('all');
  const [filterEdge, setFilterEdge] = useState('all');
  const [showBetsOnly, setShowBetsOnly] = useState(false);
  const [sortBy, setSortBy] = useState('edge');
  const [sortDir, setSortDir] = useState('desc');
  const [expandedPlayer, setExpandedPlayer] = useState(null);
  const [bankroll, setBankroll] = useState(1000);
  const [savingBets, setSavingBets] = useState(false);

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const today = new Date().toISOString().split('T')[0];
      const res = selectedDate === today 
        ? await api.getNBATodayPredictions()
        : await api.getNBAPredictionsByDate(selectedDate);
      setData(res);
      if (res.bankroll) setBankroll(res.bankroll);
    } catch (e) { 
      console.error('Failed to load NBA data:', e); 
    }
    setLoading(false);
  }, [selectedDate]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const { allPredictions, gameTeams, stats } = useMemo(() => {
    if (!data) return { allPredictions: [], gameTeams: {}, stats: {} };
    
    let predictions = [];
    const teams = {};
    
    if (data.games && Array.isArray(data.games)) {
      data.games.forEach(game => {
        const away = game.away_team_abbrev || game.away_team;
        const home = game.home_team_abbrev || game.home_team;
        const matchup = `${away} @ ${home}`;
        teams[matchup] = [away, home];
        
        (game.player_predictions || []).forEach(p => {
          predictions.push({ ...p, matchup, game_id: game.game_id || p.game_id });
        });
      });
    }
    
    // Calculate stats
    const withOdds = predictions.filter(p => p.has_odds);
    const bets = predictions.filter(p => p.is_bet_recommended);
    const totalEdge = bets.reduce((s, p) => s + (p.edge || 0), 0);
    
    return {
      allPredictions: predictions,
      gameTeams: teams,
      stats: {
        total: predictions.length,
        withOdds: withOdds.length,
        bets: bets.length,
        totalEdge: totalEdge,
        avgEdge: bets.length > 0 ? totalEdge / bets.length : 0
      }
    };
  }, [data]);

  // Group by player
  const playerGroups = useMemo(() => {
    const groups = {};
    allPredictions.forEach(pred => {
      const key = `${pred.player_id}-${pred.matchup}`;
      if (!groups[key]) {
        groups[key] = {
          player_id: pred.player_id,
          player_name: pred.player_name,
          team_abbrev: pred.team_abbrev,
          matchup: pred.matchup,
          props: []
        };
      }
      groups[key].props.push(pred);
    });
    return Object.values(groups);
  }, [allPredictions]);

  const matchups = useMemo(() => Object.keys(gameTeams).sort(), [gameTeams]);
  
  const teams = useMemo(() => {
    if (filterMatchup !== 'all' && gameTeams[filterMatchup]) {
      return gameTeams[filterMatchup].filter(Boolean).sort();
    }
    const set = new Set();
    Object.values(gameTeams).forEach(([t1, t2]) => {
      if (t1) set.add(t1);
      if (t2) set.add(t2);
    });
    return Array.from(set).sort();
  }, [gameTeams, filterMatchup]);

  const propTypes = useMemo(() => {
    const set = new Set(allPredictions.map(p => p.prop_type).filter(Boolean));
    return Array.from(set).sort();
  }, [allPredictions]);

  // Filter and sort
  const filteredPlayers = useMemo(() => {
    let result = [...playerGroups];
    
    if (filterMatchup !== 'all') {
      const teamsInMatchup = gameTeams[filterMatchup] || [];
      result = result.filter(p => p.matchup === filterMatchup || teamsInMatchup.includes(p.team_abbrev));
    }
    
    if (filterTeam !== 'all') {
      result = result.filter(p => p.team_abbrev === filterTeam);
    }
    
    if (filterProp !== 'all') {
      result = result.filter(p => p.props.some(prop => prop.prop_type === filterProp));
    }
    
    if (filterEdge !== 'all') {
      result = result.filter(p => p.props.some(prop => prop.edge_class === filterEdge));
    }
    
    if (showBetsOnly) {
      result = result.filter(p => p.props.some(prop => prop.is_bet_recommended));
    }
    
    // Sort
    result.sort((a, b) => {
      let aVal, bVal;
      switch (sortBy) {
        case 'edge':
          aVal = Math.max(...a.props.map(p => p.edge || 0));
          bVal = Math.max(...b.props.map(p => p.edge || 0));
          break;
        case 'player':
          aVal = a.player_name || '';
          bVal = b.player_name || '';
          break;
        case 'team':
          aVal = a.team_abbrev || '';
          bVal = b.team_abbrev || '';
          break;
        default:
          aVal = Math.max(...a.props.map(p => p.edge || 0));
          bVal = Math.max(...b.props.map(p => p.edge || 0));
      }
      if (typeof aVal === 'string') {
        return sortDir === 'desc' ? bVal.localeCompare(aVal) : aVal.localeCompare(bVal);
      }
      return sortDir === 'desc' ? bVal - aVal : aVal - bVal;
    });
    
    return result;
  }, [playerGroups, filterMatchup, filterTeam, filterProp, filterEdge, showBetsOnly, sortBy, sortDir, gameTeams]);

  const getBestProp = (props) => {
    if (filterProp !== 'all') {
      return props.find(p => p.prop_type === filterProp) || props[0];
    }
    return props.reduce((best, p) => (p.edge || 0) > (best.edge || 0) ? p : best, props[0]);
  };

  const saveAllBets = async () => {
    setSavingBets(true);
    try {
      const result = await api.saveAllNBABets({ min_edge: 0.02, bankroll });
      alert(`Saved ${result.bets_saved} bets!`);
      loadData();
    } catch (e) {
      alert('Error saving bets: ' + e.message);
    }
    setSavingBets(false);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="spinner w-10 h-10"></div>
      </div>
    );
  }

  return (
    <div className="space-y-5 animate-in">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <span>🏀</span> NBA Props
            {data?.odds_available && <span className="badge bg-emerald-500/20 text-emerald-400 text-xs">LIVE ODDS</span>}
          </h1>
          <p className="text-slate-400 text-sm mt-0.5">
            {data?.total_games || data?.games?.length || 0} games • {stats.total} props • {stats.withOdds} with odds • {stats.bets} bets
          </p>
        </div>
        <div className="flex gap-2">
          <input type="date" value={selectedDate} onChange={(e) => setSelectedDate(e.target.value)} className="input-field" />
          <button onClick={loadData} className="btn-secondary text-sm">Refresh</button>
          {stats.bets > 0 && (
            <button onClick={saveAllBets} disabled={savingBets} className="btn-primary text-sm">
              {savingBets ? 'Saving...' : `Save ${stats.bets} Bets`}
            </button>
          )}
        </div>
      </div>

      {/* Stats Summary */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <div className="card p-3">
          <p className="text-[10px] text-slate-500 uppercase">Total Props</p>
          <p className="text-xl font-bold">{stats.total}</p>
        </div>
        <div className="card p-3">
          <p className="text-[10px] text-slate-500 uppercase">With Odds</p>
          <p className="text-xl font-bold text-emerald-400">{stats.withOdds}</p>
        </div>
        <div className="card p-3">
          <p className="text-[10px] text-slate-500 uppercase">Recommended Bets</p>
          <p className="text-xl font-bold text-amber-400">{stats.bets}</p>
        </div>
        <div className="card p-3">
          <p className="text-[10px] text-slate-500 uppercase">Total Edge</p>
          <p className="text-xl font-bold text-emerald-400">+{(stats.totalEdge * 100).toFixed(1)}%</p>
        </div>
        <div className="card p-3">
          <p className="text-[10px] text-slate-500 uppercase">Bankroll</p>
          <p className="text-xl font-bold">${bankroll.toFixed(0)}</p>
        </div>
      </div>

      {/* Filters */}
      <div className="card p-4 flex flex-wrap gap-3 items-end">
        <div className="flex-1 min-w-[140px]">
          <label className="text-[10px] text-slate-500 uppercase block mb-1">Matchup</label>
          <select value={filterMatchup} onChange={(e) => { setFilterMatchup(e.target.value); setFilterTeam('all'); }} className="input-field w-full">
            <option value="all">All Matchups ({matchups.length})</option>
            {matchups.map(m => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>
        <div className="flex-1 min-w-[100px]">
          <label className="text-[10px] text-slate-500 uppercase block mb-1">Team</label>
          <select value={filterTeam} onChange={(e) => setFilterTeam(e.target.value)} className="input-field w-full">
            <option value="all">All Teams</option>
            {teams.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
        </div>
        <div className="flex-1 min-w-[100px]">
          <label className="text-[10px] text-slate-500 uppercase block mb-1">Prop</label>
          <select value={filterProp} onChange={(e) => setFilterProp(e.target.value)} className="input-field w-full">
            <option value="all">All Props</option>
            {propTypes.map(p => <option key={p} value={p}>{p.toUpperCase()}</option>)}
          </select>
        </div>
        <div className="flex-1 min-w-[100px]">
          <label className="text-[10px] text-slate-500 uppercase block mb-1">Edge Class</label>
          <select value={filterEdge} onChange={(e) => setFilterEdge(e.target.value)} className="input-field w-full">
            <option value="all">All</option>
            <option value="EXCEPTIONAL">Exceptional (8%+)</option>
            <option value="STRONG">Strong (5-8%)</option>
            <option value="GOOD">Good (3-5%)</option>
            <option value="MODERATE">Moderate (2-3%)</option>
          </select>
        </div>
        <div className="flex-1 min-w-[100px]">
          <label className="text-[10px] text-slate-500 uppercase block mb-1">Sort</label>
          <select value={sortBy} onChange={(e) => setSortBy(e.target.value)} className="input-field w-full">
            <option value="edge">Best Edge</option>
            <option value="player">Player</option>
            <option value="team">Team</option>
          </select>
        </div>
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="checkbox" checked={showBetsOnly} onChange={(e) => setShowBetsOnly(e.target.checked)} className="rounded" />
          <span className="text-sm text-slate-400">Bets Only</span>
        </label>
      </div>

      {/* Table */}
      <div className="card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Player</th>
                <th>Team</th>
                <th>Matchup</th>
                <th>Prop</th>
                <th className="text-right">Predicted</th>
                <th className="text-center">Odds</th>
                <th className="text-center">Edge</th>
                <th className="text-right">Bet Size</th>
                <th className="text-center">History</th>
              </tr>
            </thead>
            <tbody>
              {filteredPlayers.map((player, idx) => {
                const uniqueKey = `${player.player_id}-${idx}`;
                const isExpanded = expandedPlayer === uniqueKey;
                const bestProp = getBestProp(player.props);
                const propsToShow = filterProp !== 'all' 
                  ? player.props.filter(p => p.prop_type === filterProp)
                  : player.props.filter(p => p.has_odds).sort((a, b) => (b.edge || 0) - (a.edge || 0)).slice(0, 3);
                
                return (
                  <React.Fragment key={uniqueKey}>
                    <tr className={`${isExpanded ? 'bg-white/[0.02]' : ''} ${bestProp.is_bet_recommended ? 'border-l-2 border-l-emerald-500' : ''}`}>
                      <td className="font-medium">{player.player_name}</td>
                      <td className="text-slate-400">{player.team_abbrev}</td>
                      <td className="text-slate-400 text-xs">{player.matchup}</td>
                      <td>
                        <div className="flex flex-wrap gap-1">
                          {propsToShow.map((prop, i) => (
                            <span 
                              key={i}
                              className={`text-[10px] px-1.5 py-0.5 rounded ${
                                prop.bet_direction === 'over' 
                                  ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' 
                                  : 'bg-red-500/20 text-red-400 border border-red-500/30'
                              }`}
                              title={`Line: ${prop.dk_line} | Edge: ${((prop.edge || 0) * 100).toFixed(1)}%`}
                            >
                              {prop.prop_type?.toUpperCase()}: {prop.bet_direction === 'over' ? '↑' : '↓'} {prop.dk_line}
                            </span>
                          ))}
                        </div>
                      </td>
                      <td className="text-right font-mono">{parseFloat(bestProp.predicted_value || 0).toFixed(1)}</td>
                      <td className="text-center">
                        <OddsDisplay prediction={bestProp} />
                      </td>
                      <td className="text-center">
                        {bestProp.edge_class && (
                          <EdgeBadge edge={bestProp.edge || 0} edgeClass={bestProp.edge_class} />
                        )}
                      </td>
                      <td className="text-right">
                        {bestProp.is_bet_recommended && (
                          <span className="font-mono text-emerald-400">
                            ${((bestProp.bet_pct || 0.02) * bankroll).toFixed(0)}
                          </span>
                        )}
                      </td>
                      <td className="text-center">
                        <button
                          onClick={() => setExpandedPlayer(isExpanded ? null : uniqueKey)}
                          className="text-emerald-400 hover:text-emerald-300 text-xs px-2 py-1 rounded hover:bg-white/5"
                        >
                          {isExpanded ? 'Hide' : 'View'}
                        </button>
                      </td>
                    </tr>
                    {isExpanded && (
                      <tr>
                        <PlayerHistoryPanel 
                          playerId={player.player_id} 
                          playerName={player.player_name}
                          onClose={() => setExpandedPlayer(null)} 
                        />
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
        
        {filteredPlayers.length === 0 && (
          <div className="p-12 text-center text-slate-400">
            <span className="text-4xl block mb-2">🏀</span>
            <p>No players found for selected filters</p>
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="card p-4">
        <h3 className="font-semibold mb-3">Edge Classes</h3>
        <div className="flex flex-wrap gap-2">
          <span className={`badge border ${EDGE_COLORS.EXCEPTIONAL}`}>EXCEPTIONAL 8%+</span>
          <span className={`badge border ${EDGE_COLORS.STRONG}`}>STRONG 5-8%</span>
          <span className={`badge border ${EDGE_COLORS.GOOD}`}>GOOD 3-5%</span>
          <span className={`badge border ${EDGE_COLORS.MODERATE}`}>MODERATE 2-3%</span>
          <span className={`badge border ${EDGE_COLORS.MARGINAL}`}>MARGINAL 0-2%</span>
          <span className={`badge border ${EDGE_COLORS.NEGATIVE}`}>NEGATIVE &lt;0%</span>
        </div>
        <p className="text-xs text-slate-500 mt-3">
          Edge = Model Probability - Implied Probability from odds. Only MODERATE or better edges are recommended as bets.
        </p>
      </div>
    </div>
  );
};

export default NBAPredictions;