// src/pages/NBAPredictions.jsx
import React, { useState, useEffect, useMemo, useCallback } from 'react';
import api from '../services/api';

const PlayerHistoryPanel = ({ playerId, playerName, onClose }) => {
  const [history, setHistory] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const load = async () => {
      try {
        // Request more history records to ensure we get 5 unique games
        const data = await api.getNBAPlayerHistory(playerId, 20);
        console.log('Player history loaded:', data);
        setHistory(data.history || []);
      } catch (e) { console.error(e); }
      setLoading(false);
    };
    load();
  }, [playerId]);

  // Group history by game date
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
    // Sort by date descending and return all games (up to 5)
    return Object.values(grouped)
      .sort((a, b) => new Date(b.date) - new Date(a.date))
      .slice(0, 5);
  }, [history]);

  if (loading) {
    return (
      <td colSpan="8" className="player-expansion p-4">
        <div className="flex justify-center"><div className="spinner w-6 h-6"></div></div>
      </td>
    );
  }

  return (
    <td colSpan="8" className="player-expansion p-4 bg-slate-800/50">
      <div className="flex items-center justify-between mb-4">
        <span className="text-sm font-semibold text-emerald-400">
          {playerName} - Last 5 Games Performance
        </span>
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
                  const predicted = parseFloat(prop.predicted_value) || 0;
                  const actual = prop.actual_value;
                  const line = parseFloat(prop.line) || 0;
                  const hit = prop.hit;
                  const diff = actual !== null ? (actual - predicted).toFixed(1) : null;
                  
                  return (
                    <div 
                      key={j} 
                      className={`p-2 rounded text-xs ${
                        hit === true ? 'bg-emerald-500/10 border border-emerald-500/20' : 
                        hit === false ? 'bg-red-500/10 border border-red-500/20' : 
                        'bg-slate-600/30'
                      }`}
                    >
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
                          <span className="text-slate-400">{line}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">Predicted:</span>
                          <span className="text-slate-300">{predicted.toFixed(1)}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">Actual:</span>
                          <span className={`font-medium ${
                            hit === true ? 'text-emerald-400' : 
                            hit === false ? 'text-red-400' : 'text-white'
                          }`}>
                            {actual !== null ? actual.toFixed(1) : '-'}
                          </span>
                        </div>
                        {diff !== null && (
                          <div className="flex justify-between">
                            <span className="text-slate-500">Diff:</span>
                            <span className={parseFloat(diff) >= 0 ? 'text-emerald-400' : 'text-red-400'}>
                              {parseFloat(diff) >= 0 ? '+' : ''}{diff}
                            </span>
                          </div>
                        )}
                        <div className="flex justify-between mt-1 pt-1 border-t border-slate-600/50">
                          <span className="text-slate-500">Call:</span>
                          <span className={`font-bold ${
                            prop.recommended_bet === 'over' ? 'text-emerald-400' : 'text-red-400'
                          }`}>
                            {prop.recommended_bet?.toUpperCase()}
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
  const [sortBy, setSortBy] = useState('edge');
  const [sortDir, setSortDir] = useState('desc');
  const [expandedPlayer, setExpandedPlayer] = useState(null);

  const loadData = useCallback(async () => {
    setLoading(true);
    try {
      const today = new Date().toISOString().split('T')[0];
      const res = selectedDate === today 
        ? await api.getNBATodayPredictions()
        : await api.getNBAPredictionsByDate(selectedDate);
      setData(res);
      console.log('NBA Data loaded:', res);
    } catch (e) { 
      console.error('Failed to load NBA data:', e); 
    }
    setLoading(false);
  }, [selectedDate]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // Flatten all predictions and track game participants
  const { allPredictions, gameTeams } = useMemo(() => {
    if (!data) return { allPredictions: [], gameTeams: {} };
    
    let predictions = [];
    const teams = {}; // matchup -> [team1, team2]
    
    if (data.games && Array.isArray(data.games)) {
      data.games.forEach(game => {
        const away = game.away_team_abbrev || game.away_team;
        const home = game.home_team_abbrev || game.home_team;
        const matchup = `${away} @ ${home}`;
        teams[matchup] = [away, home];
        
        (game.player_predictions || []).forEach(p => {
          predictions.push({
            ...p,
            matchup,
            game_id: game.game_id || p.game_id
          });
        });
      });
    } else if (data.player_predictions && Array.isArray(data.player_predictions)) {
      data.player_predictions.forEach(p => {
        const matchup = `${p.opponent_abbrev || 'OPP'} @ ${p.team_abbrev || 'TEAM'}`;
        if (!teams[matchup]) teams[matchup] = [p.opponent_abbrev, p.team_abbrev];
        predictions.push({ ...p, matchup });
      });
    } else if (Array.isArray(data)) {
      data.forEach(p => {
        const matchup = `${p.opponent_abbrev || 'OPP'} @ ${p.team_abbrev || 'TEAM'}`;
        if (!teams[matchup]) teams[matchup] = [p.opponent_abbrev, p.team_abbrev];
        predictions.push({ ...p, matchup });
      });
    }
    
    return { allPredictions: predictions, gameTeams: teams };
  }, [data]);

  // Group predictions by player
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

  // Unique values for filters
  const matchups = useMemo(() => {
    return Object.keys(gameTeams).sort();
  }, [gameTeams]);

  const teams = useMemo(() => {
    // If a matchup is selected, only show teams from that matchup
    if (filterMatchup !== 'all' && gameTeams[filterMatchup]) {
      return gameTeams[filterMatchup].filter(Boolean).sort();
    }
    // Otherwise show all teams
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

  // Filter and sort player groups
  const filteredPlayers = useMemo(() => {
    let result = [...playerGroups];
    
    // Matchup filter - show BOTH teams in the matchup
    if (filterMatchup !== 'all') {
      const teamsInMatchup = gameTeams[filterMatchup] || [];
      result = result.filter(p => 
        p.matchup === filterMatchup || teamsInMatchup.includes(p.team_abbrev)
      );
    }
    
    // Team filter
    if (filterTeam !== 'all') {
      result = result.filter(p => p.team_abbrev === filterTeam);
    }
    
    // Prop filter - only include players who have that prop
    if (filterProp !== 'all') {
      result = result.filter(p => p.props.some(prop => prop.prop_type === filterProp));
    }
    
    // Sort
    result.sort((a, b) => {
      let aVal, bVal;
      switch (sortBy) {
        case 'edge':
          aVal = Math.max(...a.props.map(p => Math.abs(parseFloat(p.edge) || 0)));
          bVal = Math.max(...b.props.map(p => Math.abs(parseFloat(p.edge) || 0)));
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
          aVal = Math.max(...a.props.map(p => Math.abs(parseFloat(p.edge) || 0)));
          bVal = Math.max(...b.props.map(p => Math.abs(parseFloat(p.edge) || 0)));
      }
      if (typeof aVal === 'string') {
        return sortDir === 'desc' ? bVal.localeCompare(aVal) : aVal.localeCompare(bVal);
      }
      return sortDir === 'desc' ? bVal - aVal : aVal - bVal;
    });
    
    return result;
  }, [playerGroups, filterMatchup, filterTeam, filterProp, sortBy, sortDir, gameTeams]);

  const togglePlayer = (playerId) => {
    setExpandedPlayer(prev => prev === playerId ? null : playerId);
  };

  // Calculate best edge for display
  const getBestProp = (props) => {
    if (filterProp !== 'all') {
      return props.find(p => p.prop_type === filterProp) || props[0];
    }
    return props.reduce((best, p) => 
      Math.abs(parseFloat(p.edge) || 0) > Math.abs(parseFloat(best.edge) || 0) ? p : best
    , props[0]);
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
          </h1>
          <p className="text-slate-400 text-sm mt-0.5">
            {data?.total_games || data?.games?.length || 0} games • {filteredPlayers.length} players • {allPredictions.length} props
          </p>
        </div>
        <div className="flex gap-2">
          <input
            type="date"
            value={selectedDate}
            onChange={(e) => setSelectedDate(e.target.value)}
            className="input-field"
          />
          <button onClick={loadData} className="btn-primary text-sm">Refresh</button>
        </div>
      </div>

      {/* Filters */}
      <div className="card p-4 flex flex-wrap gap-3 items-end">
        <div className="flex-1 min-w-[160px]">
          <label className="text-[10px] text-slate-500 uppercase block mb-1">Matchup</label>
          <select value={filterMatchup} onChange={(e) => { setFilterMatchup(e.target.value); setFilterTeam('all'); }} className="input-field w-full">
            <option value="all">All Matchups ({matchups.length})</option>
            {matchups.map(m => <option key={m} value={m}>{m}</option>)}
          </select>
        </div>
        <div className="flex-1 min-w-[120px]">
          <label className="text-[10px] text-slate-500 uppercase block mb-1">Team</label>
          <select value={filterTeam} onChange={(e) => setFilterTeam(e.target.value)} className="input-field w-full">
            <option value="all">All Teams ({teams.length})</option>
            {teams.map(t => <option key={t} value={t}>{t}</option>)}
          </select>
        </div>
        <div className="flex-1 min-w-[120px]">
          <label className="text-[10px] text-slate-500 uppercase block mb-1">Prop Type</label>
          <select value={filterProp} onChange={(e) => setFilterProp(e.target.value)} className="input-field w-full">
            <option value="all">All Props</option>
            {propTypes.map(p => <option key={p} value={p}>{p.toUpperCase()}</option>)}
          </select>
        </div>
        <div className="flex-1 min-w-[120px]">
          <label className="text-[10px] text-slate-500 uppercase block mb-1">Sort By</label>
          <select value={sortBy} onChange={(e) => setSortBy(e.target.value)} className="input-field w-full">
            <option value="edge">Best Edge</option>
            <option value="player">Player Name</option>
            <option value="team">Team</option>
          </select>
        </div>
        <div className="flex-1 min-w-[100px]">
          <label className="text-[10px] text-slate-500 uppercase block mb-1">Order</label>
          <select value={sortDir} onChange={(e) => setSortDir(e.target.value)} className="input-field w-full">
            <option value="desc">High → Low</option>
            <option value="asc">Low → High</option>
          </select>
        </div>
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
                <th>Props</th>
                <th className="text-right">Best Edge</th>
                <th className="text-center">Top Call</th>
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
                  : player.props;
                
                return (
                  <React.Fragment key={uniqueKey}>
                    <tr className={isExpanded ? 'bg-white/[0.02]' : ''}>
                      <td className="font-medium">{player.player_name}</td>
                      <td className="text-slate-400">{player.team_abbrev}</td>
                      <td className="text-slate-400 text-xs">{player.matchup}</td>
                      <td>
                        <div className="flex flex-wrap gap-1">
                          {propsToShow.map((prop, i) => (
                            <span 
                              key={i}
                              className={`text-[10px] px-1.5 py-0.5 rounded ${
                                prop.recommended_bet === 'over' 
                                  ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' 
                                  : 'bg-red-500/20 text-red-400 border border-red-500/30'
                              }`}
                              title={`Pred: ${parseFloat(prop.predicted_value || 0).toFixed(1)} | Line: ${prop.line} | Edge: ${prop.edge}`}
                            >
                              {prop.prop_type?.toUpperCase()}: {parseFloat(prop.predicted_value || 0).toFixed(1)} {prop.recommended_bet === 'over' ? '↑' : '↓'}
                            </span>
                          ))}
                        </div>
                      </td>
                      <td className={`text-right font-mono ${parseFloat(bestProp.edge || 0) > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {bestProp.edge ? (parseFloat(bestProp.edge) > 0 ? '+' : '') + parseFloat(bestProp.edge).toFixed(1) : '-'}
                      </td>
                      <td className="text-center">
                        <span className={`badge ${bestProp.recommended_bet === 'over' ? 'badge-bet' : 'bg-red-500/20 text-red-400 border border-red-500/30'}`}>
                          {bestProp.prop_type?.toUpperCase()} {bestProp.recommended_bet?.toUpperCase()}
                        </span>
                      </td>
                      <td className="text-center">
                        <button
                          onClick={() => togglePlayer(isExpanded ? null : uniqueKey)}
                          className="text-emerald-400 hover:text-emerald-300 text-xs px-2 py-1 rounded hover:bg-white/5 transition-colors"
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
    </div>
  );
};

export default NBAPredictions;