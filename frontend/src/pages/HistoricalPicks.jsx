// src/pages/HistoricalPicks.jsx
import React, { useState, useEffect } from 'react';
import api from '../services/api';

const HistoricalPicks = () => {
  const [loading, setLoading] = useState(true);
  const [nbaData, setNbaData] = useState([]);
  const [betResults, setBetResults] = useState([]);
  const [trackingStats, setTrackingStats] = useState(null);
  const [days, setDays] = useState(7);
  const [activeSport, setActiveSport] = useState('nhl');
  const [filterProp, setFilterProp] = useState('all');
  const [filterResult, setFilterResult] = useState('all');
  const [page, setPage] = useState(1);
  const perPage = 30;

  useEffect(() => {
    loadHistory();
  }, [days]);

  const loadHistory = async () => {
    setLoading(true);
    try {
      const [nba, bets, stats] = await Promise.allSettled([
        api.getNBAHistory(days),
        api.getBetResults(days, 200, filterResult === 'all' ? null : filterResult.toUpperCase()),
        api.getTrackingStats(),
      ]);
      if (nba.status === 'fulfilled') setNbaData(nba.value.predictions || []);
      if (bets.status === 'fulfilled') setBetResults(bets.value.bets || []);
      if (stats.status === 'fulfilled') setTrackingStats(stats.value);
      setPage(1);
    } catch (e) { console.error(e); }
    setLoading(false);
  };

  const data = activeSport === 'nba' ? nbaData : betResults;
  const propTypes = activeSport === 'nba' ? [...new Set(nbaData.map(p => p.prop_type))].sort() : [];

  // Filter
  let filtered = data.filter(p => {
    if (activeSport === 'nba') {
      if (filterProp !== 'all' && p.prop_type !== filterProp) return false;
      if (filterResult === 'win' && !p.was_correct) return false;
      if (filterResult === 'loss' && p.was_correct !== false) return false;
      if (filterResult === 'pending' && p.was_correct !== null && p.was_correct !== undefined) return false;
    } else {
      if (filterResult === 'win' && p.bet_result !== 'WIN') return false;
      if (filterResult === 'loss' && p.bet_result !== 'LOSS') return false;
    }
    return true;
  });

  filtered.sort((a, b) => {
    const dateA = a.game_date || a.prediction_date || '';
    const dateB = b.game_date || b.prediction_date || '';
    return dateB.localeCompare(dateA);
  });

  const totalPages = Math.ceil(filtered.length / perPage);
  const paginated = filtered.slice((page - 1) * perPage, page * perPage);

  // Group by date
  const groupedByDate = paginated.reduce((acc, pred) => {
    const date = pred.game_date || pred.prediction_date || 'Unknown';
    if (!acc[date]) acc[date] = [];
    acc[date].push(pred);
    return acc;
  }, {});

  // Calculate stats for NHL
  const nhlStats = {
    total: betResults.length,
    wins: betResults.filter(p => p.bet_result === 'WIN').length,
    losses: betResults.filter(p => p.bet_result === 'LOSS').length,
    totalPnl: betResults.reduce((sum, p) => sum + parseFloat(p.pnl || 0), 0),
  };
  nhlStats.winRate = nhlStats.wins / (nhlStats.wins + nhlStats.losses) || 0;

  // Stats for NBA
  const nbaStats = {
    total: nbaData.length,
    wins: nbaData.filter(p => p.was_correct === true).length,
    losses: nbaData.filter(p => p.was_correct === false).length,
    pending: nbaData.filter(p => p.was_correct === null || p.was_correct === undefined).length,
  };
  nbaStats.winRate = nbaStats.wins / (nbaStats.wins + nbaStats.losses) || 0;

  const stats = activeSport === 'nba' ? nbaStats : nhlStats;

  return (
    <div className="space-y-5 animate-in">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <span>📜</span> Historical Picks
          </h1>
          <p className="text-slate-400 text-sm mt-0.5">{filtered.length} bets from last {days} days</p>
        </div>
        <div className="flex gap-2">
          <select value={days} onChange={(e) => setDays(Number(e.target.value))} className="input-field">
            <option value={7}>Last 7 Days</option>
            <option value={14}>Last 14 Days</option>
            <option value={30}>Last 30 Days</option>
            <option value={60}>Last 60 Days</option>
          </select>
          <button onClick={loadHistory} className="btn-primary text-sm">Refresh</button>
        </div>
      </div>

      {/* Sport Toggle */}
      <div className="flex gap-2">
        <button
          onClick={() => { setActiveSport('nhl'); setPage(1); setFilterProp('all'); }}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 ${
            activeSport === 'nhl' 
              ? 'bg-blue-500/20 text-blue-400 border border-blue-500/30' 
              : 'bg-slate-800/50 text-slate-400 hover:bg-slate-700'
          }`}
        >
          🏒 NHL Bets
        </button>
        <button
          onClick={() => { setActiveSport('nba'); setPage(1); setFilterProp('all'); }}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 ${
            activeSport === 'nba' 
              ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30' 
              : 'bg-slate-800/50 text-slate-400 hover:bg-slate-700'
          }`}
        >
          🏀 NBA Props
        </button>
      </div>

      {/* Stats Summary */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <div className="card p-3">
          <p className="text-[10px] text-slate-500 uppercase">Total Bets</p>
          <p className="text-xl font-bold">{stats.total}</p>
        </div>
        <div className="card p-3">
          <p className="text-[10px] text-slate-500 uppercase">Wins</p>
          <p className="text-xl font-bold text-emerald-400">{stats.wins}</p>
        </div>
        <div className="card p-3">
          <p className="text-[10px] text-slate-500 uppercase">Losses</p>
          <p className="text-xl font-bold text-red-400">{stats.losses}</p>
        </div>
        <div className="card p-3">
          <p className="text-[10px] text-slate-500 uppercase">Win Rate</p>
          <p className={`text-xl font-bold ${stats.winRate > 0.52 ? 'text-emerald-400' : 'text-amber-400'}`}>
            {(stats.winRate * 100).toFixed(1)}%
          </p>
        </div>
        {activeSport === 'nhl' && (
          <div className="card p-3">
            <p className="text-[10px] text-slate-500 uppercase">Total P/L</p>
            <p className={`text-xl font-bold ${nhlStats.totalPnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              {nhlStats.totalPnl >= 0 ? '+' : ''}${nhlStats.totalPnl.toFixed(2)}
            </p>
          </div>
        )}
      </div>

      {/* Filters */}
      <div className="card p-3 flex flex-wrap gap-3 items-end">
        {activeSport === 'nba' && (
          <div className="flex-1 min-w-[140px]">
            <label className="text-[10px] text-slate-500 uppercase block mb-1">Prop Type</label>
            <select value={filterProp} onChange={(e) => { setFilterProp(e.target.value); setPage(1); }} className="input-field w-full">
              <option value="all">All Props</option>
              {propTypes.map(p => <option key={p} value={p}>{p.toUpperCase()}</option>)}
            </select>
          </div>
        )}
        <div className="flex-1 min-w-[140px]">
          <label className="text-[10px] text-slate-500 uppercase block mb-1">Result</label>
          <select value={filterResult} onChange={(e) => { setFilterResult(e.target.value); setPage(1); }} className="input-field w-full">
            <option value="all">All Results</option>
            <option value="win">Wins Only</option>
            <option value="loss">Losses Only</option>
            {activeSport === 'nba' && <option value="pending">Pending</option>}
          </select>
        </div>
        <span className="text-xs text-slate-500">{paginated.length} of {filtered.length}</span>
      </div>

      {loading && (
        <div className="flex items-center justify-center h-48">
          <div className="spinner w-10 h-10"></div>
        </div>
      )}

      {/* Data Table */}
      {!loading && Object.keys(groupedByDate).length > 0 && (
        <div className="space-y-4">
          {Object.entries(groupedByDate).map(([date, preds]) => {
            const dayPnl = activeSport === 'nhl' 
              ? preds.reduce((s, p) => s + parseFloat(p.pnl || 0), 0)
              : null;
            const dayWins = preds.filter(p => activeSport === 'nhl' ? p.bet_result === 'WIN' : p.was_correct === true).length;
            
            return (
              <div key={date} className="card overflow-hidden">
                <div className="bg-slate-800/50 px-4 py-2 flex justify-between items-center border-b border-white/5">
                  <div className="flex items-center gap-3">
                    <span className="font-semibold text-sm">
                      {new Date(date).toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })}
                    </span>
                    <span className="text-xs text-slate-500">{preds.length} picks • {dayWins}W</span>
                  </div>
                  {activeSport === 'nhl' && dayPnl !== null && (
                    <span className={`font-bold ${dayPnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                      {dayPnl >= 0 ? '+' : ''}${dayPnl.toFixed(2)}
                    </span>
                  )}
                </div>
                <div className="overflow-x-auto">
                  <table className="data-table">
                    <thead>
                      <tr>
                        {activeSport === 'nba' ? (
                          <>
                            <th>Player</th>
                            <th>Matchup</th>
                            <th>Prop</th>
                            <th className="text-right">Predicted</th>
                            <th className="text-right">Line</th>
                            <th className="text-right">Actual</th>
                            <th className="text-center">Call</th>
                            <th className="text-center">Result</th>
                          </>
                        ) : (
                          <>
                            <th>Matchup</th>
                            <th>Pick</th>
                            <th className="text-right">Prob</th>
                            <th className="text-right">Edge</th>
                            <th className="text-right">Odds</th>
                            <th className="text-right">Bet Size</th>
                            <th className="text-center">Score</th>
                            <th className="text-center">Result</th>
                            <th className="text-right">P/L</th>
                          </>
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      {preds.map((pred, idx) => (
                        <tr key={idx}>
                          {activeSport === 'nba' ? (
                            <>
                              <td className="font-medium">{pred.player_name}</td>
                              <td className="text-slate-400 text-xs">{pred.team_abbrev} vs {pred.opponent_abbrev}</td>
                              <td className="uppercase font-semibold text-xs text-slate-300">{pred.prop_type}</td>
                              <td className="text-right font-mono">{parseFloat(pred.predicted_value).toFixed(1)}</td>
                              <td className="text-right font-mono text-slate-400">{pred.line || '-'}</td>
                              <td className={`text-right font-mono font-semibold ${
                                pred.actual_value !== null && pred.actual_value !== undefined
                                  ? pred.was_correct ? 'text-emerald-400' : 'text-red-400'
                                  : 'text-slate-400'
                              }`}>
                                {pred.actual_value !== null && pred.actual_value !== undefined 
                                  ? parseFloat(pred.actual_value).toFixed(1) 
                                  : '-'}
                              </td>
                              <td className="text-center">
                                {pred.recommended_bet && (
                                  <span className={`badge ${pred.recommended_bet === 'over' ? 'badge-bet' : 'bg-red-500/20 text-red-400 border border-red-500/30'}`}>
                                    {pred.recommended_bet.toUpperCase()}
                                  </span>
                                )}
                              </td>
                              <td className="text-center">
                                {pred.was_correct !== null && pred.was_correct !== undefined ? (
                                  <span className={pred.was_correct ? 'badge-win' : 'badge-loss'}>
                                    {pred.was_correct ? 'HIT' : 'MISS'}
                                  </span>
                                ) : (
                                  <span className="badge-pending">PENDING</span>
                                )}
                              </td>
                            </>
                          ) : (
                            <>
                              <td className="font-medium">{pred.away_team} @ {pred.home_team}</td>
                              <td className="text-emerald-400 font-medium">{pred.predicted_team}</td>
                              <td className="text-right">{((pred.model_probability || 0) * 100).toFixed(0)}%</td>
                              <td className={`text-right ${(pred.edge || 0) > 0 ? 'text-emerald-400' : 'text-slate-400'}`}>
                                {(pred.edge || 0) > 0 ? '+' : ''}{((pred.edge || 0) * 100).toFixed(1)}%
                              </td>
                              <td className="text-right text-slate-400">
                                {pred.american_odds > 0 ? '+' : ''}{pred.american_odds}
                              </td>
                              <td className="text-right font-mono">${parseFloat(pred.bet_size || 0).toFixed(2)}</td>
                              <td className="text-center text-slate-400">
                                {pred.away_score !== null ? `${pred.away_score}-${pred.home_score}` : '-'}
                              </td>
                              <td className="text-center">
                                <span className={pred.bet_result === 'WIN' ? 'badge-win' : 'badge-loss'}>
                                  {pred.bet_result}
                                </span>
                              </td>
                              <td className={`text-right font-mono font-bold ${
                                parseFloat(pred.pnl || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'
                              }`}>
                                {parseFloat(pred.pnl || 0) >= 0 ? '+' : ''}${parseFloat(pred.pnl || 0).toFixed(2)}
                              </td>
                            </>
                          )}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {!loading && Object.keys(groupedByDate).length === 0 && (
        <div className="card p-12 text-center">
          <span className="text-4xl block mb-2">📭</span>
          <p className="text-slate-400">No picks found</p>
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex justify-center items-center gap-3">
          <button
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page === 1}
            className="btn-secondary text-sm disabled:opacity-50"
          >
            Previous
          </button>
          <span className="text-sm text-slate-400">Page {page} of {totalPages}</span>
          <button
            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
            className="btn-secondary text-sm disabled:opacity-50"
          >
            Next
          </button>
        </div>
      )}

      {/* Cumulative Stats */}
      {trackingStats && activeSport === 'nhl' && (
        <div className="card p-4">
          <h3 className="font-semibold mb-3">Overall Tracking Stats</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <p className="text-slate-500">Current Bankroll</p>
              <p className="text-xl font-bold text-emerald-400">${(trackingStats.current_bankroll || 0).toFixed(2)}</p>
            </div>
            <div>
              <p className="text-slate-500">Starting Bankroll</p>
              <p className="text-xl font-bold">${(trackingStats.starting_bankroll || 0).toFixed(2)}</p>
            </div>
            <div>
              <p className="text-slate-500">Total Return</p>
              <p className={`text-xl font-bold ${(trackingStats.total_return_pct || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {(trackingStats.total_return_pct || 0) >= 0 ? '+' : ''}{(trackingStats.total_return_pct || 0).toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-slate-500">First Bet</p>
              <p className="font-medium">{trackingStats.overall?.first_bet || 'N/A'}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default HistoricalPicks;