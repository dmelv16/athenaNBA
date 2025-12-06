// src/pages/HistoricalPicks.jsx - Enhanced Historical Picks
import React, { useState, useEffect } from 'react';
import api from '../services/api';

const HistoricalPicks = () => {
  const [loading, setLoading] = useState(true);
  const [nbaData, setNbaData] = useState([]);
  const [nhlData, setNhlData] = useState([]);
  const [days, setDays] = useState(7);
  const [activeSport, setActiveSport] = useState('nba');
  const [filterProp, setFilterProp] = useState('all');
  const [filterResult, setFilterResult] = useState('all');
  const [page, setPage] = useState(1);
  const perPage = 25;

  useEffect(() => {
    loadHistory();
  }, [days]);

  const loadHistory = async () => {
    setLoading(true);
    try {
      const [nbaRes, nhlRes] = await Promise.allSettled([
        api.getNBAHistory(days),
        api.getNHLResults(days),
      ]);
      
      if (nbaRes.status === 'fulfilled') setNbaData(nbaRes.value.predictions || []);
      if (nhlRes.status === 'fulfilled') setNhlData(nhlRes.value.results || []);
      setPage(1);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const data = activeSport === 'nba' ? nbaData : nhlData;
  const propTypes = activeSport === 'nba' 
    ? [...new Set(nbaData.map(p => p.prop_type))].sort()
    : [];

  // Filter
  let filtered = data.filter(p => {
    if (activeSport === 'nba') {
      if (filterProp !== 'all' && p.prop_type !== filterProp) return false;
    }
    if (filterResult !== 'all') {
      if (filterResult === 'win' && !p.was_correct) return false;
      if (filterResult === 'loss' && p.was_correct !== false) return false;
      if (filterResult === 'pending' && p.was_correct !== null) return false;
    }
    return true;
  });

  // Sort by date
  filtered.sort((a, b) => (b.prediction_date || '').localeCompare(a.prediction_date || ''));

  // Paginate
  const totalPages = Math.ceil(filtered.length / perPage);
  const paginated = filtered.slice((page - 1) * perPage, page * perPage);

  // Group by date
  const groupedByDate = paginated.reduce((acc, pred) => {
    const date = pred.prediction_date;
    if (!acc[date]) acc[date] = [];
    acc[date].push(pred);
    return acc;
  }, {});

  // Calculate stats
  const stats = {
    total: filtered.length,
    wins: filtered.filter(p => p.was_correct === true).length,
    losses: filtered.filter(p => p.was_correct === false).length,
    pending: filtered.filter(p => p.was_correct === null || p.was_correct === undefined).length,
  };
  stats.winRate = stats.wins / (stats.wins + stats.losses) || 0;

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <span className="text-4xl">📜</span>
            Historical Picks
          </h1>
          <p className="text-slate-400 mt-1">
            {stats.total.toLocaleString()} predictions from the last {days} days
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          <select
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
            className="input-field"
          >
            <option value={7}>Last 7 Days</option>
            <option value={14}>Last 14 Days</option>
            <option value={30}>Last 30 Days</option>
            <option value={60}>Last 60 Days</option>
          </select>
          <button onClick={loadHistory} className="btn-primary">
            Refresh
          </button>
        </div>
      </div>

      {/* Sport Toggle */}
      <div className="flex gap-2">
        <button
          onClick={() => { setActiveSport('nba'); setPage(1); setFilterProp('all'); }}
          className={`px-5 py-2.5 rounded-xl font-medium transition-all flex items-center gap-2 ${
            activeSport === 'nba' 
              ? 'bg-amber-500/20 text-amber-400 border border-amber-500/50' 
              : 'bg-slate-800/50 text-slate-400 hover:bg-slate-700'
          }`}
        >
          <span>🏀</span> NBA Props
        </button>
        <button
          onClick={() => { setActiveSport('nhl'); setPage(1); setFilterProp('all'); }}
          className={`px-5 py-2.5 rounded-xl font-medium transition-all flex items-center gap-2 ${
            activeSport === 'nhl' 
              ? 'bg-blue-500/20 text-blue-400 border border-blue-500/50' 
              : 'bg-slate-800/50 text-slate-400 hover:bg-slate-700'
          }`}
        >
          <span>🏒</span> NHL Games
        </button>
      </div>

      {/* Stats Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="glass-card p-4">
          <p className="text-xs text-slate-500 uppercase">Total Picks</p>
          <p className="text-2xl font-bold">{stats.total}</p>
        </div>
        <div className="glass-card p-4">
          <p className="text-xs text-slate-500 uppercase">Wins</p>
          <p className="text-2xl font-bold text-emerald-400">{stats.wins}</p>
        </div>
        <div className="glass-card p-4">
          <p className="text-xs text-slate-500 uppercase">Losses</p>
          <p className="text-2xl font-bold text-red-400">{stats.losses}</p>
        </div>
        <div className="glass-card p-4">
          <p className="text-xs text-slate-500 uppercase">Win Rate</p>
          <p className={`text-2xl font-bold ${stats.winRate > 0.524 ? 'text-emerald-400' : 'text-amber-400'}`}>
            {(stats.winRate * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      {/* Filters */}
      <div className="glass-card p-4 flex flex-wrap gap-4 items-end">
        {activeSport === 'nba' && (
          <div className="flex-1 min-w-[150px]">
            <label className="text-xs text-slate-500 block mb-1.5">Prop Type</label>
            <select
              value={filterProp}
              onChange={(e) => { setFilterProp(e.target.value); setPage(1); }}
              className="input-field w-full"
            >
              <option value="all">All Props</option>
              {propTypes.map(p => (
                <option key={p} value={p}>{p.toUpperCase()}</option>
              ))}
            </select>
          </div>
        )}
        
        <div className="flex-1 min-w-[150px]">
          <label className="text-xs text-slate-500 block mb-1.5">Result</label>
          <select
            value={filterResult}
            onChange={(e) => { setFilterResult(e.target.value); setPage(1); }}
            className="input-field w-full"
          >
            <option value="all">All Results</option>
            <option value="win">Wins Only</option>
            <option value="loss">Losses Only</option>
            <option value="pending">Pending</option>
          </select>
        </div>
        
        <div className="text-sm text-slate-500 whitespace-nowrap">
          Showing {paginated.length} of {filtered.length}
        </div>
      </div>

      {/* Loading */}
      {loading && (
        <div className="flex items-center justify-center h-64">
          <div className="spinner w-12 h-12"></div>
        </div>
      )}

      {/* Data Table */}
      {!loading && Object.keys(groupedByDate).length > 0 && (
        <div className="space-y-6">
          {Object.entries(groupedByDate).map(([date, preds]) => (
            <div key={date} className="glass-card overflow-hidden">
              <div className="bg-slate-800/70 px-6 py-3 flex justify-between items-center border-b border-slate-700/50">
                <h3 className="font-semibold">
                  {new Date(date).toLocaleDateString('en-US', { 
                    weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' 
                  })}
                </h3>
                <span className="text-sm text-slate-400">{preds.length} picks</span>
              </div>
              
              <div className="overflow-x-auto">
                <table className="data-table">
                  <thead>
                    <tr>
                      {activeSport === 'nba' ? (
                        <>
                          <th className="table-header table-cell text-left">Player</th>
                          <th className="table-header table-cell text-left">Matchup</th>
                          <th className="table-header table-cell text-left">Prop</th>
                          <th className="table-header table-cell text-right">Prediction</th>
                          <th className="table-header table-cell text-right">Line</th>
                          <th className="table-header table-cell text-right">Confidence</th>
                          <th className="table-header table-cell text-center">Call</th>
                        </>
                      ) : (
                        <>
                          <th className="table-header table-cell text-left">Matchup</th>
                          <th className="table-header table-cell text-left">Pick</th>
                          <th className="table-header table-cell text-right">Probability</th>
                          <th className="table-header table-cell text-right">Edge</th>
                          <th className="table-header table-cell text-center">Action</th>
                          <th className="table-header table-cell text-center">Result</th>
                          <th className="table-header table-cell text-right">P/L</th>
                        </>
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {preds.map((pred, idx) => (
                      <tr key={idx} className="table-row">
                        {activeSport === 'nba' ? (
                          <>
                            <td className="table-cell font-medium">{pred.player_name}</td>
                            <td className="table-cell text-slate-400">{pred.team_abbrev} vs {pred.opponent_abbrev}</td>
                            <td className="table-cell uppercase font-semibold text-slate-300">{pred.prop_type}</td>
                            <td className="table-cell text-right font-mono">
                              {parseFloat(pred.predicted_value).toFixed(1)}
                            </td>
                            <td className="table-cell text-right font-mono text-slate-400">
                              {pred.line || '-'}
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
                          </>
                        ) : (
                          <>
                            <td className="table-cell font-medium">{pred.away_team} @ {pred.home_team}</td>
                            <td className="table-cell text-emerald-400 font-semibold">{pred.predicted_winner}</td>
                            <td className="table-cell text-right">
                              {((pred.model_probability || 0) * 100).toFixed(0)}%
                            </td>
                            <td className={`table-cell text-right ${(pred.edge || 0) > 0 ? 'edge-positive' : 'text-slate-400'}`}>
                              {pred.edge > 0 ? '+' : ''}{((pred.edge || 0) * 100).toFixed(1)}%
                            </td>
                            <td className="table-cell text-center">
                              <span className={`badge ${pred.action === 'BET' ? 'badge-success' : 'badge-info'}`}>
                                {pred.action || 'HOLD'}
                              </span>
                            </td>
                            <td className="table-cell text-center">
                              {pred.was_correct !== null && pred.was_correct !== undefined ? (
                                <span className={`badge ${pred.was_correct ? 'badge-success' : 'badge-danger'}`}>
                                  {pred.was_correct ? 'WIN' : 'LOSS'}
                                </span>
                              ) : (
                                <span className="badge badge-warning">PENDING</span>
                              )}
                            </td>
                            <td className={`table-cell text-right font-mono ${
                              (pred.profit_loss || 0) > 0 ? 'text-emerald-400' : 
                              (pred.profit_loss || 0) < 0 ? 'text-red-400' : 'text-slate-400'
                            }`}>
                              {pred.profit_loss ? (pred.profit_loss > 0 ? '+' : '') + '$' + pred.profit_loss.toFixed(0) : '-'}
                            </td>
                          </>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* No Data */}
      {!loading && Object.keys(groupedByDate).length === 0 && (
        <div className="glass-card p-16 text-center">
          <span className="text-5xl mb-4 block">📭</span>
          <p className="text-lg text-slate-400">No predictions found</p>
          <p className="text-sm text-slate-500 mt-1">Try adjusting your filters</p>
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex justify-center items-center gap-4">
          <button
            onClick={() => setPage(p => Math.max(1, p - 1))}
            disabled={page === 1}
            className="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Previous
          </button>
          <span className="text-slate-400">
            Page {page} of {totalPages}
          </span>
          <button
            onClick={() => setPage(p => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
            className="btn-secondary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
};

export default HistoricalPicks;