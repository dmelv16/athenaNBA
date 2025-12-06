// src/pages/NBABetHistory.jsx - NBA Prop Bet History & Performance
import React, { useState, useEffect } from 'react';
import api from '../services/api';

const EDGE_COLORS = {
  EXCEPTIONAL: 'text-purple-400',
  STRONG: 'text-emerald-400',
  GOOD: 'text-blue-400',
  MODERATE: 'text-amber-400',
  MARGINAL: 'text-slate-400',
};

const NBABetHistory = () => {
  const [loading, setLoading] = useState(true);
  const [betHistory, setBetHistory] = useState([]);
  const [pendingBets, setPendingBets] = useState([]);
  const [performance, setPerformance] = useState(null);
  const [days, setDays] = useState(30);
  const [activeTab, setActiveTab] = useState('pending');
  const [filterResult, setFilterResult] = useState('all');
  const [filterProp, setFilterProp] = useState('all');

  useEffect(() => {
    loadData();
  }, [days]);

  const loadData = async () => {
    setLoading(true);
    try {
      const [pending, history, perf] = await Promise.allSettled([
        api.getNBAPendingBets(),
        api.getNBABetHistory(days),
        api.getNBABetPerformance(days),
      ]);
      
      if (pending.status === 'fulfilled') setPendingBets(pending.value.bets || []);
      if (history.status === 'fulfilled') setBetHistory(history.value.bets || []);
      if (perf.status === 'fulfilled') setPerformance(perf.value);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  };

  const handleUpdateResult = async (betId, actualValue) => {
    try {
      await api.updateNBABetResult(betId, parseFloat(actualValue));
      loadData();
    } catch (e) {
      alert('Error updating result: ' + e.message);
    }
  };

  // Get unique prop types for filter
  const propTypes = [...new Set(betHistory.map(b => b.prop_type))].sort();

  // Filter history
  const filteredHistory = betHistory.filter(bet => {
    if (filterResult !== 'all' && bet.bet_result !== filterResult) return false;
    if (filterProp !== 'all' && bet.prop_type !== filterProp) return false;
    return true;
  });

  // Group by date
  const groupedByDate = filteredHistory.reduce((acc, bet) => {
    const date = bet.game_date || 'Unknown';
    if (!acc[date]) acc[date] = [];
    acc[date].push(bet);
    return acc;
  }, {});

  const overall = performance?.overall || {};
  const byEdge = performance?.by_edge_class || [];
  const byProp = performance?.by_prop_type || [];

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
            <span>🏀</span> NBA Prop Bet History
          </h1>
          <p className="text-slate-400 text-sm mt-0.5">
            {pendingBets.length} pending • {betHistory.length} completed
          </p>
        </div>
        <div className="flex gap-2">
          <select value={days} onChange={(e) => setDays(Number(e.target.value))} className="input-field">
            <option value={7}>Last 7 Days</option>
            <option value={14}>Last 14 Days</option>
            <option value={30}>Last 30 Days</option>
            <option value={60}>Last 60 Days</option>
          </select>
          <button onClick={loadData} className="btn-primary text-sm">Refresh</button>
        </div>
      </div>

      {/* Performance Summary */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <div className="card p-3">
          <p className="text-[10px] text-slate-500 uppercase">Total Bets</p>
          <p className="text-xl font-bold">{overall.total_bets || 0}</p>
        </div>
        <div className="card p-3">
          <p className="text-[10px] text-slate-500 uppercase">Record</p>
          <p className="text-xl font-bold">{overall.wins || 0}W - {overall.losses || 0}L</p>
        </div>
        <div className="card p-3">
          <p className="text-[10px] text-slate-500 uppercase">Win Rate</p>
          <p className={`text-xl font-bold ${((overall.wins || 0) / ((overall.wins || 0) + (overall.losses || 1))) > 0.52 ? 'text-emerald-400' : 'text-amber-400'}`}>
            {(((overall.wins || 0) / ((overall.wins || 0) + (overall.losses || 1))) * 100).toFixed(1)}%
          </p>
        </div>
        <div className="card p-3">
          <p className="text-[10px] text-slate-500 uppercase">Total P&L</p>
          <p className={`text-xl font-bold ${(overall.total_pnl || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {(overall.total_pnl || 0) >= 0 ? '+' : ''}${(overall.total_pnl || 0).toFixed(2)}
          </p>
        </div>
        <div className="card p-3">
          <p className="text-[10px] text-slate-500 uppercase">ROI</p>
          <p className={`text-xl font-bold ${((overall.total_pnl || 0) / (overall.total_staked || 1) * 100) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {(((overall.total_pnl || 0) / (overall.total_staked || 1)) * 100).toFixed(1)}%
          </p>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-white/5">
        {['pending', 'history', 'performance'].map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 text-sm font-medium transition-colors capitalize ${
              activeTab === tab 
                ? 'text-emerald-400 border-b-2 border-emerald-400' 
                : 'text-slate-400 hover:text-white'
            }`}
          >
            {tab} {tab === 'pending' && pendingBets.length > 0 && `(${pendingBets.length})`}
          </button>
        ))}
      </div>

      {/* Pending Tab */}
      {activeTab === 'pending' && (
        <div className="space-y-4">
          {pendingBets.length > 0 ? (
            <div className="card overflow-hidden">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Player</th>
                    <th>Prop</th>
                    <th className="text-right">Line</th>
                    <th className="text-center">Direction</th>
                    <th className="text-right">Odds</th>
                    <th className="text-center">Edge</th>
                    <th className="text-right">Bet</th>
                    <th className="text-center">Result</th>
                  </tr>
                </thead>
                <tbody>
                  {pendingBets.map((bet, idx) => (
                    <tr key={bet.id || idx}>
                      <td className="text-slate-400 text-sm">{bet.game_date}</td>
                      <td className="font-medium">{bet.player_name}</td>
                      <td className="uppercase text-xs text-slate-300">{bet.prop_type}</td>
                      <td className="text-right font-mono">{bet.line}</td>
                      <td className="text-center">
                        <span className={`badge ${bet.bet_direction === 'over' ? 'badge-bet' : 'bg-red-500/20 text-red-400'}`}>
                          {bet.bet_direction?.toUpperCase()}
                        </span>
                      </td>
                      <td className="text-right font-mono">{bet.odds_american}</td>
                      <td className="text-center">
                        <span className={EDGE_COLORS[bet.edge_class] || 'text-slate-400'}>
                          +{((bet.edge || 0) * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className="text-right font-mono">${parseFloat(bet.bet_size || 0).toFixed(2)}</td>
                      <td className="text-center">
                        <input
                          type="number"
                          step="0.5"
                          placeholder="Actual"
                          className="input-field w-20 text-sm"
                          onBlur={(e) => {
                            if (e.target.value) {
                              handleUpdateResult(bet.id, e.target.value);
                            }
                          }}
                          onKeyPress={(e) => {
                            if (e.key === 'Enter' && e.target.value) {
                              handleUpdateResult(bet.id, e.target.value);
                            }
                          }}
                        />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="card p-12 text-center">
              <span className="text-4xl block mb-2">✅</span>
              <p className="text-slate-400">No pending bets</p>
            </div>
          )}
        </div>
      )}

      {/* History Tab */}
      {activeTab === 'history' && (
        <div className="space-y-4">
          {/* Filters */}
          <div className="card p-3 flex gap-3 items-end">
            <div className="flex-1 min-w-[120px]">
              <label className="text-[10px] text-slate-500 uppercase block mb-1">Result</label>
              <select value={filterResult} onChange={(e) => setFilterResult(e.target.value)} className="input-field w-full">
                <option value="all">All Results</option>
                <option value="WIN">Wins</option>
                <option value="LOSS">Losses</option>
                <option value="PUSH">Pushes</option>
              </select>
            </div>
            <div className="flex-1 min-w-[120px]">
              <label className="text-[10px] text-slate-500 uppercase block mb-1">Prop Type</label>
              <select value={filterProp} onChange={(e) => setFilterProp(e.target.value)} className="input-field w-full">
                <option value="all">All Props</option>
                {propTypes.map(p => <option key={p} value={p}>{p?.toUpperCase()}</option>)}
              </select>
            </div>
            <span className="text-xs text-slate-500">{filteredHistory.length} bets</span>
          </div>

          {Object.keys(groupedByDate).length > 0 ? (
            <div className="space-y-4">
              {Object.entries(groupedByDate).sort((a, b) => b[0].localeCompare(a[0])).map(([date, bets]) => {
                const dayPnl = bets.reduce((s, b) => s + parseFloat(b.pnl || 0), 0);
                const dayWins = bets.filter(b => b.bet_result === 'WIN').length;
                
                return (
                  <div key={date} className="card overflow-hidden">
                    <div className="bg-slate-800/50 px-4 py-2 flex justify-between items-center border-b border-white/5">
                      <div className="flex items-center gap-3">
                        <span className="font-semibold text-sm">
                          {new Date(date).toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })}
                        </span>
                        <span className="text-xs text-slate-500">{bets.length} bets • {dayWins}W</span>
                      </div>
                      <span className={`font-bold ${dayPnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {dayPnl >= 0 ? '+' : ''}${dayPnl.toFixed(2)}
                      </span>
                    </div>
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Player</th>
                          <th>Prop</th>
                          <th className="text-right">Line</th>
                          <th className="text-center">Call</th>
                          <th className="text-right">Actual</th>
                          <th className="text-right">Odds</th>
                          <th className="text-right">Bet</th>
                          <th className="text-center">Result</th>
                          <th className="text-right">P&L</th>
                        </tr>
                      </thead>
                      <tbody>
                        {bets.map((bet, idx) => (
                          <tr key={bet.id || idx}>
                            <td className="font-medium">{bet.player_name}</td>
                            <td className="uppercase text-xs text-slate-300">{bet.prop_type}</td>
                            <td className="text-right font-mono">{bet.line}</td>
                            <td className="text-center">
                              <span className={`badge ${bet.bet_direction === 'over' ? 'badge-bet' : 'bg-red-500/20 text-red-400'}`}>
                                {bet.bet_direction?.toUpperCase()}
                              </span>
                            </td>
                            <td className="text-right font-mono font-semibold">
                              {bet.actual_value?.toFixed(1) || '-'}
                            </td>
                            <td className="text-right font-mono text-slate-400">{bet.odds_american}</td>
                            <td className="text-right font-mono">${parseFloat(bet.bet_size || 0).toFixed(2)}</td>
                            <td className="text-center">
                              <span className={bet.bet_result === 'WIN' ? 'badge-win' : bet.bet_result === 'LOSS' ? 'badge-loss' : 'badge-pending'}>
                                {bet.bet_result}
                              </span>
                            </td>
                            <td className={`text-right font-mono font-bold ${parseFloat(bet.pnl || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                              {parseFloat(bet.pnl || 0) >= 0 ? '+' : ''}${parseFloat(bet.pnl || 0).toFixed(2)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                );
              })}
            </div>
          ) : (
            <div className="card p-12 text-center">
              <span className="text-4xl block mb-2">📊</span>
              <p className="text-slate-400">No completed bets yet</p>
            </div>
          )}
        </div>
      )}

      {/* Performance Tab */}
      {activeTab === 'performance' && (
        <div className="space-y-4">
          {/* By Edge Class */}
          {byEdge.length > 0 && (
            <div className="card p-4">
              <h3 className="font-semibold mb-4">Performance by Edge Class</h3>
              <div className="space-y-3">
                {byEdge.map((edge, idx) => {
                  const winRate = edge.wins / (edge.bets || 1);
                  return (
                    <div key={idx} className="flex items-center justify-between p-3 bg-slate-800/30 rounded-lg">
                      <div>
                        <p className={`font-semibold ${EDGE_COLORS[edge.edge_class] || 'text-white'}`}>{edge.edge_class}</p>
                        <p className="text-xs text-slate-500">
                          {edge.bets} bets • {edge.wins}W • Avg: {((edge.avg_edge || 0) * 100).toFixed(1)}% edge
                        </p>
                      </div>
                      <div className="text-right">
                        <p className={`text-xl font-bold ${(edge.pnl || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {(edge.pnl || 0) >= 0 ? '+' : ''}${(edge.pnl || 0).toFixed(2)}
                        </p>
                        <p className="text-xs text-slate-500">{(winRate * 100).toFixed(0)}% win rate</p>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* By Prop Type */}
          {byProp.length > 0 && (
            <div className="card p-4">
              <h3 className="font-semibold mb-4">Performance by Prop Type</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {byProp.map((prop, idx) => {
                  const winRate = prop.wins / (prop.bets || 1);
                  return (
                    <div key={idx} className="bg-slate-800/30 rounded-lg p-3">
                      <p className="uppercase font-semibold text-sm text-slate-300">{prop.prop_type}</p>
                      <p className={`text-lg font-bold mt-1 ${(prop.pnl || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {(prop.pnl || 0) >= 0 ? '+' : ''}${(prop.pnl || 0).toFixed(2)}
                      </p>
                      <p className="text-xs text-slate-500">{prop.wins}/{prop.bets} ({(winRate * 100).toFixed(0)}%)</p>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Info */}
          <div className="card p-4">
            <h3 className="font-semibold mb-3">Understanding NBA Prop Performance</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-slate-400">
              <div>
                <h4 className="text-amber-400 font-medium mb-2">Breakeven Win Rates</h4>
                <ul className="space-y-1">
                  <li>• -110 odds: 52.4% needed</li>
                  <li>• -115 odds: 53.5% needed</li>
                  <li>• -120 odds: 54.5% needed</li>
                </ul>
              </div>
              <div>
                <h4 className="text-emerald-400 font-medium mb-2">Edge Thresholds</h4>
                <ul className="space-y-1">
                  <li>• <span className="text-purple-400">EXCEPTIONAL:</span> 8%+ edge</li>
                  <li>• <span className="text-emerald-400">STRONG:</span> 5-8% edge</li>
                  <li>• <span className="text-blue-400">GOOD:</span> 3-5% edge</li>
                  <li>• <span className="text-amber-400">MODERATE:</span> 2-3% edge (min bet)</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default NBABetHistory;