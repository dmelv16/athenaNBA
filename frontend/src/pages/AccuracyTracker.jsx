// src/pages/AccuracyTracker.jsx
import React, { useState, useEffect, useMemo } from 'react';
import api from '../services/api';

const ProgressBar = ({ value, label, count, threshold = 0.524 }) => {
  const pct = value * 100;
  const profitable = value > threshold;
  
  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <span className="text-sm font-medium text-slate-300 uppercase">{label}</span>
        <span className={`font-mono font-bold ${profitable ? 'text-emerald-400' : 'text-red-400'}`}>
          {pct.toFixed(1)}%
        </span>
      </div>
      <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
        <div 
          className={`h-full rounded-full transition-all duration-500 ${profitable ? 'bg-emerald-500' : 'bg-red-500'}`}
          style={{ width: `${Math.min(pct, 100)}%` }}
        />
      </div>
      <div className="flex justify-between text-[10px] text-slate-500">
        <span>{count} predictions</span>
        <span>Breakeven: {(threshold * 100).toFixed(1)}%</span>
      </div>
    </div>
  );
};

const BankrollChart = ({ history }) => {
  if (!history || history.length === 0) return null;
  
  const maxBankroll = Math.max(...history.map(h => h.bankroll));
  const minBankroll = Math.min(...history.map(h => h.bankroll));
  const range = maxBankroll - minBankroll || 1;
  
  return (
    <div className="h-32 flex items-end gap-1">
      {history.map((day, i) => {
        const height = ((day.bankroll - minBankroll) / range) * 100;
        const isPositive = day.pnl >= 0;
        return (
          <div key={i} className="flex-1 flex flex-col items-center gap-1" title={`${day.date}: $${day.bankroll.toFixed(2)}`}>
            <div 
              className={`w-full rounded-t transition-all ${isPositive ? 'bg-emerald-500' : 'bg-red-500'}`}
              style={{ height: `${Math.max(height, 5)}%` }}
            />
          </div>
        );
      })}
    </div>
  );
};

const AccuracyTracker = () => {
  const [loading, setLoading] = useState(true);
  const [nbaAccuracy, setNbaAccuracy] = useState(null);
  const [nhlAccuracy, setNhlAccuracy] = useState(null);
  const [trackingStats, setTrackingStats] = useState(null);
  const [trackingPerf, setTrackingPerf] = useState(null);
  const [days, setDays] = useState(30);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    loadData();
  }, [days]);

  const loadData = async () => {
    setLoading(true);
    try {
      const [nbaAcc, nhlAcc, stats, perf] = await Promise.allSettled([
        api.getNBAAccuracy(days),
        api.getNHLAccuracy(days),
        api.getTrackingStats(),
        api.getTrackingPerformance(days),
      ]);
      if (nbaAcc.status === 'fulfilled') setNbaAccuracy(nbaAcc.value);
      if (nhlAcc.status === 'fulfilled') setNhlAccuracy(nhlAcc.value);
      if (stats.status === 'fulfilled') setTrackingStats(stats.value);
      if (perf.status === 'fulfilled') setTrackingPerf(perf.value);
    } catch (e) { console.error(e); }
    setLoading(false);
  };

  const nbaStats = useMemo(() => {
    if (!nbaAccuracy?.accuracy_by_prop) return null;
    const props = nbaAccuracy.accuracy_by_prop;
    const totalPreds = props.reduce((s, p) => s + p.total_predictions, 0);
    const avgAccuracy = props.reduce((s, p) => s + (p.line_accuracy || 0), 0) / props.length;
    const avgError = props.reduce((s, p) => s + (p.avg_error || 0), 0) / props.length;
    const roi = (avgAccuracy * 0.91) - ((1 - avgAccuracy) * 1);
    return { totalPreds, avgAccuracy: avgAccuracy * 100, avgError, roi: roi * 100, profitable: avgAccuracy > 0.524 };
  }, [nbaAccuracy]);

  const overall = trackingStats?.overall || {};
  const byEdge = trackingStats?.by_edge_class || [];

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
            <span>🎯</span> Performance Tracker
          </h1>
          <p className="text-slate-400 text-sm mt-0.5">Track prediction accuracy and betting performance</p>
        </div>
        <div className="flex gap-2">
          <select value={days} onChange={(e) => setDays(Number(e.target.value))} className="input-field">
            <option value={7}>Last 7 Days</option>
            <option value={14}>Last 14 Days</option>
            <option value={30}>Last 30 Days</option>
            <option value={60}>Last 60 Days</option>
            <option value={90}>Last 90 Days</option>
          </select>
          <button onClick={loadData} className="btn-primary text-sm">Refresh</button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-white/5">
        {['overview', 'nhl-tracking', 'nba', 'by-edge'].map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 text-sm font-medium transition-colors capitalize ${
              activeTab === tab 
                ? 'text-emerald-400 border-b-2 border-emerald-400' 
                : 'text-slate-400 hover:text-white'
            }`}
          >
            {tab === 'nhl-tracking' ? 'NHL Tracking' : tab === 'by-edge' ? 'By Edge' : tab.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Overview */}
      {activeTab === 'overview' && (
        <div className="space-y-5">
          {/* Bankroll Hero */}
          <div className="card p-6 border-emerald-500/20 bg-gradient-to-r from-emerald-500/10 to-transparent">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div>
                <p className="text-slate-400 text-sm">Current Bankroll</p>
                <p className="text-3xl font-bold text-emerald-400">${(trackingStats?.current_bankroll || 0).toFixed(2)}</p>
              </div>
              <div>
                <p className="text-slate-400 text-sm">Total P&L</p>
                <p className={`text-3xl font-bold ${(overall.total_pnl || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {(overall.total_pnl || 0) >= 0 ? '+' : ''}${(overall.total_pnl || 0).toFixed(2)}
                </p>
              </div>
              <div>
                <p className="text-slate-400 text-sm">ROI</p>
                <p className={`text-3xl font-bold ${(overall.roi_pct || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {(overall.roi_pct || 0) >= 0 ? '+' : ''}{(overall.roi_pct || 0).toFixed(1)}%
                </p>
              </div>
              <div>
                <p className="text-slate-400 text-sm">Total Return</p>
                <p className={`text-3xl font-bold ${(trackingStats?.total_return_pct || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {(trackingStats?.total_return_pct || 0) >= 0 ? '+' : ''}{(trackingStats?.total_return_pct || 0).toFixed(1)}%
                </p>
              </div>
            </div>
          </div>

          {/* Bankroll Chart */}
          {trackingPerf?.daily_history && trackingPerf.daily_history.length > 0 && (
            <div className="card p-4">
              <h3 className="font-semibold mb-4">Bankroll Over Time</h3>
              <BankrollChart history={trackingPerf.daily_history} />
              <div className="flex justify-between text-xs text-slate-500 mt-2">
                <span>{trackingPerf.daily_history[0]?.date}</span>
                <span>{trackingPerf.daily_history[trackingPerf.daily_history.length - 1]?.date}</span>
              </div>
            </div>
          )}

          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="card p-4">
              <p className="stat-label">Total Bets</p>
              <p className="stat-lg">{overall.total_bets || 0}</p>
              <p className="text-[10px] text-slate-500 mt-1">${(overall.total_staked || 0).toFixed(0)} staked</p>
            </div>
            <div className="card p-4">
              <p className="stat-label">Record</p>
              <p className="stat-lg">{overall.wins || 0}W - {overall.losses || 0}L</p>
              <p className="text-[10px] text-slate-500 mt-1">{((overall.win_rate || 0) * 100).toFixed(1)}% win rate</p>
            </div>
            <div className="card p-4">
              <p className="stat-label">Avg Edge</p>
              <p className="stat-lg text-amber-400">{((overall.avg_edge || 0) * 100).toFixed(1)}%</p>
              <p className="text-[10px] text-slate-500 mt-1">Per bet</p>
            </div>
            <div className="card p-4">
              <p className="stat-label">Avg Probability</p>
              <p className="stat-lg">{((overall.avg_probability || 0) * 100).toFixed(1)}%</p>
              <p className="text-[10px] text-slate-500 mt-1">Model confidence</p>
            </div>
          </div>

          {/* Current Streak */}
          {trackingStats?.current_streak && (
            <div className="card p-4">
              <div className="flex items-center justify-between">
                <h3 className="font-semibold">Current Streak</h3>
                <span className={`text-2xl font-bold ${trackingStats.current_streak.type === 'WIN' ? 'text-emerald-400' : 'text-red-400'}`}>
                  {trackingStats.current_streak.length} {trackingStats.current_streak.type === 'WIN' ? 'Wins' : 'Losses'}
                </span>
              </div>
              <div className="flex gap-1 mt-3">
                {(trackingStats.recent_results || []).map((result, i) => (
                  <div 
                    key={i}
                    className={`flex-1 h-8 rounded flex items-center justify-center text-xs font-bold ${
                      result === 'WIN' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
                    }`}
                  >
                    {result === 'WIN' ? 'W' : 'L'}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* NHL Tracking Tab */}
      {activeTab === 'nhl-tracking' && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="card p-4">
              <p className="stat-label">Total Bets</p>
              <p className="stat-lg">{overall.total_bets || 0}</p>
            </div>
            <div className="card p-4">
              <p className="stat-label">Win Rate</p>
              <p className={`stat-lg ${(overall.win_rate || 0) > 0.52 ? 'text-emerald-400' : 'text-red-400'}`}>
                {((overall.win_rate || 0) * 100).toFixed(1)}%
              </p>
            </div>
            <div className="card p-4">
              <p className="stat-label">Total P&L</p>
              <p className={`stat-lg ${(overall.total_pnl || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {(overall.total_pnl || 0) >= 0 ? '+' : ''}${(overall.total_pnl || 0).toFixed(2)}
              </p>
            </div>
            <div className="card p-4">
              <p className="stat-label">ROI</p>
              <p className={`stat-lg ${(overall.roi_pct || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {(overall.roi_pct || 0) >= 0 ? '+' : ''}{(overall.roi_pct || 0).toFixed(1)}%
              </p>
            </div>
          </div>

          {/* Top Teams */}
          {trackingStats?.top_teams && trackingStats.top_teams.length > 0 && (
            <div className="card p-4">
              <h3 className="font-semibold mb-4">Top Performing Teams</h3>
              <div className="space-y-2">
                {trackingStats.top_teams.map((team, i) => (
                  <div key={i} className="flex items-center justify-between p-3 bg-slate-800/30 rounded-lg">
                    <div className="flex items-center gap-4">
                      <span className="text-2xl font-bold text-slate-600">#{i + 1}</span>
                      <div>
                        <p className="font-medium">{team.team}</p>
                        <p className="text-xs text-slate-500">{team.bets} bets • {((team.win_rate || 0) * 100).toFixed(0)}% win rate</p>
                      </div>
                    </div>
                    <p className={`text-xl font-bold ${team.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                      {team.pnl >= 0 ? '+' : ''}${team.pnl.toFixed(2)}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* NHL Accuracy by Confidence */}
          {nhlAccuracy?.by_confidence && (
            <div className="card p-4">
              <h3 className="font-semibold mb-4">Accuracy by Confidence Level</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {nhlAccuracy.by_confidence.map(bucket => (
                  <ProgressBar 
                    key={bucket.bucket} 
                    value={bucket.accuracy} 
                    label={`${bucket.bucket} Confidence`} 
                    count={bucket.count} 
                    threshold={0.52} 
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* NBA Tab */}
      {activeTab === 'nba' && (
        <div className="space-y-4">
          {nbaStats && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <div className="card p-4">
                <p className="stat-label">Total Predictions</p>
                <p className="stat-lg">{nbaStats.totalPreds.toLocaleString()}</p>
              </div>
              <div className="card p-4">
                <p className="stat-label">Avg Accuracy</p>
                <p className={`stat-lg ${nbaStats.profitable ? 'text-emerald-400' : 'text-red-400'}`}>{nbaStats.avgAccuracy.toFixed(1)}%</p>
              </div>
              <div className="card p-4">
                <p className="stat-label">Avg Error</p>
                <p className="stat-lg text-amber-400">±{nbaStats.avgError.toFixed(1)}</p>
              </div>
              <div className="card p-4">
                <p className="stat-label">Est. ROI</p>
                <p className={`stat-lg ${nbaStats.roi > 0 ? 'text-emerald-400' : 'text-red-400'}`}>{nbaStats.roi > 0 ? '+' : ''}{nbaStats.roi.toFixed(1)}%</p>
              </div>
            </div>
          )}
          {nbaAccuracy?.accuracy_by_prop && (
            <div className="card p-4">
              <h3 className="font-semibold mb-4">Accuracy by Prop Type</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {nbaAccuracy.accuracy_by_prop.map(prop => (
                  <ProgressBar key={prop.prop_type} value={prop.line_accuracy || 0} label={prop.prop_type} count={prop.total_predictions} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* By Edge Tab */}
      {activeTab === 'by-edge' && (
        <div className="space-y-4">
          <div className="card p-4">
            <h3 className="font-semibold mb-4">Performance by Edge Class</h3>
            <p className="text-sm text-slate-400 mb-4">How bets perform based on model edge at time of prediction</p>
            <div className="space-y-4">
              {byEdge.map((edge, idx) => {
                const winRate = edge.win_rate || 0;
                const isProfitable = winRate > 0.52;
                return (
                  <div key={idx} className="bg-slate-800/30 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div>
                        <span className="font-semibold text-lg">{edge.edge_class}</span>
                        <span className="text-slate-500 text-sm ml-2">({edge.bets} bets)</span>
                      </div>
                      <div className="text-right">
                        <p className={`text-xl font-bold ${edge.pnl >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                          {edge.pnl >= 0 ? '+' : ''}${edge.pnl.toFixed(2)}
                        </p>
                        <p className="text-xs text-slate-500">Avg Edge: {(edge.avg_edge * 100).toFixed(1)}%</p>
                      </div>
                    </div>
                    <div className="grid grid-cols-3 gap-4 text-center text-sm">
                      <div>
                        <p className="text-slate-400">Record</p>
                        <p className="font-semibold">{edge.wins}W - {edge.bets - edge.wins}L</p>
                      </div>
                      <div>
                        <p className="text-slate-400">Win Rate</p>
                        <p className={`font-semibold ${isProfitable ? 'text-emerald-400' : 'text-red-400'}`}>
                          {(winRate * 100).toFixed(1)}%
                        </p>
                      </div>
                      <div>
                        <p className="text-slate-400">Avg Edge</p>
                        <p className="font-semibold text-amber-400">{(edge.avg_edge * 100).toFixed(1)}%</p>
                      </div>
                    </div>
                    <div className="mt-3 h-2 bg-slate-700 rounded-full overflow-hidden">
                      <div 
                        className={`h-full ${isProfitable ? 'bg-emerald-500' : 'bg-red-500'}`}
                        style={{ width: `${winRate * 100}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {/* Info Card */}
      <div className="card p-4">
        <h3 className="font-semibold mb-3">Understanding the Numbers</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-slate-400">
          <div>
            <h4 className="text-emerald-400 font-medium mb-2">What makes a profitable bet?</h4>
            <ul className="space-y-1">
              <li>• 52.4% accuracy at -110 odds = breakeven</li>
              <li>• 55% accuracy ≈ 4.5% ROI</li>
              <li>• 60% accuracy ≈ 14% ROI</li>
            </ul>
          </div>
          <div>
            <h4 className="text-amber-400 font-medium mb-2">Edge Classes Explained</h4>
            <ul className="space-y-1">
              <li>• <strong>EXCEPTIONAL:</strong> 8%+ edge</li>
              <li>• <strong>STRONG:</strong> 5-8% edge</li>
              <li>• <strong>GOOD:</strong> 3-5% edge</li>
              <li>• <strong>MODERATE:</strong> 1-3% edge</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AccuracyTracker;