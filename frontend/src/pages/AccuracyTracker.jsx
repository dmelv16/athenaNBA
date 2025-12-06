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

const AccuracyTracker = () => {
  const [loading, setLoading] = useState(true);
  const [nbaAccuracy, setNbaAccuracy] = useState(null);
  const [nhlAccuracy, setNhlAccuracy] = useState(null);
  const [nhlResults, setNhlResults] = useState(null);
  const [days, setDays] = useState(30);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    loadData();
  }, [days]);

  const loadData = async () => {
    setLoading(true);
    try {
      const [nbaAcc, nhlAcc, nhlRes] = await Promise.allSettled([
        api.getNBAAccuracy(days),
        api.getNHLAccuracy(days),
        api.getNHLResults(days),
      ]);
      if (nbaAcc.status === 'fulfilled') setNbaAccuracy(nbaAcc.value);
      if (nhlAcc.status === 'fulfilled') setNhlAccuracy(nhlAcc.value);
      if (nhlRes.status === 'fulfilled') setNhlResults(nhlRes.value);
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
          <p className="text-slate-400 text-sm mt-0.5">Track prediction accuracy over time</p>
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

      {/* Tabs */}
      <div className="flex gap-1 border-b border-white/5">
        {['overview', 'nba', 'nhl'].map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 text-sm font-medium transition-colors capitalize ${
              activeTab === tab 
                ? 'text-emerald-400 border-b-2 border-emerald-400' 
                : 'text-slate-400 hover:text-white'
            }`}
          >
            {tab === 'overview' ? 'Overview' : tab.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Overview */}
      {activeTab === 'overview' && (
        <div className="space-y-5">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="card p-4">
              <p className="stat-label">NBA Accuracy</p>
              <p className={`stat-lg ${nbaStats?.profitable ? 'text-emerald-400' : 'text-red-400'}`}>
                {nbaStats?.avgAccuracy.toFixed(1) || 0}%
              </p>
              <p className="text-[10px] text-slate-500 mt-1">{nbaStats?.totalPreds.toLocaleString() || 0} predictions</p>
            </div>
            <div className="card p-4">
              <p className="stat-label">NHL Accuracy</p>
              <p className={`stat-lg ${(nhlAccuracy?.overall_accuracy || 0) > 0.52 ? 'text-emerald-400' : 'text-amber-400'}`}>
                {((nhlAccuracy?.overall_accuracy || 0) * 100).toFixed(1)}%
              </p>
              <p className="text-[10px] text-slate-500 mt-1">{nhlAccuracy?.total_predictions || 0} predictions</p>
            </div>
            <div className="card p-4">
              <p className="stat-label">NBA Est. ROI</p>
              <p className={`stat-lg ${(nbaStats?.roi || 0) > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {(nbaStats?.roi || 0) > 0 ? '+' : ''}{(nbaStats?.roi || 0).toFixed(1)}%
              </p>
              <p className="text-[10px] text-slate-500 mt-1">At -110 odds</p>
            </div>
            <div className="card p-4">
              <p className="stat-label">NHL P/L</p>
              <p className={`stat-lg ${(nhlResults?.total_profit || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {(nhlResults?.total_profit || 0) >= 0 ? '+' : ''}${Math.abs(nhlResults?.total_profit || 0).toFixed(0)}
              </p>
              <p className="text-[10px] text-slate-500 mt-1">{nhlResults?.wins || 0}W - {nhlResults?.losses || 0}L</p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="card p-4">
              <h3 className="font-semibold mb-4 flex items-center gap-2">🏀 NBA Props</h3>
              {nbaAccuracy?.accuracy_by_prop ? (
                <div className="space-y-4">
                  {nbaAccuracy.accuracy_by_prop.map(prop => (
                    <ProgressBar key={prop.prop_type} value={prop.line_accuracy || 0} label={prop.prop_type} count={prop.total_predictions} />
                  ))}
                </div>
              ) : (
                <p className="text-slate-500 text-center py-6">No NBA data</p>
              )}
            </div>
            <div className="card p-4">
              <h3 className="font-semibold mb-4 flex items-center gap-2">🏒 NHL Performance</h3>
              {nhlAccuracy?.by_confidence ? (
                <div className="space-y-4">
                  {nhlAccuracy.by_confidence.map(bucket => (
                    <ProgressBar key={bucket.bucket} value={bucket.accuracy} label={`${bucket.bucket} Confidence`} count={bucket.count} threshold={0.52} />
                  ))}
                </div>
              ) : (
                <p className="text-slate-500 text-center py-6">No NHL data</p>
              )}
            </div>
          </div>
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

      {/* NHL Tab */}
      {activeTab === 'nhl' && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="card p-4">
              <p className="stat-label">Total Predictions</p>
              <p className="stat-lg">{nhlAccuracy?.total_predictions || 0}</p>
            </div>
            <div className="card p-4">
              <p className="stat-label">Overall Accuracy</p>
              <p className={`stat-lg ${(nhlAccuracy?.overall_accuracy || 0) > 0.52 ? 'text-emerald-400' : 'text-red-400'}`}>
                {((nhlAccuracy?.overall_accuracy || 0) * 100).toFixed(1)}%
              </p>
            </div>
            <div className="card p-4">
              <p className="stat-label">Bet Win Rate</p>
              <p className={`stat-lg ${(nhlResults?.win_rate || 0) > 0.52 ? 'text-emerald-400' : 'text-amber-400'}`}>
                {((nhlResults?.win_rate || 0) * 100).toFixed(1)}%
              </p>
            </div>
            <div className="card p-4">
              <p className="stat-label">Total P/L</p>
              <p className={`stat-lg ${(nhlResults?.total_profit || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {(nhlResults?.total_profit || 0) >= 0 ? '+' : ''}${Math.abs(nhlResults?.total_profit || 0).toFixed(0)}
              </p>
            </div>
          </div>
          {nhlAccuracy?.by_confidence && (
            <div className="card p-4">
              <h3 className="font-semibold mb-4">Accuracy by Confidence</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {nhlAccuracy.by_confidence.map(bucket => (
                  <div key={bucket.bucket} className="bg-slate-800/50 rounded-lg p-4">
                    <div className="flex justify-between items-center mb-2">
                      <span className="capitalize text-slate-300">{bucket.bucket}</span>
                      <span className="text-xs text-slate-500">{bucket.count} games</span>
                    </div>
                    <p className={`text-2xl font-bold ${bucket.accuracy > 0.55 ? 'text-emerald-400' : bucket.accuracy > 0.50 ? 'text-amber-400' : 'text-red-400'}`}>
                      {(bucket.accuracy * 100).toFixed(1)}%
                    </p>
                    <div className="mt-2 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                      <div 
                        className={`h-full ${bucket.accuracy > 0.55 ? 'bg-emerald-500' : bucket.accuracy > 0.50 ? 'bg-amber-500' : 'bg-red-500'}`}
                        style={{width: `${bucket.accuracy * 100}%`}}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
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
            <h4 className="text-amber-400 font-medium mb-2">How we measure</h4>
            <ul className="space-y-1">
              <li>• <strong>Line Accuracy:</strong> % that beat the line</li>
              <li>• <strong>Avg Error:</strong> How far off predictions are</li>
              <li>• <strong>Edge:</strong> Model prob minus implied odds</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AccuracyTracker;