// src/pages/AccuracyTracker.jsx - Enhanced Performance Tracking
import React, { useState, useEffect, useMemo } from 'react';
import api from '../services/api';

const AccuracyBar = ({ value, label, count, threshold = 0.524 }) => {
  const percentage = value * 100;
  const isProfitable = value > threshold;
  
  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <span className="font-semibold uppercase text-sm text-slate-300">{label}</span>
        <span className={`font-mono font-bold ${isProfitable ? 'text-emerald-400' : 'text-red-400'}`}>
          {percentage.toFixed(1)}%
        </span>
      </div>
      <div className="progress-bar h-3">
        <div 
          className={`progress-bar-fill ${isProfitable ? 'bg-gradient-to-r from-emerald-500 to-emerald-400' : 'bg-gradient-to-r from-red-500 to-red-400'}`}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
      </div>
      <div className="flex justify-between text-xs text-slate-500">
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
  const [nbaHistory, setNbaHistory] = useState(null);
  const [nhlResults, setNhlResults] = useState(null);
  const [days, setDays] = useState(30);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    loadAccuracyData();
  }, [days]);

  const loadAccuracyData = async () => {
    setLoading(true);
    try {
      const [nbaAccRes, nhlAccRes, nbaHistRes, nhlResRes] = await Promise.allSettled([
        api.getNBAAccuracy(days),
        api.getNHLAccuracy(days),
        api.getNBAHistory(days),
        api.getNHLResults(days),
      ]);
      
      if (nbaAccRes.status === 'fulfilled') setNbaAccuracy(nbaAccRes.value);
      if (nhlAccRes.status === 'fulfilled') setNhlAccuracy(nhlAccRes.value);
      if (nbaHistRes.status === 'fulfilled') setNbaHistory(nbaHistRes.value);
      if (nhlResRes.status === 'fulfilled') setNhlResults(nhlResRes.value);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const nbaOverallStats = useMemo(() => {
    if (!nbaAccuracy?.accuracy_by_prop) return null;
    
    const props = nbaAccuracy.accuracy_by_prop;
    const totalPreds = props.reduce((sum, p) => sum + p.total_predictions, 0);
    const avgAccuracy = props.reduce((sum, p) => sum + (p.line_accuracy || 0), 0) / props.length;
    const avgError = props.reduce((sum, p) => sum + (p.avg_error || 0), 0) / props.length;
    const estimatedROI = (avgAccuracy * 0.91) - ((1 - avgAccuracy) * 1);
    
    return {
      totalPreds,
      avgAccuracy: avgAccuracy * 100,
      avgError,
      estimatedROI: estimatedROI * 100,
      isProfitable: avgAccuracy > 0.524,
    };
  }, [nbaAccuracy]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="spinner w-12 h-12"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <span className="text-4xl">🎯</span>
            Performance Tracker
          </h1>
          <p className="text-slate-400 mt-1">Track prediction accuracy over time</p>
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
            <option value={90}>Last 90 Days</option>
          </select>
          <button onClick={loadAccuracyData} className="btn-primary">
            Refresh
          </button>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex gap-2 border-b border-slate-700/50 pb-2">
        {['overview', 'nba', 'nhl'].map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-5 py-2.5 rounded-t-xl font-medium transition-all capitalize ${
              activeTab === tab 
                ? 'bg-emerald-500/20 text-emerald-400 border-b-2 border-emerald-500' 
                : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
            }`}
          >
            {tab === 'overview' ? 'Overview' : tab.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          {/* Combined Stats */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="glass-card p-6">
              <p className="stat-label">NBA Accuracy</p>
              <p className={`stat-value ${nbaOverallStats?.isProfitable ? 'text-emerald-400' : 'text-red-400'}`}>
                {nbaOverallStats?.avgAccuracy.toFixed(1) || 0}%
              </p>
              <p className="text-xs text-slate-500 mt-1">
                {nbaOverallStats?.totalPreds.toLocaleString() || 0} predictions
              </p>
            </div>
            
            <div className="glass-card p-6">
              <p className="stat-label">NHL Accuracy</p>
              <p className={`stat-value ${(nhlAccuracy?.overall_accuracy || 0) > 0.52 ? 'text-emerald-400' : 'text-amber-400'}`}>
                {((nhlAccuracy?.overall_accuracy || 0) * 100).toFixed(1)}%
              </p>
              <p className="text-xs text-slate-500 mt-1">
                {nhlAccuracy?.total_predictions || 0} predictions
              </p>
            </div>
            
            <div className="glass-card p-6">
              <p className="stat-label">NBA Est. ROI</p>
              <p className={`stat-value ${nbaOverallStats?.estimatedROI > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {nbaOverallStats?.estimatedROI > 0 ? '+' : ''}{nbaOverallStats?.estimatedROI.toFixed(1) || 0}%
              </p>
              <p className="text-xs text-slate-500 mt-1">At -110 odds</p>
            </div>
            
            <div className="glass-card p-6">
              <p className="stat-label">NHL Profit/Loss</p>
              <p className={`stat-value ${(nhlResults?.total_profit || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {(nhlResults?.total_profit || 0) >= 0 ? '+' : ''}${(nhlResults?.total_profit || 0).toFixed(0)}
              </p>
              <p className="text-xs text-slate-500 mt-1">
                {nhlResults?.wins || 0}W - {nhlResults?.losses || 0}L
              </p>
            </div>
          </div>

          {/* Quick Comparison */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="glass-card p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <span className="text-xl">🏀</span>
                NBA Props Summary
              </h3>
              {nbaAccuracy?.accuracy_by_prop ? (
                <div className="space-y-4">
                  {nbaAccuracy.accuracy_by_prop.map(prop => (
                    <AccuracyBar
                      key={prop.prop_type}
                      value={prop.line_accuracy || 0}
                      label={prop.prop_type}
                      count={prop.total_predictions}
                    />
                  ))}
                </div>
              ) : (
                <p className="text-slate-500 text-center py-8">No NBA data available</p>
              )}
            </div>

            <div className="glass-card p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <span className="text-xl">🏒</span>
                NHL Performance
              </h3>
              {nhlAccuracy?.by_confidence ? (
                <div className="space-y-4">
                  {nhlAccuracy.by_confidence.map(bucket => (
                    <AccuracyBar
                      key={bucket.bucket}
                      value={bucket.accuracy}
                      label={`${bucket.bucket} Confidence`}
                      count={bucket.count}
                      threshold={0.52}
                    />
                  ))}
                </div>
              ) : (
                <p className="text-slate-500 text-center py-8">No NHL data available</p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* NBA Tab */}
      {activeTab === 'nba' && (
        <div className="space-y-6">
          {/* Stats Cards */}
          {nbaOverallStats && (
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="glass-card p-6">
                <p className="stat-label">Total Predictions</p>
                <p className="stat-value text-white">{nbaOverallStats.totalPreds.toLocaleString()}</p>
              </div>
              <div className="glass-card p-6">
                <p className="stat-label">Average Accuracy</p>
                <p className={`stat-value ${nbaOverallStats.isProfitable ? 'text-emerald-400' : 'text-red-400'}`}>
                  {nbaOverallStats.avgAccuracy.toFixed(1)}%
                </p>
              </div>
              <div className="glass-card p-6">
                <p className="stat-label">Avg Prediction Error</p>
                <p className="stat-value text-amber-400">±{nbaOverallStats.avgError.toFixed(1)}</p>
              </div>
              <div className="glass-card p-6">
                <p className="stat-label">Estimated ROI</p>
                <p className={`stat-value ${nbaOverallStats.estimatedROI > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  {nbaOverallStats.estimatedROI > 0 ? '+' : ''}{nbaOverallStats.estimatedROI.toFixed(1)}%
                </p>
              </div>
            </div>
          )}

          {/* Detailed Prop Accuracy */}
          <div className="glass-card p-6">
            <h3 className="text-lg font-semibold mb-6">Accuracy by Prop Type</h3>
            {nbaAccuracy?.accuracy_by_prop && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {nbaAccuracy.accuracy_by_prop.map(prop => (
                  <AccuracyBar
                    key={prop.prop_type}
                    value={prop.line_accuracy || 0}
                    label={prop.prop_type}
                    count={prop.total_predictions}
                  />
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* NHL Tab */}
      {activeTab === 'nhl' && (
        <div className="space-y-6">
          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="glass-card p-6">
              <p className="stat-label">Total Predictions</p>
              <p className="stat-value text-white">{nhlAccuracy?.total_predictions || 0}</p>
            </div>
            <div className="glass-card p-6">
              <p className="stat-label">Overall Accuracy</p>
              <p className={`stat-value ${(nhlAccuracy?.overall_accuracy || 0) > 0.52 ? 'text-emerald-400' : 'text-red-400'}`}>
                {((nhlAccuracy?.overall_accuracy || 0) * 100).toFixed(1)}%
              </p>
            </div>
            <div className="glass-card p-6">
              <p className="stat-label">Bet Win Rate</p>
              <p className={`stat-value ${(nhlResults?.win_rate || 0) > 0.52 ? 'text-emerald-400' : 'text-amber-400'}`}>
                {((nhlResults?.win_rate || 0) * 100).toFixed(1)}%
              </p>
            </div>
            <div className="glass-card p-6">
              <p className="stat-label">Total P/L</p>
              <p className={`stat-value ${(nhlResults?.total_profit || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                {(nhlResults?.total_profit || 0) >= 0 ? '+' : ''}${(nhlResults?.total_profit || 0).toFixed(0)}
              </p>
            </div>
          </div>

          {/* Accuracy by Confidence */}
          {nhlAccuracy?.by_confidence && (
            <div className="glass-card p-6">
              <h3 className="text-lg font-semibold mb-6">Accuracy by Confidence Level</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {nhlAccuracy.by_confidence.map(bucket => (
                  <div key={bucket.bucket} className="bg-slate-800/50 rounded-xl p-5">
                    <div className="flex justify-between items-center mb-3">
                      <span className="capitalize text-slate-300">{bucket.bucket} Confidence</span>
                      <span className="text-sm text-slate-500">{bucket.count} games</span>
                    </div>
                    <p className={`text-3xl font-bold ${bucket.accuracy > 0.55 ? 'text-emerald-400' : bucket.accuracy > 0.50 ? 'text-amber-400' : 'text-red-400'}`}>
                      {(bucket.accuracy * 100).toFixed(1)}%
                    </p>
                    <div className="progress-bar mt-3">
                      <div 
                        className={`progress-bar-fill ${bucket.accuracy > 0.55 ? 'bg-emerald-500' : bucket.accuracy > 0.50 ? 'bg-amber-500' : 'bg-red-500'}`}
                        style={{ width: `${bucket.accuracy * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Understanding Section */}
      <div className="glass-card p-6">
        <h3 className="text-lg font-semibold mb-4">Understanding the Numbers</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-slate-300 text-sm">
          <div>
            <h4 className="font-semibold text-emerald-400 mb-2">What makes a profitable bet?</h4>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <span className="text-slate-500">•</span>
                At standard -110 odds, you need 52.4% accuracy to break even
              </li>
              <li className="flex items-start gap-2">
                <span className="text-slate-500">•</span>
                55% accuracy = ~4.5% ROI (solid long-term profit)
              </li>
              <li className="flex items-start gap-2">
                <span className="text-slate-500">•</span>
                60% accuracy = ~14% ROI (excellent performance)
              </li>
            </ul>
          </div>
          <div>
            <h4 className="font-semibold text-amber-400 mb-2">How we measure accuracy</h4>
            <ul className="space-y-2">
              <li className="flex items-start gap-2">
                <span className="text-slate-500">•</span>
                <strong>Line Accuracy:</strong> % of predictions that beat the betting line
              </li>
              <li className="flex items-start gap-2">
                <span className="text-slate-500">•</span>
                <strong>Avg Error:</strong> How far off our predictions are on average
              </li>
              <li className="flex items-start gap-2">
                <span className="text-slate-500">•</span>
                <strong>Confidence:</strong> Model's certainty in each prediction
              </li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AccuracyTracker;