// src/pages/AccuracyTracker.jsx
import React, { useState, useEffect } from 'react';
import api from '../services/api';

const AccuracyBar = ({ value, label, count }) => {
  const percentage = value * 100;
  const isProfit = percentage > 52.4; // Breakeven at -110 odds
  
  return (
    <div className="space-y-2">
      <div className="flex justify-between items-center">
        <span className="font-semibold uppercase">{label}</span>
        <span className={`font-mono ${isProfit ? 'text-green-400' : 'text-red-400'}`}>
          {percentage.toFixed(1)}%
        </span>
      </div>
      <div className="h-4 bg-gray-700 rounded-full overflow-hidden">
        <div 
          className={`h-full transition-all duration-500 ${isProfit ? 'bg-green-500' : 'bg-red-500'}`}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        />
      </div>
      <div className="flex justify-between text-xs text-gray-500">
        <span>{count} predictions</span>
        <span>Breakeven: 52.4%</span>
      </div>
    </div>
  );
};

const AccuracyTracker = () => {
  const [loading, setLoading] = useState(true);
  const [accuracy, setAccuracy] = useState(null);
  const [history, setHistory] = useState(null);
  const [days, setDays] = useState(30);

  useEffect(() => {
    loadAccuracyData();
  }, [days]);

  const loadAccuracyData = async () => {
    setLoading(true);
    try {
      const [accRes, histRes] = await Promise.allSettled([
        api.getNBAAccuracy(days),
        api.getNBAHistory(days),
      ]);
      
      if (accRes.status === 'fulfilled') setAccuracy(accRes.value);
      if (histRes.status === 'fulfilled') setHistory(histRes.value);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  // Calculate daily performance from history
  const dailyPerformance = React.useMemo(() => {
    if (!history?.predictions) return [];
    
    const byDate = {};
    history.predictions.forEach(pred => {
      const date = pred.prediction_date;
      if (!byDate[date]) {
        byDate[date] = { date, total: 0, count: 0 };
      }
      byDate[date].count++;
    });
    
    return Object.values(byDate).sort((a, b) => a.date.localeCompare(b.date));
  }, [history]);

  // Calculate overall stats
  const overallStats = React.useMemo(() => {
    if (!accuracy?.accuracy_by_prop) return null;
    
    const props = accuracy.accuracy_by_prop;
    const totalPreds = props.reduce((sum, p) => sum + p.total_predictions, 0);
    const avgAccuracy = props.reduce((sum, p) => sum + (p.line_accuracy || 0), 0) / props.length;
    const avgError = props.reduce((sum, p) => sum + (p.avg_error || 0), 0) / props.length;
    
    // Calculate ROI assuming -110 odds
    // Win pays 0.91 units, loss costs 1 unit
    // ROI = (wins * 0.91 - losses * 1) / total_bets
    const avgWinRate = avgAccuracy;
    const estimatedROI = (avgWinRate * 0.91) - ((1 - avgWinRate) * 1);
    
    return {
      totalPreds,
      avgAccuracy: avgAccuracy * 100,
      avgError,
      estimatedROI: estimatedROI * 100,
      isProfitable: avgAccuracy > 0.524,
    };
  }, [accuracy]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-green-500"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            🎯 Accuracy Tracker
          </h1>
          <p className="text-gray-400">Track prediction performance over time</p>
        </div>
        
        <div className="flex items-center gap-4">
          <select
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
            className="bg-gray-700 border border-gray-600 rounded-lg px-4 py-2"
          >
            <option value={7}>Last 7 Days</option>
            <option value={14}>Last 14 Days</option>
            <option value={30}>Last 30 Days</option>
            <option value={60}>Last 60 Days</option>
            <option value={90}>Last 90 Days</option>
          </select>
          <button
            onClick={loadAccuracyData}
            className="bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Overall Stats */}
      {overallStats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <p className="text-gray-400 text-sm">Total Predictions</p>
            <p className="text-3xl font-bold text-white">{overallStats.totalPreds.toLocaleString()}</p>
          </div>
          
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <p className="text-gray-400 text-sm">Average Accuracy</p>
            <p className={`text-3xl font-bold ${overallStats.isProfitable ? 'text-green-400' : 'text-red-400'}`}>
              {overallStats.avgAccuracy.toFixed(1)}%
            </p>
            <p className="text-xs text-gray-500">Breakeven: 52.4%</p>
          </div>
          
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <p className="text-gray-400 text-sm">Avg Prediction Error</p>
            <p className="text-3xl font-bold text-yellow-400">
              ±{overallStats.avgError.toFixed(1)}
            </p>
          </div>
          
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <p className="text-gray-400 text-sm">Estimated ROI</p>
            <p className={`text-3xl font-bold ${overallStats.estimatedROI > 0 ? 'text-green-400' : 'text-red-400'}`}>
              {overallStats.estimatedROI > 0 ? '+' : ''}{overallStats.estimatedROI.toFixed(1)}%
            </p>
            <p className="text-xs text-gray-500">At -110 odds</p>
          </div>
        </div>
      )}

      {/* Accuracy by Prop Type */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h2 className="text-xl font-semibold mb-6">Accuracy by Prop Type</h2>
        
        {accuracy?.accuracy_by_prop && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {accuracy.accuracy_by_prop.map(prop => (
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

      {/* Performance Legend */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h2 className="text-xl font-semibold mb-4">Understanding the Numbers</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-gray-300">
          <div>
            <h3 className="font-semibold text-green-400 mb-2">What makes a profitable bet?</h3>
            <ul className="space-y-2 text-sm">
              <li>• At standard -110 odds, you need 52.4% accuracy to break even</li>
              <li>• 55% accuracy = ~4.5% ROI (solid long-term profit)</li>
              <li>• 60% accuracy = ~14% ROI (excellent performance)</li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold text-yellow-400 mb-2">How we measure accuracy</h3>
            <ul className="space-y-2 text-sm">
              <li>• <strong>Line Accuracy:</strong> % of predictions that beat the betting line</li>
              <li>• <strong>Avg Error:</strong> How far off our predictions are on average</li>
              <li>• <strong>Confidence:</strong> Model's certainty in each prediction</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Daily Activity */}
      {dailyPerformance.length > 0 && (
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h2 className="text-xl font-semibold mb-4">Daily Prediction Volume</h2>
          <div className="h-32 flex items-end gap-1">
            {dailyPerformance.slice(-30).map((day, idx) => (
              <div
                key={day.date}
                className="flex-1 bg-green-500 rounded-t hover:bg-green-400 transition-colors"
                style={{ height: `${(day.count / Math.max(...dailyPerformance.map(d => d.count))) * 100}%` }}
                title={`${day.date}: ${day.count} predictions`}
              />
            ))}
          </div>
          <div className="flex justify-between text-xs text-gray-500 mt-2">
            <span>{dailyPerformance[Math.max(0, dailyPerformance.length - 30)]?.date}</span>
            <span>{dailyPerformance[dailyPerformance.length - 1]?.date}</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default AccuracyTracker;