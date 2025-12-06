// src/pages/NHLPredictions.jsx - Enhanced NHL Predictions Page
import React, { useState, useEffect } from 'react';
import api from '../services/api';

const NHLPredictions = () => {
  const [loading, setLoading] = useState(true);
  const [todayData, setTodayData] = useState(null);
  const [historyData, setHistoryData] = useState(null);
  const [accuracy, setAccuracy] = useState(null);
  const [activeTab, setActiveTab] = useState('today');
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);

  useEffect(() => {
    loadData();
  }, []);

  const loadData = async () => {
    setLoading(true);
    try {
      const [todayRes, historyRes, accuracyRes] = await Promise.allSettled([
        api.getNHLTodayPredictions(),
        api.getNHLHistory(30),
        api.getNHLAccuracy(30),
      ]);
      
      if (todayRes.status === 'fulfilled') setTodayData(todayRes.value);
      if (historyRes.status === 'fulfilled') setHistoryData(historyRes.value);
      if (accuracyRes.status === 'fulfilled') setAccuracy(accuracyRes.value);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const loadByDate = async (date) => {
    setLoading(true);
    try {
      const res = await api.getNHLPredictionsByDate(date);
      setTodayData(res);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  if (loading && !todayData) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="spinner w-12 h-12"></div>
      </div>
    );
  }

  const games = todayData?.games || todayData?.predictions || [];

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-3">
            <span className="text-4xl">🏒</span>
            NHL Predictions
          </h1>
          <p className="text-slate-400 mt-1">
            {games.length} games with betting recommendations
          </p>
        </div>
        
        <div className="flex items-center gap-3">
          <input
            type="date"
            value={selectedDate}
            onChange={(e) => {
              setSelectedDate(e.target.value);
              loadByDate(e.target.value);
            }}
            className="input-field"
          />
          <button onClick={loadData} className="btn-primary">
            Refresh
          </button>
        </div>
      </div>

      {/* Accuracy Summary */}
      {accuracy && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="glass-card p-5">
            <p className="stat-label">Total Predictions</p>
            <p className="stat-value text-white">{accuracy.total_predictions || 0}</p>
          </div>
          <div className="glass-card p-5">
            <p className="stat-label">Overall Accuracy</p>
            <p className={`stat-value ${(accuracy.overall_accuracy || 0) > 0.52 ? 'text-emerald-400' : 'text-red-400'}`}>
              {((accuracy.overall_accuracy || 0) * 100).toFixed(1)}%
            </p>
          </div>
          <div className="glass-card p-5">
            <p className="stat-label">Bet Accuracy</p>
            <p className={`stat-value ${(accuracy.bet_accuracy || 0) > 0.52 ? 'text-emerald-400' : 'text-amber-400'}`}>
              {((accuracy.bet_accuracy || 0) * 100).toFixed(1)}%
            </p>
          </div>
          <div className="glass-card p-5">
            <p className="stat-label">Avg Edge on Bets</p>
            <p className="stat-value text-emerald-400">
              +{((accuracy.avg_edge_on_bets || 0) * 100).toFixed(1)}%
            </p>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-2 border-b border-slate-700/50 pb-2">
        <button
          onClick={() => setActiveTab('today')}
          className={`px-5 py-2.5 rounded-t-xl font-medium transition-all ${
            activeTab === 'today' 
              ? 'bg-emerald-500/20 text-emerald-400 border-b-2 border-emerald-500' 
              : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
          }`}
        >
          Today's Games
        </button>
        <button
          onClick={() => setActiveTab('history')}
          className={`px-5 py-2.5 rounded-t-xl font-medium transition-all ${
            activeTab === 'history' 
              ? 'bg-emerald-500/20 text-emerald-400 border-b-2 border-emerald-500' 
              : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
          }`}
        >
          History & Results
        </button>
        <button
          onClick={() => setActiveTab('accuracy')}
          className={`px-5 py-2.5 rounded-t-xl font-medium transition-all ${
            activeTab === 'accuracy' 
              ? 'bg-emerald-500/20 text-emerald-400 border-b-2 border-emerald-500' 
              : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
          }`}
        >
          Performance
        </button>
      </div>

      {/* Today's Games Tab */}
      {activeTab === 'today' && (
        <div className="space-y-4">
          {games.length > 0 ? (
            games.map((game, idx) => (
              <div key={idx} className="glass-card overflow-hidden">
                <div className="p-6">
                  {/* Game Header */}
                  <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center gap-4">
                      <div className="text-center">
                        <p className="text-2xl font-bold">{game.away_team}</p>
                        <p className="text-xs text-slate-500 mt-1">Away</p>
                      </div>
                      <div className="px-4">
                        <span className="text-slate-600 text-2xl font-light">@</span>
                      </div>
                      <div className="text-center">
                        <p className="text-2xl font-bold">{game.home_team}</p>
                        <p className="text-xs text-slate-500 mt-1">Home</p>
                      </div>
                    </div>
                    
                    {game.action === 'BET' && (
                      <div className="badge-success px-3 py-1.5 text-sm">
                        ✓ RECOMMENDED BET
                      </div>
                    )}
                  </div>
                  
                  {/* Prediction Details */}
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                    <div className="bg-slate-800/50 rounded-xl p-4 text-center">
                      <p className="text-xs text-slate-500 mb-1">Predicted Winner</p>
                      <p className="text-xl font-bold text-emerald-400">{game.predicted_winner}</p>
                    </div>
                    
                    <div className="bg-slate-800/50 rounded-xl p-4 text-center">
                      <p className="text-xs text-slate-500 mb-1">Win Probability</p>
                      <p className="text-xl font-bold">
                        {((game.model_probability || game.home_win_probability) * 100).toFixed(0)}%
                      </p>
                    </div>
                    
                    <div className="bg-slate-800/50 rounded-xl p-4 text-center">
                      <p className="text-xs text-slate-500 mb-1">Edge</p>
                      <p className={`text-xl font-bold ${(game.edge || 0) > 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        {game.edge > 0 ? '+' : ''}{((game.edge || 0) * 100).toFixed(1)}%
                      </p>
                    </div>
                    
                    <div className="bg-slate-800/50 rounded-xl p-4 text-center">
                      <p className="text-xs text-slate-500 mb-1">Odds</p>
                      <p className="text-xl font-bold">
                        {game.american_odds > 0 ? '+' : ''}{game.american_odds || '-'}
                      </p>
                    </div>
                    
                    <div className="bg-slate-800/50 rounded-xl p-4 text-center">
                      <p className="text-xs text-slate-500 mb-1">Suggested Bet</p>
                      <p className="text-xl font-bold text-amber-400">
                        {((game.bet_pct_bankroll || 0) * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                  
                  {/* Results if available */}
                  {game.actual_winner && (
                    <div className={`mt-4 p-4 rounded-xl ${game.was_correct ? 'bg-emerald-500/10 border border-emerald-500/30' : 'bg-red-500/10 border border-red-500/30'}`}>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                          <span className={`text-2xl ${game.was_correct ? '' : ''}`}>
                            {game.was_correct ? '✓' : '✗'}
                          </span>
                          <div>
                            <p className={`font-semibold ${game.was_correct ? 'text-emerald-400' : 'text-red-400'}`}>
                              {game.was_correct ? 'WINNER!' : 'Loss'}
                            </p>
                            <p className="text-sm text-slate-400">
                              Final: {game.away_team} {game.away_score} - {game.home_score} {game.home_team}
                              {game.went_to_ot && ' (OT)'}
                            </p>
                          </div>
                        </div>
                        <p className="text-sm text-slate-500">Winner: {game.actual_winner}</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))
          ) : (
            <div className="glass-card p-16 text-center">
              <span className="text-5xl mb-4 block">🏒</span>
              <p className="text-lg text-slate-400">No NHL games scheduled for this date</p>
              <p className="text-sm text-slate-500 mt-2">Check back later or view historical predictions</p>
            </div>
          )}
        </div>
      )}

      {/* History Tab */}
      {activeTab === 'history' && historyData && (
        <div className="glass-card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="data-table">
              <thead>
                <tr>
                  <th className="table-header table-cell text-left">Date</th>
                  <th className="table-header table-cell text-left">Matchup</th>
                  <th className="table-header table-cell text-left">Pick</th>
                  <th className="table-header table-cell text-right">Probability</th>
                  <th className="table-header table-cell text-right">Edge</th>
                  <th className="table-header table-cell text-center">Action</th>
                  <th className="table-header table-cell text-center">Result</th>
                </tr>
              </thead>
              <tbody>
                {historyData.predictions?.slice(0, 50).map((pred, idx) => (
                  <tr key={idx} className="table-row">
                    <td className="table-cell text-slate-400">{pred.prediction_date}</td>
                    <td className="table-cell font-medium">{pred.away_team} @ {pred.home_team}</td>
                    <td className="table-cell text-emerald-400">{pred.predicted_winner}</td>
                    <td className="table-cell text-right">{((pred.model_probability || 0) * 100).toFixed(0)}%</td>
                    <td className={`table-cell text-right ${(pred.edge || 0) > 0 ? 'text-emerald-400' : 'text-slate-400'}`}>
                      {pred.edge > 0 ? '+' : ''}{((pred.edge || 0) * 100).toFixed(1)}%
                    </td>
                    <td className="table-cell text-center">
                      <span className={`badge ${pred.action === 'BET' ? 'badge-success' : 'badge-info'}`}>
                        {pred.action || 'HOLD'}
                      </span>
                    </td>
                    <td className="table-cell text-center">
                      {pred.was_correct !== null && (
                        <span className={`badge ${pred.was_correct ? 'badge-success' : 'badge-danger'}`}>
                          {pred.was_correct ? 'WIN' : 'LOSS'}
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Accuracy Tab */}
      {activeTab === 'accuracy' && accuracy && (
        <div className="space-y-6">
          {/* Accuracy by Confidence */}
          {accuracy.by_confidence && (
            <div className="glass-card p-6">
              <h3 className="text-lg font-semibold mb-4">Accuracy by Confidence Level</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {accuracy.by_confidence.map((bucket, idx) => (
                  <div key={idx} className="bg-slate-800/50 rounded-xl p-5">
                    <div className="flex items-center justify-between mb-3">
                      <span className="text-slate-400 capitalize">{bucket.bucket} Confidence</span>
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
          
          {/* Key Insights */}
          <div className="glass-card p-6">
            <h3 className="text-lg font-semibold mb-4">Key Insights</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
                  <span className="text-slate-400">Breakeven Accuracy Needed</span>
                  <span className="font-semibold">52.4%</span>
                </div>
                <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
                  <span className="text-slate-400">Current Win Rate</span>
                  <span className={`font-semibold ${(accuracy.overall_accuracy || 0) > 0.524 ? 'text-emerald-400' : 'text-red-400'}`}>
                    {((accuracy.overall_accuracy || 0) * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
                  <span className="text-slate-400">Average Confidence</span>
                  <span className="font-semibold">{((accuracy.avg_confidence || 0) * 100).toFixed(1)}%</span>
                </div>
              </div>
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
                  <span className="text-slate-400">Correct Predictions</span>
                  <span className="font-semibold text-emerald-400">{accuracy.correct_predictions || 0}</span>
                </div>
                <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
                  <span className="text-slate-400">Total Predictions</span>
                  <span className="font-semibold">{accuracy.total_predictions || 0}</span>
                </div>
                <div className="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg">
                  <span className="text-slate-400">Profit Status</span>
                  <span className={`font-semibold ${(accuracy.overall_accuracy || 0) > 0.524 ? 'text-emerald-400' : 'text-amber-400'}`}>
                    {(accuracy.overall_accuracy || 0) > 0.524 ? 'Profitable' : 'Improving'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default NHLPredictions;